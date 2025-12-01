import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import os

# ==============================================================
# 1. CORE UTILITIES
# ==============================================================
def calculate_gaussian_curvature(mesh, neighborhood_size=25):
    vertices = np.asarray(mesh.vertices)
    curvatures = np.zeros(len(vertices))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i, vertex in enumerate(vertices):
        [k, idx, _] = kdtree.search_knn_vector_3d(vertex, neighborhood_size)
        if k < 10:
            continue
        neighbors = vertices[idx[1:]]
        centered_neighbors = neighbors - vertex
        if len(centered_neighbors) >= 6:
            pca_local = PCA(n_components=3)
            pca_local.fit(centered_neighbors)
            local_coords = pca_local.transform(centered_neighbors)
            x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]
            A = np.column_stack([x**2, y**2, x * y, x, y, np.ones(len(x))])
            try:
                coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
                a, b, c = coeffs[0], coeffs[1], coeffs[2]
                curvatures[i] = a * b - (c / 2) ** 2
            except np.linalg.LinAlgError:
                curvatures[i] = 0
    return curvatures


def get_adaptive_torso_dimensions(vertices):
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    return {"width": x_range, "height": y_range, "depth": z_range}


# Helper used in the slice analysis to ignore tiny wiggles on flatter backs.
def _smoothed_depth_profile(points_pca, sigma=1.25):
    if len(points_pca) < 5:
        return points_pca[:, 1]

    sort_idx = np.argsort(points_pca[:, 0])
    sorted_depth = points_pca[sort_idx, 1]
    smoothed_sorted = gaussian_filter1d(sorted_depth, sigma=sigma, mode="nearest")

    smoothed = np.empty_like(sorted_depth)
    smoothed[sort_idx] = smoothed_sorted
    return smoothed


def _order_points_mst(points, k=10, return_indices=False):
    """
    Order scattered 2D points along a curve using a k-NN MST and its diameter.
    Falls back to sorting by x if the graph is too small/disconnected.
    """
    points = np.asarray(points)
    if len(points) < 3:
        return points

    A = kneighbors_graph(points, min(k, len(points) - 1), mode="distance", include_self=False)
    G = nx.from_scipy_sparse_array(A)
    T = nx.minimum_spanning_tree(G)

    leaves = [n for n in T.nodes() if T.degree[n] == 1]
    if len(leaves) < 2:
        idx = np.argsort(points[:, 0])
        return idx if return_indices else points[idx]

    start = leaves[0]
    lengths = nx.single_source_dijkstra_path_length(T, start)
    far1 = max(lengths, key=lengths.get)
    lengths = nx.single_source_dijkstra_path_length(T, far1)
    far2 = max(lengths, key=lengths.get)

    path = nx.shortest_path(T, far1, far2)
    return path if return_indices else points[path]


def _curvature_along_path(ordered_points):
    """
    Estimate signed curvature along an ordered 2D path.
    Negative curvature corresponds to valleys in our convention.
    """
    if len(ordered_points) < 5:
        return np.zeros(len(ordered_points))

    diffs = np.diff(ordered_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    t = np.clip(t, 1e-6, None)

    x = ordered_points[:, 0]
    y = ordered_points[:, 1]
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    denom = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


def _lift_path_above_mesh(points, mesh, offset=3.0, k=12):
    """
    Offset a polyline slightly off the mesh along local normals
    to avoid z-fighting in the viewer.
    """
    if len(points) == 0:
        return points

    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    tree = o3d.geometry.KDTreeFlann(pcd)

    lifted = []
    for pt in points:
        [k_found, idx, _] = tree.search_knn_vector_3d(pt, k)
        if k_found == 0:
            lifted.append(pt)
            continue
        local_normals = normals[idx]
        normal = np.mean(local_normals, axis=0)
        norm = np.linalg.norm(normal) + 1e-8
        normal /= norm
        lifted.append(pt + normal * offset)

    return np.array(lifted)


# ==============================================================
# 2. CORE PROJECTION LOGIC (Unchanged)
# 2.  PROJECTION LOGIC (Unchanged)
# ==============================================================

def _find_deepest_point_by_projection(points_in_slice, store_vis=False):
    if len(points_in_slice) < 20:
        return None, None

    # Use global X to keep the search corridor anchored to the torso center
    x_global = points_in_slice[:, 0]
    median_x_global = np.median(x_global)
    x_range_global = max(x_global.max() - x_global.min(), 1e-6)

    # Define a central search window
    primary_ratio = 0.15
    fallback_ratio = 0.28
    center_width = x_range_global * primary_ratio
    center_mask = np.abs(x_global - median_x_global) < (center_width / 2)

    # If the primary window is too sparse, expand it (still centered)
    if np.sum(center_mask) < 8:
        center_width = x_range_global * fallback_ratio
        center_mask = np.abs(x_global - median_x_global) < (center_width / 2)

    # Final fallback: take the closest-to-center points only (to keep search restricted)
    if np.sum(center_mask) < 6:
        max_candidates = min(len(points_in_slice), 24)
        closest_idx = np.argsort(np.abs(x_global - median_x_global))[:max_candidates]
        mask = np.zeros_like(x_global, dtype=bool)
        mask[closest_idx] = True
        center_mask = mask

    center_points_original_coords = points_in_slice[center_mask]
    center_points_2d = center_points_original_coords[:, [0, 2]]  # (X, Z)

    if len(center_points_original_coords) < 5:
        return None, None

    # Order the slice using MST diameter to respect the actual cross-section shape
    ordered_idx = _order_points_mst(center_points_2d, return_indices=True)
    ordered_2d = center_points_2d[ordered_idx]
    ordered_orig = center_points_original_coords[ordered_idx]

    # Smooth along the ordered path
    smooth_x = gaussian_filter1d(ordered_2d[:, 0], sigma=1.0, mode="nearest")
    smooth_z = gaussian_filter1d(ordered_2d[:, 1], sigma=1.0, mode="nearest")
    smooth_y = gaussian_filter1d(ordered_orig[:, 1], sigma=1.0, mode="nearest")

    # Resample uniformly along arc length for stable curvature and depth evaluation
    diffs = np.diff(np.column_stack([smooth_x, smooth_z]), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if s[-1] < 1e-6:
        return None, None
    n_resample = 200
    s_new = np.linspace(0, s[-1], n_resample)
    x_rs = np.interp(s_new, s, smooth_x)
    z_rs = np.interp(s_new, s, smooth_z)
    y_rs = np.interp(s_new, s, smooth_y)

    # Curvature in X-Z plane (signed)
    dx = np.gradient(x_rs, s_new)
    dz = np.gradient(z_rs, s_new)
    ddx = np.gradient(dx, s_new)
    ddz = np.gradient(dz, s_new)
    denom = np.power(dx ** 2 + dz ** 2, 1.5) + 1e-6
    curvature = (dx * ddz - dz * ddx) / denom

    # Depth scoring (global Z; valley corresponds to minimum Z)
    depth_range = max(z_rs.max() - z_rs.min(), 1e-6)
    depth_score = (z_rs - z_rs.min()) / depth_range

    # Favor negative curvature (valleys)
    curv_range = max(curvature.max() - curvature.min(), 1e-6)
    curv_score = (curvature - curvature.min()) / curv_range

    # Centrality in global X
    center_penalty = np.abs(x_rs - median_x_global)
    center_penalty /= center_penalty.max() + 1e-6

    combined_score = 0.65 * depth_score + 0.25 * curv_score + 0.10 * center_penalty
    best_idx = np.argmin(combined_score)
    deepest_point = np.array([x_rs[best_idx], y_rs[best_idx], z_rs[best_idx]])

    vis_data = None
    if store_vis:
        x_min = x_global.min()
        x_max = x_global.max()
        p1 = np.array([x_min, deepest_point[1], deepest_point[2]])
        p2 = np.array([x_max, deepest_point[1], deepest_point[2]])
        vis_data = {
            "points_orig": points_in_slice,
            "ref_line_points": [p1, p2],
            "deepest_point": deepest_point,
        }

    return deepest_point, vis_data

# ==============================================================
# 3. PIPELINE FUNCTION (Unchanged)
# ==============================================================

def detect_spinal_midline_projection_method(
    backMesh,
    slice_thickness=1.0,
    num_slices=200,
    top_cut=0.17,
    bottom_cut=0.15,
):
    allPoints = np.asarray(backMesh.sample_points_uniformly(number_of_points=80000).points)
    y_min, y_max = allPoints[:, 1].min(), allPoints[:, 1].max()
    total_height = y_max - y_min
    valid_min = y_min + top_cut * total_height
    valid_max = y_max - bottom_cut * total_height
    sliceHeights = np.linspace(valid_min, valid_max, num=num_slices)

    spinalMidlinePoints = []
    debug_vis_data = []
    vis_indices = np.linspace(0, num_slices - 1, 15, dtype=int)

    print("\n" + "="*70)
    print("SPINAL MIDLINE DETECTION (V8.0 - Top-Down Image Export)")
    print("="*70)

    for i, current_y in enumerate(sliceHeights):
        mask = (allPoints[:, 1] >= current_y - slice_thickness / 2) & (
            allPoints[:, 1] < current_y + slice_thickness / 2
        )
        pointsInSlice = allPoints[mask]
        if len(pointsInSlice) < 20:
            continue

        deepest_point, slice_vis_data = _find_deepest_point_by_projection(
            pointsInSlice, store_vis=(i in vis_indices)
        )

        if deepest_point is not None:
            spinalMidlinePoints.append(deepest_point)
            if slice_vis_data is not None:
                debug_vis_data.append(slice_vis_data)

    if not spinalMidlinePoints:
        return np.empty((0, 3)), None

    spinalMidlinePoints = np.array(spinalMidlinePoints)
    spinalMidlinePoints = spinalMidlinePoints[np.argsort(spinalMidlinePoints[:, 1])]
    print(f"✅ Detected {len(spinalMidlinePoints)} midline points.")

    # Smooth the midline to remove zigzag for the final visualization
    sigma = 3.0
    spinalMidlinePoints[:, 0] = gaussian_filter1d(spinalMidlinePoints[:, 0], sigma=sigma)
    spinalMidlinePoints[:, 2] = gaussian_filter1d(spinalMidlinePoints[:, 2], sigma=sigma)
    print(f"✅ Applied Gaussian smoothing with sigma={sigma} to final midline path.")

    return spinalMidlinePoints, debug_vis_data


# ==============================================================
# 4. 🧩 NEW: TOP-DOWN IMAGE SAVING FUNCTION
# ==============================================================

def save_top_down_view(slice_data, filename):
    """
    Creates a top-down visualization of a single slice and saves it to a file.
    """
    # 1. Create geometry objects for the slice
    slice_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slice_data["points_orig"]))
    slice_pcd.paint_uniform_color([1.0, 0.8, 0.0]) # Yellow

    ref_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(slice_data["ref_line_points"]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ref_line.paint_uniform_color([0.0, 0.2, 1.0]) # Blue

    valley_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.5) # Slightly larger sphere
    valley_sphere.translate(slice_data["deepest_point"])
    valley_sphere.paint_uniform_color([0.0, 0.8, 0.0]) # Green

    # 2. Setup a non-interactive visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, visible=False) # Create an off-screen window
    vis.add_geometry(slice_pcd)
    #vis.add_geometry(ref_line)
    vis.add_geometry(valley_sphere)

    # 3. Set the camera to a perfect top-down view
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Center the view on the detected green point
    center = slice_data["deepest_point"]

    # Camera position: directly above the center point along the Y-axis
    camera_position = center + np.array([0, 300, 0]) # 300mm above

    # Set camera properties
    # The camera looks from its position towards the center
    # The 'up' vector defines the top of the image (we use -Z to orient the back 'up')
    view_control.set_lookat(center)
    view_control.set_front(-(camera_position - center))
    view_control.set_up([0, 0, -1])
    view_control.set_zoom(1.2) # Adjust zoom to fit the slice nicely

    # 4. Capture the image and clean up
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()


# ==============================================================
# 5. MAIN EXECUTION AND VISUALIZATION (Modified)
# ==============================================================

def main():
    ply_path = "05310122013_back.ply" # 👈 Make sure this file exists
    output_dir = "single_cross_section_top_views"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 Created directory: {output_dir}")

    print(f"\nProcessing file: {ply_path}")
    backMesh = o3d.io.read_triangle_mesh(ply_path)
    if len(backMesh.vertices) == 0:
        print("Error: Could not load mesh.")
        return

    backMesh.compute_vertex_normals()

    spinalMidlinePoints, debug_vis_data = detect_spinal_midline_projection_method(
        backMesh,
        slice_thickness=1.5,
        num_slices=150,
        top_cut=0.17,
        bottom_cut=0.15,
    )

    if len(spinalMidlinePoints) < 5:
        print("Failed: too few points detected.")
        return

    # --- 🧩 NEW: Loop to save top-down images ---
    if debug_vis_data:
        print(f"\n📸 Saving {len(debug_vis_data)} top-down views to '{output_dir}'...")
        for i, data in enumerate(debug_vis_data):
            filename = os.path.join(output_dir, f"slice_{i:02d}.png")
            save_top_down_view(data, filename)
        print("✅ Finished saving top-down images.")

    # --- Main 3D Visualization (as before) ---
    vis_geometries = []
    backMesh.paint_uniform_color([0.6, 0.6, 0.6])
    vis_geometries.append(backMesh)

    if debug_vis_data:
        print(f"\nPreparing main 3D visualization...")
        for data in debug_vis_data:
            # Add geometries for the final 3D view
            slice_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points_orig"]))
            slice_pcd.paint_uniform_color([1.0, 0.8, 0.0])
            vis_geometries.append(slice_pcd)

            ref_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(data["ref_line_points"]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            ref_line.paint_uniform_color([0.0, 0.2, 1.0])
            vis_geometries.append(ref_line)

            valley_sphere = o3d.geometry.TriangleMesh.create_sphere(2.0)
            valley_sphere.translate(data["deepest_point"])
            valley_sphere.paint_uniform_color([0.0, 0.8, 0.0])
            vis_geometries.append(valley_sphere)

    # Draw the final smoothed black midline path (lifted off the mesh to avoid z-fighting)
    points_with_offset = _lift_path_above_mesh(spinalMidlinePoints, backMesh, offset=3.0)
    final_midline_path = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_with_offset),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points_with_offset) - 1)]),
    )
    final_midline_path.paint_uniform_color([0, 0, 0])
    vis_geometries.append(final_midline_path)

    print("✅ Displaying final 3D visualization.")
    o3d.visualization.draw_geometries(
        vis_geometries,
        window_name="SPINAL MIDLINE V8.0 (Final 3D View)",
    )

main()
