import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import os # 👈 New import for creating directories

# ==============================================================
# 1. CORE UTILITIES (Unchanged)
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


# ==============================================================
# 2. CORE PROJECTION LOGIC (Unchanged)
# ==============================================================

def _find_deepest_point_by_projection(points_in_slice, store_vis=False):
    if len(points_in_slice) < 20:
        return None, None

    # Perform 2D PCA on the X and Z coordinates to flatten the slice
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points_in_slice[:, [0, 2]])
    x_pca = points_2d[:, 0]
    slice_width = x_pca.max() - x_pca.min()

    # Define a central search window
    primary_ratio = 0.25
    fallback_ratio = 0.50
    center_width = slice_width * primary_ratio
    center_mask = np.abs(x_pca - np.median(x_pca)) < (center_width / 2)
    # If the primary window is too sparse, expand it
    if np.sum(center_mask) < 5:
        center_width = slice_width * fallback_ratio
        center_mask = np.abs(x_pca - np.median(x_pca)) < (center_width / 2)

    center_points_original_coords = points_in_slice[center_mask]
    center_points_pca_coords = points_2d[center_mask]

    if len(center_points_original_coords) < 10:
        return None, None

    # "Deepest and Most Central" Logic: Find the most central point among the top 5 deepest
    num_candidates = 5
    if len(center_points_pca_coords) < num_candidates:
        num_candidates = len(center_points_pca_coords)

    candidate_indices = np.argsort(center_points_pca_coords[:, 1])[:num_candidates]
    candidate_x_coords = center_points_pca_coords[candidate_indices, 0]
    closest_to_center_idx_in_candidates = np.argmin(np.abs(candidate_x_coords))
    final_best_index = candidate_indices[closest_to_center_idx_in_candidates]
    deepest_point = center_points_original_coords[final_best_index]
    
    vis_data = None
    if store_vis:
        p1_local = np.array([x_pca.min(), 0])
        p2_local = np.array([x_pca.max(), 0])
        p1_orig = pca.inverse_transform(p1_local)
        p2_orig = pca.inverse_transform(p2_local)
        p1 = np.array([p1_orig[0], deepest_point[1], p1_orig[1]])
        p2 = np.array([p2_orig[0], deepest_point[1], p2_orig[1]])
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
    valid_min = y_min + top_cut * (y_max - y_min)
    valid_max = y_max - bottom_cut * (y_max - y_min)
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
    vis.add_geometry(ref_line)
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
    ply_path = "07521072017_back.ply" # 👈 Make sure this file exists
    output_dir = "cross_section_top_views"
    
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
            
    # Draw the final smoothed black midline path
    z_offset = 3.0
    points_with_offset = np.copy(spinalMidlinePoints)
    points_with_offset[:, 2] += z_offset

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


if __name__ == "__main__":
    main()