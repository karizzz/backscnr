import glob
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d


def calculate_gaussian_curvature(mesh, neighborhood_size=20):
    """
    Calculate SIGNED Gaussian curvature for each vertex using local surface fitting.
    Preserves the sign to distinguish convex (positive) from concave (negative) regions.
    """
    vertices = np.asarray(mesh.vertices)
    curvatures = np.zeros(len(vertices))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for i, vertex in enumerate(vertices):
        [k, idx, _] = kdtree.search_knn_vector_3d(vertex, neighborhood_size)
        
        if k < 10:
            continue
            
        neighbors = vertices[idx[1:]]
        centered_neighbors = neighbors - vertex
        
        if len(centered_neighbors) >= 6:
            pca = PCA(n_components=3)
            pca.fit(centered_neighbors)
            local_coords = pca.transform(centered_neighbors)
            x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]
            A = np.column_stack([x**2, y**2, x*y, x, y, np.ones(len(x))])
            
            try:
                coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
                a, b, c = coeffs[0], coeffs[1], coeffs[2]
                gaussian_curvature = a * b - (c/2)**2
                curvatures[i] = gaussian_curvature
            except np.linalg.LinAlgError:
                curvatures[i] = 0
    
    return curvatures


def get_adaptive_torso_dimensions(vertices):
    """
    Calculate adaptive dimensions based on the torso size for robust thresholding.
    """
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    return {
        'width': x_range,
        'height': y_range, 
        'depth': z_range,
        'midline_radius': x_range * 0.08,
        'psis_min_separation': x_range * 0.12,
        'c7_posterior_threshold': z_range * 0.7,
        'dbscan_eps': x_range * 0.05
    }


def detect_c7_landmark(mesh, curvatures, spinal_midline_points, dimensions):
    """
    Detect C7 vertebra using anatomical constraints:
    1. Must have POSITIVE curvature (convex bump)
    2. Must be near the spinal midline
    3. From candidates, select the most posterior (largest z-value)
    """
    vertices = np.asarray(mesh.vertices)
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_range = y_max - y_min
    
    upper_threshold = y_max - 0.25 * y_range
    upper_mask = vertices[:, 1] >= upper_threshold
    
    if not np.any(upper_mask):
        return None
    
    positive_curvature_mask = curvatures > 0
    candidate_mask = upper_mask & positive_curvature_mask
    
    if not np.any(candidate_mask):
        return None
    
    candidate_vertices = vertices[candidate_mask]
    candidate_curvatures = curvatures[candidate_mask]
    
    if len(spinal_midline_points) > 0:
        upper_midline = spinal_midline_points[spinal_midline_points[:, 1] >= upper_threshold]
        
        if len(upper_midline) > 0:
            avg_x = np.mean(upper_midline[:, 0])
            avg_z = np.mean(upper_midline[:, 2])
            
            distances = np.sqrt((candidate_vertices[:, 0] - avg_x)**2 + 
                                (candidate_vertices[:, 2] - avg_z)**2)
            midline_mask = distances <= dimensions['midline_radius']
            
            if np.any(midline_mask):
                candidate_vertices = candidate_vertices[midline_mask]
                candidate_curvatures = candidate_curvatures[midline_mask]
    
    if len(candidate_vertices) == 0:
        return None
    
    if len(candidate_curvatures) > 0:
        curvature_threshold = np.percentile(candidate_curvatures, 90)
        strong_candidates_mask = candidate_curvatures >= curvature_threshold
    else:
        return None

    if not np.any(strong_candidates_mask):
        if len(candidate_curvatures) > 0:
            max_idx = np.argmax(candidate_curvatures)
            return candidate_vertices[max_idx]
        else:
            return None

    strong_candidates = candidate_vertices[strong_candidates_mask]
    
    if len(strong_candidates) > 0:
        most_posterior_idx = np.argmax(strong_candidates[:, 2])
        c7_point = strong_candidates[most_posterior_idx]
        return c7_point
    else:
        return None


def detect_psis_landmarks(mesh, curvatures, spinal_midline_points, dimensions):
    """
    Detect PSIS dimples using anatomical constraints:
    1. Must have NEGATIVE curvature (concave dimples) - using an ADAPTIVE threshold.
    2. Must be in lower back region.
    3. Must be symmetrically positioned relative to midline.
    """
    vertices = np.asarray(mesh.vertices)
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_range = y_max - y_min
    
    lower_threshold = y_min + 0.20 * y_range
    lower_mask = vertices[:, 1] <= lower_threshold
    
    initial_concave_mask = curvatures < 0
    candidate_mask = lower_mask & initial_concave_mask
    
    if not np.any(candidate_mask):
        print("  No negative curvature candidates found for PSIS.")
        return None, None
    
    all_concave_vertices = vertices[candidate_mask]
    all_concave_curvatures = curvatures[candidate_mask]

    if len(all_concave_curvatures) > 10:
        strong_curvature_threshold = np.percentile(all_concave_curvatures, 10)
        strong_mask = all_concave_curvatures <= strong_curvature_threshold
        
        candidate_vertices = all_concave_vertices[strong_mask]
        candidate_curvatures = all_concave_curvatures[strong_mask]
    else:
        candidate_vertices = all_concave_vertices
        candidate_curvatures = all_concave_curvatures

    if len(candidate_vertices) < 2:
        print("  Not enough strong negative curvature candidates for PSIS.")
        return None, None

    if len(spinal_midline_points) == 0:
        print("  No spinal midline available for PSIS symmetry analysis.")
        return None, None
    
    lower_midline = spinal_midline_points[spinal_midline_points[:, 1] <= lower_threshold]
    midline_x = np.mean(lower_midline[:, 0]) if len(lower_midline) > 0 else np.mean(spinal_midline_points[:, 0])
    
    left_mask = candidate_vertices[:, 0] < midline_x
    right_mask = candidate_vertices[:, 0] > midline_x
    
    left_candidates = candidate_vertices[left_mask]
    left_curvatures = candidate_curvatures[left_mask]
    right_candidates = candidate_vertices[right_mask]
    right_curvatures = candidate_curvatures[right_mask]
    
    psis_left, psis_right = None, None
    
    if len(left_candidates) > 0:
        deepest_left_idx = np.argmin(left_curvatures)
        psis_left = left_candidates[deepest_left_idx]
    
    if len(right_candidates) > 0:
        deepest_right_idx = np.argmin(right_curvatures)
        psis_right = right_candidates[deepest_right_idx]
    
    if psis_left is not None and psis_right is not None:
        separation = np.linalg.norm(psis_left - psis_right)
        
        if separation < dimensions['psis_min_separation']:
            if left_curvatures[deepest_left_idx] < right_curvatures[deepest_right_idx]:
                psis_right = None 
            else:
                psis_left = None
                
    return psis_left, psis_right


def trim_midline_endpoints(spinal_midline_points, trim_fraction=0.08):
    if len(spinal_midline_points) < 10:
        return spinal_midline_points
    if spinal_midline_points.shape[0] == 0:
        return spinal_midline_points
    sorted_indices = np.argsort(spinal_midline_points[:, 1])
    sorted_midline = spinal_midline_points[sorted_indices]
    n_points = len(sorted_midline)
    n_trim = max(1, int(n_points * trim_fraction))
    if n_trim * 2 >= n_points:
        return np.array([])
    trimmed_midline = sorted_midline[n_trim:-n_trim]
    return trimmed_midline


def smooth_midline_robust(midline_points, sigma=2.0):
    if len(midline_points) < 5:
        return midline_points
    smoothed = midline_points.copy()
    for i in range(3):
        smoothed[:, i] = gaussian_filter1d(midline_points[:, i], sigma=sigma, mode='nearest')
    return smoothed


def refine_midline_between_landmarks(spinal_midline_points, c7_point, psis_points):
    if c7_point is None and all(p is None for p in psis_points):
        return spinal_midline_points
    c7_y = c7_point[1] if c7_point is not None else -np.inf
    psis_y_coords = [p[1] for p in psis_points if p is not None]
    psis_y = np.mean(psis_y_coords) if psis_y_coords else np.inf
    if c7_point is None and not psis_y_coords:
        return spinal_midline_points
    y_upper = max(c7_y, psis_y)
    y_lower = min(c7_y, psis_y)
    if y_upper > y_lower:
        y_buffer = (y_upper - y_lower) * 0.05
        y_upper += y_buffer
        y_lower -= y_buffer
    else:
        y_upper += 5.0
        y_lower -= 5.0
    mask = (spinal_midline_points[:, 1] >= y_lower) & (spinal_midline_points[:, 1] <= y_upper)
    refined_midline = spinal_midline_points[mask]
    if len(refined_midline) < 5:
        return spinal_midline_points
    return refined_midline


def project_points_to_mesh_surface(points, mesh_vertices_np, mesh_kdtree):
    """
    Project a set of points onto the nearest vertex of a mesh.
    Returns the projected points that lie on the mesh surface.
    """
    projected_points_on_vertices = []
    for point in points:
        [k, idx, _] = mesh_kdtree.search_knn_vector_3d(point, 1)
        if k > 0:
            projected_points_on_vertices.append(mesh_vertices_np[idx[0]])
        else:
            projected_points_on_vertices.append(point) # Fallback if no closest point found

    return np.array(projected_points_on_vertices)


def get_normals_at_points(query_points, mesh_vertices_np, mesh_normals_np, mesh_kdtree_for_normals):
    """
    Finds the normal vector at each query point by looking up the normal of the closest mesh vertex.
    Includes a fallback to a default normal if closest vertex is not found or has zero normal.
    """
    normals = []
    default_normal = np.array([0, 0, 1.0]) # Pushes out in positive Z direction

    for q_point in query_points:
        [k, idx, _] = mesh_kdtree_for_normals.search_knn_vector_3d(q_point, 1)
        if k > 0:
            vertex_normal = mesh_normals_np[idx[0]]
            if np.linalg.norm(vertex_normal) > 1e-6:
                normals.append(vertex_normal)
            else:
                normals.append(default_normal)
        else:
            normals.append(default_normal)

    return np.array(normals)


def main():
    # --- HARDCODED FILE FOR SINGLE VIEWING ---
    backMesh = o3d.io.read_triangle_mesh("05420022014_back.ply")
    
    if len(backMesh.vertices) == 0:
        print("Error: Could not load mesh or mesh is empty")
        return
        
    backMesh.compute_vertex_normals()
    ply_path_name = "08612012017_back.ply" 
    print(f"\nProcessing {ply_path_name}")
    
    # --- Create KDTree for mesh vertices and normals ---
    mesh_vertices_np = np.asarray(backMesh.vertices)
    mesh_normals_np = np.asarray(backMesh.vertex_normals)
    mesh_pcd_for_projection = o3d.geometry.PointCloud()
    mesh_pcd_for_projection.points = backMesh.vertices
    mesh_kdtree_for_projection = o3d.geometry.KDTreeFlann(mesh_pcd_for_projection)

    # --- Adaptive dimensions ---
    vertices = np.asarray(backMesh.vertices)
    dimensions = get_adaptive_torso_dimensions(vertices)
    print(f"Adaptive dimensions - Width: {dimensions['width']:.1f}, "
          f"Midline radius: {dimensions['midline_radius']:.1f}, "
          f"PSIS min separation: {dimensions['psis_min_separation']:.1f}")
    
    # --- Sample points and slice ---
    backPointcloud = backMesh.sample_points_uniformly(number_of_points=50000)
    allPoints = np.asarray(backPointcloud.points)
    
    y_min, y_max = allPoints[:, 1].min(), allPoints[:, 1].max()
    slice_thickness = 3.0
    num_slices = 50
    sliceHeights = np.linspace(y_min, y_max, num=num_slices)
    
    spinalMidlinePoints = []
    
    for current_y in sliceHeights:
        mask = (allPoints[:, 1] >= current_y - slice_thickness / 2) & \
               (allPoints[:, 1] < current_y + slice_thickness / 2)
        pointsInSlice = allPoints[mask]
        if len(pointsInSlice) < 50:
            continue

        avg_z_slice = np.mean(pointsInSlice[:, 2])
        posterior_mask = pointsInSlice[:, 2] >= avg_z_slice - (dimensions['depth'] * 0.05)
        pointsInSlice = pointsInSlice[posterior_mask]
        
        if len(pointsInSlice) < 10:
            continue
        
        slice_pcd = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(pointsInSlice)
        slice_pcd = slice_pcd.voxel_down_sample(voxel_size=1.0)
        
        downsampledPoints = np.asarray(slice_pcd.points)
        if len(downsampledPoints) < 5:
            continue
        
        # PCA valley fitting
        pca = PCA(n_components=2)
        sliceProjectedCoords = pca.fit_transform(downsampledPoints)
        acrossBack = sliceProjectedCoords[:, 0]
        depth = sliceProjectedCoords[:, 1]
        
        xMean = acrossBack.mean()
        xStd = acrossBack.std() + 1e-9
        acrossBackNormalized = (acrossBack - xMean) / xStd
        
        designMatrix = np.column_stack([acrossBackNormalized**2, acrossBackNormalized, np.ones_like(acrossBackNormalized)])
        
        try:
            a, b, c = np.linalg.lstsq(designMatrix, depth, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        
        isFlipped = False
        if a <= 0:
            try:
                a, b, c = np.linalg.lstsq(designMatrix, -depth, rcond=None)[0]
                isFlipped = True
            except np.linalg.LinAlgError:
                continue
        
        if abs(a) < 1e-6:
            continue
            
        vertexAcrossBackNormalized = -b / (2.0 * a)
        if abs(vertexAcrossBackNormalized) > 3.0:
            continue
            
        vertexDepth = a * vertexAcrossBackNormalized**2 + b * vertexAcrossBackNormalized + c
        if isFlipped:
            vertexDepth = -vertexDepth
        
        vertexAcrossBack = vertexAcrossBackNormalized * xStd + xMean
        valley3dCoords = pca.inverse_transform([[vertexAcrossBack, vertexDepth]])[0]
        spinalMidlinePoints.append([valley3dCoords[0], current_y, valley3dCoords[2]])
    
    # --- Smooth raw midline ---
    if len(spinalMidlinePoints) >= 5:
        spinalMidlinePoints = np.array(spinalMidlinePoints)
        spinalMidlinePoints = spinalMidlinePoints[np.argsort(spinalMidlinePoints[:, 1])]
        spinalMidlinePoints = trim_midline_endpoints(spinalMidlinePoints, trim_fraction=0.08)
        spinalMidlinePoints = smooth_midline_robust(spinalMidlinePoints, sigma=1.5)
    else:
        print("Warning: Too few spinal midline points detected. Exiting.")
        return
    
    # ==============================================================
    # 🧠 NEW DENOISED SURFACE PROJECTION
    # ==============================================================
    print("Projecting spinal midline onto mesh surface (denoised)...")

    projected_points_on_mesh = []
    smoothed_normals = []
    for pt in spinalMidlinePoints:
        [k, idx, _] = mesh_kdtree_for_projection.search_knn_vector_3d(pt, 10)
        nearest_vertices = mesh_vertices_np[idx]
        nearest_normals = mesh_normals_np[idx]

        avg_vertex = np.mean(nearest_vertices, axis=0)
        avg_normal = np.mean(nearest_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal) + 1e-8

        projected_points_on_mesh.append(avg_vertex)
        smoothed_normals.append(avg_normal)

    projected_points_on_mesh = np.array(projected_points_on_mesh)
    smoothed_normals = np.array(smoothed_normals)

    OFFSET_DISTANCE_MM = 1.5
    spinalMidlinePoints = projected_points_on_mesh + smoothed_normals * OFFSET_DISTANCE_MM

    # Final smoothing to eliminate jitter
    spinalMidlinePoints = smooth_midline_robust(spinalMidlinePoints, sigma=2.0)
    print(f"Projected and denoised {len(spinalMidlinePoints)} points.")
    # ==============================================================

    # --- Continue with curvature & landmarks ---
    print("Calculating signed Gaussian curvature...")
    curvatures = calculate_gaussian_curvature(backMesh, neighborhood_size=25)
    
    print("Detecting C7 landmark...")
    c7_point = detect_c7_landmark(backMesh, curvatures, spinalMidlinePoints, dimensions)
    
    print("Detecting PSIS landmarks...")
    psis_left, psis_right = detect_psis_landmarks(backMesh, curvatures, spinalMidlinePoints, dimensions)
    
    if c7_point is not None or psis_left is not None or psis_right is not None:
        print("Refining midline between landmarks...")
        refined_midline = refine_midline_between_landmarks(spinalMidlinePoints, c7_point, [psis_left, psis_right])
    else:
        refined_midline = spinalMidlinePoints
    
    # --- Visualization ---
    geometriesToRender = []
    backMesh.paint_uniform_color([0.6, 0.6, 0.6])
    geometriesToRender.append(backMesh)
    
    if len(refined_midline) >= 2:
        spinalMidline = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(refined_midline),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(refined_midline) - 1)])
        )
        spinalMidline.paint_uniform_color([1, 0, 0])
        geometriesToRender.append(spinalMidline)
        print(f"Final midline contains {len(refined_midline)} points")
    
    # --- Landmark spheres ---
    OFFSET_LANDMARK_DISTANCE_MM = OFFSET_DISTANCE_MM + 1.0 
    
    if c7_point is not None:
        [k, idx, _] = mesh_kdtree_for_projection.search_knn_vector_3d(c7_point, 5)
        avg_normal = np.mean(mesh_normals_np[idx], axis=0)
        avg_normal /= np.linalg.norm(avg_normal) + 1e-8
        c7_point_offset = c7_point + avg_normal * OFFSET_LANDMARK_DISTANCE_MM
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=6.0)
        sphere.translate(c7_point_offset)
        sphere.paint_uniform_color([0, 1, 0])
        geometriesToRender.append(sphere)
        print(f"C7 (green) at {c7_point_offset}")
    
    for psis_point, color in zip([psis_left, psis_right], [[0, 0, 1], [0, 0, 1]]):
        if psis_point is None:
            continue
        [k, idx, _] = mesh_kdtree_for_projection.search_knn_vector_3d(psis_point, 5)
        avg_normal = np.mean(mesh_normals_np[idx], axis=0)
        avg_normal /= np.linalg.norm(avg_normal) + 1e-8
        psis_offset = psis_point + avg_normal * OFFSET_LANDMARK_DISTANCE_MM
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
        sphere.translate(psis_offset)
        sphere.paint_uniform_color(color)
        geometriesToRender.append(sphere)
    
    print("\nLaunching visualization...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Denoised Spinal Midline", width=1200, height=800)
    for geom in geometriesToRender:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.line_width = 3.0
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


main()