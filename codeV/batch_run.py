"""
Enhanced spinal midline detection with robust C7 and PSIS landmark identification.
INTEGRATED VERSION: V1's spinal midline formation + V2's bounded PSIS detection
Optimized for various back types including scoliosis and deformities.
"""
import glob
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
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
    Enhanced for different back types including scoliotic spines.
    """
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    return {
        'width': x_range,
        'height': y_range, 
        'depth': z_range,
        'midline_radius': x_range * 0.08,
        'psis_min_separation': x_range * 0.10,  # Reduced for closer PSIS in some cases
        'psis_max_separation': x_range * 0.35,  # Added max separation constraint
        'c7_posterior_threshold': z_range * 0.7,
        'psis_search_height': y_range * 0.30,   # Expanded search height for PSIS
        'psis_lateral_tolerance': x_range * 0.25 # Increased lateral tolerance for curved spines
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


def create_adaptive_psis_search_region(mesh, spinal_midline_points, dimensions):
    """
    ENHANCED: Create a more adaptive bounded search region for PSIS detection.
    Handles curved spines (scoliosis) and varying back shapes better.
    """
    vertices = np.asarray(mesh.vertices)
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_range = y_max - y_min
    
    # ENHANCED Y-axis bounds - more generous for different back lengths
    y_lower = y_min
    y_upper = y_min + dimensions['psis_search_height']  # Use adaptive height
    
    # Get midline reference in lower back - MORE ROBUST for curved spines
    lower_midline = spinal_midline_points[
        spinal_midline_points[:, 1] <= y_upper
    ]
    
    if len(lower_midline) == 0:
        # Fallback to overall midline
        midline_x = np.mean(spinal_midline_points[:, 0])
        midline_z = np.mean(spinal_midline_points[:, 2])
    else:
        # For curved spines, use WEIGHTED average - lower points have more weight
        weights = np.exp(-(lower_midline[:, 1] - y_min) / (y_range * 0.1))  # Exponential weighting
        midline_x = np.average(lower_midline[:, 0], weights=weights)
        midline_z = np.average(lower_midline[:, 2], weights=weights)
    
    # ENHANCED X-axis bounds - more tolerant for asymmetric/curved backs
    psis_lateral_min = dimensions['width'] * 0.06   # Reduced minimum (closer to midline possible)
    psis_lateral_max = dimensions['psis_lateral_tolerance']  # Increased maximum
    
    x_min = midline_x - psis_lateral_max
    x_max = midline_x + psis_lateral_max
    
    # ENHANCED Z-axis bounds - more generous depth range
    z_range_psis = dimensions['depth'] * 0.40  # Increased from 0.25
    z_min = midline_z - dimensions['depth'] * 0.20  # More anterior tolerance
    z_max = midline_z + z_range_psis  # More posterior tolerance
    
    return {
        'min_bound': np.array([x_min, y_lower, z_min]),
        'max_bound': np.array([x_max, y_upper, z_max]),
        'midline_x': midline_x,
        'midline_z': midline_z,
        'psis_lateral_min': psis_lateral_min,
        'psis_lateral_max': psis_lateral_max,
        'lower_midline_points': lower_midline
    }


def detect_psis_landmarks_enhanced_bounded(mesh, curvatures, spinal_midline_points, dimensions):
    """
    ENHANCED PSIS detection with improved bounding box and adaptive filtering.
    Better handles scoliotic and asymmetric backs.
    """
    vertices = np.asarray(mesh.vertices)
    
    # Create enhanced search region
    search_region = create_adaptive_psis_search_region(
        mesh, spinal_midline_points, dimensions
    )
    
    print(f"  Enhanced PSIS search region:")
    print(f"    X: [{search_region['min_bound'][0]:.1f}, {search_region['max_bound'][0]:.1f}]")
    print(f"    Y: [{search_region['min_bound'][1]:.1f}, {search_region['max_bound'][1]:.1f}]")
    print(f"    Z: [{search_region['min_bound'][2]:.1f}, {search_region['max_bound'][2]:.1f}]")
    print(f"    Weighted midline X: {search_region['midline_x']:.1f}")
    print(f"    Lateral distance range: [{search_region['psis_lateral_min']:.1f}, {search_region['psis_lateral_max']:.1f}]")
    
    # Filter vertices within bounding box
    bbox_mask = (
        (vertices[:, 0] >= search_region['min_bound'][0]) &
        (vertices[:, 0] <= search_region['max_bound'][0]) &
        (vertices[:, 1] >= search_region['min_bound'][1]) &
        (vertices[:, 1] <= search_region['max_bound'][1]) &
        (vertices[:, 2] >= search_region['min_bound'][2]) &
        (vertices[:, 2] <= search_region['max_bound'][2])
    )
    
    print(f"  Vertices in bounding box: {np.sum(bbox_mask)}")
    
    # Combine with negative curvature requirement
    concave_mask = curvatures < 0
    candidate_mask = bbox_mask & concave_mask
    
    print(f"  Vertices with negative curvature in box: {np.sum(candidate_mask)}")
    
    if not np.any(candidate_mask):
        print("  No PSIS candidates within bounded search region.")
        return None, None
    
    candidate_vertices = vertices[candidate_mask]
    candidate_curvatures = curvatures[candidate_mask]
    
    # ENHANCED filtering for strong negative curvature (adaptive percentile)
    if len(candidate_curvatures) > 20:
        curvature_threshold = np.percentile(candidate_curvatures, 5)  # Top 5% most concave
    elif len(candidate_curvatures) > 10:
        curvature_threshold = np.percentile(candidate_curvatures, 10) # Top 10% most concave
    else:
        curvature_threshold = np.percentile(candidate_curvatures, 20) # Top 20% most concave
    
    strong_mask = candidate_curvatures <= curvature_threshold
    candidate_vertices = candidate_vertices[strong_mask]
    candidate_curvatures = candidate_curvatures[strong_mask]
    print(f"  Strong negative curvature candidates: {len(candidate_vertices)}")
    
    if len(candidate_vertices) < 1:  # Reduced from 2 to 1 for single PSIS detection
        print("  Insufficient strong concave candidates for PSIS.")
        return None, None
    
    # ENHANCED constraint: PSIS must be at minimum distance from midline
    midline_x = search_region['midline_x']
    lateral_distance = np.abs(candidate_vertices[:, 0] - midline_x)
    min_distance_mask = (
        lateral_distance >= search_region['psis_lateral_min']
    )
    
    candidate_vertices = candidate_vertices[min_distance_mask]
    candidate_curvatures = candidate_curvatures[min_distance_mask]
    
    print(f"  Candidates at sufficient lateral distance: {len(candidate_vertices)}")
    
    if len(candidate_vertices) < 1:  # Reduced from 2 to 1
        print("  No candidates at sufficient lateral distance from midline.")
        return None, None
    
    # ENHANCED split into left and right candidates - more robust for asymmetric backs
    left_mask = candidate_vertices[:, 0] < midline_x
    right_mask = candidate_vertices[:, 0] > midline_x
    
    left_candidates = candidate_vertices[left_mask]
    left_curvatures = candidate_curvatures[left_mask]
    right_candidates = candidate_vertices[right_mask]
    right_curvatures = candidate_curvatures[right_mask]
    
    print(f"  Left candidates: {len(left_candidates)}, Right candidates: {len(right_candidates)}")
    
    psis_left, psis_right = None, None
    
    # Select deepest concavity on each side with additional proximity filtering
    if len(left_candidates) > 0:
        # For multiple candidates, prefer those closer to the expected PSIS region
        if len(search_region['lower_midline_points']) > 0:
            # Weight by proximity to lower spine
            lower_spine_y = np.mean(search_region['lower_midline_points'][:, 1])
            proximity_weights = np.exp(-np.abs(left_candidates[:, 1] - lower_spine_y) / (dimensions['height'] * 0.1))
            combined_scores = left_curvatures * -1 + proximity_weights * 0.1  # Negative curvature + proximity bonus
            deepest_idx = np.argmax(combined_scores)
        else:
            deepest_idx = np.argmin(left_curvatures)
        psis_left = left_candidates[deepest_idx]
        print(f"  PSIS Left: curvature = {left_curvatures[deepest_idx]:.4f}")
    
    if len(right_candidates) > 0:
        # Same proximity filtering for right side
        if len(search_region['lower_midline_points']) > 0:
            lower_spine_y = np.mean(search_region['lower_midline_points'][:, 1])
            proximity_weights = np.exp(-np.abs(right_candidates[:, 1] - lower_spine_y) / (dimensions['height'] * 0.1))
            combined_scores = right_curvatures * -1 + proximity_weights * 0.1
            deepest_idx = np.argmax(combined_scores)
        else:
            deepest_idx = np.argmin(right_curvatures)
        psis_right = right_candidates[deepest_idx]
        print(f"  PSIS Right: curvature = {right_curvatures[deepest_idx]:.4f}")
    
    # ENHANCED symmetry check - more tolerant for scoliotic backs
    if psis_left is not None and psis_right is not None:
        separation = np.linalg.norm(psis_left - psis_right)
        print(f"  PSIS separation: {separation:.1f}mm")
        
        # More flexible separation constraints
        expected_min = dimensions['psis_min_separation']
        expected_max = dimensions['psis_max_separation']
        
        if separation < expected_min:
            print(f"  Warning: PSIS separation too small ({separation:.1f}mm < {expected_min:.1f}mm)")
            # Keep the one with deeper curvature
            if np.min(left_curvatures) < np.min(right_curvatures):
                psis_right = None
            else:
                psis_left = None
        elif separation > expected_max:
            print(f"  Warning: PSIS separation too large ({separation:.1f}mm > {expected_max:.1f}mm)")
            # Still keep both - might be valid for very wide or scoliotic backs
    
    return psis_left, psis_right


def trim_midline_endpoints(spinal_midline_points, trim_fraction=0.08):
    """FROM V1 - Simple endpoint trimming"""
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
    """FROM V1 - Simple Gaussian smoothing"""
    if len(midline_points) < 5:
        return midline_points
    smoothed = midline_points.copy()
    for i in range(3):
        smoothed[:, i] = gaussian_filter1d(midline_points[:, i], sigma=sigma, mode='nearest')
    return smoothed


def refine_midline_between_landmarks(spinal_midline_points, c7_point, psis_points):
    """FROM V1 - Simple landmark-based refinement"""
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


def main():
    import os
    from pathlib import Path
    os.makedirs("result", exist_ok=True)

    for ply_path in sorted(glob.glob("backROI/*.ply")):
        print(f"\nProcessing {ply_path}")
        
        backMesh = o3d.io.read_triangle_mesh(ply_path)
        
        if len(backMesh.vertices) == 0:
            print(f"Error: Could not load mesh or mesh is empty for {ply_path}. Skipping.")
            continue
            
        backMesh.compute_vertex_normals()
        
        vertices = np.asarray(backMesh.vertices)
        dimensions = get_adaptive_torso_dimensions(vertices)
        print(f"  Enhanced adaptive dimensions - Width: {dimensions['width']:.1f}mm, "
              f"Height: {dimensions['height']:.1f}mm, "
              f"PSIS search height: {dimensions['psis_search_height']:.1f}mm")
        print(f"  Midline radius: {dimensions['midline_radius']:.1f}mm, "
              f"PSIS separation: [{dimensions['psis_min_separation']:.1f}, {dimensions['psis_max_separation']:.1f}]mm")
        
        # === V1's SPINAL MIDLINE FORMATION (Simple dot-connecting) ===
        print("\nV1: Sampling points and detecting spinal midline...")
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
            
            if len(pointsInSlice) < 10:
                continue

            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(pointsInSlice)
            slice_pcd = slice_pcd.voxel_down_sample(voxel_size=1.0)
            downsampledPoints = np.asarray(slice_pcd.points)
            
            if len(downsampledPoints) < 3: 
                continue
            
            pca = PCA(n_components=2)
            sliceProjectedCoords = pca.fit_transform(downsampledPoints)
            acrossBack = sliceProjectedCoords[:, 0]
            depth = sliceProjectedCoords[:, 1]
            xMean = acrossBack.mean()
            xStd = acrossBack.std() + 1e-9
            acrossBackNormalized = (acrossBack - xMean) / xStd
            designMatrix = np.column_stack([acrossBackNormalized**2,
                                            acrossBackNormalized,
                                            np.ones_like(acrossBackNormalized)])
            try:
                a, b, c = np.linalg.lstsq(designMatrix, depth, rcond=None)[0]
                isFlipped = False
                if a <= 0:
                    a_flipped, b_flipped, c_flipped = np.linalg.lstsq(designMatrix, -depth, rcond=None)[0]
                    if a_flipped > 0:
                        a, b, c = a_flipped, b_flipped, c_flipped
                        isFlipped = True
                    else:
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
            except (np.linalg.LinAlgError, ValueError):
                continue
            
        if len(spinalMidlinePoints) < 5: 
            print(f"  Warning: Too few spinal midline points ({len(spinalMidlinePoints)}) detected for {ply_path}. Skipping landmark detection.")
            backMesh.paint_uniform_color([0.6, 0.6, 0.6])
            geoms = [backMesh] 
            out_png = Path("result") / (Path(ply_path).stem + ".png")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            for g in geoms:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(out_png))
            vis.destroy_window()
            print(f"  Saved {out_png} (no midline/landmarks due to insufficient data)")
            continue
            
        # V1's simple processing
        spinalMidlinePoints = np.array(spinalMidlinePoints)
        spinalMidlinePoints = spinalMidlinePoints[np.argsort(spinalMidlinePoints[:, 1])]
        spinalMidlinePoints = trim_midline_endpoints(spinalMidlinePoints, trim_fraction=0.08)
        
        if len(spinalMidlinePoints) >= 5:
            spinalMidlinePoints = smooth_midline_robust(spinalMidlinePoints, sigma=2.0)
        else:
            print(f"  Warning: Midline too short after trimming for {ply_path}. Skipping landmark detection.")
            backMesh.paint_uniform_color([0.6, 0.6, 0.6])
            geoms = [backMesh]
            out_png = Path("result") / (Path(ply_path).stem + ".png")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            for g in geoms:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(out_png))
            vis.destroy_window()
            print(f"  Saved {out_png} (no midline/landmarks)")
            continue

        print(f"V1: Generated {len(spinalMidlinePoints)} midline points (simple dot connection)")

        # === LANDMARK DETECTION ===
        print("\nCalculating signed Gaussian curvature...")
        curvatures = calculate_gaussian_curvature(backMesh, neighborhood_size=25)
        
        print("\nDetecting C7 landmark...")
        c7_point = detect_c7_landmark(backMesh, curvatures, spinalMidlinePoints, dimensions)
        
        print("\nV2: Detecting PSIS landmarks with ENHANCED bounded search...")
        # === V2's ENHANCED BOUNDED PSIS DETECTION ===
        psis_left, psis_right = detect_psis_landmarks_enhanced_bounded(
            backMesh, curvatures, spinalMidlinePoints, dimensions
        )
        
        refined_midline = refine_midline_between_landmarks(spinalMidlinePoints, c7_point, [psis_left, psis_right])

        # Build scene
        backMesh.paint_uniform_color([0.6, 0.6, 0.6])
        geoms = [backMesh]
        
        if len(refined_midline) >= 2:
            midline_geom = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(refined_midline),
                lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(refined_midline)-1)])
            ).paint_uniform_color([1, 0, 0])
            geoms.append(midline_geom)
        else:
            print(f"  Warning: Final refined midline for {ply_path} has too few points to render.")

        if c7_point is not None:
            c7_sphere = o3d.geometry.TriangleMesh.create_sphere(5).translate(c7_point).paint_uniform_color([0, 1, 0])
            geoms.append(c7_sphere)
            print(f"  C7 detected at: [{c7_point[0]:.1f}, {c7_point[1]:.1f}, {c7_point[2]:.1f}]")
        else:
            print("  C7 not detected.")

        if psis_left is not None:
            psis_left_sphere = o3d.geometry.TriangleMesh.create_sphere(4).translate(psis_left).paint_uniform_color([0, 0, 1])
            geoms.append(psis_left_sphere)
            print(f"  PSIS Left detected at: [{psis_left[0]:.1f}, {psis_left[1]:.1f}, {psis_left[2]:.1f}]")
        else:
            print("  Left PSIS not detected.")
            
        if psis_right is not None:
            psis_right_sphere = o3d.geometry.TriangleMesh.create_sphere(4).translate(psis_right).paint_uniform_color([0, 0, 1])
            geoms.append(psis_right_sphere)
            print(f"  PSIS Right detected at: [{psis_right[0]:.1f}, {psis_right[1]:.1f}, {psis_right[2]:.1f}]")
        else:
            print("  Right PSIS not detected.")

        out_png = Path("result") / (Path(ply_path).stem + ".png")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_png))
        vis.destroy_window()
        print(f"  Saved {out_png}")

if __name__ == "__main__":
    main()