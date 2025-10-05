"""
Enhanced spinal midline detection with robust C7 and PSIS landmark identification.
Addresses issues with curvature sign, adaptive thresholds, and anatomical constraints.
This version is revised to fix midline detection issues in batch processing.
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
    
    # Build KD-tree for efficient neighbor search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for i, vertex in enumerate(vertices):
        # Find k-nearest neighbors
        [k, idx, _] = kdtree.search_knn_vector_3d(vertex, neighborhood_size)
        
        if k < 10:  # Need minimum neighbors for stable calculation
            continue
            
        neighbors = vertices[idx[1:]]  # Exclude the vertex itself
        
        # Center the neighborhood
        centered_neighbors = neighbors - vertex
        
        # Fit local quadratic surface z = ax² + by² + cxy + dx + ey + f
        # using PCA to establish local coordinate system
        if len(centered_neighbors) >= 6:
            pca = PCA(n_components=3)
            pca.fit(centered_neighbors)
            
            # Transform to local coordinates
            local_coords = pca.transform(centered_neighbors)
            
            # Fit quadratic surface in local coordinates
            x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]
            
            # Design matrix for quadratic fit
            A = np.column_stack([x**2, y**2, x*y, x, y, np.ones(len(x))])
            
            try:
                coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
                a, b, c = coeffs[0], coeffs[1], coeffs[2]
                
                # Gaussian curvature = (a*b - (c/2)²) for quadratic surface
                # KEEP THE SIGN - positive for convex (bumps), negative for concave (dimples)
                gaussian_curvature = a * b - (c/2)**2
                curvatures[i] = gaussian_curvature  # No abs() here!
                
            except np.linalg.LinAlgError:
                curvatures[i] = 0
    
    return curvatures

def get_adaptive_torso_dimensions(vertices):
    """
    Calculate adaptive dimensions based on the torso size for robust thresholding.
    """
    x_range = vertices[:, 0].max() - vertices[:, 0].min()  # Width
    y_range = vertices[:, 1].max() - vertices[:, 1].min()  # Height
    z_range = vertices[:, 2].max() - vertices[:, 2].min()  # Depth
    
    return {
        'width': x_range,
        'height': y_range, 
        'depth': z_range,
        'midline_radius': x_range * 0.08,  # 8% of torso width
        'psis_min_separation': x_range * 0.12,  # 12% of torso width
        'c7_posterior_threshold': z_range * 0.7  # Top 30% of depth range
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
    
    # Define upper back region (top 25% of height)
    upper_threshold = y_max - 0.25 * y_range
    upper_mask = vertices[:, 1] >= upper_threshold
    
    if not np.any(upper_mask):
        # print("No vertices found in upper back region for C7 detection") # Keep this silent for batch
        return None
    
    # Filter for POSITIVE curvature only (convex bumps)
    positive_curvature_mask = curvatures > 0
    candidate_mask = upper_mask & positive_curvature_mask
    
    if not np.any(candidate_mask):
        # print("No positive curvature candidates found for C7") # Keep this silent for batch
        return None
    
    candidate_vertices = vertices[candidate_mask]
    candidate_curvatures = curvatures[candidate_mask]
    
    # Filter candidates near the spinal midline
    # Ensure spinal_midline_points is not empty before attempting to filter
    if len(spinal_midline_points) > 0:
        # Get midline reference in upper region
        upper_midline = spinal_midline_points[spinal_midline_points[:, 1] >= upper_threshold]
        
        if len(upper_midline) > 0:
            avg_x = np.mean(upper_midline[:, 0])
            avg_z = np.mean(upper_midline[:, 2]) # Use this for distance calculation
            
            # Use adaptive radius instead of fixed 30mm
            distances = np.sqrt((candidate_vertices[:, 0] - avg_x)**2 + 
                              (candidate_vertices[:, 2] - avg_z)**2)
            midline_mask = distances <= dimensions['midline_radius']
            
            if np.any(midline_mask):
                candidate_vertices = candidate_vertices[midline_mask]
                candidate_curvatures = candidate_curvatures[midline_mask]
    
    if len(candidate_vertices) == 0:
        return None
    
    # From remaining candidates, select top 10% by curvature (strongest convex points)
    # Ensure there are enough points for percentile calculation
    if len(candidate_curvatures) > 0:
        curvature_threshold = np.percentile(candidate_curvatures, 90)
        strong_candidates_mask = candidate_curvatures >= curvature_threshold
    else: # If no curvatures left, no strong candidates
        strong_candidates_mask = np.array([], dtype=bool)

    
    if not np.any(strong_candidates_mask):
        # Fallback to absolute max curvature in case percentile fails or yields no points
        if len(candidate_curvatures) > 0:
            max_idx = np.argmax(candidate_curvatures)
            return candidate_vertices[max_idx]
        else:
            return None # No candidates left

    strong_candidates = candidate_vertices[strong_candidates_mask]
    
    # Among strong candidates, select the most posterior (anatomical constraint)
    # Ensure strong_candidates is not empty
    if len(strong_candidates) > 0:
        most_posterior_idx = np.argmax(strong_candidates[:, 2])
        c7_point = strong_candidates[most_posterior_idx]
        # print(f"C7 detected with positive curvature at: {c7_point}") # Keep this silent for batch
        return c7_point
    else:
        return None # No strong candidates with posterior value

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
    
    # Define lower back region 
    lower_threshold = y_min + 0.20 * y_range
    lower_mask = vertices[:, 1] <= lower_threshold
    
    # --- FIX: Use a two-step adaptive threshold instead of a fixed value ---
    
    # Step 1: Find ALL points with any negative curvature (concave points)
    initial_concave_mask = curvatures < 0
    candidate_mask = lower_mask & initial_concave_mask
    
    if not np.any(candidate_mask):
        print("  No negative curvature candidates found for PSIS.")
        return None, None
    
    all_concave_vertices = vertices[candidate_mask]
    all_concave_curvatures = curvatures[candidate_mask]

    if len(all_concave_curvatures) > 10: # Ensure enough points for a meaningful percentile
        strong_curvature_threshold = np.percentile(all_concave_curvatures, 10)
        strong_mask = all_concave_curvatures <= strong_curvature_threshold
        
        candidate_vertices = all_concave_vertices[strong_mask]
        candidate_curvatures = all_concave_curvatures[strong_mask]
    else:
        # If there are few candidates, just use them all
        candidate_vertices = all_concave_vertices
        candidate_curvatures = all_concave_curvatures

    if len(candidate_vertices) < 2:
        print("  Not enough strong negative curvature candidates for PSIS.")
        return None, None

    # Get midline reference for symmetry analysis
    if len(spinal_midline_points) == 0:
        print("  No spinal midline available for PSIS symmetry analysis.")
        return None, None
    
    lower_midline = spinal_midline_points[spinal_midline_points[:, 1] <= lower_threshold]
    midline_x = np.mean(lower_midline[:, 0]) if len(lower_midline) > 0 else np.mean(spinal_midline_points[:, 0])
    
    # Split candidates into left and right based on midline
    left_mask = candidate_vertices[:, 0] < midline_x
    right_mask = candidate_vertices[:, 0] > midline_x
    
    left_candidates = candidate_vertices[left_mask]
    left_curvatures = candidate_curvatures[left_mask]
    right_candidates = candidate_vertices[right_mask]
    right_curvatures = candidate_curvatures[right_mask]
    
    psis_left, psis_right = None, None
    
    # Find deepest dimple on left side
    if len(left_candidates) > 0:
        deepest_left_idx = np.argmin(left_curvatures)
        psis_left = left_candidates[deepest_left_idx]
    
    # Find deepest dimple on right side
    if len(right_candidates) > 0:
        deepest_right_idx = np.argmin(right_curvatures)
        psis_right = right_candidates[deepest_right_idx]
    
    # Validate separation
    if psis_left is not None and psis_right is not None:
        separation = np.linalg.norm(psis_left - psis_right)
        
        if separation < dimensions['psis_min_separation']:
            # If too close, keep only the one with the 'more negative' (deeper) curvature
            if left_curvatures[deepest_left_idx] < right_curvatures[deepest_right_idx]:
                psis_right = None 
            else:
                psis_left = None
                
    return psis_left, psis_right
def trim_midline_endpoints(spinal_midline_points, trim_fraction=0.08): # Increased trim_fraction slightly
    """
    Remove unstable endpoints from the spinal midline by trimming a fraction
    from both the top and bottom. This prevents midline from extending 
    outside the mesh geometry.
    """
    if len(spinal_midline_points) < 10: # Minimum points to attempt trimming
        return spinal_midline_points
    
    # Sort by y-coordinate to ensure proper ordering
    # Ensure it's not empty before sorting
    if spinal_midline_points.shape[0] == 0:
        return spinal_midline_points

    sorted_indices = np.argsort(spinal_midline_points[:, 1])
    sorted_midline = spinal_midline_points[sorted_indices]
    
    n_points = len(sorted_midline)
    n_trim = max(1, int(n_points * trim_fraction)) # Ensure at least 1 point trimmed if possible

    # Ensure n_trim does not exceed half the length of the midline
    if n_trim * 2 >= n_points:
        return np.array([]) # Return empty if trimming removes all points

    trimmed_midline = sorted_midline[n_trim:-n_trim]
    
    # print(f"Trimmed {n_trim} points from each end of midline ({n_points} -> {len(trimmed_midline)} points)") # Keep silent for batch
    return trimmed_midline

def smooth_midline_robust(midline_points, sigma=2.0): # Increased sigma for smoother results
    """
    Apply robust smoothing to the midline points using Gaussian filtering.
    """
    if len(midline_points) < 5:
        return midline_points
    
    smoothed = midline_points.copy()
    for i in range(3):  # Smooth x, y, z coordinates
        # Pad with 'edge' to handle endpoints more gracefully
        smoothed[:, i] = gaussian_filter1d(midline_points[:, i], sigma=sigma, mode='nearest')
    
    return smoothed

def refine_midline_between_landmarks(spinal_midline_points, c7_point, psis_points):
    """
    Refine the spinal midline to span between C7 and PSIS landmarks.
    """
    # If no C7 or no PSIS points, use the raw (trimmed and smoothed) midline
    if c7_point is None and all(p is None for p in psis_points):
        return spinal_midline_points
    
    c7_y = c7_point[1] if c7_point is not None else -np.inf # If C7 missing, treat as very low
    
    # Find the average y-coordinate of detected PSIS points, or -inf if none
    psis_y_coords = [p[1] for p in psis_points if p is not None]
    psis_y = np.mean(psis_y_coords) if psis_y_coords else np.inf # If PSIS missing, treat as very high
    
    # If both C7 and PSIS were missing, return original midline (already handled above)
    if c7_point is None and not psis_y_coords:
        return spinal_midline_points

    # Ensure proper ordering and add buffer
    y_upper = max(c7_y, psis_y)
    y_lower = min(c7_y, psis_y)
    
    # Add a small buffer to ensure we include the landmark regions in the refined midline
    # Only add buffer if there's a meaningful range to buffer
    if y_upper > y_lower:
        y_buffer = (y_upper - y_lower) * 0.05
        y_upper += y_buffer
        y_lower -= y_buffer
    else: # If C7 and PSIS are at similar y-levels (e.g., in a very short segment)
        y_upper += 5.0 # Add a small fixed buffer
        y_lower -= 5.0


    # Filter midline points within the landmark range
    mask = (spinal_midline_points[:, 1] >= y_lower) & (spinal_midline_points[:, 1] <= y_upper)
    refined_midline = spinal_midline_points[mask]
    
    # If refinement results in too few points, return the full (trimmed and smoothed) midline
    if len(refined_midline) < 5:
        # print("Warning: Refined midline too short, falling back to full trimmed/smoothed midline.") # Keep silent
        return spinal_midline_points

    return refined_midline

# Main execution code
def main():
    import os
    from pathlib import Path
    os.makedirs("result", exist_ok=True)

    for ply_path in sorted(glob.glob("backROI/*.ply")):
        print(f"\nProcessing {ply_path}") # Added newline for clarity in batch output
        
        backMesh = o3d.io.read_triangle_mesh(ply_path)
        
        if len(backMesh.vertices) == 0:
            print(f"Error: Could not load mesh or mesh is empty for {ply_path}. Skipping.")
            continue # Skip to next file
            
        backMesh.compute_vertex_normals()
        
        # Get adaptive dimensions based on torso size
        vertices = np.asarray(backMesh.vertices)
        dimensions = get_adaptive_torso_dimensions(vertices)
        print(f"  Adaptive dimensions - Width: {dimensions['width']:.1f}mm, "
              f"Midline radius: {dimensions['midline_radius']:.1f}mm, "
              f"PSIS min separation: {dimensions['psis_min_separation']:.1f}mm")
        
        backPointcloud = backMesh.sample_points_uniformly(number_of_points=50000)
        allPoints = np.asarray(backPointcloud.points)

        y_min, y_max = allPoints[:, 1].min(), allPoints[:, 1].max()
        slice_thickness = 3.0
        num_slices = 50
        sliceHeights = np.linspace(y_min, y_max, num=num_slices)

        sliceVizGeometries = []
        # --- FIX: Re-initialize spinalMidlinePoints for each new file ---
        spinalMidlinePoints = [] 

        for current_y in sliceHeights:
            mask = (allPoints[:, 1] >= current_y - slice_thickness / 2) & \
                   (allPoints[:, 1] < current_y + slice_thickness / 2)
            pointsInSlice = allPoints[mask]
            
            # Reverted to original or slightly relaxed thresholds for slice points
            if len(pointsInSlice) < 10: # Original was 10, I had it at 20, reverting to 10
                continue

            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(pointsInSlice)
            slice_pcd = slice_pcd.voxel_down_sample(voxel_size=1.0) # Reverted voxel_size for less aggressive downsampling
            downsampledPoints = np.asarray(slice_pcd.points)
            
            # Reverted to original or slightly relaxed thresholds for downsampled points
            if len(downsampledPoints) < 3: 
                continue
            
            
            #sliceVizGeometries.append(slice_pcd)

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
                # --- FIX: Restore the 'isFlipped' logic for robust valley detection ---
                a, b, c = np.linalg.lstsq(designMatrix, depth, rcond=None)[0]
                isFlipped = False
                if a <= 0: # If parabola opens upward in current PCA depth, flip the depth axis
                    # Attempt to refit with flipped depth
                    a_flipped, b_flipped, c_flipped = np.linalg.lstsq(designMatrix, -depth, rcond=None)[0]
                    # Only use flipped if it now correctly opens downwards (a_flipped > 0)
                    if a_flipped > 0:
                        a, b, c = a_flipped, b_flipped, c_flipped
                        isFlipped = True
                    else: # If even after flipping, 'a' is not positive (still a hill or too flat), skip
                        continue

                # --- Relaxed 'abs(a)' check --- removed for now, 'isFlipped' logic should cover it
                # Ensure vertex is within reasonable bounds (standard deviations from mean)
                vertexAcrossBackNormalized = -b / (2.0 * a)
                if abs(vertexAcrossBackNormalized) > 3.0: # Reverted to original 3.0 or even higher if needed
                    continue

                vertexDepth = a * vertexAcrossBackNormalized**2 + b * vertexAcrossBackNormalized + c
                if isFlipped: # Apply flip back to depth if it was flipped for fitting
                    vertexDepth = -vertexDepth

                vertexAcrossBack = vertexAcrossBackNormalized * xStd + xMean
                valley3dCoords = pca.inverse_transform([[vertexAcrossBack, vertexDepth]])[0]
                spinalMidlinePoints.append([valley3dCoords[0], current_y, valley3dCoords[2]])
            except np.linalg.LinAlgError:
                continue
            except ValueError: # Catch potential errors from np.linalg.lstsq for empty arrays
                continue
            
        # Ensure there are enough midline points before processing
        # Relaxed minimum from 10 to 5 for further processing
        if len(spinalMidlinePoints) < 5: 
            print(f"  Warning: Too few spinal midline points ({len(spinalMidlinePoints)}) detected for {ply_path}. Skipping landmark detection.")
            # Build scene with just the mesh and slices if midline is insufficient
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
            continue # Skip to next file
            
        spinalMidlinePoints = np.array(spinalMidlinePoints)
        spinalMidlinePoints = spinalMidlinePoints[np.argsort(spinalMidlinePoints[:, 1])]
        
        # --- FIX: Apply trimming before smoothing to remove endpoints more robustly ---
        spinalMidlinePoints = trim_midline_endpoints(spinalMidlinePoints, trim_fraction=0.08)
        
        # --- FIX: Robust smoothing ---
        if len(spinalMidlinePoints) >= 5: # Ensure enough points for smoothing
            spinalMidlinePoints = smooth_midline_robust(spinalMidlinePoints, sigma=2.0)
        else:
            print(f"  Warning: Midline too short after trimming for {ply_path}. Skipping landmark detection.")
            # Build scene with just the mesh and slices if midline is insufficient
            backMesh.paint_uniform_color([0.6, 0.6, 0.6])
            geoms = [backMesh] + sliceVizGeometries
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
            continue # Skip to next file


        curvatures = calculate_gaussian_curvature(backMesh, neighborhood_size=25) # Slightly larger neighborhood
        c7_point = detect_c7_landmark(backMesh, curvatures, spinalMidlinePoints, dimensions)
        psis_left, psis_right = detect_psis_landmarks(backMesh, curvatures, spinalMidlinePoints, dimensions)
        
        refined_midline = refine_midline_between_landmarks(spinalMidlinePoints, c7_point, [psis_left, psis_right])

        # Build scene
        backMesh.paint_uniform_color([0.6, 0.6, 0.6])
        
        geoms = [backMesh] + sliceVizGeometries
        
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