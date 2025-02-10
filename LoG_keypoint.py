import numpy as np
import cv2
import random

def read_calib_file(filepath):
    """
    Read KITTI calibration file and return a dictionary with matrices.
    """
    calib = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            key, value = line.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()])
    return calib

def read_label_file(filepath):
    """
    Read KITTI label file.
    Returns a list of objects with keys:
      type, bbox, dimensions (h, w, l), location, and rotation_y.
    """
    objects = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj = {
                'type': parts[0],
                'truncation': float(parts[1]),
                'occlusion': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                # Dimensions: height, width, length
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                # Location (x, y, z) in camera coordinates
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                'rotation_y': float(parts[14])
            }
            objects.append(obj)
    return objects

def compute_box_object_points(dimensions):
    """
    Compute the 3D bounding box corners in the object's coordinate system.
    Here the box is defined with its bottom center at (0,0,0) and extends:
      - along X (length): ±l/2,
      - along Y (height): 0 to -h,
      - along Z (width): ±w/2.
    Returns an array of shape (8, 3).
    """
    h, w, l = dimensions
    # 8 corners: first 4 are the bottom face, next 4 the top face
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [    0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    object_points = np.vstack((x_corners, y_corners, z_corners)).T  # shape (8, 3)
    return object_points.astype(np.float32)

def detect_log_corners(image, bbox, maxCorners=20, threshold=0.03, sigma=1.5, kernel_size=3):
    """
    Detect keypoints using a Laplacian-of-Gaussian (LoG) approach within a given ROI.
    
    This function:
      1. Crops the region of interest (ROI) defined by bbox.
      2. Converts the ROI to grayscale and scales it to [0,1].
      3. Applies Gaussian blur (with standard deviation sigma).
      4. Converts the blurred image to float64.
      5. Computes the Laplacian of the blurred image.
      6. Normalizes the absolute value of the Laplacian response.
      7. Finds local maxima via dilation and thresholds them.
      8. Returns the top 'maxCorners' points in full image coordinates.
      
    Parameters:
      image      : Full color image.
      bbox       : [xmin, ymin, xmax, ymax] (from KITTI label).
      maxCorners : Maximum number of corners (keypoints) to return.
      threshold  : Minimum normalized LoG response to consider.
      sigma      : Standard deviation for Gaussian blur.
      kernel_size: Kernel size for the Laplacian operator.
      
    Returns:
      corners_full : Array of detected keypoints (in full image coordinates) of shape (N,2)
                     or None if no keypoints are detected.
    """
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    # Crop ROI from the image.
    roi = image[ymin:ymax, xmin:xmax]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Convert to float32 and normalize to [0, 1]
    gray_roi = np.float32(gray_roi) / 255.0
    # Apply Gaussian Blur.
    blurred = cv2.GaussianBlur(gray_roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Convert blurred image to float64 to match the desired output depth.
    blurred = np.float64(blurred)
    # Compute the Laplacian.
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    # Use absolute response (corners may produce both positive and negative responses).
    lap_abs = np.abs(lap)
    # Normalize the Laplacian response to [0,1].
    lap_norm = cv2.normalize(lap_abs, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find local maxima by comparing each pixel to its neighborhood (via dilation).
    dilated = cv2.dilate(lap_norm, None)
    local_max_mask = (lap_norm == dilated)
    # Threshold the normalized response.
    thresh_mask = (lap_norm > threshold)
    # Combine the masks.
    corner_mask = np.logical_and(local_max_mask, thresh_mask)
    
    # Get coordinates of detected corners. Note: np.argwhere returns (row, col) pairs.
    coords = np.argwhere(corner_mask)
    if len(coords) == 0:
        return None
    # Get the corresponding responses.
    responses = lap_norm[corner_mask]
    # Sort corners by response (descending).
    sorted_indices = np.argsort(-responses)
    coords = coords[sorted_indices]
    # Limit to the strongest maxCorners.
    coords = coords[:maxCorners]
    # Convert from (row, col) to (x, y) coordinates.
    corners_roi = np.fliplr(coords).astype(np.float32)
    # Shift coordinates to the full image space (add ROI offset).
    corners_full = corners_roi + np.array([xmin, ymin], dtype=np.float32)
    return corners_full


def match_detected_to_projected(projected_points, detected_points, thresh=20):
    """
    For each projected (ideal) 2D point, find the closest detected point
    (from the LoG method) that is within a given threshold. Once a detected point is used,
    it is removed from further matching.
    
    Parameters:
      projected_points: Array of shape (N,2) of ideal/projection points.
      detected_points : Array of shape (M,2) of points from LoG detection.
      thresh          : Maximum distance to consider a match valid.
    
    Returns:
      matched_points: Array of shape (N,2) with the matched detected points.
                      If a match is not found for any projected point, returns None.
    """
    matched = []
    detected_list = detected_points.tolist()  # list of [x, y]
    
    for pt in projected_points:
        if len(detected_list) == 0:
            return None  # no points left to match
        # Compute Euclidean distances from pt to all available detected points.
        dists = np.linalg.norm(np.array(detected_list) - pt, axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < thresh:
            matched.append(detected_list[min_idx])
            # Remove the matched point to avoid duplicate assignment.
            detected_list.pop(min_idx)
        else:
            # If no detected point is close enough, consider the matching failed.
            return None
    return np.array(matched, dtype=np.float32)

if __name__ == '__main__':
    # --- Update these paths to point to your KITTI data files ---
    sample_number = "000018"
    calib_path = f"/home/milad/ACV/kitti/training/calib/{sample_number}.txt"          # KITTI calibration file
    label_path = f"/home/milad/ACV/kitti/training/label_2/{sample_number}.txt"         # KITTI label file for left image
    left_img_path = f"/home/milad/ACV/kitti/training/image_2/{sample_number}.png"      # Left camera image

    # Load the left view image.
    img = cv2.imread(left_img_path)
    if img is None:
        raise ValueError("Could not load image at " + left_img_path)
    
    # Read calibration and label files.
    calib = read_calib_file(calib_path)
    objects = read_label_file(label_path)
    
    # Extract intrinsic matrix from left camera projection matrix P2.
    if 'P2' not in calib:
        raise ValueError("P2 not found in calibration file.")
    P2 = calib['P2'].reshape(3, 4)
    K = P2[:, :3]  # intrinsic camera matrix
    distCoeffs = np.zeros((4, 1))  # assuming zero distortion
    
    # Process each object (only handling cars in this example).
    for obj in objects:
        if obj['type'] != 'Car':
            continue
        
        # Compute the 3D bounding box corners in the object coordinate system.
        object_points = compute_box_object_points(obj['dimensions'])
        
        # --- Obtain ground-truth pose (for simulation) ---
        rotation_y = obj['rotation_y']
        # Create a rotation matrix around Y-axis.
        R = np.array([[ np.cos(rotation_y), 0, np.sin(rotation_y)],
                      [               0,     1,             0],
                      [-np.sin(rotation_y), 0, np.cos(rotation_y)]])
        rvec_gt, _ = cv2.Rodrigues(R)  # ground-truth rotation vector
        tvec_gt = np.array(obj['location'], dtype=np.float32).reshape(3, 1)
        
        # Project the 3D object points to the image plane using the ground-truth pose.
        projected_points, _ = cv2.projectPoints(object_points, rvec_gt, tvec_gt, K, distCoeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # --- Laplacian-of-Gaussian (LoG) Corner Detection in the car bounding box ---
        bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
        detected_corners = detect_log_corners(img, bbox,
                                              maxCorners=300,
                                              threshold=0.03,
                                              sigma=1.5,
                                              kernel_size=3)
        if detected_corners is None:
            print("No LoG corners detected in bbox, skipping object.")
            continue

        # --- Match detected corners to the projected 3D box corners ---
        matched_corners = match_detected_to_projected(projected_points, detected_corners, thresh=20)
        if matched_corners is None or len(matched_corners) < 4:
            print("Not all projected corners could be matched reliably, skipping object.")
            continue

        # For this demo, if more than 8 points are matched, we take the top 8.
        if len(matched_corners) > 8:
            matched_corners = matched_corners[:8]
            projected_corr = projected_points[:8]
            object_points_corr = object_points[:8]
        else:
            projected_corr = projected_points
            object_points_corr = object_points

        # Use RANSAC-based PnP to robustly estimate the pose from the (LoG-detected) keypoints.
        success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(object_points_corr, matched_corners,
                                                                   K, distCoeffs,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   reprojectionError=8.0,
                                                                   iterationsCount=100)
        if not success:
            print("PnPRansac failed for an object.")
            continue

        # Draw the provided 2D bounding box (from labels) in blue.
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        
        # Reproject the 3D object points using the estimated pose.
        estimated_image_points, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, K, distCoeffs)
        estimated_image_points = estimated_image_points.reshape(-1, 2).astype(np.int32)
        
        # Draw the reprojected 3D bounding box (using the estimated pose) in green.
        connections = [(0,1), (1,2), (2,3), (3,0),
                       (4,5), (5,6), (6,7), (7,4),
                       (0,4), (1,5), (2,6), (3,7)]
        for i, j in connections:
            pt_i = tuple(estimated_image_points[i])
            pt_j = tuple(estimated_image_points[j])
            cv2.line(img, pt_i, pt_j, (0, 255, 0), 2)
        
        # Draw the LoG-detected keypoints (matched to projected points) as small red circles.
        for pt in matched_corners:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        
        # Optionally, draw the ideal (ground-truth) projected points in yellow for comparison.
        for pt in projected_points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
        
        # Output estimated pose information.
        print("Estimated rvec:", rvec_est.ravel())
        print("Estimated tvec:", tvec_est.ravel())
        if inliers is not None:
            print("Number of inliers:", len(inliers))
    
    # Save the output image (useful if running in an environment without display)
    output_path = f'{sample_number}_output_log_keypoints.png'
    cv2.imwrite(output_path, img)
    print("Output saved to", output_path)