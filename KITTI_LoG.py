import os
import cv2
import numpy as np
import random
import argparse
import concurrent.futures
from tqdm import tqdm

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
    The box is defined with its bottom center at (0,0,0) and extends:
      - along X (length): ±l/2,
      - along Y (height): 0 to -h,
      - along Z (width): ±w/2.
    Returns an array of shape (8, 3).
    """
    h, w, l = dimensions
    # 8 corners: first 4 are the bottom face, next 4 the top face.
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [    0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    object_points = np.vstack((x_corners, y_corners, z_corners)).T
    return object_points.astype(np.float32)

def detect_log_corners(image, bbox, maxCorners=20, threshold=0.03, sigma=1.5, kernel_size=3):
    """
    Detect keypoints using a Laplacian-of-Gaussian (LoG) approach within a given ROI.
    
    The function:
      1. Crops the ROI defined by bbox.
      2. Converts the ROI to grayscale and normalizes to [0, 1].
      3. Applies Gaussian blur (with standard deviation sigma).
      4. Converts the blurred image to float64.
      5. Computes the Laplacian.
      6. Normalizes the absolute Laplacian response to [0,1].
      7. Finds local maxima via dilation and thresholds them.
      8. Returns up to 'maxCorners' keypoints in full image coordinates.
      
    Parameters:
      image      : Full color image.
      bbox       : [xmin, ymin, xmax, ymax] from KITTI labels.
      maxCorners : Maximum number of keypoints to return.
      threshold  : Minimum normalized LoG response.
      sigma      : Gaussian blur sigma.
      kernel_size: Kernel size for Laplacian operator.
      
    Returns:
      corners_full : Array of keypoints (shape (N,2)) in full image coordinates or None.
    """
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    roi = image[ymin:ymax, xmin:xmax]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = np.float32(gray_roi) / 255.0
    blurred = cv2.GaussianBlur(gray_roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred = np.float64(blurred)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    lap_abs = np.abs(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find local maxima by dilation.
    dilated = cv2.dilate(lap_norm, None)
    local_max_mask = (lap_norm == dilated)
    thresh_mask = (lap_norm > threshold)
    corner_mask = np.logical_and(local_max_mask, thresh_mask)
    
    coords = np.argwhere(corner_mask)
    if len(coords) == 0:
        return None
    responses = lap_norm[corner_mask]
    sorted_indices = np.argsort(-responses)
    coords = coords[sorted_indices][:maxCorners]
    corners_roi = np.fliplr(coords).astype(np.float32)
    corners_full = corners_roi + np.array([xmin, ymin], dtype=np.float32)
    return corners_full

def match_detected_to_projected(projected_points, detected_points, thresh=20):
    """
    For each projected (ideal) 2D point, find the closest detected LoG keypoint
    that is within a given threshold. Once used, a detected point is removed.
    
    Parameters:
      projected_points: Array (N,2) of ideal/projection points.
      detected_points : Array (M,2) of detected keypoints.
      thresh          : Maximum Euclidean distance for a valid match.
    
    Returns:
      matched_points: Array (N,2) of the matched keypoints or None if a match fails.
    """
    matched = []
    detected_list = detected_points.tolist()
    for pt in projected_points:
        if not detected_list:
            return None
        dists = np.linalg.norm(np.array(detected_list) - pt, axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < thresh:
            matched.append(detected_list[min_idx])
            detected_list.pop(min_idx)
        else:
            return None
    return np.array(matched, dtype=np.float32)

def compute_iou(boxA, boxB):
    """
    Compute Intersection-over-Union (IOU) between two axis-aligned bounding boxes.
    Each box is [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def process_sample(sample_number, calib_dir, label_dir, image_dir, output_dir=None):
    """
    Process one KITTI sample:
      - Load calibration, label, and image.
      - For each 'Car' object, run pose estimation using LoG keypoints.
      - Compute the predicted 2D bounding box (axis–aligned from reprojected 3D points)
        and compare with the ground-truth bounding box.
      - Draw boxes and keypoints (optional) and return a list of IOU scores.
    """
    calib_path = os.path.join(calib_dir, sample_number + ".txt")
    label_path = os.path.join(label_dir, sample_number + ".txt")
    image_path = os.path.join(image_dir, sample_number + ".png")
    
    img = cv2.imread(image_path)
    if img is None:
        return []  # Skip if image cannot be loaded.
    
    calib = read_calib_file(calib_path)
    objects = read_label_file(label_path)
    
    if 'P2' not in calib:
        return []
    P2 = calib['P2'].reshape(3, 4)
    K = P2[:, :3]
    distCoeffs = np.zeros((4, 1))
    
    sample_ious = []
    
    for obj in objects:
        if obj['type'] != 'Car':
            continue
        
        object_points = compute_box_object_points(obj['dimensions'])
        
        # Ground-truth pose (rotation about Y-axis).
        rotation_y = obj['rotation_y']
        R = np.array([[ np.cos(rotation_y), 0, np.sin(rotation_y)],
                      [                 0, 1,               0],
                      [-np.sin(rotation_y), 0, np.cos(rotation_y)]])
        rvec_gt, _ = cv2.Rodrigues(R)
        tvec_gt = np.array(obj['location'], dtype=np.float32).reshape(3, 1)
        
        # Project 3D object points using the ground-truth pose.
        projected_points, _ = cv2.projectPoints(object_points, rvec_gt, tvec_gt, K, distCoeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        gt_bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
        
        # Detect LoG keypoints in the ground-truth bbox region.
        detected_corners = detect_log_corners(img, gt_bbox, maxCorners=300, threshold=0.03, sigma=1.5, kernel_size=3)
        if detected_corners is None:
            continue
        
        # Match detected keypoints to the projected ideal points.
        matched_corners = match_detected_to_projected(projected_points, detected_corners, thresh=20)
        if matched_corners is None or len(matched_corners) < 4:
            continue
        
        # If more than 8 matches, use the first 8.
        if len(matched_corners) > 8:
            matched_corners = matched_corners[:8]
            object_points_corr = object_points[:8]
        else:
            object_points_corr = object_points
        
        # Estimate pose using RANSAC-based PnP.
        success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
            object_points_corr, matched_corners, K, distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, iterationsCount=100)
        if not success:
            continue
        
        # Reproject the full 3D bounding box using the estimated pose.
        estimated_image_points, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, K, distCoeffs)
        estimated_image_points = estimated_image_points.reshape(-1, 2)
        
        # Compute an axis-aligned bounding box from the estimated points.
        x_min_pred = np.min(estimated_image_points[:, 0])
        y_min_pred = np.min(estimated_image_points[:, 1])
        x_max_pred = np.max(estimated_image_points[:, 0])
        y_max_pred = np.max(estimated_image_points[:, 1])
        pred_bbox = [x_min_pred, y_min_pred, x_max_pred, y_max_pred]
        
        iou = compute_iou(gt_bbox, pred_bbox)
        sample_ious.append(iou)
        
        # Draw ground-truth bbox (blue) and predicted bbox (green).
        pt1 = (int(gt_bbox[0]), int(gt_bbox[1]))
        pt2 = (int(gt_bbox[2]), int(gt_bbox[3]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        pt1_pred = (int(x_min_pred), int(y_min_pred))
        pt2_pred = (int(x_max_pred), int(y_max_pred))
        cv2.rectangle(img, pt1_pred, pt2_pred, (0, 255, 0), 2)
        
        # Optionally, draw the matched keypoints (red) and reprojected points (yellow).
        for pt in matched_corners:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        for pt in estimated_image_points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
    
    # Optionally save the output image.
    if output_dir is not None:
        out_path = os.path.join(output_dir, sample_number + "_output.png")
        cv2.imwrite(out_path, img)
    
    return sample_ious

def main(args):
    calib_dir = args.calib_dir
    label_dir = args.label_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    num_workers = args.num_workers

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all sample numbers from the image directory.
    sample_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    sample_numbers = sorted([os.path.splitext(f)[0] for f in sample_files])
    
    all_ious = []
    
    # Process samples in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, sample_number, calib_dir, label_dir, image_dir, output_dir): sample_number
                   for sample_number in sample_numbers}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            sample_number = futures[future]
            try:
                sample_ious = future.result()
                all_ious.extend(sample_ious)
            except Exception as e:
                print(f"Error processing sample {sample_number}: {e}")
    
    if all_ious:
        mean_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        print(f"Processed {len(sample_numbers)} samples with {len(all_ious)} valid objects.")
        print(f"Mean IOU: {mean_iou:.3f}, Std IOU: {std_iou:.3f}")
    else:
        print("No valid IOU computed from the dataset.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KITTI Pose Estimation & IOU Computation using LoG keypoints")
    parser.add_argument('--calib_dir', type=str, default="/home/milad/ACV/kitti/training/calib",
                        help="Directory containing KITTI calibration files")
    parser.add_argument('--label_dir', type=str, default="/home/milad/ACV/kitti/training/label_2",
                        help="Directory containing KITTI label files")
    parser.add_argument('--image_dir', type=str, default="/home/milad/ACV/kitti/training/image_2",
                        help="Directory containing KITTI images")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Optional directory to save output images")
    parser.add_argument('--num_workers', type=int, default=32,
                        help="Number of parallel workers to use")
    args = parser.parse_args()
    main(args)