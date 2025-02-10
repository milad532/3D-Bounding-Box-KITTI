import cv2
import numpy as np

# --- Utility functions ---

def read_calib(calib_file):
    """
    Reads a KITTI calibration file.
    Expected lines for projection matrices (e.g. "P2:") contain 12 numbers.
    If only 9 numbers are provided (a 3x3 intrinsic matrix), we append a zero column.
    Returns a dictionary with keys like 'P2' and 'P3' containing 3x4 numpy arrays.
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            key, value = line.split(":", 1)
            numbers = [float(x) for x in value.split()]
            if len(numbers) == 12:
                calib[key] = np.array(numbers).reshape(3, 4)
            elif len(numbers) == 9:
                intrinsic = np.array(numbers).reshape(3, 3)
                calib[key] = np.hstack([intrinsic, np.zeros((3, 1))])
            else:
                raise ValueError(f"Unexpected number of numbers for {key}: {len(numbers)}")
    return calib

def parse_kitti_label(label_file):
    """
    Parses a KITTI label file.
    Each line is expected to have at least 15 fields:
      type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom,
      h, w, l, x, y, z, rotation_y
    Returns a list of dictionaries (one per object).
    """
    objects = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],   # [left, top, right, bottom]
                "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                "location": [float(parts[11]), float(parts[12]), float(parts[13])],
                "rotation_y": float(parts[14])
            }
            objects.append(obj)
    return objects

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    return image

def detect_sift_keypoints(image, roi=None):
    """
    Detects SIFT keypoints (and descriptors) in the image.
    If an ROI (x, y, w, h) is provided, detection is limited to that area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if roi is not None:
        x, y, w, h = roi
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
    else:
        mask = None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors

def match_keypoints(des1, des2):
    """
    Matches SIFT descriptors using BFMatcher with Lowe’s ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for match in raw_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    return good_matches

def triangulate_matches(kp_left, kp_right, matches, P_left, P_right):
    """
    Triangulates 3D points from matching keypoints between left and right images.
    Returns an array of 3D points (in Euclidean coordinates).
    """
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches]).T  # shape (2, N)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches]).T  # shape (2, N)
    pts_4d = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
    pts_3d = pts_4d[:3, :] / pts_4d[3, :]
    return pts_3d.T  # shape (N, 3)

def compute_3d_bbox(points_3d):
    """
    Computes an axis-aligned 3D bounding box from a set of 3D points.
    Returns the 8 corners of the box.
    """
    if points_3d.size == 0:
        return None
    x_min, y_min, z_min = np.min(points_3d, axis=0)
    x_max, y_max, z_max = np.max(points_3d, axis=0)
    corners = np.array([
        [x_max, y_max, z_max],
        [x_max, y_max, z_min],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_min, y_max, z_max],
        [x_min, y_max, z_min],
        [x_min, y_min, z_min],
        [x_min, y_min, z_max]
    ], dtype=np.float32)
    return corners

def project_box(P, corners_3d):
    """
    Projects 3D points into the image using projection matrix P.
    Returns the 2D projected points.
    """
    num_pts = corners_3d.shape[0]
    corners_hom = np.hstack([corners_3d, np.ones((num_pts, 1))])
    proj_pts_hom = np.dot(P, corners_hom.T).T
    proj_pts = proj_pts_hom[:, :2] / proj_pts_hom[:, 2:3]
    return proj_pts

def draw_box_3d(image, proj_pts, color=(0, 255, 0), thickness=2):
    """
    Draws a 3D bounding box on the image from the 2D projected points.
    Assumes ordering: first 4 points for bottom face, next 4 for top face.
    """
    proj_pts = proj_pts.astype(np.int32)
    # Draw bottom face
    for i in range(4):
        pt1 = tuple(proj_pts[i])
        pt2 = tuple(proj_pts[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    # Draw top face
    for i in range(4, 8):
        pt1 = tuple(proj_pts[i])
        pt2 = tuple(proj_pts[4 + (i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    # Draw vertical lines
    for i in range(4):
        pt_bottom = tuple(proj_pts[i])
        pt_top = tuple(proj_pts[i + 4])
        cv2.line(image, pt_bottom, pt_top, color, thickness)
    return image

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Each bbox is [x_min, y_min, x_max, y_max].
    """
    x_left   = max(bbox1[0], bbox2[0])
    y_top    = max(bbox1[1], bbox2[1])
    x_right  = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area
    return iou

def get_2d_bbox_from_points(points):
    """
    Given an array of 2D points (N, 2), computes the axis-aligned bounding box.
    Returns [x_min, y_min, x_max, y_max].
    """
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return [x_min, y_min, x_max, y_max]

# --- Main pipeline with stereo and IoU integration ---
def main():
    sample_number = "000019"
    calib_file = f"/home/milad/ACV/kitti/training/calib/{sample_number}.txt"          # KITTI calibration file
    label_file = f"/home/milad/ACV/kitti/training/label_2/{sample_number}.txt"         # KITTI label file for left image
    left_image_path = f"/home/milad/ACV/kitti/training/image_2/{sample_number}.png"    # Left camera image
    right_image_path = f"/home/milad/ACV/kitti/training/image_3/{sample_number}.png"   # Right camera image

    # Load images, calibration and labels
    left_img = load_image(left_image_path)
    right_img = load_image(right_image_path)
    calib = read_calib(calib_file)
    P2 = calib.get("P2")  # Left projection matrix
    P3 = calib.get("P3")  # Right projection matrix
    if P2 is None or P3 is None:
        print("Calibration file missing P2 or P3")
        return

    objects = parse_kitti_label(label_file)
    # Process only objects labeled as 'Car'
    car_objs = [obj for obj in objects if obj["type"] == "Car"]
    if not car_objs:
        print("No car objects in label file.")
        return
    sum_iou=0
    for car in car_objs:
        # Ground truth 2D box from label file: [left, top, right, bottom]
        gt_bbox = [int(v) for v in car["bbox"]]
        x, y, x2, y2 = gt_bbox
        roi = (x, y, x2 - x, y2 - y)

        # Detect SIFT keypoints restricted to the ROI in both left and right images.
        kp_left, des_left = detect_sift_keypoints(left_img, roi=roi)
        kp_right, des_right = detect_sift_keypoints(right_img, roi=roi)
        print(f"Car ROI: Left view {len(kp_left)} keypoints, Right view {len(kp_right)} keypoints")

        if des_left is None or des_right is None:
            print("No descriptors in one of the views.")
            continue

        # Match keypoints between left and right views
        matches = match_keypoints(des_left, des_right)
        print(f"Found {len(matches)} good stereo matches for this car.")

        if len(matches) < 3:
            print("Not enough matches for triangulation.")
            continue

        # Triangulate matching keypoints to obtain 3D points
        pts_3d = triangulate_matches(kp_left, kp_right, matches, P2, P3)
        pts_3d = pts_3d[pts_3d[:, 2] > 0]  # Keep points with positive depth
        print(f"Triangulated {pts_3d.shape[0]} 3D points for this car.")

        if pts_3d.shape[0] == 0:
            print("No valid 3D points after filtering.")
            continue

        # Compute a 3D bounding box (axis-aligned) from the triangulated 3D points.
        bbox_3d_corners = compute_3d_bbox(pts_3d)
        if bbox_3d_corners is None:
            print("Failed to compute 3D bounding box.")
            continue

        # Project the 3D bounding box onto the left image using P2.
        proj_box = project_box(P2, bbox_3d_corners)

        # For IoU calculation, compute the predicted 2D bounding box by taking the
        # min and max of the projected points.
        pred_bbox = get_2d_bbox_from_points(proj_box)
        iou = calculate_iou(pred_bbox, gt_bbox)
        sum_iou = sum_iou + iou
        print(f"Predicted bbox: {pred_bbox}, Ground Truth bbox: {gt_bbox}, IoU: {iou:.2f}")

        # Draw the predicted 3D bounding box (wireframe) and the ground truth 2D box for comparison.
        left_img = draw_box_3d(left_img, proj_box, color=(0, 255, 0), thickness=2)
        cv2.rectangle(left_img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 0, 0), 2)
        
        # Optionally, display IoU on the image
        # cv2.putText(left_img, f"IoU: {iou:.2f}", (gt_bbox[0], gt_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display and save the final result
    # cv2.imshow("Improved SIFT Detection with IoU", left_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cars_number=len(car_objs)
    if len(car_objs)==0:
        cars_number=1    
    print(f"IOU = {sum_iou/cars_number}")
    cv2.imwrite(f"{sample_number}_improved_sift_3d_box_iou.png", left_img)

if __name__ == "__main__":
    main()