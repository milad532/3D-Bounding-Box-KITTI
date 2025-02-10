import numpy as np
import cv2
import random
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def add_noise_to_keypoints(keypoints, noise_std=3.0, outlier_ratio=0.3, image_shape=None):
    """
    Add Gaussian noise to keypoints and randomly perturb some points to simulate outliers.
    """
    noisy_keypoints = keypoints.copy()
    for i in range(len(noisy_keypoints)):
        if random.random() < outlier_ratio:
            # Replace with a random point within the image dimensions if provided
            if image_shape is not None:
                h, w = image_shape[:2]
                noisy_keypoints[i, 0] = random.uniform(0, w)
                noisy_keypoints[i, 1] = random.uniform(0, h)
            else:
                noisy_keypoints[i] += np.random.normal(0, noise_std * 5, 2)
        else:
            noisy_keypoints[i] += np.random.normal(0, noise_std, 2)
    return noisy_keypoints

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are [x_min, y_min, x_max, y_max].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_axis_aligned_bbox(points):
    """
    Given a set of 2D points (Nx2), compute the axis-aligned bounding box.
    Returns [x_min, y_min, x_max, y_max].
    """
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]

def process_sample(sample):
    """
    Process one KITTI sample: for each 'Car' in the sample,
    use RANSAC-based PnP to estimate pose from noisy keypoints,
    reproject the 3D bounding box, and compute the IoU between the
    predicted 2D box and ground truth.
    
    The sample is a dictionary with:
       'img_path', 'calib_path', 'label_path'
    
    Returns a list of IoU values (one per Car object) for this sample.
    """
    try:
        # Load image
        img = cv2.imread(sample['img_path'])
        if img is None:
            return []
        
        # Read calibration and extract intrinsic matrix (from P2)
        calib = read_calib_file(sample['calib_path'])
        if 'P2' not in calib:
            return []
        P2 = calib['P2'].reshape(3, 4)
        K = P2[:, :3]
        distCoeffs = np.zeros((4, 1))  # assume zero distortion
        
        # Read label file
        objects = read_label_file(sample['label_path'])
        iou_list = []
        
        for obj in objects:
            if obj['type'] != 'Car':
                continue
            
            # Compute the 3D object points in the object coordinate system.
            object_points = compute_box_object_points(obj['dimensions'])
            
            # Ground truth pose (from label): rotation around Y-axis and translation.
            rotation_y = obj['rotation_y']
            R = np.array([[ np.cos(rotation_y), 0, np.sin(rotation_y)],
                          [               0,     1,             0],
                          [-np.sin(rotation_y), 0, np.cos(rotation_y)]])
            rvec_gt, _ = cv2.Rodrigues(R)
            tvec_gt = np.array(obj['location'], dtype=np.float32).reshape(3, 1)
            
            # Project object points using the ground truth pose to obtain ideal keypoints.
            image_points, _ = cv2.projectPoints(object_points, rvec_gt, tvec_gt, K, distCoeffs)
            image_points = image_points.reshape(-1, 2)
            
            # Simulate noisy keypoint estimation (with outliers)
            noisy_image_points = add_noise_to_keypoints(image_points, noise_std=3.0,
                                                        outlier_ratio=0.3, image_shape=img.shape)
            
            # Use RANSAC-based PnP to robustly estimate the pose.
            success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
                object_points, noisy_image_points, K, distCoeffs,
                flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, iterationsCount=100
            )
            if not success:
                continue
            
            # Reproject the 3D object points using the estimated pose.
            estimated_image_points, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, K, distCoeffs)
            estimated_image_points = estimated_image_points.reshape(-1, 2)
            
            # Compute predicted 2D bounding box (axis-aligned) from the reprojected points.
            pred_bbox = get_axis_aligned_bbox(estimated_image_points)
            # Ground truth 2D bbox from the label.
            gt_bbox = obj['bbox']  # [x_min, y_min, x_max, y_max]
            iou = compute_iou(pred_bbox, gt_bbox)
            iou_list.append(iou)
            
        return iou_list
    except Exception as e:
        print("Error processing sample:", sample['img_path'], e)
        return []

if __name__ == '__main__':
    # --- Update these paths to point to your KITTI dataset directories ---
    image_dir = "/home/milad/ACV/kitti/training/image_2"   # left images directory
    label_dir = "/home/milad/ACV/kitti/training/label_2"     # label files directory
    calib_dir = "/home/milad/ACV/kitti/training/calib"         # calibration files directory

    # Build list of samples (each sample is a dict with paths for image, label, and calib)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    samples = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        calib_path = os.path.join(calib_dir, base_name + ".txt")
        if os.path.exists(label_path) and os.path.exists(calib_path):
            samples.append({
                "img_path": img_path,
                "label_path": label_path,
                "calib_path": calib_path
            })

    print("Found {} samples in the dataset.".format(len(samples)))
    
    # Use multiprocessing for efficient computation
    num_workers = 32  # adjust the number of workers as needed
    all_iou = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in samples]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            iou_list = future.result()
            all_iou.extend(iou_list)
    
    if all_iou:
        mean_iou = np.mean(all_iou)
        std_iou = np.std(all_iou)
        print("Overall IoU over dataset: {:.3f} ± {:.3f}".format(mean_iou, std_iou))
    else:
        print("No IoU values were computed. Check your dataset and processing functions.")