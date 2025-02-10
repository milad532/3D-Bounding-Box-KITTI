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

if __name__ == '__main__':
    # --- Update these paths to point to your KITTI data files ---
    sample_number = "000019"
    calib_path = f"/home/milad/ACV/kitti/training/calib/{sample_number}.txt"          # KITTI calibration file
    label_path = f"/home/milad/ACV/kitti/training/label_2/{sample_number}.txt"         # KITTI label file for left image
    left_img_path = f"/home/milad/ACV/kitti/training/image_2/{sample_number}.png"    # Left camera image
    
    # Load the left view image
    img = cv2.imread(left_img_path)
    if img is None:
        raise ValueError("Could not load image at " + left_img_path)
    
    # Read calibration and label files
    calib = read_calib_file(calib_path)
    objects = read_label_file(label_path)
    
    # Extract intrinsic matrix from left camera projection matrix P2
    if 'P2' not in calib:
        raise ValueError("P2 not found in calibration file.")
    P2 = calib['P2'].reshape(3, 4)
    K = P2[:, :3]  # intrinsic camera matrix
    distCoeffs = np.zeros((4, 1))  # assuming zero distortion
    
    # Process each object (we handle only cars in this example)
    for obj in objects:
        if obj['type'] != 'Car':
            continue
        
        # Compute the 3D box corners in the object coordinate system.
        object_points = compute_box_object_points(obj['dimensions'])
        
        # Ground truth pose (from label): rotation around Y-axis and translation.
        rotation_y = obj['rotation_y']
        # Create a rotation matrix around Y-axis.
        R = np.array([[ np.cos(rotation_y), 0, np.sin(rotation_y)],
                      [               0,     1,             0],
                      [-np.sin(rotation_y), 0, np.cos(rotation_y)]])
        rvec_gt, _ = cv2.Rodrigues(R)  # ground truth rotation vector
        tvec_gt = np.array(obj['location'], dtype=np.float32).reshape(3, 1)
        
        # Project the object points to the image plane using the ground truth pose.
        image_points, _ = cv2.projectPoints(object_points, rvec_gt, tvec_gt, K, distCoeffs)
        image_points = image_points.reshape(-1, 2)
        
        # Simulate noisy keypoint estimation (some points may be outliers)
        noisy_image_points = add_noise_to_keypoints(image_points, noise_std=3.0,
                                                    outlier_ratio=0.3, image_shape=img.shape)
        
        # Use RANSAC-based PnP to robustly estimate the pose from the noisy keypoints.
        success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(object_points, noisy_image_points,
                                                                   K, distCoeffs,
                                                                   flags=cv2.SOLVEPNP_ITERATIVE,
                                                                   reprojectionError=8.0,
                                                                   iterationsCount=100)
        if not success:
            print("PnPRansac failed for an object.")
            continue
        
        # Draw the provided 2D bounding box (from labels) in blue.
        bbox = obj['bbox']
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        
        # Reproject the 3D object points using the estimated pose.
        estimated_image_points, _ = cv2.projectPoints(object_points, rvec_est, tvec_est, K, distCoeffs)
        estimated_image_points = estimated_image_points.reshape(-1, 2).astype(np.int32)
        
        # Define the connections of the 3D bounding box.
        connections = [(0,1), (1,2), (2,3), (3,0),
                       (4,5), (5,6), (6,7), (7,4),
                       (0,4), (1,5), (2,6), (3,7)]
        for i, j in connections:
            pt_i = tuple(estimated_image_points[i])
            pt_j = tuple(estimated_image_points[j])
            cv2.line(img, pt_i, pt_j, (0, 255, 0), 2)
        
        # Optionally, draw the noisy keypoints as small red circles.
        for pt in noisy_image_points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        
        # Output some information about the estimated pose.
        print("Estimated rvec:", rvec_est.ravel())
        print("Estimated tvec:", tvec_est.ravel())
        print("Number of inliers:", len(inliers))
    
    # Save the output image (since remote servers may not support visualization)
    output_path = f'{sample_number}_output_ransac_keypoints.png'
    cv2.imwrite(output_path, img)
    print("Output saved to", output_path)