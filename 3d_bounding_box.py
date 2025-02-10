import cv2
import numpy as np

def parse_kitti_label(label_file):
    """
    Parses a KITTI label file.
    Each line has the following fields:
      type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom,
      h, w, l, x, y, z, rotation_y
    Returns a list of dictionaries (one per object).
    """
    objects = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue  # skip incomplete lines
            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],  # [left, top, right, bottom]
                # KITTI dimensions: height, width, length
                "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],
                # KITTI location: x, y, z (in camera coordinates)
                "location": [float(parts[11]), float(parts[12]), float(parts[13])],
                "rotation_y": float(parts[14])
            }
            objects.append(obj)
    return objects

def read_calib(calib_file):
    """
    Reads a KITTI calibration file.
    Expected lines are for example:
      P2: fx 0 cx Tx; 0 fy cy Ty; 0 0 1 0
    If only 9 numbers are provided, assume it's a 3x3 intrinsic matrix and append a zero column.
    Returns a dictionary with keys such as 'P2' and 'P3' containing 3x4 numpy arrays.
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
                # If only 9 numbers are provided, assume it is a 3x3 intrinsic matrix.
                # Append a zero column to convert it into a 3x4 matrix.
                intrinsic = np.array(numbers).reshape(3, 3)
                calib[key] = np.hstack([intrinsic, np.zeros((3, 1))])
            else:
                raise ValueError(f"Unexpected number of calibration numbers for {key}: {len(numbers)}")
    return calib

def compute_box_3d(dim, location, rotation_y):
    """
    Computes the 3D bounding box corners of an object in camera coordinates.
    KITTI provides dimensions as (h, w, l).
    The model assumes the bottom center is at the object location.
    
    The 3D bounding box is defined in the object coordinate system.
    Here we define the eight corners in the object's coordinate system:
      (l/2, 0,  w/2), (l/2, 0, -w/2), (-l/2, 0, -w/2), (-l/2, 0, w/2)  --> bottom face
      and the same with y = -h for the top face.
    Then we rotate these corners around the y-axis by rotation_y and translate by location.
    """
    h, w, l = dim  # height, width, length
    # 8 corners of the box in the object coordinate system
    # Note: In KITTI, y is downward so the bottom face is at y=0.
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ 0,    0,    0,    0,   -h,   -h,   -h,   -h   ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]
    corners = np.array([x_corners, y_corners, z_corners])  # shape (3,8)

    # Rotation matrix around y-axis
    R = np.array([
        [ np.cos(rotation_y), 0, np.sin(rotation_y)],
        [ 0,                 1, 0                ],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    # Rotate and translate corners
    corners_3d = np.dot(R, corners)  # shape (3,8)
    corners_3d = corners_3d + np.array(location).reshape(3, 1)
    return corners_3d.T  # shape (8, 3)

def draw_box_3d(image, proj_points, color=(0, 255, 0), thickness=2):
    """
    Draws a 3D bounding box on the image given projected 2D points.
    proj_points: an array of shape (8,2) with the projected 2D points.
    The order of points is assumed to be:
      0-3: bottom face (clockwise), 4-7: top face.
    """
    proj_points = proj_points.astype(np.int32)

    # Draw bottom face
    for i in range(4):
        pt1 = tuple(proj_points[i])
        pt2 = tuple(proj_points[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    # Draw top face
    for i in range(4, 8):
        pt1 = tuple(proj_points[i])
        pt2 = tuple(proj_points[4 + (i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    # Draw vertical lines
    for i in range(4):
        pt_bottom = tuple(proj_points[i])
        pt_top = tuple(proj_points[i + 4])
        cv2.line(image, pt_bottom, pt_top, color, thickness)
    return image

def project_box(P, corners_3d):
    """
    Projects 3D box corners into the image using projection matrix P (3x4).
    Returns the 2D projected points as an (8, 2) array.
    """
    # Convert corners_3d to homogeneous coordinates
    num_points = corners_3d.shape[0]
    ones = np.ones((num_points, 1))
    corners_hom = np.hstack([corners_3d, ones])  # (num_points, 4)
    
    # Project points: result shape (num_points, 3)
    proj_points_hom = np.dot(P, corners_hom.T).T
    proj_points = proj_points_hom[:, :2] / proj_points_hom[:, 2:3]
    return proj_points

def main():
    
    calib_file = "/home/milad/ACV/kitti/training/calib/000018.txt"          # KITTI calibration file for the scene
    label_file = "/home/milad/ACV/kitti/training/label_2/000018.txt"         # KITTI label file for this image (usually for left image)
    left_image_path = "/home/milad/ACV/kitti/training/image_2/000018.png"    # Left camera image
    right_image_path = "/home/milad/ACV/kitti/training/image_3/000018.png"   # Right camera image

    # Load images
    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)
    if left_img is None or right_img is None:
        print("Could not load one or both images.")
        return

    # Read calibration and label files
    calib = read_calib(calib_file)
    # For KITTI, P2 is the left camera projection matrix and P3 is the right camera projection matrix.
    P2 = calib.get("P2")
    P3 = calib.get("P3")
    if P2 is None or P3 is None:
        print("Calibration file does not contain P2/P3")
        return

    objects = parse_kitti_label(label_file)
    # For this example, we only consider objects of type 'Car'
    car_objects = [obj for obj in objects if obj["type"] == "Car"]

    # For each car, compute its 3D bounding box and project it on both images
    for obj in car_objects:
        # KITTI dimensions: [h, w, l]
        dims = obj["dimensions"]
        loc = obj["location"]
        ry = obj["rotation_y"]
        # Compute the 3D bounding box corners (8 corners)
        box3d_corners = compute_box_3d(dims, loc, ry)  # shape (8,3)
        # Project to left image using P2 and right image using P3
        proj_left = project_box(P2, box3d_corners)
        proj_right = project_box(P3, box3d_corners)
        
        # Draw the 3D bounding box on both images
        left_img = draw_box_3d(left_img, proj_left, color=(0, 255, 0), thickness=2)
        right_img = draw_box_3d(right_img, proj_right, color=(0, 255, 0), thickness=2)
    
    # Optionally, draw also the 2D bounding box (from the label) on the left image
    for obj in car_objects:
        bbox = obj["bbox"]
        # bbox order: [left, top, right, bottom]
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(left_img, pt1, pt2, (255, 0, 0), 2)
    
    # # Show the images
    # cv2.imshow("Left Image with 3D Boxes", left_img)
    # cv2.imshow("Right Image with 3D Boxes", right_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save results
    cv2.imwrite("left_result.png", left_img)
    cv2.imwrite("right_result.png", right_img)

if __name__ == "__main__":
    main()