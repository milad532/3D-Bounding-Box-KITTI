import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def load_image(image_path):
    """Loads an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    return image

def detect_sift_keypoints(image):
    """
    Converts the image to grayscale and detects SIFT keypoints and descriptors.
    Returns keypoints and descriptors.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def cluster_keypoints(keypoints, eps=30, min_samples=5):
    """
    Clusters keypoints by their (x, y) coordinates using DBSCAN.
    The parameters eps (distance in pixels) and min_samples can be tuned.
    Returns a list of cluster labels corresponding to each keypoint.
    """
    if not keypoints:
        return np.array([])
    coords = np.array([kp.pt for kp in keypoints])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return clustering.labels_

def get_cluster_bounding_box(keypoints, labels, target_label):
    """
    For a given target cluster (by label), computes the axis-aligned bounding box.
    Returns (x_min, y_min, x_max, y_max) if the cluster has enough points.
    """
    coords = np.array([kp.pt for kp in keypoints])
    cluster_coords = coords[labels == target_label]
    if cluster_coords.size == 0:
        return None
    x_min = int(np.min(cluster_coords[:, 0]))
    y_min = int(np.min(cluster_coords[:, 1]))
    x_max = int(np.max(cluster_coords[:, 0]))
    y_max = int(np.max(cluster_coords[:, 1]))
    return (x_min, y_min, x_max, y_max)

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draws a rectangle on the image with given bounding box coordinates."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def main():
    # Replace this with the path to your KITTI image (e.g., a left camera image)
    image_path = "/home/milad/ACV/kitti/training/image_2/000018.png"
    image = load_image(image_path)

    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = detect_sift_keypoints(image)
    print(f"Detected {len(keypoints)} keypoints.")

    # Cluster the keypoints using DBSCAN.
    # (eps and min_samples may need to be adjusted based on image resolution and object size)
    labels = cluster_keypoints(keypoints, eps=30, min_samples=5)
    unique_labels = set(labels)
    print(f"Found clusters: {unique_labels}")

    # Here, we assume that the car of interest produces one of the larger clusters.
    # For illustration, we pick the cluster with the most keypoints (ignoring noise, label -1).
    best_label = None
    best_count = 10
    for label in unique_labels:
        if label == -1:
            continue
        count = np.sum(labels == label)
        if count > best_count:
            best_count = count
            best_label = label

    if best_label is None:
        print("No valid cluster found.")
        return

    bbox = get_cluster_bounding_box(keypoints, labels, best_label)
    if bbox is None:
        print("Failed to compute bounding box for cluster.")
        return

    # Draw the bounding box on a copy of the image.
    output_img = image.copy()
    output_img = draw_bounding_box(output_img, bbox, color=(0, 255, 0), thickness=2)

    # Optionally, draw all keypoints for visualization
    output_img = cv2.drawKeypoints(output_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("Classic Keypoint-Based Box", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("classic_keypoint_bbox.png", output_img)

if __name__ == "__main__":
    main()