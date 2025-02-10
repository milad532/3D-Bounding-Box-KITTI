# 3D Bounding Box Car Detection on the KITTI Dataset

## Overview
This project implements 3D bounding box detection for cars using the KITTI dataset. The approach leverages different feature detection techniques to estimate 3D bounding boxes. The methods explored include:
- **SIFT (Scale-Invariant Feature Transform)**
- **RANSAC (Random Sample Consensus)**
- **LoG (Laplacian of Gaussian)**

The implemented algorithms process images from the KITTI dataset and generate 3D bounding boxes around detected vehicles.

## Dataset
The KITTI dataset is a well-known benchmark for computer vision tasks in autonomous driving, providing images with rich annotations for object detection, tracking, and segmentation.

### Dataset Structure
The dataset follows this directory structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```

## Implementation
The codebase includes implementations for multiple feature detection techniques:
- `3d_bounding_box.py` - Main script for 3D bounding box estimation.
- `3d_sift_box.py` - 3D bounding box estimation using SIFT.
- `KITTI_LoG.py` - 3D bounding box estimation using LoG.
- `KITTI_RANSAC.py` - 3D bounding box estimation using RANSAC.
- `KiTTi_SIFT_3D_box.py` - Additional script for SIFT-based 3D bounding box detection.
- `LoG_keypoint.py` - Keypoint detection using LoG.
- `RANSAC_keypoint.py` - Keypoint detection using RANSAC.
- `SIFT_keypoint_detector.py` - Keypoint detection using SIFT.

## Results
The performance of each method is evaluated using the Intersection over Union (IoU) metric:
- **SIFT**: Mean IoU = **0.25**, Std IoU = **0.05**
- **RANSAC**: Mean IoU = **0.80**, Std IoU = **0.19**
- **LoG**: Mean IoU = **0.87**, Std IoU = **0.07**

### Sample Output
Below is an example of detected 3D bounding boxes visualized on an image from the KITTI dataset:

![3D Bounding Box Results](3D_bbox_results.png)
## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/3D-Bounding-Box-KITTI.git
   cd 3D-Bounding-Box-KITTI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the desired script:
   ```bash
   python KITTI_RANSAC.py
   ```
   or
   ```bash
   python 3d_sift_box.py
   ```

## Dependencies
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

## License
This project is licensed under the MIT License.

## Author
[Milad Hosseini]

