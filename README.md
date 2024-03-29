# F-Matrix and Relative Pose Estimation

## Introduction
This repository contains the code and solutions for Task 3, which involves estimating the fundamental matrix between two images (`left.jpg` and `right.jpg`), drawing epipolar lines, and finding the relative pose (rotation and translation) expressed in the left image's frame.

### Task 3a (2pt): Fundamental Matrix Estimation
The fundamental matrix is calculated using the `cv2.findFundamentalMat` function from the OpenCV library. The fundamental matrix relates corresponding points in two images and is a crucial component in stereo vision. The calculated fundamental matrix is provided in the answer section.

### Task 3b (1pt): Drawing Epipolar Lines
Epipolar lines in both images are drawn, showcasing the geometric relationship between the two views. Visualization of epipolar lines is included in the answer section.

![Epipolar Lines](path/to/Figure9_Epipolar_Lines.jpg)

### Task 3c (1pt): Relative Pose Estimation
The relative pose (rotation matrix `R` and translation vector `t`) between the two images is found. The decomposition involves calculating the essential matrix `E` and extracting `R` and `t` from it. Answers to the questions regarding the process are provided in the answer section.

Rotation Matrix (R):
```
[[ 0.50491267  0.86315416  0.00530047]
 [ 0.86316452 -0.50487697 -0.00680119]
 [-0.00319439  0.00800919 -0.99996282]]
```

Translation Vector (t):
```
[ 0.52035892 -0.85393759  0.00414637]
```

## Author
- Name: Ashiq Rahman Anwar Batcha
