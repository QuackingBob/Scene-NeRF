import cv2
import numpy as np

# Load the images
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Create an ORB object for feature detection and description
orb = cv2.ORB_create()

# Detect and compute features in the images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match the features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

# Find the fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Extract the inliers
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find the essential matrix from the fundamental matrix
K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
E = K.T @ F @ K

# Find the camera poses
_, R, t, _ = cv2.recoverPose(E, pts1, pts2)

# Triangulate the 3D points
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
P2 = np.hstack((R, t))
points_3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d /= points_3d[3]

# TODO Perform Bundle Adjustment to refine the camera poses and 3D points

