import cv2
import numpy as np

# Step 1: Extract feature points from images
sift = cv2.SIFT_create()
kp_list = []
desc_list = []
for i in range(len(images)):
    kp, desc = sift.detectAndCompute(images[i], None)
    kp_list.append(kp)
    desc_list.append(desc)

# # Step 2: Match feature points across images
# matcher = cv2.FlannBasedMatcher()
# matches_list = []
# for i in range(len(images) - 1):
#     matches = matcher.match(desc_list[i], desc_list[i+1])
#     matches_list.append(matches)

# # Step 3: Compute essential matrix
# K = np.eye(3)  # Camera intrinsic matrix
# E_list = []
# for matches in matches_list:
#     pts1 = np.float32([kp_list[i][m.queryIdx].pt for m in matches])
#     pts2 = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches])
#     E, mask = cv2.findEssentialMat(pts1, pts2, K)
#     E_list.append(E)

# # Step 4: Decompose essential matrix
# R_list = []
# t_list = []
# for E in E_list:
#     _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
#     R_list.append(R)
#     t_list.append(t)

# # Step 5: Reconstruct camera poses
# pose_list = [np.eye(4)]
# for i in range(len(images) - 1):
#     pose = np.zeros((4, 4))
#     pose[:3, :3] = R_list[i]
#     pose[:3, 3] = t_list[i].reshape(3)
#     pose_list.append(pose @ pose_list[-1])

for i in range(len(images) - 1):
    # Step 3: Detect and match keypoints
    kp1, des1 = detect_and_compute(images[i])
    kp2, des2 = detect_and_compute(images[i+1])

    # Match keypoints using FLANN matcher
    matches = flann_matcher(des1, des2)

    # Filter matches using Lowe's ratio test
    good_matches = lowe_ratio_test(matches)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Step 4: Estimate fundamental matrix and essential matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)

    # Step 5: Recover camera poses
    E = mtx.T @ F @ mtx
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)

    pose = np.zeros((4, 4))
    pose[:3, :3] = R
    pose[:3, 3] = t.ravel()
    pose[3, 3] = 1
    pose_list[i+1] = pose.dot(pose_list[i])


# Step 6: Refine camera poses (optional)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_points = np.zeros((len(kp_list), 1, 3), dtype=np.float32)
obj_points[:, :, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 1, 2)
obj_points_list = [obj_points] * len(images)
ret


# Step 6: Refine camera poses (optional)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_points = np.zeros((len(kp_list), 1, 3), dtype=np.float32)
obj_points[:, :, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 1, 2)
obj_points_list = [obj_points] * len(images)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, kp_list, images[0].shape[::-1], None, None)

for i in range(len(images)):
    _, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, kp_list[i], mtx, dist)
    pose = np.zeros((4, 4))
    pose[:3, :3], _ = cv2.Rodrigues(rvec)
    pose[:3, 3] = tvec.reshape(3)
    pose_list[i] = pose

    # Refine camera poses and 3D points using bundle adjustment
    obj_points_new = np.zeros((len(kp_list[i]), 1, 3), dtype=np.float32)
    obj_points_new[:, :, :2] = np.array([kp.pt for kp in kp_list[i]]).reshape(-1, 1, 2)
    _, pose_list[i], obj_points, _ = cv2.solvePnPRansac(obj_points_new, kp_list[i], mtx, dist, pose_list[i][:3, :4], criteria=criteria)
# Visualize camera poses
for i in range(len(images)):
    image = images[i].copy()
    axis_len = min(image.shape[:2]) // 2
    cv2.drawFrameAxes(image, mtx, dist, pose_list[i][:3, :4], axis_len)
    cv2.imshow(f"Image {i}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
