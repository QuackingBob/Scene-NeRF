import cv2
import numpy as np
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

image_paths = glob.glob('calibration_images/*.jpg')
images = [cv2.imread(i) for i in image_paths]

import numpy as np
import cv2

def detect_and_compute(image):
    """Detects and computes keypoints and descriptors in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None) # keypoint and description
    return kp, des

def flann_matcher(des1, des2):
    """Matches keypoints in two sets of descriptors using FLANN matcher."""
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def lowe_ratio_test(matches, threshold=0.7):
    """Filters matches using Lowe's ratio test."""
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_camera_pose(images, mtx, dist):
    """Estimates the camera pose from a sequence of images and camera matrix."""
    # Initialize camera pose list
    pose_list = [np.eye(4)] * len(images)

    # Iterate through image sequence
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
        # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
        # E = mtx.T @ F @ mtx
        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.RANSAC, prob=0.99, threshold=1.0)

        # Step 5: Recover camera poses
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)

        pose = np.zeros((4, 4))
        pose[:3, :3] = R
        pose[:3, 3] = t.ravel()
        pose[3, 3] = 1
        pose_list[i+1] = pose.dot(pose_list[i])

    return pose_list


def visualize_camera_poses(pose_list, scale=1):
    """Visualizes the camera poses in 3D using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract camera positions
    positions = [pose[:3, 3] for pose in pose_list]

    # Scale positions
    positions = np.array(positions) * scale

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c='r')

    # Set plot limits
    ax.set_xlim3d([np.min(positions[:, 0]), np.max(positions[:, 0])])
    ax.set_ylim3d([np.min(positions[:, 1]), np.max(positions[:, 1])])
    ax.set_zlim3d([np.min(positions[:, 2]), np.max(positions[:, 2])])

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses')

    plt.show()


def visualize_camera_poses_and_orientation(pose_list, scale=1):
    """Visualizes the camera poses in 3D using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract camera positions and orientations
    positions = [pose[:3, 3] for pose in pose_list]
    orientations = [pose[:3, :3] for pose in pose_list]

    # Scale positions
    positions = np.array(positions) * scale

    # Plot camera positions and orientations
    for i in range(len(positions)):
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], c='r', marker='o')

        x, y, z = positions[i]
        u, v, w = orientations[i] @ np.array([1, 0, 0])

        ax.quiver(x, y, z, u, v, w, length=0.1, color='b')

    # Set plot limits
    ax.set_xlim3d([np.min(positions[:, 0]), np.max(positions[:, 0])])
    ax.set_ylim3d([np.min(positions[:, 1]), np.max(positions[:, 1])])
    ax.set_zlim3d([np.min(positions[:, 2]), np.max(positions[:, 2])])

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses')

    plt.show()


def heading_from_pose(pose):
    """From a 4x4 pose matrix, extract rotation matrix and get euler angles
    
    pose:
    R R R T
    R R R T
    R R R T
    0 0 0 1

    rotation matrix:
    R R R
    R R R
    R R R

    """
    # Define a rotation matrix
    R = pose[:3, :3]

    # Define the identity matrix with known directional vectors
    I = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])

    # Compute the inverse of the rotation matrix
    Rinv = np.linalg.inv(R)

    # Apply the inverse of the rotation matrix to the identity matrix
    Irot = Rinv @ I

    # Extract the theta and phi angles from the resulting matrix
    theta = np.arctan2(Irot[2, 1], Irot[2, 2])
    phi = np.arctan2(-Irot[2, 0], np.sqrt(Irot[0, 0]**2 + Irot[1, 0]**2))

    return (theta, phi)


def main():
    fx = 1.50839653e3 # focal lengths x
    fy = 1.49710157e3 # focal lengths y
    cx = 9.98912401e2 # principal point of the camera x
    cy = 5.53356162e2 # principal point of the camera y

    # Define the camera matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    distortion_coeff = np.array([-3.23935068e-1, -6.68856908e-1, -1.79469111e-3, -3.29352539e-3, 3.10882365])

    pose_list = estimate_camera_pose(images, K, distortion_coeff)
    print(pose_list)
    angle_list = [heading_from_pose(i) for i in pose_list]
    print(angle_list)

    # Visualize camera poses
    visualize_camera_poses_and_orientation(pose_list, scale=0.1)


if __name__ == "__main__": 
    main()