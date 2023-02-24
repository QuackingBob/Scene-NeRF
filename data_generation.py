import cv2
from PIL import Image
import numpy

import cv2
import numpy as np
import glob

fx = 1.50889127e03 # focal lengths x
fy = 1.49796532e03 # focal lengths y
cx = 9.98070558e02 # principal point of the camera x
cy = 5.51627933e+02 # principal point of the camera y

# Define the camera matrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Initialize the SfM parameters
sfm_params = cv2.SfM_InitParameters()
sfm_params.calc_pose = True
sfm_params.use_fisheye_model = False
sfm_params.use_gpu = False

# Create the SfM object
sfm = cv2.StructureFromMotion_create(sfm_params)

# Initialize the image list and camera poses
image_list = []
camera_poses = []

images = glob.glob('calibration_images/*.jpg')
num_images = len(images)

# Loop over the images and estimate the camera poses
for i in range(num_images):
    # Load the image
    image = cv2.imread(images[i])

    # Extract feature points from the image
    keypoints = cv2.xfeatures2d.SIFT_create().detect(image)
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(image, keypoints)

    # Add the image and feature points to the SfM object
    sfm.update(image, keypoints)

    # Estimate the camera pose
    pose = sfm.estimatePartial()

    # Add the camera pose to the list
    camera_poses.append(pose)

    # Add the image to the image list
    image_list.append(image)

    # Print progress
    print(f"Processed image {i+1}/{num_images}")

# Refine the camera poses and 3D points
reprojection_error, camera_poses, points3D = sfm.finalize()

# Print the reprojection error
print(f"Reprojection error: {reprojection_error}")

# Save the camera poses and 3D points to a file
np.savetxt("camera_poses.txt", np.array(camera_poses))
np.savetxt("points3D.txt", np.array(points3D))

