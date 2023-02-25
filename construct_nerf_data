import cv2
import numpy as np
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pose_estimation
from calibration import ResizeWithAspectRatio
import os
from sklearn.model_selection import train_test_split
import json

def save_json(images, poses, fname, path, fov, angle):
    """Save images and poses into proper train format"""
    if not os.path.exists(os.path.join(path, fname)):
        os.makedirs(os.path.join(path, fname))
    
    json_dict = {
        "camera_angle_x": fov,
        "frames": []
    }

    counter = 0

    for image, pose in list(zip(images, poses)):
        img_path = f"./{fname}/i_{counter}"
        img_dict = {
            "file_path": img_path,
            "rotation": angle,
            "transform_matrix": pose.tolist()
        }
        json_dict["frames"].append(img_dict)
        cv2.imwrite(os.path.join(path, fname, f"i_{counter}.png"), image)
        counter += 1
    
    with open(os.path.join(path, fname + ".json"), "w") as jsonfile:
        json.dump(json_dict, jsonfile)



def main():
    if not os.path.exists('nerf_data'):
        os.makedirs('nerf_data')
    
    if not os.path.exists('nerf_data/train'):
        os.makedirs('nerf_data/train')
        os.makedirs('nerf_data/test')
        os.makedirs('nerf_data/val')

    width = 1000
    
    image_paths = glob.glob('output/*.jpg')
    print(image_paths)
    images = [ResizeWithAspectRatio(cv2.imread(i), width=width) for i in image_paths]

    fx = 1.00294889e+03 # focal lengths x
    fy = 1.00168472e+03 # focal lengths y
    cx = 4.88588193e+02 # principal point of the camera x
    cy = 6.61380661e+02 # principal point of the camera y

    # Define the camera matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # define distortion coeff
    distortion_coeff = np.array([0.06622152, -0.0631438, -0.00363991, -0.0006693, -0.3078563])

    # get field of view param (we need fovx to train the nerf)
    fov_x = np.rad2deg(2 * np.arctan2(width, 2 * fx))
    # fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    # get camera pose matrices and visualize them
    pose_list = pose_estimation.estimate_camera_pose(images, K, distortion_coeff)
    pose_estimation.visualize_camera_poses_and_orientation(pose_list, scale=0.1)

    # combine datalists together
    data = list(zip(images, pose_list))
    
    # split into train, test, val
    train_and_val, test = train_test_split(data, test_size=0.1, random_state=42) # the meaning of life, the universe and everything
    train, val = train_test_split(train_and_val, test_size=0.1, random_state=42)

    # seperate data into individual lists
    train_images, train_poses = zip(*train)
    test_images, test_poses = zip(*test)
    val_images, val_poses = zip(*val)

    save_json(train_images, train_poses, "train", "nerf_data", fov_x, 0.012566370614359171)
    save_json(test_images, test_poses, "test", "nerf_data", fov_x, 0.012566370614359171)
    save_json(val_images, val_poses, "val", "nerf_data", fov_x, 0.012566370614359171)


if __name__ == "__main__":
    main()