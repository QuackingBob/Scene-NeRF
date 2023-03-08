import numpy as np
import cv2
import pose_estimation
import json
import matplotlib.pyplot as plt
import math

def trans_t(t, theta, phi):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]], dtype=np.float32)
    # return np.array([
    #     [1, 0, 0, t * np.cos(theta) * np.cos(phi)],
    #     [0, 1, 0, t * np.cos(theta) * np.sin(phi)],
    #     [0, 0, 1, t * np.sin(theta)],
    #     [0, 0, 0, 1]], dtype=np.float32)

# def rot_phi(phi):
#     return np.array([
#         [np.cos(phi), -np.sin(phi), 0, 0],
#         [np.sin(phi), np.cos(phi), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]], dtype=np.float32)

def rot_phi(phi):
    return np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]], dtype=np.float32)

def rot_theta(th):
    return np.array([
        [np.cos(th), 0,-np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]], dtype=np.float32)

def pose_spherical(radius, theta, phi):
    '''
        The codes for generating random pose is based on https://github.com/yenchenlin/nerf-pytorch/blob/ec26d1c17d9ba2a897bc2ab254a0e15fce0d83b8/load_LINEMOD.py.
        and modified to make each coordinate indicates below.

               theta
                 ^
        - phi <= o => phi 
                /
            radius 

        Args:
            components of spherical coordinates.
    '''
    # c2w = np.eye(4)
    c2w = trans_t(radius, theta, phi)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    # c2w[:3, 3] = np.array([t * np.cos(theta) * np.cos(phi), t * np.cos(theta) * np.sin(phi), t * np.sin(theta)])
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

json_file = open('testjson.json', 'r')
data = json.load(json_file)
poses = [np.array(i["transform_matrix"]) for i in data["frames"]]
filtered = []

for pose in poses:
    r, t, p = pose_estimation.spherical_from_pose(pose)
    if r > 1 and r < 1.3:
        filtered.append(pose) 
filtered.append(np.eye(4))

pose_estimation.visualize_camera_poses(filtered)
pose_estimation.visualize_camera_poses_and_orientation(filtered)

x = []
y = []

reconstructed_pose = []

for i in filtered:
    r, t, p = pose_estimation.spherical_from_pose(i)
    print(f'r {r:.2f}, t {t:.3f}, p {p:.3f}')
    print(pose)
    new_pose = pose_spherical(r, t, p)
    print(new_pose)
    reconstructed_pose.append(new_pose)
    print("_" * 80)
    x.append(t)
    y.append(p)

pose_estimation.visualize_camera_poses(reconstructed_pose)
pose_estimation.visualize_camera_poses_and_orientation(reconstructed_pose)


def func(x):
    return -math.pi/2.0 * np.sin(x) + math.pi/2.0

x2 = np.linspace(-math.pi, math.pi, 100)
y2 = func(x2)

plt.scatter(y, x)
plt.plot(x2, y2)
plt.show()