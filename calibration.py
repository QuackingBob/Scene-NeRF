
import numpy as np
import cv2
import glob

# Define the calibration pattern size and type
pattern_size = (7, 7)
pattern_type = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE

# Create the object points (3D coordinates of the calibration pattern corners)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

# Create arrays to store the 2D image points and 3D object points
img_points = [] # 2D points in image plane.
obj_points = [] # 3D points in real world space

# Load the calibration images
images = glob.glob('calibration_images/*.jpg')

# Loop over the images and detect the calibration pattern
for fname in images:
    # Load the image and convert to grayscale
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the calibration pattern
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=pattern_type)

    print(ret)
    print(corners)

    # If the pattern is found, add the points to the arrays
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)

# Calibrate the camera
ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera matrix:")
print(K)
print("Distortion coefficients:")
print(dist_coeff)

# Evaluate the calibration
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeff)
    error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
    mean_error += error

print("Mean reprojection error: {}".format(mean_error/len(obj_points)))

info_file = open("camerainfo.txt", "w")
info_file.write(f"Camera matrix:\n{K}\nDistortion Coefficients:\n{dist_coeff}\nMean reprojection error:\n{mean_error/len(obj_points)}")
info_file.close()