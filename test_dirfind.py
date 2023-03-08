import os
import numpy as np
import cv2
from einops import rearrange

w, h = 1000, 1000

data_dir = "temp"

image_depth_fname_dict = {}
sample_image = None

for root, dirs, files in os.walk(data_dir):
    for file_name in files:
        if file_name.endswith(".jpg"):  # assuming your image files end with .png
            if "img" in root.split(os.path.sep):
                img_path = os.path.join(root, file_name)
                depth_dir = os.path.join(root[:root.find("img")-1], "depthmap")
                depth_path = os.path.join(depth_dir, os.path.basename(file_name)[:-3] + "h5")
                image_depth_fname_dict[img_path] = depth_path
                if sample_image is None:
                    img_temp = cv2.imread(img_path)
                    img_temp = cv2.resize(img_temp, (w, h))
                    img_temp = np.divide(img_temp, 255.)
                    img_temp = rearrange(img_temp, 'h w c -> c h w')
                    sample_image = img_temp

print(image_depth_fname_dict)

cv2.imshow('sample', sample_image)
cv2.waitKey(0)
