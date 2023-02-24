import opensfm
from opensfm import reconstruction, transformations
import glob

reconstruction_data = reconstruction.ReconstructionManager()

image_paths = glob.glob('calibration_images/*.jpg')

for image_path in image_paths:
    reconstruction_data.add_image(image_path)

reconstruction_data.invent_reference_lla()
reconstruction_data.invent_reference()
reconstruction_data.add_reconstruction()
reconstruction_data.bundle()

for image_path in image_paths:
    image = reconstruction_data.images[image_path]
    pose = image.pose.get_origin_and_rotation()
    R = transformations.rotation_matrix(pose[2], [0, 0, 1])[:3, :3]
    t = pose[0]