import cv2
import os

# Open the video file
vidcap = cv2.VideoCapture('temp/scene.mp4')

# Time interval between each frame
dt = 1.0  # in seconds

# Create a folder to store the output images
if not os.path.exists('temp/sceneoutput'):
    os.makedirs('temp/sceneoutput')

# Initialize variables
success = True
count = 0
time = 0.0

while success:
    # Read the next frame from the video
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    success, image = vidcap.read()
    
    # Calculate the time of the current frame
    time += dt
    
    # Save the image to disk at regular intervals
    if success:
        cv2.imwrite(os.path.join('temp/sceneoutput', f"frame{count:04}.jpg"), image)
        count += 1

# Release the video file and close the output window
vidcap.release()
cv2.destroyAllWindows()
