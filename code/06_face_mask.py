###############################################################################
### Simple demo with video input of face
### Input : Live video of face
### Output: Overlay 3D face mesh on image
###############################################################################

import cv2
import time

from utils_display import DisplayFaceMask
from utils_mediapipe import MediaPipeFace


# Load mediapipe class
pipe = MediaPipeFace(static_image_mode=False, max_num_faces=1)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

# Load display class
ret, img = cap.read(0) # Read in a sample image
disp = DisplayFaceMask(img=img, draw3d=True, max_num_faces=1)

prev_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Preprocess image if necessary
    # img = cv2.flip(img, 1) # Flip image for 3rd person view
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = pipe.forward(img)

    # Compute FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    param[0]['fps'] = fps
    prev_time = curr_time

    img.flags.writeable = True
    cv2.imshow('img', img)

    # Display 3D
    img = disp.draw3d(param, img)

    key = cv2.waitKey(1)
    if key==27:
        break

pipe.pipe.close()
cap.release()
