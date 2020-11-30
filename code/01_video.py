###############################################################################
### Simple demo with video input
### Input : Live video of hand
### Output: 2D   display of hand keypoint
###         2.5D display of hand keypoint
###         3D   display of hand joint
###############################################################################

import cv2
import time
import numpy as np
from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand


# Load mediapipe hand class
hand = MediaPipeHand(static_image_mode=False, max_num_hands=2)

# Load display class
disp = DisplayHand(draw3d=True)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

counter = 0
logging = False
prev_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Flip image for 3rd person view
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = hand.forward(img)

    # Compute FPS
    curr_time = time.time()
    param[0]['fps'] = 1/(curr_time-prev_time)
    prev_time = curr_time

    img.flags.writeable = True

    # Display 2D keypoint
    cv2.imshow('img 2D', disp.draw2d(img.copy(), param))
    # Display 2.5D keypoint
    cv2.imshow('img 2.5D', disp.draw2d_(img.copy(), param))
    # Display 3D
    disp.draw3d(param)

    key = cv2.waitKey(1)
    if key==27:
        break

hand.hand.close()
cap.release()
