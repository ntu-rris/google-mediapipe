###############################################################################
### Simple demo with video input
### Input : Live video of upper body
### Output: 2D display of hand and body keypoint
###############################################################################

import cv2
import time
import numpy as np
from utils_display import DisplayHand, DisplayBody
from utils_mediapipe import MediaPipeHand, MediaPipeBody


# Load mediapipe class
body = MediaPipeBody(static_image_mode=False)
hand = MediaPipeHand(static_image_mode=False, max_num_hands=2)

# Load display class
disp_body = DisplayBody(draw3d=False)
disp_hand = DisplayHand(draw3d=False)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0
# cap = cv2.VideoCapture('../data/video.mp4') # By default webcam is index 0

# Create video
# fps = 30
# width, height = 720, 480
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
# video = cv2.VideoWriter('../data/video_.mp4', fourcc, fps, (width, height))

prev_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract body param
    param_body = body.forward(img)
    param_hand = hand.forward(img)

    # Compute FPS
    curr_time = time.time()
    param_hand[0]['fps'] = 1/(curr_time-prev_time)
    param_body['fps'] = 1/(curr_time-prev_time)
    prev_time = curr_time

    img.flags.writeable = True

    # Display 2D keypoint
    img = disp_body.draw2d(img.copy(), param_body)
    img = disp_hand.draw2d(img.copy(), param_hand)
    cv2.imshow('img 2D', img)

    # Write to video
    # video.write(img)   

    key = cv2.waitKey(1)
    if key==27:
        break

body.body.close()
hand.hand.close()
# video.release()
cap.release()
