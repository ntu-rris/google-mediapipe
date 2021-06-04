###############################################################################
### Simple demo with video input of human
### Input : Live video of human
### Output: Segments human for selfie effect/video conferencing
###############################################################################

import cv2
import time
import argparse

from utils_mediapipe import MediaPipeSeg


# User select mode
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='general',
    help='Select mode: general / landscape')
args = parser.parse_args()
if args.mode=='general': mode = 0
elif args.mode=='landscape': mode = 1

# Load mediapipe class
pipe = MediaPipeSeg(model_selection=mode)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

prev_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Feedforward to perform segmentation
    img = pipe.forward(img)

    # Compute FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(img, 'FPS: %.1f' % (fps),
        (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('img', img)

    key = cv2.waitKey(1)
    if key==27:
        break

pipe.pipe.close()
cap.release()