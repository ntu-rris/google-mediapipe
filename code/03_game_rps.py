###############################################################################
### Simple demo on playing rock paper scissor
### Input : Live video of 2 hands playing rock paper scissor
### Output: 2D display of hand keypoint 
###         with gesture classification (rock=fist, paper=five, scissor=three/yeah)
###############################################################################

import cv2

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition


# Load mediapipe hand class
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2)

# Load display class
disp = DisplayHand(max_num_hands=2)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

# Load gesture recognition class
gest = GestureRecognition(mode='eval')

counter = 0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Flip image for 3rd person view
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = pipe.forward(img)
    # Evaluate gesture for all hands
    for p in param:
        if p['class'] is not None:
            p['gesture'] = gest.eval(p['angle'])

    img.flags.writeable = True

    # Display keypoint and result of rock paper scissor game
    cv2.imshow('Game: Rock Paper Scissor', disp.draw_game_rps(img.copy(), param))

    key = cv2.waitKey(1)
    if key==27:
        break

pipe.pipe.close()
cap.release()
