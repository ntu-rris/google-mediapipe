###############################################################################
### Simple demo on gesture recognition
### Input : Live video of hand
### Output: 2D display of hand keypoint 
###         with gesture classification
### Usage : python 02_gesture.py -m train (to log data)
###       : python 02_gesture.py -m eval  (to perform gesture recognition)
###############################################################################

import cv2
import argparse

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='eval', help='train / eval')
args = parser.parse_args()
mode = args.mode

# Load mediapipe hand class
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)

# Load display class
disp = DisplayHand(max_num_hands=1)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

# Load gesture recognition class
gest = GestureRecognition(mode)

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
    if (param[0]['class'] is not None) and (mode=='eval'):
        param[0]['gesture'] = gest.eval(param[0]['angle'])

    img.flags.writeable = True

    # Display keypoint
    cv2.imshow('img 2D', disp.draw2d(img.copy(), param))

    key = cv2.waitKey(1)
    if key==27:
        break
    if key==32 and (param[0]['class'] is not None) and (mode=='train'):
        # Press spacebar to log training data
        # Note: Need to manually change class label
        # 'fist','one','two','three','four','five','six',
        # 'rock','spiderman',
        # 'yeah','ok',
        gest.train(param[0]['angle'], gest.gesture['fist'])
        print('Saved', counter) # Log around 10 for each class
        counter += 1
    if key==32 and (param[0]['class'] is not None) and (mode=='eval'):
        cv2.waitKey(0) # Pause display until user press any key        

pipe.pipe.close()
cap.release()
