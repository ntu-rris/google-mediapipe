###############################################################################
### Simple demo on measuring hand rom
### Input : Live video of hand
### Output: 2D display of hand keypoint 
###         with hand rom classification and corresponding joint angle
### Usage : python 04_hand_rom.py -m train (to log data)
###       : python 04_hand_rom.py -m eval  (to perform hand rom recognition)
###############################################################################

import cv2
import argparse

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import HandRomRecognition


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

# Load hand rom recognition class
gest = HandRomRecognition(mode)

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
        # 'Finger MCP Flexion'    :0,
        # 'Finger PIP DIP Flexion':1,
        # 'Thumb MCP Flexion'     :2,
        # 'Thumb IP Flexion'      :3,
        # 'Thumb Radial Abduction':4,
        # 'Thumb Palmar Abduction':5, # From this class onwards hard to differentiate
        # 'Thumb Opposition'      :6,
        # 'Forearm Pronation'     :7, # Not done yet
        # 'Forearm Supination'    :8, # Not done yet
        # 'Wrist Flexion'         :9, # Not done yet
        # 'Wrist Extension'       :10,# Not done yet
        # 'Wrist Radial Deviation':11,# Not done yet
        # 'Wrist Ulnar Deviation' :12,# Not done yet
        gest.train(param[0]['angle'], gest.gesture['Finger MCP Flexion'])
        print('Saved', counter) # Log around 10 for each class
        counter += 1
    if key==32 and (param[0]['class'] is not None) and (mode=='eval'):
        cv2.waitKey(0) # Pause display until user press any key

pipe.pipe.close()
cap.release()
