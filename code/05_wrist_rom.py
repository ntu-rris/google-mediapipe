###############################################################################
### Simple demo on measuring wrist and forearm rom
### Input : Live video of hand and upper body
### Output: 2D display of hand and upper body keypoint 
###         with corresponding joint angle
### Usage : python 05_wrist_rom.py -m 0 -s right
###         python 05_wrist_rom.py -m 1 -s right
###         python 05_wrist_rom.py -m 2 -s right
###         python 05_wrist_rom.py -m 0 -s left
###         python 05_wrist_rom.py -m 1 -s left
###         python 05_wrist_rom.py -m 2 -s left
###############################################################################

import cv2
import sys
import argparse

from utils_display import DisplayHolistic, DisplayHand
from utils_mediapipe import MediaPipeHolistic, MediaPipeHand
from utils_joint_angle import WristArmRom


# User select mode of measurement
# 0: Wrist flexion/extension
# 1: Wrist radial/ulnar deviation
# 2: Forearm pronation/supination
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--side', default='right')
parser.add_argument('-m', '--mode', default=2, 
    help='Select mode: \
    0:Wrist flex/ext, \
    1:Wrist radial/ulnar dev, \
    2:Forearm pronation/supination')
args = parser.parse_args()
mode = int(args.mode)
side = args.side

# Load mediapipe and display class
if mode==0 or mode==1: # Note; To determine wrist rom need upper body elbow joint
    pipe = MediaPipeHolistic(static_image_mode=False, model_complexity=1)
    disp = DisplayHolistic(draw3d=True)
elif mode==2:
    pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)
    disp = DisplayHand(draw3d=True, max_num_hands=1)
else:
    print('Undefined mode only 3 modes are available: \n \
        0:Wrist flex/ext, \n \
        1:Wrist radial/ulnar dev, \n \
        2:Forearm pronation/supination')
    sys.exit()

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

# Load wrist and forearm rom class
rom = WristArmRom(mode, side)

while cap.isOpened():
    ret, img = cap.read()
    # img = cv2.imread('../data/sample/wrist_extension.png') # python 05_wrist_rom.py -m 0 -s right
    # img = cv2.imread('../data/sample/wrist_flexion.png')   # python 05_wrist_rom.py -m 0 -s right
    # img = cv2.imread('../data/sample/wrist_radial.png')    # python 05_wrist_rom.py -m 1 -s right
    # img = cv2.imread('../data/sample/wrist_ulnar.png')     # python 05_wrist_rom.py -m 1 -s right
    if not ret:
        break

    # Flip image for 3rd person view
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = pipe.forward(img)

    img.flags.writeable = True

    # Compute wrist/forearm range of motion
    param = rom.eval(param)

    # Display keypoint
    cv2.imshow('img 2D', disp.draw2d(img.copy(), param))
    # Display 3D
    disp.draw3d(param)
    disp.vis.update_geometry(None)
    disp.vis.poll_events()
    disp.vis.update_renderer()    

    key = cv2.waitKey(1)
    if key==27:
        break
    if key==32:
        cv2.imwrite('img.png', disp.draw2d(img.copy(), param))
        cv2.waitKey(0) # Pause display until user press any key

pipe.pipe.close()
cap.release()
