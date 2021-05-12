###############################################################################
### Simple demo with a single color image
### Input : Color image of face / hand / body
### Output: 2D/2.5D/3D display of face, hand, body keypoint/joint
### Usage : python 00_image.py -m face
###         python 00_image.py -m hand
###         python 00_image.py -m body
###         python 00_image.py -m holistic
###############################################################################

import cv2
import sys
import argparse

from utils_display import DisplayFace, DisplayHand, DisplayBody, DisplayHolistic
from utils_mediapipe import MediaPipeFace, MediaPipeHand, MediaPipeBody, MediaPipeHolistic


# User select mode
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='hand', 
    help='Select mode: face / hand / body / holistic')
args = parser.parse_args()
mode = args.mode

# Load mediapipe and display class
if mode=='face':
    pipe = MediaPipeFace(static_image_mode=True, max_num_faces=1)
    disp = DisplayFace(draw3d=True)
    file = '../data/sample/mona.png'
elif mode=='hand':
    pipe = MediaPipeHand(static_image_mode=True, max_num_hands=1)
    disp = DisplayHand(draw3d=True, max_num_hands=1)
    file = '../data/sample/hand.png'
elif mode=='body':
    pipe = MediaPipeBody(static_image_mode=True, model_complexity=1)
    disp = DisplayBody(draw3d=True)
    file = '../data/sample/upper_limb4.png'
elif mode=='holistic':
    pipe = MediaPipeHolistic(static_image_mode=True, model_complexity=1)
    disp = DisplayHolistic(draw3d=True)
    file = '../data/sample/lower_limb4.png'
else:
    print('Undefined mode only the following modes are available: \nface / hand / body / holistic')
    sys.exit()

# Read in image (Note: You can change the file path to your own test image)
img  = cv2.imread(file)

# # Preprocess image if necessary
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
img = cv2.flip(img, 1)
# # Select ROI
# r = cv2.selectROI(img)
# # Crop image
# img = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

# Feedforward to extract pose param
param = pipe.forward(img)

# Display 2D keypoint
cv2.imshow('img 2D', disp.draw2d(img.copy(), param))
# Display 2.5D keypoint
cv2.imshow('img 2.5D', disp.draw2d_(img.copy(), param))
cv2.waitKey(0) # Press escape to dispay 3D view

# Display 3D joint
disp.draw3d(param)
disp.vis.update_geometry(None)
disp.vis.poll_events()
disp.vis.update_renderer()
disp.vis.run()

pipe.pipe.close()
