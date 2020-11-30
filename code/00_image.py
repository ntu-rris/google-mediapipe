###############################################################################
### Simple demo with a single color image
### Input : Color image of hand
### Output: 2D   display of hand keypoint
###         2.5D display of hand keypoint
###         3D   display of hand joint
###############################################################################

import os
import cv2
from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand


# Load mediapipe hand class
hand = MediaPipeHand(static_image_mode=True, max_num_hands=2)

# Load display class
disp = DisplayHand(draw3d=True)

# Download test image of hand if it does not exist
if not os.path.exists('../data/hand.jpg'):
    hand.download_image()
# Read in image
img = cv2.imread('../data/hand.jpg')

# # Preprocess image
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
img = cv2.flip(img, 1)
# # Select ROI
# r = cv2.selectROI(img)
# # Crop image
# img = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

# Feedforward to extract hand param
param = hand.forward(img)

# Display 2D keypoint
cv2.imshow('img 2D', disp.draw2d(img.copy(), param))
# Display 2.5D keypoint
cv2.imshow('img 2.5D', disp.draw2d_(img.copy(), param))
cv2.waitKey(0) # Press escape to dispay 3D view

# Display 3D
disp.draw3d(param)
disp.vis.run()

hand.hand.close()
