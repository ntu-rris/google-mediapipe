###############################################################################
### Simple demo with a single color image
### Input : Color image of upper body
### Output: 2D display of hand and body keypoint
###############################################################################

import os
import cv2
from utils_display import DisplayHand, DisplayBody
from utils_mediapipe import MediaPipeHand, MediaPipeBody


# Load mediapipe class
body = MediaPipeBody(static_image_mode=True)
hand = MediaPipeHand(static_image_mode=True, max_num_hands=2)

# Load display class
disp_body = DisplayBody(draw3d=False)
disp_hand = DisplayHand(draw3d=False)

# Read in image
img = cv2.imread('../data/body.jpg')
# img = cv2.imread('../data/mona.jpg')

# # Preprocess image
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# img = cv2.flip(img, 1)
# # Select ROI
# r = cv2.selectROI(img)
# # Crop image
# img = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

# Feedforward to extract body param
param_body = body.forward(img)
param_hand = hand.forward(img)

img = disp_body.draw2d(img.copy(), param_body)
img = disp_hand.draw2d(img.copy(), param_hand)

# Display 2D keypoint
cv2.imshow('img 2D', img)
cv2.waitKey(0)

body.body.close()
hand.hand.close()
