###############################################################################
### Simple demo with a single color image
### Input : Color image of object
### Output: 2D/3D display of object's bounding box
### Usage : python 09_objectron.py -m shoe
###         python 09_objectron.py -m chair
###         python 09_objectron.py -m cup
###         python 09_objectron.py -m camera
###############################################################################

import cv2
import argparse

from utils_display import DisplayObjectron
from utils_mediapipe import MediaPipeObjectron


# User select mode
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='shoe', 
    help='Select mode: shoe / chair / cup / camera')
args = parser.parse_args()

# Read in image (Note: You can change the file path to your own test image)
file = '../data/sample/object.png'
img  = cv2.imread(file)

# Estimate camera intrinsic
img_width  = img.shape[1]
img_height = img.shape[0]
intrin = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}

# Load mediapipe
pipe = MediaPipeObjectron(static_image_mode=True, max_num_objects=5, 
  model_name=args.mode, intrin=intrin)

# Load display class
disp = DisplayObjectron(draw3d=True, draw_camera=True, intrin=intrin, max_num_objects=5)

# Feedforward to extract pose param
param = pipe.forward(img)

# # Display 2D keypoint
cv2.imshow('img', disp.draw2d(img, param))
cv2.waitKey(0) # Press escape to dispay 3D view

# Display 3D joint
disp.draw3d(param, img)
disp.vis.update_geometry(None)
disp.vis.poll_events()
disp.vis.update_renderer()
disp.vis.run()

pipe.pipe.close()