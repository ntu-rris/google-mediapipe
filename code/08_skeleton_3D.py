###############################################################################
### Simple demo on displaying 3D hand/body skeleton
### Input : Live video of hand/body
### Output: 3D display of hand/body skeleton 
### Usage : python 08_skeleton_3D.py -m hand
###       : python 08_skeleton_3D.py -m body
###       : python 08_skeleton_3D.py -m holistic
###############################################################################

import cv2
import time
import argparse

from utils_display import DisplayHand, DisplayBody, DisplayHolistic
from utils_mediapipe import MediaPipeHand, MediaPipeBody, MediaPipeHolistic


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='hand', help=' Select mode: hand / body / holistic')
args = parser.parse_args()
mode = args.mode

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0
# cap = cv2.VideoCapture('../data/video.mp4') # Read from .mp4 file

# Read in sample image to estimate camera intrinsic
ret, img = cap.read(0)
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
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

# Load mediapipe and display class
if mode=='hand':
    pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2, intrin=intrin)
    disp = DisplayHand(draw3d=True, draw_camera=True, max_num_hands=2, intrin=intrin)
elif mode=='body':
    # Note: As of version 0.8.3 3D joint estimation is only available in full body mode
    pipe = MediaPipeBody(static_image_mode=False, model_complexity=1, intrin=intrin)
    disp = DisplayBody(draw3d=True, draw_camera=True, intrin=intrin)
elif mode=='holistic':
    # Note: As of version 0.8.3 3D joint estimation is only available in full body mode
    pipe = MediaPipeHolistic(static_image_mode=False, model_complexity=1, intrin=intrin)
    disp = DisplayHolistic(draw3d=True, draw_camera=True, intrin=intrin)

prev_time = time.time()
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back
        ret, img = cap.read()
        # break

    # Flip image for 3rd person view
    img = cv2.flip(img, 1)
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feedforward to extract keypoint
    param = pipe.forward(img)

    # Compute FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    if mode=='body':
        param['fps'] = fps
    elif mode=='face' or mode=='hand':
        param[0]['fps'] = fps
    elif mode=='holistic':
        for p in param:
            p['fps'] = fps
    prev_time = curr_time    

    img.flags.writeable = True

    # Display keypoint
    cv2.imshow('img 2D', disp.draw2d(img, param))
    # Display 3D
    disp.draw3d(param,)
    disp.draw3d_(param, img)
    disp.vis.update_geometry(None)
    disp.vis.poll_events()
    disp.vis.update_renderer()    

    key = cv2.waitKey(1)
    if key==27:
        break
    if key==ord('r'): # Press 'r' to reset camera view
        disp.camera.reset_view()

pipe.pipe.close()
cap.release()
