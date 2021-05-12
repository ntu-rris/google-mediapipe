###############################################################################
### Simple demo with at least 2 cameras for triangulation
### Input : Live videos of face / hand / body
###       : Calibrated camera intrinsics and extrinsics
### Output: 2D/3D (triangulated) display of hand, body keypoint/joint
### Usage : python 07_triangulate.py -m body --use_panoptic_dataset
###############################################################################

import cv2
import sys
import time
import argparse
import numpy as np
import open3d as o3d

from utils_display import DisplayHand, DisplayBody, DisplayHolistic
from utils_mediapipe import MediaPipeHand, MediaPipeBody, MediaPipeHolistic
from utils_3d_reconstruct import Triangulation


# User select mode
parser = argparse.ArgumentParser()
parser.add_argument('--use_panoptic_dataset', action='store_true')
parser.add_argument('-m', '--mode', default='body',
    help='Select mode: hand / body / holistic')
args = parser.parse_args()
mode = args.mode

# Define list of camera index
# cam_idx = [4,10] # Note: Hardcoded for my setup
# Read from .mp4 file
if args.use_panoptic_dataset:
    # Test with 2 views
    cam_idx = ['../data/171204_pose1_sample/hdVideos/hd_00_00.mp4',
               '../data/171204_pose1_sample/hdVideos/hd_00_11.mp4']

    # # Test with n views
    # num_views = 5 # Note: Maximum 31 hd cameras but processing time will be extremely slow
    # cam_idx = []
    # for i in range(num_views):
    #     cam_idx.append(
    #         '../data/171204_pose1_sample/hdVideos/hd_00_'+str(i).zfill(2)+'.mp4')

# Start video capture
cap = [cv2.VideoCapture(cam_idx[i]) for i in range(len(cam_idx))] 

# Define list of other variable
img   = [None for i in range(len(cam_idx))] # Store image
pipe  = [None for i in range(len(cam_idx))] # MediaPipe class
disp  = [None for i in range(len(cam_idx))] # Display class
param = [None for i in range(len(cam_idx))] # Store pose parameter
prev_time = [time.time() for i in range(len(cam_idx))]

# Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)
vis.get_render_option().point_size = 5.0

# Load triangulation class
tri = Triangulation(cam_idx=cam_idx, vis=vis, 
    use_panoptic_dataset=args.use_panoptic_dataset)

# Load mediapipe and display class
if mode=='hand':
    for i in range(len(cam_idx)):
        pipe[i] = MediaPipeHand(static_image_mode=False, max_num_hands=1)
        disp[i] = DisplayHand(draw3d=True, max_num_hands=1, vis=vis)
elif mode=='body':
    for i in range(len(cam_idx)):
        pipe[i] = MediaPipeBody(static_image_mode=False, model_complexity=1)
        disp[i] = DisplayBody(draw3d=True, vis=vis)
elif mode=='holistic':
    for i in range(len(cam_idx)):
        pipe[i] = MediaPipeHolistic(static_image_mode=False, model_complexity=1)
        disp[i] = DisplayHolistic(draw3d=True, vis=vis)
else:
    print('Undefined mode only the following modes are available: \n hand / body / holistic')
    sys.exit()

while True:
    # Loop through video capture
    for i, c in enumerate(cap):
        if not c.isOpened():
            break
        ret, img[i] = c.read()
        if not ret:
            break

        # Preprocess image if necessary
        # img[i] = cv2.flip(img[i], 1) # Flip image for 3rd person view

        # To improve performance, optionally mark image as not writeable to pass by reference
        img[i].flags.writeable = False

        # Feedforward to extract keypoint
        param[i] = pipe[i].forward(img[i])

        img[i].flags.writeable = True

        # Compute FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time[i])
        if mode=='body':
            param[i]['fps'] = fps
        elif mode=='hand':
            param[i][0]['fps'] = fps
        elif mode=='holistic':
            for p in param[i]:
                p['fps'] = fps
        prev_time[i] = curr_time

    # Perform triangulation
    if args.use_panoptic_dataset:
        if len(cam_idx)==2:
            param = tri.triangulate_2views(param, mode)
        else:
            param = tri.triangulate_nviews(param, mode)
    
    for i in range(len(cam_idx)):
        # Display 2D keypoint
        img[i] = disp[i].draw2d(img[i].copy(), param[i])
        img[i] = cv2.resize(img[i], None, fx=0.5, fy=0.5)
        cv2.imshow('img'+str(i), img[i])
        # Display 3D
        disp[i].draw3d(param[i])

    vis.update_geometry(None)
    vis.poll_events()
    vis.update_renderer()

    key = cv2.waitKey(1)
    if key==27:
        break

# vis.run() # Keep 3D display for visualization

for p, c in zip(pipe, cap):
    p.pipe.close()
    c.release()
