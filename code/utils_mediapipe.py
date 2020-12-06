###############################################################################
### Google MediaPipe for hand and upper body pose estimation
### https://github.com/google/mediapipe
###############################################################################

import cv2
import numpy as np
import mediapipe as mp
from utils_joint_angle import convert_3d_joint_to_angle
from utils_joint_angle import convert_relative_to_actual_3d_joint


class MediaPipeHand:
    def __init__(self, static_image_mode=True, max_num_hands=2, idx=None):
        super(MediaPipeHand, self).__init__()

        # Access MediaPipe Solutions Python API
        mp_hands = mp.solutions.hands
        # help(mp_hands.Hands)

        # Initialize MediaPipe Hands
        # Video: static_image_mode=False
        # Picture: static_image_mode=True
        self.hand = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)

        # Define hand parameters
        self.param = []
        for i in range(max_num_hands):
            p = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in absolute coordinate (m)
                'visible' : np.zeros(21), # Visibility (Note: Its always zero!)
                'presence': np.zeros(21), # Presence (Note: Its always zero!)
                'class'   : None, # Left / right hand
                'score'   : 0, # Confidence score
                'angle'   : np.zeros(15), # Joint angles
                'gesture' : None, # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        # Convert mediapipe hand result to my own param
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['class'] = None

        if result.multi_hand_landmarks is not None:
            # Loop through different hands
            for i, res in enumerate(result.multi_handedness):
                self.param[i]['class'] = res.classification[0].label
                self.param[i]['score'] = res.classification[0].score

            # Loop through different hands
            for i, res in enumerate(result.multi_hand_landmarks):
                # Loop through 21 landmark for each hand
                for j, lm in enumerate(res.landmark):
                    self.param[i]['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

                    # Ignore it first
                    # self.param[i]['visible'][j] = lm.visibility
                    # self.param[i]['presence'][j] = lm.presence

                # Convert relative 3D joint to angle
                self.param[i]['angle'] = convert_3d_joint_to_angle(self.param[i]['joint'])

        return self.param


    def download_image(self, url=None, dest='../data/hand.jpg'):
        import urllib.request
        if url==None:
            url = 'https://upload.wikimedia.org/wikipedia/commons/3/32/Human-Hands-Front-Back.jpg'
        print('Downloading image from', url)
        urllib.request.urlretrieve(url, dest)
        print('Saved image to', dest)


    def forward(self, img):
        # Preprocess image
        # img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract hand result
        result = self.hand.process(img)

        # Convert hand result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeBody:
    def __init__(self, static_image_mode=True):
        super(MediaPipeBody, self).__init__()

        # Access MediaPipe Solutions Python API
        mp_body = mp.solutions.pose

        # Initialize MediaPipe Body
        # Video: static_image_mode=False
        # Picture: static_image_mode=True
        self.body = mp_body.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=0.5)

        # Define body parameter
        self.param = {
                'keypt'   : np.zeros((25,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((25,3)), # 3D joint in relative coordinate
                'visible' : np.zeros(25), # Visibility
                'presence': np.zeros(25), # Presence (Note: Its always zero!)
                'detect'  : False, # Whether upper body is detected
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        # Convert mediapipe body result to my own param
        img_height, img_width, _ = img.shape

        if result.pose_landmarks is None:
            self.param['detect'] = False
        else:
            self.param['detect'] = True

            # Loop through 25 landmark of upper body
            for j, lm in enumerate(result.pose_landmarks.landmark):
                self.param['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param['joint'][j,0] = lm.x
                self.param['joint'][j,1] = lm.y
                self.param['joint'][j,2] = lm.z # Note: z value should be discarded as the model is currently not fully trained to predict depth

                self.param['visible'][j] = lm.visibility
                self.param['presence'][j] = lm.presence

        return self.param


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract body result
        result = self.body.process(img)

        # Convert body result to my own param
        param = self.result_to_param(result, img)

        return param
