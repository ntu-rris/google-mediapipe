###############################################################################
### Wrapper for Google MediaPipe face, hand, body and holistic pose estimation
### https://github.com/google/mediapipe
###############################################################################

import cv2
import numpy as np
import mediapipe as mp
from utils_joint_angle import convert_3d_joint_to_angle


class MediaPipeFace:
    def __init__(self, static_image_mode=True, max_num_faces=1):
        super(MediaPipeFace, self).__init__()

        # Access MediaPipe Solutions Python API
        mp_faces = mp.solutions.face_mesh

        # Initialize MediaPipe FaceMesh
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize face to reduce latency
        #   For unrelated images set to True: 
        #   To allow face detection to runs on every input images
        
        # max_num_faces:
        #   Maximum number of faces to detect
        
        # min_detection_confidence:
        #   Confidence value [0,1] from face detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for face landmarks to be considered tracked successfully, 
        #   or otherwise face detection will be invoked automatically on the next input image.
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is true, where face detection simply runs on every image.

        self.pipe = mp_faces.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define face parameter
        self.param = []
        for i in range(max_num_faces):
            p = {
                'detect': False, # Boolean to indicate whether a face is detected
                'keypt' : np.zeros((468,2)), # 2D keypt in image coordinate (pixel)
                'joint' : np.zeros((468,3)), # 3D joint in relative coordinate
                'fps'   : -1, # Frame per sec
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        # Convert mediapipe face result to my own param
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['detect'] = False

        if result.multi_face_landmarks is not None:
            # Loop through different faces
            for i, res in enumerate(result.multi_face_landmarks):
                self.param[i]['detect'] = True
                # Loop through 468 landmark for each face
                for j, lm in enumerate(res.landmark):
                    self.param[i]['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

        return self.param


    def forward(self, img):
        # Preprocess image
        # img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract face result
        result = self.pipe.process(img)

        # Convert face result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeHand:
    def __init__(self, static_image_mode=True, max_num_hands=1):
        super(MediaPipeHand, self).__init__()
        self.max_num_hands = max_num_hands

        # Access MediaPipe Solutions Python API
        mp_hands = mp.solutions.hands
        # help(mp_hands.Hands)

        # Initialize MediaPipe Hands
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize hand to reduce latency
        #   For unrelated images set to True: 
        #   To allow hand detection to runs on every input images
        
        # max_num_hands:
        #   Maximum number of hands to detect
        
        # min_detection_confidence:
        #   Confidence value [0,1] from hand detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for hand landmarks to be considered tracked successfully, 
        #   or otherwise hand detection will be invoked automatically on the next input image.
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is true, where hand detection simply runs on every image.

        self.pipe = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define hand parameter
        self.param = []
        for i in range(max_num_hands):
            p = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in absolute coordinate (m)
                'class'   : None, # Left / right hand
                'score'   : 0, # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15), # Joint angles
                'gesture' : None, # Type of hand gesture
                'fps'     : -1, # Frame per sec
                # https://github.com/google/mediapipe/issues/1351
                # 'visible' : np.zeros(21), # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                # 'presence': np.zeros(21), # Presence: Likelihood [0,1] of being present in the image or if its located outside the image
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
                if i>self.max_num_hands-1: break # Note: Need to check if exceed max number of hand
                self.param[i]['class'] = res.classification[0].label
                self.param[i]['score'] = res.classification[0].score

            # Loop through different hands
            for i, res in enumerate(result.multi_hand_landmarks):
                if i>self.max_num_hands-1: break # Note: Need to check if exceed max number of hand
                # Loop through 21 landmark for each hand
                for j, lm in enumerate(res.landmark):
                    self.param[i]['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

                    # Ignore it https://github.com/google/mediapipe/issues/1320
                    # self.param[i]['visible'][j] = lm.visibility
                    # self.param[i]['presence'][j] = lm.presence

                # Convert relative 3D joint to angle
                self.param[i]['angle'] = convert_3d_joint_to_angle(self.param[i]['joint'])

        return self.param


    def forward(self, img):
        # Preprocess image
        # img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract hand result
        result = self.pipe.process(img)

        # Convert hand result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeBody:
    def __init__(self, static_image_mode=True, upper_body_only=True):
        super(MediaPipeBody, self).__init__()

        # Access MediaPipe Solutions Python API
        mp_body = mp.solutions.pose

        # Initialize MediaPipe Body
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize a person to reduce latency
        #   For unrelated images set to True: 
        #   To allow detection of the most prominent person to runs on every input images
        
        # upper_body_only:
        #   If set to true, outputs only 25 upper-body pose landmarks
        #   Otherwise, outputs full set of 33 pose landmarks
        #   Note that upper-body-only prediction may be more accurate 
        #   for use cases where the lower-body parts are mostly out of view
        
        # smooth_landmarks:
        #   If set to true, filters pose landmarks across different input images
        #   to reduce jitter, but ignored if static_image_mode is also set to true

        # min_detection_confidence:
        #   Confidence value [0,1] from person detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for pose landmarks to be considered tracked successfully, 
        #   or otherwise person detection will be invoked automatically on the next input image.
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is true, where person detection simply runs on every image.

        self.pipe = mp_body.Pose(
            static_image_mode=static_image_mode,
            upper_body_only=upper_body_only,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        if upper_body_only:
            num_landmark = 25
        else:
            num_landmark = 33

        # Define body parameter
        self.param = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((num_landmark,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((num_landmark,3)), # 3D joint in relative coordinate
                'visible' : np.zeros(num_landmark), # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'presence': np.zeros(num_landmark), # Presence: Likelihood [0,1] of being present in the image or if its located outside the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        # Convert mediapipe body result to my own param
        img_height, img_width, _ = img.shape

        if result.pose_landmarks is None:
            self.param['detect'] = False
        else:
            self.param['detect'] = True

            # Loop through landmark of body
            for j, lm in enumerate(result.pose_landmarks.landmark):
                self.param['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param['joint'][j,0] = lm.x
                self.param['joint'][j,1] = lm.y
                # Note: z should be discarded as model is not fully trained to predict depth
                # but this is something on the roadmap
                self.param['joint'][j,2] = lm.z

                self.param['visible'][j] = lm.visibility
                self.param['presence'][j] = lm.presence

        return self.param


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract body result
        result = self.pipe.process(img)

        # Convert body result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeHolistic:
    def __init__(self, static_image_mode=True, upper_body_only=True):
        super(MediaPipeHolistic, self).__init__()

        # Access MediaPipe Solutions Python API
        mp_holisitic = mp.solutions.holistic

        # Initialize MediaPipe Holistic
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize a person to reduce latency
        #   For unrelated images set to True: 
        #   To allow detection of the most prominent person to runs on every input images
        
        # upper_body_only:
        #   If set to true, outputs only 25 upper-body pose landmarks (535 in total)
        #   Otherwise, outputs full set of 33 pose landmarks (543 in total)
        #   Note that upper-body-only prediction may be more accurate 
        #   for use cases where the lower-body parts are mostly out of view
        
        # smooth_landmarks:
        #   If set to true, filters pose landmarks across different input images
        #   to reduce jitter, but ignored if static_image_mode is also set to true

        # min_detection_confidence:
        #   Confidence value [0,1] from person detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for pose landmarks to be considered tracked successfully, 
        #   or otherwise person detection will be invoked automatically on the next input image.
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is true, where person detection simply runs on every image.

        self.pipe = mp_holisitic.Holistic(
            static_image_mode=static_image_mode,
            upper_body_only=upper_body_only,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define face parameter
        self.param_fc = {
                'detect': False, # Boolean to indicate whether a face is detected
                'keypt' : np.zeros((468,2)), # 2D keypt in image coordinate (pixel)
                'joint' : np.zeros((468,3)), # 3D joint in relative coordinate
                'fps'   : -1, # Frame per sec
            }

        # Define left and right hand parameter
        self.param_lh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in absolute coordinate (m)
                'class'   : None, # Left / none hand
                'score'   : 0, # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15), # Joint angles
                'gesture' : None, # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }
        self.param_rh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in absolute coordinate (m)
                'class'   : None, # None / right hand
                'score'   : 0, # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15), # Joint angles
                'gesture' : None, # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }

        if upper_body_only:
            num_landmark = 25
        else:
            num_landmark = 33            

        # Define body parameter
        self.param_bd = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((num_landmark,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((num_landmark,3)), # 3D joint in relative coordinate
                'visible' : np.zeros(num_landmark), # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'presence': np.zeros(num_landmark), # Presence: Likelihood [0,1] of being present in the image or if its located outside the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        # Convert mediapipe holistic result to my own param
        img_height, img_width, _ = img.shape

        ############
        ### Face ###
        ############
        if result.face_landmarks is None:
            self.param_fc['detect'] = False
        else:
            self.param_fc['detect'] = True

            # Loop through landmark of face
            for j, lm in enumerate(result.face_landmarks.landmark):
                self.param_fc['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_fc['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_fc['joint'][j,0] = lm.x
                self.param_fc['joint'][j,1] = lm.y
                self.param_fc['joint'][j,2] = lm.z

        #################
        ### Left Hand ###
        #################
        if result.left_hand_landmarks is None:
            # Reset hand param
            self.param_lh['class'] = None
        else:
            self.param_lh['class'] = 'left'

            # Loop through landmark of hands
            for j, lm in enumerate(result.left_hand_landmarks.landmark):
                self.param_lh['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_lh['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_lh['joint'][j,0] = lm.x
                self.param_lh['joint'][j,1] = lm.y
                self.param_lh['joint'][j,2] = lm.z

                # Ignore it https://github.com/google/mediapipe/issues/1320
                # self.param_lh['visible'][j] = lm.visibility
                # self.param_lh['presence'][j] = lm.presence

                # Convert relative 3D joint to angle
                self.param_lh['angle'] = convert_3d_joint_to_angle(self.param_lh['joint'])   

        ##################
        ### Right Hand ###
        ##################
        if result.right_hand_landmarks is None:
            # Reset hand param
            self.param_rh['class'] = None
        else:
            self.param_rh['class'] = 'right'

            # Loop through landmark of hands
            for j, lm in enumerate(result.right_hand_landmarks.landmark):
                self.param_rh['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_rh['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_rh['joint'][j,0] = lm.x
                self.param_rh['joint'][j,1] = lm.y
                self.param_rh['joint'][j,2] = lm.z

                # Ignore it https://github.com/google/mediapipe/issues/1320
                # self.param_rh['visible'][j] = lm.visibility
                # self.param_rh['presence'][j] = lm.presence

                # Convert relative 3D joint to angle
                self.param_rh['angle'] = convert_3d_joint_to_angle(self.param_lh['joint'])    

        ############
        ### Pose ###
        ############
        if result.pose_landmarks is None:
            self.param_bd['detect'] = False
        else:
            self.param_bd['detect'] = True

            # Loop through landmark of body
            for j, lm in enumerate(result.pose_landmarks.landmark):
                self.param_bd['keypt'][j,0] = lm.x * img_width # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_bd['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_bd['joint'][j,0] = lm.x
                self.param_bd['joint'][j,1] = lm.y
                # Note: z should be discarded as model is not fully trained to predict depth
                # but this is something on the roadmap
                self.param_bd['joint'][j,2] = lm.z

                self.param_bd['visible'][j] = lm.visibility
                self.param_bd['presence'][j] = lm.presence

        return (self.param_fc, self.param_lh, self.param_rh, self.param_bd)


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract holistic result
        result = self.pipe.process(img)

        # Convert holistic result to my own param
        param = self.result_to_param(result, img)

        return param
