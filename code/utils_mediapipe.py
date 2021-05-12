###############################################################################
### Wrapper for Google MediaPipe face, hand, body, holistic and object pose estimation
### https://github.com/google/mediapipe
###############################################################################

import cv2
import numpy as np
import mediapipe as mp


# Define default camera intrinsic
img_width  = 640
img_height = 480
intrin_default = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}


class MediaPipeFace:
    def __init__(self, static_image_mode=True, max_num_faces=1):
        # Access MediaPipe Solutions Python API
        mp_faces = mp.solutions.face_mesh

        # Initialize MediaPipe FaceMesh
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize face to reduce latency
        #   For unrelated images set to True: 
        #   To allow face detection to run on every input images
        
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
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((468,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((468,3)), # 3D joint in relative coordinate
                'fps'     : -1, # Frame per sec
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
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
                    self.param[i]['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

        return self.param


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeHand:
    def __init__(self, static_image_mode=True, max_num_hands=1, intrin=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Access MediaPipe Solutions Python API
        mp_hands = mp.solutions.hands
        # help(mp_hands.Hands)

        # Initialize MediaPipe Hands
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize hand to reduce latency
        #   For unrelated images set to True: 
        #   To allow hand detection to run on every input images
        
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
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
                # https://github.com/google/mediapipe/issues/1351
                # 'visible' : np.zeros(21), # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                # 'presence': np.zeros(21), # Presence: Likelihood [0,1] of being present in the image or if its located outside the image
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
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
                    self.param[i]['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

                    # Ignore it https://github.com/google/mediapipe/issues/1320
                    # self.param[i]['visible'][j] = lm.visibility
                    # self.param[i]['presence'][j] = lm.presence

                # Convert relative 3D joint to angle
                self.param[i]['angle'] = self.convert_3d_joint_to_angle(self.param[i]['joint'])
                # Convert relative 3D joint to actual 3D joint in camera coordinate
                self.convert_relative_to_actual_3d_joint(self.param[i], self.intrin)

        return self.param

    
    def convert_3d_joint_to_angle(self, joint):
        # Get direction vector of bone from parent to child
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3]
        # Normalize v
        v = v/np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        return np.degrees(angle) # Convert radian to degree


    def convert_relative_to_actual_3d_joint(self, param, intrin):
        # Note: MediaPipe hand model uses weak perspective (scaled orthographic) projection
        # https://github.com/google/mediapipe/issues/742#issuecomment-639104199

        # Weak perspective projection = (X,Y,Z) -> (X,Y) -> (SX, SY) = (x,y) in image coor
        # https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf (slide 35) 
        # Step 1) Orthographic projection = (X,Y,Z) -> (X,Y) discard Z depth
        # Step 2) Uniform scaling by a factor S = f/Zavg, (X,Y) -> (SX, SY)
        # Therefore, to backproject 2D -> 3D:
        # x = SX + cx -> X = (x - cx) / S
        # y = SY + cy -> Y = (y - cy) / S
        # z = SZ      -> Z = z / S

        # Note: Output of mediapipe 3D hand joint X' and Y' are normalized to [0,1]
        # Need to convert normalized 3D (X',Y') to 2D image coor (x,y)
        # x = X' * img_width
        # y = Y' * img_height

        # Note: For scaling of mediapipe 3D hand joint Z'
        # Since it is mentioned in mcclanahoochie's comment to the above github issue
        # 'z is scaled proportionally along with x and y (via weak projection), and expressed in the same units as x & y.'
        # And also in the paper for MediaPipe face: 2019 Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs
        # '3D positions are re-scaled so that a fixed aspect ratio is maintained between the span of x-coor and the span of z-coor'
        # Therefore, I think that Z' is scaled similar to X'
        # z = Z' * img_width
        # z = SZ -> Z = z/S

        # Note: For full-body pose the magnitude of z uses roughly the same scale as x
        # https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
        
        # De-normalized 3D hand joint
        param['joint_3d'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
        param['joint_3d'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
        param['joint_3d'][:,2] = param['joint'][:,2]*intrin['width']

        # Assume average depth is fixed at 0.6 m (works best when the hand is around 0.5 to 0.7 m from camera)
        Zavg = 0.6
        # Average focal length of fx and fy
        favg = (intrin['fx']+intrin['fy'])*0.5
        # Compute scaling factor S
        S = favg/Zavg
        # Uniform scaling
        param['joint_3d'] /= S

        # Estimate wrist depth using similar triangle
        D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
        # Dist btw wrist and index finger MCP keypt (in 2D image coor)
        d = np.linalg.norm(param['keypt'][0] - param['keypt'][9])
        # d/f = D/Z -> Z = D/d*f
        Zwrist = D/d*favg
        # Add wrist depth to all joints
        param['joint_3d'][:,2] += Zwrist      


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeBody:
    def __init__(self, static_image_mode=True, model_complexity=1, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Access MediaPipe Solutions Python API
        mp_body = mp.solutions.pose

        # Initialize MediaPipe Body
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize a person to reduce latency
        #   For unrelated images set to True: 
        #   To allow detection of the most prominent person to run on every input images
        
        # model_complexity:
        #   Complexity of the pose landmark model: 0, 1 or 2. 
        #   Landmark accuracy as well as inference latency generally 
        #   go up with the model complexity. Default to 1.
        
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
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define body parameter
        self.param = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((33,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((33,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((33,3)), # 3D joint in camera coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
        img_height, img_width, _ = img.shape

        if result.pose_landmarks is None:
            self.param['detect'] = False
        else:
            self.param['detect'] = True

            # Loop through landmark of body
            for j, lm in enumerate(result.pose_landmarks.landmark):
                self.param['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param['joint'][j,0] = lm.x
                self.param['joint'][j,1] = lm.y
                self.param['joint'][j,2] = lm.z

                self.param['visible'][j] = lm.visibility

            # Convert relative 3D joint to actual 3D joint in m
            self.convert_relative_to_actual_3d_joint(self.param, self.intrin)

        return self.param


    def convert_relative_to_actual_3d_joint(self, param, intrin):
        # De-normalized 3D body joint
        param['joint_3d'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
        param['joint_3d'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
        param['joint_3d'][:,2] = param['joint'][:,2]*intrin['width'] * 0.25
        # Note: Seems like need to further scale down z by 0.25 else will get elongated forearm and feet
        # Could it be beacuse z ranges btw a smaller range of -ve XX to +ve XX unlike x and y which range from 0 to 1

        # Compute center of shoulder and hip joint
        center_shoulder = (param['joint_3d'][11] + param['joint_3d'][12])*0.5
        center_hip      = (param['joint_3d'][23] + param['joint_3d'][24])*0.5

        # Translate to new origin at center of hip
        param['joint_3d'] -= center_hip

        # Scale from relative to actual 3D joint in m
        D = 0.55 # Note: Hardcode actual dist btw shoulder and hip joint as 0.55 m
        d = np.linalg.norm(center_shoulder - center_hip) # Dist btw shoulder and hip joint (in relative 3D coor)
        param['joint_3d'] *= D/d

        # Note: Unlike hand where the Zavg is relatively constant around 0.6 m, 
        # it is quite hard to define Zavg for body, 
        # thus the step to convert to camera coor is ignored


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeHolistic:
    def __init__(self, static_image_mode=True, model_complexity=1, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Access MediaPipe Solutions Python API
        mp_holisitic = mp.solutions.holistic

        # Initialize MediaPipe Holistic
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize a person to reduce latency
        #   For unrelated images set to True: 
        #   To allow detection of the most prominent person to run on every input images
        
        # model_complexity:
        #   Complexity of the pose landmark model: 0, 1 or 2. 
        #   Landmark accuracy as well as inference latency generally 
        #   go up with the model complexity. Default to 1.
        
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
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define face parameter
        self.param_fc = {
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((468,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((468,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((468,3)), # 3D joint in camera coordinate (m)
                'fps'     : -1, # Frame per sec
            }

        # Define left and right hand parameter
        self.param_lh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }
        self.param_rh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }

        # Define body parameter
        self.param_bd = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((33,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((33,3)), # 3D joint in relative coordinate
                'joint_3d': np.zeros((33,3)), # 3D joint in camera coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'fps'     : -1, # Frame per sec
            }


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
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
                self.param_fc['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
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
                self.param_lh['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_lh['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_lh['joint'][j,0] = lm.x
                self.param_lh['joint'][j,1] = lm.y
                self.param_lh['joint'][j,2] = lm.z

                # Ignore it https://github.com/google/mediapipe/issues/1320
                # self.param_lh['visible'][j] = lm.visibility
                # self.param_lh['presence'][j] = lm.presence

            # Convert relative 3D joint to angle
            self.param_lh['angle'] = self.convert_3d_joint_to_angle(self.param_lh['joint'])

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
                self.param_rh['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_rh['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_rh['joint'][j,0] = lm.x
                self.param_rh['joint'][j,1] = lm.y
                self.param_rh['joint'][j,2] = lm.z

                # Ignore it https://github.com/google/mediapipe/issues/1320
                # self.param_rh['visible'][j] = lm.visibility
                # self.param_rh['presence'][j] = lm.presence

            # Convert relative 3D joint to angle
            self.param_rh['angle'] = self.convert_3d_joint_to_angle(self.param_rh['joint'])

        ############
        ### Pose ###
        ############
        if result.pose_landmarks is None:
            self.param_bd['detect'] = False
        else:
            self.param_bd['detect'] = True

            # Loop through landmark of body
            for j, lm in enumerate(result.pose_landmarks.landmark):
                self.param_bd['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                self.param_bd['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                self.param_bd['joint'][j,0] = lm.x
                self.param_bd['joint'][j,1] = lm.y
                self.param_bd['joint'][j,2] = lm.z

                self.param_bd['visible'][j] = lm.visibility

            # Convert relative 3D joint to actual 3D joint in camera coordinate
            self.convert_relative_to_actual_3d_joint(
                self.param_fc, self.param_lh, self.param_rh, self.param_bd, self.intrin)

        return (self.param_fc, self.param_lh, self.param_rh, self.param_bd)

    
    def convert_3d_joint_to_angle(self, joint):
        # Get direction vector of bone from parent to child
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3]
        # Normalize v
        v = v/np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        return np.degrees(angle) # Convert radian to degree


    def convert_relative_to_actual_3d_joint(self, param_fc, param_lh, param_rh, param_bd, intrin):
        if param_bd['detect']:
            # De-normalized 3D body joint
            param_bd['joint_3d'][:,0] = param_bd['joint'][:,0]*intrin['width'] -intrin['cx']
            param_bd['joint_3d'][:,1] = param_bd['joint'][:,1]*intrin['height']-intrin['cy']
            param_bd['joint_3d'][:,2] = param_bd['joint'][:,2]*intrin['width'] * 0.25
            # Note: Seems like need to further scale down z by 0.25 else will get elongated forearm and feet
            # Could it be beacuse z ranges btw a smaller range of -ve XX to +ve XX unlike x and y which range from 0 to 1

            # Compute center of shoulder and hip joint
            center_shoulder = (param_bd['joint_3d'][11] + param_bd['joint_3d'][12])*0.5
            center_hip      = (param_bd['joint_3d'][23] + param_bd['joint_3d'][24])*0.5

            # Translate to new origin at center of hip
            param_bd['joint_3d'] -= center_hip

            # Scale from relative to actual 3D joint in m
            D = 0.55 # Note: Hardcode actual dist btw shoulder and hip joint as 0.55 m
            d = np.linalg.norm(center_shoulder - center_hip) # Dist btw shoulder and hip joint (in relative 3D coor)
            param_bd['joint_3d'] *= D/d

        if param_fc['detect']:
            param_fc['joint_3d'] = param_fc['joint'].copy()
            
            # Scale from relative to actual 3D joint in m
            D = 0.07 # Note: Hardcode actual dist btw left and right eye as 0.07 m
            d = np.linalg.norm(param_fc['joint_3d'][386] - param_fc['joint_3d'][159]) # Dist btw left and right eye (in relative 3D coor)
            param_fc['joint_3d'] *= D/d

            # Translate to face nose joint then add body nose joint
            param_fc['joint_3d'] += -param_fc['joint_3d'][4] + param_bd['joint_3d'][0] # Nose joint             

        if param_lh['class'] is not None:
            # De-normalized 3D hand joint
            param_lh['joint_3d'][:,0] = param_lh['joint'][:,0]*intrin['width'] -intrin['cx']
            param_lh['joint_3d'][:,1] = param_lh['joint'][:,1]*intrin['height']-intrin['cy']
            param_lh['joint_3d'][:,2] = param_lh['joint'][:,2]*intrin['width']

            # Scale from relative to actual 3D joint in m
            D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
            d = np.linalg.norm(param_lh['joint_3d'][0] - param_lh['joint_3d'][9]) # Dist btw wrist and index finger MCP joint
            param_lh['joint_3d'] *= D/d

            # Translate to original hand wrist then add body wrist joint
            param_lh['joint_3d'] += -param_lh['joint_3d'][0] + param_bd['joint_3d'][15] # Left wrist joint

        if param_rh['class'] is not None:
            # De-normalized 3D hand joint
            param_rh['joint_3d'][:,0] = param_rh['joint'][:,0]*intrin['width'] -intrin['cx']
            param_rh['joint_3d'][:,1] = param_rh['joint'][:,1]*intrin['height']-intrin['cy']
            param_rh['joint_3d'][:,2] = param_rh['joint'][:,2]*intrin['width']
            
            # Scale from relative to actual 3D joint in m
            D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
            d = np.linalg.norm(param_rh['joint_3d'][0] - param_rh['joint_3d'][9]) # Dist btw wrist and index finger MCP joint
            param_rh['joint_3d'] *= D/d

            # Translate to original hand wrist then add body wrist joint
            param_rh['joint_3d'] += -param_rh['joint_3d'][0] + param_bd['joint_3d'][16] # Right wrist joint


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeObjectron:
    def __init__(self, static_image_mode=True, max_num_objects=5, model_name='Shoe', intrin=None):
        self.max_num_objects = max_num_objects

        # Access MediaPipe Solutions Python API
        mp_obj = mp.solutions.objectron

        # Initialize MediaPipe Objectron
        # static_image_mode:
        #   For video processing set to False: 
        #   Will use previous frame to localize face to reduce latency
        #   For unrelated images set to True: 
        #   To allow object detection to run on every input images
        
        # max_num_objects:
        #   Maximum number of objects to detect
        
        # min_detection_confidence:
        #   Confidence value [0,1] from face detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for the 3D bounding box landmarks to be considered tracked successfully, 
        #   or otherwise object detection will be invoked automatically on the next input image. 
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is true, where object detection simply runs on every image

        # model_name:
        #   Currently supports the below 4 classes:
        #   Shoe / Chair / Cup / Camera

        if intrin is None:
            self.pipe = mp_obj.Objectron(
                static_image_mode=static_image_mode,
                max_num_objects=max_num_objects,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.99,
                model_name=model_name.capitalize())
        else:
            self.pipe = mp_obj.Objectron(
                static_image_mode=static_image_mode,
                max_num_objects=max_num_objects,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.99,
                principal_point=(intrin['cx'],intrin['cy']),
                focal_length=(intrin['fx'],intrin['fy']),
                image_size=(intrin['width'],intrin['height']),
                model_name=model_name.capitalize())

        # Define face parameter
        self.param = []
        for i in range(max_num_objects):
            p = {
                'detect'      : False,           # Boolean to indicate whether a face is detected
                'landmarks_2d': np.zeros((9,2)), # 2D landmarks of object's 3D bounding box in image coordinate (pixel) Note: first value is center of bounding box, remaining 8 values are corners of bounding box
                'landmarks_3d': np.zeros((9,3)), # 3D landmarks of object's 3D bounding box in camera coordinate Note: first value is center of bounding box, remaining 8 values are corners of bounding box
                'rotation'    : np.eye(3),       # Rotation matrix from object coordinate frame to camera coordinate frame
                'translation' : np.zeros(3),     # Translation vector from object coordinate frame to camera coordinate frame
                'scale'       : np.zeros(3),     # Relative scale of the object along x, y and z directions
                'fps'         : -1,              # Frame per sec
            }
            self.param.append(p)

        # Change fr objectron to Open3D camera coor
        # Objectron camera coor:
        # +x pointing right, +y pointing up and +z pointing away from the scene
        # Open3D camera coor:
        # +x pointing right, +y pointing down and +z pointing to the scene
        # Thus, only need to reflect y and z axis        
        self.coc = np.eye(3) # Change of coor 3 by 3 matrix
        self.coc[1,1] = -1 # y axis
        self.coc[2,2] = -1 # z axis


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['detect'] = False

        if result.detected_objects is not None:
            # Loop through different objects
            for i, res in enumerate(result.detected_objects):
                self.param[i]['detect'] = True
                # Loop through 9 2D landmark for each object
                for j, lm in enumerate(res.landmarks_2d.landmark):
                    self.param[i]['landmarks_2d'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['landmarks_2d'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                # Loop through 9 3D landmark for each object
                for j, lm in enumerate(res.landmarks_3d.landmark):
                    self.param[i]['landmarks_3d'][j,0] = lm.x
                    self.param[i]['landmarks_3d'][j,1] = lm.y
                    self.param[i]['landmarks_3d'][j,2] = lm.z

                self.param[i]['scale']       = res.scale
                self.param[i]['rotation']    = res.rotation
                self.param[i]['translation'] = res.translation

                # # Just to check computation of landmarks_3d_ = rotation * scale * unit_box + translation
                # # Define a origin centered unit box with scale
                # d = 0.5
                # unit_box = np.array([ 
                #     [ 0, 0, 0], # 0
                #     [-d,-d,-d], # 1
                #     [-d,-d, d], # 2
                #     [-d, d,-d], # 3
                #     [-d, d, d], # 4
                #     [ d,-d,-d], # 5
                #     [ d,-d, d], # 6
                #     [ d, d,-d], # 7
                #     [ d, d, d]])# 8
                # unit_box *= self.param[i]['scale']
                # landmarks_3d = unit_box @ self.param[i]['rotation'].T + self.param[i]['translation']
                # print('Check computation of landmarks_3d', 
                #     np.allclose(landmarks_3d, self.param[i]['landmarks_3d'])) # Should get True

                # Change fr objectron to Open3D camera coor
                self.param[i]['landmarks_3d'] = self.param[i]['landmarks_3d'] @ self.coc.T
                self.param[i]['rotation']     = self.coc @ self.param[i]['rotation']
                self.param[i]['translation']  = self.coc @ self.param[i]['translation']

        return self.param


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param