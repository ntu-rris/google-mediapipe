###############################################################################
### Wrapper for Google MediaPipe face, hand, body, holistic, object pose estimation
### and selfie segmentation
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


class MediaPipeFaceDetect:
    def __init__(self, model_selection=1, max_num_faces=5):
        # Access MediaPipe Solutions Python API
        mp_face_detect = mp.solutions.face_detection

        # Initialize MediaPipe FaceDetection
        # model_selection:
        #   An integer index 0 or 1
        #   Use 0 for short-range model that works best for faces within 2 meters from the camera
        #   Use 1 for a full-range model best for faces within 5 meters from the camera

        # min_detection_confidence:
        #   Confidence value [0,1] from face detection model
        #   for detection to be considered successful

        self.pipe = mp_face_detect.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5)

        # Define face parameter
        self.param = []
        for i in range(max_num_faces):
            p = {
                'detect': False,           # Boolean to indicate whether a face is detected
                'score' : 0,               # Probability of detection
                'bbox'  : (0,0,0,0),       # Bounding box (xmin,ymin,width,height) (pixel)
                'keypt' : np.zeros((6,2)), # 2D keypt in image coordinate (pixel)
                'fps'   : -1,              # Frame per sec
            }
            self.param.append(p)


    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['detect'] = False

        if result.detections is not None:
            # Loop through different faces
            for i, res in enumerate(result.detections):
                self.param[i]['detect'] = True
                self.param[i]['score']  = res.score[0]

                # Get bbox parameter
                xmin   = res.location_data.relative_bounding_box.xmin  * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                ymin   = res.location_data.relative_bounding_box.ymin  * img_height # Convert normalized coor to pixel [0,1] -> [0,height]
                width  = res.location_data.relative_bounding_box.width * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                height = res.location_data.relative_bounding_box.height* img_height # Convert normalized coor to pixel [0,1] -> [0,height]
                self.param[i]['bbox'] = (xmin, ymin, width, height)

                # Loop through 6 landmarks for each face
                # Right eye, left eye, nose, mouth, right ear, left ear
                for j, lm in enumerate(res.location_data.relative_keypoints):
                    self.param[i]['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

        return self.param


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeFace:
    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=True):
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

        # refine_landmarks
        #   Set to True to further refine the landmark coordinates around the eyes and lips
        #   and output additional landmarks around the irises. Default to false.

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
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define face parameter
        self.param = []
        for i in range(max_num_faces):
            num = 478 if refine_landmarks else 468
            p = {
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((num,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((num,3)), # 3D joint in relative coordinate
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
    def __init__(self, static_image_mode=True, max_num_hands=1,
        model_complexity=1, intrin=None):
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

        # model_complexity:
        #   Complexity of the hand landmark model: 0 or 1.
        #   Landmark accuracy as well as inference latency generally
        #   go up with the model complexity. Default to 1.
        
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
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define hand parameter
        self.param = []
        for i in range(max_num_hands):
            p = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'rvec'    : np.zeros(3),      # Global rotation vector Note: this term is only used for solvepnp initialization
                'tvec'    : np.asarray([0,0,0.6]), # Global translation vector (m) Note: Init z direc to some +ve dist (i.e. in front of camera), to prevent solvepnp from wrongly estimating z as -ve
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

                    # Ignore it https://github.com/google/mediapipe/issues/1320
                    # self.param[i]['visible'][j] = lm.visibility
                    # self.param[i]['presence'][j] = lm.presence

        if result.multi_hand_world_landmarks is not None:
            for i, res in enumerate(result.multi_hand_world_landmarks):
                if i>self.max_num_hands-1: break # Note: Need to check if exceed max number of hand
                # Loop through 21 landmark for each hand
                for j, lm in enumerate(res.landmark):
                    self.param[i]['joint'][j,0] = lm.x
                    self.param[i]['joint'][j,1] = lm.y
                    self.param[i]['joint'][j,2] = lm.z

                # Convert relative 3D joint to angle
                self.param[i]['angle'] = self.convert_joint_to_angle(self.param[i]['joint'])
                # Convert relative 3D joint to camera coordinate
                self.convert_joint_to_camera_coor(self.param[i], self.intrin)

        return self.param

    
    def convert_joint_to_angle(self, joint):
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


    def convert_joint_to_camera_coor(self, param, intrin, use_solvepnp=True):
        # MediaPipe version 0.8.9.1 onwards:
        # Given real-world 3D joint centered at middle MCP joint -> J_origin
        # To estimate the 3D joint in camera coordinate          -> J_camera = J_origin + tvec,
        # We need to find the unknown translation vector         -> tvec = [tx,ty,tz]
        # Such that when J_camera is projected to the 2D image plane
        # It matches the 2D keypoint locations

        # Considering all 21 keypoints,
        # Each keypoints will form 2 eq, in total we have 42 eq 3 unknowns
        # Since the equations are linear wrt [tx,ty,tz]
        # We can solve the unknowns using linear algebra A.x = b, where x = [tx,ty,tz]

        # Consider a single keypoint (pixel x) and joint (X,Y,Z)
        # Using the perspective projection eq:
        # (x - cx)/fx = (X + tx) / (Z + tz)
        # Similarly for pixel y:
        # (y - cy)/fy = (Y + ty) / (Z + tz)
        # Rearranging the above linear equations by keeping constants to the right hand side:
        # fx.tx - (x - cx).tz = -fx.X + (x - cx).Z
        # fy.ty - (y - cy).tz = -fy.Y + (y - cy).Z
        # Therefore, we can factor out the unknowns and form a matrix eq:
        # [fx  0 (x - cx)][tx]   [-fx.X + (x - cx).Z]
        # [ 0 fy (y - cy)][ty] = [-fy.Y + (y - cy).Z]
        #                 [tz]

        idx = [i for i in range(21)] # Use all landmarks

        if use_solvepnp:
            # Method 1: OpenCV solvePnP
            fx, fy = intrin['fx'], intrin['fy']
            cx, cy = intrin['cx'], intrin['cy']
            intrin_mat = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
            dist_coeff = np.zeros(4)

            ret, param['rvec'], param['tvec'] = cv2.solvePnP(
                param['joint'][idx], param['keypt'][idx],
                intrin_mat, dist_coeff, param['rvec'], param['tvec'],
                useExtrinsicGuess=True)
            # Add tvec to all joints
            param['joint'] += param['tvec']

        else:
            # Method 2:
            A = np.zeros((len(idx),2,3))
            b = np.zeros((len(idx),2))

            A[:,0,0] = intrin['fx']
            A[:,1,1] = intrin['fy']
            A[:,0,2] = -(param['keypt'][idx,0] - intrin['cx'])
            A[:,1,2] = -(param['keypt'][idx,1] - intrin['cy'])

            b[:,0] = -intrin['fx'] * param['joint'][idx,0] \
                     + (param['keypt'][idx,0] - intrin['cx']) * param['joint'][idx,2]
            b[:,1] = -intrin['fy'] * param['joint'][idx,1] \
                     + (param['keypt'][idx,1] - intrin['cy']) * param['joint'][idx,2]

            A = A.reshape(-1,3) # [8,3]
            b = b.flatten() # [8]

            # Use the normal equation AT.A.x = AT.b to minimize the sum of the sq diff btw left and right sides
            x = np.linalg.solve(A.T @ A, A.T @ b)
            # Add tvec to all joints
            param['joint'] += x


    def convert_joint_to_camera_coor_(self, param, intrin):
        # Note: With verion 0.8.9.1 this function is obsolete, refer to the above function
        # as the joint is already in real-world 3D coor with origin at middle finger MCP

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
        param['joint'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
        param['joint'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
        param['joint'][:,2] = param['joint'][:,2]*intrin['width']

        # Assume average depth is fixed at 0.6 m (works best when the hand is around 0.5 to 0.7 m from camera)
        Zavg = 0.6
        # Average focal length of fx and fy
        favg = (intrin['fx']+intrin['fy'])*0.5
        # Compute scaling factor S
        S = favg/Zavg
        # Uniform scaling
        param['joint'] /= S

        # Estimate wrist depth using similar triangle
        D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
        # Dist btw wrist and index finger MCP keypt (in 2D image coor)
        d = np.linalg.norm(param['keypt'][0] - param['keypt'][9])
        # d/f = D/Z -> Z = D/d*f
        Zwrist = D/d*favg
        # Add wrist depth to all joints
        param['joint'][:,2] += Zwrist


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeBody:
    def __init__(self, static_image_mode=True, model_complexity=1,
        enable_segmentation=True, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        self.enable_segmentation = enable_segmentation

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
        #   If set to True, filters pose landmarks across different input images
        #   to reduce jitter, but ignored if static_image_mode is also set to True

        # enable_segmentation
        #   Set to True to generate segmentation mask. Default to false.

        # smooth_segmentation
        #   Set to True to filter segmentation masks across different input images
        #   to reduce jitter. Ignored if enable_segmentation is false or static_image_mode is True.
        #   Default to True.

        # min_detection_confidence:
        #   Confidence value [0,1] from person detection model
        #   for detection to be considered successful
        
        # min_tracking_confidence:
        #   Minimum confidence value [0,1] from landmark-tracking model
        #   for pose landmarks to be considered tracked successfully, 
        #   or otherwise person detection will be invoked automatically on the next input image.
        #   Setting it to a higher value can increase robustness of the solution, 
        #   at the expense of a higher latency. 
        #   Ignored if static_image_mode is True, where person detection simply runs on every image.

        self.pipe = mp_body.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define body parameter
        self.param = {
                'detect'  : False, # Boolean to indicate whether a person is detected
                'keypt'   : np.zeros((33,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((33,3)), # 3D joint in real world coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'rvec'    : np.zeros(3),           # Global rotation vector Note: this term is only used for solvepnp initialization
                'tvec'    : np.asarray([0,0,1.0]), # Global translation vector (m) Note: Init z direc to some +ve dist (i.e. in front of camera), to prevent solvepnp from wrongly estimating z as -ve
                'mask'    : None, # Binary segmentation mask 0:Backgnd, 1:Human
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
                # Note: lm.z is ignored refer to pose_world_landmarks for 3D coordinated

                self.param['visible'][j] = lm.visibility

            # Loop through estimated real world 3D coordinates in meters origin at center between hips
            for j, lm in enumerate(result.pose_world_landmarks.landmark):
                self.param['joint'][j,0] = lm.x
                self.param['joint'][j,1] = lm.y
                self.param['joint'][j,2] = lm.z

            # Segmentation mask
            if self.enable_segmentation:
                self.param['mask'] = result.segmentation_mask

            # Convert 3D joint to camera coordinate
            self.convert_body_joint_to_camera_coor(self.param, self.intrin)

        return self.param


    def convert_body_joint_to_camera_coor(self, param, intrin, scale_body=False, use_solvepnp=True):
        # MediaPipe version 0.8.6 onwards:
        # Given real-world 3D joint centered at hip joint -> J_origin
        # To estimate the 3D joint in camera coordinate   -> J_camera = J_origin + tvec,
        # We need to find the unknown translation vector  -> tvec = [tx,ty,tz]
        # Such that when J_camera is projected to the 2D image plane
        # It matches the 2D keypoint locations

        # Let's choose 4 keypoints (left/right shoulder/hip)
        # Since they are more stable as compared to the limbs
        # Each keypoints will form 2 eq, in total we have 8 eq 3 unknowns
        # Since the equations are linear wrt [tx,ty,tz]
        # We can solve the unknowns using linear algebra A.x = b, where x = [tx,ty,tz]

        # Consider a single keypoint (pixel x) and joint (X,Y,Z)
        # Using the perspective projection eq:
        # (x - cx)/fx = (X + tx) / (Z + tz)
        # Similarly for pixel y:
        # (y - cy)/fy = (Y + ty) / (Z + tz)
        # Rearranging the above linear equations by keeping constants to the right hand side:
        # fx.tx - (x - cx).tz = -fx.X + (x - cx).Z
        # fy.ty - (y - cy).tz = -fy.Y + (y - cy).Z
        # Therefore, we can factor out the unknowns and form a matrix eq:
        # [fx  0 (x - cx)][tx]   [-fx.X + (x - cx).Z]
        # [ 0 fy (y - cy)][ty] = [-fy.Y + (y - cy).Z]
        #                 [tz]

        # Scale body joint
        if scale_body:
            self.scale_body_joint(param)

        # idx = [11,12,23,24] # Index of left/right shoulder/hip
        idx = [i for i in range(33)] # Use all landmarks

        if use_solvepnp:
            # Method 1: OpenCV solvePnP
            fx, fy = intrin['fx'], intrin['fy']
            cx, cy = intrin['cx'], intrin['cy']
            intrin_mat = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
            dist_coeff = np.zeros(4)

            ret, param['rvec'], param['tvec'] = cv2.solvePnP(
                param['joint'][idx], param['keypt'][idx],
                intrin_mat, dist_coeff, param['rvec'], param['tvec'],
                useExtrinsicGuess=True)
            # Add tvec to all joints
            param['joint'] += param['tvec']

        else:
            # Method 2:
            A = np.zeros((len(idx),2,3))
            b = np.zeros((len(idx),2))

            A[:,0,0] = intrin['fx']
            A[:,1,1] = intrin['fy']
            A[:,0,2] = -(param['keypt'][idx,0] - intrin['cx'])
            A[:,1,2] = -(param['keypt'][idx,1] - intrin['cy'])

            b[:,0] = -intrin['fx'] * param['joint'][idx,0] \
                     + (param['keypt'][idx,0] - intrin['cx']) * param['joint'][idx,2]
            b[:,1] = -intrin['fy'] * param['joint'][idx,1] \
                     + (param['keypt'][idx,1] - intrin['cy']) * param['joint'][idx,2]

            A = A.reshape(-1,3) # [8,3]
            b = b.flatten() # [8]

            # Use the normal equation AT.A.x = AT.b to minimize the sum of the sq diff btw left and right sides
            x = np.linalg.solve(A.T @ A, A.T @ b)
            # Add tvec to all joints
            param['joint'] += x


    def scale_body_joint(self, param):
        # TODO: Try normalize body joint using spherical coor
        # Something like Pose Locality Constrained Representation for 3D Human Pose Reconstruction
        # Section onPose Data Normalization
        # https://cse.sc.edu/~fan23/doc/plcr_eccv2014.pdf
        # Convert cartesian to spherical
        # Convert spherical to cartesian

        # Desired body dimension in meter
        # Adapted from https://www.researchgate.net/figure/Dimensions-of-average-male-human-being-23_fig1_283532449
        hip     = 0.14 # Width
        trunk   = 0.51 # Length from hip to shoulder
        arm     = 0.30 # Length
        forearm = 0.27 # Length
        thigh   = 0.46 # Length
        leg     = 0.45 # Length

        # Compute estimated body dimension
        hip_        = np.linalg.norm(param['joint'][23] - param['joint'][24])
        trunk_left  = np.linalg.norm(param['joint'][11] - param['joint'][23])
        trunk_right = np.linalg.norm(param['joint'][12] - param['joint'][24])

        arm_left     = np.linalg.norm(param['joint'][11] - param['joint'][13])
        forearm_left = np.linalg.norm(param['joint'][13] - param['joint'][15])
        thigh_left   = np.linalg.norm(param['joint'][23] - param['joint'][25])
        leg_left     = np.linalg.norm(param['joint'][25] - param['joint'][27])

        arm_right     = np.linalg.norm(param['joint'][12] - param['joint'][14])
        forearm_right = np.linalg.norm(param['joint'][14] - param['joint'][16])
        thigh_right   = np.linalg.norm(param['joint'][24] - param['joint'][26])
        leg_right     = np.linalg.norm(param['joint'][26] - param['joint'][28])

        # Perform scaling of estimated body joint to match desired dimension
        joint = np.zeros_like(param['joint']) # Temp storage for joint
        joint[[23,24]] = param['joint'][[23,24]] * hip / hip_
        joint[11] = param['joint'][11] * trunk / trunk_left
        joint[12] = param['joint'][12] * trunk / trunk_right

        def scale_limb(param, idx1, idx2, len1, len2):
            j1 = param['joint'][idx1]
            j2 = param['joint'][idx2]
            return (j2 - j1) * len1 / len2

        joint[13] = joint[11] + scale_limb(param, 11, 13, arm, arm_left)
        joint[14] = joint[12] + scale_limb(param, 12, 14, arm, arm_right)
        joint[15] = joint[13] + scale_limb(param, 13, 15, forearm, forearm_left)
        joint[16] = joint[14] + scale_limb(param, 14, 16, forearm, forearm_right)

        joint[25] = joint[23] + scale_limb(param, 23, 25, thigh, thigh_left)
        joint[26] = joint[24] + scale_limb(param, 24, 26, thigh, thigh_right)
        joint[27] = joint[25] + scale_limb(param, 25, 27, leg, leg_left)
        joint[28] = joint[26] + scale_limb(param, 26, 28, leg, leg_right)

        # Ignore scaling of hand and feet joint
        # simply translate to new wrist and ankle joint
        joint[[17,19,21]] = param['joint'][[17,19,21]] - param['joint'][15] + joint[15]
        joint[[18,20,22]] = param['joint'][[18,20,22]] - param['joint'][16] + joint[16]
        joint[[29,31]] = param['joint'][[29,31]] - param['joint'][27] + joint[27]
        joint[[30,32]] = param['joint'][[30,32]] - param['joint'][28] + joint[28]

        param['joint'][11:] = joint[11:] # Note: Ignore scaling of face joints


    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


class MediaPipeHolistic:
    def __init__(self, static_image_mode=True, model_complexity=1,
        enable_segmentation=True, refine_face_landmarks=True, intrin=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        self.enable_segmentation = enable_segmentation

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

        # enable_segmentation
        #   Set to True to generate segmentation mask. Default to false.

        # smooth_segmentation
        #   Set to True to filter segmentation masks across different input images
        #   to reduce jitter. Ignored if enable_segmentation is false or static_image_mode is True.
        #   Default to True.

        # refine_face_landmarks
        #   Set to True to further refine the landmark coordinates around the eyes and lips
        #   and output additional landmarks around the irises. Default to false.

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
            enable_segmentation=enable_segmentation,
            smooth_segmentation=True,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define face parameter
        num = 478 if refine_face_landmarks else 468
        self.param_fc = {
                'detect'  : False, # Boolean to indicate whether a face is detected
                'keypt'   : np.zeros((num,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((num,3)), # 3D joint in camera coordinate (m)
                'fps'     : -1, # Frame per sec
            }

        # Define left and right hand parameter
        self.param_lh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in camera coordinate (m)
                'class'   : None,             # Left / right / none hand
                'score'   : 0,                # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle'   : np.zeros(15),     # Flexion joint angles in degree
                'gesture' : None,             # Type of hand gesture
                'fps'     : -1, # Frame per sec
            }
        self.param_rh = {
                'keypt'   : np.zeros((21,2)), # 2D keypt in image coordinate (pixel)
                'joint'   : np.zeros((21,3)), # 3D joint in camera coordinate (m)
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
                'joint'   : np.zeros((33,3)), # 3D joint in real world coordinate (m)
                'visible' : np.zeros(33),     # Visibility: Likelihood [0,1] of being visible (present and not occluded) in the image
                'rvec'    : np.zeros(3),           # Global rotation vector
                'tvec'    : np.asarray([0,0,1.0]), # Global translation vector (m)
                'mask'    : None, # Binary segmentation mask 0:Backgnd, 1:Human
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

            # Scale face vert to meter
            D = 0.07 # Note: Hardcode actual dist btw left and right eye as 0.07 m
            d = np.linalg.norm(self.param_fc['joint'][386] - self.param_fc['joint'][159]) # Dist btw left and right eye (in relative 3D coor)
            self.param_fc['joint'] *= D/d

            # Translate face nose joint to origin
            self.param_fc['joint'] -= self.param_fc['joint'][4] # Nose joint

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

            # Convert relative 3D joint to angle
            self.param_lh['angle'] = self.convert_joint_to_angle(self.param_lh['joint'])
            # Convert relative 3D joint to camera coordinate
            self.convert_joint_to_camera_coor(self.param_lh, self.intrin)

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

            # Convert relative 3D joint to angle
            self.param_rh['angle'] = self.convert_joint_to_angle(self.param_rh['joint'])
            # Convert relative 3D joint to camera coordinate
            self.convert_joint_to_camera_coor(self.param_rh, self.intrin)

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

                self.param_bd['visible'][j] = lm.visibility

            # Loop through estimated real world 3D coordinates in meters origin at center between hips
            for j, lm in enumerate(result.pose_world_landmarks.landmark):
                self.param_bd['joint'][j,0] = lm.x
                self.param_bd['joint'][j,1] = lm.y
                self.param_bd['joint'][j,2] = lm.z

            # Segmentation mask
            if self.enable_segmentation:
                self.param_bd['mask'] = result.segmentation_mask

            # Convert 3D joint to camera coordinate
            self.convert_body_joint_to_camera_coor(self.param_bd, self.intrin)

            # Translate to face nose joint to body nose joint
            self.param_fc['joint'] += self.param_bd['joint'][0] # Nose joint

            # Translate to hand wrist to body wrist joint
            self.param_lh['joint'] += self.param_bd['joint'][15] # Left wrist joint
            self.param_rh['joint'] += self.param_bd['joint'][16] # Right wrist joint


        return (self.param_fc, self.param_lh, self.param_rh, self.param_bd)

    
    def convert_joint_to_angle(self, joint):
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


    def convert_joint_to_camera_coor(self, param, intrin):
        # De-normalized 3D hand joint
        param['joint'][:,0] = param['joint'][:,0]*intrin['width']
        param['joint'][:,1] = param['joint'][:,1]*intrin['height']
        param['joint'][:,2] = param['joint'][:,2]*intrin['width']

        # Scale hand joint to meter
        D = 0.08 # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
        d = np.linalg.norm(param['joint'][0] - param['joint'][9]) # Dist btw wrist and index finger MCP joint
        param['joint'] *= D/d

        # Translate wrist to origin
        param['joint'] -= param['joint'][0]


    def convert_body_joint_to_camera_coor(self, param, intrin):
        idx = [i for i in range(33)] # Use all landmarks

        fx, fy = intrin['fx'], intrin['fy']
        cx, cy = intrin['cx'], intrin['cy']
        intrin_mat = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
        dist_coeff = np.zeros(4)

        ret, param['rvec'], param['tvec'] = cv2.solvePnP(
            param['joint'][idx], param['keypt'][idx],
            intrin_mat, dist_coeff, param['rvec'], param['tvec'],
            useExtrinsicGuess=True)
        # Add tvec to all joints
        param['joint'] += param['tvec']


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


class MediaPipeSeg:
    def __init__(self, model_selection=0):
        # Access MediaPipe Solutions Python API
        mp_seg = mp.solutions.selfie_segmentation

        # Initialize MediaPipe Selfie Segmentation
        # model_selection:
        #   An integer index 0 or 1. 
        #   Use 0 to select the general model
        #   Use 1 to select the landscape model 
        
        self.pipe = mp_seg.SelfieSegmentation(
            model_selection=model_selection)


    def forward(self, img):
        # Preprocess image
        # img = cv2.flip(img, 1) # Flip image for 3rd person view
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        img.flags.writeable = False # To improve performance, optionally mark image as not writeable to pass by reference
        result = self.pipe.process(img)
        img.flags.writeable = True

        # Convert imput image back to original color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Extract segmentation mask
        msk = result.segmentation_mask # [h,w] float32 range from 0 to 1
        msk = cv2.bilateralFilter(msk, 9, 75, 75) # Apply bilateral filter to smooth the low res mask
        msk = np.stack((msk,) * 3, axis=-1) > 0.1 # Convert those pixel>thres to boolean

        # Create background image
        bg_img = cv2.GaussianBlur(img,(55,55),0) # Blurred input image
        # bg_img = np.zeros(img.shape, dtype=np.uint8) # Black background
        
        # Overlay segmentated img on background image
        out = np.where(msk, img, bg_img)

        return out
