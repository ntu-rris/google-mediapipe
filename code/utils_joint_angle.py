###############################################################################
### Useful function for converting 3D joint to joint angles
### Joint angle is then used for:
### 1) Gesture recognition 
### 2) Hand ROM recognition 
###############################################################################

import cv2
import numpy as np


def convert_3d_joint_to_angle(joint):
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


def convert_relative_to_actual_3d_joint(param, intrin):
    # Note: MediaPipe hand model uses weak perspective (scaled orthographic) projection
    # https://github.com/google/mediapipe/issues/742#issuecomment-639104199

    # Weak perspective projection = (X,Y,Z) -> (x,y) -> (Sx, Sy)
    # https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf (slide 35) 
    # Step 1) Orthographic projection = (X,Y,Z) -> (x,y)
    # Step 2) Uniform scaling by a factor S = f/Zavg, (x,y) -> (Sx, Sy)
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
    # Therefore, I think that z' is scaled similar to x'
    # z = Z' * img_width
    # z = SZ -> Z = z/S
    
    # De-normalized 3D hand joint
    param['joint_3d'][:,0] = param['joint'][:,0]*intrin['width'] -intrin['cx']
    param['joint_3d'][:,1] = param['joint'][:,1]*intrin['height']-intrin['cy']
    param['joint_3d'][:,2] = param['joint'][:,2]*intrin['width']
    # Assume average depth is fixed at 60 cm (works best when the hand is around 50 to 70 cm from camera)
    Zavg = 0.6
    # Average focal length of fx and fy
    favg = (intrin['fx']+intrin['fy'])/2
    # Compute scaling factor S
    S = favg/Zavg 
    # Uniform scaling
    param['joint_3d'] /= S

    # Estimate 3D position of Zwrist using similar triangle
    # Note: Hardcode actual dist btw wrist and index finger MCP as 8cm
    D = 0.08 
    # Dist btw wrist and index finger MCP keypt (in 2D image coor)
    d = np.linalg.norm(param['keypt'][0] - param['keypt'][9])
    # By similar triangle: d/f = D/Z -> Z = D/d*f
    Zwrist = D/d*favg
    # Compute actual depth
    param['joint_3d'][:,2] += Zwrist

    return param['joint_3d']


def convert_relative_to_actual_3d_joint_(param, intrin):
    # Adapted from Iqbal et al.
    # (ECCV 2018) Hand Pose Estimation via Latent 2.5D Heatmap Regression
    # But there are some errors in equation:
    # (4) missing camera intrinsic
    # (6) missing a constant multiplication of 2

    # Select wrist joint (n) and index finger MCP (m)
    xn, yn = param['keypt'][0] # Wrist
    xm, ym = param['keypt'][9] # Index finger MCP
    xn = (xn-intrin['cx'])/intrin['fx']
    xm = (xm-intrin['cx'])/intrin['fx']
    yn = (yn-intrin['cy'])/intrin['fy']
    ym = (ym-intrin['cy'])/intrin['fy']

    Zn = param['joint'][0,2] # Relative Z coor of wrist
    Zm = param['joint'][9,2] # Relative Z coor of index finger MCP

    # Precalculate value for computing Zroot
    xx = xn-xm
    yy = yn-ym
    xZ = xn*Zn - xm*Zm
    yZ = yn*Zn - ym*Zm
    ZZ = Zn-Zm

    # Compute Zroot relative
    C = 1
    a = xx*xx + yy*yy
    b = 2*(xx*xZ + yy*yZ)
    c = xZ*xZ + yZ*yZ + ZZ*ZZ - C*C
    Zroot = (-b + np.sqrt(b*b - 4*a*c))/(2*a)

    # Convert to actual scale
    s = 0.08 # Note: Hardcode distance from wrist to index finger MCP as 8cm
    Zroot *= s / C

    # Compute actual depth
    param['joint_3d'][:,2] = param['joint'][:,2] + Zroot

    # Compute X and Y
    param['joint_3d'][:,0] = (param['keypt'][:,0]-intrin['cx'])/intrin['fx'] 
    param['joint_3d'][:,1] = (param['keypt'][:,1]-intrin['cy'])/intrin['fy']
    param['joint_3d'][:,:2] *= param['joint_3d'][:,2:3]

    return param['joint_3d']


#############################################################
### Simple gesture recognition from joint angle using KNN ###
#############################################################
class GestureRecognition:
    def __init__(self, mode='train'):
        super(GestureRecognition, self).__init__()

        # 11 types of gesture 'name':class label
        self.gesture = {
            'fist':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,
            'rock':7,'spiderman':8,'yeah':9,'ok':10,
        }

        if mode=='train':
            # Create .csv file to log training data
            self.file = open('../data/gesture_train.csv', 'a+')
        elif mode=='eval':
            # Load training data
            file = np.genfromtxt('../data/gesture_train.csv', delimiter=',')
            # Extract input joint angles
            angle = file[:,:-1].astype(np.float32)
            # Extract output class label
            label = file[:, -1].astype(np.float32)
            # Use OpenCV KNN
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)


    def train(self, angle, label):
        # Log training data
        data = np.append(angle, label) # Combine into one array
        np.savetxt(self.file, [data], delimiter=',', fmt='%f')
        

    def eval(self, angle):
        # Use KNN for gesture recognition
        data = np.asarray([angle], dtype=np.float32)
        ret, results, neighbours ,dist = self.knn.findNearest(data, 3)
        idx = int(results[0][0]) # Index of class label

        return list(self.gesture)[idx] # Return name of class label


##############################################################
### Simple hand ROM recognition from joint angle using KNN ###
##############################################################
class HandRomRecognition:
    def __init__(self, mode='train'):
        super(HandRomRecognition, self).__init__()

        # 13 types of hand ROM 'name':class label
        self.gesture = {
            'Finger MCP Flexion'    :0,
            'Finger PIP DIP Flexion':1,
            'Thumb MCP Flexion'     :2,
            'Thumb IP Flexion'      :3,
            'Thumb Radial Abduction':4,
            'Thumb Palmar Abduction':5, # From this class onwards hard to differentiate
            'Thumb Opposition'      :6, 
            'Forearm Pronation'     :7, # Not done yet
            'Forearm Supination'    :8, # Not done yet
            'Wrist Flexion'         :9, # Not done yet
            'Wrist Extension'       :10,# Not done yet
            'Wrist Radial Deviation':11,# Not done yet
            'Wrist Ulnar Deviation' :12,# Not done yet
        }

        if mode=='train':
            # Create .csv file to log training data
            self.file = open('../data/handrom_train.csv', 'a+')
        elif mode=='eval':
            # Load training data
            file = np.genfromtxt('../data/handrom_train.csv', delimiter=',')
            # Extract input joint angles
            angle = file[:,:-1].astype(np.float32)
            # Extract output class label
            label = file[:, -1].astype(np.float32)
            # Use OpenCV KNN
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)


    def train(self, angle, label):
        # Log training data
        data = np.append(angle, label) # Combine into one array
        np.savetxt(self.file, [data], delimiter=',', fmt='%f')
        

    def eval(self, angle):
        # Use KNN for gesture recognition
        data = np.asarray([angle], dtype=np.float32)
        ret, results, neighbours ,dist = self.knn.findNearest(data, 3)
        idx = int(results[0][0]) # Index of class label

        return list(self.gesture)[idx] # Return name of class label
