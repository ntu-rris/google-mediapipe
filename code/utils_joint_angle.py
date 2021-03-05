###############################################################################
### Useful function for converting 3D joint to joint angles
### Joint angle is then used for:
### 1) Gesture recognition 
### 2) Hand ROM recognition 
###############################################################################

import cv2
import numpy as np


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
            'Wrist Flexion'         :7, # Refer to WristArmRom
            'Wrist Extension'       :8, # Refer to WristArmRom
            'Wrist Radial Deviation':9, # Refer to WristArmRom
            'Wrist Ulnar Deviation' :10,# Refer to WristArmRom
            'Forearm Neutral'       :11,# Refer to WristArmRom
            'Forearm Pronation'     :12,# Refer to WristArmRom
            'Forearm Supination'    :13,# Refer to WristArmRom
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


###########################################################
### Simple measurement of wrist and forearm joint angle ###
###########################################################
class WristArmRom:
    def __init__(self, mode=2, side='right'):
        super(WristArmRom, self).__init__()
        
        # 3 modes of measurement
        # 0: Wrist flexion/extension
        # 1: Wrist radial/ulnar deviation
        # 2: Forearm pronation/supination
        self.mode = mode
        self.side = side # left / right

        # 13 types of hand ROM 'name':class label
        self.gesture = {
            'Finger MCP Flexion'    :0, # Refer to HandRomRecognition
            'Finger PIP DIP Flexion':1, # Refer to HandRomRecognition
            'Thumb MCP Flexion'     :2, # Refer to HandRomRecognition
            'Thumb IP Flexion'      :3, # Refer to HandRomRecognition
            'Thumb Radial Abduction':4, # Refer to HandRomRecognition
            'Thumb Palmar Abduction':5, # Refer to HandRomRecognition
            'Thumb Opposition'      :6, # Refer to HandRomRecognition
            'Wrist Flex/Extension'  :7, # Use mode 0 # Note: hard to differentiate
            # 'Wrist Extension'       :8, # Note: hard to differentiate
            'Wrist Radial/Ulnar Dev':9, # Use mode 1 # Note: hard to differentiate
            # 'Wrist Ulnar Deviation' :10, # Note: hard to differentiate
            'Forearm Neutral'       :11,# Use mode 2
            'Forearm Pronation'     :12,# Use mode 2
            'Forearm Supination'    :13,# Use mode 2
        }


    def eval(self, param):
        # Wrist flexion/extension, wrist radial/ulnar deviation
        if self.mode==0 or self.mode==1:
            # Assume camera is placed such that upper body of the subject
            # is visible especially the elbow joints

            # Wrist joint angle can be simply calculated from angle between
            # Vector 1: joining body elbow joint[13/14] to wrist[15/16] 
            # Vector 2: joining hand wrist[0] to middle finger MCP joint[9]
            _, param_lh, param_rh, param_bd = param

            if self.side=='left':
                v1 = param_bd['joint'][15]- param_bd['joint'][13]
                v2 = param_lh['joint'][9] - param_lh['joint'][0]
            elif self.side=='right':
                v1 = param_bd['joint'][16]- param_bd['joint'][14]
                v2 = param_rh['joint'][9] - param_rh['joint'][0]

            # Normalize vector
            v1 = v1/(np.linalg.norm(v1)+1e-6)
            v2 = v2/(np.linalg.norm(v2)+1e-6)
            # Get angle using arcos of dot product between the two vectors
            angle = np.arccos(np.dot(v1, v2))
            angle = np.degrees(angle) # Convert radian to degree

            if self.side=='left':
                param_lh['angle'][0] = angle # Note: Temporary make use hand joint angles            
                if self.mode==0 and angle>10:
                    param_lh['gesture'] = 'Wrist Flex/Extension'
                elif self.mode==1 and angle>10:
                    param_lh['gesture'] = 'Wrist Radial/Ulnar Dev'
            elif self.side=='right':
                param_rh['angle'][0] = angle # Note: Temporary make use hand joint angles            
                if self.mode==0 and angle>10:
                    param_rh['gesture'] = 'Wrist Flex/Extension'
                elif self.mode==1 and angle>10:
                    param_rh['gesture'] = 'Wrist Radial/Ulnar Dev'


        # Forearm pronation/supination
        elif self.mode==2:
            # Assume camera is placed at the same level as the hand
            # such that palmar side of the hand is directly facing camera
            # Forearm pronation/supination can be simply calculated from palm direction normal

            # Get normal of plane joining wrist[0], index finger MCP[5] and little finger MCP[17]
            # Note: Palm normal points from palmar side to dorsal side
            v1 = param[0]['joint'][5] - param[0]['joint'][0] # Vector pointing from wrist to index finger MCP
            v2 = param[0]['joint'][17]- param[0]['joint'][0] # Vector pointing from wrist to little finger MCP
            n  = np.cross(v1,v2) # Palm normal vector
            # Normalize normal vector
            n  = n/(np.linalg.norm(n)+1e-6)

            # When forearm is in neutral position, palm is directly facing camera
            # Palm normal vector shld be parallel to camera z-axis
            # Get angle using arcos of dot product between palm normal and z-axis
            angle = np.arccos(np.dot(n,np.array([0,0,1])))
            angle = np.degrees(angle) # Convert radian to degree
            # Get direction by checking on y component of normal vector
            direc = n[1]

            if self.side=='left':
                # Reverse angle and direction for left hand
                angle = 180-angle
                direc = -direc

            if angle<30:
                param[0]['gesture'] = 'Forearm Neutral'
            elif angle>30 and direc<0:
                param[0]['gesture'] = 'Forearm Pronation'
            elif angle>30 and direc>0:
                param[0]['gesture'] = 'Forearm Supination'
            
            param[0]['angle'][0] = angle # Note: Temporary make use hand joint angles

        return param
