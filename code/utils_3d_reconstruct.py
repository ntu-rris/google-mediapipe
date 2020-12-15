###############################################################################
### Useful function for 3D reconstruction from multiple images
### 
### First requirement is to obtain camera intrinsics and extrinsics parameters
### Refer to Calibration class
### 
### Next use triangulation/direct linear transformation (DLT)
### To reconstruct 3D points from 2D image points
###############################################################################

import cv2
import glob
import json
import yaml
import numpy as np
import open3d as o3d


class Calibration:
    def __init__(self, chessboard_size=(6,5), chessboard_sq_size=0.015):
        super(Calibration, self).__init__()

        self.chessboard_size    = chessboard_size
        self.chessboard_sq_size = chessboard_sq_size
        
        # Prepare 3D object points in real world space
        self.obj_pts = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        # [[0,0,0], [1,0,0], [2,0,0] ....,[9,6,0]]
        self.obj_pts[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2) 
        self.obj_pts *=  chessboard_sq_size # Convert length of each black square to units in meter
        
        # Termination criteria for cornerSubPix
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)

        # Flag for findChessboardCorners
        self.flags_findChessboard = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

        # Termination criteria for stereoCalibrate
        self.criteria_stereoCalibrate = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # Flag for stereoCalibrate
        self.flags_stereoCalibrate = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_CB_FAST_CHECK


    def get_intrin(self, folder):
        # Use chessboard pattern to calib intrinsic of color camera

        # Read in filename of .png from folder
        file = glob.glob(folder+'*.png')
        file.sort()
        
        # List to store object point and image point from all images
        objpt = [] # 3d point in real world space
        imgpt = [] # 2d point in image plane
        file_ = [] # Filename of images with success findChessboardCorners
                
        for f in file:
            # Read in image
            img = cv2.imread(f)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, self.flags_findChessboard)

            # If found, add object points, image points (after refining them)
            if ret==True:
                corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.criteria) # (chessboard_size[0]*chessboard_size[1], 1, 2}
                imgpt.append(corners2)
                objpt.append(self.obj_pts)
                file_.append(f)
                print('Found ChessboardCorners', f)
            else:
                print('Cannot find ChessboardCorners', f)
                
        # Calibration
        if len(objpt)>0 and len(imgpt)>0:
            print('Calibrating ...')
            ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpt, imgpt, gray.shape[::-1], None, None)
            print('Calibrating done')

            # Draw projected xyz axis on the image
            for i, f in enumerate(file_):
                # Read in image
                img = cv2.imread(f)

                # Draw corners
                img = cv2.drawChessboardCorners(img, self.chessboard_size, imgpt[i], True)                

                # solvePnp will return the transformation matrix to transform 3D model coordinate to 2D camera coordinate
                ret, rvec, tvec = cv2.solvePnP(objpt[i], imgpt[i], mat, dist)     
                self.project_3Daxis_to_2Dimage(img, mat, dist, rvec, tvec)

                # Save image with new name and extension
                f_ = f[:-4] + '_.jpg'
                cv2.imwrite(f_, img)

                # Get reprojection error
                error = self.get_reprojection_error(
                            np.asarray(objpt[i]).reshape(-1, 3), # To convert from m list of (n, 3) to (m*n, 3)
                            np.asarray(imgpt[i]).reshape(-1, 2), # To convert from m list of (n, 1, 2) to (m*n, 2)
                            mat, dist, rvec, tvec)
                print('Img', i, f, 'reprojection error', error)

            # Save camera intrinsic
            img = cv2.imread(file[0])
            data = dict(intrin_mat=mat.tolist(),
                        dist_coeff=dist.tolist(),
                        img_height=img.shape[0],
                        img_width=img.shape[1])
            filepath = folder + 'intrin.yaml'
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                print('Saved camera intrinsic to', filepath)


    def get_extrin(self, folder):
        # Use chessboard pattern to get extrinsic of color cameras
        # Note: Simplified calibration with only one image per camera view

        # Read in filename of .png from folder
        file = glob.glob(folder+'*.png')
        file.sort()

        for i, f in enumerate(file):
            # Extract camera index from last few characters of filename
            cam_idx = f.split('/')[-1]
            cam_idx = cam_idx.split('.')[0] # Note: f = '../data/calib_extrin/cam_X.png', where X is camera index
            
            # Read in camera intrinsic
            filepath = '../data/calib_intrin/'+cam_idx+'/intrin.yaml'
            param = yaml.load(open(filepath), Loader=yaml.FullLoader)
            mat = np.asarray(param['intrin_mat'])
            dist = np.asarray(param['dist_coeff'])

            # Read in image
            img = cv2.imread(f)

            # Convert to grayscale first
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, self.flags_findChessboard)

            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.criteria) # (chessboard_size[0]*chessboard_size[1], 1, 2}

                # Draw corners
                img = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                
                # solvePnp will return the transformation matrix to transform 3D model coordinate to 2D camera coordinate
                ret, rvec, tvec = cv2.solvePnP(self.obj_pts, corners2, mat, dist)
                self.project_3Daxis_to_2Dimage(img, mat, dist, rvec, tvec)              
                
                # Save image with new name and extension
                f_ = f[:-4] + '_.jpg'
                cv2.imwrite(f_, img)
                # Display image
                cv2.imshow('img'+str(i), img)
                cv2.waitKey(1)                

                # Get reprojection error
                error = self.get_reprojection_error(
                            np.asarray(self.obj_pts).reshape(-1, 3), # To convert from m list of (n, 3) to (m*n, 3)
                            np.asarray(corners2).reshape(-1, 2), # To convert from m list of (n, 1, 2) to (m*n, 2)
                            mat, dist, rvec, tvec)                
                print('Img', f, 'reprojection error', error)

                # Create 4 by 4 homo matrix [R|T] to transform 3D model coordinate to 3D camera coordinate 
                homo_matrix = np.hstack((cv2.Rodrigues(rvec)[0], tvec)) # 3 by 4 matrix
                homo_matrix = np.vstack((homo_matrix, np.array([0,0,0,1]))) # 4 by 4 matrix

                # Save camera extrinsic
                data = dict(extrin_mat=homo_matrix.tolist())
                filepath = f[:-4] + '_extrin.yaml'
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
                    print('Saved camera extrinsic to', filepath)


    def get_extrin_mirror(self, folder, idx=0):
        # Use chessboard to calib extrin of color cameras
        # Note: Simplified calibration with only one image per camera view
        # Note: Assume setup contains one camera with two plane mirrors -> total of 3 camera views

        print('Assume setup contains one camera with two plane mirrors')
        print('Select region of interest (ROI) containing chessboard pattern in the following order:')
        print(' 1) Select ROI of actual camera view')
        print(' 2) Select ROI of 1st virtual camera view')
        print(' 3) Select ROI of 2nd virtual camera view')

        # Read in camera intrinsic
        cam_idx = 'cam_'+str(idx).zfill(2)
        filepath = '../data/calib_intrin/'+cam_idx+'/intrin.yaml'
        param = yaml.load(open(filepath), Loader=yaml.FullLoader)
        mat = np.asarray(param['intrin_mat'])
        dist = np.asarray(param['dist_coeff'])

        # Read in image
        img = cv2.imread(folder+'image.png')
        # Keep original image for drawing
        ori = cv2.imread(folder+'image.png')

        for i in range(3): # Note: Hardcode 3 views
            # Manually select ROI
            print('')
            roi = cv2.selectROI(img) # Return top left x, y, width, height
            # Mask out non ROI
            tmp = self.mask_non_roi(img, roi)

            # Convert to grayscale
            gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, self.flags_findChessboard)

            if ret==False:
                print('Cannot find ChessboardCorners')
            else:
                # If found, add object points, image points (after refining them)
                corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.criteria) # (chessboard_size[0]*chessboard_size[1], 1, 2}

                # For next two views need to flip corners as they are from virtual camera
                if i>0:
                    corners2 = self.flip_corners(corners2)
                    # Need to mirror reflect about the x axis also
                    corners2[:,:,0] = img.shape[1] - corners2[:,:,0]
                    img = cv2.flip(img, flipCode=1)
                    ori = cv2.flip(ori, flipCode=1)

                # Draw corners
                ori = cv2.drawChessboardCorners(ori, self.chessboard_size, corners2, ret)

                # solvePnp will return the transformation matrix to transform 3D model coordinate to 2D camera coordinate
                ret, rvec, tvec = cv2.solvePnP(self.obj_pts, corners2, mat, dist)
                self.project_3Daxis_to_2Dimage(ori, mat, dist, rvec, tvec)

                # Mask out chessboard
                img = self.mask_chessboard(img, corners2)

                # Flip back image
                if i>0:
                    img = cv2.flip(img, flipCode=1)
                    ori = cv2.flip(ori, flipCode=1)                    

                # Display image
                cv2.imshow('ori', ori)
                cv2.waitKey(1)

                # Get reprojection error
                error = self.get_reprojection_error(
                            np.asarray(self.obj_pts).reshape(-1, 3), # To convert from m list of (n, 3) to (m*n, 3)
                            np.asarray(corners2).reshape(-1, 2), # To convert from m list of (n, 1, 2) to (m*n, 2)
                            mat, dist, rvec, tvec)                
                print('Img', i, 'reprojection error', error)

                # Create 4 by 4 homo matrix [R|T] to transform 3D model coordinate to 3D camera coordinate 
                homo_matrix = np.hstack((cv2.Rodrigues(rvec)[0], tvec)) # 3 by 4 matrix
                homo_matrix = np.vstack((homo_matrix, np.array([0,0,0,1]))) # 4 by 4 matrix
                # Note: Inverse matrix to get transformation from image to camera frame
                homo_matrix = np.linalg.inv(homo_matrix)                

                # Save camera extrinsic
                data = dict(extrin_mat=homo_matrix.tolist())
                filepath = folder + 'cam_' + str(i).zfill(2) + '_extrin.yaml'
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
                    print('Saved camera extrinsic to', filepath)

        # Save image with new name and extension
        cv2.imwrite(folder+'image_.jpg', ori)


    def project_3Daxis_to_2Dimage(self, img, mat, dist, rvec, tvec):
        axis_3D = np.float32([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]) * 2 * self.chessboard_sq_size # Create a 3D axis that is twice the length of chessboard_sq_size
        axis_2D = cv2.projectPoints(axis_3D, rvec, tvec, mat, dist)[0].reshape(-1, 2)
        if axis_2D.shape[0]==4:
            colours = [(0,0,255),(0,255,0),(255,0,0)] # BGR
            for i in range(1,4):
                (x0, y0), (x1, y1) = axis_2D[0], axis_2D[i]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), colours[i-1], 3)    


    def get_reprojection_error(self, p3D, p2D, mat, dist, rvec, tvec):
        p2D_reproj = cv2.projectPoints(p3D, rvec, tvec, mat, dist)[0].reshape(-1, 2)
        error = cv2.norm(p2D, p2D_reproj, cv2.NORM_L2) / len(p2D)

        return error # https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html


    def mask_non_roi(self, img, roi):
        x, y, width, height = roi

        # Create binary mask
        msk = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        msk[y:y+height,x:x+width] = 255

        # Mask non ROI
        tmp = cv2.bitwise_and(img, img, mask=msk)

        return tmp


    def mask_chessboard(self, img, corners):
        # Get four corners
        c0 = corners[0,:,:].astype(np.int32)
        c1 = corners[self.chessboard_size[0]-1,:,:].astype(np.int32)
        c2 = corners[-self.chessboard_size[0],:,:].astype(np.int32)
        c3 = corners[-1,:,:].astype(np.int32)
        c_ = np.array([c0,c1,c3,c2])

        tmp = img.copy()
        cv2.fillPoly(tmp, [c_], color=(255,255,255)) # Mask out the corners with white polygon

        return tmp


    def flip_corners(self, corners):
        col, row = self.chessboard_size[0], self.chessboard_size[1]
        temp = corners.copy()

        # Note: For (col=odd,row=even) checkerboard
        if row%2==0:
            for r in range(row):
              temp[r*col:r*col+col, :, :] = corners[(row-r-1)*col:(row-r-1)*col+col:, :, :]
        
        # Note: For (col=even,row=odd) checkerboard
        elif col%2==0:
            for r in range(row):
                for c in range(col):
                    temp[r*col+c, :, :] = corners[r*col+(col-c-1), :, :]

        return temp


    def visualize_cam_pose(self, folder):
        file = glob.glob(folder+'*.yaml')
        file.sort()
        color = [[1,0,0],[0,1,0],[0,0,1]] # Note: maximum 3 cameras
        cam_frame = []
        for i, f in enumerate(file):
            # Read in camera extrin
            param = yaml.load(open(f), Loader=yaml.FullLoader)
            extrin = np.asarray(param['extrin_mat'])
            # Create camera frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(np.linalg.inv(extrin))
            frame.paint_uniform_color(color[i])
            cam_frame.append(frame)

        # Create world reference frame
        ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.chessboard_sq_size*4)

        # Create chessboard pattern
        mesh = self.create_chessboard_pattern_open3d()

        o3d.visualization.draw_geometries([ref_frame, mesh] + cam_frame)


    def create_chessboard_pattern_open3d(self):
        # Create chessboard pattern for Open3D visualization

        # Origin at top left of the checkerboard
        # Axis is +ve X points to right, +ve Y points downwards
        # o-->X
        # |
        # Y
        # Vertex order is anti clockwise
        # 0 ---- 3
        # |      |
        # 1 ---- 2
        vertices  = []
        triangles = []    # Must be anti clockwise order when view from outside of the mesh
        black     = True  # Use to alternate between black and white square when loop across row and col
        index     = 0     # To keep track of the number of vertices
        size      = self.chessboard_sq_size
        for i in range(self.chessboard_size[1]+1): # In +ve Y axis direction
            for j in range(self.chessboard_size[0]+1): # In +ve X axis direction
                    if black:
                        x0, y0 = j*size, i*size           # In anti clockwise order from top left
                        x1, y1 = j*size, i*size+size      # bottom left
                        x2, y2 = j*size+size, i*size+size # bottom right
                        x3, y3 = j*size+size, i*size      # top right
                        vertices.append([x0, y0, 0])
                        vertices.append([x1, y1, 0])
                        vertices.append([x2, y2, 0])
                        vertices.append([x3, y3, 0])
                        triangles.append([index, index+1, index+2])
                        triangles.append([index, index+2, index+3])
                        index += 4

                    black = not black # Toggle the flag for next square
            
            if (self.chessboard_size[0]+1)%2 == 0: # Important: Need to check if col is even else will get parallel black strips as for even col the sq in the next row follw the same color
                black = not black

        # To shift the origin to the bottom right of first top left black square
        vertices = np.asarray(vertices) - np.array([size, size, 0])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0,0,0]) # Black color

        return mesh        


class Triangulation:
    def __init__(self, cam_idx, vis=None, use_panoptic_dataset=False):
        super(Triangulation, self).__init__()

        #############################
        ### Load camera parameter ###
        #############################
        if use_panoptic_dataset:
            data_path = '../data/'
            seq_name  = '171204_pose1_sample'
            # Load camera calibration param
            with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as f:
                calib = json.load(f)

            # Cameras are identified by a tuple of (panel#,node#)
            # Note: 31 HD cameras (0,0) - (0,30), where the zero in the first index means HD camera 
            cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

            # Convert data into numpy arrays for convenience
            for k, cam in cameras.items():    
                cam['K'] = np.matrix(cam['K'])
                cam['distCoef'] = np.array(cam['distCoef'])
                cam['R'] = np.matrix(cam['R'])
                cam['t'] = np.array(cam['t'])*0.01 # Convert cm to m

            # Extract camera index integer from video file name
            cam_idx_ = []
            for c in cam_idx:
                # Example of file name
                # ../data/171204_pose1_sample/hdVideos/hd_00_00.mp4
                value = c.split('_')[-1] # Select the last split (XX.mp4)
                value = value.split('.')[0] # Select the first split (XX)
                cam_idx_.append(int(value))
            
            # Compute projection matrix
            self.pmat = []
            for i in range(len(cam_idx_)):
                cam = cameras[(0,cam_idx_[i])]
                extrin_mat = np.zeros((3,4))
                extrin_mat[:3,:3] = cam['R']
                extrin_mat[:3,3:] = cam['t']
                self.pmat.append(cam['K'] @ extrin_mat)

            #############################
            ### Visualize camera pose ###
            #############################
            # Draw world frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            vis.add_geometry(frame)
            
            # Draw camera axis
            hd_cam_idx = zip([0] * 30,range(0,30)) # Choose only HD cameras
            hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]
            for i, cam in enumerate(hd_cameras):
                if i in cam_idx_: # Show only those selected camera
                    extrin_mat = np.eye(4)
                    extrin_mat[:3,:3] = cam['R']
                    extrin_mat[:3,3:] = cam['t']
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                    axis.transform(np.linalg.inv(extrin_mat))
                    vis.add_geometry(axis)


    def triangulate_2views(self, param, mode):

        if mode=='body':
            p0 = param[0]['keypt'] # [nPt,2]
            p1 = param[1]['keypt'] # [nPt,2]

        elif mode=='holistic':
            _, param_lh, param_rh, param_bd = param[0]
            p0 = np.vstack((param_lh['keypt'], 
                            param_rh['keypt'],
                            param_bd['keypt'])) # [21+21+33/25,2]
            
            _, param_lh, param_rh, param_bd = param[1]
            p1 = np.vstack((param_lh['keypt'], 
                            param_rh['keypt'],
                            param_bd['keypt'])) # [21+21+33/25,2]


        # Note: OpenCV can only triangulate from 2 views
        p3d = cv2.triangulatePoints(
            self.pmat[0], self.pmat[1],
            p0.T, p1.T) # Note: triangulatePoints requires 2xN arrays, so transpose
        
        # However, homgeneous point is returned so need to divide by last term
        p3d /= p3d[3] # [4,nPt]
        p3d = p3d[:3,:].T # [nPt,3]

        # Update param 3D joint
        if mode=='body':
            param[0]['joint'] = p3d # [nPt,3]
            param[1]['joint'] = p3d # [nPt,3]

        elif mode=='holistic':
            _, param_lh, param_rh, param_bd = param[0]
            param_lh['joint'] = p3d[  :21]
            param_rh['joint'] = p3d[21:42]
            param_bd['joint'] = p3d[42:]

            _, param_lh, param_rh, param_bd = param[1]
            param_lh['joint'] = p3d[  :21]
            param_rh['joint'] = p3d[21:42]
            param_bd['joint'] = p3d[42:]

        return param


    def triangulate_nviews(self, param, mode):

        p2d = [] # List of len nCam to store [nPt,2] for each view
        if mode=='body':
            for p in param:
                p2d.append(p['keypt']) # [nPt,2]

        elif mode=='holistic':
            for p in param:
                _, param_lh, param_rh, param_bd = p
                p2d.append(np.vstack((
                            param_lh['keypt'], 
                            param_rh['keypt'],
                            param_bd['keypt'])) # [21+21+33/25,2]
                           )

        # Convert list into a single array
        p2d = np.concatenate(p2d, axis=1) # [nPt,2*nCam]
        nPt = p2d.shape[0]

        p3d = np.zeros((nPt,3))
        for i in range(nPt):
            p3d[i,:] = self.triangulate_point(p2d[i,:].reshape(-1,2))

        # Update param 3D joint
        if mode=='body':
            for p in param:
                p['joint'] = p3d # [nPt,3]

        elif mode=='holistic':
            for p in param:
                _, param_lh, param_rh, param_bd = p
                param_lh['joint'] = p3d[  :21]
                param_rh['joint'] = p3d[21:42]
                param_bd['joint'] = p3d[42:]

        return param


    def triangulate_point(self, point):
        # Modified from 
        # https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914

        # Other possible python implementation
        # https://www.mail-archive.com/floatcanvas@mithis.com/msg00513.html

        # Also its worthwhile to read through the below link
        # http://kwon3d.com/theory/dlt/dlt.html 
        # For indepth explanation on DLT and many other useful theories
        # required by multicam mocap by Prof Young-Hoo Kwon

        # Use DLT to triangulate a 3D point from N image points in N camera views
        N = len(self.pmat) # Number of camera views
        M = np.zeros((3*N, 4+N))
        for i in range(N):
            M[3*i:3*i+3, :4] = self.pmat[i] # [3,4]
            M[3*i:3*i+2,4+i] = -point[i] # [2,1]
            M[3*i+2    ,4+i] = -1  # Homogeneous coordinate
        V = np.linalg.svd(M)[-1] # [4+N,4+N]
        X = V[-1,:4] # [4]
        X = X / X[3] # [4]

        return X[:3] # [3]


class PanopticDataset:
    def __init__(self, data_path='../data/', seq_name='171204_pose1_sample'):
        super(PanopticDataset, self).__init__()

        # Load camera calibration parameters
        with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as f:
            calib = json.load(f)

        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

        # Convert data into numpy arrays for convenience
        for k, cam in cameras.items():    
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3,1)) * 0.01 # Convert cm to m
            
        # Choose only HD cameras for visualization
        hd_cam_idx = zip([0] * 30,range(0,30))
        hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]

        # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera 
        # cam = cameras[(0,5)]        

        # Visualize 3D camera pose
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)

        # Draw world frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # Draw camera axis
        axes = []
        for cam in hd_cameras:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            tmat = np.eye(4)
            tmat[:3,:3] = cam['R']
            tmat[:3,3:] = cam['t']
            axis.transform(np.linalg.inv(tmat))
            axes.append(axis)

        # Draw body pose
        hd_idx = 0
        hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
        # Edges between joints in the body skeleton
        body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as f:
            bframe = json.load(f)

        # Bodies
        for ids in range(len(bframe['bodies'])):
            body = bframe['bodies'][ids]
            skel = np.array(body['joints19']).reshape((-1,4))[:,:3] * 0.01
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(skel)

            bone = o3d.geometry.LineSet()
            bone.points = o3d.utility.Vector3dVector(skel)
            bone.lines  = o3d.utility.Vector2iVector(body_edges)

            self.vis.add_geometry(pcd)
            self.vis.add_geometry(bone)

        self.vis.add_geometry(frame)
        for a in axes:
            self.vis.add_geometry(a)

        self.vis.run()


    def projectPoints(X, K, R, t, Kd):
        """ Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        
        Roughly, x = K*(R*X + t) + distortion
        
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """
        
        x = np.asarray(R*X + t)
        
        x[0:2,:] = x[0:2,:]/x[2,:]
        
        r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
        
        x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
        x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

        x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
        x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
        
        return x


###############################################################################
### Simple example to test program                                          ###
###############################################################################
if __name__ == '__main__':

    pano = PanopticDataset()
