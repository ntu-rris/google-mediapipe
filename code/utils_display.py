###############################################################################
### Useful function for visualization of 
### face, hand, body, holistic and object pose estimation
###############################################################################

import cv2
import numpy as np
import open3d as o3d


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


class DisplayFace:
    def __init__(self, draw3d=False, intrin=None, max_num_faces=1, vis=None):
        self.max_num_faces = max_num_faces
        self.nPt = 468 # Define number of keypoints/joints
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin        

        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
                self.vis.get_render_option().point_size = 3.0
            joint = np.zeros((self.nPt,3))

            # Draw face mesh
            # .obj file adapted from https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/data
            # self.mesh = o3d.io.read_triangle_mesh('../data/canonical_face_model.obj') # Seems like Open3D ver 0.11.2 have some issue with reading .obj https://github.com/intel-isl/Open3D/issues/2614
            self.mesh = o3d.io.read_triangle_mesh('../data/canonical_face_model.ply')
            self.mesh.paint_uniform_color([255/255, 172/255, 150/255]) # Skin color
            self.mesh.compute_vertex_normals()
            self.mesh.scale(0.01, np.array([0,0,0]))

            # Draw world reference frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

            # Add geometry to visualize
            self.vis.add_geometry(frame)
            self.vis.add_geometry(self.mesh)

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_up([0,-1,0]) # Set up as -y axis
            ctr.set_front([0,0,-1]) # Set to looking towards -z axis
            ctr.set_lookat([0.5,0.5,0]) # Set to center of view
            ctr.set_zoom(1)


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        # Loop through different faces
        for p in param:
            if p['detect']:
                # Draw contours around eyes, eyebrows, lips and entire face
                for connect in FACE_CONNECTIONS:
                    x = int(p['keypt'][connect[0],0])
                    y = int(p['keypt'][connect[0],1])
                    x_= int(p['keypt'][connect[1],0])
                    y_= int(p['keypt'][connect[1],1])
                    if x_>0 and y_>0 and x_<img_width and y_<img_height and \
                       x >0 and y >0 and x <img_width and y <img_height:
                        cv2.line(img, (x_, y_), (x, y), (0,255,0), 1) # Green

                # Loop through keypoint for each face
                for i in range(self.nPt):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Draw keypoint
                        cv2.circle(img, (x, y), 1, (0,0,255), -1) # Red

            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   

        return img


    def draw2d_(self, img, param):
        # Different from draw2d
        # draw2d_ draw 2.5D with relative depth info
        # The closer the landmark is towards the camera
        # The lighter the color of circle

        img_height, img_width, _ = img.shape

        # Loop through different hands
        for p in param:
            if p['detect']:
                # Draw contours around eyes, eyebrows, lips and entire face
                for connect in FACE_CONNECTIONS:
                    x = int(p['keypt'][connect[0],0])
                    y = int(p['keypt'][connect[0],1])
                    x_= int(p['keypt'][connect[1],0])
                    y_= int(p['keypt'][connect[1],1])
                    if x_>0 and y_>0 and x_<img_width and y_<img_height and \
                       x >0 and y >0 and x <img_width and y <img_height:
                        cv2.line(img, (x_, y_), (x, y), (255,255,255), 1) # White

                min_depth = min(p['joint'][:,2])
                max_depth = max(p['joint'][:,2])

                # Loop through keypt and joint for each face
                for i in range(self.nPt):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Convert depth to color nearer white, further black
                        depth = (max_depth-p['joint'][i,2]) / (max_depth-min_depth)
                        color = [int(255*depth), int(255*depth), int(255*depth)]

                        # Draw keypoint
                        cv2.circle(img, (x, y), 2, color, -1)
            
            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)                           

        return img


    def draw3d(self, param):
        for i in range(self.max_num_faces):
            if param[i]['detect']:
                self.mesh.vertices = o3d.utility.Vector3dVector(param[i]['joint'])
            else:
                self.mesh.vertices = o3d.utility.Vector3dVector(np.zeros((self.nPt,3)))


    def draw3d_(self, param):
        # Different from draw3d
        # draw3d_ draw the actual 3d joint in camera coordinate
        for i in range(self.max_num_faces):
            if param[i]['detect']:
                self.mesh.vertices = o3d.utility.Vector3dVector(param[i]['joint_3d'])
            else:
                self.mesh.vertices = o3d.utility.Vector3dVector(np.zeros((self.nPt,3)))


class DisplayFaceMask:
    def __init__(self, img, draw3d=False, max_num_faces=1):
        # Note: This class is specially created for demo 07_face_mask.py
        self.max_num_faces = max_num_faces
        self.nPt = 468 # Define number of keypoints/joints

        ############################
        ### Open3D visualization ###
        ############################
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=img.shape[1], height=img.shape[0])
        self.vis.get_render_option().point_size = 3.0
        joint = np.zeros((self.nPt,3))

        # Draw face mesh
        # .obj file adapted from https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/data
        # self.mesh = o3d.io.read_triangle_mesh('../data/canonical_face_model.obj') # Seems like Open3D ver 0.11.2 have some issue with reading .obj https://github.com/intel-isl/Open3D/issues/2614
        self.mesh = o3d.io.read_triangle_mesh('../data/canonical_face_model.ply')        
        self.mesh.paint_uniform_color([255/255, 172/255, 150/255]) # Skin color
        self.mesh.compute_vertex_normals()
        self.mesh.scale(0.01, [0,0,0])

        # Draw 2D image plane in 3D space
        self.mesh_img = self.create_mesh_img(img)

        # Draw world reference frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # Add geometry to visualize
        self.vis.add_geometry(frame)
        self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.mesh_img)

        # Set camera view
        ctr = self.vis.get_view_control()
        ctr.set_up([0,-1,0]) # Set up as -y axis
        ctr.set_front([0,0,-1]) # Set to looking towards -z axis
        ctr.set_lookat([0.5,0.5*img.shape[0]/img.shape[1],0]) # Set to center of view
        ctr.set_zoom(0.6)  


    def create_mesh_img(self, img):
        h, w, _ = img.shape

        mesh = o3d.geometry.TriangleMesh()
        # Convention of 4 vertices
        # --> right x
        # Down y
        # Vertex 0 (-x,-y) -- Vertex 1 (x,-y)
        # Vertex 3 (-x,y)  -- Vertex 2 (x,y)
        mesh.vertices = o3d.utility.Vector3dVector(
            [[-w,-h,0],[w,-h,0],
             [w,h,0],[-w,h,0]])
        # Anti-clockwise direction (4 triangles to allow two-sided views)
        mesh.triangles = o3d.utility.Vector3iVector(
            [[0,2,1],[0,3,2],  # Front face
             [0,1,2],[0,2,3]]) # Back face
        # Define the uvs to match img coor to the order of triangles
        # Top left    (0,0) -- Top right    (1,0)
        # Bottom left (0,1) -- Bottom right (1,1)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            [[0,0],[1,1],[1,0], [0,0],[0,1],[1,1], # Front face
             [0,0],[1,0],[1,1], [0,0],[1,1],[0,1]]) # Back face
        # Image to be displayed
        mesh.textures = [o3d.geometry.Image(img)]

        num_face = np.asarray(mesh.triangles).shape[0]
        mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(num_face, dtype=np.int32))

        mesh.translate([w,h,0]) # Shift origin of image to top left corner
        mesh.scale(1/(2*w), [0,0,0]) # Scale down such image width = 1 unit in 3D space

        return mesh        


    def draw3d(self, param, img):
        for i in range(self.max_num_faces):
            if param[i]['detect']:
                param[i]['joint'][:,1] *= img.shape[0]/img.shape[1] # To match for scaling of mesh image
                param[i]['joint'][:,2] -= 0.05 # Shift face mask slightly forward
                self.mesh.vertices = o3d.utility.Vector3dVector(param[i]['joint'])
            else:
                self.mesh.vertices = o3d.utility.Vector3dVector(np.zeros((self.nPt,3)))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.mesh_img.textures = [o3d.geometry.Image(img)]
        
        self.vis.update_geometry(None)
        self.vis.poll_events()
        self.vis.update_renderer()

        # # Capture screen image
        # img = self.vis.capture_screen_float_buffer()
        # # Convert to OpenCV format
        # img = (np.asarray(img)*255).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # return img


class DisplayHand:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, max_num_hands=1, vis=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Define kinematic tree linking keypoint together to form skeleton
        self.ktree = [0,          # Wrist
                      0,1,2,3,    # Thumb
                      0,5,6,7,    # Index
                      0,9,10,11,  # Middle
                      0,13,14,15, # Ring
                      0,17,18,19] # Little

        # Define color for 21 keypoint
        self.color = [[0,0,0], # Wrist black
                      [255,0,0],[255,60,0],[255,120,0],[255,180,0], # Thumb
                      [0,255,0],[60,255,0],[120,255,0],[180,255,0], # Index
                      [0,255,0],[0,255,60],[0,255,120],[0,255,180], # Middle
                      [0,0,255],[0,60,255],[0,120,255],[0,180,255], # Ring
                      [0,0,255],[60,0,255],[120,0,255],[180,0,255]] # Little
        self.color = np.asarray(self.color)
        self.color_ = self.color / 255 # For Open3D RGB
        self.color[:,[0,2]] = self.color[:,[2,0]] # For OpenCV BGR
        self.color = self.color.tolist()


        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
            self.vis.get_render_option().point_size = 8.0
            joint = np.zeros((21,3))

            # Draw 21 joints
            self.pcd = []
            for i in range(max_num_hands):
                p = o3d.geometry.PointCloud()
                p.points = o3d.utility.Vector3dVector(joint)
                p.colors = o3d.utility.Vector3dVector(self.color_)
                self.pcd.append(p)
            
            # Draw 20 bones
            self.bone = []
            for i in range(max_num_hands):
                b = o3d.geometry.LineSet()
                b.points = o3d.utility.Vector3dVector(joint)
                b.colors = o3d.utility.Vector3dVector(self.color_[1:])
                b.lines  = o3d.utility.Vector2iVector(
                    [[0,1], [1,2],  [2,3],  [3,4],    # Thumb
                     [0,5], [5,6],  [6,7],  [7,8],    # Index
                     [0,9], [9,10], [10,11],[11,12],  # Middle
                     [0,13],[13,14],[14,15],[15,16],  # Ring
                     [0,17],[17,18],[18,19],[19,20]]) # Little
                self.bone.append(b)

            # Draw world reference frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

            # Add geometry to visualize
            self.vis.add_geometry(frame)
            for i in range(max_num_hands):
                self.vis.add_geometry(self.pcd[i])
                self.vis.add_geometry(self.bone[i])

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_up([0,-1,0]) # Set up as -y axis
            ctr.set_front([0,0,-1]) # Set to looking towards -z axis
            ctr.set_lookat([0.5,0.5,0]) # Set to center of view
            ctr.set_zoom(1)
            
            if draw_camera:
                # Remove previous frame
                self.vis.remove_geometry(frame)
                # Draw camera reference frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                # Draw camera frustum
                self.camera = DisplayCamera(self.vis, self.intrin)
                frustum = self.camera.create_camera_frustum()
                # Draw 2D image plane in 3D space
                self.mesh_img = self.camera.create_mesh_img()
                # Add geometry to visualize
                self.vis.add_geometry(frame)
                self.vis.add_geometry(frustum)
                self.vis.add_geometry(self.mesh_img)
                # Reset camera view
                self.camera.reset_view()


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        # Loop through different hands
        for p in param:
            if p['class'] is not None:
                # Label left or right hand
                x = int(p['keypt'][0,0]) - 30
                y = int(p['keypt'][0,1]) + 40
                # cv2.putText(img, '%s %.3f' % (p['class'], p['score']), (x, y), 
                cv2.putText(img, '%s' % (p['class']), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red
                
                # Loop through keypoint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), self.color[i], 2) 

                        # Draw keypoint
                        cv2.circle(img, (x, y), 5, self.color[i], -1)
                        # cv2.circle(img, (x, y), 3, self.color[i], -1)

                        # # Number keypoint
                        # cv2.putText(img, '%d' % (i), (x, y), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1, self.color[i])

                        # # Label visibility and presence
                        # cv2.putText(img, '%.1f, %.1f' % (p['visible'][i], p['presence'][i]),
                        #     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color[i])
                
		        # Label gesture
                if p['gesture'] is not None:
                    size = cv2.getTextSize(p['gesture'].upper(), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    x = int((img_width-size[0]) / 2)
                    cv2.putText(img, p['gesture'].upper(),
                        # (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                        (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    # Label joint angle
                    self.draw_joint_angle(img, p)

            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   

        return img


    def draw2d_(self, img, param):
        # Different from draw2d
        # draw2d_ draw 2.5D with relative depth info
        # The closer the landmark is towards the camera
        # The lighter and larger the circle

        img_height, img_width, _ = img.shape

        # Loop through different hands
        for p in param:
            if p['class'] is not None:
                # Extract wrist pixel
                x = int(p['keypt'][0,0]) - 30
                y = int(p['keypt'][0,1]) + 40
                # Label left or right hand
                cv2.putText(img, '%s %.3f' % (p['class'], p['score']), (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red

                min_depth = min(p['joint'][:,2])
                max_depth = max(p['joint'][:,2])

                # Loop through keypt and joint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Convert depth to color nearer white, further black
                        depth = (max_depth-p['joint'][i,2]) / (max_depth-min_depth)
                        color = [int(255*depth), int(255*depth), int(255*depth)]
                        size = int(10*depth)+2
                        # size = 2

                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), color, 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), size, color, size)
            
            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)                           

        return img


    def draw3d(self, param):
        for i in range(self.max_num_hands):
            if param[i]['class'] is None:
                self.pcd[i].points = o3d.utility.Vector3dVector(np.zeros((21,3)))
                self.bone[i].points = o3d.utility.Vector3dVector(np.zeros((21,3)))
            else:
                self.pcd[i].points = o3d.utility.Vector3dVector(param[i]['joint'])
                self.bone[i].points = o3d.utility.Vector3dVector(param[i]['joint'])


    def draw3d_(self, param, img=None):
        # Different from draw3d
        # draw3d_ draw the actual 3d joint in camera coordinate
        for i in range(self.max_num_hands):
            if param[i]['class'] is None:
                self.pcd[i].points = o3d.utility.Vector3dVector(np.zeros((21,3)))
                self.bone[i].points = o3d.utility.Vector3dVector(np.zeros((21,3)))
            else:
                self.pcd[i].points = o3d.utility.Vector3dVector(param[i]['joint_3d'])
                self.bone[i].points = o3d.utility.Vector3dVector(param[i]['joint_3d'])

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.mesh_img.textures = [o3d.geometry.Image(img)]            


    def draw_joint_angle(self, img, p):
        # Create text
        text = None
        if p['gesture']=='Finger MCP Flexion':
            text = 'Index : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f' % \
                (p['angle'][3], p['angle'][6], p['angle'][9], p['angle'][12])

        elif p['gesture']=='Finger PIP DIP Flexion':
            text = 'PIP: \
                  \nIndex : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f \
                  \nDIP: \
                  \nIndex : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f' % \
                (p['angle'][4], p['angle'][7], p['angle'][10], p['angle'][13],
                 p['angle'][5], p['angle'][8], p['angle'][11], p['angle'][14])

        elif p['gesture']=='Thumb MCP Flexion':
            text = 'Angle: %.1f' % p['angle'][1]

        elif p['gesture']=='Thumb IP Flexion':
            text = 'Angle: %.1f' % p['angle'][2]

        elif p['gesture']=='Thumb Radial Abduction':
            text = 'Angle: %.1f' % p['angle'][0]

        elif p['gesture']=='Thumb Palmar Abduction':
            text = 'Angle: %.1f' % p['angle'][0]

        elif p['gesture']=='Thumb Opposition':
            # Dist btw thumb and little fingertip
            dist = np.linalg.norm(p['joint'][4] - p['joint'][20])
            text = 'Dist: %.3f' % dist
        
        elif p['gesture']=='Forearm Neutral' or \
             p['gesture']=='Forearm Pronation' or \
             p['gesture']=='Forearm Supination' or \
             p['gesture']=='Wrist Flex/Extension' or \
             p['gesture']=='Wrist Radial/Ulnar Dev':
            text = 'Angle: %.1f' % p['angle'][0]

        if text is not None:
            x0 = 10 # Starting x coor for placing text
            y0 = 60 # Starting y coor for placing text
            dy = 25 # Change in text vertical spacing        
            vert = len(text.split('\n'))
            # Draw black background
            cv2.rectangle(img, (x0, y0), (140, y0+vert*dy+10), (0,0,0), -1)
            # Draw text
            for i, line in enumerate(text.split('\n')):
                y = y0 + (i+1)*dy
                cv2.putText(img, line,
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


    def draw_game_rps(self, img, param):
        img_height, img_width, _ = img.shape

        # Init result of 2 hands to none
        res = [None, None]

        # Loop through different hands
        for j, p in enumerate(param):
            # Only allow maximum of two hands
            if j>1:
                break

            if p['class'] is not None:                
                # Loop through keypoint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), self.color[i], 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), 5, self.color[i], -1)

                # Label gesture 
                text = None
                if p['gesture']=='fist':
                    text = 'rock'
                elif p['gesture']=='five':
                    text = 'paper'
                elif (p['gesture']=='three') or (p['gesture']=='yeah'):
                    text = 'scissor'
                res[j] = text

                # Label result
                if text is not None:
                    x = int(p['keypt'][0,0]) - 30
                    y = int(p['keypt'][0,1]) + 40
                    cv2.putText(img, '%s' % (text.upper()), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red

        # Determine winner
        text = None
        winner = None
        if res[0]=='rock':
            if res[1]=='rock'     : text = 'Tie'
            elif res[1]=='paper'  : text = 'Paper wins'  ; winner = 1
            elif res[1]=='scissor': text = 'Rock wins'   ; winner = 0
        elif res[0]=='paper':
            if res[1]=='rock'     : text = 'Paper wins'  ; winner = 0
            elif res[1]=='paper'  : text = 'Tie'
            elif res[1]=='scissor': text = 'Scissor wins'; winner = 1
        elif res[0]=='scissor':
            if res[1]=='rock'     : text = 'Rock wins'   ; winner = 1
            elif res[1]=='paper'  : text = 'Scissor wins'; winner = 0
            elif res[1]=='scissor': text = 'Tie'

        # Label gesture
        if text is not None:
            size = cv2.getTextSize(text.upper(), 
                # cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = int((img_width-size[0]) / 2)
            cv2.putText(img, text.upper(),
                # (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)            

        # Draw winner text
        if winner is not None:
            x = int(param[winner]['keypt'][0,0]) - 30
            y = int(param[winner]['keypt'][0,1]) + 80
            cv2.putText(img, 'WINNER', (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) # Yellow            

        return img


class DisplayBody:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, vis=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin  

        # Define kinematic tree linking 33 keypoint together to form skeleton
        self.ktree = [
            0,     # Nose
            0,1,2, # Left eye
            0,4,5, # Right eye
            3,6,   # Ear
            10,9,  # Mouth
            23,11,11,12,13,14, # Upper body and arm
            15,16,17,18,15,16, # Hand
            24,12, # Torso
            23,24,25,26,27,28,27,28] # Leg

        # Define color for 33 keypoint
        self.color = [[60,0,255], # Nose
                      [60,0,255],[120,0,255],[180,0,255], # Left eye
                      [60,0,255],[120,0,255],[180,0,255], # Right eye
                      [240,0,255],[240,0,255], # Ear
                      [255,0,255],[255,0,255], # Mouth
                      [0,255,0],[60,255,0],[255,60,0],[0,255,60],[255,120,0],[0,255,120],
                      [255,180,0],[0,255,180],[255,240,0],[0,255,240],[255,255,0],[0,255,255],
                      [0,120,255],[0,180,255], # Torso
                      [255,60,0],[0,255,60],[255,120,0],[0,255,120],
                      [255,180,0],[0,255,180],[255,240,0],[0,255,240]]
        self.color = np.asarray(self.color)
        self.color_ = self.color / 255 # For Open3D RGB
        self.color[:,[0,2]] = self.color[:,[2,0]] # For OpenCV BGR
        self.color = self.color.tolist()

        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
                self.vis.get_render_option().point_size = 8.0
            joint = np.zeros((33,3))

            # Draw joints
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(joint)
            self.pcd.colors = o3d.utility.Vector3dVector(self.color_)
            
            # Draw bones
            self.bone = o3d.geometry.LineSet()
            self.bone.points = o3d.utility.Vector3dVector(joint)
            self.bone.colors = o3d.utility.Vector3dVector(self.color_[1:])
            bone_connections = [[0,1],  [1,2], [2,3],    # Left eye
                                [0,4],  [4,5], [5,6],    # Right eye
                                [3,7],  [6,8],           # Ear
                                [9,10], [9,10],          # Mouth
                                [11,23],[11,12],[11,13],[12,14],[13,15],[14,16], # Upper body and arm
                                [15,17],[16,18],[17,19],[18,20],[15,21],[16,22], # Hand
                                [23,24],[12,24], # Torso
                                [23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[27,31],[28,32]] # Leg
            self.bone.lines  = o3d.utility.Vector2iVector(bone_connections)

            # Draw world reference frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

            # Add geometry to visualize
            self.vis.add_geometry(frame)
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.bone)

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_up([0,-1,0]) # Set up as -y axis
            ctr.set_front([0,0,-1]) # Set to looking towards -z axis
            ctr.set_lookat([0.5,0.5,0]) # Set to center of view
            ctr.set_zoom(1)

            if draw_camera:
                # Draw camera frustum
                self.camera = DisplayCamera(self.vis, self.intrin)
                frustum = self.camera.create_camera_frustum(depth=[1,2])
                # Draw 2D image plane in 3D space
                self.mesh_img = self.camera.create_mesh_img(depth=2)
                # Add geometry to visualize
                self.vis.add_geometry(frustum)
                self.vis.add_geometry(self.mesh_img)
                # Reset camera view
                self.camera.reset_view()


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        p = param
        if p['detect']:
            # Loop through keypoint for body
            for i in range(33):
                x = int(p['keypt'][i,0])
                y = int(p['keypt'][i,1])
                if x>0 and y>0 and x<img_width and y<img_height:
                    # Draw skeleton
                    start = p['keypt'][self.ktree[i],:]
                    x_ = int(start[0])
                    y_ = int(start[1])
                    if x_>0 and y_>0 and x_<img_width and y_<img_height:
                        cv2.line(img, (x_, y_), (x, y), self.color[i], 2)

                    # Draw keypoint
                    cv2.circle(img, (x, y), 3, self.color[i], -1)

                    # Number keypoint
                    # cv2.putText(img, '%d' % (i), (x, y), 
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color[i])

                    # Label visibility and presence
                    # cv2.putText(img, '%.1f, %.1f' % (p['visible'][i], p['presence'][i]),
                    #     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color[i])

        # Label fps
        if p['fps']>0:
            cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   

        return img


    def draw2d_(self, img, param):
        # Different from draw2d
        # draw2d_ draw 2.5D with relative depth info
        # The closer the landmark is towards the camera
        # The lighter and larger the circle

        img_height, img_width, _ = img.shape

        # Loop through different hands
        p = param
        if p['detect']:
            min_depth = min(p['joint'][:,2])
            max_depth = max(p['joint'][:,2])

            # Loop through keypt and joint for body hand
            for i in range(33):
                x = int(p['keypt'][i,0])
                y = int(p['keypt'][i,1])
                if x>0 and y>0 and x<img_width and y<img_height:
                    # Convert depth to color nearer white, further black
                    depth = (max_depth-p['joint'][i,2]) / (max_depth-min_depth)
                    color = [int(255*depth), int(255*depth), int(255*depth)]
                    size = int(5*depth)+2

                    # Draw skeleton
                    start = p['keypt'][self.ktree[i],:]
                    x_ = int(start[0])
                    y_ = int(start[1])
                    if x_>0 and y_>0 and x_<img_width and y_<img_height:
                        cv2.line(img, (x_, y_), (x, y), color, 2)

                    # Draw keypoint
                    cv2.circle(img, (x, y), size, color, size)
            
        # Label fps
        if p['fps']>0:
            cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)                           

        return img


    def draw3d(self, param):
        if param['detect']:
            self.pcd.points = o3d.utility.Vector3dVector(param['joint'])
            self.bone.points = o3d.utility.Vector3dVector(param['joint'])
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.zeros((33,3)))
            self.bone.points = o3d.utility.Vector3dVector(np.zeros((33,3)))


    def draw3d_(self, param, img=None):
        # Different from draw3d
        # draw3d_ draw the actual 3d joint in camera coordinate
        if param['detect']:
            # Translate all joint_3d forward by 1 m
            param['joint_3d'][:,2] += 1.0             
            self.pcd.points = o3d.utility.Vector3dVector(param['joint_3d'])
            self.bone.points = o3d.utility.Vector3dVector(param['joint_3d'])
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.zeros((33,3)))
            self.bone.points = o3d.utility.Vector3dVector(np.zeros((33,3)))

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.mesh_img.textures = [o3d.geometry.Image(img)]


class DisplayHolistic:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, vis=None):
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:     
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
                self.vis.get_render_option().point_size = 8.0

            self.disp_face = DisplayFace(draw3d=True, vis=self.vis, intrin=self.intrin)
            self.disp_hand = DisplayHand(draw3d=True, vis=self.vis, intrin=self.intrin,
                max_num_hands=2)
            self.disp_body = DisplayBody(draw3d=True, vis=self.vis, intrin=self.intrin)

            if draw_camera:
                # Draw camera frustum
                self.camera = DisplayCamera(self.vis, self.intrin)
                frustum = self.camera.create_camera_frustum(depth=[1,2])
                # Draw 2D image plane in 3D space
                self.mesh_img = self.camera.create_mesh_img(depth=2)
                # Add geometry to visualize
                self.vis.add_geometry(frustum)
                self.vis.add_geometry(self.mesh_img)
                # Reset camera view
                self.camera.reset_view()

        else:
            self.disp_face = DisplayFace(draw3d=False)
            self.disp_hand = DisplayHand(draw3d=False, max_num_hands=2)
            self.disp_body = DisplayBody(draw3d=False, draw_camera=False)


    def draw2d(self, img, param):
        param_fc, param_lh, param_rh, param_bd = param
        self.disp_face.draw2d(img, [param_fc])
        self.disp_body.draw2d(img, param_bd)
        self.disp_hand.draw2d(img, [param_lh, param_rh])

        return img


    def draw2d_(self, img, param):
        param_fc, param_lh, param_rh, param_bd = param
        self.disp_face.draw2d_(img, [param_fc])
        self.disp_body.draw2d_(img, param_bd)
        self.disp_hand.draw2d_(img, [param_lh, param_rh])

        return img


    def draw3d(self, param):
        param_fc, param_lh, param_rh, param_bd = param
        self.disp_face.draw3d([param_fc])
        self.disp_hand.draw3d([param_lh, param_rh])
        self.disp_body.draw3d(param_bd)


    def draw3d_(self, param, img):
        # Different from draw3d
        # draw3d_ draw the actual 3d joint in camera coordinate
        param_fc, param_lh, param_rh, param_bd = param
        # Note: Collapse body hand joint as there is full hand joint from hand
        param_bd['joint_3d'][[17,19,21]] = param_bd['joint_3d'][15]
        param_bd['joint_3d'][[18,20,22]] = param_bd['joint_3d'][16]
        # Note: Collapse body face joint as there is full face mesh
        param_bd['joint_3d'][[0,1,2,3,4,5,6,7,8,9,10]] = np.zeros(3)
        # Translate all joint_3d forward by 1 m
        param_fc['joint_3d'][:,2] += 1.0 
        param_lh['joint_3d'][:,2] += 1.0 
        param_rh['joint_3d'][:,2] += 1.0 
        # param_bd['joint_3d'][:,2] += 1.0 
        self.disp_face.draw3d_([param_fc])
        self.disp_hand.draw3d_([param_lh, param_rh])
        self.disp_body.draw3d_(param_bd)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.mesh_img.textures = [o3d.geometry.Image(img)]


class DisplayCamera:
    def __init__(self, vis, intrin=None):
        self.vis = vis
        
        # Get camera intrinsics param
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin            
        
        # Note: Need to subtract optical center by 0.5
        # https://github.com/intel-isl/Open3D/issues/727
        self.intrin['cx'] -= 0.5
        self.intrin['cy'] -= 0.5

        # For reset_view
        self.pinhole = o3d.camera.PinholeCameraParameters()
        self.pinhole.extrinsic = np.eye(4)
        self.pinhole.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intrin['width'], self.intrin['height'], 
            self.intrin['fx'], self.intrin['fy'], 
            self.intrin['cx'], self.intrin['cy'])            


    def reset_view(self):
        # Reset camera view to this camera
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
            self.pinhole)


    def unproject_pt(self, u, v, depth):
        # Transform 2D pixels to 3D points
        # Given pixel coordinates and depth in an image
        # with no distortion or inverse distortion coefficients
        # compute the corresponding point in 3D space relative to the same camera
        x = (u - self.intrin['cx'])/self.intrin['fx']*depth
        y = (v - self.intrin['cy'])/self.intrin['fy']*depth
        z = depth

        return [x, y, z]


    def create_camera_frustum(self, depth=[0.5,1.0]):
        # Get camera intrinsics param
        w  = self.intrin['width']
        h  = self.intrin['height']

        # Each frustum 8 lines, each line 2 pts, and plot at 2 different depths
        points, lines = [], []
        c = 0 # Counter
        points.append([0,0,0]) # Origin
        for d in depth: # Plot at different depth
            points.append(self.unproject_pt(0, 0, d)) # Top left
            points.append(self.unproject_pt(w, 0, d)) # Top right
            points.append(self.unproject_pt(w, h, d)) # Bottom left
            points.append(self.unproject_pt(0, h, d)) # Bottom right
            lines.append([0,c+1]);   lines.append([0,c+2])
            lines.append([0,c+3]);   lines.append([0,c+4])
            lines.append([c+1,c+2]); lines.append([c+2,c+3])
            lines.append([c+3,c+4]); lines.append([c+4,c+1])
            c += 4
            
        # Set to uniform light gray color
        colors = [[0.75,0.75,0.75] for i in range(len(lines))]

        line = o3d.geometry.LineSet()
        line.lines  = o3d.utility.Vector2iVector(lines)
        line.points = o3d.utility.Vector3dVector(points)
        line.colors = o3d.utility.Vector3dVector(colors)

        return line


    def create_mesh_img(self, img=None, depth=1.0):
        # Get camera intrinsics param
        w  = self.intrin['width']
        h  = self.intrin['height']

        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)

        vert = []
        vert.append(self.unproject_pt( 0, 0, depth)) # Top left
        vert.append(self.unproject_pt( w, 0, depth)) # Top right
        vert.append(self.unproject_pt( w, h, depth)) # Bottom left
        vert.append(self.unproject_pt( 0, h, depth)) # Bottom right

        mesh = o3d.geometry.TriangleMesh()
        # Convention of 4 vertices
        # --> right x
        # Down y
        # Vertex 0 (-x,-y) -- Vertex 1 (x,-y)
        # Vertex 3 (-x,y)  -- Vertex 2 (x,y)
        mesh.vertices = o3d.utility.Vector3dVector(vert)
        # Anti-clockwise direction (4 triangles to allow two-sided views)
        mesh.triangles = o3d.utility.Vector3iVector(
            [[0,2,1],[0,3,2],  # Front face
             [0,1,2],[0,2,3]]) # Back face
        # Define the uvs to match img coor to the order of triangles
        # Top left    (0,0) -- Top right    (1,0)
        # Bottom left (0,1) -- Bottom right (1,1)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            [[0,0],[1,1],[1,0], [0,0],[0,1],[1,1], # Front face
             [0,0],[1,0],[1,1], [0,0],[1,1],[0,1]]) # Back face
        # Image to be displayed
        mesh.textures = [o3d.geometry.Image(img)]

        num_face = np.asarray(mesh.triangles).shape[0]
        mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(num_face, dtype=np.int32))

        return mesh


class DisplayObjectron:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, max_num_objects=1, vis=None):
        self.max_num_objects = max_num_objects
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # For drawing projected axis on 2D image
        self.intrin_mat = np.asarray([
            [self.intrin['fx'], 0, self.intrin['cx']],
            [0, self.intrin['fy'], self.intrin['cy']],
            [0, 0, 1]])      

        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
                self.vis.get_render_option().point_size = 3.0

            point = np.zeros((9,3))
            color = [[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]] # Note: Max 5 colors RGB
            self.box  = []
            self.axis = []
            for i in range(max_num_objects):
                # Draw bounding box
                b = o3d.geometry.LineSet()
                b.points = o3d.utility.Vector3dVector(point)
                b.lines  = o3d.utility.Vector2iVector(    
                    [(1, 2), (1, 3), (1, 5), (2, 4),
                     (2, 6), (3, 4), (3, 7), (4, 8),
                     (5, 6), (5, 7), (6, 8), (7, 8)])
                b.paint_uniform_color(color[i])
                self.box.append(b)

                # Draw object frame
                a = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                self.axis.append(a)

            # Add geometry to visualize
            for b in self.box:
                self.vis.add_geometry(b)
            for a in self.axis:
                self.vis.add_geometry(a)

            if draw_camera:
                # Draw camera reference frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # Draw camera frustum
                self.camera = DisplayCamera(self.vis, self.intrin)
                frustum = self.camera.create_camera_frustum()
                # Draw 2D image plane in 3D space
                self.mesh_img = self.camera.create_mesh_img()
                # Add geometry to visualize
                self.vis.add_geometry(frame)
                self.vis.add_geometry(frustum)
                self.vis.add_geometry(self.mesh_img)
                # Reset camera view
                self.camera.reset_view()                


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        # Loop through different objects
        color = [[0,0,255],[0,255,0],[255,0,0],[255,0,255],[255,255,0]] # Note: Max 5 colors BGR
        for j, p in enumerate(param):
            if p['detect']:
                # Draw bounding box
                for connect in BOX_CONNECTIONS:
                    x = int(p['landmarks_2d'][connect[0],0])
                    y = int(p['landmarks_2d'][connect[0],1])
                    x_= int(p['landmarks_2d'][connect[1],0])
                    y_= int(p['landmarks_2d'][connect[1],1])
                    if x_>0 and y_>0 and x_<img_width and y_<img_height and \
                       x >0 and y >0 and x <img_width and y <img_height:
                        cv2.line(img, (x_, y_), (x, y), color[j], 1)

                # Draw center of object
                x = int(p['landmarks_2d'][0,0])
                y = int(p['landmarks_2d'][0,1])
                if x>0 and y>0 and x<img_width and y<img_height:
                    cv2.circle(img, (x, y), 2, (0,255,255), -1)
                    cv2.line(img, (x-5, y), (x+5, y), (0,255,255), 2)
                    cv2.line(img, (x, y-5), (x, y+5), (0,255,255), 2)

                # Draw 2d landmarks
                for i in range(1,9):
                    x = int(p['landmarks_2d'][i,0])
                    y = int(p['landmarks_2d'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        cv2.circle(img, (x, y), 4, color[j], -1)

                # Draw projected axis
                axis_3D = np.float32([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]) * 0.05 # Create a 3D axis with length of 5 cm
                rvec = cv2.Rodrigues(p['rotation'])[0]
                tvec = p['translation']
                mat  = self.intrin_mat
                dist = np.zeros(4) # Distortion coeff
                axis_2D = cv2.projectPoints(axis_3D, rvec, tvec, mat, dist)[0].reshape(-1, 2)
                if axis_2D.shape[0]==4:
                    color = [(0,0,255),(0,255,0),(255,0,0)] # BGR
                    for i in range(1,4):
                        (x0, y0), (x1, y1) = axis_2D[0], axis_2D[i]
                        cv2.arrowedLine(img, (int(x0), int(y0)), (int(x1), int(y1)), color[i-1], 2)                 

            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   

        return img


    def draw3d(self, param, img=None):
        for i, b in enumerate(self.box):
            if param[i]['detect']:
                b.points = o3d.utility.Vector3dVector(param[i]['landmarks_3d'])
            else:
                b.points = o3d.utility.Vector3dVector(np.zeros((9,3)))

        for i, a in enumerate(self.axis):
            if param[i]['detect']:
                a.rotate(param[i]['rotation'], center=(0,0,0))
                a.translate(param[i]['translation'])

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.mesh_img.textures = [o3d.geometry.Image(img)]                  


# Adapted from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py
FACE_CONNECTIONS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])  

# Adapted from https://github.com/google/mediapipe/blob/350fbb2100ad531bc110b93aaea23d96af5a5064/mediapipe/python/solutions/objectron.py
BOX_CONNECTIONS = frozenset([
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 4),
    (3, 7),
    (4, 8),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 8),
])
