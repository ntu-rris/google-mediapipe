###############################################################################
### Useful function for visualization of hand and upper body tracking
###############################################################################

import cv2
import numpy as np
import open3d as o3d


class DisplayHand:
    def __init__(self, draw3d=False, max_num_hands=2):
        super(DisplayHand, self).__init__()
        self.max_num_hands = max_num_hands

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
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=640, height=480)
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
            ctr.set_zoom(1.5)
            # ctr.set_up([0,-1,0]) # Set up as -y axis
            # ctr.set_front([1,0,0]) # Set to looking towards x axis
            # ctr.set_lookat([0,0,0.5]) # Set to center of view
            # ctr.set_zoom(0.5)


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
                # self.pcd[i].points = o3d.utility.Vector3dVector(param[i]['joint_3d'])
                # self.bone[i].points = o3d.utility.Vector3dVector(param[i]['joint_3d'])

        self.vis.update_geometry(None)
        self.vis.poll_events()
        self.vis.update_renderer()


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


    def drawGameRPS(self, img, param):
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
    def __init__(self, draw3d=False):
        super(DisplayBody, self).__init__()

        # Define kinematic tree linking keypoint together to form skeleton
        self.ktree = [
            # Face
            0,       # Nose
            2,0,2,   # Right eye
            5,0,5,   # Left eye
            2,5,     # Ear
            10,9,    # Mouth
            # Arm
            23,11,11,12,13,14, # Upper and lower arm
            15,16,17,18,15,16, # Hand
            # Torso
            24,12]

        # Define color for 25 keypoint
        self.color = [[60,0,255], # Nose
                      [60,0,255],[120,0,255],[180,0,255], # Right eye
                      [60,0,255],[120,0,255],[180,0,255], # Left eye
                      [240,0,255],[240,0,255], # Ear
                      [255,0,255],[255,0,255], # Mouth
                      [0,255,0],[60,255,0],[255,60,0],[0,255,60],[255,120,0],[0,255,120],
                      [255,180,0],[0,255,180],[255,240,0],[0,255,240],[255,255,0],[0,255,255],
                      [0,120,255],[0,180,255]]
        self.color = np.asarray(self.color)
        self.color_ = self.color / 255 # For Open3D RGB
        self.color[:,[0,2]] = self.color[:,[2,0]] # For OpenCV BGR
        self.color = self.color.tolist()


        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=640, height=480)
            self.vis.get_render_option().point_size = 8.0
            joint = np.zeros((25,3))

            # Draw 25 joints
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(joint)
            self.pcd.colors = o3d.utility.Vector3dVector(self.color_)
            
            # Draw 20 bones
            self.bone = o3d.geometry.LineSet()
            self.bone.points = o3d.utility.Vector3dVector(joint)
            self.bone.colors = o3d.utility.Vector3dVector(self.color_[1:])
            self.bone.lines  = o3d.utility.Vector2iVector(
                [[0,2],  [1,2], [2,3],    # Right eye
                 [0,5],  [4,5], [5,6],    # Left eye
                 [2,7],  [6,8],           # Ear
                 [9,10], [9,10],          # Mouth
                 [11,23],[11,12],[11,13],[12,14],[13,15],[14,16],
                 [15,17],[16,18],[17,19],[18,20],[15,21],[16,22],
                 [23,24],[12,24]])

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
            ctr.set_zoom(1.5)


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        p = param
        if p['detect']:
            # Loop through keypoint for upper body
            for i in range(25):
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



    def draw3d(self, param):

        if param['detect']:
            self.pcd.points = o3d.utility.Vector3dVector(param['joint'])
            self.bone.points = o3d.utility.Vector3dVector(param['joint'])
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.zeros((25,3)))
            self.bone.points = o3d.utility.Vector3dVector(np.zeros((25,3)))

        self.vis.update_geometry(None)
        self.vis.poll_events()
        self.vis.update_renderer()
