import numpy as np
import skeleton_utils
from skeleton_utils import normalize


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




class Skeleton(object):
    """ Class for calculating joint rotations from skeleton keypoints """
    def __init__(self, joint_names, parents, root_joint="HipCenter"):
        self.joints = joint_names
        self.parents = parents
        self.root_joint = root_joint
        self.root_define_joints = ['HipLeft', 'Spine']
        
    def load_keypoints(self, keypoints):
        self.keypoints = keypoints
        self.joint_hierarchy = {}
        self.bone_lengths = {}
        self.offset_directions = {}
        self.base_skeleton = {}
        self.offset_directions = {}
        self.joint_rotations = []
        self.__calculate_joint_hierarchy()
        self.__calculate_offset_directions()
        self.__calculate_bone_lengths()
        self.__get_base_skeleton()
        
    def __calculate_joint_hierarchy(self):
        """ Calculate joint hierarchy as path from root to the joint"""
        for joint_idx, joint in enumerate(self.joints):
            if joint not in self.joint_hierarchy:
                self.joint_hierarchy[joint] = []
            parent = self.parents[joint_idx]
            while parent != -1:
                self.joint_hierarchy[joint].append(self.joints[parent])
                parent = self.parents[parent]
        # print(self.joint_hierarchy)
        
    def __calculate_offset_directions(self):
        """ Calculate offset direction of joint wrt it's parent joint """
        first_kpt = self.keypoints[0]
        for joint_idx, joint in enumerate(self.joints):
            if joint == self.root_joint:
                continue
            joint_coor = first_kpt[joint_idx]
            parent_idx = self.parents[joint_idx]
            parent_coor = first_kpt[parent_idx]
            direction = joint_coor - parent_coor
            self.offset_directions[joint] = normalize(direction)
                
    def __calculate_bone_lengths(self):
        """ Calculate bone lengths as median of bone lengths of all frames """
        for joint_idx, joint in enumerate(self.joints):
            if joint == self.root_joint:
                continue
            # parent = self.joint_hierarchy[joint][0]
            parent_idx = self.parents[joint_idx]

            joint_kpts = self.keypoints[:, joint_idx]
            parent_kpts = self.keypoints[:, parent_idx]

            _bone = joint_kpts - parent_kpts
            _bone_lengths = np.linalg.norm(_bone, axis = -1)

            _bone_length = np.median(_bone_lengths)
            self.bone_lengths[joint] = _bone_length
        
    def __get_base_skeleton(self):
        
        body_lengths = self.bone_lengths
        
        self.base_skeleton[self.root_joint] = np.array([0,0,0])
        
        def _set_length(joint_type):
            self.base_skeleton[joint_type + 'Left'] = self.offset_directions[joint_type + 'Left'] * ((body_lengths[joint_type + 'Left'] + body_lengths[joint_type + 'Right']) / 2)
            self.base_skeleton[joint_type + 'Right'] = self.offset_directions[joint_type + 'Right'] * ((body_lengths[joint_type + 'Left'] + body_lengths[joint_type + 'Right']) / 2)

        _set_length('Hip')
        _set_length('Ankle')
        _set_length('Knee')
        _set_length('Foot')
        _set_length('Shoulder')
        _set_length('Elbow')
        _set_length('Wrist')
        _set_length('Hand')
        
        self.base_skeleton['Head'] = self.offset_directions['Head'] * (body_lengths['Head'])
        self.base_skeleton['Spine'] = self.offset_directions['Spine'] * (body_lengths['Spine'])
        self.base_skeleton['ShoulderCenter'] = self.offset_directions['ShoulderCenter'] * (body_lengths['ShoulderCenter'])
        
        
    def get_hips_position_and_rotation(self, frame_pos):
        
        #root position is saved directly
        root_position = frame_pos[self.root_joint]

        #calculate unit vectors of root joint
        root_u = frame_pos[self.root_define_joints[0]] - frame_pos[self.root_joint]
        root_u = normalize(root_u)
        root_v = frame_pos[self.root_define_joints[1]] - frame_pos[self.root_joint]
        root_v = normalize(root_v)
        root_w = np.cross(root_u, root_v)

        #Make the rotation matrix
        C = np.array([root_u, root_v, root_w]).T
        thetaz,thetay, thetax = skeleton_utils.Decompose_R_ZXY(C)
        root_rotation = np.array([thetaz, thetax, thetay])

        return root_position, root_rotation
    
    def get_joint_rotations(self, joint_name, frame_rotations, frame_pos):
    
        _invR = np.eye(3)
        for i, parent_name in enumerate(self.joint_hierarchy[joint_name]):
            if i == 0: continue
            _r_angles = frame_rotations[parent_name]
            R = skeleton_utils.get_R_z(_r_angles[0]) @ skeleton_utils.get_R_x(_r_angles[1]) @ skeleton_utils.get_R_y(_r_angles[2])
            _invR = _invR@R.T

        b = _invR @ (frame_pos[joint_name] - frame_pos[self.joint_hierarchy[joint_name][0]])

        _R = skeleton_utils.Get_R2(self.offset_directions[joint_name], b)
        tz, ty, tx = skeleton_utils.Decompose_R_ZXY(_R)
        joint_rs = np.array([tz, tx, ty])

        return joint_rs
    
    def calculate_joint_angles(self):
    

        total_frames = self.keypoints.shape[0]
        for frame_idx in range(total_frames):

            #get the keypoints positions in the current frame
            frame_pos = {}
            for joint_idx, joint in enumerate(self.joints):
                frame_pos[joint] = self.keypoints[frame_idx][joint_idx]

            root_position, root_rotation = self.get_hips_position_and_rotation(frame_pos)

            frame_rotations = {self.root_joint : root_rotation}

            #center the body pose
            for joint in self.joints:
                frame_pos[joint] = frame_pos[joint] - root_position

            #get the max joints connectsion
            max_connected_joints = 0
            for joint in self.joints:
                if len(self.joint_hierarchy[joint]) > max_connected_joints:
                    max_connected_joints = len(self.joint_hierarchy[joint])

            depth = 2
            while(depth <= max_connected_joints):
                for joint in self.joints:
                    if len(self.joint_hierarchy[joint]) == depth:
                        joint_rs = self.get_joint_rotations(joint, frame_rotations, frame_pos)
                        parent = self.joint_hierarchy[joint][0]
                        frame_rotations[parent] = joint_rs
                depth += 1

            for joint in frame_rotations:
                frame_rotations[joint] = np.array(frame_rotations[joint])
                
            self.joint_rotations.append(frame_rotations)
        
    def get_rotation_chain(self, joint, hierarchy, frame_rotations):
    
        hierarchy = hierarchy[::-1]

        #this code assumes ZXY rotation order
        R = np.eye(3)
        for parent in hierarchy:
            angles = frame_rotations[parent]
            _R = skeleton_utils.get_R_z(angles[0])@skeleton_utils.get_R_x(angles[1])@skeleton_utils.get_R_y(angles[2])
            R = R @ _R

        return R
    
    def draw_skeleton_from_joint_angles(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        total_frames = self.keypoints.shape[0]
        for frame_idx in range(total_frames):

            #get a dictionary containing the rotations for the current frame
            frame_rotations = self.joint_rotations[frame_idx]

            #for plotting
            for idx, _j in enumerate(self.joints):
                if _j == self.root_joint: 
                    continue

                #get hierarchy of how the joint connects back to root joint
                hierarchy = self.joint_hierarchy[_j]
                root_joint_idx = self.joints.index(self.root_joint)
                #get the current position of the parent joint
                r1 = self.keypoints[frame_idx][root_joint_idx]
                for parent in hierarchy:
                    if parent == self.root_joint: 
                        continue
                    R = self.get_rotation_chain(parent, self.joint_hierarchy[parent], frame_rotations)
                    r1 = r1 + R @ self.base_skeleton[parent]

                #get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
                r2 = r1 + self.get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ self.base_skeleton[_j]
                plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'red')
                ax.text(r2[0], r2[1], r2[2], str(idx + 1))
                # plt.scatter(*r1, color='b')
                # plt.scatter(*r2, color='b')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.azim = -90
            ax.elev = 90
            ax.set_title('Pose from joint angles')
            # ax.set_xlim3d(-4, 4)
            ax.set_xlabel('x')
            # ax.set_ylim3d(-4, 4)
            ax.set_ylabel('y')
            # ax.set_zlim3d(-4, 4)
            ax.set_zlabel('z')
            plt.pause(1/30)
            ax.cla()
        plt.close()
    
    def get_joint_position_frames(self, joint_rotations_frames, root_point):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        result = []
        total_frames = len(joint_rotations_frames)
        for frame_idx in range(total_frames):
            current_frame = [root_point[frame_idx]]
            frame_rotations = joint_rotations_frames[frame_idx]

            #for plotting
            for idx, _j in enumerate(self.joints):
                if _j == self.root_joint: 
                    continue

                #get hierarchy of how the joint connects back to root joint
                hierarchy = self.joint_hierarchy[_j]
                r1 = root_point[idx]
                for parent in hierarchy:
                    if parent == self.root_joint: 
                        continue
                    R = self.get_rotation_chain(parent, self.joint_hierarchy[parent], frame_rotations)
                    r1 = r1 + R @ self.base_skeleton[parent]

                #get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
                r2 = r1 + self.get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ self.base_skeleton[_j]
                current_frame.append(r2)
            result.append(current_frame)
        
        return np.array(result)
    
    def get_joint_position_frames_1(self, joint_rotations, root_points):
        joint_rotations_frames = []
        for rotation in joint_rotations:
            current_frame = {}
            i = 0
            for joint in self.joints:
                if joint == "Head" or joint.startswith("Foot") or joint.startswith("Hand"):
                    continue
                current_frame[joint] = rotation[i]
                i += 1
            
            joint_rotations_frames.append(current_frame)
        
        return self.get_joint_position_frames(joint_rotations_frames, root_points)
    
    def draw_skeleton_from_joint_coordinates(self):
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        total_frames = self.keypoints.shape[0]
        for frame_idx in range(total_frames):
            # print(framenum)
            # if framenum%2 == 0: continue #skip every 2nd frame

            for _j in self.joints:
                if _j == self.root_joint: 
                    continue
                _p = self.joint_hierarchy[_j][0] #get the name of the parent joint
                _p_idx = self.joints.index(_p)
                _j_idx = self.joints.index(_j)
                r1 = self.keypoints[frame_idx][_p_idx]
                r2 = self.keypoints[frame_idx][_j_idx]
                plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'blue')

            #ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.azim = -90
            ax.elev = 90
            ax.set_title('Pose from joint angles')
            # ax.set_xlim3d(-4, 4)
            ax.set_xlabel('x')
            # ax.set_ylim3d(-4, 4)
            ax.set_ylabel('y')
            # ax.set_zlim3d(-4, 4)
            ax.set_zlabel('z')
            plt.pause(1/30)
            ax.cla()
        plt.close()
        