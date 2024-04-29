import numpy as np
import os
import sys
from torch.utils.data import Dataset
import pickle
from skeleton import Skeleton

from scipy.spatial.transform import Rotation as R

def euler_to_quat(euler_angles):
    return R.from_euler('zxy', euler_angles).as_quat()


class MotionClipsDataset(Dataset):
    def __init__(self, motion_files_dir, motion_lengths_file, skeleton, type, clip_size=192, masked_window=64):
        self.data = []
        self.type = type
        self.clip_size = clip_size
        self.masked_window = masked_window
        self.kinect_dir = motion_files_dir
        self.skeleton = skeleton

        motion_lengths = np.loadtxt(
            motion_lengths_file, delimiter=",", dtype=str)
        for adavu_file, frame_count in motion_lengths:
            frame_count = int(frame_count)
            clip_start_indices = np.arange(1, frame_count - clip_size + 1, clip_size)
            self.data += [(adavu_file, idx) for idx in clip_start_indices]

    def __len__(self):
        return len(self.data)

    # def get_initial_pose

    def get_joint_parameters(self):
        joint_rotations = []
        for rotations in self.skeleton.joint_rotations:
            current_frame = []
            for joint in self.skeleton.joints:
                if joint in rotations:
                    current_frame.append(rotations[joint])
            joint_rotations.append(current_frame[1:])
        
        return np.array(joint_rotations)[1:]

    def get_root_parameter(self):
        hip_positions = np.array([positions[0] for positions in self.skeleton.keypoints])
        # delta_t = 30
        hip_velocity = (hip_positions[:-1] - hip_positions[1:]) 
        hip_rotations = np.array(
            [rotation["HipCenter"] for rotation in self.skeleton.joint_rotations][1:]
        )
        # hip_rotations = hip_rotations
        return np.hstack((hip_velocity, hip_rotations))

    def get_parameters(self, path, start_idx):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # joint_keypoints = self.get_joint_keypoints(data, start_idx-1, start_idx + self.clip_size)
        joint_keypoints = data[start_idx-1 : start_idx + self.clip_size]
        self.skeleton.load_keypoints(joint_keypoints)
        self.skeleton.calculate_joint_angles()

        if self.type == 'J':
            return self.get_joint_parameters()
        if self.type == 'R':
            return self.get_root_parameter()

    def get_joint_position_frames(self, idx):
        adavu_file, start_frame_idx = self.data[idx]
        adavu_file_path = os.path.join(self.kinect_dir, adavu_file)

        with open(adavu_file_path, 'rb') as f:
            frames = pickle.load(f)
            
        return frames, adavu_file

    def get_joint_keypoints(self, frames, start_frame_idx, end_frame_idx):
        joint_position_frames = []
        for frame in frames[start_frame_idx: end_frame_idx]:
            pose = [list(joint.values()) for joint in frame['joints'].values()]
            joint_position_frames.append(pose)
        return np.array(joint_position_frames)

    def __getitem__(self, idx):
        adavu_file, start_frame_idx = self.data[idx]
        adavu_file_path = os.path.join(self.kinect_dir, adavu_file)
        frames = self.get_parameters(adavu_file_path, start_frame_idx)
        frames = frames.astype(np.float32)
        clip1 = frames[:self.clip_size // 2]
        clip2 = frames[self.clip_size // 2:]
        return clip1, clip2


if __name__ == '__main__':
    joints = [
        'HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 
        'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight','ElbowRight', 
        'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 
        'HipRight', 'KneeRight', 'AnkleRight', 'FootRight'
    ]
    joint_parent_indices = [-1,  0,  1,  2,  2,  4,  5,  6,  2,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18]
    skeleton = Skeleton(joints, joint_parent_indices)

    dataset_path = r"data"
    kinect_dir = os.path.join(dataset_path, "modified_kinect")
    motion_lengths_file = os.path.join(dataset_path, "adavu_frames_len.csv")
    
    
    
    dataset = MotionClipsDataset(
        kinect_dir, motion_lengths_file, skeleton, type=sys.argv[1])
    print(len(dataset), dataset[0][0].shape, dataset[0][0].dtype)
    # for i, (clip1, clip2) in enumerate(dataset):
    #     print(i, np.isnan(np.min(clip1)), np.isnan(np.min(clip1)))
