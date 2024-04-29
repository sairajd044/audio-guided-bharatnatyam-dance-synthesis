import pickle
import sys
import torch
import numpy as np
import os
import argparse

from model import JointRotationModel, RootPointModel
# from dataset import MotionClipsDataset
from skeleton import Skeleton
from visualize import animate


def get_keypoints(path):
    with open(path, "rb") as fp:
        frames = pickle.load(fp)
    return frames
    
# adava_1_path = r"data\modified_kinect\1_tatta\1\Dancer1\tatta_1_Dancer1.pkl"
# adava_2_path = r"data\modified_kinect\2_natta\2\Dancer1\natta_2_Dancer1.pkl"


def get_root_parameters(data, root_model):
    clip1, clip2 = data
    clip1 = torch.tensor(clip1, dtype=torch.float32).unsqueeze(0)
    clip2 = torch.tensor(clip2, dtype=torch.float32).unsqueeze(0)
    root_pred = root_model(clip1, clip2).squeeze(0)
    return root_pred[:, :3].detach().numpy(), root_pred[:, 3:].detach().numpy()

def get_joint_rotations(data, joint_model):
    clip1, clip2 = data
    clip1 = torch.tensor(clip1, dtype=torch.float32).unsqueeze(0)
    clip2 = torch.tensor(clip2, dtype=torch.float32).unsqueeze(0)
    joint_pred = joint_model(clip1, clip2).squeeze(0)
    return joint_pred.detach().numpy()

def get_parameters(clip, skeleton):
    skeleton.load_keypoints(clip)
    skeleton.calculate_joint_angles()
    joint_rotations = skeleton.get_joint_rotation_frames()[1:]
    hip_rotations = joint_rotations[:, 0]
    hip_velocity = (clip[1:, 0] - clip[0:-1, 0]) / 30
    # hip_rotations = np.expand_dims(hip_rotations, axis=1)
    return joint_rotations[:, 1:], np.hstack((hip_velocity, hip_rotations))
    

def inpaint_adavu(adavu_1_frames, adavu_2_frames, root_model, joint_model, clip_size=192, mask_window=64):
    joints = [
        'HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 
        'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight','ElbowRight', 
        'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 
        'HipRight', 'KneeRight', 'AnkleRight', 'FootRight'
    ]
    joint_parent_indices = [-1,  0,  1,  2,  2,  4,  5,  6,  2,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18]
    
    initial_pose = adavu_1_frames[-clip_size // 2 - 1 :-clip_size // 2]
    clip1 = adavu_1_frames[-clip_size // 2:]
    clip2 = adavu_2_frames[:clip_size // 2]
    combined = np.concatenate((initial_pose, clip1, clip2))
    s = Skeleton(joints, joint_parent_indices)
    joint_parameters, root_parameters =  get_parameters(combined, s)
    # s.load_keypoints(clip2)
    # s.get_joint_position_frames(s.joint_rotations, s.keypoints[:, 0])
    # data = np.concatenate((clip1, clip2), axis=0)
    root_velocities, root_rotations = get_root_parameters(np.vsplit(root_parameters, 2), root_model)
    joint_rotations = get_joint_rotations(np.vsplit(joint_parameters, 2), joint_model)
    
    root_rotations = np.expand_dims(root_rotations, axis=1)
    
    root_points = [initial_pose[0][0]]
    for velocity in root_velocities:
        prev_point = root_points[-1]
        root_points.append(prev_point + velocity / 900)
        
    root_points = np.array(root_points[1:])
    
    joint_rotations = np.concatenate((root_rotations, joint_rotations), axis=1)
    
    
    
    new_skeleton = Skeleton(joints, joint_parent_indices)
    new_skeleton.load_keypoints(initial_pose)
    
    inpainted_frames = new_skeleton.get_joint_position_frames_1(
        np.array(joint_rotations), root_points
    )
    
    # data[-clip_size]
    total_motion = np.concatenate((adavu_1_frames[:-clip_size // 2], inpainted_frames, adavu_2_frames[clip_size // 2:]), axis=0)
    return total_motion

def main(args):
    
    adava_1_path = args.first_dance
    adava_2_path = args.second_dance
    
    adavu_1_frames = get_keypoints(adava_1_path)
    adavu_2_frames = get_keypoints(adava_2_path)
    
    weights_directory = args.saved_weight_dir
    
    root_model_path = os.path.join(weights_directory, "root", "best_loss_checkpoint.pth")
    joint_model_path = os.path.join(weights_directory, "joint", "best_loss_checkpoint.pth")
    
    root_model = RootPointModel()
    root_model.load_weights(root_model_path)
    joint_model = JointRotationModel()
    joint_model.load_weights(joint_model_path)
    
    final_motion = inpaint_adavu(adavu_1_frames, adavu_2_frames, root_model, joint_model)
    
    animate(adavu_1_frames, path=os.path.join(args.output_dir, "first.mp4"))
    animate(adavu_2_frames, path=os.path.join(args.output_dir, "second.mp4"))
    animate(final_motion, path = os.path.join(args.output_dir, "combined.mp4"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--saved_weight_dir', type=str, help='saved weights directory')
    parser.add_argument('-f', '--first_dance', type=str, help='.pkl file for first dance')
    parser.add_argument('-s', '--second_dance', type=str, help='.pkl file for second dance')
    parser.add_argument('-o', '--output_dir', type=str, help='output for generated video')
    args = parser.parse_args()
    main(args)
    

