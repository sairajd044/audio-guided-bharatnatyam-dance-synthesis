import torch
import numpy as np
import os
import sys
import pickle
import random
import argparse

from model import RootPointModel, JointRotationModel
from dataset import MotionClipsDataset
from skeleton import Skeleton
from visualize import animate

def save_dance_frames(frames, path):
    with open(path, 'wb') as f:
        pickle.dump(frames, f)

                                                                       

def get_root_parameters(data, root_model):
    clip1, clip2 = data
    clip1 = torch.tensor(clip1).unsqueeze(0)
    clip2 = torch.tensor(clip2).unsqueeze(0)
    root_pred = root_model(clip1, clip2).squeeze(0)
    return root_pred[:, :3].detach().numpy(), root_pred[:, 3:].detach().numpy()

def get_joint_rotations(data, joint_model):
    clip1, clip2 = data
    clip1 = torch.tensor(clip1).unsqueeze(0)
    clip2 = torch.tensor(clip2).unsqueeze(0)
    joint_pred = joint_model(clip1, clip2).squeeze(0)
    return joint_pred.detach().numpy()

def main(args):
    dataset_path = r"data"
    kinect_dir = os.path.join(dataset_path, "modified_kinect")
    motion_lengths_file = os.path.join(dataset_path, "adavu_frames_len.csv")
    
    joints = [
        'HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 
        'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight','ElbowRight', 
        'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 
        'HipRight', 'KneeRight', 'AnkleRight', 'FootRight'
    ]
    joint_parent_indices = [-1,  0,  1,  2,  2,  4,  5,  6,  2,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18]
    skeleton = Skeleton(joints, joint_parent_indices)
    
    root_dataset = MotionClipsDataset(kinect_dir, motion_lengths_file, skeleton, type='R')
    joint_dataset = MotionClipsDataset(kinect_dir, motion_lengths_file, skeleton, type='J')
    
    weights_directory = args.saved_weight_dir
    
    root_model_path = os.path.join(weights_directory, "root", "best_loss_checkpoint.pth")
    joint_model_path = os.path.join(weights_directory, "joint", "best_loss_checkpoint.pth")
    
    
    root_model = RootPointModel()
    root_model.load_weights(root_model_path)
    joint_model = JointRotationModel()
    joint_model.load_weights(joint_model_path)
    
    # print(len(root_dataset), len(joint_dataset))
    i = random.randint(0, len(root_dataset) - 1)
    
    # joint_pred = root_model(joint_dataset[i])
    # print(len(root_pred[0]), type(root_pred[0]))
    
    joint_position_frames, adavu_name = root_dataset.get_joint_position_frames(i)
    initial_pose = joint_position_frames[0]
    root_velocities, root_rotations = get_root_parameters(root_dataset[i], root_model)
    joint_rotations = get_joint_rotations(joint_dataset[i], joint_model)
    
    root_rotations = np.expand_dims(root_rotations, axis=1)
    
    root_points = [initial_pose[0]]
    for velocity in root_velocities:
        prev_point = root_points[-1]
        root_points.append(prev_point + velocity * 30)
        
    root_points = np.array(root_points[1:])
    
    # print(root_points)
    
    joint_rotations = np.concatenate((root_rotations, joint_rotations), axis=1)
    
    
    new_skeleton = Skeleton(joints, joint_parent_indices)
    new_skeleton.load_keypoints(joint_position_frames[:1])
    generated_frames = new_skeleton.get_joint_position_frames_1(
        joint_rotations, root_points
    )
    # print(data.shape)
    # generated_frames = get_joint_position_frames(
    #     initial_pose=intial_pose, 
    #     root_velocities=root_velocities, 
    #     bone_orientations=joint_rotations, 
    #     limbs=limbs)
    # print(frames.shape)
    output_dir = args.output_dir
    name = os.path.basename(os.path.splitext(adavu_name)[0])
    animate(joint_position_frames, os.path.join(output_dir, name) + "_original.mp4")
    animate(generated_frames, os.path.join(output_dir, name) + "_generated.mp4")
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--saved_weight_dir', type=str, help='saved weights directory')
    parser.add_argument('-o', '--output_dir', type=str, help='output for generated video')
    args = parser.parse_args()
    main(args)