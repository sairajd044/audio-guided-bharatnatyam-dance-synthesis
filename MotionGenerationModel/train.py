import time
import os

import torch.nn as nn
import torch.optim as optim

from dataset import MotionClipsDataset
from model import JointRotationModel, RootPointModel

from trainer import Trainer
from options import parser

from skeleton import Skeleton

def main(args):
    
    dataset_path = args.dataset_dir
    kinect_dir = os.path.join(dataset_path, "modified_kinect1")
    motion_lengths_file = os.path.join(dataset_path, "adavu_frames_len.csv")
    if args.model == 'J':
        model = JointRotationModel()
        loss_fn = nn.L1Loss(reduction="sum")
    elif args.model == 'R':
        model = RootPointModel()
        loss_fn = nn.L1Loss(reduction="sum")
    else:
        SystemExit()
        
        
    joints = [
        'HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', \
        'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight','ElbowRight', \
        'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', \
        'HipRight', 'KneeRight', 'AnkleRight', 'FootRight'
    ]
    joint_parent_indices = [-1,  0,  1,  2,  2,  4,  5,  6,  2,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18]
    skeleton = Skeleton(joints, joint_parent_indices)
    dataset = MotionClipsDataset(kinect_dir, motion_lengths_file, skeleton, args.model)
    
    training_result_dir = args.output_dir
    # shutil.rmtree(training_result_dir, ignore_errors=False) 
    os.makedirs(training_result_dir, exist_ok=True)
    
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    
    Trainer(
        dataset=dataset,
        model=model,
        device=args.device,
        train_test_split_ratio=0.7,
        NUM_EPOCHS=args.epoch,
        batch_size=args.batch_size,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        training_result_dir=training_result_dir,
    ).train(resume=args.resume)

if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings('ignore')
    args = parser.parse_args()
    main(args)
    