import os

import torch.nn.functional as F
import torch.optim as optim

from dataset import Music_CAU_Dataset
from model import CauPredictionModel
from utils import *
from options import parser
from trainer import Trainer

def custom_loss(prediction, target):
    pred_len = len(prediction)
    target_len = len(target)
    min_length = min(pred_len, target_len)
    return 0.01 * abs(pred_len - target_len) + F.nll_loss(prediction[:min_length], target[:min_length])

def main(args):
    # dataset_path = r"/home/mt0/22CS60R52/ChoreoNet/data"
    dataset_path = args.dataset_dir
    music_dir = os.path.join(dataset_path, 'music')
    choreography_path = os.path.join(dataset_path, 'choreography')
    csv_file = os.path.join(dataset_path, "movement_interval.csv")
    
    training_result_dir = args.ouptut_dir
    os.makedirs(training_result_dir, exist_ok=True)
    
    cau_vocab = get_cau_vocabulary(
        csv_file=csv_file,
        special_tokens=['SOD', 'EOD', 'START', 'HOLD']
    )
    
    cau_beats = get_beats_of_cau(csv_file=csv_file, START = 16, HOLD = 4, SOD=1, EOD = 0)

    cau_pred_model = CauPredictionModel(
        cau_vocab=cau_vocab, 
        cau_beats=cau_beats, 
        device=args.device, 
        bidirectional=args.bidirectional,
        window=args.window
    )
    dataset = Music_CAU_Dataset(choreography_path, music_dir, cau_vocab)
    
    
    
    loss_fn = custom_loss
    learning_rate = 1e-3
    optimizer = optim.RMSprop(cau_pred_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.9)

    train_test_split_ratio = 0.7
    
    # print(args.device)
    
    import warnings
    # warnings.filterwarnings('ignore')
    Trainer(
        model=cau_pred_model,
        dataset=dataset,
        train_test_split_ratio=train_test_split_ratio,
        device=args.device,
        NUM_EPOCHS=args.epoch,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        training_result_dir=training_result_dir,
        cau_vocab=cau_vocab
    ).train(resume=args.resume)
    # warnings.filterwarnings('default')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    