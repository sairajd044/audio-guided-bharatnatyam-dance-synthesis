import os
import argparse

from model import CauPredictionModel
from dataset import get_audio_data
from utils import *

def main(args):
    dataset_path = args.dataset_dir
    csv_file = os.path.join(dataset_path, "movement_interval.csv")
    
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
    
    cau_pred_model.load_weights(args.saved_weights)
    audio_data = get_audio_data(args.music_file, args.start, args.end, args.window)
    pred = cau_pred_model(audio_data)
    pred_cau_indices = pred.argmax(dim=1)
    result = [cau_vocab.get_word_from_index(int(idx)) for idx in pred_cau_indices]
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-', '--resume', help='Resumes the training', action='store_true')
    parser.add_argument('-b', '--bidirectional', action='store_true')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], help='(cpu / cuda)', default='cuda')
    parser.add_argument('-c', '--saved_weights', type=str)
    parser.add_argument('-w', '--window', type=int, help='window length for music feature extraction')
    parser.add_argument('-s', '--start', type=int, help='starting time (in ms)', default=0)
    parser.add_argument('-e', '--end', type=int, help='end time (in ms)', default=None)
    parser.add_argument('-m', '--music_file', type=str, help='training result directory')
    parser.add_argument('-f', '--dataset_dir', type=str, help='dataset directory')

    args = parser.parse_args()
    main(args)
    