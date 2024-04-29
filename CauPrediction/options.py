import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', help='Resumes the training', action='store_true')
parser.add_argument('-e', '--epoch', type=int, help='Number of epochs to train', default=1000)
parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], help='(cpu / cuda)', default='cuda')
parser.add_argument('-b', '--bidirectional', action='store_true')
parser.add_argument('-w', '--window', type=int, help='window length for music feature extraction')
parser.add_argument('-o', '--ouptut_dir', type=str, help='training result directory')
parser.add_argument('-f', '--dataset_dir', type=str, help='dataset directory')


    # print(args.boom.upper())


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)