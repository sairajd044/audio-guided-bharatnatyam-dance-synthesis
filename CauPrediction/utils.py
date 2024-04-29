import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

class Vocab:
    def __init__(self, special_tokens):
        self.size = 0
        self.word2index = {}
        self.index2word = {}
        for token in special_tokens:
            self.add_word(token)
    
    def add_word(self, word):
        if word not in self.word2index:
            idx = len(self.word2index)
            self.word2index[word] = idx
            self.index2word[idx] = word
            self.size += 1
        
    def get_index(self, word):
        return self.word2index[word]
    
    def get_word_from_index(self, idx):
        return self.index2word[idx]
    
def get_cau_vocabulary(csv_file, special_tokens):
    vocab = Vocab(special_tokens)
    df = pd.read_csv(csv_file)
    cau_list = df['movement_tag'].values
    for cau in cau_list:
        vocab.add_word(cau)
    return vocab
        
def get_beats_of_cau(csv_file, **extra_caus):
    mapping = {**extra_caus}
    df = pd.read_csv(csv_file)
    for cau, beat in df[['movement_tag', 'beats']].values:
        try:
            mapping[cau] = int(beat)
        except ValueError:
            pass
    return mapping
        
def bleu_score(actual_indices, prediction):
    predicted_indices = prediction.argmax(dim=-1)
    predicted_seq =  [cau_vocab.get_word_from_index(index.item()) for index in predicted_indices]
    actual_seq = [[cau_vocab.get_word_from_index(index.item()) for index in actual_indices]]
    return corpus_bleu(actual_indices, predicted_indices)

def get_predicted_cau_sequence(prediction, cau_vocab):
    predicted_index = prediction.argmax(dim=-1)
    return [cau_vocab.get_word_from_index(index.item()) for index in predicted_index]

def seconds_to_time(seconds):
    mint, sec = divmod(seconds, 60)
    hour, mint = divmod(mint, 60)
    # return (hour, mint, round(sec, 2))
    return f'{hour} hr {mint} min {sec:.2f} s'
    
def time_to_seconds(time_str):
    h, _, m, _, s, _ = time_str.split()
    return float(h) * 3600 + float(m) * 60 + float(s)

def save_checkpoint(epoch, model, optimizer, scheduler, file):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }, file)

def save_plot(train_losses, train_blue, val_losses, val_bleu, path):
    plt.subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.plot(train_losses)
    plt.title("train_losses")
    
    plt.subplot(2, 2, 2)
    plt.plot(train_blue)
    plt.title("train_blue")
    
    plt.subplot(2, 2, 3)
    plt.plot(val_losses)
    plt.title("val_losses")
    
    plt.subplot(2, 2, 4)
    plt.plot(val_bleu)
    plt.title("val_bleu")
    
    plt.savefig(path)