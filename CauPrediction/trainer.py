import torch
from torch.utils.data import random_split
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from utils import seconds_to_time, time_to_seconds

class Trainer(object):
    """ Class for training model on dataset """
    def __init__(self, 
                 model, cau_vocab, dataset, train_test_split_ratio, device, NUM_EPOCHS, 
                 loss_fn, optimizer, scheduler, training_result_dir, resume=False):
        """ Initialize parameters """
        self.model = model.to(device)
        self.cau_vocab = cau_vocab
        self.dataset = dataset
        self.tts_ratio = train_test_split_ratio
        self.device = device
        self.NUM_EPOCHS = NUM_EPOCHS
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_result_dir = training_result_dir
        
        self.epoch = 0
        self.last_trained_time = 0
        self.checkpoint_path = os.path.join(self.training_result_dir, "checkpoint.pth")
        self.best_loss_checkpoint_path = os.path.join(self.training_result_dir, "best_loss_checkpoint.pth")
        self.best_bleu_checkpoint_path = os.path.join(self.training_result_dir, "best_bleu_checkpoint.pth")
        self.log_file_path = os.path.join(training_result_dir, "train_log.csv")
        self.graph_path = os.path.join(training_result_dir, "graph-{epoch}.png")
        
        self.best_loss = float('inf')
        self.best_bleu = float('-inf')
        
        self.train_losses = []
        self.val_losses = []
        self.train_bleu_scores = []
        self.val_bleu_scores = []
        
    def load_checkpoint(self):
        """ Initialize parameters from checkpoint file for resumption """
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        
        best_loss_checkpoint = torch.load(self.best_loss_checkpoint_path)
        self.best_loss = best_loss_checkpoint['loss']
        best_bleu_checkpoint = torch.load(self.best_bleu_checkpoint_path)
        self.best_bleu = best_bleu_checkpoint['bleu']
            
        # if model trained for more epochs than asked 
        if self.NUM_EPOCHS <= self.epoch:
            print(f"Training already completed till {self.epoch}")
            sys.exit(1)
        
    def load_metrics(self):
        """ Load training loss and bleu scores for resumption """
        data = np.genfromtxt(
            fname=self.log_file_path,
            skip_header=1,
            delimiter=',',
            dtype=object
        )
        # print(data.shape, data.dtype)
        data = data.reshape(-1, 6)
        # print(data.shape, data.dtype)
        self.last_trained_time = time_to_seconds(data[-1, -1])
        data = data[:, :-1].astype(np.float32)
        epochs, self.train_losses, self.val_losses, self.train_bleu_scores, self.val_bleu_scores = data.T
        self.train_losses = list(self.train_losses)
        self.train_bleu_scores = list(self.train_bleu_scores)
        self.val_losses = list(self.val_losses)
        self.val_bleu_scores = list(self.val_bleu_scores)
    
    def save_checkpoint(self, loss, bleu):
        """Saves states of model, optimizer, scheduler for each epoch and the best model

        Args:
            loss (float): loss for current epoch.
            bleu (float): bleu for current epoch.
        """
        torch.save({
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }, self.checkpoint_path)
        
        #Save best model based on loss
        if loss < self.best_loss:
            torch.save({
                "model_state": self.model.state_dict(), 
                "loss": loss
            }, self.best_loss_checkpoint_path)
            self.best_loss = loss
        
        #Save best model based on bleu score
        if bleu > self.best_bleu:
            torch.save({
                "model_state": self.model.state_dict(), 
                "bleu": bleu
            }, self.best_bleu_checkpoint_path)
            self.best_bleu = bleu
        
    
    def init_log(self):
        """ Initialized log csv file """
        with open(self.log_file_path, mode='w') as f:
            header = f'''epoch,training_loss,training_bleu,validation_loss,validation_bleu,time\n'''
            f.write(header)
    
    def write_log(self, *args):
        """ Write into log csv file """
        with open(self.log_file_path, mode='a') as f:
            log = ','.join([str(x) for x in args])
            f.write(log + '\n')
            
    def save_plot(self):
        """ Saves plot of train and validation losses and bleu scores """
        plt.subplot(2, 2, 1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.plot(self.train_losses)
        plt.title("train_losses")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.train_bleu_scores)
        plt.title("train_blue")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.val_losses)
        plt.title("val_losses")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.val_bleu_scores)
        plt.title("val_bleu")
        
        plt.savefig(self.graph_path.format(epoch = self.epoch))
        try:
            os.remove(self.graph_path.format(epoch = self.epoch - 1))
        except FileNotFoundError:
            pass
    
    def bleu_score(self, actual_indices, prediction):
        """Gets bleu score.

        Args:
            actual indices (list): Index of each CAU in vocabulary
            prediction (list): Predicted CAU tokens

        Returns:
            int: bleu score
        """
        predicted_indices = prediction[1:].argmax(dim=-1)
        predicted_seq =  [self.cau_vocab.get_word_from_index(index.item()) for index in predicted_indices]
        actual_seq = [self.cau_vocab.get_word_from_index(index.item()) for index in actual_indices]
        # print("Predicted", predicted_seq)
        # print("Actual", actual_seq)
        return sentence_bleu([actual_seq], predicted_seq, weights=(0.1, 0.2, 0.3, 0.4)) # Giving more weightage of 4-grams, then 3-grams and so on

    def train_one_epoch(self, dataset):
        epoch_loss = 0
        epoch_bleu = 0
        self.model.train()
        for (audio_data, cau_sequence) in dataset:
            cau_sequence = cau_sequence.to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(audio_data)
            loss = self.loss_fn(prediction, cau_sequence)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_bleu += self.bleu_score(cau_sequence, prediction)
            del cau_sequence, prediction
        return epoch_loss / len(dataset), epoch_bleu / len(dataset)
    
    def val_one_epoch(self, dataset):
        epoch_loss = 0
        epoch_bleu = 0
        self.model.train()
        for (audio_data, cau_sequence) in dataset:
            cau_sequence = cau_sequence.to(self.device)
            prediction = self.model(audio_data)
            loss = self.loss_fn(prediction, cau_sequence)
            epoch_loss += loss.item()
            epoch_bleu += self.bleu_score(cau_sequence, prediction)
            del cau_sequence, prediction
        return epoch_loss / len(dataset), epoch_bleu / len(dataset)
    
    def train(self, resume=False):
        if resume:
            self.load_checkpoint()
            self.load_metrics()
        else:
            self.init_log()
            
        
        train_ds, val_ds = random_split(self.dataset, [self.tts_ratio, 1 - self.tts_ratio])
        start_time = time.time()
        
        #Performs training and validation for each epoch
        while self.epoch < self.NUM_EPOCHS:
            self.epoch += 1
            train_loss, train_bleu = self.train_one_epoch(train_ds)
            self.train_losses.append(train_loss)
            self.train_bleu_scores.append(train_bleu)
            val_loss, val_bleu = self.val_one_epoch(val_ds)
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(val_bleu)
            self.scheduler.step(train_loss)
            epoch_end_time = time.time()
            # if self.epoch % 10 == 0:
            self.save_checkpoint(val_loss, val_bleu)
            time_elapsed = seconds_to_time(self.last_trained_time + epoch_end_time - start_time)
            print(f'EPOCH: {self.epoch}, train_loss: {train_loss}, train_bleu: {train_bleu}, val_loss: {val_loss}, val_bleu: {val_bleu}, {time_elapsed}')
            self.write_log(self.epoch, train_loss, train_bleu, val_loss, val_bleu, time_elapsed)
            self.save_plot()
        print(f"Training completed for {self.epoch} epochs")
            
            
            
        
    
    