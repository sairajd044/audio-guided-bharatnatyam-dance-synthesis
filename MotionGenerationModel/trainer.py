import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from model import JointRotationModel, RootPointModel

from utils import seconds_to_time, time_to_seconds

class Trainer(object):
    """ Class for training model on dataset """
    def __init__(self, 
                 model, dataset, train_test_split_ratio, device, 
                 NUM_EPOCHS, loss_fn, optimizer, scheduler, batch_size, 
                 training_result_dir):
        self.model = model.to(device)
        # self.cau_vocab = cau_vocab
        self.dataset = dataset
        self.tts_ratio = train_test_split_ratio
        self.device = device
        self.NUM_EPOCHS = NUM_EPOCHS
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.training_result_dir = training_result_dir
        
        self.epoch = 0
        self.last_trained_time = 0
        self.checkpoint_path = os.path.join(self.training_result_dir, "checkpoint.pth")
        self.best_loss_checkpoint_path = os.path.join(self.training_result_dir, "best_loss_checkpoint.pth")
        self.log_file_path = os.path.join(training_result_dir, "train_log.csv")
        self.graph_path = os.path.join(training_result_dir, "graph-{epoch}.png")
        
        self.best_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
        
        
    def get_dataloaders(self):
        """ Returns train and validaton dataloaders"""
        train_ds, test_ds = random_split(self.dataset, [self.tts_ratio, 1 - self.tts_ratio])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_dl, test_dl    

        
    def load_checkpoint(self):
        """ Initialize parameters from checkpoint file for resumption """
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        # self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        
        best_loss_checkpoint = torch.load(self.best_loss_checkpoint_path)
        self.best_loss = best_loss_checkpoint['loss']
        
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
        data = data.reshape(-1, 4)
        # print(data.shape, data.dtype)
        self.last_trained_time = time_to_seconds(data[-1, -1])
        data = data[:, :-1].astype(np.float32)
        epochs, self.train_losses, self.val_losses = data.T
        self.train_losses = list(self.train_losses)
        self.val_losses = list(self.val_losses)
    
    def save_checkpoint(self, loss):
        """Saves states of model, optimizer, scheduler for each epoch and the best model

        Args:
            loss (float): loss for current epoch.
        """
        torch.save({
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }, self.checkpoint_path)
        
        #Saves the best model
        if loss < self.best_loss:
            torch.save({
                "model_state": self.model.state_dict(), 
                "loss": loss
            }, self.best_loss_checkpoint_path)
            self.best_loss = loss
        
        
    
    def init_log(self):
        """ Initialized log csv file """
        with open(self.log_file_path, mode='w') as f:
            header = f'''epoch,training_loss,validation_loss,time\n'''
            f.write(header)
    
    def write_log(self, *args):
        """ Write into log csv file """
        with open(self.log_file_path, mode='a') as f:
            log = ','.join([str(x) for x in args])
            f.write(log + '\n')
            
    def save_plot(self):
        """ Saves plot of train and validation losses and bleu scores """
        plt.tight_layout(pad=3)
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses)
        plt.title("train_losses")
        
        plt.subplot(2, 1, 2)
        plt.plot(self.val_losses)
        plt.title("val_losses")
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(self.graph_path.format(epoch = self.epoch))
        try:
            os.remove(self.graph_path.format(epoch = self.epoch - 1))
        except FileNotFoundError:
            pass

    def train_one_epoch(self, dataloader):
        epoch_loss = 0
        self.model.train()
        for (clips1, clips2) in dataloader:
            batch_size = clips1.shape[0]
            clips1 = clips1.to(self.device)
            clips2 = clips2.to(self.device)
            ground_truth = torch.cat([clips1, clips2], dim=1)
            
            self.optimizer.zero_grad()
            prediction = self.model(clips1, clips2)

            loss = self.calculate_loss(prediction, ground_truth)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() / batch_size
            del clips1, clips2, prediction
            # torch.cuda.empty_cache()
        return epoch_loss / len(dataloader)
    
    def calculate_loss(self, prediction, ground_truth):
        """Reshapes the prediction and ground_truth before passing into loss function

        Args:
            prediction (N, 3) or (N, 6)
            ground_truth (N, 3) or (N, 6)

        Returns:
            tensor: loss
        """
        if isinstance(self.model, JointRotationModel):
            return self.loss_fn(
                prediction.view(-1, 3),
                ground_truth.view(-1, 3),
            )
        elif isinstance(self.model, RootPointModel):
            return self.loss_fn(
                prediction.view(-1, 6),
                ground_truth.view(-1, 6),
            )
    
    def val_one_epoch(self, dataloader):
        epoch_loss = 0
        self.model.eval()
        for (clips1, clips2) in dataloader:
            batch_size = clips1.shape[0]
            clips1 = clips1.to(self.device)
            clips2 = clips2.to(self.device)
            ground_truth = torch.cat([clips1, clips2], dim=1)
            
            # optimizer.zero_grad()
            prediction = self.model(clips1, clips2)
            loss = self.calculate_loss(prediction, ground_truth)
            epoch_loss += loss.item() / batch_size
            del clips1, clips2, prediction
            # torch.cuda.empty_cache()
        return epoch_loss / len(dataloader)
    
    def train(self, resume=False):
        if resume:
            self.load_checkpoint()
            self.load_metrics()
        else:
            self.init_log()
            
        train_dl, val_dl = self.get_dataloaders()
        start_time = time.time()
        
        #Performs training and validation for each epoch
        while self.epoch < self.NUM_EPOCHS:
            self.epoch += 1
            train_loss = self.train_one_epoch(train_dl)
            self.train_losses.append(train_loss)
            val_loss = self.val_one_epoch(val_dl)
            self.val_losses.append(val_loss)
            self.scheduler.step(train_loss)
            epoch_end_time = time.time()
            # if self.epoch % 10 == 0:
            self.save_checkpoint(val_loss)
            time_elapsed = seconds_to_time(self.last_trained_time + epoch_end_time - start_time)
            print(f'EPOCH: {self.epoch}, train_loss: {train_loss}, val_loss: {val_loss}, {time_elapsed}')
            self.write_log(self.epoch, train_loss, val_loss, time_elapsed)
            self.save_plot()
        print(f"Training completed for {self.epoch} epochs")
            
            
            
        
    
    