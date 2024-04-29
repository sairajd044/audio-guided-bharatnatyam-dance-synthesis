import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CauPredictionModel(nn.Module):
    def __init__(self, device, cau_vocab, cau_beats, bidirectional=False, window=10):
        super().__init__()
        self.device = device
        self.window = window
        self.cau_vocab = cau_vocab
        self.cau_beats = cau_beats
        self.SOD_token = cau_vocab.get_index('SOD')
        self.EOD_token = cau_vocab.get_index('EOD')
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=2,out_channels=4, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4,out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8,out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16,out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        music_features_dim = [32, 13 * window * 2 * 10]
        for _ in range(5):
            music_features_dim[1] = np.ceil((music_features_dim[1] - 2) / 2)
        
        self.encoder_fc = nn.Linear(in_features=music_features_dim[0] * int(music_features_dim[1]), out_features=64)
        self.emb = nn.Embedding(num_embeddings=cau_vocab.size, embedding_dim=128)
        self.mlp = nn.Linear(in_features=192, out_features=64)
        self.decoder_gru = nn.GRU(input_size=64, hidden_size=64)
        self.decoder_fc = nn.Linear(in_features=64, out_features=cau_vocab.size)
        if bidirectional:
            self.decoder_fc.in_features *= 2

    def get_beats_of_cau(self, cau):
        cau = cau.item()
        return self.cau_beats[self.cau_vocab.get_word_from_index(cau)]
        
    def get_audio_features(self, y, sr, t):
        """Acoustic features

        Args:
            y (list): audio samples
            sr (int): sample rate
            t (float): current time
        Returns:
            tensor (N, 13): Onset strength (N, 1) + Chroma features (N, 12)
        """
        # Obtaining fixed-size window of clips 
        start_idx = int(max((t - self.window) * sr, 0))
        end_idx = int((t + self.window) * sr)
        y_window = y[start_idx : end_idx]
        # Chroma features (N, 12)
        chroma_features = librosa.feature.chroma_stft(y=y_window, sr=sr, hop_length=int(sr/10)) #fps 10
        # onset_strength (N,)
        onset_strength = librosa.onset.onset_strength(y=y_window, sr=sr, hop_length=int(sr/10)) #fps 10
        onset_strength = np.expand_dims(onset_strength, axis=0) # (N, 1)
        
        #Both concatenated
        music_features = np.vstack((onset_strength, chroma_features))
        feature_length = self.window * 10 * 2 # window size * fps 
        music_features = music_features[:, : feature_length]
        
        # Applying necessary padding 
        if t - self.window < 0:
            music_features = np.pad(music_features, ((0,0), (feature_length-music_features.shape[1], 0)), 'constant', constant_values=((0, 0),(0,0)))
        elif end_idx > len(y) or music_features.shape[1] != feature_length:
            music_features = np.pad(music_features, ((0,0), (0,feature_length-music_features.shape[1])), 'constant', constant_values=((0, 0),(0,0)))
        music_features = np.expand_dims(music_features.T.flatten(), axis = 0)
        music_features_tensor =  torch.FloatTensor(music_features)
        music_features_tensor = music_features_tensor.to(self.device)
        return music_features_tensor

    def forward(self, music):
        y, sr, beat_samples, start_time, end_time, initial_beat_idx = music
        t = start_time
        current_beat_idx = initial_beat_idx
        #Initializing hidden layer of GRU
        hidden = self.init_hidden()
        #current token, initialzied as SOD
        y_p = torch.empty(1, dtype=torch.int).fill_(self.SOD_token)
        y_p = y_p.to(self.device)
        y_gen = [] # Store CAU tokens logits
        while t < end_time and y_p.item() != self.EOD_token:
            acoustic_features = self.get_audio_features(y, sr, t)
            acoustic_features = acoustic_features.view(1, -1)
            m_t = self.encoder_cnn(acoustic_features)
            m_t = m_t.view(1, -1)
            #Encoded music features
            m_t = self.encoder_fc(m_t)
            #Embedding previously generated CAU token
            y_p_emb = self.emb(y_p)
            x = torch.cat((m_t, y_p_emb), dim=1)
            x = self.mlp(x)
            out, hidden = self.decoder_gru(x, hidden)
            # Logits of generated CAU token 
            out = self.decoder_fc(out)
            y_gen.append(out)
            # Obtain CAU token from logits
            y_hat = out.argmax(dim=-1)
            y_p = y_hat.clone().detach()
            # Updating current beat index 
            current_beat_idx += self.get_beats_of_cau(y_hat)
            # Checking if time is consumed
            if current_beat_idx >= len(beat_samples):
                break
            # Shifting the current time
            t = beat_samples[current_beat_idx] / sr
        # print(y_gen[0].shape)
        # y_gen.append(self.)
        # if t >= end_time or current_beat_idx >= len(beat_samples):
        #     stat['music_ended'] += 1
        # if y_p.item() == self.EOD_token:
        #     stat['eod_found'] += 1
        
        del acoustic_features
        y_gen = torch.cat(y_gen)
        return F.log_softmax(y_gen, dim=-1) 
    
    # @staticmethod
    def init_hidden(self):
        hidden =  torch.zeros(1, 64)
        hidden = hidden.to(self.device)
        return hidden
    
    def load_weights(self, checkpoint_path):
        #Load model weights from saved file
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state'])
        self = self.to(self.device)