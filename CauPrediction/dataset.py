import os
import bisect
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


def get_audio_data(audio_file_path, music_start, music_end, window):
    """
    Args:
        audio_file_path (str): path of audio file
        music_start (float): start of music in audio file in seconds
        music_end (float): end of music in audio file in seconds
        window (int): fixed length window size
    Returns:
        y: audio samples
        sr: sample rate
        beat_samples: audio samples where beat occurs
        new_start: modified start time in seconds
        new_end: modified start time in seconds
        initial_beats_idx: index of first beat in beat_samples array
    """
    y, sr = librosa.load(audio_file_path)
    offset = max(music_start - window, 0)
    if music_end == None: 
        music_end = len(y) / sr
        
    # start_frame = offset * sr
    offset_sample = int(sr * offset)
    end_sample = int(sr * (music_end + window))
    y = y[offset_sample : end_sample]
    new_start = offset
    new_end = music_end - offset
    tempo, beat_samples = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    # initial_beats_idx = len(list(filter(lambda x : x < sr * offset, beat_samples)))
    initial_beats_idx = bisect.bisect_left(beat_samples, offset_sample)
    return y, sr, beat_samples, new_start, new_end, initial_beats_idx

class Music_CAU_Dataset(Dataset):
    """ Music <-> CAU dataset """
    def __init__(self, choreography_path, music_directory, cau_vocab, audio_fps=25, window=10):
        """ Initialize the data structures"""
        self.audio_fps = audio_fps
        self.window = window // 2
        choreo_dfs = []
        for choreo_file in os.listdir(choreography_path):
            if os.path.splitext(choreo_file)[-1] == '.json':
                temp_df = pd.read_json(os.path.join(choreography_path, choreo_file), dtype={'dance_id': 'string'})
                choreo_dfs.append(temp_df)
        choreo_df = pd.concat(choreo_dfs)
        
        #Preparing music
        choreo_df['music_file'] = choreo_df.apply(
            lambda row : os.path.join(music_directory, row['dance_type'], row['dance_id'] + '.mp3'), axis = 1
        )
        choreo_df['music_start_time'] = choreo_df['start_pos'] / audio_fps
        choreo_df['music_end_time'] = choreo_df['end_pos'] / audio_fps
        self.music_info = choreo_df[['music_file', 'music_start_time', 'music_end_time']].values
        EOD_token = cau_vocab.get_index('EOD')
        
        #Prepapring CAU sequence
        choreo_df['cau_sequence'] = choreo_df['movements'].apply(lambda cau_seq : [cau_vocab.get_index(cau) for cau in cau_seq] + [EOD_token])
        self.cau_sequences = choreo_df['cau_sequence'].values

    def __len__(self):
        return len(self.cau_sequences)
    
    def __getitem__(self, idx):
        audio_file_path, music_start, music_end = self.music_info[idx]
        audio_data = get_audio_data(audio_file_path, music_start, music_end, self.window)
        # input_ = (audio_data, self.cau_sequences[idx])
        cau_sequence = torch.Tensor(self.cau_sequences[idx])
        return audio_data, cau_sequence.long()
    