import torch
import torch.nn as nn
# from data import get_random_data
from unet import UNET

class MotionGenerationModel(nn.Module):
    """ Base class for both Motion Generation models"""
    def __init__(self, mask_window=64):
        """ Initialize model parameters """
        super(MotionGenerationModel, self).__init__()
        self.mask_window = mask_window
        
        self.encoder_layers = []
        self.decoder_layers = []
        channels = [1, 16, 32, 64, 32, 16, 1]
        # strides = [2, 2, 2, 1, 1, 1]
        for i in range(6):
            self.encoder_layers.append(
                nn.ConvTranspose1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3)
            )
            self.decoder_layers.append(
                nn.Conv1d(in_channels=channels[i+1], out_channels=channels[i], kernel_size=3)
            )
        
        self.unet = UNET(in_channels=[1, 32, 32, 32], out_channels=1)
        
    def reshape_input(self, input_clips):
        """ Reshape input appropriately before passing into frame encoder"""
        pass
        
    def reshape_output(self, decoded_clips):
        """ Reshape output appropriately after getting output from the frame decoder"""
        pass
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        motion_clip_len = x.shape[1]
        
        # Masking middle of concatenated clip
        x[:, (motion_clip_len - self.mask_window) // 2 : (motion_clip_len + self.mask_window) // 2] = 0
        x = self.reshape_input(x)
        encoded_clips = [] 
        
        # Encoding clip framewise 
        for clip in x:
            encoded_clip = self.encoder(clip)
            encoded_clip = encoded_clip.squeeze(1)
            # print("Encoded clip", encoded_clip.shape)
            encoded_clips.append(encoded_clip)
        
        # Concatenating frames together 
        encoded_clips = torch.stack(encoded_clips)
        encoded_clips = encoded_clips.unsqueeze(1) # Adding a channel 
        #Passing through UNET
        masked_clips = self.unet(encoded_clips)
        masked_clips = masked_clips.permute(0, 2, 1, 3)
        
        decoded_clips = []
        #Decoding impainted clip framewise
        for clip in masked_clips:
            decoded_clip = self.decoder(clip)
            decoded_clip = decoded_clip.squeeze(1)
            # print("Decoded clips", decoded_clip.shape)
            decoded_clips.append(decoded_clip)
        
        # Concatenating frames together 
        decoded_clips = torch.stack(decoded_clips)
        return self.reshape_output(decoded_clips)
    
    def load_weights(self, checkpoint_path):
        #Load model weights from saved file
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state'])


class JointRotationModel(MotionGenerationModel):
    def __init__(self, mask_window = 64):
        super().__init__(mask_window)

        self.encoder_layers.append(nn.Linear(54, 84))
        self.decoder_layers.append(nn.Linear(84, 54))
        self.decoder_layers.reverse()
        
        #Initializing encoder and decoder
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)
        
        del self.encoder_layers, self.decoder_layers
        
    def reshape_input(self, input_clips):
        x = input_clips.flatten(start_dim=-2)
        x = x.unsqueeze(2)
        return x
        
    def reshape_output(self, decoded_clips):
        return decoded_clips.view(decoded_clips.shape[0], decoded_clips.shape[1], 14, 3)
    
    
class RootPointModel(MotionGenerationModel):
    def __init__(self):
        super().__init__()
        
        self.encoder_layers.append(nn.Linear(18, 84))
        self.decoder_layers.append(nn.Linear(84, 18))
        self.decoder_layers.reverse()
        
        #Initializing encoder and decoder
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)
        
        del self.encoder_layers, self.decoder_layers
    
    def reshape_input(self, input_clips):
        return input_clips.unsqueeze(2)
      
    def reshape_output(self, decoded_clips):
        return decoded_clips.view(decoded_clips.shape[0], decoded_clips.shape[1], 6)

    
if __name__ == '__main__':
    x1, x2 = torch.randn(1, 96, 14, 4).cuda(), torch.randn(1, 96, 14, 4).cuda()
    # input_data = torch
    # print("Input data", input_data.size())
    model = RootPointModel().cuda()
    model = JointRotationModel().cuda()
    pred = model(x1, x2)
    print(pred.shape)
    # print(pred.shape)
          
          