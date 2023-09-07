# import os
# import torch
# import torchaudio
# from torchaudio import transforms
# import torchaudio.functional as F
# audio_path = '/home/nikk/dataset/Formalin_Ultrasound_recording.wav'

# waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
# print(waveform.size())

# resample_rate = 1500
# resampled_audio = F.resample(waveform, sample_rate, 1500, rolloff=0.99)
# print(resampled_audio.size()) 
# Array = []
# for i in range (4650, len(resampled_audio[0]), 50):
#     arr = resampled_audio[0, i:i+50]
#     if(len(arr)==50):
#         Array.append(arr)
#     else:
#         pass


# Array = torch.stack(Array)
# print(Array.shape)

import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import random_split
from dataset import MouseDataset
from model import MouseModel

video_path ="/home/nikk/dataset/frame_folder"
pred_path ='/home/nikk/dataset/alphapose-results.json'
label_path = '/home/nikk/dataset/Formalin_acute_pain_1.csv'
audio_path='/home/nikk/dataset/Formalin_Ultrasound_recording.wav'

   
dataset = MouseDataset(video_path, pred_path, label_path, audio_path, 1500)
audio = dataset.behavior_feat
print(audio.shape)

    