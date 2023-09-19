import os
import torch
import torchaudio
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchaudio.functional as F
import torch.nn.functional as Ff
from scipy.signal import hilbert, chirp
from pydub import AudioSegment
from natsort import natsorted
import torchaudio.transforms as T
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def audio_trimming(audio_path, trim_time):
    audio = AudioSegment.from_file(audio_path)
    start_time = trim_time * 1000 #As pydub deals in milliseconds
    trimmed_audio = audio
    sample_rate = trimmed_audio.frame_rate
    audio_array = np.array(trimmed_audio.get_array_of_samples())
    audio_tensor = torch.from_numpy(audio_array).float()

    #print(audio_tensor.size())
    return audio_tensor, sample_rate

def one_hot_encode(labels):
    one_hot_labels = [0, 0]
    if ' flinching' in labels:
        one_hot_labels[0] = 1 
    if ' licking' in labels:
        one_hot_labels[1] = 1
    return one_hot_labels

def interploate_pose(pose_pred):
    pose_keypoints = []
    for i in range(len(pose_pred)):
        if pose_pred[i] == [0]:
            prev_valid = None
            for j in range(i-1, -1, -1):
                if not pose_pred[j] == [0]:
                    prev_valid = pose_pred[j]
                    break

            next_valid = None
            for j in range(i+1, len(pose_pred)):
                if not pose_pred[j] == [0]:
                    next_valid = pose_pred[j]
                    break

            if prev_valid is not None and next_valid is not None:
                keypoints = (np.array(prev_valid['keypoints']) + np.array(next_valid['keypoints'])) / 2
            elif prev_valid is not None:
                keypoints = prev_valid['keypoints']
            elif next_valid is not None:
                keypoints = next_valid['keypoints']
            pose_keypoints.append(keypoints)
        else:
            pose_keypoints.append(pose_pred[i]['keypoints'])    

    pose_keypoints = np.array(pose_keypoints)
    pose_keypoints = np.delete(pose_keypoints, [2,5,8,11,14,17], axis=1) # remove score
    return pose_keypoints

def time_to_frame(time):
    time_str = str(time)
    if len(time_str) > 5:
        # print("First one", time_str)
        time_str = time_str[1:6] # handle 29:48:00 in Formalin_acute_pain_1.csv
    if '.' in time_str:
        # print("Second one", time_str)
        minutes, seconds = map(int, time_str.split('.'))
    elif ':' in time_str:
        # print("Third one", time_str)
        minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds * 30

def sliding_window(pose_keypoints, audio, label_data, step=30, stride=10):
    bias = len(pose_keypoints)%step
    
    subarrays = [] 
    for i in range(bias, len(pose_keypoints), stride):
        if len(pose_keypoints[i:i+step]) == 30:
            subarray = pose_keypoints[i:i+step]  # Extract a subarray of length 'step'
            subarrays.append(subarray)
        else:
            continue

    behavior_feat = np.asarray(subarrays)
    subarrays = []
    for i in range(bias, len(audio), stride):
        if len(audio[i:i+step]) == 30:
            subarray = audio[i:i+step]  # Extract a subarray of length 'step'
            subarrays.append(subarray)
        else:
            continue
    
    audio_feat = np.asarray(subarrays)
    subarrays = []
    for i in range(bias, len(label_data), stride):
        if len(label_data[i:i+step]) == 30:
            subarray = label_data[i:i+step]  # Extract a subarray of length 'step'
            subarrays.append(subarray)
        else:
            continue

    label_window = np.asarray(subarrays)
    subarrays = []
    
    one_hot_labels = np.array([one_hot_encode(list(set(x))) for x in label_window])

    valid_indices = ~(np.all(one_hot_labels == [0,0], axis=1))
    behavior_feat = behavior_feat[valid_indices]
    
    audio_feat = audio_feat[valid_indices]
    one_hot_labels = one_hot_labels[valid_indices]

    return behavior_feat, audio_feat, one_hot_labels, valid_indices

def listing(resampled_audio):
    x = []
    for i in range(0, len(resampled_audio), 1000):
        arr = np.array(resampled_audio[i:i+1000])
        if len(arr) == 1000:
            x.append(arr)
        else:
            pass
    x= np.asarray(x)
    x = x[:-93, :]
    x = torch.from_numpy(x)
    return x

def count_labels(labels):
    count_linking = 0
    count_flinching = 0
    for i in range(len(labels)):
        if labels[i,1] == 0:
            count_linking = count_linking + 1
        else:
            count_flinching = count_flinching + 1
    print(count_linking)
    print(count_flinching)


def mfcc_lfcc(sampled_array, sample_rate, n_fft=127, hop_length=16, n_lfcc = 64, win_length = None):
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    lfcc_spectrogram = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )
    
    audio_feat = []
    # for i in tqdm(range(len(sampled_array)), desc="Processing"):
    for i in tqdm(range(len(sampled_array))):
        mfcc_spec = spectrogram(sampled_array[i])
        lfcc_spec = lfcc_spectrogram(sampled_array[i])
        
        padding = [0, 1]
        mfcc_spec = Ff.pad(mfcc_spec, padding, mode='constant', value=0)
        mfcc_spec = mfcc_spec.numpy()
        lfcc_spec = Ff.pad(lfcc_spec, padding, mode='constant', value=0)
        lfcc_spec = lfcc_spec.numpy()
        audio_feat.append([mfcc_spec, lfcc_spec])
    return audio_feat

def feature_extraction(audio_path):
    waveform, sample_rate = audio_trimming(audio_path, 653)
    resampled_audio = F.resample(waveform, sample_rate, 30000, rolloff=0.99)
    sampled_array = listing(resampled_audio)
    features = mfcc_lfcc(sampled_array, 30000)
    features = np.asarray(features)
    # features = torch.from_numpy(features)
    return(features)

def listing(resampled_audio):
    x = []
    for i in range(0, len(resampled_audio), 1000):
        arr = np.array(resampled_audio[i:i+1000])
        if len(arr) == 1000:
            x.append(arr)
        else:
            pass
    x= np.asarray(x)
    x = x[:-93, :]
    x = torch.from_numpy(x)
    return x

def count_labels(labels):
    count_linking = 0
    count_flinching = 0
    for i in range(len(labels)):
        if labels[i,1] == 0:
            count_linking = count_linking + 1
        else:
            count_flinching = count_flinching + 1
    print(count_linking)
    print(count_flinching)


def mfcc_lfcc(sampled_array, sample_rate, n_fft=127, hop_length=16, n_lfcc = 64, win_length = None):
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    lfcc_spectrogram = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        },
    )
    
    audio_feat = []
    # for i in tqdm(range(len(sampled_array)), desc="Processing"):
    for i in tqdm(range(len(sampled_array))):
        mfcc_spec = spectrogram(sampled_array[i])
        lfcc_spec = lfcc_spectrogram(sampled_array[i])
        
        #As the spec will be a dimension of (64, 63) So we have done a pad of 1 row with zero
        padding = [0, 1]
        mfcc_spec = Ff.pad(mfcc_spec, padding, mode='constant', value=0)
        mfcc_spec = mfcc_spec.numpy()
        lfcc_spec = Ff.pad(lfcc_spec, padding, mode='constant', value=0)
        lfcc_spec = lfcc_spec.numpy()
        audio_feat.append([mfcc_spec, lfcc_spec])
    return audio_feat

def feature_extraction(audio_path):
    waveform, sample_rate = audio_trimming(audio_path, 653)
    resampled_audio = F.resample(waveform, sample_rate, 30000, rolloff=0.99)
    sampled_array = listing(resampled_audio)
    features = mfcc_lfcc(sampled_array, 30000)
    features = np.asarray(features)
    # features = torch.from_numpy(features)
    return(features)

class MouseDataset(Dataset):
    def __init__(self, frames_folder, pred_path, label_path, audio_path, resampling_rate):
        super(MouseDataset, self).__init__()

        self.image_files = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]
        self.image_files = natsorted(self.image_files)
        self.image_files = self.image_files[19465:]
        self.frames_folder = frames_folder
        print(self.frames_folder)
        self.pred_path = pred_path
        self.label_path = label_path
        self.audio_path = audio_path

        self.sliding_window = True
        self.step = 30
        self.stride = 10
        self.bias = len(self.image_files) % self.step
        # self.resampling_rate = resampling_rate
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # audio feature extraction in the form of (64, 64) for mfcc and lfcc
        
        audio_features  = feature_extraction(self.audio_path)
        pose_pred = self.load_predictions(len(self.image_files))
        pose_keypoints = interploate_pose(pose_pred)

        label_data = self.load_Formalin_labels(len(self.image_files))
        
        # sliding window
        if self.sliding_window:
            self.behavior_feat, self.audio_feat, self.labels, self.valid_indices = sliding_window(pose_keypoints, audio_features, label_data)

        self.all_indices = np.arange((len(self.image_files) - self.bias - self.step) // self.stride + 1)

    def read_label(self, image_file):
        label = image_file.split('_')[-1].split('.')[0]
        return int(label)

    def load_predictions(self, total_frames):
        #print(total_frames)
        with open(self.pred_path) as f:
            pose_top = json.load(f)
            # print(len(pose_top))

        # Single-mouse pose_pred
        pose_pred = [[0]] * total_frames
        # Sort annotations
        for i in range(len(pose_top)):
            image_id = pose_top[i]['image_id']
            #print(pose_pred[image_id])
            #print([image_id, total_frames])
            if image_id < total_frames:
                if not pose_pred[image_id] == [0]:
                    if pose_top[i]['score'] > pose_pred[image_id]['score']:
                        pose_pred[image_id] = pose_top[i]
                else:
                    pose_pred[image_id] = pose_top[i]
            else:
                continue

        return pose_pred

    def load_Formalin_labels(self, total_frames):
        label_dataframe = pd.read_csv(self.label_path)

        label_data = ['no behavior'] * total_frames
        for index, record in label_dataframe.iterrows():
            start_frame = time_to_frame(record[0])
            end_frame = time_to_frame(record[1])
            behavior = record[2]
            for i in range(start_frame, end_frame):
                label_data[i] = behavior
        return label_data

    def __len__(self):
        return np.sum(self.valid_indices)

    def __getitem__(self, idx):

    
        actual_idx = np.where(self.all_indices)[0][idx]
        start_idx = self.bias + actual_idx * self.stride
        end_idx = start_idx + self.step
        if end_idx > len(self.image_files):
            print(start_idx, end_idx)
            print(len(self.image_files))
            print(self.bias, idx, self.stride)

        images = []
        for i in range(start_idx, end_idx):
            image_path = os.path.join(self.frames_folder, self.image_files[i])
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images_tensor = torch.stack(images)
        
        # iterate labels
        labels = self.labels[idx]
        labels_tensor = torch.tensor(labels)

        # iterate behavior_feat
        behavior_feat = self.behavior_feat[idx]   
        behavior_feat_tensor = torch.tensor(behavior_feat)

        #iterate over audio features
        audio_feat = self.audio_feat[idx]   
        audio_feat_tensor = torch.tensor(audio_feat)

        return images_tensor, behavior_feat_tensor, audio_feat_tensor, labels_tensor