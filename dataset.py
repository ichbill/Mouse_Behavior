import os
import torch
# import torchaudio
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchaudio.functional as F

import json
import numpy as np
import pandas as pd

def one_hot_encode(labels):
    one_hot_labels = [0,0]
    # if ' habituation' in labels:
    #     one_hot_labels[0] = 1
    if ' flinching' in labels:
        one_hot_labels[0] = 1
    if ' licking' in labels:
        one_hot_labels[1] = 1
    return one_hot_labels

def interploate_pose(pose_pred):
    pose_keypoints = []
    for i in range(len(pose_pred)):
        if pose_pred[i] == [0]: # if no pose prediction
            # search two nearest valid pose prediction
            # search previous valid pose prediction
            prev_valid = None
            for j in range(i-1, -1, -1):
                if not pose_pred[j] == [0]:
                    prev_valid = pose_pred[j]
                    break
            
            # search next valid pose prediction
            next_valid = None
            for j in range(i+1, len(pose_pred)):
                if not pose_pred[j] == [0]:
                    next_valid = pose_pred[j]
                    break

            # interpolate the pose prediction by averaging two nearest valid pose prediction
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
        time_str = time_str[:5] # handle 29:48:00 in Formalin_acute_pain_1.csv
    if '.' in time_str:
        minutes, seconds = map(int, time_str.split('.'))
    elif ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds * 30

def sliding_window(pose_keypoints, label_data):
    step = 10
    stride = 10
    # sliding window for pose
    # if args.local_rank == 0:
    print('pose_keypoints shape:', pose_keypoints.shape)
    bias = len(pose_keypoints)%step
    behavior_feat = np.array([pose_keypoints[bias:][i:i+step] for i in range(0,len(pose_keypoints[bias:]),stride)])
    # if args.local_rank == 0:
    print('pose_keypoints sliding window shape:', behavior_feat.shape)

    # sliding window for audio
    # if args.local_rank == 0:
    #     print('audio_feat shape:', audio.shape)
    # audio_feat = np.array([audio[bias:][i:i+step] for i in range(0,len(audio[bias:]),stride)])
    # if args.local_rank == 0:
    #     print('audio_feat sliding window shape:', audio_feat.shape)

    # sliding window for labels
    # if args.local_rank == 0:
    print('label_data shape:', len(label_data))
    label_window = np.array([label_data[bias:][i:i+step] for i in range(0,len(label_data[bias:]),stride)])
    one_hot_labels = np.array([one_hot_encode(list(set(x))) for x in label_window])
    # if args.local_rank == 0:
    print('label_data sliding window shape:', one_hot_labels.shape)
    # one_hot_labels = np.array([one_hot_encode(x) for x in labels])

    # valid_indices = ~(np.all(one_hot_labels == [0,0], axis=1))
    valid_indices = np.ones((len(one_hot_labels),), dtype=bool)
    behavior_feat = behavior_feat[valid_indices]
    # audio_feat = audio_feat[valid_indices]
    one_hot_labels = one_hot_labels[valid_indices]

    print('one_hot_labels shape:', len(one_hot_labels))
    print('behavior_feat shape:', len(behavior_feat))
    print('valid_indices shape:', len(valid_indices))
    return behavior_feat, one_hot_labels, valid_indices

class MouseDataset(Dataset):
    def __init__(self, pred_path, label_path):
        super(MouseDataset, self).__init__()

        # self.image_files = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]
        # self.frames_folder = frames_folder
        self.pred_path = pred_path
        self.label_path = label_path
        # self.audio_path = audio_path

        self.sliding_window = True
        self.step = 10
        self.stride = 10
        self.bias = 1
        # len(self.image_files) % self.step
        # self.resampling_rate = args.resampling_rate
        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor()
        # ])

        # load audio
        # self.waveform, self.sample_rate = torchaudio.load(self.audio_path)
        # if args.local_rank == 0:
        #     print('Original audio shape: ', self.waveform.shape)
        # resampled_audio = F.resample(self.waveform, self.sample_rate, self.resampling_rate, rolloff=0.99)
        # if args.local_rank == 0:
        #     print('Resampled audio shape: ', resampled_audio.shape)
        
        # audio_tensor = []
        # for i in range (4650, len(resampled_audio[0]), 50):
        #     arr = np.array(resampled_audio[0, i:i+50])
        #     if(len(arr)==50):
        #         audio_tensor.append(arr)
        #     else:
        #         pass
        # audio_array = np.asarray(audio_tensor)
        # if args.local_rank == 0:
        #     print('Audio shape divided into frames: ', audio_array.shape)

        # load pose prediction
        total_frames = 108211
        pose_pred = self.load_predictions(total_frames)
        pose_keypoints = interploate_pose(pose_pred)

        # load labels
        # if 'CQ' in self.frames_folder:
            # label_data = self.load_CQ_labels(len(self.image_files))
        # elif 'Formalin' in self.frames_folder:
        label_data = self.load_Formalin_labels(total_frames)
        print(len(label_data))
        # sliding window
        if self.sliding_window:
            self.behavior_feat, self.labels, self.valid_indices = sliding_window(pose_keypoints, label_data)
        self.all_indices = np.arange((total_frames - self.bias - self.step) // self.stride + 1)

    def read_label(self, image_file):
        label = image_file.split('_')[-1].split('.')[0]
        return int(label)

    def load_predictions(self, total_frames):
        with open(self.pred_path) as f:
            pose_top = json.load(f)
            # print(len(pose_top))

        # Single-mouse pose_pred
        pose_pred = [[0]] * total_frames
        # Sort annotations
        for i in range(len(pose_top)):
            image_id = pose_top[i]['image_id']
            if not pose_pred[image_id] == [0]:
                if pose_top[i]['score'] > pose_pred[image_id]['score']:
                    pose_pred[image_id] = pose_top[i]
            else:
                pose_pred[image_id] = pose_top[i]

        # multi-mouse pose_pred
        # pose_pred = [[0]]*np.array(video_data).shape[0]
        # # sort annotations
        # for i in range(len(pose_top)):
        #     image_id = pose_top[i]['image_id']
        #     if pose_pred[image_id] == [0]:
        #         pose_pred[image_id] = [pose_top[i]]
        #     else:
        #         pose_pred[image_id].append(pose_top[i])

        return pose_pred

    def load_CQ_labels(self, total_frames):
        label_dataframe = pd.read_excel(self.label_path)
        
        label_data = ['no behavior'] * total_frames
        for index, record in label_dataframe.iterrows():
            start_frame = time_to_frame(record[0])
            if pd.isna(record[1]) and index + 1 < len(label_dataframe):
                end_frame = time_to_frame(label_dataframe.iloc[index + 1, 0])
            else:
                end_frame = time_to_frame(record[1])
            behavior = record[2]
            for i in range(start_frame, end_frame):
                label_data[i] = behavior
        return label_data

    def load_Formalin_labels(self, total_frames):
        label_dataframe = pd.read_excel(self.label_path)

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
        # load image
        # actual_idx = np.where(self.all_indices)[0][idx]
        # start_idx = self.bias + actual_idx * self.stride
        # end_idx = start_idx + self.step
        # if end_idx > len(self.image_files):
        #     print(start_idx, end_idx)
        #     print(len(self.image_files))
        #     print(self.bias, idx, self.stride)

        # images = []
        # for i in range(start_idx, end_idx):
        #     image_path = os.path.join(self.frames_folder, self.image_files[i])
        #     image = Image.open(image_path)
        #     if self.transform:
        #         image = self.transform(image)
        #     images.append(image)
        # images_tensor = torch.stack(images)
        
        # iterate labels
        labels = self.labels[idx]
        labels_tensor = torch.tensor(labels)

        # iterate behavior_feat
        behavior_feat = self.behavior_feat[idx]   
        behavior_feat_tensor = torch.tensor(behavior_feat)

        #iterate over audio features
        # audio_feat = self.audio_feat[idx]   
        # audio_feat_tensor = torch.tensor(audio_feat)

        return behavior_feat_tensor, labels_tensor
