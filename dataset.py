import os
import torch
import torchaudio
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import json
import numpy as np
import pandas as pd

def one_hot_encode(labels):
    one_hot_labels = [0,0,0]
    if ' habituation' in labels:
        one_hot_labels[0] = 1
    if ' flinching' in labels:
        one_hot_labels[1] = 1
    if ' licking' in labels:
        one_hot_labels[2] = 1
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
        time_str = time_str[:5] # handle 29:48:00 in Formalin_acute_pain_1.csv
    if '.' in time_str:
        minutes, seconds = map(int, time_str.split('.'))
    elif ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds * 30

def sliding_window(pose_keypoints, label_data, step=10, stride=10):
    # sliding window for pose
    bias = len(pose_keypoints)%step
    behavior_feat = np.array([pose_keypoints[bias:][i:i+step] for i in range(0,len(pose_keypoints[bias:]),stride)])

    # sliding window for labels
    label_window = np.array([label_data[bias:][i:i+step] for i in range(0,len(label_data[bias:]),stride)])
    one_hot_labels = np.array([one_hot_encode(list(set(x))) for x in label_window])
    # one_hot_labels = np.array([one_hot_encode(x) for x in labels])

    valid_indices = ~(np.all(one_hot_labels == [0,0,0], axis=1))
    behavior_feat = behavior_feat[valid_indices]
    one_hot_labels = one_hot_labels[valid_indices]
    # print(behavior_feat.shape, one_hot_labels.shape)
    # breakpoint()

    return behavior_feat, one_hot_labels, valid_indices

class MouseDataset(Dataset):
    def __init__(self, frames_folder, pred_path, label_path, audio_path):
        super(MouseDataset, self).__init__()

        self.image_files = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]
        self.frames_folder = frames_folder
        print(self.frames_folder)
        self.pred_path = pred_path
        self.label_path = label_path
        self.audio_path = audio_path

        self.sliding_window = True
        self.step = 10
        self.stride = 10
        self.bias = len(self.image_files) % self.step
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # load audio
        self.waveform, self.sample_rate = torchaudio.load(self.audio_path)
        
        # load pose prediction
        pose_pred = self.load_predictions(len(self.image_files))
        pose_keypoints = interploate_pose(pose_pred)

        # load labels
        #check the rame folder path
        '''
        if 'CQ' in self.frames_folder:
            print('CQ')
            label_data = self.load_CQ_labels(len(self.image_files))
        elif 'Formalin' in self.frames_folder:
            print('Formalin')
            label_data = self.load_Formalin_labels(len(self.image_files))
        else:
            print("Your frame folder doesn;'t have either CQ nor Formaline")
        '''
        label_data = self.load_Formalin_labels(len(self.image_files))
        # sliding window
        if self.sliding_window:
            self.behavior_feat, self.labels, self.valid_indices = sliding_window(pose_keypoints, label_data)
        self.all_indices = np.arange((len(self.image_files) - self.bias - self.step) // self.stride + 1)

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
        # load image
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

        return images_tensor, behavior_feat_tensor, labels_tensor
