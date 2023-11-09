import torch
import os
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.optim import Adam
import torch.nn as nn
from dataset import MouseDataset
from model import MouseModel, VisionModel, AudioModel, BehaviorModel
from tqdm import tqdm
import argparse
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # data path
    # parser.add_argument('--video_path', default="/data/zhaozhenghao/Projects/Mouse_behavior/dataset/Formalin/frame_folder", type=str, dest='video_path', help='Video path.')
    parser.add_argument('--pred_path', default='alphapose-results.json', type=str, dest='pred_path', help='Prediction path.')
    parser.add_argument('--label_path', default='Formalin_acute_pain_1.xlsx', type=str, dest='label_path', help='Label path.')
    # parser.add_argument('--audio_path', default='/data/zhaozhenghao/Projects/Mouse_behavior/dataset/Formalin/Formalin_Ultrasound_recording.wav', type=str, dest='audio_path', help='Audio path.')

    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)

    # parameters
    parser.add_argument('--resampling_rate', type=int, default=1500, help='Resampling rate for audio.')
    parser.add_argument('--step', type=int, default=10, help='Step size for sliding window.')
    parser.add_argument('--stride', type=int, default=10, help='Stride size for sliding window.')

    # tensorboard
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # ddp
    # parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    return args

def uniform_dist(dataset):
    labels = np.array([labels for _, labels in dataset])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    labels_str = [f"{label_tensor[0].item()}{label_tensor[1].item()}" 
                  for label_tensor in labels]
    # Placeholder for X since we only need to stratify based on the labels
    X_dummy = np.zeros(len(labels))

    # Use 'labels_str' for stratification
    for train_index, test_index in sss.split(X_dummy, labels_str):
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
    
    return train_dataset, test_dataset

def get_flat_labels(dataset):
    labels = []
    for _, label_tensor in dataset:
        labels.append(label_tensor[0].item() * 2 + label_tensor[1].item())  # This maps [0,0]->0, [0,1]->1, [1,0]->2, [1,1]->3
    return labels

# Calculate sample weights
def calculate_weights(dataset):
    flat_labels = get_flat_labels(dataset)
    class_sample_count = torch.tensor([(torch.tensor(flat_labels) == t).sum() for t in torch.unique(torch.tensor(flat_labels), sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in flat_labels])
    print(sample_weights)
    return sample_weights

def main(args):
    # local_rank = args.local_rank
    # dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = 0
    # torch.cuda.set_device(local_rank)
    # device = torch.device('cuda', local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.tensorboard and local_rank == 0:
        writer = SummaryWriter(args.log_dir)

    # dataset
    dataset = MouseDataset(args.pred_path, args.label_path)

    #Maintaining the distribution in train and test set
    train_dataset, test_dataset = uniform_dist(dataset)
    
    #
    train_weights = calculate_weights(train_dataset)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Calculate weights for the testing dataset
    test_weights = calculate_weights(test_dataset)
    test_sampler = WeightedRandomSampler(test_weights, len(test_weights))

    # Now, use the samplers in your DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    # for steps, (behavior_feat, labels) in enumerate(tqdm(train_loader)):
    #     rnn_1_neuron =RNNVanilla(behavior_feat[0][0],1)


    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, num_workers=0)
    # for steps, (behavior_feat, labels) in enumerate(tqdm(train_loader)):
    #     print("behavior_feat : ", behavior_feat)
    #     print("behavior_feat_size :",behavior_feat.shape)
    #     print("behavior_feat[0] : ", behavior_feat[0])
    #     print("behavior_feat[0].size :",behavior_feat[0].shape)
    #     print("labels : ",labels)
    #     print("labels_size :",labels.shape)
    #     break
    
    #'''
    # model
    # model = VisionModel(num_meta_features=12, num_classes=3).to(device)
    # model = AudioModel(num_audio_features=50, num_classes=3).to(device)
    # model = MouseModel(num_meta_features=12, num_audio_features=50, num_classes=3).to(device)
    model = BehaviorModel(num_features=12, num_classes=2).to(device) #
    # model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCELoss()
    ## criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    best_valid_loss = float('inf')
    test_loss_list = []
    test_accuracy_list=[]
    for epoch in range(args.num_epochs):
        model.train()
        all_preds = []
        all_labels = []
        total_loss = 0
        for steps, (behavior_feat, labels) in enumerate(tqdm(train_loader)):
            behavior_feat, labels =  behavior_feat.to(device), labels.to(device)

            # outputs = model(images, behavior_feat)
            outputs = model(behavior_feat)
            # outputs = model(audio)
            # outputs = model(images, behavior_feat, audio)
            # print(outputs, labels)
            loss = criterion(outputs, labels.float())
            # loss = creiterion(outputs, behavior_feat.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(all_labels, all_preds) 
        train_loss = total_loss / len(train_loader)
        if args.tensorboard and local_rank == 0:
            writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")

        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for behavior_feat, labels in test_loader:
                behavior_feat, labels = behavior_feat.to(device), labels.to(device)
                # outputs = model(images, behavior_feat)
                outputs = model(behavior_feat)
                # outputs = model(audio)
                # outputs = model(images, behavior_feat, audio)
                loss = criterion(outputs, labels.float())
                total_loss += loss
                
                # probs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()  # Convert logits to binary predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            
            test_loss = total_loss / len(test_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            test_loss_list.append(test_loss)
            test_accuracy_list.append(val_accuracy)
            if test_loss < best_valid_loss:
                best_valid_loss = test_loss
                checkpoint_path = 'best_model_checkpoint.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    }, checkpoint_path)
                            
            
            if args.tensorboard and local_rank == 0:
                writer.add_scalar('Test/Accuracy', val_accuracy, epoch)
                writer.add_scalar('Test/Loss', test_loss, epoch)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Testing Loss: {test_loss:.4f}, Test Accuracy: {val_accuracy:.2f}")

    print("Training finished!")
    n=[i for i in range(50)]
    if args.tensorboard:
        writer.close()
    plt.subplot(2,1,1)
    plt.plot(n,test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(2,1,2)
    plt.plot(n,test_accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0,1)
    plt.show()

    #'''
if __name__ == "__main__":
    args = parse_args()
    main(args)
    # torch.multiprocessing.spawn(main, args=(args,))