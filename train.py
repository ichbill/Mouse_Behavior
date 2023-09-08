import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from dataset import MouseDataset
from model import MouseModel, VisionModel, AudioModel
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--video_path', default="/data/zhaozhenghao/Projects/Mouse_behavior/dataset/Formalin/frame_folder", type=str, dest='video_path', help='Video path.')
    parser.add_argument('--pred_path', default='/data/zhaozhenghao/Projects/Mouse_behavior/track_result/Formalin/sideview_pose_ckpt1/alphapose-results.json', type=str, dest='pred_path', help='Prediction path.')
    parser.add_argument('--label_path', default='/data/zhaozhenghao/Projects/Mouse_behavior/dataset/Formalin/Formalin_acute_pain_1.csv', type=str, dest='label_path', help='Label path.')
    parser.add_argument('--audio_path', default='/data/zhaozhenghao/Projects/Mouse_behavior/dataset/Formalin/Formalin_Ultrasound_recording.wav', type=str, dest='audio_path', help='Audio path.')

    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    # parameters
    parser.add_argument('--resampling_rate', type=int, default=1500, help='Resampling rate for audio.')
    parser.add_argument('--step', type=int, default=10, help='Step size for sliding window.')
    parser.add_argument('--stride', type=int, default=10, help='Stride size for sliding window.')

    # tensorboard
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # ddp
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    return args

def main(local_rank, args):
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    if args.tensorboard and local_rank == 0:
        writer = SummaryWriter(args.log_dir)

    # dataset
    dataset = MouseDataset(args.video_path, args.pred_path, args.label_path, args.audio_path, args)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, num_workers=0)

    # model
    model = VisionModel(num_meta_features=12, num_classes=3).to(device)
    # model = AudioModel(num_audio_features=50, num_classes=3).to(device)
    # model = MouseModel(num_meta_features=12, num_audio_features=50, num_classes=3).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        model.train()
        for steps, (images, behavior_feat, audio, labels) in enumerate(tqdm(train_loader)):
            images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)

            outputs = model(images, behavior_feat)
            # outputs = model(audio)
            # outputs = model(images, behavior_feat, audio)
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.tensorboard and local_rank == 0:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + steps)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, behavior_feat, audio, labels in test_loader:
                images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)
                outputs = model(images, behavior_feat)
                # outputs = model(audio)
                # outputs = model(images, behavior_feat, audio)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).all(dim=1).sum().item()
                        
        accuracy = 100 * correct / total
        if args.tensorboard and local_rank == 0:
            writer.add_scalar('Test/Accuracy', accuracy, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")

    print("Training finished!")
    if args.tensorboard and local_rank == 0:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args,))