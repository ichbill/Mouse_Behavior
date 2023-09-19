# credits:
# how to use DDP module with DDP sampler: https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
# how to setup a basic DDP example from scratch: https://pytorch.org/tutorials/intermediate/dist_tuto.html
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
import torch.nn as nn
from dataset import MouseDataset
from model import MouseModel, VisionModel, AudioModel
from tqdm import tqdm
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--video_path', default="/home/nikk/dataset/frame_folder", type=str, dest='video_path', help='Video path.')
    parser.add_argument('--pred_path', default='/home/nikk/dataset/alphapose-results.json', type=str, dest='pred_path', help='Prediction path.')
    parser.add_argument('--label_path', default='/home/nikk/dataset/Formalin_acute_pain_1.csv', type=str, dest='label_path', help='Label path.')
    parser.add_argument('--audio_path', default='/home/nikk/dataset/Formalin_Ultrasound_recording.wav', type=str, dest='audio_path', help='Audio path.')

    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)

    # tensorboard
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # ddp
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    return args

def main(args):
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.manual_seed(1234)
    if args.tensorboard and local_rank == 0:
        writer = SummaryWriter(args.log_dir)

    # dataset
    dataset = MouseDataset(args.video_path, args.pred_path, args.label_path, args.audio_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, num_workers=0)

    # model
    #model = VisionModel(num_meta_features=12, num_classes=2).to(device)
    model = AudioModel(num_audio_features=1000, num_classes=2).to(device)
    # model = MouseModel(num_meta_features=12, num_audio_features=1000, num_classes=2).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    #Instead of dropping any postions from our 'licking' and 'fliching' the wieghts, we have used a better option to penalise the weights of each class.
    #A minority class will get a higher weight such as 2.0 for our 'flinching' class.
    pos_weight = torch.tensor([2.0, 1.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    best_valid_loss = float('inf')

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds_train = []
        all_labels_train = []
        model.train()
        for steps, (images, behavior_feat, audio, labels) in enumerate(tqdm(train_loader)):
            images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)

            optimizer.zero_grad()
            # outputs = model(images, behavior_feat)
            outputs = model(audio)
            # outputs = model(images, behavior_feat, audio)
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # Convert logits to binary predictions
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
            
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / len(train_loader)
        if args.tensorboard and local_rank == 0:
            writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, behavior_feat, audio, labels in test_loader:
                images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)

                outputs = model(images, behavior_feat)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()  # Convert logits to binary predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        test_loss = val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
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
    if args.tensorboard:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args,))
