import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from dataset import MouseDataset
from model import MouseModel, VisionModel, AudioModel
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score

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
    
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    device = torch.device(f"cuda:{local_rank}")

    print(local_rank)
    
    print("Prcocess Started")
    if args.tensorboard and local_rank == 0:
        writer = SummaryWriter(args.log_dir)
    print("Ok")
    dist.init_process_group(backend='nccl', world_size=8, rank=local_rank)
    print("Ok2")
    torch.manual_seed(1234)
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
    # model = VisionModel(num_meta_features=12, num_classes=3).to(device)
    model = AudioModel(num_classes=2).to(device)
    # model = MouseModel(num_meta_features=12, num_audio_features=50, num_classes=3).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    print("Train")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        model.train()
        all_preds = []
        all_labels = []
        total_loss = 0
        for steps, (images, behavior_feat, audio, labels) in enumerate(tqdm(train_loader)):
            images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels.float())
            
            # Perform backward pass
            loss.backward()
        
            # Perform optimization
            optimizer.step()
            total_loss += loss

            # probs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / len(train_loader)
        if args.tensorboard and local_rank == 0:
            writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")


        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for images, behavior_feat, audio, labels in test_loader:
                images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)
                
                outputs = model(audio)
                # outputs = outputs.squeeze(dim=1)
                loss = criterion(outputs, labels.float())
                
                total_loss += loss
                
                # probs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()  # Convert logits to binary predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            
            test_loss = total_loss / len(test_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            if test_loss < best_valid_loss:
                best_valid_loss = test_loss
                checkpoint_path = 'combined_new_best_model_checkpoint.pt'
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
    world_size = 8
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    torch.multiprocessing.spawn(main, args=(args,), nprocs=world_size)