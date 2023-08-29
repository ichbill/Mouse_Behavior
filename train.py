import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from dataset import MouseDataset
from model import MouseModel
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

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

    # tensorboard
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs')

    args = parser.parse_args()

    return args

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.tensorboard:
        writer = SummaryWriter(args.log_dir)

    # dataset
    dataset = MouseDataset(args.video_path, args.pred_path, args.label_path, args.audio_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model
    model = MouseModel(num_meta_features=12, num_classes=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        model.train()
        for steps, (images, behavior_feat, labels) in enumerate(tqdm(train_loader)):
            images, behavior_feat, labels = images.to(device), behavior_feat.to(device), labels.to(device)

            outputs = model(images, behavior_feat)
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.tensorboard:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + steps)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, behavior_feat, labels in test_loader:
                images, behavior_feat, labels = images.to(device), behavior_feat.to(device), labels.to(device)
                outputs = model(images, behavior_feat)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).all(dim=1).sum().item()
                        
        accuracy = 100 * correct / total
        if args.tensorboard:
            writer.add_scalar('Test/Accuracy', accuracy, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")

    print("Training finished!")
    if args.tensorboard:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)