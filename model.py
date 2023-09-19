import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MouseModel(nn.Module):
    def __init__(self, num_meta_features, num_audio_features, num_classes):
        super(MouseModel, self).__init__()

        self.vision_model = VisionModel(num_meta_features, num_classes)
        self.audio_model = AudioModel(num_audio_features, num_classes)
        
        self.classifier = nn.Linear(256*2 + 512, num_classes)

    def forward(self, image, meta, audio):
        vision_output = self.vision_model.forward_features(image, meta)
        audio_output = self.audio_model.forward_features(audio)

        combined = torch.cat((vision_output, audio_output), dim=1)
        output = self.classifier(combined)
        
        return output
    
class VisionModel(nn.Module):
    def __init__(self, num_meta_features, num_classes):
        super(VisionModel, self).__init__()

        #self.resnet = models.resnet18(pretrained=True)
        #self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        #self.image_fc = nn.Linear(512, 256) 

        #self.meta_fc = nn.Sequential(
        #    nn.Linear(num_meta_features, 128),
        #    nn.ReLU(),
        #    nn.Dropout(0.25),
        #    nn.Linear(128, 256)
        #)

        self.audio_fc = nn.Sequential(
            nn.Linear(num_meta_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Linear(256*2, num_classes)

    def forward_features(self, image, meta):
        batch_size, num_frames, c, h, w = image.shape
        image = image.view(batch_size*num_frames, c, h, w)

        #image = self.resnet(image)
        #image = image.view(batch_size, num_frames, -1)
        #image = self.image_fc(image)
        #image = image.mean(dim=1) 

        audio = audio.to(torch.float32)
        meta = self.audio_fc(audio)
        meta = meta.mean(dim=1)

        combined = torch.cat((image, meta), dim=1)
        return combined

    def forward(self, image, meta):
        combined = self.forward_features(image, meta)
        output = self.classifier(combined)
        return output
    

class AudioModel(nn.Module):
    def __init__(self, num_audio_features, num_classes):
        super(AudioModel, self).__init__()

        self.conv1 = nn.Conv3d(2, 8, kernel_size=(3,3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(8)
        self.drop1 = nn.Dropout(0.5)
 
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.drop3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        # self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 1, 1))

        self.flat = nn.Flatten()
        self.drop7 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(24576, 512)
        self.act3 = nn.ReLU()

        self.drop6 = nn.Dropout(0.5)
 
        self.fc6 = nn.Linear(512, 2)
 
    def forward(self, x):
        # input stepx2x64x64, output 32x32x32
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size, c, num_frames, h, w)

        x = self.act1(self.conv1(x))
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        x = self.act3(self.conv3(x))
        x = self.pool3(x)

        x = self.act4(self.conv4(x))
        x = self.pool4(x)

        x = self.act5(self.conv5(x))
        x = self.pool5(x)

        
        x = self.flat(x)
        x = self.drop7(x)
        x = self.act3(self.fc3(x))

        x = self.drop6(x)
        
        x = self.fc6(x)
        return x

