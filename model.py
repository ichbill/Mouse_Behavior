import torch
import torch.nn as nn
import torchvision.models as models

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

        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.image_fc = nn.Linear(512, 256) 

        self.meta_fc = nn.Sequential(
            nn.Linear(num_meta_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 256)
        )

        self.classifier = nn.Linear(256*2, num_classes)

    def forward_features(self, image, meta):
        batch_size, num_frames, c, h, w = image.shape
        image = image.view(batch_size*num_frames, c, h, w)

        image = self.resnet(image)
        image = image.view(batch_size, num_frames, -1)
        image = self.image_fc(image)
        image = image.mean(dim=1) 

        meta = meta.to(torch.float32)
        meta = self.meta_fc(meta)
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

        self.audio_fc = nn.Sequential(
            nn.Linear(num_audio_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Linear(256*2, num_classes)

    def forward_features(self, audio):
        audio = audio.to(torch.float32)
        meta = self.audio_fc(audio)
        meta = meta.mean(dim=1)
        return meta

    def forward(self, audio):
        meta = self.forward_features(audio)
        output = self.classifier(meta)
        return output