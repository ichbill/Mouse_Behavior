import torch
import torch.nn as nn
import torchvision.models as models

class MouseModel(nn.Module):
    def __init__(self, num_meta_features, num_classes):
        super(MouseModel, self).__init__()

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

    def forward(self, audio):
        #batch_size, num_frames, c, h, w = image.shape
        #image = image.view(batch_size*num_frames, c, h, w)

        #image = self.resnet(image)
        #image = image.view(batch_size, num_frames, -1)
        #image = self.image_fc(image)
        #image = image.mean(dim=1) 

        audio = audio.to(torch.float32)
        meta = self.audio_fc(audio)
        meta = meta.mean(dim=1)

        #combined = torch.cat((image, meta), dim=1)
        
        output = self.classifier(meta)
        return output