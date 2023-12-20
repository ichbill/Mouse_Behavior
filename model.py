import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import MinMaxScaler


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
        metda = meta.mean(dim=1)
        return meta

    def forward(self, audio):
        meta = self.forward_features(audio)
        output = self.classifier(meta)
        return output

class BehaviorModel(nn.Module):
    def __init__(self,num_features,num_classes):
        super(BehaviorModel,self).__init__()
        self.input_size = num_features
        self.hidden_size= 512        #hidden_size #40
        self.layer_size = 1         #layer_size
        self.output_size = num_classes
        self.window_size= 0
        # self.rnn = nn.RNN(self.input_size, self.hidden_size, self.layer_size, batch_first=True, nonlinearility = 'relu')
        # self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers= self.layer_size, batch_first=True)
        
        # self.layer_norm = nn.LayerNorm(self.hidden_size)
        #self.fc = nn.Sequential(nn.Linear(self.hidden_size*10,self.output_size)) # 10 is window size
        # self.fc  = nn.Linear(self.hidden_size*5,self.output_size, bias=True)
        # for ANN

        self.fc = nn.Sequential(nn.Linear(self.input_size,256),  
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(256,128),                                                             
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(128,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(64,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(32,16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(16,8),
                                nn.BatchNorm1d(8),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(8,self.output_size),
                                nn.BatchNorm1d(self.output_size),
                                )

        # self.fc = nn.Sequential(nn.Linear(self.hidden_size*self.window_size, 8),
        #                         nn.BatchNorm1d(8),
        #                         nn.ReLU(),
        #                         nn.Linear(8,self.output_size),
        #                         nn.BatchNorm1d(self.output_size),
        #                         )
        
        # self.fc = nn.Sequential(nn.Linear(self.hidden_size*5, 64), 
        #                         # nn.ReLU(),
        #                         # nn.Linear(128,256),
        
        #                         # nn.BatchNorm1d(256),
        #                         # nn.ReLU(),
        #                         # nn.Linear(256,128),
        #                         # nn.BatchNorm1d(128),
        #                         # #nn.Dropout(0.1),
        #                         # nn.ReLU(),
        #                         # nn.Linear(128,64),
        #                         # nn.BatchNorm1d(64),
        #                         # # nn.Dropout(0.1),
        #                         # nn.ReLU(),
        #                         nn.Linear(64,32),
        #                         nn.BatchNorm1d(32),
        #                         # nn.Dropout(0.1),
        #                         nn.ReLU(),
        #                         nn.Linear(32,16),
        #                         nn.BatchNorm1d(16),
        #                         # nn.Dropout(0.25),
        #                         nn.ReLU(),
        #                         # nn.Dropout(0.1),
        #                         nn.Linear(16, self.output_size))   

    def forward(self,behavior_feat):
        # Convert input data to torch.float32
        behavior_feat = behavior_feat.to(torch.float32)
        # Instantitate hidden_state at timestamp 0 
        # hidden_state = torch.zeros(self.layer_size, behavior_feat[0], self.hidden_size)
        # behavior_feat.size(0) : batch_size 
        #hidden_state = (num_layer, batch_size, hidden_size)

        # hidden_state = torch.zeros(self.layer_size, behavior_feat.size(0), self.hidden_size)
        # cell_state = torch.zeros(self.layer_size, behavior_feat.size(0), self.hidden_size)
        # hidden_state = hidden_state.requires_grad_()

        # m = nn.BatchNorm1d(1)
        # behavior_feat = m(behavior_feat)
        # output, _ = self.rnn(behavior_feat,hidden_state.detach())
        # output, _ = self.rnn(behavior_feat,(hidden_state.detach(),cell_state.detach()))
        # output = self.layer_norm(output)
        # output = output.reshape(output.shape[0],-1)
        # print(output.shape)
        output = self.fc(behavior_feat)
        return output
    
class VisionModel_JJ(nn.Module):
    def __init__(self, num_meta_features, num_classes):
        super(VisionModel, self).__init__()
        # LeNet-5
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1),
            nn.Tanh())
        self.classfier=nn.Sequential(
            nn.Linear(in_features=120,out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84,out_features=num_classes)
        )
    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x,1)
        logits = self.classfier(x)
        # probs = F.softmax(logits,dim=1)
        return logits








        
        