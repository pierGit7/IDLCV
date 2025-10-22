
import torch.nn.functional as F
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # input size: 3x128x128
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # ->(64,128,128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # ->(64,128,128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->(64,64,64)
            
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # ->(128,64,64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # ->(128,32,32)
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # ->(128,32,32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # ->(256,16,16)

            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # ->(512,8,8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
            #softmax will be applied in the loss function
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LateFusionVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # reuse the feature extractor from your VGG16
        self.features = VGG16().features

        # spatial pooling to reduce (H, W) → (1, 1)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layer for classification (after temporal pooling)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # rearrange so we can process all frames in batch mode
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]

        # run 2D CNN feature extractor on each frame
        features = self.features(x)  # [B*T, 512, 8, 8]

        # average pool spatially
        features = self.spatial_pool(features)  # [B*T, 512, 1, 1]
        features = features.view(B, T, 512)     # [B, T, 512]

        # average pool temporally across frames
        clip_features = features.mean(dim=1)    # [B, 512]

        # classify the video
        out = self.fc(clip_features)            # [B, num_classes]
        return out
