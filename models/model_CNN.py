# cnn_baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Conv1: full sensor height (19), wide temporal kernel (300), stride 100
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(19, 300), stride=(1, 100)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(inplace=True),

            # Conv2: 1x1 conv to increase feature depth
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True)
        )

        # After conv: output shape = (B, 8, 1, T)
        # T = floor((1800 - 300)/100 + 1) = 16 → 1x16
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (B, 8*1*16) = (B, 128)
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 3)             # Output: HC / LC / GC
        )

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 1, 19, 1800)  # reshape input
        x = self.features(x)
        x = self.classifier(x)
        return x                     # logits (no softmax)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
