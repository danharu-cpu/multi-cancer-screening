import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseClassifier(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(BaseClassifier, self).__init__()

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

        # Flattened output = (B, 8, 1, 16) → 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        self.output_layer = nn.Linear(32, 2)  # Binary classifier output (HC vs Cancer or LC vs GC)

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 1, 19, 1800)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten (B, 128)
        x = self.classifier(x)     # (B, 32)
        return self.output_layer(x)  # (B, 2)

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

class HD_CNN(nn.Module):
    def __init__(self):
        super(HD_CNN, self).__init__()
        self.coarse = BaseClassifier(dropout_prob=0.65)
        self.fine = BaseClassifier(dropout_prob=0.5)

    def forward(self, x):
        coarse_out = self.coarse(x)  # (B, 2)
        fine_out = self.fine(x)      # (B, 2)

        prob_coarse = F.softmax(coarse_out, dim=1)
        prob_fine = F.softmax(fine_out, dim=1)

        final_logits = torch.zeros(x.size(0), 3).to(x.device)
        final_logits[:, 0] = prob_coarse[:, 0]                      # Healthy
        final_logits[:, 1] = prob_coarse[:, 1] * prob_fine[:, 0]    # Lung
        final_logits[:, 2] = prob_coarse[:, 1] * prob_fine[:, 1]    # Gastric

        return final_logits, coarse_out, fine_out
