import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4):  # adjust num_classes as needed
        super().__init__()
        concat_dim = 768 + 192  # BERT + ECAPA features = 960
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(concat_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, num_classes)

    def forward(self, text_embed, audio_embed):
        x = torch.cat((text_embed, audio_embed), dim=1)
        x = self.dropout(F.gelu(self.linear1(x)))
        x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))
        logits = self.linear4(x)
        return logits  # softmax removed for CrossEntropyLoss