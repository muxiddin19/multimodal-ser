import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4, text_dim=768, audio_dim=192, hidden_dim=256, dropout=0.5):
        super().__init__()
        
        # Text encoder
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Audio encoder
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text_embed, audio_embed):
        text_feat = self.text_fc(text_embed)
        audio_feat = self.audio_fc(audio_embed)
        
        # Concatenate
        fused = torch.cat([text_feat, audio_feat], dim=-1)
        fused = self.fusion(fused)
        
        logits = self.classifier(fused)
        softmax = torch.softmax(logits, dim=-1)
        
        return logits, softmax