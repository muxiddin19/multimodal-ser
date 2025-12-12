"""
SOTA Multimodal Speech Emotion Recognition
Fixed for BERT (768) + ECAPA-TDNN (192) features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score, 
    confusion_matrix, classification_report
)
import numpy as np
import random
from collections import Counter

from training.CustomizedDataset import CustomizedDataset

# ============================================================
# 1. CONFIGURATION - FIXED DIMENSIONS
# ============================================================
class Config:
    # Model - FIXED: Match your actual feature dimensions
    text_dim = 768   # BERT
    audio_dim =  768  # emotion2vec 192  # ECAPA-TDNN (your actual dimension)
    hidden_dim = 256
    num_heads = 8
    num_layers = 2
    num_classes = 4  # IEMOCAP 4-class
    dropout = 0.3
    
    # Training
    batch_size = 16
    lr = 2e-5
    weight_decay = 0.01
    epochs = 100
    patience = 15
    warmup_ratio = 0.1
    
    # VAD values for emotions
    vad_dict = {
        0: [-0.666, 0.730, 0.314],   # anger
        1: [0.960, 0.648, 0.588],    # happiness
        2: [-0.062, -0.632, -0.286], # neutral
        3: [-0.896, -0.424, -0.672], # sadness
    }
    
    seed = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ============================================================
# 2. GATED CROSS-MODAL ATTENTION
# ============================================================
class GatedCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.a2t_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.t2a_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.audio_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.norm_a = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn_a = nn.LayerNorm(dim)
        self.norm_ffn_t = nn.LayerNorm(dim)
        
    def forward(self, audio_feat, text_feat):
        a2t_out, a2t_w = self.a2t_attn(audio_feat, text_feat, text_feat)
        t2a_out, t2a_w = self.t2a_attn(text_feat, audio_feat, audio_feat)
        
        audio_combined = torch.cat([audio_feat, a2t_out], dim=-1)
        audio_gate = self.audio_gate(audio_combined)
        audio_out = self.norm_a(audio_feat + audio_gate * a2t_out)
        
        text_combined = torch.cat([text_feat, t2a_out], dim=-1)
        text_gate = self.text_gate(text_combined)
        text_out = self.norm_t(text_feat + text_gate * t2a_out)
        
        audio_out = self.norm_ffn_a(audio_out + self.ffn(audio_out))
        text_out = self.norm_ffn_t(text_out + self.ffn(text_out))
        
        return audio_out, text_out, (a2t_w, t2a_w)


# ============================================================
# 3. CONTRASTIVE LEARNING
# ============================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        mask_sum = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
        
        return -mean_log_prob.mean()


# ============================================================
# 4. MAIN MODEL
# ============================================================
class SOTAMultimodalSER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Separate projections for different input dimensions
        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_dim, config.hidden_dim),  # 192 -> 256
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),   # 768 -> 256
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Modality embeddings
        self.audio_embed = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.text_embed = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Self-attention
        self.audio_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )
        
        self.text_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )
        
        # Cross-modal attention layers
        self.cross_attn_layers = nn.ModuleList([
            GatedCrossModalAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # VAD regression head
        self.vad_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # Contrastive projection
        self.contrast_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 128)
        )
        
        self.modality_dropout = 0.1
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, audio_feat, text_feat):
        # Input: audio_feat [B, 192], text_feat [B, 768]
        
        # Add sequence dimension if needed
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)  # [B, 1, 192]
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)    # [B, 1, 768]
        
        # Project to common hidden dimension
        audio_h = self.audio_proj(audio_feat)  # [B, 1, 256]
        text_h = self.text_proj(text_feat)     # [B, 1, 256]
        
        # Add modality embeddings
        audio_h = audio_h + self.audio_embed
        text_h = text_h + self.text_embed
        
        # Modality dropout during training
        if self.training and random.random() < self.modality_dropout:
            if random.random() < 0.5:
                audio_h = torch.zeros_like(audio_h)
            else:
                text_h = torch.zeros_like(text_h)
        
        # Self-attention
        audio_h = self.audio_self_attn(audio_h)
        text_h = self.text_self_attn(text_h)
        
        # Cross-modal attention
        for layer in self.cross_attn_layers:
            audio_h, text_h, _ = layer(audio_h, text_h)
        
        # Pool
        audio_pooled = audio_h.mean(dim=1)
        text_pooled = text_h.mean(dim=1)
        
        # Fusion
        fused = torch.cat([audio_pooled, text_pooled], dim=-1)
        fused = self.fusion(fused)
        
        # Outputs
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        vad = self.vad_head(fused)
        contrast_feat = self.contrast_proj(fused)
        
        return {
            'logits': logits,
            'probs': probs,
            'vad': vad,
            'features': fused,
            'contrast_features': contrast_feat
        }


# ============================================================
# 5. MULTI-TASK LOSS
# ============================================================
class MultiTaskLoss(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)
        
    def forward(self, outputs, labels):
        device = labels.device
        
        # Classification loss
        cls_loss = self.ce_loss(outputs['logits'], labels)
        
        # VAD targets from labels
        vad_targets = torch.tensor([
            self.config.vad_dict[l.item()] for l in labels
        ], dtype=torch.float32).to(device)
        vad_loss = self.mse_loss(outputs['vad'], vad_targets)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(outputs['contrast_features'], labels)
        
        # Combined loss
        total_loss = cls_loss + 0.3 * vad_loss + 0.2 * contrastive_loss
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'vad': vad_loss,
            'contrastive': contrastive_loss
        }


# ============================================================
# 6. EVALUATION
# ============================================================
def evaluate(model, dataloader, device, config):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for text_feat, audio_feat, labels in dataloader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            
            outputs = model(audio_feat, text_feat)
            preds = outputs['probs'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        'WA': accuracy_score(all_labels, all_preds),
        'UA': balanced_accuracy_score(all_labels, all_preds),
        'WF1': f1_score(all_labels, all_preds, average='weighted'),
        'Macro_F1': f1_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }


# ============================================================
# 7. TRAINING
# ============================================================
def get_class_weights(dataset, device):
    labels = [dataset[i][2].item() if torch.is_tensor(dataset[i][2]) else dataset[i][2] 
              for i in range(len(dataset))]
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(len(counts))], 
                          dtype=torch.float32)
    return weights.to(device)


def train_and_evaluate(config, train_dataset, val_dataset, num_runs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*50}")
        
        set_seed(config.seed + run)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        model = SOTAMultimodalSER(config).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        class_weights = get_class_weights(train_dataset, device)
        criterion = MultiTaskLoss(config, class_weights)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        total_steps = len(train_loader) * config.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr * 10,
            total_steps=total_steps,
            pct_start=config.warmup_ratio
        )
        
        best_ua = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []
            
            for text_feat, audio_feat, labels in train_loader:
                text_feat = text_feat.to(device)
                audio_feat = audio_feat.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(audio_feat, text_feat)
                losses = criterion(outputs, labels)
                
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += losses['total'].item()
                train_preds.extend(outputs['probs'].argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_wa = accuracy_score(train_labels, train_preds)
            train_ua = balanced_accuracy_score(train_labels, train_preds)
            
            # Validation
            val_results = evaluate(model, val_loader, device, config)
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {total_loss/len(train_loader):.4f}, WA: {train_wa:.4f}, UA: {train_ua:.4f} | "
                  f"Val WA: {val_results['WA']:.4f}, UA: {val_results['UA']:.4f}, WF1: {val_results['WF1']:.4f}")
            
            if val_results['UA'] > best_ua:
                best_ua = val_results['UA']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best UA: {best_ua:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final = evaluate(model, val_loader, device, config)
        all_results.append(final)
        
        print(f"\nRun {run+1} Best: WA={final['WA']:.4f}, UA={final['UA']:.4f}, WF1={final['WF1']:.4f}")
        print(f"Confusion Matrix:\n{final['confusion_matrix']}")
    
    # Summary
    wa = [r['WA'] for r in all_results]
    ua = [r['UA'] for r in all_results]
    wf1 = [r['WF1'] for r in all_results]
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS ({num_runs} runs)")
    print(f"{'='*50}")
    print(f"WA:  {np.mean(wa)*100:.2f} ± {np.std(wa)*100:.2f}")
    print(f"UA:  {np.mean(ua)*100:.2f} ± {np.std(ua)*100:.2f}")
    print(f"WF1: {np.mean(wf1)*100:.2f} ± {np.std(wf1)*100:.2f}")
    
    return all_results


# ============================================================
# 8. MAIN
# ============================================================
if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)
    
    train_dataset = CustomizedDataset("features/IEMOCAP_BERT_ECAPA_train.pkl")
    val_dataset = CustomizedDataset("features/IEMOCAP_BERT_ECAPA_val.pkl")
    
    # Verify dimensions
    sample = train_dataset[0]
    print(f"Text feature dim: {sample[0].shape}")
    print(f"Audio feature dim: {sample[1].shape}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    results = train_and_evaluate(config, train_dataset, val_dataset, num_runs=5)