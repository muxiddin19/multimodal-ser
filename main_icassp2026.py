"""
ICASSP 2026 Submission: Multimodal Speech Emotion Recognition
Addresses reviewer feedback from AAAI-26 rejection:
1. Proper architecture matching paper description
2. Support for 4/6/7 class configurations
3. Comprehensive metrics (WA, UA, WF1, Macro-F1, confusion matrix)
4. Cross-dataset evaluation support
5. Rigorous ablation study framework
6. Statistical significance with confidence intervals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)
import numpy as np
import random
from collections import Counter
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os

# ============================================================
# 1. CONFIGURATION - Reviewer-Addressed Design
# ============================================================
@dataclass
class Config:
    """Configuration aligned with paper and reviewer requirements."""
    # Feature dimensions - support multiple audio encoders
    text_dim: int = 768  # BERT-base
    audio_dim: int = 768  # Wav2Vec2-base (paper) or emotion2vec
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.3

    # Emotion configuration - support multiple class setups
    num_classes: int = 4  # Can be 4, 6, or 7
    emotion_config: str = "iemocap_4"  # iemocap_4, iemocap_6, ekman_7

    # Training
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 15
    warmup_ratio: float = 0.1

    # Multi-run evaluation (reviewer requirement)
    num_runs: int = 5

    # Loss weights
    cls_weight: float = 1.0
    vad_weight: float = 0.3
    contrastive_weight: float = 0.2

    seed: int = 42

    # Dataset paths
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""

    # Ablation flags
    use_audio: bool = True
    use_text: bool = True
    use_cross_attention: bool = True
    use_vad_loss: bool = True
    use_contrastive_loss: bool = True


# VAD values for different emotion configurations (from paper Table 1)
VAD_CONFIGS = {
    "iemocap_4": {
        # anger, happiness, neutral, sadness
        0: [-0.666, 0.730, 0.314],   # anger
        1: [0.960, 0.648, 0.588],    # happiness
        2: [-0.062, -0.632, -0.286], # neutral
        3: [-0.896, -0.424, -0.672], # sadness
    },
    "iemocap_6": {
        # happiness, sadness, neutral, anger, excitement, frustration
        0: [0.960, 0.648, 0.588],    # happiness
        1: [-0.896, -0.424, -0.672], # sadness
        2: [-0.062, -0.632, -0.286], # neutral
        3: [-0.666, 0.730, 0.314],   # anger
        4: [0.850, 0.750, 0.450],    # excitement (similar to joy)
        5: [-0.500, 0.600, -0.200],  # frustration
    },
    "ekman_7": {
        # Ekman's 6 basic emotions + neutral (as per paper)
        0: [0.960, 0.648, 0.588],    # joy
        1: [-0.666, 0.730, 0.314],   # anger
        2: [-0.896, -0.424, -0.672], # sadness
        3: [-0.854, 0.680, -0.414],  # fear
        4: [0.750, 0.750, 0.124],    # surprise
        5: [-0.896, 0.550, -0.366],  # disgust
        6: [-0.062, -0.632, -0.286], # neutral
    }
}

EMOTION_LABELS = {
    "iemocap_4": ["anger", "happiness", "neutral", "sadness"],
    "iemocap_6": ["happiness", "sadness", "neutral", "anger", "excitement", "frustration"],
    "ekman_7": ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"],
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. DATASET - Flexible for multiple datasets
# ============================================================
class MultimodalEmotionDataset(Dataset):
    """Dataset supporting multiple formats and datasets."""

    def __init__(self, data_path: str, dataset_type: str = "iemocap"):
        super().__init__()
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        text = item['text_embed']
        if not torch.is_tensor(text):
            text = torch.tensor(text, dtype=torch.float32)
        else:
            text = text.clone().detach().float()

        audio = item['audio_embed']
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        else:
            audio = audio.clone().detach().float()

        label = item['label']
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return text, audio, label


# ============================================================
# 3. MODEL COMPONENTS - Matching Paper Description
# ============================================================

class MultiHeadCrossAttention(nn.Module):
    """
    Standard Multi-Head Cross-Attention as described in paper Section 3.2.
    Implements Eq. (4): H'_{A→T} = MHCA(Q=H_A, K=H_T, V=H_T)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, Nq, D]
            key: [B, Nk, D]
            value: [B, Nk, D]
        Returns:
            output: [B, Nq, D]
        """
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention Module as per paper Figure 3.
    Audio-to-Text and Text-to-Audio attention.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Audio attends to Text (A→T)
        self.audio_to_text = MultiHeadCrossAttention(dim, num_heads, dropout)
        # Text attends to Audio (T→A)
        self.text_to_audio = MultiHeadCrossAttention(dim, num_heads, dropout)

        # Feed-forward networks
        self.ffn_audio = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.norm_audio = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_feat: [B, Na, D] - Audio features
            text_feat: [B, Nt, D] - Text features
        Returns:
            audio_out: [B, Na, D] - Enhanced audio features
            text_out: [B, Nt, D] - Enhanced text features
        """
        # Cross-attention
        audio_cross, a2t_weights = self.audio_to_text(audio_feat, text_feat, text_feat)
        text_cross, t2a_weights = self.text_to_audio(text_feat, audio_feat, audio_feat)

        # Feed-forward with residual
        audio_out = self.norm_audio(audio_cross + self.ffn_audio(audio_cross))
        text_out = self.norm_text(text_cross + self.ffn_text(text_cross))

        return audio_out, text_out


class SelfAttentionEncoder(nn.Module):
    """Self-attention encoder for unimodal processing."""
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss for emotion embeddings."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mask_sum = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        return -mean_log_prob.mean()


# ============================================================
# 4. MAIN MODEL - Paper-Aligned Architecture
# ============================================================

class SentimentogramSER(nn.Module):
    """
    Multimodal SER model matching paper Section 3.2.

    Architecture:
    1. Feature projection to common dimension
    2. Self-attention for each modality (optional joint encoding)
    3. Bidirectional cross-attention fusion
    4. Classification head
    5. VAD regression head (auxiliary)
    6. Contrastive projection head
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Input projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Modality embeddings (learnable)
        self.text_embed = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)

        # Self-attention encoders
        self.text_self_attn = SelfAttentionEncoder(
            config.hidden_dim, config.num_heads, num_layers=1, dropout=config.dropout
        )
        self.audio_self_attn = SelfAttentionEncoder(
            config.hidden_dim, config.num_heads, num_layers=1, dropout=config.dropout
        )

        # Cross-attention layers (paper: multiple layers)
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(config.hidden_dim, config.num_heads, config.dropout)
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

        # VAD regression head (auxiliary task)
        self.vad_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3),
            nn.Tanh()  # VAD values are in [-1, 1]
        )

        # Contrastive projection head
        self.contrast_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 128)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        use_cross_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_feat: [B, D_text] or [B, N, D_text]
            audio_feat: [B, D_audio] or [B, N, D_audio]
            use_cross_attention: Whether to use cross-attention (for ablation)
        """
        # Add sequence dimension if needed
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        # Project to common dimension
        text_h = self.text_proj(text_feat)  # [B, Nt, D]
        audio_h = self.audio_proj(audio_feat)  # [B, Na, D]

        # Add modality embeddings
        text_h = text_h + self.text_embed
        audio_h = audio_h + self.audio_embed

        # Self-attention
        text_h = self.text_self_attn(text_h)
        audio_h = self.audio_self_attn(audio_h)

        # Cross-attention fusion (if enabled)
        if use_cross_attention and self.config.use_cross_attention:
            for cross_layer in self.cross_attn_layers:
                audio_h, text_h = cross_layer(audio_h, text_h)

        # Pooling (mean over sequence)
        text_pooled = text_h.mean(dim=1)  # [B, D]
        audio_pooled = audio_h.mean(dim=1)  # [B, D]

        # Fusion
        fused = torch.cat([text_pooled, audio_pooled], dim=-1)
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
            'contrast_features': contrast_feat,
            'text_features': text_pooled,
            'audio_features': audio_pooled
        }


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

class MultiTaskLoss(nn.Module):
    """Multi-task loss with classification, VAD regression, and contrastive learning."""

    def __init__(self, config: Config, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.vad_dict = VAD_CONFIGS.get(config.emotion_config, VAD_CONFIGS["iemocap_4"])

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = labels.device

        # Classification loss
        cls_loss = self.ce_loss(outputs['logits'], labels)

        losses = {
            'cls': cls_loss,
            'total': self.config.cls_weight * cls_loss
        }

        # VAD regression loss (if enabled)
        if self.config.use_vad_loss:
            vad_targets = torch.tensor([
                self.vad_dict.get(l.item(), [0, 0, 0]) for l in labels
            ], dtype=torch.float32, device=device)
            vad_loss = self.mse_loss(outputs['vad'], vad_targets)
            losses['vad'] = vad_loss
            losses['total'] = losses['total'] + self.config.vad_weight * vad_loss

        # Contrastive loss (if enabled)
        if self.config.use_contrastive_loss:
            contrastive_loss = self.contrastive_loss(outputs['contrast_features'], labels)
            losses['contrastive'] = contrastive_loss
            losses['total'] = losses['total'] + self.config.contrastive_weight * contrastive_loss

        return losses


# ============================================================
# 6. COMPREHENSIVE METRICS (Reviewer Requirement)
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, emotion_labels: List[str]) -> Dict:
    """
    Compute comprehensive metrics as requested by reviewers:
    - WA (Weighted Accuracy)
    - UA (Unweighted Accuracy / Balanced Accuracy) - PRIMARY METRIC
    - WF1 (Weighted F1)
    - Macro-F1
    - Per-class metrics
    - Confusion matrix
    """
    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(emotion_labels[:len(set(y_true))]):
        per_class[label] = {
            'precision': float(precision[i]) if i < len(precision) else 0,
            'recall': float(recall[i]) if i < len(recall) else 0,
            'f1': float(f1[i]) if i < len(f1) else 0,
            'support': int(support[i]) if i < len(support) else 0
        }

    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'WA': wa,
        'UA': ua,
        'WF1': wf1,
        'Macro_F1': macro_f1,
        'per_class': per_class,
        'confusion_matrix': conf_matrix.tolist()
    }


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config
) -> Dict:
    """Evaluate model and compute comprehensive metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for text_feat, audio_feat, labels in dataloader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)

            # Handle ablation: zero out modality if disabled
            if not config.use_text:
                text_feat = torch.zeros_like(text_feat)
            if not config.use_audio:
                audio_feat = torch.zeros_like(audio_feat)

            outputs = model(text_feat, audio_feat, use_cross_attention=config.use_cross_attention)
            preds = outputs['probs'].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    emotion_labels = EMOTION_LABELS.get(config.emotion_config, EMOTION_LABELS["iemocap_4"])
    return compute_metrics(all_labels, all_preds, emotion_labels)


# ============================================================
# 8. TRAINING
# ============================================================

def get_class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    """Calculate class weights for imbalanced data."""
    labels = [dataset[i][2].item() if torch.is_tensor(dataset[i][2]) else dataset[i][2]
              for i in range(len(dataset))]
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(counts)
    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    )
    return weights.to(device)


def train_single_run(
    config: Config,
    train_dataset: Dataset,
    val_dataset: Dataset,
    run_idx: int,
    device: torch.device
) -> Dict:
    """Train model for a single run."""

    set_seed(config.seed + run_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    model = SentimentogramSER(config).to(device)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    class_weights = get_class_weights(train_dataset, device)
    criterion = MultiTaskLoss(config, class_weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr * 10,
        total_steps=total_steps,
        pct_start=config.warmup_ratio
    )

    best_ua = 0
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for text_feat, audio_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            labels = labels.to(device)

            # Handle ablation
            if not config.use_text:
                text_feat = torch.zeros_like(text_feat)
            if not config.use_audio:
                audio_feat = torch.zeros_like(audio_feat)

            optimizer.zero_grad()
            outputs = model(text_feat, audio_feat, use_cross_attention=config.use_cross_attention)
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

        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / len(train_loader),
            'train_wa': train_wa,
            'train_ua': train_ua,
            'val_wa': val_results['WA'],
            'val_ua': val_results['UA'],
            'val_wf1': val_results['WF1'],
            'val_macro_f1': val_results['Macro_F1']
        })

        print(f"  Epoch {epoch+1:3d} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train UA: {train_ua:.4f} | "
              f"Val UA: {val_results['UA']:.4f}, WF1: {val_results['WF1']:.4f}")

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    -> New best UA: {best_ua:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_results = evaluate(model, val_loader, device, config)
    final_results['history'] = history

    return final_results, model


def train_and_evaluate(
    config: Config,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None
) -> Dict:
    """
    Train and evaluate with multiple runs for statistical significance.
    Returns mean ± std for all metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config.emotion_config}, {config.num_classes} classes")
    print(f"Ablation settings: text={config.use_text}, audio={config.use_audio}, "
          f"cross_attn={config.use_cross_attention}")

    all_results = []
    all_test_results = []
    best_model = None
    best_val_ua = 0

    for run in range(config.num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{config.num_runs}")
        print(f"{'='*60}")

        run_results, model = train_single_run(
            config, train_dataset, val_dataset, run, device
        )
        all_results.append(run_results)

        # Track best model
        if run_results['UA'] > best_val_ua:
            best_val_ua = run_results['UA']
            best_model = model

        print(f"\nRun {run+1} Val Results:")
        print(f"  WA: {run_results['WA']:.4f}")
        print(f"  UA: {run_results['UA']:.4f}")
        print(f"  WF1: {run_results['WF1']:.4f}")
        print(f"  Macro-F1: {run_results['Macro_F1']:.4f}")

        # Test evaluation if test set provided
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            test_results = evaluate(model, test_loader, device, config)
            all_test_results.append(test_results)
            print(f"\nRun {run+1} Test Results:")
            print(f"  WA: {test_results['WA']:.4f}")
            print(f"  UA: {test_results['UA']:.4f}")
            print(f"  WF1: {test_results['WF1']:.4f}")
            print(f"  Macro-F1: {test_results['Macro_F1']:.4f}")

    # Aggregate validation results with confidence intervals
    metrics = ['WA', 'UA', 'WF1', 'Macro_F1']
    summary = {'validation': {}, 'test': {}}

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION RESULTS ({config.num_runs} runs)")
    print(f"{'='*60}")

    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = 1.96 * std / np.sqrt(len(values))  # 95% CI

        summary['validation'][metric] = {
            'mean': mean,
            'std': std,
            'ci_95': ci_95,
            'values': values
        }

        print(f"{metric}: {mean*100:.2f} ± {std*100:.2f}% (95% CI: ±{ci_95*100:.2f}%)")

    # Average validation confusion matrix
    avg_conf_matrix = np.mean([np.array(r['confusion_matrix']) for r in all_results], axis=0)
    summary['validation']['confusion_matrix'] = avg_conf_matrix.tolist()
    summary['validation']['per_class'] = all_results[-1]['per_class']

    print(f"\nValidation Confusion Matrix (averaged):")
    emotion_labels = EMOTION_LABELS.get(config.emotion_config, EMOTION_LABELS["iemocap_4"])
    print("    " + " ".join([f"{l[:3]:>6}" for l in emotion_labels[:config.num_classes]]))
    for i, row in enumerate(avg_conf_matrix):
        print(f"{emotion_labels[i][:3]:>3} " + " ".join([f"{v:>6.1f}" for v in row]))

    # Aggregate test results if available
    if test_dataset is not None and all_test_results:
        print(f"\n{'='*60}")
        print(f"FINAL TEST RESULTS ({config.num_runs} runs)")
        print(f"{'='*60}")

        for metric in metrics:
            values = [r[metric] for r in all_test_results]
            mean = np.mean(values)
            std = np.std(values)
            ci_95 = 1.96 * std / np.sqrt(len(values))

            summary['test'][metric] = {
                'mean': mean,
                'std': std,
                'ci_95': ci_95,
                'values': values
            }

            print(f"{metric}: {mean*100:.2f} ± {std*100:.2f}% (95% CI: ±{ci_95*100:.2f}%)")

        # Average test confusion matrix
        avg_test_conf_matrix = np.mean([np.array(r['confusion_matrix']) for r in all_test_results], axis=0)
        summary['test']['confusion_matrix'] = avg_test_conf_matrix.tolist()
        summary['test']['per_class'] = all_test_results[-1]['per_class']

        print(f"\nTest Confusion Matrix (averaged):")
        print("    " + " ".join([f"{l[:3]:>6}" for l in emotion_labels[:config.num_classes]]))
        for i, row in enumerate(avg_test_conf_matrix):
            print(f"{emotion_labels[i][:3]:>3} " + " ".join([f"{v:>6.1f}" for v in row]))

    # Save best model
    if best_model is not None:
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/best_{config.emotion_config}_{config.num_classes}class.pt"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': config,
            'val_ua': best_val_ua
        }, model_path)
        print(f"\nBest model saved to {model_path}")

    return summary


# ============================================================
# 9. ABLATION STUDY FRAMEWORK
# ============================================================

def run_ablation_study(
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_config: Config
) -> Dict:
    """
    Run comprehensive ablation study as requested by reviewers.
    Tests:
    1. Text-only
    2. Audio-only
    3. Simple fusion (concat, no cross-attention)
    4. Full model
    """

    ablation_configs = [
        ("Text-only", {"use_text": True, "use_audio": False, "use_cross_attention": False}),
        ("Audio-only", {"use_text": False, "use_audio": True, "use_cross_attention": False}),
        ("Simple Fusion", {"use_text": True, "use_audio": True, "use_cross_attention": False}),
        ("Full Model", {"use_text": True, "use_audio": True, "use_cross_attention": True}),
    ]

    results = {}

    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)

    for name, config_overrides in ablation_configs:
        print(f"\n--- {name} ---")

        # Create config with overrides
        config = Config(
            text_dim=base_config.text_dim,
            audio_dim=base_config.audio_dim,
            hidden_dim=base_config.hidden_dim,
            num_classes=base_config.num_classes,
            emotion_config=base_config.emotion_config,
            num_runs=3,  # Fewer runs for ablation
            **config_overrides
        )

        summary = train_and_evaluate(config, train_dataset, val_dataset)
        results[name] = summary

    # Print ablation summary table
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Configuration':<20} {'WA':>12} {'UA':>12} {'WF1':>12} {'Macro-F1':>12}")
    print("-"*70)

    for name in ["Text-only", "Audio-only", "Simple Fusion", "Full Model"]:
        r = results[name]['validation']
        print(f"{name:<20} "
              f"{r['WA']['mean']*100:>6.2f}±{r['WA']['std']*100:>4.2f} "
              f"{r['UA']['mean']*100:>6.2f}±{r['UA']['std']*100:>4.2f} "
              f"{r['WF1']['mean']*100:>6.2f}±{r['WF1']['std']*100:>4.2f} "
              f"{r['Macro_F1']['mean']*100:>6.2f}±{r['Macro_F1']['std']*100:>4.2f}")

    return results


# ============================================================
# 10. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sentimentogram SER - ICASSP 2026")
    parser.add_argument("--train", type=str, default="features/IEMOCAP_BERT_ECAPA_train.pkl")
    parser.add_argument("--val", type=str, default="features/IEMOCAP_BERT_ECAPA_val.pkl")
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--audio_dim", type=int, default=768, help="Audio feature dimension")
    parser.add_argument("--text_dim", type=int, default=768, help="Text feature dimension")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of emotion classes")
    parser.add_argument("--emotion_config", type=str, default="iemocap_4",
                        choices=["iemocap_4", "iemocap_6", "ekman_7"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    # Hyperparameter tuning arguments
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of cross-attention layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    args = parser.parse_args()

    config = Config(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        num_classes=args.num_classes,
        emotion_config=args.emotion_config,
        num_runs=args.num_runs,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience
    )

    set_seed(config.seed)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalEmotionDataset(args.train)
    val_dataset = MultimodalEmotionDataset(args.val)
    test_dataset = MultimodalEmotionDataset(args.test) if args.test else None

    # Verify dimensions
    sample = train_dataset[0]
    print(f"Text feature dim: {sample[0].shape}")
    print(f"Audio feature dim: {sample[1].shape}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")

    if args.ablation:
        results = run_ablation_study(train_dataset, val_dataset, config)
    else:
        results = train_and_evaluate(config, train_dataset, val_dataset, test_dataset)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
