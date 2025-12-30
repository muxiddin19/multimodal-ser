"""
Ablation Study for ACL 2026 Submission
=======================================

Systematic removal of each component to quantify contribution:

1. Full Model (all components)
2. w/o VAD-Guided Attention (λ=0)
3. w/o Constrained Fusion (use standard EAAF)
4. w/o Hard Negative Mining (standard MICL)
5. w/o Focal Loss (use CrossEntropy)
6. w/o Data Augmentation (no mixup/noise)
7. w/o MICL entirely (micl_weight=0)
8. Audio-only baseline
9. Text-only baseline

Each configuration runs 5 times for statistical significance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import numpy as np
import random
from collections import Counter
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

# Import our models
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_components import (
    EnhancedMultimodalSER,
    EnhancedMultiTaskLoss,
    FocalLoss,
    ConstrainedAdaptiveFusion,
    HardNegativeMICL
)
from models.novel_components import (
    NovelMultimodalSER,
    NovelMultiTaskLoss,
    EmotionAwareAdaptiveFusion
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MultimodalEmotionDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text_embed', item.get('text_features'))
        audio = item.get('audio_embed', item.get('audio_features'))
        label = item.get('label', item.get('emotion'))

        if not torch.is_tensor(text):
            text = torch.tensor(text, dtype=torch.float32)
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return text.float(), audio.float(), label.long()


def get_class_weights(dataset, device, num_classes):
    labels = [dataset[i][2].item() for i in range(len(dataset))]
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    name: str
    description: str

    # Model settings
    use_vga: bool = True           # VAD-Guided Attention
    vad_lambda: float = 0.5
    use_constrained_fusion: bool = True
    use_hard_negatives: bool = True
    use_focal_loss: bool = True
    use_augmentation: bool = True
    use_micl: bool = True
    micl_weight: float = 0.3

    # Modality settings
    use_text: bool = True
    use_audio: bool = True

    # Training
    num_runs: int = 5
    epochs: int = 100
    patience: int = 15
    batch_size: int = 16
    lr: float = 2e-5


# Define ablation configurations
ABLATION_CONFIGS = [
    AblationConfig(
        name="full_model",
        description="Full Enhanced Model (all components)",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=True,
        use_hard_negatives=True,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_vga",
        description="w/o VAD-Guided Attention (λ=0)",
        use_vga=False, vad_lambda=0.0,
        use_constrained_fusion=True,
        use_hard_negatives=True,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_constrained_fusion",
        description="w/o Constrained Fusion (standard gates)",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=False,
        use_hard_negatives=True,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_hard_negatives",
        description="w/o Hard Negative Mining (standard MICL)",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=True,
        use_hard_negatives=False,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_focal_loss",
        description="w/o Focal Loss (standard CE)",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=True,
        use_hard_negatives=True,
        use_focal_loss=False,
        use_augmentation=True,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_augmentation",
        description="w/o Data Augmentation",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=True,
        use_hard_negatives=True,
        use_focal_loss=True,
        use_augmentation=False,
        use_micl=True, micl_weight=0.3
    ),
    AblationConfig(
        name="no_micl",
        description="w/o MICL Loss",
        use_vga=True, vad_lambda=0.5,
        use_constrained_fusion=True,
        use_hard_negatives=True,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=False, micl_weight=0.0
    ),
    AblationConfig(
        name="audio_only",
        description="Audio-only Baseline",
        use_vga=False, vad_lambda=0.0,
        use_constrained_fusion=False,
        use_hard_negatives=False,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=False, micl_weight=0.0,
        use_text=False, use_audio=True
    ),
    AblationConfig(
        name="text_only",
        description="Text-only Baseline",
        use_vga=False, vad_lambda=0.0,
        use_constrained_fusion=False,
        use_hard_negatives=False,
        use_focal_loss=True,
        use_augmentation=True,
        use_micl=False, micl_weight=0.0,
        use_text=True, use_audio=False
    ),
]


class AblationModel(nn.Module):
    """Flexible model for ablation study with toggleable components."""

    def __init__(
        self,
        config: AblationConfig,
        text_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 384,
        num_classes: int = 5
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_text = config.use_text
        self.use_audio = config.use_audio

        # Input projections
        if self.use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )

        if self.use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )

        # Cross-attention (with or without VGA)
        if self.use_text and self.use_audio:
            if config.use_vga and config.vad_lambda > 0:
                from models.novel_components import VADGuidedBidirectionalAttention
                self.cross_attn = VADGuidedBidirectionalAttention(
                    hidden_dim, num_heads=8, dropout=0.3, vad_lambda=config.vad_lambda
                )
            else:
                # Standard cross-attention
                self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.3, batch_first=True)
                self.cross_attn_norm = nn.LayerNorm(hidden_dim)

            # Fusion
            if config.use_constrained_fusion:
                self.fusion = ConstrainedAdaptiveFusion(hidden_dim, dropout=0.3)
            else:
                self.fusion = EmotionAwareAdaptiveFusion(hidden_dim, dropout=0.3)

        # MICL projector
        if config.use_micl and self.use_text and self.use_audio:
            from models.novel_components import MICLProjector
            self.micl_projector = MICLProjector(hidden_dim, 128, hidden_dim)
            if config.use_hard_negatives:
                self.micl_loss = HardNegativeMICL(temperature=0.07)
            else:
                from models.novel_components import ModalityInvariantContrastiveLoss
                self.micl_loss = ModalityInvariantContrastiveLoss(temperature=0.07)

        # Classifier
        input_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_feat, audio_feat, labels=None):
        # Handle single modality
        if not self.use_text:
            h = self.audio_proj(audio_feat.unsqueeze(1) if audio_feat.dim() == 2 else audio_feat)
            h = h.mean(dim=1)
            logits = self.classifier(h)
            return {'logits': logits, 'probs': torch.softmax(logits, dim=-1), 'micl_loss': torch.tensor(0.0)}

        if not self.use_audio:
            h = self.text_proj(text_feat.unsqueeze(1) if text_feat.dim() == 2 else text_feat)
            h = h.mean(dim=1)
            logits = self.classifier(h)
            return {'logits': logits, 'probs': torch.softmax(logits, dim=-1), 'micl_loss': torch.tensor(0.0)}

        # Both modalities
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        text_h = self.text_proj(text_feat)
        audio_h = self.audio_proj(audio_feat)

        # Cross-attention
        if self.config.use_vga and self.config.vad_lambda > 0:
            audio_h, text_h = self.cross_attn(audio_h, text_h)
        else:
            # Standard cross-attention
            attn_out, _ = self.cross_attn(text_h, audio_h, audio_h)
            text_h = self.cross_attn_norm(text_h + attn_out)

        # Pool
        text_pooled = text_h.mean(dim=1)
        audio_pooled = audio_h.mean(dim=1)

        # Fusion
        fused, fusion_aux = self.fusion(text_pooled, audio_pooled)

        # MICL loss
        micl_loss = torch.tensor(0.0, device=text_feat.device)
        if self.config.use_micl and hasattr(self, 'micl_projector'):
            text_proj, audio_proj = self.micl_projector(text_pooled, audio_pooled)
            if self.config.use_hard_negatives:
                micl_loss, _ = self.micl_loss(text_proj, audio_proj, labels)
            else:
                micl_loss, _ = self.micl_loss(text_proj, audio_proj, labels)

        # Classify
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'probs': torch.softmax(logits, dim=-1),
            'micl_loss': micl_loss,
            'text_gate': fusion_aux.get('text_gate', torch.tensor(0.5)),
            'audio_gate': fusion_aux.get('audio_gate', torch.tensor(0.5))
        }


def train_single_run(
    config: AblationConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    run_idx: int,
    device: torch.device,
    num_classes: int = 5
) -> Dict:
    """Train one run of an ablation configuration."""

    set_seed(42 + run_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = AblationModel(config, num_classes=num_classes).to(device)

    # Loss function
    class_weights = get_class_weights(train_dataset, device, num_classes)
    if config.use_focal_loss:
        cls_criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr * 10,
        total_steps=len(train_loader) * config.epochs,
        pct_start=0.1
    )

    best_ua = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        for text_feat, audio_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(text_feat, audio_feat, labels)

            cls_loss = cls_criterion(outputs['logits'], labels)
            micl_loss = outputs['micl_loss'] if config.use_micl else 0
            total_loss = cls_loss + config.micl_weight * micl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for text_feat, audio_feat, labels in val_loader:
                text_feat = text_feat.to(device)
                audio_feat = audio_feat.to(device)
                outputs = model(text_feat, audio_feat)
                preds = outputs['probs'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_ua = balanced_accuracy_score(all_labels, all_preds)

        if val_ua > best_ua:
            best_ua = val_ua
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Final evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_feat, audio_feat, labels in val_loader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            outputs = model(text_feat, audio_feat)
            preds = outputs['probs'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        'UA': balanced_accuracy_score(all_labels, all_preds),
        'WA': accuracy_score(all_labels, all_preds),
        'WF1': f1_score(all_labels, all_preds, average='weighted'),
        'Macro_F1': f1_score(all_labels, all_preds, average='macro')
    }


def run_ablation_study(
    train_path: str,
    val_path: str,
    num_classes: int = 5,
    output_path: str = "results/ablation_study.json"
):
    """Run complete ablation study."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dataset = MultimodalEmotionDataset(train_path)
    val_dataset = MultimodalEmotionDataset(val_path)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    all_results = {}

    for config in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")

        run_results = []
        for run in range(config.num_runs):
            print(f"  Run {run+1}/{config.num_runs}...", end=" ", flush=True)
            result = train_single_run(config, train_dataset, val_dataset, run, device, num_classes)
            run_results.append(result)
            print(f"UA: {result['UA']*100:.2f}%")

        # Aggregate
        metrics = ['UA', 'WA', 'WF1', 'Macro_F1']
        summary = {'config': config.description, 'runs': run_results}
        for m in metrics:
            values = [r[m] for r in run_results]
            summary[m] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

        all_results[config.name] = summary
        print(f"  Mean UA: {summary['UA']['mean']*100:.2f} ± {summary['UA']['std']*100:.2f}%")

    # Statistical significance tests
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)

    full_model_ua = all_results['full_model']['UA']['values']
    significance_results = {}

    for name, result in all_results.items():
        if name == 'full_model':
            continue

        other_ua = result['UA']['values']

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(full_model_ua, other_ua)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = stats.wilcoxon(full_model_ua, other_ua)
        except:
            w_stat, w_pvalue = np.nan, np.nan

        significance_results[name] = {
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'w_statistic': float(w_stat) if not np.isnan(w_stat) else None,
            'w_pvalue': float(w_pvalue) if not np.isnan(w_pvalue) else None,
            'significant_005': t_pvalue < 0.05,
            'significant_001': t_pvalue < 0.01
        }

        sig_marker = "**" if t_pvalue < 0.01 else ("*" if t_pvalue < 0.05 else "")
        diff = (all_results['full_model']['UA']['mean'] - result['UA']['mean']) * 100
        print(f"  {name}: Δ={diff:+.2f}%, p={t_pvalue:.4f} {sig_marker}")

    all_results['statistical_tests'] = significance_results

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':<35} {'UA':>12} {'Δ UA':>10} {'p-value':>10}")
    print("-"*80)

    full_ua = all_results['full_model']['UA']['mean']
    for name in ['full_model'] + [c.name for c in ABLATION_CONFIGS if c.name != 'full_model']:
        result = all_results[name]
        ua = result['UA']['mean']
        std = result['UA']['std']
        diff = (ua - full_ua) * 100 if name != 'full_model' else 0

        if name in significance_results:
            pval = significance_results[name]['t_pvalue']
            pval_str = f"{pval:.4f}" if pval >= 0.001 else "<0.001"
        else:
            pval_str = "-"

        print(f"{result['config']:<35} {ua*100:>6.2f}±{std*100:.2f}% {diff:>+8.2f}% {pval_str:>10}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/ablation_study.json")
    args = parser.parse_args()

    run_ablation_study(args.train, args.val, args.num_classes, args.output)
