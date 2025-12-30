"""
ACL 2026: Enhanced Multimodal Speech Emotion Recognition
=========================================================

This script implements our SOTA approach with novel contributions:

CORE CONTRIBUTIONS (from novel_components.py):
1. VAD-Guided Cross-Attention (VGA) - Psychological theory-guided attention
2. Emotion-Aware Adaptive Fusion (EAAF) - Dynamic modality weighting
3. Modality-Invariant Contrastive Learning (MICL) - Cross-modal alignment

ENHANCEMENTS (from enhanced_components.py):
4. Focal Loss - Better class imbalance handling (+3-5% UA expected)
5. Constrained Adaptive Fusion - Gates sum to 1 for interpretability
6. Hard Negative Mining for MICL - Focus on difficult pairs
7. Feature Augmentation - Dropout, noise, mixup
8. Curriculum Learning - Progressive 4→5→6 class training

TARGET: 80%+ UA on IEMOCAP 6-class (vs current SOTA ~78%)

Usage:
    # Train on IEMOCAP 5-class
    python main_acl2026_enhanced.py \
        --train features/IEMOCAP_5class_emotion2vec_train.pkl \
        --val features/IEMOCAP_5class_emotion2vec_val.pkl \
        --test features/IEMOCAP_5class_emotion2vec_test.pkl \
        --emotion_config iemocap_5 --num_classes 5

    # Train on MELD
    python main_acl2026_enhanced.py \
        --train features/MELD_4class_emotion2vec_train.pkl \
        --val features/MELD_4class_emotion2vec_val.pkl \
        --test features/MELD_4class_emotion2vec_test.pkl \
        --emotion_config meld_4 --num_classes 4

    # Train on CREMA-D 6-class
    python main_acl2026_enhanced.py \
        --train features/CREMAD_6class_emotion2vec_train.pkl \
        --val features/CREMAD_6class_emotion2vec_val.pkl \
        --test features/CREMAD_6class_emotion2vec_test.pkl \
        --emotion_config cremad_6 --num_classes 6

    # With curriculum learning
    python main_acl2026_enhanced.py --train ... --curriculum --curriculum_phases 30,60,100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
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
from datetime import datetime

from models.enhanced_components import (
    EnhancedMultimodalSER,
    EnhancedMultiTaskLoss,
    CurriculumScheduler,
    FocalLoss,
    ConstrainedAdaptiveFusion,
    HardNegativeMICL,
    FeatureAugmentation,
    mixup_batch,
    mixup_criterion
)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EnhancedConfig:
    """Configuration for enhanced ACL 2026 experiments."""
    # Feature dimensions
    text_dim: int = 768      # BERT-base
    audio_dim: int = 1024    # emotion2vec_plus_large
    hidden_dim: int = 384
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.3

    # Emotion configuration
    num_classes: int = 5
    emotion_config: str = "iemocap_5"

    # Training
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 15
    warmup_ratio: float = 0.1

    # Multi-run evaluation
    num_runs: int = 5
    seed: int = 42

    # Loss weights (optimized based on research)
    cls_weight: float = 1.0
    vad_weight: float = 0.5   # Increased from 0.3
    micl_weight: float = 0.3  # Increased from 0.2

    # Novel component settings (optimized)
    vad_lambda: float = 0.5   # Increased from 0.1 for stronger VAD guidance
    micl_temp: float = 0.07
    micl_dim: int = 128
    focal_gamma: float = 2.0  # Focal loss focusing parameter

    # Data augmentation
    use_augmentation: bool = True
    mixup_alpha: float = 0.4
    augment_prob: float = 0.5

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_phases: List[int] = field(default_factory=lambda: [30, 60, 100])
    curriculum_classes: List[int] = field(default_factory=lambda: [4, 5, 6])


# Emotion configurations
EMOTION_LABELS = {
    "iemocap_4": ["anger", "happiness", "neutral", "sadness"],
    "iemocap_5": ["happy_excited", "sadness", "neutral", "anger", "frustration"],
    "iemocap_6": ["happiness", "sadness", "neutral", "anger", "excitement", "frustration"],
    "cremad_4": ["anger", "happiness", "neutral", "sadness"],
    "cremad_6": ["anger", "disgust", "fear", "happiness", "neutral", "sadness"],
    "meld_4": ["anger", "joy", "neutral", "sadness"],
    "meld_7": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
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
# DATASET
# ============================================================

class MultimodalEmotionDataset(Dataset):
    """Dataset for multimodal emotion recognition."""

    def __init__(self, data_path: str, emotion_config: str = "iemocap_5"):
        super().__init__()
        self.emotion_config = emotion_config

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        # Text features
        text = item.get('text_embed', item.get('text_features', None))
        if text is None:
            raise KeyError(f"No text features found in item. Keys: {item.keys()}")
        if not torch.is_tensor(text):
            text = torch.tensor(text, dtype=torch.float32)
        else:
            text = text.clone().detach().float()

        # Audio features
        audio = item.get('audio_embed', item.get('audio_features', None))
        if audio is None:
            raise KeyError(f"No audio features found in item. Keys: {item.keys()}")
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
        else:
            audio = audio.clone().detach().float()

        # Label
        label = item.get('label', item.get('emotion', None))
        if label is None:
            raise KeyError(f"No label found in item. Keys: {item.keys()}")
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return text, audio, label

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution for analysis."""
        labels = [self[i][2].item() for i in range(len(self))]
        return dict(Counter(labels))


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, emotion_labels: List[str]) -> Dict:
    """Compute comprehensive metrics."""
    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Per-class metrics
    per_class = {}
    unique_labels = sorted(set(y_true))
    for i, label_idx in enumerate(unique_labels):
        label_name = emotion_labels[label_idx] if label_idx < len(emotion_labels) else f"class_{label_idx}"
        per_class[label_name] = {
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


def get_class_weights(dataset: Dataset, device: torch.device, num_classes: int) -> torch.Tensor:
    """Calculate class weights for focal loss."""
    labels = [dataset[i][2].item() if torch.is_tensor(dataset[i][2]) else dataset[i][2]
              for i in range(len(dataset))]
    counts = Counter(labels)
    total = len(labels)

    # Handle missing classes
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weights.append(total / (num_classes * counts[i]))
        else:
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights.to(device)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: EnhancedConfig
) -> Tuple[Dict, Dict]:
    """Evaluate model and return metrics + auxiliary info."""
    model.eval()
    all_preds, all_labels = [], []
    total_text_gate, total_audio_gate, total_inter_gate = 0.0, 0.0, 0.0
    n_samples = 0

    with torch.no_grad():
        for text_feat, audio_feat, labels in dataloader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)

            outputs = model(text_feat, audio_feat, labels.to(device), apply_augmentation=False)
            preds = outputs['probs'].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Track fusion gate statistics
            total_text_gate += outputs['text_gate'].sum().item()
            total_audio_gate += outputs['audio_gate'].sum().item()
            total_inter_gate += outputs['interaction_gate'].sum().item()
            n_samples += len(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    emotion_labels = EMOTION_LABELS.get(config.emotion_config, EMOTION_LABELS["iemocap_5"])
    metrics = compute_metrics(all_labels, all_preds, emotion_labels)

    # Auxiliary info about enhanced components
    aux_info = {
        'avg_text_gate': total_text_gate / n_samples,
        'avg_audio_gate': total_audio_gate / n_samples,
        'avg_interaction_gate': total_inter_gate / n_samples,
    }

    return metrics, aux_info


# ============================================================
# TRAINING
# ============================================================

def train_single_run(
    config: EnhancedConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    run_idx: int,
    device: torch.device
) -> Tuple[Dict, nn.Module]:
    """Train model for a single run with all enhancements."""

    set_seed(config.seed + run_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Create enhanced model
    model = EnhancedMultimodalSER(
        text_dim=config.text_dim,
        audio_dim=config.audio_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        vad_lambda=config.vad_lambda,
        micl_dim=config.micl_dim,
        use_augmentation=config.use_augmentation
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Enhanced loss function with focal loss
    class_weights = get_class_weights(train_dataset, device, config.num_classes)
    criterion = EnhancedMultiTaskLoss(
        num_classes=config.num_classes,
        emotion_config=config.emotion_config,
        class_weights=class_weights,
        cls_weight=config.cls_weight,
        vad_weight=config.vad_weight,
        micl_weight=config.micl_weight,
        focal_gamma=config.focal_gamma
    )

    # Curriculum scheduler (if enabled)
    curriculum = None
    if config.use_curriculum:
        curriculum = CurriculumScheduler(
            total_epochs=config.epochs,
            phase_epochs=config.curriculum_phases,
            class_schedule=config.curriculum_classes
        )
        print(f"  Curriculum Learning: {config.curriculum_classes} classes over {config.curriculum_phases} epochs")

    # Optimizer and scheduler
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
        total_cls_loss = 0
        total_vad_loss = 0
        total_micl_loss = 0
        train_preds, train_labels = [], []
        n_batches_used = 0

        # Get curriculum info if enabled
        if curriculum:
            phase_info = curriculum.get_phase_info(epoch)
            current_num_classes = phase_info['num_classes']
        else:
            current_num_classes = config.num_classes

        for text_feat, audio_feat, labels in train_loader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            labels = labels.to(device)

            # Apply curriculum mask if enabled
            curriculum_mask = None
            if curriculum:
                curriculum_mask = curriculum.get_class_mask(epoch, labels)
                if curriculum_mask.sum() == 0:
                    continue

            # Apply mixup augmentation
            use_mixup = config.use_augmentation and random.random() < 0.3
            if use_mixup and curriculum_mask is None:
                text_feat, audio_feat, labels_a, labels_b, lam = mixup_batch(
                    text_feat, audio_feat, labels, config.mixup_alpha
                )
            else:
                labels_a = labels_b = labels
                lam = 1.0

            optimizer.zero_grad()
            outputs = model(
                text_feat, audio_feat, labels,
                epoch=epoch, max_epochs=config.epochs
            )

            # Compute loss
            if use_mixup and curriculum_mask is None:
                # Mixup loss
                losses_a = criterion(outputs, labels_a, curriculum_mask)
                losses_b = criterion(outputs, labels_b, curriculum_mask)
                loss = lam * losses_a['total'] + (1 - lam) * losses_b['total']
                losses = {
                    'total': loss,
                    'cls': lam * losses_a['cls'] + (1 - lam) * losses_b['cls'],
                    'vad': lam * losses_a['vad'] + (1 - lam) * losses_b['vad'],
                    'micl': lam * losses_a['micl'] + (1 - lam) * losses_b['micl']
                }
            else:
                losses = criterion(outputs, labels, curriculum_mask)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += losses['total'].item()
            total_cls_loss += losses['cls'].item()
            total_vad_loss += losses['vad'].item()
            total_micl_loss += losses['micl'].item()

            # Track predictions (only for valid samples in curriculum)
            if curriculum_mask is not None:
                valid_preds = outputs['probs'][curriculum_mask].argmax(dim=1)
                valid_labels = labels[curriculum_mask]
                train_preds.extend(valid_preds.cpu().numpy())
                train_labels.extend(valid_labels.cpu().numpy())
            else:
                train_preds.extend(outputs['probs'].argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            n_batches_used += 1

        if n_batches_used == 0:
            continue

        train_ua = balanced_accuracy_score(train_labels, train_preds)

        # Validation
        val_results, val_aux = evaluate(model, val_loader, device, config)

        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / n_batches_used,
            'train_cls_loss': total_cls_loss / n_batches_used,
            'train_vad_loss': total_vad_loss / n_batches_used,
            'train_micl_loss': total_micl_loss / n_batches_used,
            'train_ua': train_ua,
            'val_ua': val_results['UA'],
            'val_wa': val_results['WA'],
            'text_gate': val_aux['avg_text_gate'],
            'audio_gate': val_aux['avg_audio_gate'],
            'interaction_gate': val_aux['avg_interaction_gate'],
            'curriculum_classes': current_num_classes if curriculum else config.num_classes
        })

        # Print progress
        curriculum_str = f" [Curr: {current_num_classes}c]" if curriculum else ""
        print(f"  Epoch {epoch+1:3d}{curriculum_str} | "
              f"Loss: {total_loss/n_batches_used:.4f} (cls:{total_cls_loss/n_batches_used:.3f}, "
              f"vad:{total_vad_loss/n_batches_used:.3f}, micl:{total_micl_loss/n_batches_used:.3f}) | "
              f"Train UA: {train_ua:.4f} | Val UA: {val_results['UA']:.4f}")

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    -> New best UA: {best_ua:.4f} "
                  f"(gates: T={val_aux['avg_text_gate']:.3f}, "
                  f"A={val_aux['avg_audio_gate']:.3f}, "
                  f"I={val_aux['avg_interaction_gate']:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_results, final_aux = evaluate(model, val_loader, device, config)
    final_results['history'] = history
    final_results['aux'] = final_aux

    return final_results, model


def train_and_evaluate(
    config: EnhancedConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None
) -> Dict:
    """Train and evaluate with multiple runs."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"ENHANCED MULTIMODAL SER - ACL 2026")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Configuration: {config.emotion_config}, {config.num_classes} classes")
    print(f"Enhancements:")
    print(f"  - VGA (λ={config.vad_lambda})")
    print(f"  - Constrained Adaptive Fusion")
    print(f"  - MICL with Hard Negative Mining (τ={config.micl_temp})")
    print(f"  - Focal Loss (γ={config.focal_gamma})")
    print(f"  - Augmentation: {config.use_augmentation} (mixup α={config.mixup_alpha})")
    print(f"  - Curriculum: {config.use_curriculum}")
    print(f"Loss weights: cls={config.cls_weight}, vad={config.vad_weight}, micl={config.micl_weight}")

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

        if run_results['UA'] > best_val_ua:
            best_val_ua = run_results['UA']
            best_model = model

        print(f"\nRun {run+1} Val Results:")
        print(f"  WA: {run_results['WA']*100:.2f}%, UA: {run_results['UA']*100:.2f}%")
        print(f"  WF1: {run_results['WF1']*100:.2f}%, Macro-F1: {run_results['Macro_F1']*100:.2f}%")

        # Per-class results
        print(f"  Per-class F1:")
        for label, metrics in run_results['per_class'].items():
            print(f"    {label}: {metrics['f1']*100:.1f}% (n={metrics['support']})")

        # Test evaluation
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            test_results, test_aux = evaluate(model, test_loader, device, config)
            all_test_results.append(test_results)
            print(f"\nRun {run+1} Test Results:")
            print(f"  WA: {test_results['WA']*100:.2f}%, UA: {test_results['UA']*100:.2f}%")

    # Aggregate results
    metrics = ['WA', 'UA', 'WF1', 'Macro_F1']
    summary = {
        'validation': {},
        'test': {},
        'config': {
            'emotion_config': config.emotion_config,
            'num_classes': config.num_classes,
            'vad_lambda': config.vad_lambda,
            'micl_weight': config.micl_weight,
            'focal_gamma': config.focal_gamma,
            'use_augmentation': config.use_augmentation,
            'use_curriculum': config.use_curriculum,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers
        },
        'per_run_results': [
            {'WA': r['WA'], 'UA': r['UA'], 'WF1': r['WF1'], 'Macro_F1': r['Macro_F1']}
            for r in all_results
        ]
    }

    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION RESULTS ({config.num_runs} runs)")
    print(f"{'='*70}")

    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = 1.96 * std / np.sqrt(len(values))
        summary['validation'][metric] = {'mean': mean, 'std': std, 'ci_95': ci_95}
        print(f"{metric}: {mean*100:.2f} ± {std*100:.2f}% (95% CI: ±{ci_95*100:.2f}%)")

    # Fusion gate analysis
    text_gates = [r['aux']['avg_text_gate'] for r in all_results]
    audio_gates = [r['aux']['avg_audio_gate'] for r in all_results]
    inter_gates = [r['aux']['avg_interaction_gate'] for r in all_results]
    print(f"\nConstrained Adaptive Fusion Analysis (gates sum to 1):")
    print(f"  Avg Text Gate: {np.mean(text_gates):.4f} ± {np.std(text_gates):.4f}")
    print(f"  Avg Audio Gate: {np.mean(audio_gates):.4f} ± {np.std(audio_gates):.4f}")
    print(f"  Avg Interaction Gate: {np.mean(inter_gates):.4f} ± {np.std(inter_gates):.4f}")

    if test_dataset is not None and all_test_results:
        print(f"\n{'='*70}")
        print(f"FINAL TEST RESULTS ({config.num_runs} runs)")
        print(f"{'='*70}")

        for metric in metrics:
            values = [r[metric] for r in all_test_results]
            mean = np.mean(values)
            std = np.std(values)
            ci_95 = 1.96 * std / np.sqrt(len(values))
            summary['test'][metric] = {'mean': mean, 'std': std, 'ci_95': ci_95}
            print(f"{metric}: {mean*100:.2f} ± {std*100:.2f}% (95% CI: ±{ci_95*100:.2f}%)")

    # Save best model
    if best_model is not None:
        os.makedirs("saved_models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"saved_models/enhanced_{config.emotion_config}_{config.num_classes}class_{timestamp}.pt"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': config,
            'val_ua': best_val_ua,
            'summary': summary
        }, model_path)
        print(f"\nBest model saved to {model_path}")

    return summary


# ============================================================
# CROSS-DATASET EVALUATION
# ============================================================

def cross_dataset_evaluate(
    model_path: str,
    datasets: Dict[str, str],
    device: torch.device
) -> Dict:
    """Evaluate trained model on multiple datasets."""

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = EnhancedMultimodalSER(
        text_dim=config.text_dim,
        audio_dim=config.audio_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        vad_lambda=config.vad_lambda,
        micl_dim=config.micl_dim,
        use_augmentation=False
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nCross-Dataset Evaluation")
    print(f"Model trained on: {config.emotion_config}")
    print(f"{'='*60}")

    results = {}
    for name, path in datasets.items():
        dataset = MultimodalEmotionDataset(path)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        metrics, aux = evaluate(model, dataloader, device, config)
        results[name] = metrics

        print(f"\n{name}:")
        print(f"  WA: {metrics['WA']*100:.2f}%, UA: {metrics['UA']*100:.2f}%")
        print(f"  WF1: {metrics['WF1']*100:.2f}%, Macro-F1: {metrics['Macro_F1']*100:.2f}%")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced ACL 2026: Multimodal SER")
    parser.add_argument("--train", type=str, required=True, help="Training data path")
    parser.add_argument("--val", type=str, required=True, help="Validation data path")
    parser.add_argument("--test", type=str, default=None, help="Test data path")

    # Model architecture
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--emotion_config", type=str, default="iemocap_5",
                        choices=["iemocap_4", "iemocap_5", "iemocap_6",
                                 "cremad_4", "cremad_6", "meld_4", "meld_7"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    # Enhanced components
    parser.add_argument("--vad_lambda", type=float, default=0.5,
                        help="VGA attention guidance strength (increased from 0.1)")
    parser.add_argument("--micl_weight", type=float, default=0.3,
                        help="MICL loss weight (increased from 0.2)")
    parser.add_argument("--vad_weight", type=float, default=0.5,
                        help="VAD loss weight (increased from 0.3)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")

    # Augmentation
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="Enable data augmentation")
    parser.add_argument("--no_augmentation", action="store_false", dest="augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Mixup alpha parameter")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning")
    parser.add_argument("--curriculum_phases", type=str, default="30,60,100",
                        help="Curriculum phase epochs (comma-separated)")
    parser.add_argument("--curriculum_classes", type=str, default="4,5,6",
                        help="Curriculum class schedule (comma-separated)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")

    args = parser.parse_args()

    # Parse curriculum settings
    curriculum_phases = [int(x) for x in args.curriculum_phases.split(',')]
    curriculum_classes = [int(x) for x in args.curriculum_classes.split(',')]

    config = EnhancedConfig(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        emotion_config=args.emotion_config,
        num_runs=args.num_runs,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        vad_lambda=args.vad_lambda,
        micl_weight=args.micl_weight,
        vad_weight=args.vad_weight,
        focal_gamma=args.focal_gamma,
        use_augmentation=args.augmentation,
        mixup_alpha=args.mixup_alpha,
        use_curriculum=args.curriculum,
        curriculum_phases=curriculum_phases,
        curriculum_classes=curriculum_classes
    )

    set_seed(config.seed)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalEmotionDataset(args.train, args.emotion_config)
    val_dataset = MultimodalEmotionDataset(args.val, args.emotion_config)
    test_dataset = MultimodalEmotionDataset(args.test, args.emotion_config) if args.test else None

    # Show data info
    sample = train_dataset[0]
    print(f"Text feature dim: {sample[0].shape}")
    print(f"Audio feature dim: {sample[1].shape}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")

    # Show class distribution
    print(f"\nTraining class distribution:")
    train_dist = train_dataset.get_class_distribution()
    emotion_labels = EMOTION_LABELS.get(args.emotion_config, [])
    for label_idx, count in sorted(train_dist.items()):
        label_name = emotion_labels[label_idx] if label_idx < len(emotion_labels) else f"class_{label_idx}"
        print(f"  {label_name}: {count} ({count/len(train_dataset)*100:.1f}%)")

    # Train and evaluate
    results = train_and_evaluate(config, train_dataset, val_dataset, test_dataset)

    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/enhanced_{args.emotion_config}_{timestamp}.json"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
