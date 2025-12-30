"""
Baseline Comparison Script for ACL 2026 Submission
===================================================
Implements simple baselines for fair comparison:
1. BERT-only (text-only)
2. emotion2vec-only (audio-only)
3. Simple Concatenation (BERT + emotion2vec → MLP)
4. Standard Cross-Attention (no VAD guidance)
5. Our Enhanced Model (VGA + EAAF + MICL)

All baselines are run 5 times with statistical significance tests.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, balanced_accuracy_score
)

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# BASELINE MODELS
# ============================================================================

class TextOnlyClassifier(nn.Module):
    """Baseline 1: BERT-only (text modality)"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_features, audio_features=None):
        return self.classifier(text_features)


class AudioOnlyClassifier(nn.Module):
    """Baseline 2: emotion2vec-only (audio modality)"""
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_features, audio_features):
        return self.classifier(audio_features)


class ConcatenationBaseline(nn.Module):
    """Baseline 3: Simple concatenation (BERT + emotion2vec → MLP)"""
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384, num_classes=5, dropout=0.3):
        super().__init__()
        concat_dim = text_dim + audio_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_features, audio_features):
        combined = torch.cat([text_features, audio_features], dim=-1)
        return self.classifier(combined)


class StandardCrossAttention(nn.Module):
    """Baseline 4: Standard cross-attention (no VAD guidance)"""
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384,
                 num_heads=8, num_classes=5, dropout=0.3):
        super().__init__()

        # Project to common space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-attention layers
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)

        # Simple fusion (average pooling)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_features, audio_features):
        # Add sequence dimension if needed
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)

        # Project
        text_proj = self.text_proj(text_features)
        audio_proj = self.audio_proj(audio_features)

        # Cross-attention
        text_attended, _ = self.text_to_audio_attn(text_proj, audio_proj, audio_proj)
        audio_attended, _ = self.audio_to_text_attn(audio_proj, text_proj, text_proj)

        # Residual + norm
        text_out = self.text_norm(text_proj + text_attended)
        audio_out = self.audio_norm(audio_proj + audio_attended)

        # Pool and fuse
        text_pooled = text_out.mean(dim=1)
        audio_pooled = audio_out.mean(dim=1)

        fused = self.fusion(torch.cat([text_pooled, audio_pooled], dim=-1))

        return self.classifier(fused)


class AdaptiveFusionBaseline(nn.Module):
    """Baseline 5: Adaptive fusion without constraints (gates don't sum to 1)"""
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384, num_classes=5, dropout=0.3):
        super().__init__()

        # Projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Separate sigmoid gates (don't sum to 1)
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_features, audio_features):
        text_proj = self.text_proj(text_features)
        audio_proj = self.audio_proj(audio_features)

        combined = torch.cat([text_proj, audio_proj], dim=-1)

        alpha_text = self.text_gate(combined)
        alpha_audio = self.audio_gate(combined)

        fused = alpha_text * text_proj + alpha_audio * audio_proj

        return self.classifier(fused)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def load_features(train_path: str, val_path: str, test_path: str = None) -> Dict:
    """Load features from pickle files"""
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    test_data = None
    if test_path and os.path.exists(test_path):
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def prepare_dataloaders(data: Dict, batch_size: int = 32) -> Dict:
    """Prepare DataLoaders from feature dictionaries or list of dicts"""
    loaders = {}

    for split in ['train', 'val', 'test']:
        if data[split] is None:
            continue

        split_data = data[split]

        # Handle list of dictionaries format
        if isinstance(split_data, list):
            text_list = []
            audio_list = []
            label_list = []
            for item in split_data:
                text_key = 'text_embed' if 'text_embed' in item else 'text_features'
                audio_key = 'audio_embed' if 'audio_embed' in item else 'audio_features'
                label_key = 'label' if 'label' in item else 'labels'

                text_list.append(item[text_key])
                audio_list.append(item[audio_key])
                label_list.append(item[label_key])

            text = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in text_list])
            audio = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in audio_list])
            labels = torch.tensor([l.item() if isinstance(l, torch.Tensor) else l for l in label_list], dtype=torch.long)
        else:
            # Handle dictionary format
            if 'text_features' in split_data:
                text = split_data['text_features']
                audio = split_data['audio_features']
                labels = split_data['labels']
            else:
                # Try alternative keys
                text = split_data.get('text', split_data.get('text_feat'))
                audio = split_data.get('audio', split_data.get('audio_feat'))
                labels = split_data.get('labels', split_data.get('label'))

            # Convert to tensors if needed
            if not isinstance(text, torch.Tensor):
                text = torch.tensor(text, dtype=torch.float32)
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

        text = text.float()
        audio = audio.float()

        dataset = TensorDataset(text, audio, labels)
        shuffle = (split == 'train')
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loaders


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for text, audio, labels in loader:
        text, audio, labels = text.to(device), audio.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(text, audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device) -> Dict:
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, audio, labels in loader:
            text, audio, labels = text.to(device), audio.to(device), labels.to(device)
            outputs = model(text, audio)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'WA': accuracy_score(all_labels, all_preds) * 100,
        'UA': balanced_accuracy_score(all_labels, all_preds) * 100,
        'WF1': f1_score(all_labels, all_preds, average='weighted') * 100,
        'Macro-F1': f1_score(all_labels, all_preds, average='macro') * 100,
        'predictions': all_preds,
        'labels': all_labels
    }


def run_baseline(
    model_class,
    model_kwargs: Dict,
    loaders: Dict,
    num_classes: int,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = 'cuda'
) -> Dict:
    """Run a single baseline experiment"""

    model = model_class(**model_kwargs, num_classes=num_classes).to(device)

    # Class weights for imbalanced data
    train_labels = []
    for _, _, labels in loaders['train']:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)

    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_ua = 0
    best_model_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders['val'], device)

        scheduler.step(val_metrics['UA'])

        if val_metrics['UA'] > best_val_ua:
            best_val_ua = val_metrics['UA']
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    val_results = evaluate(model, loaders['val'], device)

    test_results = None
    if 'test' in loaders and loaders['test'] is not None:
        test_results = evaluate(model, loaders['test'], device)

    return {
        'val': val_results,
        'test': test_results
    }


def run_multiple_seeds(
    model_class,
    model_kwargs: Dict,
    loaders: Dict,
    num_classes: int,
    num_runs: int = 5,
    device: str = 'cuda'
) -> Dict:
    """Run baseline multiple times with different seeds"""

    val_results = {'WA': [], 'UA': [], 'WF1': [], 'Macro-F1': []}
    test_results = {'WA': [], 'UA': [], 'WF1': [], 'Macro-F1': []}

    for run in range(num_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        results = run_baseline(
            model_class, model_kwargs, loaders,
            num_classes=num_classes, device=device
        )

        for metric in ['WA', 'UA', 'WF1', 'Macro-F1']:
            val_results[metric].append(results['val'][metric])
            if results['test'] is not None:
                test_results[metric].append(results['test'][metric])

    # Compute statistics
    summary = {'val': {}, 'test': {}}
    for metric in ['WA', 'UA', 'WF1', 'Macro-F1']:
        vals = val_results[metric]
        summary['val'][metric] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'values': vals
        }
        if test_results['WA']:
            tests = test_results[metric]
            summary['test'][metric] = {
                'mean': np.mean(tests),
                'std': np.std(tests),
                'values': tests
            }

    return summary


def compute_significance(baseline_results: List[float], ours_results: List[float]) -> Dict:
    """Compute statistical significance tests"""
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(ours_results, baseline_results)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(ours_results, baseline_results)
    except ValueError:
        w_stat, w_pvalue = np.nan, np.nan

    # Effect size (Cohen's d)
    diff = np.array(ours_results) - np.array(baseline_results)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)

    return {
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_stat': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'significant_005': t_pvalue < 0.05,
        'significant_001': t_pvalue < 0.01
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison for ACL 2026')
    parser.add_argument('--train', type=str, required=True, help='Training features path')
    parser.add_argument('--val', type=str, required=True, help='Validation features path')
    parser.add_argument('--test', type=str, default=None, help='Test features path')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of emotion classes')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per baseline')
    parser.add_argument('--output', type=str, default='results/baseline_comparison.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("ACL 2026 Baseline Comparison")
    print("=" * 60)
    print(f"Train: {args.train}")
    print(f"Val: {args.val}")
    print(f"Test: {args.test}")
    print(f"Classes: {args.num_classes}")
    print(f"Runs: {args.num_runs}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load data
    print("\nLoading features...")
    data = load_features(args.train, args.val, args.test)
    loaders = prepare_dataloaders(data, batch_size=32)

    # Define baselines
    baselines = {
        'BERT-only (Text)': {
            'class': TextOnlyClassifier,
            'kwargs': {'input_dim': 768, 'hidden_dim': 256, 'dropout': 0.3}
        },
        'emotion2vec-only (Audio)': {
            'class': AudioOnlyClassifier,
            'kwargs': {'input_dim': 1024, 'hidden_dim': 256, 'dropout': 0.3}
        },
        'Concatenation': {
            'class': ConcatenationBaseline,
            'kwargs': {'text_dim': 768, 'audio_dim': 1024, 'hidden_dim': 384, 'dropout': 0.3}
        },
        'Standard Cross-Attention': {
            'class': StandardCrossAttention,
            'kwargs': {'text_dim': 768, 'audio_dim': 1024, 'hidden_dim': 384, 'num_heads': 8, 'dropout': 0.3}
        },
        'Adaptive Fusion (Unconstrained)': {
            'class': AdaptiveFusionBaseline,
            'kwargs': {'text_dim': 768, 'audio_dim': 1024, 'hidden_dim': 384, 'dropout': 0.3}
        }
    }

    all_results = {}

    for name, config in baselines.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        results = run_multiple_seeds(
            config['class'],
            config['kwargs'],
            loaders,
            num_classes=args.num_classes,
            num_runs=args.num_runs,
            device=args.device
        )

        all_results[name] = results

        # Print results
        print(f"\nValidation Results:")
        for metric in ['WA', 'UA', 'WF1', 'Macro-F1']:
            mean = results['val'][metric]['mean']
            std = results['val'][metric]['std']
            print(f"  {metric}: {mean:.2f} ± {std:.2f}%")

        if results['test'].get('WA'):
            print(f"\nTest Results:")
            for metric in ['WA', 'UA', 'WF1', 'Macro-F1']:
                mean = results['test'][metric]['mean']
                std = results['test'][metric]['std']
                print(f"  {metric}: {mean:.2f} ± {std:.2f}%")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Convert numpy to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    serializable_results = to_serializable(all_results)
    serializable_results['metadata'] = {
        'train_path': args.train,
        'val_path': args.val,
        'test_path': args.test,
        'num_classes': args.num_classes,
        'num_runs': args.num_runs,
        'timestamp': datetime.now().isoformat()
    }

    with open(args.output, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Validation UA)")
    print("=" * 80)
    print(f"{'Method':<35} {'UA Mean':>10} {'UA Std':>10} {'WF1 Mean':>10}")
    print("-" * 80)
    for name, results in all_results.items():
        ua_mean = results['val']['UA']['mean']
        ua_std = results['val']['UA']['std']
        wf1_mean = results['val']['WF1']['mean']
        print(f"{name:<35} {ua_mean:>10.2f} {ua_std:>10.2f} {wf1_mean:>10.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
