"""
Combined training on multiple datasets (IEMOCAP + CREMA-D)
This approach trains on merged data from both domains.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import argparse
import json
import pickle

from main_icassp2026 import (
    Config, SentimentogramSER, MultimodalEmotionDataset,
    MultiTaskLoss, evaluate, get_class_weights, set_seed, EMOTION_LABELS
)


def train_combined(
    config,
    train_datasets,  # List of training datasets
    val_datasets,    # Dict of {name: dataset} for validation
    test_datasets,   # Dict of {name: dataset} for testing
    num_runs=3
):
    """Train on combined datasets and evaluate on each separately"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Combine training data
    combined_train = ConcatDataset(train_datasets)
    print(f"Combined training samples: {len(combined_train)}")

    all_results = {name: [] for name in test_datasets.keys()}

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*60}")

        set_seed(config.seed + run)

        # Data loaders
        train_loader = DataLoader(
            combined_train, batch_size=config.batch_size, shuffle=True, drop_last=True
        )

        val_loaders = {
            name: DataLoader(ds, batch_size=config.batch_size, shuffle=False)
            for name, ds in val_datasets.items()
        }

        test_loaders = {
            name: DataLoader(ds, batch_size=config.batch_size, shuffle=False)
            for name, ds in test_datasets.items()
        }

        # Model
        model = SentimentogramSER(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Loss - get class weights from combined dataset
        class_weights = get_class_weights(combined_train, device)
        criterion = MultiTaskLoss(config, class_weights)

        # Scheduler
        total_steps = len(train_loader) * config.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr * 10,
            total_steps=total_steps,
            pct_start=config.warmup_ratio
        )

        best_avg_ua = 0
        patience_counter = 0
        best_state = None

        for epoch in range(config.epochs):
            # Training
            model.train()
            total_loss = 0

            for text_feat, audio_feat, labels in train_loader:
                text_feat = text_feat.to(device)
                audio_feat = audio_feat.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(text_feat, audio_feat)
                losses = criterion(outputs, labels)

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += losses['total'].item()

            # Validation on each dataset
            val_results = {}
            for name, loader in val_loaders.items():
                val_results[name] = evaluate(model, loader, device, config)

            avg_ua = np.mean([r['UA'] for r in val_results.values()])

            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | " +
                  " | ".join([f"{name} UA: {r['UA']:.4f}" for name, r in val_results.items()]))

            if avg_ua > best_avg_ua:
                best_avg_ua = avg_ua
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"    -> New best avg UA: {best_avg_ua:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model and evaluate on test sets
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        print(f"\nRun {run+1} Test Results:")
        for name, loader in test_loaders.items():
            test_result = evaluate(model, loader, device, config)
            all_results[name].append(test_result)
            print(f"  {name}: WA={test_result['WA']:.4f}, UA={test_result['UA']:.4f}, WF1={test_result['WF1']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL COMBINED TRAINING RESULTS ({num_runs} runs)")
    print(f"{'='*60}")

    summary = {}
    for name, results in all_results.items():
        print(f"\n{name} Test Set:")
        summary[name] = {}
        for metric in ['WA', 'UA', 'WF1', 'Macro_F1']:
            values = [r[metric] for r in results]
            mean = np.mean(values)
            std = np.std(values)
            summary[name][metric] = {'mean': mean, 'std': std}
            print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Combined Training for SER")
    parser.add_argument("--iemocap_train", type=str, required=True)
    parser.add_argument("--iemocap_val", type=str, required=True)
    parser.add_argument("--iemocap_test", type=str, required=True)
    parser.add_argument("--cremad_train", type=str, required=True)
    parser.add_argument("--cremad_val", type=str, required=True)
    parser.add_argument("--cremad_test", type=str, required=True)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_combined.json")

    args = parser.parse_args()

    config = Config(
        audio_dim=args.audio_dim,
        hidden_dim=384,
        num_layers=2,
        dropout=0.3,
        weight_decay=0.01,
        epochs=100,
        patience=15
    )

    # Load datasets
    print("Loading datasets...")
    iemocap_train = MultimodalEmotionDataset(args.iemocap_train)
    iemocap_val = MultimodalEmotionDataset(args.iemocap_val)
    iemocap_test = MultimodalEmotionDataset(args.iemocap_test)

    cremad_train = MultimodalEmotionDataset(args.cremad_train)
    cremad_val = MultimodalEmotionDataset(args.cremad_val)
    cremad_test = MultimodalEmotionDataset(args.cremad_test)

    print(f"IEMOCAP: train={len(iemocap_train)}, val={len(iemocap_val)}, test={len(iemocap_test)}")
    print(f"CREMA-D: train={len(cremad_train)}, val={len(cremad_val)}, test={len(cremad_test)}")

    # Train
    results = train_combined(
        config=config,
        train_datasets=[iemocap_train, cremad_train],
        val_datasets={'IEMOCAP': iemocap_val, 'CREMAD': cremad_val},
        test_datasets={'IEMOCAP': iemocap_test, 'CREMAD': cremad_test},
        num_runs=args.num_runs
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
