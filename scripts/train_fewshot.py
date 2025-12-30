"""
Few-shot fine-tuning: Pretrain on source, fine-tune on limited target data (10%, 20%, 50%)
This tests how well the model adapts with minimal target domain data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import json
import random

from main_icassp2026 import (
    Config, SentimentogramSER, MultimodalEmotionDataset,
    MultiTaskLoss, evaluate, get_class_weights, set_seed, EMOTION_LABELS
)


def create_few_shot_subset(dataset, percentage, seed=42):
    """Create a stratified subset of the dataset"""
    random.seed(seed)
    np.random.seed(seed)

    # Get all labels
    labels = [dataset[i][2].item() for i in range(len(dataset))]

    # Group indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # Sample from each class proportionally
    selected_indices = []
    for label, indices in label_to_indices.items():
        n_samples = max(1, int(len(indices) * percentage))
        selected = random.sample(indices, n_samples)
        selected_indices.extend(selected)

    random.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def pretrain_on_source(config, train_dataset, val_dataset, device, run_idx):
    """Pretrain model on source domain"""
    print("\n--- Pretraining on Source Domain ---")

    set_seed(config.seed + run_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    model = SentimentogramSER(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    class_weights = get_class_weights(train_dataset, device)
    criterion = MultiTaskLoss(config, class_weights)

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

        val_results = evaluate(model, val_loader, device, config)

        if epoch % 10 == 0:
            print(f"  Pretrain Epoch {epoch+1:3d} | Val UA: {val_results['UA']:.4f}")

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Pretrain early stopping at epoch {epoch+1}, Best UA: {best_ua:.4f}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def finetune_few_shot(model, config, train_subset, val_dataset, test_dataset, device, run_idx):
    """Fine-tune on few-shot target data"""

    set_seed(config.seed + run_idx + 100)

    # Handle small batch sizes for few-shot
    batch_size = min(config.batch_size, len(train_subset) // 2)
    batch_size = max(4, batch_size)  # At least 4

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, drop_last=len(train_subset) > batch_size
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Lower learning rate for few-shot fine-tuning
    finetune_lr = config.lr * 0.1

    optimizer = optim.AdamW(
        model.parameters(), lr=finetune_lr, weight_decay=config.weight_decay
    )

    class_weights = get_class_weights(train_subset, device)
    criterion = MultiTaskLoss(config, class_weights)

    best_ua = 0
    patience_counter = 0
    best_state = None

    # Shorter training for few-shot
    finetune_epochs = 30
    finetune_patience = 8

    for epoch in range(finetune_epochs):
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

            total_loss += losses['total'].item()

        val_results = evaluate(model, val_loader, device, config)

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= finetune_patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_val = evaluate(model, val_loader, device, config)
    final_test = evaluate(model, test_loader, device, config)

    return final_val, final_test


def run_few_shot_experiment(
    config,
    source_train, source_val,
    target_train, target_val, target_test,
    percentages=[0.1, 0.2, 0.5, 1.0],
    num_runs=3
):
    """Run few-shot experiments with different percentages of target data"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = {}

    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Few-shot: {int(pct*100)}% target data")
        print(f"{'='*60}")

        pct_results = []

        for run in range(num_runs):
            print(f"\n--- Run {run+1}/{num_runs} ---")

            # Create few-shot subset
            train_subset = create_few_shot_subset(target_train, pct, seed=config.seed + run)
            print(f"  Training samples: {len(train_subset)} (from {len(target_train)})")

            # Pretrain on source
            model = pretrain_on_source(config, source_train, source_val, device, run)

            # Fine-tune on few-shot target
            print(f"\n--- Fine-tuning on {int(pct*100)}% target data ---")
            val_result, test_result = finetune_few_shot(
                model, config, train_subset, target_val, target_test, device, run
            )

            pct_results.append(test_result)
            print(f"  Test UA: {test_result['UA']:.4f}, WA: {test_result['WA']:.4f}")

        # Summary for this percentage
        all_results[f"{int(pct*100)}%"] = {
            'n_samples': len(train_subset),
            'results': []
        }

        print(f"\n{int(pct*100)}% Target Data Results:")
        for metric in ['WA', 'UA', 'WF1', 'Macro_F1']:
            values = [r[metric] for r in pct_results]
            mean = np.mean(values)
            std = np.std(values)
            all_results[f"{int(pct*100)}%"][metric] = {'mean': mean, 'std': std}
            print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    # Final summary table
    print(f"\n{'='*60}")
    print("FEW-SHOT FINE-TUNING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Data %':<10} {'Samples':<10} {'UA':<15} {'WA':<15}")
    print("-" * 50)

    for pct_key, data in all_results.items():
        ua = data['UA']
        wa = data['WA']
        print(f"{pct_key:<10} {data['n_samples']:<10} "
              f"{ua['mean']*100:.2f}±{ua['std']*100:.2f}%  "
              f"{wa['mean']*100:.2f}±{wa['std']*100:.2f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Few-shot Fine-tuning for SER")
    parser.add_argument("--source_train", type=str, required=True)
    parser.add_argument("--source_val", type=str, required=True)
    parser.add_argument("--target_train", type=str, required=True)
    parser.add_argument("--target_val", type=str, required=True)
    parser.add_argument("--target_test", type=str, required=True)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--percentages", type=str, default="10,20,50,100",
                        help="Comma-separated percentages (e.g., '10,20,50,100')")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_fewshot.json")

    args = parser.parse_args()

    # Parse percentages
    percentages = [int(p) / 100 for p in args.percentages.split(",")]

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
    source_train = MultimodalEmotionDataset(args.source_train)
    source_val = MultimodalEmotionDataset(args.source_val)
    target_train = MultimodalEmotionDataset(args.target_train)
    target_val = MultimodalEmotionDataset(args.target_val)
    target_test = MultimodalEmotionDataset(args.target_test)

    print(f"Source: train={len(source_train)}, val={len(source_val)}")
    print(f"Target: train={len(target_train)}, val={len(target_val)}, test={len(target_test)}")

    # Run few-shot experiments
    results = run_few_shot_experiment(
        config=config,
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        percentages=percentages,
        num_runs=args.num_runs
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
