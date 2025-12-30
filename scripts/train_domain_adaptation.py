"""
Training script with domain adaptation for cross-dataset SER
Supports: DANN, MMD, CORAL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix

# Import from main training script
from main_icassp2026 import (
    Config, SentimentogramSER, MultimodalEmotionDataset,
    MultiTaskLoss, evaluate, get_class_weights, set_seed, EMOTION_LABELS
)
from domain_adaptation import (
    DomainDiscriminator, MMDLoss, CoralLoss,
    train_with_domain_adaptation
)


def train_with_da(
    config,
    source_train_dataset,
    source_val_dataset,
    target_train_dataset,
    target_test_dataset,
    adaptation_method='mmd',
    lambda_domain=0.1,
    num_runs=3
):
    """Train with domain adaptation"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Domain adaptation method: {adaptation_method}")
    print(f"Lambda domain: {lambda_domain}")

    all_source_results = []
    all_target_results = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*60}")

        set_seed(config.seed + run)

        # Data loaders
        source_train_loader = DataLoader(
            source_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        source_val_loader = DataLoader(
            source_val_dataset, batch_size=config.batch_size, shuffle=False
        )
        target_train_loader = DataLoader(
            target_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        target_test_loader = DataLoader(
            target_test_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Model
        model = SentimentogramSER(config).to(device)

        # Domain discriminator (for DANN)
        domain_disc = None
        if adaptation_method == 'dann':
            domain_disc = DomainDiscriminator(config.hidden_dim).to(device)

        # Optimizer
        params = list(model.parameters())
        if domain_disc is not None:
            params += list(domain_disc.parameters())

        optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

        # Loss
        class_weights = get_class_weights(source_train_dataset, device)
        criterion = MultiTaskLoss(config, class_weights)

        # Training
        best_source_ua = 0
        best_target_ua = 0
        patience_counter = 0
        best_state = None

        for epoch in range(config.epochs):
            # Train with domain adaptation
            train_stats = train_with_domain_adaptation(
                model=model,
                domain_disc=domain_disc,
                source_loader=source_train_loader,
                target_loader=target_train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_epochs=config.epochs,
                adaptation_method=adaptation_method,
                lambda_domain=lambda_domain
            )

            # Evaluate on source val
            source_results = evaluate(model, source_val_loader, device, config)

            # Evaluate on target test
            target_results = evaluate(model, target_test_loader, device, config)

            print(f"  Epoch {epoch+1:3d} | "
                  f"CLS Loss: {train_stats['cls_loss']:.4f} | "
                  f"DA Loss: {train_stats['domain_loss']:.4f} | "
                  f"Src UA: {source_results['UA']:.4f} | "
                  f"Tgt UA: {target_results['UA']:.4f}")

            # Track best based on target performance (or source if you prefer)
            if target_results['UA'] > best_target_ua:
                best_target_ua = target_results['UA']
                best_source_ua = source_results['UA']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"    -> New best Target UA: {best_target_ua:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model and final evaluation
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        final_source = evaluate(model, source_val_loader, device, config)
        final_target = evaluate(model, target_test_loader, device, config)

        print(f"\nRun {run+1} Final Results:")
        print(f"  Source Val - WA: {final_source['WA']:.4f}, UA: {final_source['UA']:.4f}")
        print(f"  Target Test - WA: {final_target['WA']:.4f}, UA: {final_target['UA']:.4f}")

        all_source_results.append(final_source)
        all_target_results.append(final_target)

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({num_runs} runs) - {adaptation_method.upper()}")
    print(f"{'='*60}")

    metrics = ['WA', 'UA', 'WF1', 'Macro_F1']

    print("\nSource Domain (Validation):")
    for metric in metrics:
        values = [r[metric] for r in all_source_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    print("\nTarget Domain (Test):")
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in all_target_results]
        mean = np.mean(values)
        std = np.std(values)
        summary[metric] = {'mean': mean, 'std': std}
        print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation for SER")
    parser.add_argument("--source_train", type=str, required=True)
    parser.add_argument("--source_val", type=str, required=True)
    parser.add_argument("--target_train", type=str, required=True)
    parser.add_argument("--target_test", type=str, required=True)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--method", type=str, default="mmd",
                        choices=["dann", "mmd", "coral", "none"])
    parser.add_argument("--lambda_domain", type=float, default=0.1)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_da.json")

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
    source_train = MultimodalEmotionDataset(args.source_train)
    source_val = MultimodalEmotionDataset(args.source_val)
    target_train = MultimodalEmotionDataset(args.target_train)
    target_test = MultimodalEmotionDataset(args.target_test)

    print(f"Source train: {len(source_train)}, Source val: {len(source_val)}")
    print(f"Target train: {len(target_train)}, Target test: {len(target_test)}")

    # Train
    results = train_with_da(
        config=config,
        source_train_dataset=source_train,
        source_val_dataset=source_val,
        target_train_dataset=target_train,
        target_test_dataset=target_test,
        adaptation_method=args.method,
        lambda_domain=args.lambda_domain,
        num_runs=args.num_runs
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
