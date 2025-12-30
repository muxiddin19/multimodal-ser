"""
Fine-tuning approach: Pretrain on source domain (IEMOCAP), then fine-tune on target domain (CREMA-D)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os

from main_icassp2026 import (
    Config, SentimentogramSER, MultimodalEmotionDataset,
    MultiTaskLoss, evaluate, get_class_weights, set_seed, EMOTION_LABELS
)


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

        if epoch % 5 == 0 or val_results['UA'] > best_ua:
            print(f"  Pretrain Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val UA: {val_results['UA']:.4f}")

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Pretrain early stopping at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Pretrain complete. Best UA: {best_ua:.4f}")

    return model


def finetune_on_target(
    model,
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    run_idx,
    finetune_lr_factor=0.1,
    freeze_layers=None
):
    """Fine-tune pretrained model on target domain"""
    print("\n--- Fine-tuning on Target Domain ---")

    set_seed(config.seed + run_idx + 100)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Optionally freeze some layers
    if freeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
                print(f"  Frozen: {name}")

    # Lower learning rate for fine-tuning
    finetune_lr = config.lr * finetune_lr_factor

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr,
        weight_decay=config.weight_decay
    )

    class_weights = get_class_weights(train_dataset, device)
    criterion = MultiTaskLoss(config, class_weights)

    best_ua = 0
    patience_counter = 0
    best_state = None

    # Shorter fine-tuning
    finetune_epochs = min(50, config.epochs)
    finetune_patience = min(10, config.patience)

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

        print(f"  Finetune Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val UA: {val_results['UA']:.4f}")

        if val_results['UA'] > best_ua:
            best_ua = val_results['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    -> New best UA: {best_ua:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= finetune_patience:
            print(f"  Finetune early stopping at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation
    final_val = evaluate(model, val_loader, device, config)
    final_test = evaluate(model, test_loader, device, config)

    return final_val, final_test


def train_with_finetuning(
    config,
    source_train, source_val,
    target_train, target_val, target_test,
    num_runs=3,
    finetune_lr_factor=0.1,
    freeze_encoder=False
):
    """Full pipeline: pretrain on source, finetune on target"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Fine-tune LR factor: {finetune_lr_factor}")
    print(f"Freeze encoder: {freeze_encoder}")

    all_val_results = []
    all_test_results = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*60}")

        # Step 1: Pretrain on source
        model = pretrain_on_source(config, source_train, source_val, device, run)

        # Step 2: Fine-tune on target
        freeze_layers = ['text_proj', 'audio_proj', 'cross_attn'] if freeze_encoder else None

        val_result, test_result = finetune_on_target(
            model=model,
            config=config,
            train_dataset=target_train,
            val_dataset=target_val,
            test_dataset=target_test,
            device=device,
            run_idx=run,
            finetune_lr_factor=finetune_lr_factor,
            freeze_layers=freeze_layers
        )

        all_val_results.append(val_result)
        all_test_results.append(test_result)

        print(f"\nRun {run+1} Final Results:")
        print(f"  Target Val - UA: {val_result['UA']:.4f}, WA: {val_result['WA']:.4f}")
        print(f"  Target Test - UA: {test_result['UA']:.4f}, WA: {test_result['WA']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL FINE-TUNING RESULTS ({num_runs} runs)")
    print(f"{'='*60}")

    summary = {'validation': {}, 'test': {}}

    print("\nTarget Validation:")
    for metric in ['WA', 'UA', 'WF1', 'Macro_F1']:
        values = [r[metric] for r in all_val_results]
        mean = np.mean(values)
        std = np.std(values)
        summary['validation'][metric] = {'mean': mean, 'std': std}
        print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    print("\nTarget Test:")
    for metric in ['WA', 'UA', 'WF1', 'Macro_F1']:
        values = [r[metric] for r in all_test_results]
        mean = np.mean(values)
        std = np.std(values)
        summary['test'][metric] = {'mean': mean, 'std': std}
        print(f"  {metric}: {mean*100:.2f} ± {std*100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning for Cross-Dataset SER")
    parser.add_argument("--source_train", type=str, required=True)
    parser.add_argument("--source_val", type=str, required=True)
    parser.add_argument("--target_train", type=str, required=True)
    parser.add_argument("--target_val", type=str, required=True)
    parser.add_argument("--target_test", type=str, required=True)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--finetune_lr_factor", type=float, default=0.1,
                        help="Learning rate factor for fine-tuning (default: 0.1)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder layers during fine-tuning")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_finetune.json")

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
    target_val = MultimodalEmotionDataset(args.target_val)
    target_test = MultimodalEmotionDataset(args.target_test)

    print(f"Source: train={len(source_train)}, val={len(source_val)}")
    print(f"Target: train={len(target_train)}, val={len(target_val)}, test={len(target_test)}")

    # Train
    results = train_with_finetuning(
        config=config,
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        num_runs=args.num_runs,
        finetune_lr_factor=args.finetune_lr_factor,
        freeze_encoder=args.freeze_encoder
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
