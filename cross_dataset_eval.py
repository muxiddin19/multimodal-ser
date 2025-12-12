"""
Cross-Dataset Evaluation for SER
Addresses reviewer concern: "Only IEMOCAP evaluation - need cross-dataset validation"

This script evaluates trained models on multiple datasets to demonstrate generalization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from typing import Dict, List
from collections import Counter

from main_icassp2026 import (
    Config, SentimentogramSER, MultimodalEmotionDataset,
    compute_metrics, set_seed, EMOTION_LABELS
)


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    model = SentimentogramSER(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_cross_dataset(
    model: nn.Module,
    dataset_path: str,
    config: Config,
    device: torch.device,
    label_mapping: Dict[int, int] = None
) -> Dict:
    """
    Evaluate model on a different dataset.

    Args:
        model: Trained model
        dataset_path: Path to target dataset features
        config: Model configuration
        device: Device
        label_mapping: Optional mapping from source to target labels
    """
    dataset = MultimodalEmotionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for text_feat, audio_feat, labels in dataloader:
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)

            outputs = model(text_feat, audio_feat)
            preds = outputs['probs'].argmax(dim=1)

            # Apply label mapping if provided
            if label_mapping:
                labels = torch.tensor([label_mapping.get(l.item(), l.item()) for l in labels])

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get appropriate labels
    num_classes = len(set(all_labels))
    emotion_labels = EMOTION_LABELS.get(config.emotion_config,
                                         [f"class_{i}" for i in range(num_classes)])

    return compute_metrics(all_labels, all_preds, emotion_labels)


def cross_dataset_evaluation_suite(
    model_checkpoint: str,
    model_config: Config,
    dataset_paths: Dict[str, str],
    output_path: str = "cross_dataset_results.json"
):
    """
    Run comprehensive cross-dataset evaluation.

    Args:
        model_checkpoint: Path to trained model
        model_config: Model configuration
        dataset_paths: Dict mapping dataset names to feature paths
        output_path: Where to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_checkpoint, model_config, device)

    results = {}

    print("\n" + "="*70)
    print("CROSS-DATASET EVALUATION")
    print("="*70)

    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\nEvaluating on {dataset_name}...")

        try:
            metrics = evaluate_cross_dataset(
                model, dataset_path, model_config, device
            )
            results[dataset_name] = metrics

            print(f"  WA: {metrics['WA']*100:.2f}%")
            print(f"  UA: {metrics['UA']*100:.2f}%")
            print(f"  WF1: {metrics['WF1']*100:.2f}%")
            print(f"  Macro-F1: {metrics['Macro_F1']*100:.2f}%")

        except Exception as e:
            print(f"  Error: {e}")
            results[dataset_name] = {"error": str(e)}

    # Summary table
    print("\n" + "="*70)
    print("CROSS-DATASET SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'WA':>10} {'UA':>10} {'WF1':>10} {'Macro-F1':>10}")
    print("-"*70)

    for dataset_name, metrics in results.items():
        if "error" not in metrics:
            print(f"{dataset_name:<20} "
                  f"{metrics['WA']*100:>9.2f}% "
                  f"{metrics['UA']*100:>9.2f}% "
                  f"{metrics['WF1']*100:>9.2f}% "
                  f"{metrics['Macro_F1']*100:>9.2f}%")
        else:
            print(f"{dataset_name:<20} {'Error':<40}")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset SER evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--datasets", type=str, nargs="+",
                        help="Dataset paths in format name:path")
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--emotion_config", type=str, default="iemocap_4")
    parser.add_argument("--output", type=str, default="cross_dataset_results.json")

    args = parser.parse_args()

    config = Config(
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        num_classes=args.num_classes,
        emotion_config=args.emotion_config
    )

    # Parse dataset paths
    dataset_paths = {}
    if args.datasets:
        for ds in args.datasets:
            name, path = ds.split(":")
            dataset_paths[name] = path
    else:
        # Default: use available datasets
        import os
        if os.path.exists("features/IEMOCAP_BERT_wav2vec_val.pkl"):
            dataset_paths["IEMOCAP (val)"] = "features/IEMOCAP_BERT_wav2vec_val.pkl"
        if os.path.exists("features/MELD_BERT_wav2vec_test.pkl"):
            dataset_paths["MELD (test)"] = "features/MELD_BERT_wav2vec_test.pkl"

    cross_dataset_evaluation_suite(args.model, config, dataset_paths, args.output)


if __name__ == "__main__":
    main()
