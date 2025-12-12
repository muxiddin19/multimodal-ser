# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal Speech Emotion Recognition (SER) system using late fusion of text and audio modalities. The project targets the IEMOCAP dataset with 4/6/7-class emotion classification.

**Research Goal**: Improve upon current SOTA results (WA ~76%, WF1 ~76%) to make the approach competitive for top-tier venues like ICASSP 2026.

## AAAI-26 Reviewer Feedback Summary

The paper was rejected from AAAI-26 with the following critical feedback:

1. **Incremental architecture** - Cross-attention fusion is well-studied
2. **Narrow dataset scope** - Only IEMOCAP; need MOSEI, CREMA-D, multilingual
3. **Missing metrics** - Need UA, macro-F1, confusion matrices, confidence intervals
4. **User study deficient** - No sample size, demographics, statistical significance
5. **Technical inaccuracies** - Inconsistent emotion label sets, VAD formulation mismatch
6. **Missing baselines** - emotion2vec, TelME, CARAT, MultiEMO not compared

## Architecture

### Feature Extraction Pipeline
1. **Text Features**: BERT (`bert-base-uncased`) → 768-dim embeddings from [CLS] token
2. **Audio Features**:
   - Wav2Vec2 (paper description) → 768-dim contextual features
   - emotion2vec (`iic/emotion2vec_plus_large`) → 768-dim emotion-aware features
   - ECAPA-TDNN (speechbrain) → 192-dim speaker embeddings (baseline)

### Model Variants
- `training/BERT_ECAPA.py`: Simple late fusion MLP (FlexibleMMSER) - baseline
- `main1.py`: Original model (has discrepancies with paper)
- `main_icassp2026.py`: **NEW** - Paper-aligned model addressing reviewer feedback:
  - Standard bidirectional cross-attention (matches paper Figure 3)
  - Support for 4/6/7-class configurations
  - Comprehensive metrics (WA, UA, WF1, Macro-F1, per-class)
  - Multi-run evaluation with confidence intervals
  - Built-in ablation study framework

### Paper vs Code Discrepancies (in main1.py)
| Paper Claims | main1.py Implementation |
|--------------|------------------------|
| Wav2Vec2 audio | ECAPA-TDNN/emotion2vec |
| 6+1 classes | 4 classes |
| Standard MHCA | Gated cross-attention |

## Commands

### Training (ICASSP 2026 Version)
```bash
# Full training with 5-run evaluation
python main_icassp2026.py --train features/IEMOCAP_BERT_wav2vec_train.pkl \
                          --val features/IEMOCAP_BERT_wav2vec_val.pkl \
                          --num_runs 5

# Run ablation study
python main_icassp2026.py --train features/... --val features/... --ablation

# Different emotion configurations
python main_icassp2026.py --emotion_config iemocap_6 --num_classes 6
python main_icassp2026.py --emotion_config ekman_7 --num_classes 7
```

### Cross-Dataset Evaluation
```bash
python cross_dataset_eval.py --model saved_models/best.pt \
                             --datasets "IEMOCAP:features/iemocap_val.pkl" \
                                        "MELD:features/meld_test.pkl"
```

### Feature Extraction (Generic)
```bash
# Extract features for any dataset
python feature_extract/extract_features_generic.py \
    --metadata metadata/iemocap_train.csv \
    --output features/IEMOCAP_BERT_wav2vec_train.pkl \
    --dataset iemocap_4 \
    --audio_model wav2vec2

# Supported datasets: iemocap_4, iemocap_6, mosei, cremad, ravdess, meld
```

### Legacy Commands
```bash
python main.py      # Basic BERT+ECAPA fusion model
python main1.py     # Original SOTA model (has paper discrepancies)
```

## Key Metrics
- **WA**: Weighted Accuracy
- **UA**: Unweighted Accuracy (balanced accuracy - PRIMARY metric for imbalanced data)
- **WF1**: Weighted F1-score
- **Macro-F1**: Macro-averaged F1 (treats all classes equally)

## Emotion Configurations

### iemocap_4 (Current)
0=anger, 1=happiness, 2=neutral, 3=sadness

### iemocap_6 (Paper evaluation)
0=happiness, 1=sadness, 2=neutral, 3=anger, 4=excitement, 5=frustration

### ekman_7 (Paper Table 1)
0=joy, 1=anger, 2=sadness, 3=fear, 4=surprise, 5=disgust, 6=neutral

## CLI Agents (in `/cli/agents/`)
Custom Claude Code agents for this project:
- `research-orchestrator`: Coordinate comprehensive research workflows
- `academic-researcher`: Literature review and paper analysis
- `technical-researcher`: Code and implementation analysis
- `code-reviewer`: Review code changes for quality/security
- `debugger`: Root cause analysis for errors

## Development Notes

### Dataset Split Protocol
Standard SER protocol: Sessions 1-3 train, Session 4 val, Session 5 test

### Class Imbalance Handling
- Class weights calculated from training distribution
- Label smoothing (0.1) in CrossEntropyLoss

### Dependencies
Key packages: torch, transformers, speechbrain, funasr (for emotion2vec), sklearn, torchaudio

### Paths
- Metadata CSVs: `metadata/`
- Feature pickles: `features/`
- Saved models: `saved_models/`
- Results: `results/`
