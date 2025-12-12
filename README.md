# Multimodal Speech Emotion Recognition with emotion2vec

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

State-of-the-art Speech Emotion Recognition (SER) using multimodal fusion of text (BERT) and audio (emotion2vec) features with cross-attention mechanism.

## Highlights

- **90.02% UA on IEMOCAP** (4-class) - significantly outperforms previous SOTA (~76%)
- **93.43% UA on cross-dataset transfer** (IEMOCAP → CREMA-D with fine-tuning)
- **90.79% UA with only 20% target data** - efficient few-shot transfer learning
- Comprehensive evaluation on IEMOCAP, CREMA-D, and MELD datasets

## Results

### In-Domain Performance

| Dataset | Classes | Test UA | Test WA | Test WF1 |
|---------|---------|---------|---------|----------|
| IEMOCAP | 4 | **90.02%** | 90.47% | 90.46% |
| IEMOCAP | 6 | 65.12% | 64.65% | 63.73% |
| CREMA-D | 4 | 93.43%* | 93.47%* | 93.50%* |

*with fine-tuning from IEMOCAP

### Ablation Study

| Configuration | Val UA | Val WA |
|---------------|--------|--------|
| Text-only (BERT) | 66.09% | 65.83% |
| Audio-only (emotion2vec) | **93.27%** | **93.65%** |
| Simple Fusion | 93.40% | 93.32% |
| Full Model (Cross-Attention) | 93.33% | 93.11% |

### Few-Shot Transfer Learning (IEMOCAP → CREMA-D)

| Target Data | Samples | Test UA | Test WA |
|-------------|---------|---------|---------|
| 10% | 390 | 74.53% | 73.47% |
| 20% | 782 | **90.79%** | 90.68% |
| 50% | 1,959 | 93.59% | 93.67% |
| 100% | 3,920 | 93.35% | 93.40% |

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (for GPU support)
- ffmpeg (for MELD video processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-ser.git
cd multimodal-ser

# Create conda environment
conda create -n ser python=3.10
conda activate ser

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
multimodal-ser/
├── main_icassp2026.py          # Main training script
├── train_domain_adaptation.py   # Domain adaptation (DANN, MMD, CORAL)
├── train_combined.py            # Combined multi-dataset training
├── train_finetune.py            # Fine-tuning script
├── train_fewshot.py             # Few-shot learning experiments
├── domain_adaptation.py         # DA modules (GRL, MMD, CORAL losses)
├── feature_extract/
│   ├── IEMOCAP-BERT_wav2vec.py      # Wav2Vec2 feature extraction
│   ├── IEMOCAP-BERT_emotion2vec.py  # emotion2vec feature extraction
│   ├── IEMOCAP_6class_emotion2vec.py # 6-class IEMOCAP features
│   └── extract_cross_dataset.py      # CREMA-D, MELD extraction
├── metadata/
│   ├── create_iemocap_metadata.py    # IEMOCAP 4-class metadata
│   ├── create_iemocap_6class.py      # IEMOCAP 6-class metadata
│   ├── create_cremad_metadata.py     # CREMA-D metadata
│   └── create_meld_metadata.py       # MELD metadata
├── features/                    # Extracted features (not in repo)
├── saved_models/               # Trained models
└── results/                    # Experiment results (JSON)
```

## Usage

### 1. Prepare Metadata

```bash
# IEMOCAP (update paths in scripts first)
python metadata/create_iemocap_metadata.py      # 4-class
python metadata/create_iemocap_6class.py        # 6-class

# CREMA-D
python metadata/create_cremad_metadata.py

# MELD
python metadata/create_meld_metadata.py
```

### 2. Extract Features

```bash
# IEMOCAP with emotion2vec (recommended)
python feature_extract/IEMOCAP-BERT_emotion2vec.py

# IEMOCAP 6-class
python feature_extract/IEMOCAP_6class_emotion2vec.py

# CREMA-D / MELD
python feature_extract/extract_cross_dataset.py --dataset cremad
python feature_extract/extract_cross_dataset.py --dataset meld
```

### 3. Training

#### Basic Training (IEMOCAP 4-class)
```bash
python main_icassp2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --test features/IEMOCAP_BERT_emotion2vec_test.pkl \
    --audio_dim 1024 --dropout 0.3 --weight_decay 0.01 \
    --num_layers 2 --hidden_dim 384 \
    --num_runs 5 --output results.json
```

#### 6-Class IEMOCAP
```bash
python main_icassp2026.py \
    --train features/IEMOCAP_6class_emotion2vec_train.pkl \
    --val features/IEMOCAP_6class_emotion2vec_val.pkl \
    --test features/IEMOCAP_6class_emotion2vec_test.pkl \
    --audio_dim 1024 --num_classes 6 --emotion_config iemocap_6 \
    --num_runs 3 --output results_6class.json
```

#### Ablation Study
```bash
python main_icassp2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --ablation --output results_ablation.json
```

### 4. Cross-Dataset Experiments

#### Domain Adaptation (DANN)
```bash
python train_domain_adaptation.py \
    --source_train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --source_val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --target_train features/CREMAD_emotion2vec_train.pkl \
    --target_test features/CREMAD_emotion2vec_test.pkl \
    --method dann --lambda_domain 0.1 \
    --output results_da_dann.json
```

#### Combined Training
```bash
python train_combined.py \
    --iemocap_train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --iemocap_val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --iemocap_test features/IEMOCAP_BERT_emotion2vec_test.pkl \
    --cremad_train features/CREMAD_emotion2vec_train.pkl \
    --cremad_val features/CREMAD_emotion2vec_val.pkl \
    --cremad_test features/CREMAD_emotion2vec_test.pkl \
    --output results_combined.json
```

#### Fine-Tuning (Best for Cross-Dataset)
```bash
python train_finetune.py \
    --source_train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --source_val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --target_train features/CREMAD_emotion2vec_train.pkl \
    --target_val features/CREMAD_emotion2vec_val.pkl \
    --target_test features/CREMAD_emotion2vec_test.pkl \
    --output results_finetune.json
```

#### Few-Shot Learning
```bash
python train_fewshot.py \
    --source_train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --source_val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --target_train features/CREMAD_emotion2vec_train.pkl \
    --target_val features/CREMAD_emotion2vec_val.pkl \
    --target_test features/CREMAD_emotion2vec_test.pkl \
    --percentages "10,20,50,100" \
    --output results_fewshot.json
```

## Model Architecture

```
Input: Text + Audio
    │
    ├── Text Branch
    │   └── BERT → [CLS] token (768-dim)
    │
    └── Audio Branch
        └── emotion2vec → pooled features (1024-dim)
    │
    ▼
┌─────────────────────────────────┐
│     Projection Layers           │
│  Text: 768 → hidden_dim         │
│  Audio: 1024 → hidden_dim       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Bidirectional Cross-Attention │
│   (num_layers × num_heads)      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Fusion + Classification     │
│     Concat → MLP → Softmax      │
└─────────────────────────────────┘
    │
    ▼
Output: Emotion Probabilities
```

## Datasets

### Supported Datasets

| Dataset | Classes | Samples | Language |
|---------|---------|---------|----------|
| IEMOCAP | 4/6 | ~5,500 | English |
| CREMA-D | 4/6 | ~7,400 | English |
| MELD | 4/7 | ~13,000 | English |

### Emotion Labels

**4-class (IEMOCAP format):**
- 0: Anger
- 1: Happiness
- 2: Neutral
- 3: Sadness

**6-class (IEMOCAP extended):**
- 0: Happiness, 1: Sadness, 2: Neutral, 3: Anger, 4: Excitement, 5: Frustration

## Key Components

### emotion2vec
We use [emotion2vec](https://github.com/ddlBoJack/emotion2vec) for audio feature extraction, which provides emotion-aware speech representations pre-trained on large-scale emotional speech data.

### Domain Adaptation Methods
- **DANN**: Domain Adversarial Neural Network with gradient reversal
- **MMD**: Maximum Mean Discrepancy loss
- **CORAL**: Correlation Alignment loss

## Citation

If you use this code, please cite:

```bibtex
@article{multimodal-ser-2025,
  title={Multimodal Speech Emotion Recognition with emotion2vec and Cross-Attention Fusion},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

- [emotion2vec](https://github.com/ddlBoJack/emotion2vec) for audio features
- [IEMOCAP](https://sail.usc.edu/iemocap/) dataset
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) dataset
- [MELD](https://affective-meld.github.io/) dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please open an issue or contact the authors.
