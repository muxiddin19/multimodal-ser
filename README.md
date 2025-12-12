# Multimodal Speech Emotion Recognition with Novel Fusion Mechanisms

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

State-of-the-art Speech Emotion Recognition (SER) using multimodal fusion of text (BERT) and audio (emotion2vec) features with **three novel contributions**: VAD-Guided Cross-Attention, Emotion-Aware Adaptive Fusion, and Modality-Invariant Contrastive Learning.

## Highlights

- **90.02% UA on IEMOCAP** (4-class) - significantly outperforms previous SOTA (~76%)
- **93.43% UA on cross-dataset transfer** (IEMOCAP → CREMA-D with fine-tuning)
- **90.79% UA with only 20% target data** - efficient few-shot transfer learning
- **Three novel contributions** with rigorous mathematical formulations

---

## Novel Contributions

### 1. VAD-Guided Cross-Attention (VGA)

Incorporates Valence-Arousal-Dominance (VAD) psychological theory into the attention mechanism to guide cross-modal interactions toward emotionally relevant features.

**Mathematical Formulation:**

Standard cross-attention computes:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

We augment this with VAD-based guidance:

$$A_{\text{guided}} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \lambda \cdot M_{\text{VAD}}\right)$$

where the VAD affinity matrix is computed as:

$$M_{\text{VAD}}(i,j) = -\|v_i - v_j\|_2$$

and $v_i, v_j \in \mathbb{R}^3$ are learned VAD projections of query and key features.

**Key Innovation:** By incorporating VAD structure, attention focuses on cross-modal features with similar emotional content, leveraging psychological emotion theory for better generalization.

---

### 2. Emotion-Aware Adaptive Fusion (EAAF)

Dynamically weights modalities based on emotion-discriminative confidence, unlike fixed-weight fusion approaches.

**Mathematical Formulation:**

Given text features $\mathbf{h}_t \in \mathbb{R}^d$ and audio features $\mathbf{h}_a \in \mathbb{R}^d$:

1. **Compute modality confidence gates:**
$$\mathbf{g} = [\mathbf{h}_t; \mathbf{h}_a; \mathbf{h}_t \odot \mathbf{h}_a] \in \mathbb{R}^{3d}$$
$$\alpha_t = \sigma(W_t \cdot \mathbf{g} + b_t) \in \mathbb{R}^d$$
$$\alpha_a = \sigma(W_a \cdot \mathbf{g} + b_a) \in \mathbb{R}^d$$

2. **Adaptive fusion with learned interaction:**
$$\mathbf{h}_{\text{fused}} = \alpha_t \odot \mathbf{h}_t + \alpha_a \odot \mathbf{h}_a + \beta \cdot (\mathbf{h}_t \odot \mathbf{h}_a)$$

where $\odot$ denotes element-wise multiplication, $\sigma$ is sigmoid, and $\beta$ is a learnable scalar.

**Key Innovation:** Sample-specific modality importance addresses the observation that some emotions are better expressed vocally (e.g., anger) while others have clearer lexical markers (e.g., sadness).

---

### 3. Modality-Invariant Contrastive Learning (MICL)

Learns emotion representations that are invariant across modalities through cross-modal contrastive learning.

**Mathematical Formulation:**

Given text embeddings $\mathbf{z}_t \in \mathbb{R}^{B \times D}$ and audio embeddings $\mathbf{z}_a \in \mathbb{R}^{B \times D}$:

1. **Cross-modal contrastive loss (text-to-audio):**
$$\mathcal{L}_{t \rightarrow a} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\text{sim}(\mathbf{z}_t^i, \mathbf{z}_a^i)/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_t^i, \mathbf{z}_a^j)/\tau)}$$

2. **Symmetric MICL loss:**
$$\mathcal{L}_{\text{MICL}} = \frac{1}{2}(\mathcal{L}_{t \rightarrow a} + \mathcal{L}_{a \rightarrow t})$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ is cosine similarity and $\tau$ is temperature.

**Key Innovation:** MICL enforces that the same utterance's text and audio representations are close in embedding space, leading to better cross-modal alignment and improved few-shot transfer.

---

## Total Training Objective

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}} + \lambda_{\text{vad}} \cdot \mathcal{L}_{\text{vad}} + \lambda_{\text{micl}} \cdot \mathcal{L}_{\text{MICL}}$$

where:
- $\mathcal{L}_{\text{cls}}$: Cross-entropy classification loss with label smoothing
- $\mathcal{L}_{\text{vad}}$: MSE loss for VAD regression (auxiliary task)
- $\mathcal{L}_{\text{MICL}}$: Modality-Invariant Contrastive Loss

Default: $\lambda_{\text{cls}}=1.0$, $\lambda_{\text{vad}}=0.3$, $\lambda_{\text{micl}}=0.2$

---

## Results

### In-Domain Performance

| Dataset | Classes | Test UA | Test WA | Test WF1 |
|---------|---------|---------|---------|----------|
| IEMOCAP | 4 | **90.02%** | 90.47% | 90.46% |
| IEMOCAP | 6 | 65.12% | 64.65% | 63.73% |
| CREMA-D | 4 | 93.43%* | 93.47%* | 93.50%* |

*with fine-tuning from IEMOCAP

### Ablation Study (Novel Components)

| Configuration | Val UA | Val WA | Δ UA |
|---------------|--------|--------|------|
| Baseline (Standard Cross-Attention) | 93.33% | 93.11% | - |
| + VGA (VAD-Guided Attention) | 93.52% | 93.28% | +0.19% |
| + EAAF (Adaptive Fusion) | 93.61% | 93.35% | +0.28% |
| + MICL (Contrastive Learning) | 93.48% | 93.22% | +0.15% |
| **Full Model (VGA + EAAF + MICL)** | **93.85%** | **93.62%** | **+0.52%** |

### Few-Shot Transfer Learning (IEMOCAP → CREMA-D)

| Target Data | Samples | Test UA | Test WA |
|-------------|---------|---------|---------|
| 10% | 390 | 74.53% | 73.47% |
| 20% | 782 | **90.79%** | 90.68% |
| 50% | 1,959 | 93.59% | 93.67% |
| 100% | 3,920 | 93.35% | 93.40% |

---

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
│  VAD-Guided Cross-Attention     │  ← Novel Component 1
│  (Bidirectional, num_layers)    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Emotion-Aware Adaptive Fusion  │  ← Novel Component 2
│  (Dynamic Modality Weighting)   │
└─────────────────────────────────┘
    │
    ├── MICL Projector ───────────────→ L_MICL  ← Novel Component 3
    │
    ▼
┌─────────────────────────────────┐
│     Classification Head         │
│     MLP → Softmax               │
└─────────────────────────────────┘
    │
    ▼
Output: Emotion Probabilities + VAD
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (for GPU support)
- ffmpeg (for MELD video processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/muxiddin19/multimodal-ser.git
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
├── main_acl2026.py              # Novel model training (ACL 2026)
├── main_icassp2026.py           # Baseline training script
├── models/
│   ├── __init__.py
│   └── novel_components.py      # VGA, EAAF, MICL implementations
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

### Training with Novel Components (ACL 2026)

```bash
python main_acl2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --test features/IEMOCAP_BERT_emotion2vec_test.pkl \
    --audio_dim 1024 --hidden_dim 384 \
    --vad_lambda 0.1 --micl_weight 0.2 \
    --num_runs 5 --output results_acl2026.json
```

### Novel Component Ablation Study

```bash
python main_acl2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --ablation --output results_ablation_novel.json
```

### Baseline Training (Standard Cross-Attention)

```bash
python main_icassp2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --test features/IEMOCAP_BERT_emotion2vec_test.pkl \
    --audio_dim 1024 --num_runs 5 --output results_baseline.json
```

### Cross-Dataset Fine-Tuning

```bash
python train_finetune.py \
    --source_train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --source_val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --target_train features/CREMAD_emotion2vec_train.pkl \
    --target_val features/CREMAD_emotion2vec_val.pkl \
    --target_test features/CREMAD_emotion2vec_test.pkl \
    --output results_finetune.json
```

### Few-Shot Learning

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

## Datasets

### Supported Datasets

| Dataset | Classes | Samples | Language |
|---------|---------|---------|----------|
| IEMOCAP | 4/6 | ~5,500 | English |
| CREMA-D | 4/6 | ~7,400 | English |
| MELD | 4/7 | ~13,000 | English |

### Emotion Labels

**4-class (IEMOCAP format):**
- 0: Anger, 1: Happiness, 2: Neutral, 3: Sadness

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
@inproceedings{multimodal-ser-acl2026,
  title={VAD-Guided Multimodal Fusion with Adaptive Weighting for Speech Emotion Recognition},
  author={Your Name},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
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
