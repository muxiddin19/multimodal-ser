# Sentimentogram: Interpretable Multimodal Speech Emotion Recognition with VAD-Guided Attention and Emotion-Aware Typography Visualization

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5+-red.svg" alt="PyTorch 2.5+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-green.svg" alt="ACL 2026"></a>
</p>

<p align="center">
  <b>State-of-the-art interpretable Speech Emotion Recognition (SER)</b><br>
  Multimodal fusion of text (BERT) and audio (emotion2vec) with psychological grounding
</p>

---

## Abstract

We present **Sentimentogram**, an interpretable multimodal speech emotion recognition system that combines **VAD-guided cross-attention**, **constrained adaptive fusion**, and **emotion-aware typography visualization**. Our approach achieves state-of-the-art results on IEMOCAP (93.0% WA on 4-class) while providing transparent modality contribution analysis. The system includes a novel subtitle visualization pipeline that renders word-level emotions with culturally-adaptive typography, and a preference-learning module for user personalization.

**Key Results:**
| Dataset | Configuration | WA (%) | UA (%) |
|---------|--------------|--------|--------|
| IEMOCAP | 4-class | **93.0** | **93.0** |
| IEMOCAP | 5-class | 78.0 | 78.0 |
| IEMOCAP | 6-class | 69.2 | 68.8 |
| CREMA-D | 6-class | 92.9 | 92.9 |

---

## Novel Contributions

| # | Contribution | Description |
|---|--------------|-------------|
| 1 | **VAD-Guided Cross-Attention** | Psychological grounding via Valence-Arousal-Dominance affinity |
| 2 | **Constrained Adaptive Fusion** | Interpretable gates with simplex constraint (sum-to-one) |
| 3 | **Hard Negative Mining MICL** | Cross-modal contrastive learning with curriculum sampling |
| 4 | **Emotion-Aware Typography (Sentimentogram)** | Word-level emotion visualization for video subtitles |
| 5 | **Preference-Learning Personalization** | Data-driven style adaptation from pairwise feedback |

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VGA-FUSION ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                              ┌──────────────┐             │
│  │    BERT      │                              │  emotion2vec │             │
│  │  (768-dim)   │                              │  (1024-dim)  │             │
│  └──────┬───────┘                              └──────┬───────┘             │
│         │                                             │                      │
│         ▼                                             ▼                      │
│  ┌──────────────┐                              ┌──────────────┐             │
│  │  Linear+LN   │                              │  Linear+LN   │             │
│  │  (384-dim)   │                              │  (384-dim)   │             │
│  └──────┬───────┘                              └──────┬───────┘             │
│         │                                             │                      │
│         └─────────────────┬───────────────────────────┘                      │
│                           │                                                  │
│                           ▼                                                  │
│         ┌─────────────────────────────────────┐                             │
│         │   ★ VAD-GUIDED CROSS-ATTENTION      │                             │
│         │   ─────────────────────────────     │                             │
│         │   Q_t ←─── h_t    h_a ───→ K_a, V_a │                             │
│         │                                     │                             │
│         │   A = softmax(QK^T/√d + λ·M_VAD)   │ ←── VAD Affinity Matrix     │
│         │                                     │     (Valence, Arousal,      │
│         │   Bidirectional: t→a and a→t       │      Dominance)             │
│         └─────────────────┬───────────────────┘                             │
│                           │                                                  │
│                           ▼                                                  │
│         ┌─────────────────────────────────────┐                             │
│         │   ★ CONSTRAINED ADAPTIVE FUSION     │                             │
│         │   ─────────────────────────────     │                             │
│         │   z = α_t·h_t + α_a·h_a + α_i·h_i  │                             │
│         │                                     │                             │
│         │   Constraint: α_t + α_a + α_i = 1  │ ←── Interpretable Gates     │
│         │   (Simplex constraint)              │                             │
│         └─────────────────┬───────────────────┘                             │
│                           │                                                  │
│         ┌─────────────────┴───────────────────┐                             │
│         │                                     │                             │
│         ▼                                     ▼                             │
│  ┌──────────────┐                      ┌──────────────┐                     │
│  │  MLP + Softmax│                      │ ★ MICL Loss  │                     │
│  │  (Emotion)   │                      │ (Contrastive)│                     │
│  └──────────────┘                      └──────────────┘                     │
│                                                                              │
│  Multi-Task Loss: L = L_focal + λ_v·L_VAD + λ_m·L_MICL                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              ★ = Novel Contribution
```

---

## Sentimentogram Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SENTIMENTOGRAM PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │  Input  │───▶│    ASR      │───▶│    Feature      │───▶│  VGA-Fusion  │  │
│  │  Video  │    │  (Whisper)  │    │   Extraction    │    │    Model     │  │
│  └─────────┘    └─────────────┘    └─────────────────┘    └──────┬───────┘  │
│                       │                    │                      │          │
│                       │                    │                      │          │
│                       ▼                    ▼                      ▼          │
│                 [Word-level        [BERT + emotion2vec]    [Emotion +       │
│                  timestamps]        embeddings]             VAD scores]     │
│                                                                   │          │
│                                                                   ▼          │
│                                          ┌────────────────────────────────┐  │
│                                          │     VAD → Style Mapping        │  │
│                                          │  ────────────────────────────  │  │
│                                          │  Valence → Color Hue           │  │
│                                          │  Arousal → Font Size/Weight    │  │
│                                          │  Dominance → Font Style        │  │
│                                          └────────────────┬───────────────┘  │
│                                                           │                  │
│                                                           ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      EMOTION-AWARE SUBTITLES                          │   │
│  │  ──────────────────────────────────────────────────────────────────  │   │
│  │                                                                       │   │
│  │   "I am"  →  [neutral, gray, 1.0x]                                   │   │
│  │   "SO"    →  [anger, RED, 1.3x, BOLD]                                │   │
│  │   "happy" →  [happiness, gold, 1.2x, italic]                         │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Optional: Preference-Learning Personalization (user-specific styling)      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Demo

**Watch Sentimentogram in action on TED Talks:**

[![Demo Video](https://img.shields.io/badge/Demo-Google%20Drive-blue)](https://drive.google.com/file/d/1jCQJbIAbtNDGf2GunXnjgWqmZWq9kvY6/view?usp=drive_link)

<table>
<tr>
<td><img src="demo/capture/sentimentogram_honest.jpg" width="400"/></td>
<td><img src="demo/capture/sentimentogram_think.jpg" width="400"/></td>
</tr>
<tr>
<td align="center"><em>"BEING HONEST" (anger) + "you think of" (happiness)</em></td>
<td align="center"><em>"I think" (happiness) + "MOST PEOPLE" (anger)</em></td>
</tr>
</table>

---

## Technical Details

### VAD-Guided Cross-Attention

Standard cross-attention:
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

VAD-guided attention with psychological grounding:
$$A_{\text{guided}} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \lambda \cdot M_{\text{VAD}}\right)$$

where $M_{\text{VAD}}(i,j) = -\|v_i - v_j\|_2$ is the VAD affinity matrix based on Russell's circumplex model of affect.

### Constrained Adaptive Fusion

Interpretable modality weighting with simplex constraint:
$$\mathbf{z} = \alpha_t \cdot \mathbf{h}_t + \alpha_a \cdot \mathbf{h}_a + \alpha_i \cdot (\mathbf{h}_t \odot \mathbf{h}_a)$$
$$\text{s.t. } \alpha_t + \alpha_a + \alpha_i = 1, \quad \alpha_i \geq 0$$

### VAD-to-Typography Mapping

| VAD Dimension | Low Value | High Value | Visual Effect |
|---------------|-----------|------------|---------------|
| **Valence** | Cool colors (blue, gray) | Warm colors (yellow, gold) | Color hue |
| **Arousal** | Small, light font | Large, bold font | Size & weight |
| **Dominance** | Italic, thin | Upright, heavy | Font style |

---

## Resources

### Core Components

| Resource | Description | Link |
|----------|-------------|------|
| **Preference Learning Module** | Bradley-Terry pairwise ranking model | [preference_learning.py](https://github.com/muxiddin19/multimodal-ser/blob/main/models/preference_learning.py) |
| **Experiment Runner** | Ablation study and evaluation scripts | [run_preference_learning.py](https://github.com/muxiddin19/multimodal-ser/blob/main/experiments/run_preference_learning.py) |
| **Novel Components** | VGA, EAAF, MICL implementations | [novel_components.py](https://github.com/muxiddin19/multimodal-ser/blob/main/models/novel_components.py) |

### Preference Learning Data

| Resource | Description | Link |
|----------|-------------|------|
| **Synthetic Dataset** | 20 users × 12 comparisons (240 pairs) | [preference_data_synthetic.json](https://github.com/muxiddin19/multimodal-ser/blob/main/acl2026/data/preference_data_synthetic.json) |
| **Real Data Template** | Template for 5 real users (60 pairs) | [preference_data_real.json](https://github.com/muxiddin19/multimodal-ser/blob/main/acl2026/data/preference_data_real.json) |
| **Collection Guide** | Instructions for collecting user preferences | [data_collection_guide.md](https://github.com/muxiddin19/multimodal-ser/blob/main/acl2026/data/data_collection_guide.md) |

### Paper & Documentation

| Resource | Description | Link |
|----------|-------------|------|
| **ACL 2026 Paper** | Full paper with appendix | [acl_latex.pdf](acl2026/acl_latex.pdf) |
| **Demo Video** | Sentimentogram on TED Talks | [Google Drive](https://drive.google.com/file/d/1jCQJbIAbtNDGf2GunXnjgWqmZWq9kvY6/view) |

---

## Preference-Learning Personalization

Instead of hard-coding subtitle styles based on cultural assumptions (which risk stereotyping), we learn user preferences from minimal pairwise feedback.

**Problem Formulation (Bradley-Terry Model):**
$$P(s_A \succ s_B | u, c) = \sigma(f(u, c, s_A) - f(u, c, s_B))$$

where $u$ = user attributes, $c$ = emotional context, $s$ = subtitle style.

**Hybrid Dataset:**
- **Synthetic:** 20 users × 12 comparisons = 240 pairs
- **Real:** 5 users × 12 comparisons = 60 pairs
- **Total:** 300 preference pairs

**Results:**

| Method | Accuracy | p-value |
|--------|----------|---------|
| Random | 50.3% ± 2.2% | - |
| Rule-based | 43.8% ± 2.6% | 0.08 |
| **Learned (Ours)** | **58.3% ± 4.9%** | **0.012** |

**Key Finding:** Rule-based performs *worse* than random, demonstrating that demographic assumptions do not reliably predict individual preferences.

---

## Installation

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

---

## Usage

### Training VGA-Fusion Model

```bash
python main_acl2026.py \
    --train features/IEMOCAP_BERT_emotion2vec_train.pkl \
    --val features/IEMOCAP_BERT_emotion2vec_val.pkl \
    --test features/IEMOCAP_BERT_emotion2vec_test.pkl \
    --audio_dim 1024 --hidden_dim 384 \
    --vad_lambda 0.1 --micl_weight 0.2 \
    --num_runs 5 --output results_acl2026.json
```

### Running Preference Learning Experiments

```bash
python experiments/run_preference_learning.py
```

### Sentimentogram Demo

```bash
python demo/sentimentogram_demo_v3.py \
    --video demo/videos/tedx1.mp4 \
    --output demo/output/result.html \
    --culture western
```

---

## Project Structure

```
multimodal-ser/
├── main_acl2026.py                    # VGA-Fusion training script
├── models/
│   ├── novel_components.py            # VGA, EAAF, MICL implementations
│   └── preference_learning.py         # Bradley-Terry preference model
├── experiments/
│   └── run_preference_learning.py     # Preference learning evaluation
├── demo/
│   ├── sentimentogram_demo_v3.py      # Visualization demo
│   └── capture/                       # Demo screenshots
├── acl2026/
│   ├── acl_latex.tex                  # Paper source
│   └── data/
│       ├── preference_data_synthetic.json
│       ├── preference_data_real.json
│       └── data_collection_guide.md
└── feature_extract/                   # Feature extraction scripts
```

---

## Citation

```bibtex
@inproceedings{author2026sentimentogram,
  title={Sentimentogram: Interpretable Multimodal Speech Emotion Recognition
         with VAD-Guided Attention and Emotion-Aware Typography Visualization},
  author={Author, First and Author, Second},
  booktitle={Proceedings of the 64th Annual Meeting of the Association
             for Computational Linguistics (ACL 2026)},
  year={2026}
}
```

---

## Acknowledgments

This work builds upon:
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec) for audio emotion features
- [IEMOCAP](https://sail.usc.edu/iemocap/) dataset
- Russell's circumplex model of affect for VAD grounding

---

## License

MIT License - see [LICENSE](LICENSE) for details.
