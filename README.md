# Sentimentogram: Learning Personalized Emotion Visualizations from User Preferences

<p align="center">
  <a href="https://openreview.net/forum?id=CivSckXgby"><img src="https://img.shields.io/badge/ACL%202026-Findings-blue" alt="ACL 2026 Findings"></a>
  <a href="https://github.com/muxiddin19/multimodal-ser/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.5%2B-orange" alt="PyTorch 2.5+">
  <a href="https://github.com/muxiddin19/multimodal-ser/stargazers"><img src="https://img.shields.io/github/stars/muxiddin19/multimodal-ser" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=CivSckXgby">📄 Paper</a> •
  <a href="https://drive.google.com/file/d/1jCsz9mba0D7AxjaY1RV1uYc8XR9xeqy7/view">🎥 Demo Video</a> •
  <a href="#dataset--model-release">📦 Dataset & Models</a> •
  <a href="#citation">📖 Citation</a>
</p>

> **Accepted at Findings of ACL 2026**
> Mukhiddin Toshpulatov, Seungkyu Oh, Suan Lee, Jo'ra Kuvandikov, Wookey Lee

---

## Overview

Current speech emotion recognition (SER) systems produce predictions that users cannot interpret, visualize, or personalize — limiting real-world adoption. **Sentimentogram** is a human-centered framework that learns personalized emotion visualization preferences from pairwise comparisons rather than relying on demographic-based heuristics.

**Key finding:** Rule-based demographic adaptation performs **significantly below chance** (43.8% vs 50.1%, *p*=0.014). Learning from just 10–12 pairwise comparisons (~3 minutes of user effort) achieves **61.2% accuracy** (+7.7% over the best baseline, *p*<0.001), with A/B study confirming improved satisfaction (+8.7%) and comprehension (+5.8%).

---

## Pipeline

```
Accurate SER  →  Interpretable Fusion  →  Emotion-Aware Visualization  →  Preference Learning
  (77.97% UA)     (α_t + α_a + α_i = 1)    (dynamic typography)           (61.2% accuracy)
```

Given video input:
1. **Whisper ASR** transcribes and segments audio into utterances
2. **BERT** (text) + **emotion2vec** (audio) extract features
3. **VGA-Fusion** predicts emotion with interpretable modality attribution
4. **Sentimentogram renderer** applies personalized emotion-aware typography to subtitles

---

## Results

### Speech Emotion Recognition

| Dataset | Protocol | UA (%) |
|---|---|---|
| IEMOCAP 4-class | Validation | 93.02 ± 0.17 |
| IEMOCAP 5-class | Validation | **77.97 ± 0.33** |
| IEMOCAP 5-class | LOSO (5-fold CV) | 75.3 ± 1.1 |
| IEMOCAP 6-class | Validation | 68.75 ± 0.58 |
| CREMA-D | Validation | 92.90 ± 0.34 |
| MELD | Validation | 63.66 ± 0.72 |

### Preference Learning

| Method | Accuracy | *p*-value |
|---|---|---|
| Random | 50.1 ± 2.2% | — |
| Rule-based (demographic) | 43.8 ± 3.1% | 0.014 (below chance) |
| Collaborative filtering | 53.5 ± 2.4% | 0.08 |
| Hierarchical Bradley-Terry | 52.8 ± 2.5% | 0.12 |
| **Learned (Ours)** | **61.2 ± 2.8%** | **<0.001** |

### ASR Robustness
At 44.3% WER (Whisper on spontaneous speech), UA drops by only **−0.96%** — multimodal fusion naturally shifts weight toward audio when transcription quality degrades.

---

## Model Architecture

### VGA-Fusion

```
Text: BERT-base (768d) ─→ Linear+LN (384d) ─┐
                                              ├─→ VAD-Guided Cross-Attention ─→ Constrained Fusion ─→ Classifier
Audio: emotion2vec (1024d) ─→ Linear+LN (384d)─┘        ↑                           ↑
                                               VAD affinity bias         α_t + α_a + α_i = 1
```

**VAD-Guided Cross-Attention** incorporates a psychologically grounded Valence-Arousal-Dominance bias:

```
VGA(Q, K, V) = softmax( QKᵀ/√d + λ·M_VAD ) · V
```

where M_VAD(i,j) = −‖v_t^(i) − v_a^(j)‖₂ is the pairwise VAD affinity.

**Constrained Adaptive Fusion** enforces interpretability:

```
[α_t, α_a, α_i] = softmax(W_g · [h_t; h_a; h_t⊙h_a])
h_fused = α_t·h_t + α_a·h_a + α_i·(h_t⊙h_a)
```

If α_a = 0.76, audio contributes exactly 76% to the prediction — enabling per-sample explanations such as *"76% audio, 24% text"*.

**Training objective:**

```
L = L_focal + λ_micl · L_MICL + λ_vad · L_VAD
```

---

## Emotion-Aware Typography

Sentimentogram renders emotion predictions as dynamic subtitle typography:

| Emotion | Font Style | Color | Size |
|---|---|---|---|
| Anger | **BOLD UPPERCASE** | Red | 1.3× |
| Happy | *Bouncy* | Gold | 1.15× |
| Sad | *Italic* | Blue | 0.92× |
| Neutral | Regular | Gray | 1.0× |

VAD dimensions map systematically: Valence → color hue (cool to warm), Arousal → font size and weight, Dominance → font style (italic to upright). All colors meet WCAG 2.1 AA contrast standards.

---

## Dataset & Model Release

### Preference Dataset
**1,500 pairwise style comparisons** from 50 real users across 6 emotion categories.

> 📦 Download: [`data/preference_dataset/`](data/preference_dataset/) *(upload in progress — available before April 19, 2026)*

**Demographics:** Age groups (18–65+), accessibility needs (low vision, color blind, dyslexia, hearing impaired), cultural backgrounds (Western, East Asian, South Asian, Middle Eastern, African, Latin American), professions (student, healthcare, educator, tech, creative).

Each comparison contains:
- User profile vector (age group, accessibility needs, language region, device type)
- Two 5-dimensional style vectors (font size, color intensity, emphasis, animation, contrast)
- Binary preference label

### Pre-trained Models
> 📦 Models: *(upload in progress — available before April 19, 2026)*

| Model | Dataset | UA (%) | Download |
|---|---|---|---|
| VGA-Fusion | IEMOCAP 5-class | 77.97 | Coming soon |
| VGA-Fusion | CREMA-D | 92.90 | Coming soon |
| Preference model | 50-user study | 61.2 | Coming soon |

---

## Installation

```bash
git clone https://github.com/muxiddin19/multimodal-ser.git
cd multimodal-ser

conda create -n sentimentogram python=3.10
conda activate sentimentogram

pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.5+
- transformers (BERT-base-uncased)
- emotion2vec-plus-large
- openai-whisper

---

## Usage

### Train VGA-Fusion (SER)
```bash
python main_acl2026.py \
  --dataset iemocap \
  --num_classes 5 \
  --hidden_dim 384 \
  --num_heads 8 \
  --k_views 4 \
  --vad_lambda 0.5 \
  --micl_weight 0.3 \
  --epochs 100 \
  --lr 2e-5 \
  --batch_size 16 \
  --seed 42
```

### Run Preference Learning Experiments
```bash
python experiments/preference_learning.py \
  --data_path data/preference_dataset/ \
  --eval_mode within_user \
  --train_comparisons 12 \
  --test_comparisons 18
```

### Run Sentimentogram Demo (Video Subtitling)
```bash
python demo/sentimentogram_demo.py \
  --video_path your_video.mp4 \
  --model_path models/vga_fusion_iemocap5.pt \
  --user_profile data/user_profiles/example.json \
  --output_path output_styled.mp4
```

---

## Project Structure

```
multimodal-ser/
├── main_acl2026.py          # VGA-Fusion training entry point
├── requirements.txt
├── run_all_experiments.sh   # Reproduce all paper results
│
├── models/
│   ├── vga_fusion.py        # VAD-Guided Cross-Attention + Constrained Fusion
│   ├── micl.py              # Supervised Contrastive MICL
│   └── preference_model.py  # Bradley-Terry preference learning
│
├── experiments/
│   ├── preference_learning.py
│   ├── typography_eval.py
│   └── ablation_study.py
│
├── feature_extract/
│   ├── bert_extractor.py
│   └── emotion2vec_extractor.py
│
├── demo/
│   ├── sentimentogram_demo.py
│   └── screenshots/
│
├── data/
│   └── preference_dataset/  # 1,500 pairwise comparisons (50 users)
│
└── scripts/
    └── preprocess_datasets.py
```

---

## Reproducing Paper Results

All experiments reported in the paper can be reproduced with:

```bash
bash run_all_experiments.sh
```

Key hyperparameters (full table in paper Appendix A):

| Parameter | Value |
|---|---|
| Hidden dimension | 384 |
| Attention heads | 8 |
| K views | 4 |
| VAD guidance λ | 0.5 |
| MICL weight | 0.3 |
| VAD loss weight | 0.5 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Early stopping patience | 15 |
| Seeds | 5 runs (reported as mean ± std) |

---

## Demo

📹 **[Watch Demo Video](https://drive.google.com/file/d/1jCsz9mba0D7AxjaY1RV1uYc8XR9xeqy7/view)** — TED Talk subtitles rendered with emotion-aware typography.

Example renderings:
- *"I'm fine"* (sarcastic, anger detected) → **BOLD RED UPPERCASE**
- *"That's wonderful news!"* (happiness) → **Gold bouncy text, 1.15×**
- *"I don't know..."* (sadness) → *italic blue, 0.92×*

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{toshpulatov2026sentimentogram,
  title     = {Sentimentogram: Learning Personalized Emotion Visualizations from User Preferences},
  author    = {Toshpulatov, Mukhiddin and Oh, Seungkyu and Lee, Suan and Kuvandikov, Jo'ra and Lee, Wookey},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://openreview.net/forum?id=CivSckXgby}
}
```

---

## Acknowledgments

This work was supported by:
- **IITP–ITRC** grant funded by the Korea government (Ministry of Science and ICT), XVoice (RS-2022-II220641)
- **National Research Foundation of Korea (NRF)** (RS-2025-24534935)
- **Inha University**

We thank the IEMOCAP team (USC), CREMA-D contributors, and MELD dataset creators. Audio features use [emotion2vec](https://github.com/ddlBoJack/emotion2vec) (Ma et al., 2024). VAD grounding uses the [NRC-VAD lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) (Mohammad, 2018).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Ethics

User studies conducted under IRB Protocol #2024-0847 with informed consent. Preference data released with no personally identifying information. We encourage deployment only in transparent, opt-in settings. See the paper's Ethics Statement for full details.
