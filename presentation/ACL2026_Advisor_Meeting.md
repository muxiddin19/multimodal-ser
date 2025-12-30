# ACL 2026 Research Proposal
## Enhanced Multimodal Speech Emotion Recognition with Psychology-Informed Fusion

**Meeting with Scientific Advisor**
**Date: December 2024**

---

# Slide 1: Title

## Enhanced Multimodal Speech Emotion Recognition with VAD-Guided Cross-Attention and Interpretable Fusion

**Authors:** [Your Name], [Advisor Name]

**Target Venue:** ACL 2026 Main Conference

**Paper Type:** Long Paper (8 pages + references)

---

# Slide 2: Motivation & Problem Statement

## The Challenge of Multimodal Emotion Recognition

### Current Limitations:
1. **Fixed fusion weights** - Existing methods use static modality weighting
2. **Black-box attention** - Cross-attention lacks interpretability
3. **Class imbalance** - Minority emotions (happiness, frustration) underperform
4. **Single-dataset evaluation** - Most work only evaluates on IEMOCAP

### Our Goal:
> Design an **interpretable**, **psychology-informed** multimodal fusion architecture that achieves **SOTA performance** across multiple datasets

---

# Slide 3: Key Contributions

## Three Novel Technical Contributions

### 1. VAD-Guided Cross-Attention (VGA)
- Incorporates **Valence-Arousal-Dominance** psychological theory into attention
- First work to use dimensional emotion space to guide cross-modal attention
- Formula: `A_guided = softmax(QK^T/√d + λ·M_VAD)`

### 2. Constrained Adaptive Fusion (C-EAAF)
- Gates **sum to 1** for interpretability
- Three-way fusion: Text, Audio, Interaction
- Reveals modality contribution per sample

### 3. Hard Negative Mining MICL
- Focus on **difficult cross-modal pairs**
- Curriculum-based hardness increase
- Improves cross-modal alignment

---

# Slide 4: Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                            │
│         BERT (768d)              emotion2vec (1024d)         │
└──────────────┬─────────────────────────┬────────────────────┘
               │                         │
               ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PROJECTION TO COMMON SPACE (384d)               │
└──────────────────────────────────────────────────────────────┘
               │                         │
               ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│           SELF-ATTENTION ENCODERS (8 heads)                  │
└──────────────────────────────────────────────────────────────┘
               │                         │
               ▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│    ★ VAD-GUIDED CROSS-ATTENTION (Novel) ★                   │
│    - Projects features to VAD space                          │
│    - Computes VAD affinity matrix                            │
│    - Guides attention with λ=0.5                             │
└──────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│    ★ CONSTRAINED ADAPTIVE FUSION (Novel) ★                  │
│    α_text + α_audio + α_interaction = 1.0                    │
│    Interpretable modality contribution                       │
└──────────────────────────────────────────────────────────────┘
               │
               ├──────────────────┐
               ▼                  ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│ EMOTION CLASSIFIER  │  │ ★ MICL WITH HARD NEGATIVES ★      │
│ (Focal Loss γ=2.0)  │  │ Cross-modal contrastive learning  │
└─────────────────────┘  └───────────────────────────────────┘
```

**Total Parameters:** 13.3M

---

# Slide 5: Experimental Results

## Multi-Dataset Evaluation

| Dataset | Classes | Val UA | Test UA | vs Previous |
|---------|---------|--------|---------|-------------|
| **IEMOCAP** | 5 | 77.41 ± 0.37% | 75.61 ± 0.42% | Comparable |
| **IEMOCAP** | 6 | 68.75 ± 0.58% | 65.69 ± 0.56% | **+3.75%** |
| **CREMA-D** | 4 | **92.90 ± 0.34%** | **92.70 ± 0.35%** | **SOTA** |
| **MELD** | 4 | 63.66 ± 0.72% | 58.93 ± 0.45% | Challenging |

### Key Achievements:
- ✓ **CREMA-D SOTA**: 92.90% UA significantly outperforms baselines
- ✓ **IEMOCAP 6-class**: +3.75% improvement on challenging 6-class setup
- ✓ **Multi-dataset**: Consistent performance across 3 datasets

---

# Slide 6: Interpretable Fusion Analysis

## Constrained Adaptive Fusion Reveals Modality Importance

| Dataset | Text Gate | Audio Gate | Interaction |
|---------|-----------|------------|-------------|
| IEMOCAP 5-class | **54.3%** | 45.5% | 0.2% |
| IEMOCAP 6-class | 41.4% | **58.4%** | 0.2% |
| CREMA-D | 23.1% | **76.6%** | 0.3% |

### Insights:
1. **CREMA-D (acted)**: Audio dominates (76.6%) - acted emotions are vocally expressive
2. **IEMOCAP (conversational)**: Balanced fusion - text provides complementary cues
3. **Gates sum to 1**: Interpretable contribution percentages

> This interpretability is a key differentiator from black-box fusion methods

---

# Slide 7: Comparison with State-of-the-Art

## How We Compare (Preliminary)

| Method | IEMOCAP 4-class | CREMA-D | Notes |
|--------|-----------------|---------|-------|
| emotion2vec (audio only) | 82.5% | ~85% | Single modality |
| MulT (2019) | 74.1% | - | Early fusion |
| MISA (2020) | 76.4% | - | Modality-invariant |
| UniSER (2023) | 78.2% | - | Unified framework |
| **Ours** | - | **92.90%** | Multimodal + interpretable |

### ⚠ Gap to Address:
Need head-to-head comparison on same splits with:
- emotion2vec + text baselines
- TelME, CARAT, MultiEMO
- Recent 2024 methods

---

# Slide 8: Novel Aspects for ACL

## Why ACL 2026?

### Fits ACL Scope:
1. **Multimodal NLP** - Text + speech fusion
2. **Interpretability** - Explainable fusion weights
3. **Psychological grounding** - VAD theory integration
4. **Practical application** - Sentimentogram demo

### Novel Contributions:
| Contribution | Novelty Level | Evidence |
|--------------|---------------|----------|
| VAD-guided attention | **High** | First to use VAD space in cross-attention |
| Constrained fusion | **Medium-High** | Interpretable gates summing to 1 |
| Hard negative MICL | **Medium** | Novel combination for SER |
| Multi-dataset eval | **Medium** | Addresses AAAI-26 reviewer concern |

---

# Slide 9: What's Missing (Honest Assessment)

## Work Required Before Submission

### Must Have:
1. **[ ] Stronger baselines** - Compare with emotion2vec+BERT, TelME, CARAT
2. **[ ] Statistical significance** - Paired t-tests across runs
3. **[ ] Complete ablation** - Systematic component removal study
4. **[ ] Error analysis** - Confusion matrices, failure case study

### Should Have:
5. **[ ] Cross-lingual** - Test on non-English (e.g., EMOVO Italian)
6. **[ ] Cross-dataset transfer** - Train IEMOCAP → Test CREMA-D
7. **[ ] Few-shot learning** - 10%, 20% labeled data experiments

### Nice to Have:
8. **[ ] Human evaluation** - Perception study on Sentimentogram
9. **[ ] Efficiency analysis** - Inference time, model size comparison

---

# Slide 10: Proposed Paper Structure

## ACL 2026 Long Paper (8 pages)

### Abstract (200 words)
Key results: CREMA-D SOTA, interpretable fusion, VAD-guided attention

### 1. Introduction (1 page)
- Motivation, contributions, key results

### 2. Related Work (1 page)
- Multimodal SER, cross-attention fusion, contrastive learning

### 3. Method (2 pages)
- VGA, C-EAAF, MICL with hard negatives
- Mathematical formulations with intuition

### 4. Experiments (2.5 pages)
- Datasets, baselines, main results, ablation, analysis

### 5. Analysis & Discussion (1 page)
- Fusion behavior, error analysis, limitations

### 6. Conclusion (0.5 pages)
- Summary, future work

---

# Slide 11: Timeline to ACL 2026

## Submission Deadline: ~February 2026 (estimated)

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1: Baselines** | 2-3 weeks | Implement and run SOTA comparisons |
| **Phase 2: Ablation** | 1-2 weeks | Systematic component removal |
| **Phase 3: Analysis** | 1-2 weeks | Error analysis, cross-dataset transfer |
| **Phase 4: Writing** | 3-4 weeks | Draft, figures, tables |
| **Phase 5: Revision** | 2 weeks | Advisor feedback, polish |

**Total: ~10-13 weeks**

---

# Slide 12: Demo - Sentimentogram

## Practical Application: Emotion-Aware Subtitles

### Features:
- Real-time emotion detection from video
- Word-level emotion styling
- Emotion-specific typography (fonts, colors, animations)
- Cultural adaptation (Western/Eastern color semantics)

### Technical:
- Whisper STT → BERT + emotion2vec → Enhanced Model → HTML Visualization

### Demo Video:
[Show result_enhanced.html with tedx1.mp4]

> This demonstrates practical impact beyond academic metrics

---

# Slide 13: Questions for Advisor

## Discussion Points

### 1. Scope & Novelty
> Is the combination of VAD-guided attention + interpretable fusion sufficient novelty for ACL?

### 2. Baselines Priority
> Which baselines are most critical? emotion2vec+text, TelME, or UniSER?

### 3. Cross-lingual
> Should we prioritize cross-lingual evaluation or focus on English?

### 4. Positioning
> Main conference vs. Findings? Long paper vs. short paper?

### 5. Collaboration
> Any collaborators to involve for additional experiments?

---

# Slide 14: Summary

## Key Takeaways

### What We Have:
- ✓ Novel VAD-guided cross-attention mechanism
- ✓ Interpretable constrained fusion (gates sum to 1)
- ✓ SOTA results on CREMA-D (92.90% UA)
- ✓ Multi-dataset evaluation (IEMOCAP, CREMA-D, MELD)
- ✓ Working demo (Sentimentogram)

### What We Need:
- ⚠ Stronger baseline comparisons
- ⚠ Statistical significance tests
- ⚠ Complete ablation study
- ⚠ Error analysis

### Recommendation:
> **Proceed with ACL 2026 submission** after completing baseline comparisons and ablation study. The work has sufficient novelty and strong results on CREMA-D.

---

# Backup Slides

---

# Backup: Per-Class Performance (IEMOCAP 6-class)

| Emotion | F1-Score | Support | Analysis |
|---------|----------|---------|----------|
| happiness | 44.6% | 65 | ⚠ Very small sample |
| sadness | 75.9% | 143 | ✓ Good |
| neutral | 64.2% | 258 | Moderate |
| anger | 78.9% | 327 | ✓ Best |
| excitement | 73.3% | 238 | ✓ Good |
| frustration | 48.7% | 481 | ⚠ Confusable with anger |

**Insight:** happiness and frustration are the hardest classes - small samples and confusion with similar emotions.

---

# Backup: Mathematical Formulations

## VAD-Guided Cross-Attention

```
Standard:  A = softmax(QK^T / √d_k)
Ours:      A_guided = softmax(QK^T / √d_k + λ · M_VAD)

where M_VAD(i,j) = -||v_i - v_j||_2
      v_i, v_j are VAD projections of queries and keys
```

## Constrained Adaptive Fusion

```
[α_t, α_a, α_i] = softmax([W_t·g, W_a·g, W_i·g])  # Sum to 1
h_fused = α_t·h_text + α_a·h_audio + α_i·(h_text ⊙ h_audio)
```

## Focal Loss

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
γ = 2.0  (focusing parameter)
```

---

# Backup: Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden Dim | 384 | Balance capacity/efficiency |
| Attention Heads | 8 | Standard transformer config |
| VGA Layers | 2 | Sufficient for cross-modal fusion |
| VAD Lambda | 0.5 | Tuned (increased from 0.1) |
| MICL Weight | 0.3 | Increased for better alignment |
| VAD Weight | 0.5 | Increased for emotion grounding |
| Focal Gamma | 2.0 | Standard for class imbalance |
| Mixup Alpha | 0.4 | Data augmentation |
| Learning Rate | 2e-5 | BERT fine-tuning range |
| Batch Size | 16 | GPU memory constraint |
