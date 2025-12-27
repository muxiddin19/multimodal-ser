# Preference Data Collection Guide

## Overview

This guide explains how to collect real user preference data for the Sentimentogram subtitle personalization system. The goal is to gather pairwise preferences from 5-10 real users to complement the synthetic data.

## Target: 5 Real Users

- **Minimum**: 5 users (60 comparisons)
- **Target**: 10 users (120 comparisons)
- **Comparisons per user**: 12 pairs
- **Time required**: ~5-10 minutes per user

## User Profile Collection

Before starting comparisons, collect these anonymous attributes:

```
1. Age Group: [ ] Young (18-35) [ ] Middle (36-55) [ ] Senior (56+)
2. Region: [ ] Western [ ] Eastern [ ] Other
3. Accessibility Needs: [ ] Yes [ ] No
4. Primary Device: [ ] Mobile [ ] Tablet [ ] Desktop
```

## Comparison Procedure

### Step 1: Show Video Clip with Two Subtitle Styles

For each comparison, show a short video clip (5-10 seconds) displaying the same emotional utterance with two different subtitle styles (A and B).

### Step 2: Present Style Options

| Style A | Style B |
|---------|---------|
| [font_size, color_intensity, emphasis, animation, contrast] | [different values] |

### Style Parameter Examples:

**Style A (Subtle):**
- Font size: 1.0x (normal)
- Color: Muted gray/blue
- Emphasis: Light italic
- Animation: None
- Contrast: Standard

**Style B (Expressive):**
- Font size: 1.3x (larger)
- Color: Vivid red/yellow
- Emphasis: Bold uppercase
- Animation: Bounce/shake
- Contrast: High

### Step 3: Record Preference

Ask: *"Which subtitle style better conveys the emotion of this speech?"*

Record:
- Preferred style: A or B
- Confidence: 1-5 scale (1=guess, 5=very confident)

## Emotion Contexts (2 comparisons each)

| Emotion | Example Utterance |
|---------|-------------------|
| Anger | "I can't believe you did that!" |
| Happiness | "This is the best day ever!" |
| Sadness | "I really miss those times..." |
| Neutral | "The meeting is at three o'clock." |
| Fear | "Did you hear that noise?" |
| Surprise | "Wait, you're getting married?!" |

## Data Format

Record each comparison in JSON format:

```json
{
  "user_id": "real_001",
  "profile": {
    "age_group": "young",
    "language_region": "western",
    "accessibility_needs": false,
    "device_type": "mobile"
  },
  "comparisons": [
    {
      "emotion": "anger",
      "style_a": [1.3, 0.9, 0.8, 0.6, 1.2],
      "style_b": [1.0, 0.5, 0.4, 0.2, 1.0],
      "preferred": "A",
      "confidence": 0.85
    }
  ]
}
```

## Style Vector Format

Each style is a 5-element vector:
```
[font_size, color_intensity, emphasis_strength, animation_level, contrast_ratio]
```

| Parameter | Range | Low Example | High Example |
|-----------|-------|-------------|--------------|
| font_size | 0.8-1.5 | 0.9 (small) | 1.4 (large) |
| color_intensity | 0-1 | 0.3 (muted) | 0.9 (vivid) |
| emphasis_strength | 0-1 | 0.2 (subtle) | 0.8 (strong) |
| animation_level | 0-1 | 0.1 (static) | 0.7 (animated) |
| contrast_ratio | 0.5-2 | 0.8 (low) | 1.6 (high) |

## Pre-defined Style Pairs

Use these pre-generated pairs for consistency:

### Pair Set 1 (Anger context)
- Style A: [1.30, 0.90, 0.80, 0.60, 1.20] - Expressive
- Style B: [1.00, 0.50, 0.40, 0.20, 1.00] - Subtle

### Pair Set 2 (Happiness context)
- Style A: [1.25, 0.85, 0.75, 0.70, 1.15] - Warm/animated
- Style B: [0.95, 0.45, 0.35, 0.15, 0.95] - Neutral

### Pair Set 3 (Sadness context)
- Style A: [0.85, 0.30, 0.25, 0.10, 0.80] - Subdued
- Style B: [1.10, 0.65, 0.55, 0.45, 1.15] - Normal

### Pair Set 4 (Neutral context)
- Style A: [1.00, 0.50, 0.45, 0.30, 1.00] - Standard
- Style B: [1.15, 0.70, 0.60, 0.50, 1.15] - Slightly enhanced

### Pair Set 5 (Fear context)
- Style A: [1.15, 0.75, 0.65, 0.50, 1.20] - Tense
- Style B: [0.90, 0.40, 0.35, 0.20, 0.95] - Calm

### Pair Set 6 (Surprise context)
- Style A: [1.30, 0.85, 0.75, 0.65, 1.25] - Excited
- Style B: [1.00, 0.50, 0.40, 0.30, 1.00] - Normal

## Quick Collection Method (Google Forms)

Create a Google Form with:

1. **Section 1: Demographics** (4 questions)
2. **Section 2-7: Emotion Comparisons** (2 per emotion = 12 total)

For each comparison question:
- Show two subtitle style images/videos
- Ask: "Which style better matches the emotion?"
- Options: Style A / Style B
- Follow-up: "How confident? (1-5)"

## Video Demo Clips

Use clips from the Sentimentogram demo or create new ones:

1. Extract 5-10 second clips with clear emotional content
2. Render same clip with Style A and Style B subtitles
3. Present side-by-side or sequential A/B comparison

## Consent Statement

*"This anonymous survey collects your preferences for subtitle styles. No personally identifiable information is stored. Data will be used for academic research on accessible video subtitling. By proceeding, you consent to participate."*

## After Collection

1. Export data to JSON format matching `preference_data_real.json` schema
2. Validate data completeness (all 12 comparisons per user)
3. Merge with synthetic data for training
4. Update paper with actual participant count

## Contact

For questions about data collection methodology, contact the research team.

---

**Dataset Statistics Target:**
- Synthetic users: 20 (240 comparisons)
- Real users: 5-10 (60-120 comparisons)
- Total: 300-360 comparisons
- Train/Test split: 80/20
