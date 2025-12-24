# Sentimentogram Demo

Emotion-Aware Subtitle Visualization System

## Overview

This demo implements the full Sentimentogram pipeline:
1. **Speech-to-Text**: Whisper for transcript extraction
2. **Emotion Detection**: Our trained SER model (5-class)
3. **Emotion Visualization**: VAD-based color/font/size mapping
4. **Personalization**: Culture (Western/Eastern) and Age-based adaptations

## Installation

```bash
# Required packages
pip install openai-whisper torch torchaudio transformers funasr

# For video processing
sudo apt install ffmpeg
```

## Usage

```bash
# Basic usage
python demo/sentimentogram_demo.py --video demo/videos/sample.mp4 --output demo/output/result.html

# With personalization
python demo/sentimentogram_demo.py \
    --video demo/videos/sample.mp4 \
    --output demo/output/result.html \
    --model saved_models/novel_iemocap_5_5class.pt \
    --culture western \
    --age adult \
    --language en
```

## Suggested Famous Video Clips (< 1 minute)

### Emotional Movie Scenes

1. **Anger/Frustration**
   - "You can't handle the truth!" - A Few Good Men
   - "I'm mad as hell" - Network (1976)

2. **Happiness/Joy**
   - "I'm the king of the world!" - Titanic
   - Singing in the Rain - title song scene

3. **Sadness**
   - "I could have saved more" - Schindler's List
   - "My precious" - Lord of the Rings (Gollum)

4. **Mixed Emotions**
   - "Here's looking at you, kid" - Casablanca
   - "Life is like a box of chocolates" - Forrest Gump

### YouTube Sources (for fair use/education)

Search for these clips on YouTube (usually available as fair use):
- Movie scene compilations with clear emotions
- TED talk segments (clear speech, varied emotions)
- Interview clips with emotional moments

## Emotion-Color Mapping

### Western Culture (Default)
| Emotion | Color | Font Size | Style |
|---------|-------|-----------|-------|
| Joy/Happy | Gold (#FFD700) | 1.2em | Bold |
| Anger | Red (#FF4444) | 1.3em | Bold |
| Sadness | Blue (#4169E1) | 0.95em | Normal |
| Fear | Purple (#800080) | 1.1em | Normal |
| Surprise | Orange (#FFA500) | 1.15em | Bold |
| Neutral | White (#FFFFFF) | 1.0em | Normal |
| Frustration | Crimson (#DC143C) | 1.1em | Bold |

### Eastern Culture
| Emotion | Color | Notes |
|---------|-------|-------|
| Joy | Red | Lucky color in Asian cultures |
| Sadness | White | Mourning color |
| Anger | Black | - |

### Age-Based Font Scaling
| Age Group | Scale |
|-----------|-------|
| Child | 1.3x |
| Teen | 1.1x |
| Adult | 1.0x |
| Senior | 1.25x |

## Output Files

The pipeline generates:
1. `result.html` - Interactive visualization with video player
2. `result_data.json` - Segment data with emotions and timestamps

## Architecture

```
Input Video
    │
    ├── FFmpeg ──────────────> Audio (.wav)
    │                              │
    │                              v
    │                         Whisper STT
    │                              │
    │                              v
    │                    Transcribed Segments
    │                     (text, timestamps)
    │                              │
    └── Video Copy                 v
           │              SER Model (BERT + emotion2vec)
           │                       │
           │                       v
           │               Emotion Labels + VAD
           │                       │
           v                       v
    ┌─────────────────────────────────────┐
    │        HTML Visualization            │
    │  ┌─────────────────────────────────┐ │
    │  │ Video Player with Subtitles    │ │
    │  │ (color/font based on emotion)  │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ Transcript with Emotion Tags   │ │
    │  └─────────────────────────────────┘ │
    │  ┌─────────────────────────────────┐ │
    │  │ Personalization Controls       │ │
    │  │ (Culture, Age Group)           │ │
    │  └─────────────────────────────────┘ │
    └─────────────────────────────────────┘
```

## Demo for Human Evaluation

For user studies:
1. Prepare 5-10 video clips with varied emotions
2. Generate visualizations for each
3. Compare with standard (non-emotional) subtitles
4. Collect ratings on: Clarity, Emotion Reflection, Engagement, etc.
