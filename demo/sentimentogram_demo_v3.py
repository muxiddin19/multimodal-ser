"""
Sentimentogram Demo V3: Multi-Level Emotion Typography with Enhanced Model
===========================================================================

Features:
1. Emotion-specific FONTS (not just colors)
2. Multi-level sizing: sentence → word → character
3. Typography effects: letter-spacing, transforms, animations
4. Character-level emphasis for high-confidence emotions

Enhanced Model Integration (ACL 2026):
- VAD-Guided Cross-Attention (λ=0.5)
- Constrained Adaptive Fusion (gates sum to 1)
- Hard Negative Mining MICL
- Focal Loss for class imbalance

Performance:
- IEMOCAP 5-class: 77.41% UA (Val), 75.61% UA (Test)
- CREMA-D 4-class: 92.90% UA (SOTA)

Usage:
    python demo/sentimentogram_demo_v3.py --video input.mp4 --output output.html

    # With specific model:
    python demo/sentimentogram_demo_v3.py --video input.mp4 \\
        --model saved_models/enhanced_iemocap_5_5class_20251224_081430.pt
"""

import os
import sys
import argparse
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import subprocess

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# CONFIGURATION - EMOTION-SPECIFIC TYPOGRAPHY
# ============================================================

@dataclass
class EmotionTypography:
    """Complete typography configuration for each emotion."""
    # Colors
    color: str
    background: str
    text_shadow: str

    # Font
    font_family: str
    font_weight: str
    font_style: str  # normal, italic

    # Sizing (multipliers)
    sentence_scale: float  # Base scale for whole sentence
    word_scale: float      # Additional scale for individual words
    char_scale_range: Tuple[float, float]  # Min/max for character variation

    # Effects
    letter_spacing: str
    text_transform: str  # none, uppercase, lowercase
    animation: str       # CSS animation name or none


# Emotion-specific font families (Google Fonts)
GOOGLE_FONTS = [
    "Poppins",           # Clean, modern (neutral)
    "Fredoka One",       # Rounded, playful (happy)
    "Merriweather",      # Elegant serif (sadness)
    "Bebas Neue",        # Bold, impactful (anger)
    "Oswald",            # Condensed, tense (frustration)
]

# Western Culture Typography
TYPOGRAPHY_WESTERN = {
    "happy_excited": EmotionTypography(
        color="#FFD700",
        background="rgba(255,215,0,0.12)",
        text_shadow="0 0 15px rgba(255,215,0,0.6), 0 0 30px rgba(255,215,0,0.3)",
        font_family="'Fredoka One', cursive",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.15,
        word_scale=1.1,
        char_scale_range=(0.95, 1.15),
        letter_spacing="0.05em",
        text_transform="none",
        animation="bounce"
    ),
    "sadness": EmotionTypography(
        color="#6B9BD2",
        background="rgba(107,155,210,0.08)",
        text_shadow="none",
        font_family="'Merriweather', serif",
        font_weight="300",
        font_style="italic",
        sentence_scale=0.92,
        word_scale=0.95,
        char_scale_range=(0.9, 1.0),
        letter_spacing="0.02em",
        text_transform="none",
        animation="fade"
    ),
    "neutral": EmotionTypography(
        color="#C0C0C0",
        background="transparent",
        text_shadow="none",
        font_family="'Poppins', sans-serif",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.0,
        word_scale=1.0,
        char_scale_range=(1.0, 1.0),
        letter_spacing="normal",
        text_transform="none",
        animation="none"
    ),
    "anger": EmotionTypography(
        color="#FF3333",
        background="rgba(255,51,51,0.15)",
        text_shadow="0 0 20px rgba(255,51,51,0.8), 2px 2px 4px rgba(0,0,0,0.5)",
        font_family="'Bebas Neue', sans-serif",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.3,
        word_scale=1.2,
        char_scale_range=(1.0, 1.3),
        letter_spacing="0.08em",
        text_transform="uppercase",
        animation="shake"
    ),
    "frustration": EmotionTypography(
        color="#FF8C00",
        background="rgba(255,140,0,0.1)",
        text_shadow="0 0 10px rgba(255,140,0,0.5)",
        font_family="'Oswald', sans-serif",
        font_weight="500",
        font_style="normal",
        sentence_scale=1.1,
        word_scale=1.05,
        char_scale_range=(0.95, 1.1),
        letter_spacing="0.03em",
        text_transform="none",
        animation="pulse"
    ),
}

# Eastern Culture Typography
TYPOGRAPHY_EASTERN = {
    "happy_excited": EmotionTypography(
        color="#FF2222",  # Red = lucky
        background="rgba(255,34,34,0.12)",
        text_shadow="0 0 15px rgba(255,34,34,0.6)",
        font_family="'Fredoka One', cursive",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.15,
        word_scale=1.1,
        char_scale_range=(0.95, 1.15),
        letter_spacing="0.05em",
        text_transform="none",
        animation="bounce"
    ),
    "sadness": EmotionTypography(
        color="#E0E0E0",  # White = mourning
        background="rgba(200,200,200,0.15)",
        text_shadow="1px 1px 2px rgba(0,0,0,0.3)",
        font_family="'Merriweather', serif",
        font_weight="300",
        font_style="italic",
        sentence_scale=0.92,
        word_scale=0.95,
        char_scale_range=(0.9, 1.0),
        letter_spacing="0.02em",
        text_transform="none",
        animation="fade"
    ),
    "neutral": EmotionTypography(
        color="#A0A0A0",
        background="transparent",
        text_shadow="none",
        font_family="'Poppins', sans-serif",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.0,
        word_scale=1.0,
        char_scale_range=(1.0, 1.0),
        letter_spacing="normal",
        text_transform="none",
        animation="none"
    ),
    "anger": EmotionTypography(
        color="#1A1A1A",  # Black
        background="rgba(0,0,0,0.2)",
        text_shadow="0 0 10px rgba(255,255,255,0.3)",
        font_family="'Bebas Neue', sans-serif",
        font_weight="400",
        font_style="normal",
        sentence_scale=1.3,
        word_scale=1.2,
        char_scale_range=(1.0, 1.3),
        letter_spacing="0.08em",
        text_transform="uppercase",
        animation="shake"
    ),
    "frustration": EmotionTypography(
        color="#8B4513",  # Brown
        background="rgba(139,69,19,0.12)",
        text_shadow="none",
        font_family="'Oswald', sans-serif",
        font_weight="500",
        font_style="normal",
        sentence_scale=1.1,
        word_scale=1.05,
        char_scale_range=(0.95, 1.1),
        letter_spacing="0.03em",
        text_transform="none",
        animation="pulse"
    ),
}

CULTURE_TYPOGRAPHY = {
    "western": TYPOGRAPHY_WESTERN,
    "eastern": TYPOGRAPHY_EASTERN,
}

# Age-based font size multipliers
AGE_FONT_SCALE = {
    "child": 1.3,
    "teen": 1.1,
    "adult": 1.0,
    "senior": 1.25,
}

# Legend colors
EMOTION_LEGEND_COLORS = {
    "happy_excited": "#FFD700",
    "sadness": "#6B9BD2",
    "neutral": "#C0C0C0",
    "anger": "#FF3333",
    "frustration": "#FF8C00",
}


# ============================================================
# AUDIO/VIDEO PROCESSING
# ============================================================

def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    """Extract audio from video file using ffmpeg."""
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        # Put audio file in same directory as output, not video
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        output_path = f"/tmp/{video_name}_audio.wav"

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    print(f"Extracted audio to: {output_path}")
    return output_path


# ============================================================
# SPEECH-TO-TEXT (Whisper)
# ============================================================

def transcribe_audio(audio_path: str, language: str = "en") -> List[Dict]:
    """Transcribe audio using Whisper with word-level timestamps."""
    try:
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper

    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "words": seg.get("words", [])
        })

    print(f"Transcribed {len(segments)} segments")
    return segments


# ============================================================
# SPEECH EMOTION RECOGNITION - WORD LEVEL
# ============================================================

class SERPredictorV3:
    """Enhanced Speech Emotion Recognition with word-level prediction.

    Now uses the EnhancedMultimodalSER model with:
    - VAD-Guided Cross-Attention (λ=0.5)
    - Constrained Adaptive Fusion
    - Hard Negative Mining MICL
    - Focal Loss support
    """

    def __init__(self, model_path: str, config_type: str = "iemocap_5"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_type = config_type

        # Try to load enhanced model first, fall back to novel model
        try:
            from models.enhanced_components import EnhancedMultimodalSER
            use_enhanced = True
        except ImportError:
            from models.novel_components import NovelMultimodalSER
            use_enhanced = False
            print("Warning: Enhanced model not available, using NovelMultimodalSER")

        model_params = {
            "text_dim": 768,
            "audio_dim": 1024,
            "hidden_dim": 384,
            "num_heads": 8,
            "num_layers": 2,
            "num_classes": 5,
            "dropout": 0.3,
            "vad_lambda": 0.5,  # Increased for enhanced model
            "micl_dim": 128
        }

        if use_enhanced:
            model_params["use_augmentation"] = False  # Disable augmentation for inference

        print(f"Loading {'Enhanced' if use_enhanced else 'Novel'} SER model from: {model_path}")

        if use_enhanced:
            self.model = EnhancedMultimodalSER(**model_params).to(self.device)
        else:
            model_params.pop("use_augmentation", None)
            model_params["vad_lambda"] = 0.1
            self.model = NovelMultimodalSER(**model_params).to(self.device)

        # Inject Config class for pickle compatibility
        import __main__
        from dataclasses import dataclass as dc, field as dataclass_field
        from typing import List

        @dc
        class EnhancedConfig:
            text_dim: int = 768
            audio_dim: int = 1024
            hidden_dim: int = 384
            num_heads: int = 8
            num_layers: int = 2
            num_classes: int = 5
            dropout: float = 0.3
            vad_lambda: float = 0.5
            micl_dim: int = 128
            vad_weight: float = 0.5
            micl_weight: float = 0.3
            micl_temp: float = 0.07
            batch_size: int = 16
            lr: float = 2e-5
            weight_decay: float = 0.01
            epochs: int = 100
            patience: int = 15
            warmup_ratio: float = 0.1
            num_runs: int = 5
            seed: int = 42
            cls_weight: float = 1.0
            focal_gamma: float = 2.0
            use_augmentation: bool = True
            mixup_alpha: float = 0.4
            augment_prob: float = 0.5
            use_curriculum: bool = False
            curriculum_phases: List[int] = dataclass_field(default_factory=lambda: [30, 60, 100])
            curriculum_classes: List[int] = dataclass_field(default_factory=lambda: [4, 5, 6])
            emotion_config: str = "iemocap_5"

        __main__.EnhancedConfig = EnhancedConfig
        # Also add as Config for backward compatibility
        __main__.Config = EnhancedConfig

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.labels = ["happy_excited", "sadness", "neutral", "anger", "frustration"]
        self._load_feature_extractors()
        print(f"SER model loaded ({type(self.model).__name__}). Classes: {self.labels}")

    def _load_feature_extractors(self):
        """Load BERT and emotion2vec for feature extraction."""
        from transformers import BertTokenizer, BertModel
        from funasr import AutoModel

        print("Loading BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert.eval()

        print("Loading emotion2vec...")
        self.emotion2vec = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    def extract_audio_features(self, audio_path: str, start: float, end: float) -> torch.Tensor:
        """Extract audio features for a time segment."""
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)

        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment = audio[:, start_sample:end_sample]

        MIN_SAMPLES = 8000
        if segment.shape[1] == 0:
            segment = torch.zeros(1, MIN_SAMPLES)
        elif segment.shape[1] < MIN_SAMPLES:
            padding = torch.zeros(1, MIN_SAMPLES - segment.shape[1])
            segment = torch.cat([segment, padding], dim=1)

        temp_path = "/tmp/segment_temp_v3.wav"
        torchaudio.save(temp_path, segment, 16000)

        try:
            result = self.emotion2vec.generate(temp_path, output_dir=None, granularity="utterance")
            if isinstance(result, list) and len(result) > 0:
                audio_feat = result[0].get('feats', None)
                if audio_feat is not None:
                    if isinstance(audio_feat, np.ndarray):
                        audio_feat = torch.from_numpy(audio_feat).float()
                else:
                    audio_feat = torch.zeros(1024)
            else:
                audio_feat = torch.zeros(1024)
        except Exception:
            audio_feat = torch.zeros(1024)

        return audio_feat

    def extract_text_features(self, text: str) -> torch.Tensor:
        """Extract text features using BERT."""
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=100)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            text_output = self.bert(**tokens)
            text_feat = text_output.last_hidden_state[:, 0, :].squeeze(0)
        return text_feat.cpu()

    def predict_segment(self, text: str, audio_path: str, start: float, end: float) -> Dict:
        """Predict emotion for a whole segment."""
        text_feat = self.extract_text_features(text)
        audio_feat = self.extract_audio_features(audio_path, start, end)

        with torch.no_grad():
            text_feat = text_feat.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device)

            outputs = self.model(text_feat, audio_feat, apply_augmentation=False)
            probs = outputs['probs'].cpu().numpy()[0]
            pred_idx = probs.argmax()

            vad = outputs.get('vad', None)
            if vad is not None:
                vad = vad.cpu().numpy()[0].tolist()

            # Capture fusion gate info from enhanced model
            fusion_gates = {}
            if 'text_gate' in outputs:
                fusion_gates['text'] = float(outputs['text_gate'].cpu().numpy()[0])
            if 'audio_gate' in outputs:
                fusion_gates['audio'] = float(outputs['audio_gate'].cpu().numpy()[0])
            if 'interaction_gate' in outputs:
                fusion_gates['interaction'] = float(outputs['interaction_gate'].cpu().numpy()[0])

        result = {
            "emotion": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "vad": vad
        }

        if fusion_gates:
            result["fusion_gates"] = fusion_gates

        return result

    def predict_words(self, words: List[Dict], audio_path: str) -> List[Dict]:
        """Predict emotion for each word or word group."""
        if not words:
            return []

        MIN_DURATION = 0.3
        word_results = []

        i = 0
        while i < len(words):
            group_words = [words[i]]
            group_start = words[i].get("start", 0)
            group_end = words[i].get("end", group_start + 0.1)
            group_text = words[i].get("word", "").strip()

            while (group_end - group_start) < MIN_DURATION and i + 1 < len(words):
                i += 1
                group_words.append(words[i])
                group_end = words[i].get("end", group_end)
                group_text += " " + words[i].get("word", "").strip()

            if group_text.strip():
                text_feat = self.extract_text_features(group_text)
                audio_feat = self.extract_audio_features(audio_path, group_start, group_end)

                with torch.no_grad():
                    text_feat_batch = text_feat.unsqueeze(0).to(self.device)
                    audio_feat_batch = audio_feat.unsqueeze(0).to(self.device)

                    outputs = self.model(text_feat_batch, audio_feat_batch, apply_augmentation=False)
                    probs = outputs['probs'].cpu().numpy()[0]
                    pred_idx = probs.argmax()

                emotion = self.labels[pred_idx]
                confidence = float(probs[pred_idx])
                all_probs = {self.labels[j]: float(probs[j]) for j in range(len(self.labels))}

                # Capture fusion gates for word-level analysis
                fusion_gates = {}
                if 'text_gate' in outputs:
                    fusion_gates['text'] = float(outputs['text_gate'].cpu().numpy()[0])
                if 'audio_gate' in outputs:
                    fusion_gates['audio'] = float(outputs['audio_gate'].cpu().numpy()[0])
            else:
                emotion = "neutral"
                confidence = 0.5
                all_probs = {}
                fusion_gates = {}

            for w in group_words:
                word_data = {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "emotion": emotion,
                    "confidence": confidence,
                    "all_probs": all_probs
                }
                if fusion_gates:
                    word_data["fusion_gates"] = fusion_gates
                word_results.append(word_data)

            i += 1

        return word_results


# ============================================================
# CHARACTER-LEVEL STYLING
# ============================================================

def generate_char_styled_word(word: str, emotion: str, confidence: float,
                               typography: EmotionTypography, age_scale: float) -> str:
    """Generate HTML for a word with character-level size variations."""
    if not word.strip():
        return word

    chars = list(word)
    char_min, char_max = typography.char_scale_range

    # Only apply character variation for high confidence emotions
    if confidence > 0.6 and emotion != "neutral":
        # Create wave-like size variation
        styled_chars = []
        for i, char in enumerate(chars):
            if char.isalpha():
                # Sine wave variation based on position
                variation = (char_max - char_min) * (0.5 + 0.5 * np.sin(i * 0.8))
                char_size = (char_min + variation) * typography.word_scale * age_scale
                styled_chars.append(
                    f'<span style="font-size:{char_size:.2f}em">{char}</span>'
                )
            else:
                styled_chars.append(char)
        return ''.join(styled_chars)
    else:
        # No character variation
        return word


def generate_word_html(word_data: Dict, typography: EmotionTypography,
                       age_scale: float, use_char_level: bool = True) -> str:
    """Generate styled HTML for a single word."""
    word = word_data["word"]
    emotion = word_data["emotion"]
    confidence = word_data["confidence"]

    # Base word scale
    word_scale = typography.word_scale * age_scale

    # Transform text if needed
    display_word = word
    if typography.text_transform == "uppercase":
        display_word = word.upper()
    elif typography.text_transform == "lowercase":
        display_word = word.lower()

    # Character-level styling for high-confidence non-neutral emotions
    if use_char_level and confidence > 0.65 and emotion != "neutral":
        inner_html = generate_char_styled_word(
            display_word, emotion, confidence, typography, age_scale
        )
    else:
        inner_html = display_word

    # Animation class
    anim_class = f"anim-{typography.animation}" if typography.animation != "none" else ""

    # Build style
    style = (
        f"color:{typography.color};"
        f"font-family:{typography.font_family};"
        f"font-weight:{typography.font_weight};"
        f"font-style:{typography.font_style};"
        f"font-size:{word_scale}em;"
        f"letter-spacing:{typography.letter_spacing};"
        f"text-shadow:{typography.text_shadow};"
        f"background:{typography.background};"
        f"padding:2px 4px;"
        f"border-radius:3px;"
        f"display:inline-block;"
        f"margin:1px;"
    )

    return f'<span class="emotion-word {anim_class}" style="{style}">{inner_html}</span>'


# ============================================================
# HTML GENERATION - V3 WITH TYPOGRAPHY
# ============================================================

def generate_html_output_v3(
    segments: List[Dict],
    video_path: str,
    output_path: str,
    culture: str = "western",
    age_group: str = "adult"
) -> str:
    """Generate HTML visualization with emotion-specific typography."""

    typography = CULTURE_TYPOGRAPHY[culture]
    age_scale = AGE_FONT_SCALE[age_group]

    # Build subtitle data
    subtitles_js = []
    for seg in segments:
        words_data = seg.get("word_emotions", [])
        subtitles_js.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": words_data,
            "dominant_emotion": seg.get("emotion", "neutral"),
            "confidence": seg.get("confidence", 0.5)
        })

    # Build transcript entries with word-level typography
    transcript_entries = []
    for idx, seg in enumerate(segments):
        time_str = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
        words_data = seg.get("word_emotions", [])
        dominant_emotion = seg.get("emotion", "neutral")
        confidence = seg.get("confidence", 0.5)

        if words_data:
            words_html = []
            for w in words_data:
                typo = typography.get(w["emotion"], typography["neutral"])
                words_html.append(generate_word_html(w, typo, age_scale, use_char_level=True))
            text_html = "".join(words_html)
        else:
            typo = typography.get(dominant_emotion, typography["neutral"])
            text_html = f'<span style="color:{typo.color};font-family:{typo.font_family};font-size:{typo.sentence_scale * age_scale}em">{seg["text"]}</span>'

        transcript_entries.append(f'''
            <div class="transcript-entry" id="entry-{idx}" data-start="{seg['start']}" data-end="{seg['end']}">
                <span class="time">{time_str}</span>
                <span class="text">{text_html}</span>
                <span class="emotion-tag" style="background:{EMOTION_LEGEND_COLORS.get(dominant_emotion, '#888')}">{dominant_emotion} ({confidence:.0%})</span>
            </div>
        ''')

    transcript_html = "\n".join(transcript_entries)

    # Typography data for JavaScript
    typography_js = {}
    for culture_name, culture_typo in CULTURE_TYPOGRAPHY.items():
        typography_js[culture_name] = {}
        for emotion, typo in culture_typo.items():
            typography_js[culture_name][emotion] = {
                "color": typo.color,
                "background": typo.background,
                "textShadow": typo.text_shadow,
                "fontFamily": typo.font_family,
                "fontWeight": typo.font_weight,
                "fontStyle": typo.font_style,
                "sentenceScale": typo.sentence_scale,
                "wordScale": typo.word_scale,
                "charScaleRange": typo.char_scale_range,
                "letterSpacing": typo.letter_spacing,
                "textTransform": typo.text_transform,
                "animation": typo.animation
            }

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentimentogram V3 - Emotion Typography</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Fredoka+One&family=Merriweather:ital,wght@0,300;1,300&family=Oswald:wght@500&family=Poppins:wght@400;600&display=swap" rel="stylesheet">

    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        h1 {{
            text-align: center;
            margin-bottom: 8px;
            color: #fff;
            font-size: 2.2em;
            font-weight: 600;
            letter-spacing: 2px;
        }}

        .subtitle-info {{
            text-align: center;
            color: #888;
            margin-bottom: 25px;
            font-size: 0.95em;
        }}

        .video-container {{
            position: relative;
            width: 100%;
            max-width: 950px;
            margin: 0 auto 35px;
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 15px 50px rgba(0,0,0,0.6);
        }}

        video {{
            width: 100%;
            display: block;
        }}

        .subtitle-overlay {{
            position: absolute;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            padding: 15px 30px;
            border-radius: 10px;
            max-width: 88%;
            background: rgba(0,0,0,0.75);
            backdrop-filter: blur(8px);
            transition: all 0.2s ease;
            font-size: 1.5em;
            line-height: 1.6;
        }}

        .controls {{
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}

        .control-group {{
            background: rgba(22, 33, 62, 0.9);
            padding: 18px 25px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .control-group label {{
            display: block;
            margin-bottom: 10px;
            color: #999;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}

        select {{
            padding: 12px 22px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #0f3460;
            color: #fff;
            cursor: pointer;
            font-size: 1em;
            font-family: 'Poppins', sans-serif;
        }}

        .emotion-legend {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 35px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 18px;
            background: rgba(22, 33, 62, 0.8);
            border-radius: 25px;
            font-size: 0.9em;
            border: 1px solid rgba(255,255,255,0.05);
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            box-shadow: 0 0 12px currentColor;
        }}

        .legend-font {{
            font-size: 0.75em;
            color: #666;
            margin-left: 5px;
        }}

        .transcript {{
            background: rgba(22, 33, 62, 0.85);
            border-radius: 15px;
            padding: 25px;
            max-height: 500px;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.05);
        }}

        .transcript h2 {{
            margin-bottom: 20px;
            color: #aaa;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 3px;
            font-weight: 400;
        }}

        .transcript-entry {{
            display: flex;
            align-items: flex-start;
            gap: 18px;
            padding: 14px 18px;
            border-radius: 10px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid transparent;
        }}

        .transcript-entry:hover {{
            background: rgba(255,255,255,0.05);
            border-color: rgba(255,255,255,0.1);
        }}

        .transcript-entry.active {{
            background: rgba(255,255,255,0.08);
            border-left: 4px solid #FFD700;
        }}

        .transcript-entry .time {{
            color: #555;
            font-size: 0.85em;
            min-width: 50px;
            font-family: monospace;
        }}

        .transcript-entry .text {{
            flex: 1;
            line-height: 1.8;
        }}

        .transcript-entry .emotion-tag {{
            font-size: 0.65em;
            padding: 5px 12px;
            border-radius: 15px;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
            font-weight: 600;
        }}

        .emotion-word {{
            transition: all 0.2s ease;
        }}

        /* Animations */
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-3px); }}
        }}

        @keyframes shake {{
            0%, 100% {{ transform: translateX(0); }}
            25% {{ transform: translateX(-2px); }}
            75% {{ transform: translateX(2px); }}
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.85; }}
        }}

        @keyframes fade {{
            0%, 100% {{ opacity: 0.9; }}
            50% {{ opacity: 0.7; }}
        }}

        .anim-bounce {{ animation: bounce 0.6s ease-in-out infinite; }}
        .anim-shake {{ animation: shake 0.3s ease-in-out infinite; }}
        .anim-pulse {{ animation: pulse 1.5s ease-in-out infinite; }}
        .anim-fade {{ animation: fade 2s ease-in-out infinite; }}

        ::-webkit-scrollbar {{ width: 10px; }}
        ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); border-radius: 5px; }}
        ::-webkit-scrollbar-thumb {{ background: #444; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SENTIMENTOGRAM V3</h1>
        <p class="subtitle-info">Multi-Level Emotion Typography: Sentence + Word + Character Styling</p>

        <div class="controls">
            <div class="control-group">
                <label>Culture Setting</label>
                <select id="cultureSelect" onchange="updateStyles()">
                    <option value="western" {"selected" if culture == "western" else ""}>Western</option>
                    <option value="eastern" {"selected" if culture == "eastern" else ""}>Eastern</option>
                </select>
            </div>
            <div class="control-group">
                <label>Age Group</label>
                <select id="ageSelect" onchange="updateStyles()">
                    <option value="child">Child (1.3x)</option>
                    <option value="teen">Teen (1.1x)</option>
                    <option value="adult" {"selected" if age_group == "adult" else ""}>Adult (1.0x)</option>
                    <option value="senior">Senior (1.25x)</option>
                </select>
            </div>
        </div>

        <div class="emotion-legend">
            <div class="legend-item">
                <div class="legend-color" style="background:#FFD700;color:#FFD700"></div>
                <span style="font-family:'Fredoka One',cursive">Happy</span>
                <span class="legend-font">(Fredoka)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#6B9BD2;color:#6B9BD2"></div>
                <span style="font-family:'Merriweather',serif;font-style:italic">Sadness</span>
                <span class="legend-font">(Merriweather)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#C0C0C0;color:#C0C0C0"></div>
                <span style="font-family:'Poppins',sans-serif">Neutral</span>
                <span class="legend-font">(Poppins)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#FF3333;color:#FF3333"></div>
                <span style="font-family:'Bebas Neue',sans-serif;text-transform:uppercase">ANGER</span>
                <span class="legend-font">(Bebas Neue)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#FF8C00;color:#FF8C00"></div>
                <span style="font-family:'Oswald',sans-serif">Frustration</span>
                <span class="legend-font">(Oswald)</span>
            </div>
        </div>

        <div class="video-container">
            <video id="video" controls>
                <source src="{os.path.basename(video_path)}" type="video/mp4">
            </video>
            <div class="subtitle-overlay" id="subtitle"></div>
        </div>

        <div class="transcript">
            <h2>Transcript with Emotion Typography</h2>
            {transcript_html}
        </div>
    </div>

    <script>
        const subtitles = {json.dumps(subtitles_js)};
        const typography = {json.dumps(typography_js)};
        const ageScales = {json.dumps(AGE_FONT_SCALE)};

        const video = document.getElementById('video');
        const subtitleEl = document.getElementById('subtitle');

        function generateStyledWord(word, emotion, confidence, typo, ageScale) {{
            let displayWord = word;
            if (typo.textTransform === 'uppercase') displayWord = word.toUpperCase();
            else if (typo.textTransform === 'lowercase') displayWord = word.toLowerCase();

            const wordScale = typo.wordScale * ageScale;
            const animClass = typo.animation !== 'none' ? `anim-${{typo.animation}}` : '';

            // Character-level variation for high confidence
            let innerHtml = displayWord;
            if (confidence > 0.65 && emotion !== 'neutral') {{
                const chars = displayWord.split('');
                const [charMin, charMax] = typo.charScaleRange;
                innerHtml = chars.map((char, i) => {{
                    if (/[a-zA-Z]/.test(char)) {{
                        const variation = (charMax - charMin) * (0.5 + 0.5 * Math.sin(i * 0.8));
                        const charSize = (charMin + variation) * wordScale;
                        return `<span style="font-size:${{charSize.toFixed(2)}}em">${{char}}</span>`;
                    }}
                    return char;
                }}).join('');
            }}

            return `<span class="emotion-word ${{animClass}}" style="
                color:${{typo.color}};
                font-family:${{typo.fontFamily}};
                font-weight:${{typo.fontWeight}};
                font-style:${{typo.fontStyle}};
                font-size:${{wordScale}}em;
                letter-spacing:${{typo.letterSpacing}};
                text-shadow:${{typo.textShadow}};
                background:${{typo.background}};
                padding:2px 4px;
                border-radius:3px;
                display:inline-block;
                margin:1px;
            ">${{innerHtml}}</span>`;
        }}

        video.addEventListener('timeupdate', function() {{
            const currentTime = video.currentTime;
            let activeSubtitle = null;

            subtitles.forEach((sub, idx) => {{
                const entry = document.getElementById(`entry-${{idx}}`);
                if (currentTime >= sub.start && currentTime <= sub.end) {{
                    activeSubtitle = sub;
                    entry.classList.add('active');
                    entry.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    entry.classList.remove('active');
                }}
            }});

            if (activeSubtitle) {{
                const culture = document.getElementById('cultureSelect').value;
                const age = document.getElementById('ageSelect').value;
                const scale = ageScales[age];
                const typos = typography[culture];

                if (activeSubtitle.words && activeSubtitle.words.length > 0) {{
                    let html = '';
                    activeSubtitle.words.forEach(w => {{
                        const typo = typos[w.emotion] || typos['neutral'];
                        html += generateStyledWord(w.word, w.emotion, w.confidence, typo, scale);
                    }});
                    subtitleEl.innerHTML = html;
                }} else {{
                    const typo = typos[activeSubtitle.dominant_emotion] || typos['neutral'];
                    const size = typo.sentenceScale * scale;
                    subtitleEl.innerHTML = `<span style="
                        color:${{typo.color}};
                        font-family:${{typo.fontFamily}};
                        font-size:${{size}}em
                    ">${{activeSubtitle.text}}</span>`;
                }}
                subtitleEl.style.display = 'block';
            }} else {{
                subtitleEl.style.display = 'none';
            }}
        }});

        document.querySelectorAll('.transcript-entry').forEach(entry => {{
            entry.addEventListener('click', () => {{
                const start = parseFloat(entry.dataset.start);
                video.currentTime = start;
                video.play();
            }});
        }});

        function updateStyles() {{
            location.reload();  // Simplest way to re-render with new styles
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated visualization: {output_path}")
    return output_path


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    video_path: str,
    output_path: str,
    model_path: str = None,
    language: str = "en",
    culture: str = "western",
    age_group: str = "adult",
    word_level: bool = True
):
    """Run the V3 Sentimentogram pipeline with emotion typography."""

    print("=" * 60)
    print("SENTIMENTOGRAM V3 - Emotion Typography System")
    print("=" * 60)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract audio
    print("\n[1/4] Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    # Step 2: Transcribe
    print("\n[2/4] Transcribing speech with word timestamps...")
    segments = transcribe_audio(audio_path, language)

    # Step 3: Emotion recognition
    print("\n[3/4] Detecting emotions (sentence + word level)...")
    if model_path and Path(model_path).exists():
        predictor = SERPredictorV3(model_path)

        for i, seg in enumerate(segments):
            print(f"  Segment {i+1}/{len(segments)}: {seg['text'][:40]}...")

            result = predictor.predict_segment(seg["text"], audio_path, seg["start"], seg["end"])
            seg.update(result)

            if word_level and seg.get("words"):
                word_emotions = predictor.predict_words(seg["words"], audio_path)
                seg["word_emotions"] = word_emotions
            else:
                seg["word_emotions"] = []
    else:
        print("  Warning: No SER model. Using placeholder emotions.")
        for seg in segments:
            seg["emotion"] = "neutral"
            seg["confidence"] = 0.5
            seg["word_emotions"] = []

    # Step 4: Generate visualization
    print("\n[4/4] Generating typography visualization...")

    import shutil
    video_output = output_dir / Path(video_path).name
    if not video_output.exists():
        shutil.copy(video_path, video_output)
        print(f"  Copied video to: {video_output}")

    html_path = generate_html_output_v3(
        segments,
        str(video_output),
        output_path,
        culture=culture,
        age_group=age_group
    )

    json_path = output_path.replace('.html', '_data.json')
    with open(json_path, 'w') as f:
        json.dump(segments, f, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Visualization: {html_path}")
    print(f"  Data: {json_path}")
    print(f"  Video: {video_output}")
    print("=" * 60)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Sentimentogram V3 - Emotion Typography")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="demo/output/result_v3.html", help="Output HTML path")
    parser.add_argument("--model", type=str, default="saved_models/enhanced_iemocap_5_5class_20251224_081430.pt",
                        help="SER model path (enhanced model recommended)")
    parser.add_argument("--language", type=str, default="en", help="Language for transcription")
    parser.add_argument("--culture", type=str, default="western", choices=["western", "eastern"])
    parser.add_argument("--age", type=str, default="adult", choices=["child", "teen", "adult", "senior"])
    parser.add_argument("--no-word-level", action="store_true", help="Disable word-level emotion")

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        language=args.language,
        culture=args.culture,
        age_group=args.age,
        word_level=not args.no_word_level
    )


if __name__ == "__main__":
    main()
