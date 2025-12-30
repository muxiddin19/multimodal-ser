"""
Sentimentogram Demo V2: Enhanced Word-Level Emotion Visualization
==================================================================

Improvements over V1:
1. Word-level emotion detection (not just sentence-level)
2. Distinct colors for ALL 5 emotions
3. Variable font sizes and weights per word
4. Smoother color transitions

Usage:
    python demo/sentimentogram_demo_v2.py --video input.mp4 --output output.html
"""

import os
import sys
import argparse
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import subprocess

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# CONFIGURATION - ENHANCED COLOR SCHEME
# ============================================================

@dataclass
class EmotionStyle:
    """Style configuration for each emotion."""
    color: str           # Hex color
    font_size: float     # Relative size (1.0 = base)
    font_weight: str     # normal, bold
    background: str      # Background color (optional)
    text_shadow: str     # Text shadow for emphasis

# Enhanced color palette - distinct colors for all 5 emotions
EMOTION_STYLES_WESTERN = {
    "happy_excited": EmotionStyle(
        color="#FFD700",           # Bright Gold
        font_size=1.25,
        font_weight="bold",
        background="rgba(255,215,0,0.15)",
        text_shadow="0 0 10px rgba(255,215,0,0.5)"
    ),
    "sadness": EmotionStyle(
        color="#4A90D9",           # Soft Blue
        font_size=0.9,
        font_weight="normal",
        background="rgba(74,144,217,0.1)",
        text_shadow="none"
    ),
    "neutral": EmotionStyle(
        color="#B8B8B8",           # Light Gray
        font_size=1.0,
        font_weight="normal",
        background="transparent",
        text_shadow="none"
    ),
    "anger": EmotionStyle(
        color="#FF4444",           # Bright Red
        font_size=1.35,
        font_weight="bold",
        background="rgba(255,68,68,0.15)",
        text_shadow="0 0 12px rgba(255,68,68,0.6)"
    ),
    "frustration": EmotionStyle(
        color="#FF8C00",           # Dark Orange
        font_size=1.15,
        font_weight="bold",
        background="rgba(255,140,0,0.12)",
        text_shadow="0 0 8px rgba(255,140,0,0.4)"
    ),
}

EMOTION_STYLES_EASTERN = {
    "happy_excited": EmotionStyle(
        color="#FF3333",           # Red (lucky in Eastern cultures)
        font_size=1.25,
        font_weight="bold",
        background="rgba(255,51,51,0.15)",
        text_shadow="0 0 10px rgba(255,51,51,0.5)"
    ),
    "sadness": EmotionStyle(
        color="#E8E8E8",           # White-ish (mourning)
        font_size=0.9,
        font_weight="normal",
        background="rgba(200,200,200,0.2)",
        text_shadow="none"
    ),
    "neutral": EmotionStyle(
        color="#A0A0A0",           # Gray
        font_size=1.0,
        font_weight="normal",
        background="transparent",
        text_shadow="none"
    ),
    "anger": EmotionStyle(
        color="#1A1A1A",           # Black
        font_size=1.35,
        font_weight="bold",
        background="rgba(0,0,0,0.3)",
        text_shadow="0 0 5px rgba(255,255,255,0.3)"
    ),
    "frustration": EmotionStyle(
        color="#8B4513",           # Brown
        font_size=1.15,
        font_weight="bold",
        background="rgba(139,69,19,0.15)",
        text_shadow="none"
    ),
}

CULTURE_STYLES = {
    "western": EMOTION_STYLES_WESTERN,
    "eastern": EMOTION_STYLES_EASTERN,
}

# Age-based font size multipliers
AGE_FONT_SCALE = {
    "child": 1.3,
    "teen": 1.1,
    "adult": 1.0,
    "senior": 1.25,
}

# Emotion label colors for legend (consistent across cultures)
EMOTION_LEGEND_COLORS = {
    "happy_excited": "#FFD700",
    "sadness": "#4A90D9",
    "neutral": "#B8B8B8",
    "anger": "#FF4444",
    "frustration": "#FF8C00",
}


# ============================================================
# AUDIO/VIDEO PROCESSING
# ============================================================

def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    """Extract audio from video file using ffmpeg."""
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '_audio.wav'

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        output_path
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    print(f"Extracted audio to: {output_path}")
    return output_path


# ============================================================
# SPEECH-TO-TEXT (Whisper)
# ============================================================

def transcribe_audio(audio_path: str, language: str = "en") -> List[Dict]:
    """
    Transcribe audio using Whisper with word-level timestamps.
    Returns list of segments with text, start, end times, and words.
    """
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

class SERPredictorV2:
    """Enhanced Speech Emotion Recognition with word-level prediction."""

    def __init__(self, model_path: str, config_type: str = "iemocap_5"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_type = config_type

        # Import model class
        from models.novel_components import NovelMultimodalSER

        # Model parameters for 5-class
        model_params = {
            "text_dim": 768,
            "audio_dim": 1024,
            "hidden_dim": 384,
            "num_heads": 8,
            "num_layers": 2,
            "num_classes": 5,
            "dropout": 0.3,
            "vad_lambda": 0.1,
            "micl_dim": 128
        }

        # Load model
        print(f"Loading SER model from: {model_path}")
        self.model = NovelMultimodalSER(**model_params).to(self.device)

        # Load checkpoint - add Config class to __main__ namespace
        import __main__

        @dataclass
        class Config:
            text_dim: int = 768
            audio_dim: int = 1024
            hidden_dim: int = 384
            num_heads: int = 8
            num_layers: int = 2
            num_classes: int = 5
            dropout: float = 0.3
            vad_lambda: float = 0.1
            micl_dim: int = 128
            vad_weight: float = 0.3
            micl_weight: float = 0.2
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
            emotion_config: str = "iemocap_5"

        __main__.Config = Config

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Emotion labels
        self.labels = ["happy_excited", "sadness", "neutral", "anger", "frustration"]

        # Load feature extractors
        self._load_feature_extractors()

        print(f"SER model loaded. Classes: {self.labels}")

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

        # Extract segment
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment = audio[:, start_sample:end_sample]

        # Minimum segment length
        MIN_SAMPLES = 8000  # 0.5 seconds
        if segment.shape[1] == 0:
            segment = torch.zeros(1, MIN_SAMPLES)
        elif segment.shape[1] < MIN_SAMPLES:
            padding = torch.zeros(1, MIN_SAMPLES - segment.shape[1])
            segment = torch.cat([segment, padding], dim=1)

        # Save temporary segment
        temp_path = "/tmp/segment_temp_v2.wav"
        torchaudio.save(temp_path, segment, 16000)

        # Extract emotion2vec features
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
        except Exception as e:
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

            outputs = self.model(text_feat, audio_feat)
            probs = outputs['probs'].cpu().numpy()[0]
            pred_idx = probs.argmax()

            vad = outputs.get('vad', None)
            if vad is not None:
                vad = vad.cpu().numpy()[0].tolist()

        return {
            "emotion": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "vad": vad
        }

    def predict_words(self, words: List[Dict], audio_path: str) -> List[Dict]:
        """
        Predict emotion for each word or word group.
        Groups adjacent words if they're too short for reliable audio analysis.
        """
        if not words:
            return []

        # Group words into chunks (minimum 0.3 seconds for audio analysis)
        MIN_DURATION = 0.3
        word_results = []

        i = 0
        while i < len(words):
            # Start a new group
            group_words = [words[i]]
            group_start = words[i].get("start", 0)
            group_end = words[i].get("end", group_start + 0.1)
            group_text = words[i].get("word", "").strip()

            # Extend group if too short
            while (group_end - group_start) < MIN_DURATION and i + 1 < len(words):
                i += 1
                group_words.append(words[i])
                group_end = words[i].get("end", group_end)
                group_text += " " + words[i].get("word", "").strip()

            # Get prediction for this group
            if group_text.strip():
                text_feat = self.extract_text_features(group_text)
                audio_feat = self.extract_audio_features(audio_path, group_start, group_end)

                with torch.no_grad():
                    text_feat_batch = text_feat.unsqueeze(0).to(self.device)
                    audio_feat_batch = audio_feat.unsqueeze(0).to(self.device)

                    outputs = self.model(text_feat_batch, audio_feat_batch)
                    probs = outputs['probs'].cpu().numpy()[0]
                    pred_idx = probs.argmax()

                emotion = self.labels[pred_idx]
                confidence = float(probs[pred_idx])
            else:
                emotion = "neutral"
                confidence = 0.5

            # Assign result to each word in the group
            for w in group_words:
                word_results.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "emotion": emotion,
                    "confidence": confidence
                })

            i += 1

        return word_results


# ============================================================
# HTML GENERATION - ENHANCED WITH WORD-LEVEL STYLING
# ============================================================

def generate_html_output_v2(
    segments: List[Dict],
    video_path: str,
    output_path: str,
    culture: str = "western",
    age_group: str = "adult"
) -> str:
    """Generate enhanced HTML visualization with word-level emotion styling."""

    styles = CULTURE_STYLES[culture]
    age_scale = AGE_FONT_SCALE[age_group]

    # Build subtitle data with word-level emotions
    subtitles_js = []
    for seg in segments:
        words_data = seg.get("word_emotions", [])
        if words_data:
            # Word-level styling
            words_js = []
            for w in words_data:
                words_js.append({
                    "word": w["word"],
                    "emotion": w["emotion"],
                    "confidence": w["confidence"]
                })
            subtitles_js.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": words_js,
                "dominant_emotion": seg.get("emotion", "neutral")
            })
        else:
            # Fallback to sentence-level
            subtitles_js.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": [],
                "dominant_emotion": seg.get("emotion", "neutral")
            })

    # Build transcript entries with word-level coloring
    transcript_entries = []
    for idx, seg in enumerate(segments):
        time_str = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
        words_data = seg.get("word_emotions", [])

        if words_data:
            # Build word-by-word HTML
            words_html = []
            for w in words_data:
                style = styles.get(w["emotion"], styles["neutral"])
                size = style.font_size * age_scale
                words_html.append(
                    f'<span style="color:{style.color};font-size:{size}em;'
                    f'font-weight:{style.font_weight};text-shadow:{style.text_shadow}">'
                    f'{w["word"]}</span>'
                )
            text_html = "".join(words_html)
            dominant_emotion = seg.get("emotion", "neutral")
            confidence = seg.get("confidence", 0.5)
        else:
            emotion = seg.get("emotion", "neutral")
            confidence = seg.get("confidence", 0.5)
            style = styles.get(emotion, styles["neutral"])
            size = style.font_size * age_scale
            text_html = f'<span style="color:{style.color};font-size:{size}em;font-weight:{style.font_weight}">{seg["text"]}</span>'
            dominant_emotion = emotion

        transcript_entries.append(f'''
            <div class="transcript-entry" id="entry-{idx}" data-start="{seg['start']}" data-end="{seg['end']}">
                <span class="time">{time_str}</span>
                <span class="text">{text_html}</span>
                <span class="emotion-tag" style="background:{EMOTION_LEGEND_COLORS.get(dominant_emotion, '#888')}">{dominant_emotion} ({confidence:.0%})</span>
            </div>
        ''')

    transcript_html = "\n".join(transcript_entries)

    # JavaScript styles data
    styles_js = {}
    for culture_name, culture_styles in CULTURE_STYLES.items():
        styles_js[culture_name] = {}
        for emotion, style in culture_styles.items():
            styles_js[culture_name][emotion] = {
                "color": style.color,
                "fontSize": style.font_size,
                "fontWeight": style.font_weight,
                "background": style.background,
                "textShadow": style.text_shadow
            }

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentimentogram Demo V2 - Word-Level Emotion</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
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
            margin-bottom: 10px;
            color: #eee;
            font-size: 2em;
        }}
        .subtitle-info {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .video-container {{
            position: relative;
            width: 100%;
            max-width: 900px;
            margin: 0 auto 30px;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        video {{
            width: 100%;
            display: block;
        }}
        .subtitle-overlay {{
            position: absolute;
            bottom: 70px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            padding: 12px 24px;
            border-radius: 8px;
            max-width: 85%;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(5px);
            transition: all 0.2s ease;
            font-size: 1.4em;
            line-height: 1.5;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .control-group {{
            background: rgba(22, 33, 62, 0.8);
            padding: 15px 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        .control-group label {{
            display: block;
            margin-bottom: 8px;
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        select {{
            padding: 10px 20px;
            border-radius: 6px;
            border: 1px solid #333;
            background: #0f3460;
            color: #fff;
            cursor: pointer;
            font-size: 1em;
        }}
        .emotion-legend {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(22, 33, 62, 0.8);
            border-radius: 25px;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            box-shadow: 0 0 10px currentColor;
        }}
        .transcript {{
            background: rgba(22, 33, 62, 0.8);
            border-radius: 12px;
            padding: 20px;
            max-height: 450px;
            overflow-y: auto;
            backdrop-filter: blur(10px);
        }}
        .transcript h2 {{
            margin-bottom: 15px;
            color: #aaa;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .transcript-entry {{
            display: flex;
            align-items: flex-start;
            gap: 15px;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .transcript-entry:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .transcript-entry.active {{
            background: rgba(255,255,255,0.1);
            border-left: 3px solid #FFD700;
        }}
        .transcript-entry .time {{
            color: #666;
            font-size: 0.85em;
            min-width: 45px;
            font-family: monospace;
        }}
        .transcript-entry .text {{
            flex: 1;
            line-height: 1.6;
        }}
        .transcript-entry .emotion-tag {{
            font-size: 0.7em;
            padding: 4px 10px;
            border-radius: 12px;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }}
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #444;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentimentogram Demo V2</h1>
        <p class="subtitle-info">Word-Level Emotion Visualization with Distinct Colors</p>

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
                <span>Happy/Excited</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#4A90D9;color:#4A90D9"></div>
                <span>Sadness</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#B8B8B8;color:#B8B8B8"></div>
                <span>Neutral</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#FF4444;color:#FF4444"></div>
                <span>Anger</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background:#FF8C00;color:#FF8C00"></div>
                <span>Frustration</span>
            </div>
        </div>

        <div class="video-container">
            <video id="video" controls>
                <source src="{os.path.basename(video_path)}" type="video/mp4">
            </video>
            <div class="subtitle-overlay" id="subtitle"></div>
        </div>

        <div class="transcript">
            <h2>Transcript with Word-Level Emotions</h2>
            {transcript_html}
        </div>
    </div>

    <script>
        const subtitles = {json.dumps(subtitles_js)};
        const cultureStyles = {json.dumps(styles_js)};
        const ageScales = {json.dumps(AGE_FONT_SCALE)};
        const legendColors = {json.dumps(EMOTION_LEGEND_COLORS)};

        const video = document.getElementById('video');
        const subtitleEl = document.getElementById('subtitle');

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
                const styles = cultureStyles[culture];

                if (activeSubtitle.words && activeSubtitle.words.length > 0) {{
                    // Word-level styling
                    let html = '';
                    activeSubtitle.words.forEach(w => {{
                        const style = styles[w.emotion] || styles['neutral'];
                        const size = style.fontSize * scale;
                        html += `<span style="color:${{style.color}};font-size:${{size}}em;font-weight:${{style.fontWeight}};text-shadow:${{style.textShadow}}">${{w.word}}</span>`;
                    }});
                    subtitleEl.innerHTML = html;
                }} else {{
                    // Sentence-level fallback
                    const style = styles[activeSubtitle.dominant_emotion] || styles['neutral'];
                    const size = style.fontSize * scale;
                    subtitleEl.innerHTML = `<span style="color:${{style.color}};font-size:${{size}}em;font-weight:${{style.fontWeight}}">${{activeSubtitle.text}}</span>`;
                }}
                subtitleEl.style.display = 'block';
            }} else {{
                subtitleEl.style.display = 'none';
            }}
        }});

        // Click on transcript entry to seek
        document.querySelectorAll('.transcript-entry').forEach(entry => {{
            entry.addEventListener('click', () => {{
                const start = parseFloat(entry.dataset.start);
                video.currentTime = start;
                video.play();
            }});
        }});

        function updateStyles() {{
            const culture = document.getElementById('cultureSelect').value;
            const age = document.getElementById('ageSelect').value;
            const scale = ageScales[age];
            const styles = cultureStyles[culture];

            // Update transcript entries
            subtitles.forEach((sub, idx) => {{
                const entry = document.getElementById(`entry-${{idx}}`);
                const textSpan = entry.querySelector('.text');

                if (sub.words && sub.words.length > 0) {{
                    let html = '';
                    sub.words.forEach(w => {{
                        const style = styles[w.emotion] || styles['neutral'];
                        const size = style.fontSize * scale;
                        html += `<span style="color:${{style.color}};font-size:${{size}}em;font-weight:${{style.fontWeight}};text-shadow:${{style.textShadow}}">${{w.word}}</span>`;
                    }});
                    textSpan.innerHTML = html;
                }} else {{
                    const style = styles[sub.dominant_emotion] || styles['neutral'];
                    const size = style.fontSize * scale;
                    textSpan.innerHTML = `<span style="color:${{style.color}};font-size:${{size}}em;font-weight:${{style.fontWeight}}">${{sub.text}}</span>`;
                }}
            }});
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
    """Run the enhanced Sentimentogram pipeline with word-level emotions."""

    print("=" * 60)
    print("SENTIMENTOGRAM V2 - Word-Level Emotion Visualization")
    print("=" * 60)

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract audio
    print("\n[1/4] Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    # Step 2: Transcribe
    print("\n[2/4] Transcribing speech with word timestamps...")
    segments = transcribe_audio(audio_path, language)

    # Step 3: Emotion recognition
    print("\n[3/4] Detecting emotions...")
    if model_path and Path(model_path).exists():
        predictor = SERPredictorV2(model_path)

        for i, seg in enumerate(segments):
            print(f"  Processing segment {i+1}/{len(segments)}: {seg['text'][:50]}...")

            # Sentence-level prediction
            result = predictor.predict_segment(seg["text"], audio_path, seg["start"], seg["end"])
            seg.update(result)

            # Word-level prediction if enabled and words available
            if word_level and seg.get("words"):
                word_emotions = predictor.predict_words(seg["words"], audio_path)
                seg["word_emotions"] = word_emotions
            else:
                seg["word_emotions"] = []

    else:
        print("  Warning: No SER model provided. Using placeholder emotions.")
        for seg in segments:
            seg["emotion"] = "neutral"
            seg["confidence"] = 0.5
            seg["vad"] = [0, 0, 0]
            seg["word_emotions"] = []

    # Step 4: Generate visualization
    print("\n[4/4] Generating enhanced visualization...")

    # Copy video to output directory
    import shutil
    video_output = output_dir / Path(video_path).name
    if not video_output.exists() or not video_output.samefile(video_path):
        shutil.copy(video_path, video_output)
        print(f"  Copied video to: {video_output}")

    html_path = generate_html_output_v2(
        segments,
        str(video_output),
        output_path,
        culture=culture,
        age_group=age_group
    )

    # Save segments data
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
    parser = argparse.ArgumentParser(description="Sentimentogram Demo V2 - Word-Level Emotion")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="demo/output/result_v2.html", help="Output HTML path")
    parser.add_argument("--model", type=str, default="saved_models/novel_iemocap_5_5class.pt", help="SER model path")
    parser.add_argument("--language", type=str, default="en", help="Language for transcription")
    parser.add_argument("--culture", type=str, default="western", choices=["western", "eastern"])
    parser.add_argument("--age", type=str, default="adult", choices=["child", "teen", "adult", "senior"])
    parser.add_argument("--no-word-level", action="store_true", help="Disable word-level emotion detection")

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
