"""
Sentimentogram Demo: Emotion-Aware Subtitle Visualization
=========================================================

This demo implements the full Sentimentogram pipeline:
1. Extract audio from video
2. Transcribe speech using Whisper (STT)
3. Detect emotions using our trained SER model
4. Map emotions to colors/fonts based on VAD theory
5. Generate personalized subtitle visualization

Usage:
    python demo/sentimentogram_demo.py --video input.mp4 --output output.html
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
# CONFIGURATION
# ============================================================

@dataclass
class EmotionStyle:
    """Style configuration for each emotion."""
    color: str           # Hex color
    font_size: str       # CSS font size
    font_weight: str     # normal, bold
    background: str      # Background color (optional)

# VAD values from paper (Table 1)
VAD_VALUES = {
    "joy": [0.960, 0.648, 0.588],
    "anger": [-0.666, 0.730, 0.314],
    "sadness": [-0.896, -0.424, -0.672],
    "fear": [-0.854, 0.680, -0.414],
    "surprise": [0.750, 0.750, 0.124],
    "disgust": [-0.896, 0.550, -0.366],
    "neutral": [-0.062, -0.632, -0.286],
    # 5-class IEMOCAP
    "happy_excited": [0.905, 0.699, 0.519],
    "frustration": [-0.500, 0.600, -0.200],
}

# Default emotion-color mapping (Western culture)
EMOTION_STYLES_DEFAULT = {
    "joy": EmotionStyle("#FFD700", "1.2em", "bold", "rgba(255,215,0,0.1)"),        # Gold
    "happy_excited": EmotionStyle("#FFD700", "1.2em", "bold", "rgba(255,215,0,0.1)"),
    "anger": EmotionStyle("#FF4444", "1.3em", "bold", "rgba(255,68,68,0.1)"),       # Red
    "sadness": EmotionStyle("#4169E1", "0.95em", "normal", "rgba(65,105,225,0.1)"), # Blue
    "fear": EmotionStyle("#800080", "1.1em", "normal", "rgba(128,0,128,0.1)"),      # Purple
    "surprise": EmotionStyle("#FFA500", "1.15em", "bold", "rgba(255,165,0,0.1)"),   # Orange
    "disgust": EmotionStyle("#228B22", "1.0em", "normal", "rgba(34,139,34,0.1)"),   # Green
    "neutral": EmotionStyle("#FFFFFF", "1.0em", "normal", "transparent"),           # White
    "frustration": EmotionStyle("#DC143C", "1.1em", "bold", "rgba(220,20,60,0.1)"), # Crimson
}

# Culture-specific color mappings
CULTURE_MAPPINGS = {
    "western": EMOTION_STYLES_DEFAULT,
    "eastern": {
        "joy": EmotionStyle("#FF0000", "1.2em", "bold", "rgba(255,0,0,0.1)"),       # Red (lucky)
        "happy_excited": EmotionStyle("#FF0000", "1.2em", "bold", "rgba(255,0,0,0.1)"),
        "anger": EmotionStyle("#000000", "1.3em", "bold", "rgba(0,0,0,0.2)"),       # Black
        "sadness": EmotionStyle("#FFFFFF", "0.95em", "normal", "rgba(200,200,200,0.3)"), # White (mourning)
        "fear": EmotionStyle("#800080", "1.1em", "normal", "rgba(128,0,128,0.1)"),
        "surprise": EmotionStyle("#FFA500", "1.15em", "bold", "rgba(255,165,0,0.1)"),
        "disgust": EmotionStyle("#228B22", "1.0em", "normal", "rgba(34,139,34,0.1)"),
        "neutral": EmotionStyle("#CCCCCC", "1.0em", "normal", "transparent"),
        "frustration": EmotionStyle("#8B0000", "1.1em", "bold", "rgba(139,0,0,0.1)"),
    }
}

# Age-based font size adjustments
AGE_FONT_SCALE = {
    "child": 1.3,      # Larger fonts for children
    "teen": 1.1,
    "adult": 1.0,
    "senior": 1.25,    # Larger for readability
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
    Returns list of segments with text, start, end times.
    """
    try:
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper

    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

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
# SPEECH EMOTION RECOGNITION
# ============================================================

class SERPredictor:
    """Speech Emotion Recognition predictor using trained model."""

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

        # Load checkpoint - add Config class to __main__ namespace first
        import sys

        # Define Config class that matches the saved one
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

        # Inject Config into __main__ namespace (where pickle expects to find it)
        import __main__
        __main__.Config = Config

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Emotion labels
        self.labels = {
            "iemocap_4": ["anger", "joy", "neutral", "sadness"],
            "iemocap_5": ["happy_excited", "sadness", "neutral", "anger", "frustration"],
            "iemocap_6": ["joy", "sadness", "neutral", "anger", "surprise", "frustration"],
        }[config_type]

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
        self.emotion2vec = AutoModel(model="iic/emotion2vec_plus_large")

    def extract_features(self, text: str, audio_path: str, start: float, end: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract text and audio features for a segment."""
        # Text features (BERT)
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=100)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            text_output = self.bert(**tokens)
            text_feat = text_output.last_hidden_state[:, 0, :].squeeze(0)

        # Audio features (emotion2vec) - extract segment
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)

        # Extract segment
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment = audio[:, start_sample:end_sample]

        # Check minimum segment length (emotion2vec needs at least ~0.5s = 8000 samples for stable results)
        MIN_SAMPLES = 8000  # 0.5 seconds at 16kHz
        if segment.shape[1] == 0:
            # Empty segment - create silent audio
            segment = torch.zeros(1, MIN_SAMPLES)
        elif segment.shape[1] < MIN_SAMPLES:
            # Pad short segments with silence
            padding = torch.zeros(1, MIN_SAMPLES - segment.shape[1])
            segment = torch.cat([segment, padding], dim=1)

        # Save temporary segment
        temp_path = "/tmp/segment_temp.wav"
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
            print(f"  Warning: emotion2vec failed for segment ({start:.2f}-{end:.2f}s): {e}")
            audio_feat = torch.zeros(1024)

        return text_feat.cpu(), audio_feat.cpu()

    def predict(self, text: str, audio_path: str, start: float, end: float) -> Dict:
        """Predict emotion for a segment."""
        text_feat, audio_feat = self.extract_features(text, audio_path, start, end)

        with torch.no_grad():
            text_feat = text_feat.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device)

            outputs = self.model(text_feat, audio_feat)
            probs = outputs['probs'].cpu().numpy()[0]
            pred_idx = probs.argmax()

            # Get VAD prediction if available
            vad = outputs.get('vad', None)
            if vad is not None:
                vad = vad.cpu().numpy()[0]

        return {
            "emotion": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": {self.labels[i]: float(p) for i, p in enumerate(probs)},
            "vad": vad.tolist() if vad is not None else None
        }


# ============================================================
# VISUALIZATION
# ============================================================

def vad_to_color(vad: List[float]) -> str:
    """Convert VAD values to HSL color."""
    v, a, d = vad

    # Map Valence to Hue (negative=blue, positive=yellow/red)
    hue = int((v + 1) / 2 * 60)  # 0-60 range (red to yellow)
    if v < 0:
        hue = int(240 - (v + 1) * 60)  # Blue range for negative

    # Map Arousal to Saturation
    saturation = int(50 + (a + 1) / 2 * 50)  # 50-100%

    # Map Dominance to Lightness
    lightness = int(30 + (d + 1) / 2 * 40)  # 30-70%

    return f"hsl({hue}, {saturation}%, {lightness}%)"


def generate_html_output(
    segments: List[Dict],
    video_path: str,
    output_path: str,
    culture: str = "western",
    age_group: str = "adult",
    title: str = "Sentimentogram Demo"
):
    """Generate HTML visualization with emotion-colored subtitles."""

    style_map = CULTURE_MAPPINGS.get(culture, EMOTION_STYLES_DEFAULT)
    font_scale = AGE_FONT_SCALE.get(age_group, 1.0)

    # Build subtitle entries
    subtitle_entries = []
    for seg in segments:
        emotion = seg.get("emotion", "neutral")
        style = style_map.get(emotion, style_map["neutral"])

        # Scale font size
        base_size = float(style.font_size.replace("em", ""))
        scaled_size = base_size * font_scale

        # Use VAD-based color if available
        if seg.get("vad"):
            color = vad_to_color(seg["vad"])
        else:
            color = style.color

        subtitle_entries.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "emotion": emotion,
            "confidence": seg.get("confidence", 0),
            "color": color,
            "fontSize": f"{scaled_size}em",
            "fontWeight": style.font_weight,
            "background": style.background,
        })

    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #fff;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #eee;
        }}
        .video-container {{
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 0 auto 30px;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }}
        video {{
            width: 100%;
            display: block;
        }}
        .subtitle-overlay {{
            position: absolute;
            bottom: 60px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            padding: 10px 20px;
            border-radius: 5px;
            max-width: 90%;
            transition: all 0.3s ease;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .control-group {{
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
        }}
        .control-group label {{
            display: block;
            margin-bottom: 8px;
            color: #888;
            font-size: 0.9em;
        }}
        select {{
            padding: 8px 15px;
            border-radius: 5px;
            border: none;
            background: #0f3460;
            color: #fff;
            cursor: pointer;
        }}
        .emotion-legend {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            background: #16213e;
            border-radius: 20px;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
        }}
        .transcript {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .transcript-entry {{
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .transcript-entry:hover {{
            transform: translateX(5px);
        }}
        .transcript-entry.active {{
            box-shadow: 0 0 10px rgba(255,255,255,0.3);
        }}
        .time-stamp {{
            font-size: 0.8em;
            color: #888;
            margin-right: 10px;
        }}
        .emotion-tag {{
            font-size: 0.75em;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 10px;
            background: rgba(255,255,255,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <div class="control-group">
                <label>Culture Setting</label>
                <select id="cultureSelect" onchange="updateStyles()">
                    <option value="western">Western</option>
                    <option value="eastern">Eastern (Asian)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Age Group</label>
                <select id="ageSelect" onchange="updateStyles()">
                    <option value="adult">Adult</option>
                    <option value="child">Child</option>
                    <option value="teen">Teen</option>
                    <option value="senior">Senior</option>
                </select>
            </div>
        </div>

        <div class="emotion-legend" id="legend"></div>

        <div class="video-container">
            <video id="video" controls>
                <source src="{os.path.basename(video_path)}" type="video/mp4">
            </video>
            <div class="subtitle-overlay" id="subtitle"></div>
        </div>

        <h2 style="margin-bottom: 15px;">Transcript</h2>
        <div class="transcript" id="transcript"></div>
    </div>

    <script>
        const subtitles = {json.dumps(subtitle_entries, indent=2)};

        const cultureStyles = {json.dumps({
            "western": {k: {"color": v.color, "fontSize": v.font_size, "fontWeight": v.font_weight, "background": v.background}
                       for k, v in EMOTION_STYLES_DEFAULT.items()},
            "eastern": {k: {"color": v.color, "fontSize": v.font_size, "fontWeight": v.font_weight, "background": v.background}
                       for k, v in CULTURE_MAPPINGS["eastern"].items()}
        })};

        const ageScales = {json.dumps(AGE_FONT_SCALE)};

        const video = document.getElementById('video');
        const subtitleEl = document.getElementById('subtitle');
        const transcriptEl = document.getElementById('transcript');
        const legendEl = document.getElementById('legend');

        // Build transcript
        subtitles.forEach((sub, idx) => {{
            const entry = document.createElement('div');
            entry.className = 'transcript-entry';
            entry.style.background = sub.background;
            entry.style.color = sub.color;
            entry.innerHTML = `
                <span class="time-stamp">${{formatTime(sub.start)}}</span>
                ${{sub.text}}
                <span class="emotion-tag">${{sub.emotion}} (${{(sub.confidence * 100).toFixed(0)}}%)</span>
            `;
            entry.onclick = () => video.currentTime = sub.start;
            entry.id = `entry-${{idx}}`;
            transcriptEl.appendChild(entry);
        }});

        // Build legend
        const emotions = [...new Set(subtitles.map(s => s.emotion))];
        emotions.forEach(emotion => {{
            const sub = subtitles.find(s => s.emotion === emotion);
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${{sub.color}}"></div>
                <span>${{emotion}}</span>
            `;
            legendEl.appendChild(item);
        }});

        // Update subtitles on video time update
        video.addEventListener('timeupdate', () => {{
            const time = video.currentTime;
            const currentSub = subtitles.find(s => time >= s.start && time <= s.end);

            if (currentSub) {{
                subtitleEl.textContent = currentSub.text;
                subtitleEl.style.color = currentSub.color;
                subtitleEl.style.fontSize = currentSub.fontSize;
                subtitleEl.style.fontWeight = currentSub.fontWeight;
                subtitleEl.style.background = currentSub.background;
                subtitleEl.style.display = 'block';

                // Highlight transcript
                document.querySelectorAll('.transcript-entry').forEach(e => e.classList.remove('active'));
                const idx = subtitles.indexOf(currentSub);
                document.getElementById(`entry-${{idx}}`).classList.add('active');
            }} else {{
                subtitleEl.style.display = 'none';
            }}
        }});

        function formatTime(seconds) {{
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${{m}}:${{s.toString().padStart(2, '0')}}`;
        }}

        function updateStyles() {{
            const culture = document.getElementById('cultureSelect').value;
            const age = document.getElementById('ageSelect').value;
            const scale = ageScales[age];
            const styles = cultureStyles[culture];

            subtitles.forEach((sub, idx) => {{
                const style = styles[sub.emotion] || styles['neutral'];
                const baseSize = parseFloat(style.fontSize);
                sub.color = style.color;
                sub.fontSize = (baseSize * scale) + 'em';
                sub.fontWeight = style.fontWeight;
                sub.background = style.background;

                const entry = document.getElementById(`entry-${{idx}}`);
                entry.style.color = sub.color;
                entry.style.background = sub.background;
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
    age_group: str = "adult"
):
    """Run the full Sentimentogram pipeline."""

    print("=" * 60)
    print("SENTIMENTOGRAM - Emotion-Aware Subtitle Visualization")
    print("=" * 60)

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract audio
    print("\n[1/4] Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    # Step 2: Transcribe
    print("\n[2/4] Transcribing speech...")
    segments = transcribe_audio(audio_path, language)

    # Step 3: Emotion recognition
    print("\n[3/4] Detecting emotions...")
    if model_path and Path(model_path).exists():
        predictor = SERPredictor(model_path)
        for seg in segments:
            result = predictor.predict(seg["text"], audio_path, seg["start"], seg["end"])
            seg.update(result)
    else:
        print("  Warning: No SER model provided. Using placeholder emotions.")
        # Placeholder: assign based on text sentiment (simple heuristic)
        for seg in segments:
            seg["emotion"] = "neutral"
            seg["confidence"] = 0.5
            seg["vad"] = [0, 0, 0]

    # Step 4: Generate visualization
    print("\n[4/4] Generating visualization...")

    # Copy video to output directory
    import shutil
    video_output = output_dir / Path(video_path).name
    if not video_output.exists():
        shutil.copy(video_path, video_output)

    html_path = generate_html_output(
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
    print("=" * 60)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Sentimentogram Demo")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="demo/output/result.html", help="Output HTML path")
    parser.add_argument("--model", type=str, default="saved_models/novel_iemocap_5_5class.pt", help="SER model path")
    parser.add_argument("--language", type=str, default="en", help="Language for transcription")
    parser.add_argument("--culture", type=str, default="western", choices=["western", "eastern"])
    parser.add_argument("--age", type=str, default="adult", choices=["child", "teen", "adult", "senior"])

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        language=args.language,
        culture=args.culture,
        age_group=args.age
    )


if __name__ == "__main__":
    main()
