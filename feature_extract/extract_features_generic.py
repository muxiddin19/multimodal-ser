"""
Generic Feature Extraction for Multiple Datasets
Supports: IEMOCAP, MOSEI, CREMA-D, RAVDESS, MELD

This addresses reviewer concern about "narrow dataset scope"
"""

import os
import pickle
import argparse
import torch
import torchaudio
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2Processor, Wav2Vec2Model
)
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for different datasets."""
    name: str
    audio_col: str
    text_col: str
    label_col: str
    sample_rate: int = 16000
    emotion_mapping: Optional[Dict] = None


# Dataset configurations
DATASET_CONFIGS = {
    "iemocap_4": DatasetConfig(
        name="IEMOCAP (4-class)",
        audio_col="audio_path",
        text_col="transcription",
        label_col="emotion",
        emotion_mapping={
            "ang": 0, "anger": 0,
            "hap": 1, "happiness": 1, "exc": 1, "excited": 1,
            "neu": 2, "neutral": 2,
            "sad": 3, "sadness": 3
        }
    ),
    "iemocap_6": DatasetConfig(
        name="IEMOCAP (6-class)",
        audio_col="audio_path",
        text_col="transcription",
        label_col="emotion",
        emotion_mapping={
            "hap": 0, "happiness": 0,
            "sad": 1, "sadness": 1,
            "neu": 2, "neutral": 2,
            "ang": 3, "anger": 3,
            "exc": 4, "excited": 4,
            "fru": 5, "frustration": 5
        }
    ),
    "mosei": DatasetConfig(
        name="CMU-MOSEI",
        audio_col="audio_path",
        text_col="text",
        label_col="sentiment",
        emotion_mapping={
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    ),
    "cremad": DatasetConfig(
        name="CREMA-D",
        audio_col="audio_path",
        text_col="sentence",
        label_col="emotion",
        emotion_mapping={
            "ANG": 0, "anger": 0,
            "DIS": 1, "disgust": 1,
            "FEA": 2, "fear": 2,
            "HAP": 3, "happy": 3,
            "NEU": 4, "neutral": 4,
            "SAD": 5, "sad": 5
        }
    ),
    "ravdess": DatasetConfig(
        name="RAVDESS",
        audio_col="audio_path",
        text_col="text",
        label_col="emotion",
        emotion_mapping={
            "neutral": 0,
            "calm": 1,
            "happy": 2,
            "sad": 3,
            "angry": 4,
            "fearful": 5,
            "disgust": 6,
            "surprised": 7
        }
    ),
    "meld": DatasetConfig(
        name="MELD",
        audio_col="audio_path",
        text_col="Utterance",
        label_col="Emotion",
        emotion_mapping={
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
    )
}


class AudioFeatureExtractor:
    """Extract audio features using Wav2Vec2 or emotion2vec."""

    def __init__(self, model_name: str = "wav2vec2", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if model_name == "wav2vec2":
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h"
            ).to(self.device)
        elif model_name == "emotion2vec":
            # Using funasr for emotion2vec
            try:
                from funasr import AutoModel as FunAutoModel
                self.model = FunAutoModel(model="iic/emotion2vec_plus_large")
            except ImportError:
                print("Warning: funasr not installed. Falling back to wav2vec2.")
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                ).to(self.device)
                self.model_name = "wav2vec2"

        self.model.eval()

    def extract(self, audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """Extract audio features from audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze()

            if self.model_name == "wav2vec2":
                inputs = self.processor(
                    waveform.numpy(),
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling over time
                    features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            elif self.model_name == "emotion2vec":
                with torch.no_grad():
                    res = self.model.generate(audio_path, granularity="utterance")
                    features = res[0]["feats"].squeeze().cpu().numpy()

            return features

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(768)  # Return zero vector on error


class TextFeatureExtractor:
    """Extract text features using BERT."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, text: str) -> np.ndarray:
        """Extract text features from text string."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token
                features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            return features

        except Exception as e:
            print(f"Error processing text: {e}")
            return np.zeros(768)


def extract_features_from_metadata(
    metadata_path: str,
    output_path: str,
    dataset_type: str = "iemocap_4",
    audio_model: str = "wav2vec2",
    text_model: str = "bert-base-uncased",
    device: str = "cuda"
):
    """
    Extract features from metadata CSV file.

    Args:
        metadata_path: Path to metadata CSV
        output_path: Path to save features pickle
        dataset_type: One of the supported dataset types
        audio_model: Audio model to use (wav2vec2, emotion2vec)
        text_model: Text model to use
        device: Device to use
    """
    config = DATASET_CONFIGS.get(dataset_type)
    if config is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Extracting features for {config.name}")
    print(f"Audio model: {audio_model}")
    print(f"Text model: {text_model}")

    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples")

    # Initialize extractors
    audio_extractor = AudioFeatureExtractor(audio_model, device)
    text_extractor = TextFeatureExtractor(text_model, device)

    features_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = row[config.audio_col]
        text = str(row[config.text_col]) if pd.notna(row[config.text_col]) else ""
        label_raw = row[config.label_col]

        # Map label
        if config.emotion_mapping:
            label = config.emotion_mapping.get(str(label_raw).lower(), -1)
            if label == -1:
                continue  # Skip unknown labels
        else:
            label = int(label_raw)

        # Extract features
        audio_embed = audio_extractor.extract(audio_path)
        text_embed = text_extractor.extract(text)

        features_list.append({
            'audio_embed': torch.tensor(audio_embed, dtype=torch.float32),
            'text_embed': torch.tensor(text_embed, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'audio_path': audio_path,
            'text': text
        })

    # Save features
    with open(output_path, 'wb') as f:
        pickle.dump(features_list, f)

    print(f"Saved {len(features_list)} samples to {output_path}")

    # Print label distribution
    labels = [f['label'].item() for f in features_list]
    from collections import Counter
    print("Label distribution:", Counter(labels))


def main():
    parser = argparse.ArgumentParser(description="Extract features for SER")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--output", type=str, required=True, help="Output pickle path")
    parser.add_argument("--dataset", type=str, default="iemocap_4",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--audio_model", type=str, default="wav2vec2",
                        choices=["wav2vec2", "emotion2vec"])
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    extract_features_from_metadata(
        args.metadata,
        args.output,
        args.dataset,
        args.audio_model,
        args.text_model,
        args.device
    )


if __name__ == "__main__":
    main()
