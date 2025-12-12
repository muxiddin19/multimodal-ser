"""
Extract BERT + emotion2vec features for cross-dataset evaluation
Supports: CREMA-D, MELD (extracts audio from video if needed)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import pickle
import pandas as pd
import subprocess
import tempfile
from transformers import BertTokenizer, BertModel
from funasr import AutoModel

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TEXT_MAX_LENGTH = 100

# Load BERT model
print("Loading BERT model...")
tokenizer_eng = BertTokenizer.from_pretrained('bert-base-uncased')
text_model_eng = BertModel.from_pretrained('bert-base-uncased', use_safetensors=True).to(device)

# Load emotion2vec model
print("Loading emotion2vec model...")
emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.5")

print("Models loaded successfully")


def extract_audio_from_video(video_path, output_path=None):
    """Extract audio from video file using ffmpeg"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None


def extract_emotion2vec_features(audio_path):
    """Extract emotion2vec features from audio file"""
    try:
        result = emotion2vec_model.generate(audio_path, output_dir=None, granularity="utterance")
        if isinstance(result, list) and len(result) > 0:
            feats = result[0].get('feats', None)
            if feats is not None:
                if isinstance(feats, np.ndarray):
                    return torch.from_numpy(feats).float()
                return feats
        return None
    except Exception as e:
        print(f"Error extracting emotion2vec: {e}")
        return None


def process_dataset(dataset_path, output_file, is_video=False):
    """Processes a dataset and saves the features as a pickle file."""
    dataset = pd.read_csv(dataset_path)
    processed_data = []
    failed = 0

    print(f"Processing {len(dataset)} samples from {dataset_path}...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(dataset)}")

            try:
                # Process text with BERT
                text = str(dataset['raw_text'][idx])
                text_token = tokenizer_eng(text, return_tensors="pt", truncation=True, max_length=TEXT_MAX_LENGTH)
                text_token = text_token.to(device)
                text_outputs = text_model_eng(**text_token)
                text_embed = text_outputs.last_hidden_state[:, 0, :][0].cpu()

                # Process audio with emotion2vec
                audio_file = dataset['audio_file'][idx]

                # Extract audio from video if needed
                temp_audio = None
                if is_video or audio_file.endswith(('.mp4', '.avi', '.mkv')):
                    temp_audio = extract_audio_from_video(audio_file)
                    if temp_audio is None:
                        failed += 1
                        continue
                    audio_file = temp_audio

                audio_embed = extract_emotion2vec_features(audio_file)

                # Clean up temp file
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)

                if audio_embed is None:
                    failed += 1
                    continue

                # Get label
                label = dataset['label'][idx]
                label = torch.tensor(int(label))

                processed_data.append({
                    'text_embed': text_embed,
                    'audio_embed': audio_embed,
                    'label': label
                })
            except Exception as e:
                print(f"  Error at idx {idx}: {e}")
                failed += 1

    print(f"  Processed: {len(processed_data)}, Failed: {failed}")

    # Save to pickle
    with open(output_file, "wb") as file:
        pickle.dump(processed_data, file)
    print(f"Processed data saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cremad", "meld"])
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    if args.dataset == "cremad":
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        for split in splits:
            input_path = f"metadata/CREMAD_4class_{split}.csv"
            output_path = f"features/CREMAD_emotion2vec_{split}.pkl"
            if os.path.exists(input_path):
                process_dataset(input_path, output_path, is_video=False)
            else:
                print(f"Metadata not found: {input_path}")
                print("Run: python metadata/create_cremad_metadata.py")

    elif args.dataset == "meld":
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        for split in splits:
            input_path = f"metadata/MELD_4class_{split}.csv"
            output_path = f"features/MELD_emotion2vec_{split}.pkl"
            if os.path.exists(input_path):
                process_dataset(input_path, output_path, is_video=True)
            else:
                print(f"Metadata not found: {input_path}")
                print("Run: python metadata/create_meld_metadata.py")


if __name__ == "__main__":
    main()
