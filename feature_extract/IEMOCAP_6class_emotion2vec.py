"""
Extract BERT + emotion2vec features for 6-class IEMOCAP
Classes: happiness, sadness, neutral, anger, excitement, frustration
"""
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from funasr import AutoModel

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TEXT_MAX_LENGTH = 100

# Paths
IEMOCAP_TRAIN_PATH = "metadata/IEMOCAP_6class_train.csv"
IEMOCAP_VAL_PATH = "metadata/IEMOCAP_6class_val.csv"
IEMOCAP_TEST_PATH = "metadata/IEMOCAP_6class_test.csv"

OUTPUT_PATH = "features/"

# Load BERT model
print("Loading BERT model...")
tokenizer_eng = BertTokenizer.from_pretrained('bert-base-uncased')
text_model_eng = BertModel.from_pretrained('bert-base-uncased', use_safetensors=True).to(device)

# Load emotion2vec model
print("Loading emotion2vec model...")
emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.5")

print("Models loaded successfully")


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


def process_dataset(dataset_path, output_file):
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
                text = dataset['raw_text'][idx]
                text_token = tokenizer_eng(text, return_tensors="pt", truncation=True, max_length=TEXT_MAX_LENGTH)
                text_token = text_token.to(device)
                text_outputs = text_model_eng(**text_token)
                text_embed = text_outputs.last_hidden_state[:, 0, :][0].cpu()

                # Process audio with emotion2vec
                audio_file = dataset['audio_file'][idx]
                audio_embed = extract_emotion2vec_features(audio_file)

                if audio_embed is None:
                    print(f"  Failed to extract emotion2vec for {audio_file}")
                    failed += 1
                    continue

                # Get label
                label = dataset['label'][idx]
                label = torch.tensor(label)

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
    """Main function to process IEMOCAP 6-class datasets."""
    # Process training set
    process_dataset(
        IEMOCAP_TRAIN_PATH,
        f"{OUTPUT_PATH}IEMOCAP_6class_emotion2vec_train.pkl"
    )

    # Process validation set
    process_dataset(
        IEMOCAP_VAL_PATH,
        f"{OUTPUT_PATH}IEMOCAP_6class_emotion2vec_val.pkl"
    )

    # Process test set
    process_dataset(
        IEMOCAP_TEST_PATH,
        f"{OUTPUT_PATH}IEMOCAP_6class_emotion2vec_test.pkl"
    )


if __name__ == "__main__":
    main()
