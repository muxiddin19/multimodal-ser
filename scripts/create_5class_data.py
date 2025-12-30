"""
Create 5-class IEMOCAP data by merging happiness and excitement.

Original 6-class labels:
  0: happiness, 1: sadness, 2: neutral, 3: anger, 4: excitement, 5: frustration

New 5-class labels:
  0: happy/excited (merged), 1: sadness, 2: neutral, 3: anger, 4: frustration
"""
import pickle
import pandas as pd
import torch
from pathlib import Path

# Label mapping: old -> new
LABEL_MAP = {
    0: 0,  # happiness -> happy/excited
    1: 1,  # sadness -> sadness
    2: 2,  # neutral -> neutral
    3: 3,  # anger -> anger
    4: 0,  # excitement -> happy/excited (MERGED)
    5: 4,  # frustration -> frustration
}

LABEL_NAMES = ['happy_excited', 'sadness', 'neutral', 'anger', 'frustration']

def convert_metadata(input_path, output_path):
    """Convert 6-class metadata CSV to 5-class."""
    df = pd.read_csv(input_path)
    df['label'] = df['label'].map(LABEL_MAP)
    df.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"  Class distribution: {df['label'].value_counts().sort_index().to_dict()}")
    return df

def convert_features(input_path, output_path):
    """Convert 6-class feature pickle to 5-class."""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    converted = []
    for item in data:
        label = item['label']
        if torch.is_tensor(label):
            old_label = label.item()
        else:
            old_label = int(label)

        new_label = LABEL_MAP[old_label]

        converted.append({
            'text_embed': item['text_embed'],
            'audio_embed': item['audio_embed'],
            'label': torch.tensor(new_label, dtype=torch.long)
        })

    with open(output_path, 'wb') as f:
        pickle.dump(converted, f)

    # Count class distribution
    labels = [item['label'].item() for item in converted]
    from collections import Counter
    dist = Counter(labels)
    print(f"Created {output_path}")
    print(f"  Samples: {len(converted)}, Distribution: {dict(sorted(dist.items()))}")

def main():
    # Create output directories
    Path('metadata').mkdir(exist_ok=True)
    Path('features').mkdir(exist_ok=True)

    print("=" * 60)
    print("Creating 5-class IEMOCAP data (happiness + excitement merged)")
    print("=" * 60)

    # Convert metadata
    print("\n--- Converting Metadata ---")
    for split in ['train', 'val', 'test']:
        convert_metadata(
            f'metadata/IEMOCAP_6class_{split}.csv',
            f'metadata/IEMOCAP_5class_{split}.csv'
        )

    # Convert emotion2vec features
    print("\n--- Converting emotion2vec Features ---")
    for split in ['train', 'val', 'test']:
        convert_features(
            f'features/IEMOCAP_6class_emotion2vec_{split}.pkl',
            f'features/IEMOCAP_5class_emotion2vec_{split}.pkl'
        )

    # Convert wav2vec features if they exist
    print("\n--- Converting Wav2Vec2 Features ---")
    for split in ['train', 'val', 'test']:
        wav2vec_path = f'features/IEMOCAP_6class_wav2vec_{split}.pkl'
        if Path(wav2vec_path).exists() and Path(wav2vec_path).stat().st_size > 100:
            convert_features(
                wav2vec_path,
                f'features/IEMOCAP_5class_wav2vec_{split}.pkl'
            )
        else:
            print(f"  Skipping {wav2vec_path} (not found or empty)")

    print("\n" + "=" * 60)
    print("5-class Label Mapping:")
    print("  0: happy_excited (happiness + excitement merged)")
    print("  1: sadness")
    print("  2: neutral")
    print("  3: anger")
    print("  4: frustration")
    print("=" * 60)

if __name__ == "__main__":
    main()
