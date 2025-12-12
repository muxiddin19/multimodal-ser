"""
Create MELD metadata for emotion recognition
MELD has 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
We'll map to 4-class to match IEMOCAP: anger, happiness(joy), neutral, sadness
"""
import os
import csv
import pandas as pd
from pathlib import Path

MELD_PATH = "/nas/dataset/meld.raw"
OUTPUT_DIR = "metadata"

# Map MELD emotions to 4-class IEMOCAP format
# MELD: anger, disgust, fear, joy, neutral, sadness, surprise
# IEMOCAP 4-class: 0=anger, 1=happiness, 2=neutral, 3=sadness
emotion_map_4class = {
    "anger": 0,
    "joy": 1,      # map to happiness
    "neutral": 2,
    "sadness": 3,
    # disgust, fear, surprise excluded for 4-class
}


def find_audio_file(splits_dir, dialogue_id, utterance_id):
    """Find audio/video file for given dialogue and utterance"""
    # MELD stores files as dia{dialogue_id}_utt{utterance_id}.mp4
    filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"

    # Check directly in splits_dir first
    file_path = os.path.join(splits_dir, filename)
    if os.path.exists(file_path):
        return file_path

    return None


def process_meld_split(csv_path, splits_dir, split_name):
    """Process a MELD split (train/dev/test)"""
    df = pd.read_csv(csv_path)
    rows = []
    missing = 0

    print(f"Processing {split_name}: {len(df)} utterances")

    for idx, row in df.iterrows():
        emotion = row['Emotion'].lower()
        utterance = row['Utterance']
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']

        # Only include 4-class emotions
        if emotion not in emotion_map_4class:
            continue

        # Find audio file
        audio_path = find_audio_file(splits_dir, dialogue_id, utterance_id)

        if audio_path is None:
            missing += 1
            continue

        rows.append([
            audio_path,
            utterance,
            emotion_map_4class[emotion]
        ])

    print(f"  Found: {len(rows)}, Missing: {missing}")
    return rows


def main():
    all_rows = {
        "train": [],
        "val": [],
        "test": []
    }

    # Process train split
    train_csv = os.path.join(MELD_PATH, "train_sent_emo.csv")
    train_splits = os.path.join(MELD_PATH, "train_splits")
    if os.path.exists(train_csv) and os.path.exists(train_splits):
        all_rows["train"] = process_meld_split(train_csv, train_splits, "train")

    # Process dev/val split
    dev_csv = os.path.join(MELD_PATH, "dev_sent_emo.csv")
    dev_splits = os.path.join(MELD_PATH, "dev_splits_complete")
    if os.path.exists(dev_csv) and os.path.exists(dev_splits):
        all_rows["val"] = process_meld_split(dev_csv, dev_splits, "val")

    # Process test split
    test_csv = os.path.join(MELD_PATH, "test_sent_emo.csv")
    test_splits = os.path.join(MELD_PATH, "output_repeated_splits_test")
    if os.path.exists(test_csv) and os.path.exists(test_splits):
        all_rows["test"] = process_meld_split(test_csv, test_splits, "test")

    # Save metadata
    for split_name, rows in all_rows.items():
        if not rows:
            print(f"Warning: No data for {split_name}")
            continue

        output_path = f"{OUTPUT_DIR}/MELD_4class_{split_name}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["audio_file", "raw_text", "label"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} samples to {output_path}")

        # Print class distribution
        from collections import Counter
        labels = [r[2] for r in rows]
        print(f"  Class distribution: {dict(sorted(Counter(labels).items()))}")


if __name__ == "__main__":
    main()
