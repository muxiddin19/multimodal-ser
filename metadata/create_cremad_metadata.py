"""
Create CREMA-D metadata for emotion recognition
CREMA-D emotions: ANG, DIS, FEA, HAP, NEU, SAD
We'll map to 4-class to match IEMOCAP: anger, happiness, neutral, sadness
"""
import os
import csv
import random

CREMAD_PATH = "/nas/dataset/crema-d/AudioWAV"
OUTPUT_DIR = "metadata"

# Map CREMA-D emotions to 4-class IEMOCAP format
# CREMA-D: ANG, DIS, FEA, HAP, NEU, SAD
# IEMOCAP 4-class: 0=anger, 1=happiness, 2=neutral, 3=sadness
emotion_map_4class = {
    "ANG": 0,  # anger
    "HAP": 1,  # happiness
    "NEU": 2,  # neutral
    "SAD": 3,  # sadness
    # DIS (disgust) and FEA (fear) excluded for 4-class
}

# For 6-class mapping (if needed later)
emotion_map_6class = {
    "ANG": 0,  # anger
    "DIS": 1,  # disgust
    "FEA": 2,  # fear
    "HAP": 3,  # happiness
    "NEU": 4,  # neutral
    "SAD": 5,  # sadness
}


def parse_filename(filename):
    """
    Parse CREMA-D filename format: ActorID_Sentence_Emotion_Level.wav
    Example: 1001_DFA_ANG_XX.wav
    """
    parts = filename.replace(".wav", "").split("_")
    if len(parts) >= 3:
        actor_id = parts[0]
        sentence = parts[1]
        emotion = parts[2]
        level = parts[3] if len(parts) > 3 else "XX"
        return actor_id, sentence, emotion, level
    return None, None, None, None


def main():
    rows_4class = []
    rows_6class = []

    # Get all WAV files
    wav_files = [f for f in os.listdir(CREMAD_PATH) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} audio files")

    for filename in wav_files:
        actor_id, sentence, emotion, level = parse_filename(filename)

        if emotion is None:
            continue

        audio_path = os.path.join(CREMAD_PATH, filename)

        # For simplicity, use sentence code as text (CREMA-D doesn't have transcriptions readily available)
        # Sentence codes: IEO, TIE, IOM, IWW, TAI, MTI, IWL, ITH, DFA, ITS, TSI, WSI
        sentence_texts = {
            "IEO": "It's eleven o'clock.",
            "TIE": "That is exactly what happened.",
            "IOM": "I'm on my way to the meeting.",
            "IWW": "I wonder what this is about.",
            "TAI": "The airplane is almost full.",
            "MTI": "Maybe tomorrow it will be cold.",
            "IWL": "I would like a new alarm clock.",
            "ITH": "I think I have a doctor's appointment.",
            "DFA": "Don't forget a jacket.",
            "ITS": "I think I've seen this before.",
            "TSI": "The surface is slick.",
            "WSI": "We'll stop in a couple of minutes.",
        }
        text = sentence_texts.get(sentence, f"Sentence {sentence}")

        # 4-class (matching IEMOCAP)
        if emotion in emotion_map_4class:
            rows_4class.append([audio_path, text, emotion_map_4class[emotion]])

        # 6-class (all CREMA-D emotions)
        if emotion in emotion_map_6class:
            rows_6class.append([audio_path, text, emotion_map_6class[emotion]])

    print(f"\n4-class samples: {len(rows_4class)}")
    print(f"6-class samples: {len(rows_6class)}")

    # Shuffle and split (80% train, 10% val, 10% test)
    random.seed(42)

    for name, rows in [("4class", rows_4class), ("6class", rows_6class)]:
        random.shuffle(rows)

        n = len(rows)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train_rows = rows[:train_end]
        val_rows = rows[train_end:val_end]
        test_rows = rows[val_end:]

        # Save splits
        for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            output_path = f"{OUTPUT_DIR}/CREMAD_{name}_{split_name}.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["audio_file", "raw_text", "label"])
                writer.writerows(split_rows)
            print(f"  {name} {split_name}: {len(split_rows)} samples -> {output_path}")

        # Print class distribution
        from collections import Counter
        labels = [r[2] for r in rows]
        print(f"\n{name} class distribution:")
        for label, count in sorted(Counter(labels).items()):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
