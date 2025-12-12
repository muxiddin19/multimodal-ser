"""
Create 6-class IEMOCAP metadata
Classes: happiness, sadness, neutral, anger, excitement, frustration
"""
import os
import csv
import re

IEMOCAP_PATH = "/nas/dataset/IEMOCAP"
OUTPUT_CSV = "metadata/IEMOCAP_6class_metadata.csv"

# 6-class emotion mapping (as per paper)
emotion_map_6 = {
    "hap": 0,  # happiness
    "sad": 1,  # sadness
    "neu": 2,  # neutral
    "ang": 3,  # anger
    "exc": 4,  # excitement
    "fru": 5,  # frustration
}

def read_transcriptions(session_path):
    transcriptions = {}
    tran_dir = os.path.join(session_path, "dialog", "transcriptions")

    for fname in os.listdir(tran_dir):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(tran_dir, fname), "r", errors="ignore") as f:
            for line in f:
                match = re.match(r"(.+?)\s+(.+)", line.strip())
                if match:
                    utt_id = match.group(1)
                    text = match.group(2)
                    transcriptions[utt_id] = text

    return transcriptions


def read_emotions(session_path):
    emotions = {}
    eval_dir = os.path.join(session_path, "dialog", "EmoEvaluation")

    for fname in os.listdir(eval_dir):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(eval_dir, fname), "r", errors="ignore") as f:
            for line in f:
                if line.startswith("["):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        utt_id = parts[1]
                        emo = parts[2]

                        if emo in emotion_map_6:
                            emotions[utt_id] = emotion_map_6[emo]

    return emotions


def get_session_number(session_name):
    """Extract session number from session folder name"""
    match = re.search(r'Session(\d+)', session_name)
    return int(match.group(1)) if match else 0


def main():
    rows = []

    for sess in sorted(os.listdir(IEMOCAP_PATH)):
        if not sess.startswith("Session"):
            continue

        session_path = os.path.join(IEMOCAP_PATH, sess)
        session_num = get_session_number(sess)

        transcriptions = read_transcriptions(session_path)
        emotions = read_emotions(session_path)

        wav_dir = os.path.join(session_path, "sentences", "wav")

        for root, _, files in os.walk(wav_dir):
            for f in files:
                if not f.endswith(".wav"):
                    continue

                utt_id = f.replace(".wav", "")
                audio_path = os.path.join(root, f)

                if utt_id in transcriptions and utt_id in emotions:
                    rows.append([
                        audio_path,
                        transcriptions[utt_id],
                        emotions[utt_id],
                        session_num
                    ])

    # Save full metadata
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["audio_file", "raw_text", "label", "session"])
        writer.writerows(rows)

    print(f"Full metadata saved to {OUTPUT_CSV}. Total rows: {len(rows)}")

    # Print class distribution
    from collections import Counter
    labels = [r[2] for r in rows]
    print("\nClass distribution:")
    label_names = {0: "happiness", 1: "sadness", 2: "neutral", 3: "anger", 4: "excitement", 5: "frustration"}
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label_names[label]}: {count}")

    # Split by session (standard protocol: 1-3 train, 4 val, 5 test)
    train_rows = [r for r in rows if r[3] in [1, 2, 3]]
    val_rows = [r for r in rows if r[3] == 4]
    test_rows = [r for r in rows if r[3] == 5]

    # Save splits
    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        output_path = f"metadata/IEMOCAP_6class_{split_name}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["audio_file", "raw_text", "label", "session"])
            writer.writerows(split_rows)
        print(f"{split_name}: {len(split_rows)} samples saved to {output_path}")


if __name__ == "__main__":
    main()
