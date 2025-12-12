import os
import csv
import re

IEMOCAP_PATH = r"C:\Users\Maftuna\IEMOCAP"
OUTPUT_CSV = r"C:\Users\Maftuna\singyu_model\recognition\Multimodal-Speech-Emotion-Recognition-main\Multimodal-Speech-Emotion-Recognition-main\metadata\IEMOCAP_metadata.csv"

emotion_map = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3
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
                    utt_id = parts[1]
                    emo = parts[2]

                    if emo in emotion_map:
                        emotions[utt_id] = emotion_map[emo]

    return emotions


def main():
    rows = []

    for sess in os.listdir(IEMOCAP_PATH):
        if not sess.startswith("Session"):
            continue

        session_path = os.path.join(IEMOCAP_PATH, sess)
        
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
                        audio_path.replace("\\", "/"),
                        transcriptions[utt_id],
                        emotions[utt_id],
                    ])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["audio_file", "raw_text", "label"])
        writer.writerows(rows)

    print(f"Metadata saved to {OUTPUT_CSV}. Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
