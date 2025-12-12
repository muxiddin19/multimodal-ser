import pandas as pd
import os

# Path to your full metadata CSV
INPUT_CSV = r"C:\Users\Maftuna\singyu_model\recognition\Multimodal-Speech-Emotion-Recognition-main\Multimodal-Speech-Emotion-Recognition-main\metadata\IEMOCAP_metadata.csv"

# Output files
OUTPUT_DIR = os.path.dirname(INPUT_CSV)
TRAIN_CSV = os.path.join(OUTPUT_DIR, "IEMOCAP_metadata_train.csv")
VAL_CSV   = os.path.join(OUTPUT_DIR, "IEMOCAP_metadata_val.csv")
TEST_CSV  = os.path.join(OUTPUT_DIR, "IEMOCAP_metadata_test.csv")

df = pd.read_csv(INPUT_CSV)

# Extract session number from the audio paths
df["session"] = df["audio_file"].str.extract(r"Session(\d)")

# Split using standard SER protocol
train_df = df[df["session"].isin(["1", "2", "3"])]
val_df   = df[df["session"] == "4"]
test_df  = df[df["session"] == "5"]

# Save the splits
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("Saved:")
print(f" Train: {len(train_df)} -> {TRAIN_CSV}")
print(f" Val:   {len(val_df)} -> {VAL_CSV}")
print(f" Test:  {len(test_df)} -> {TEST_CSV}")
