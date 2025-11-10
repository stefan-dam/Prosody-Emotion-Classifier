import pandas as pd
import json
import numpy as np
import os
import sys

dataset = sys.argv[1]  # e.g. "RAVDESS"
csv_path = f"data/{dataset}/metadata.csv"
df = pd.read_csv(csv_path)

speakers = df["speaker_id"].unique()
np.random.shuffle(speakers)

n = len(speakers)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train_speakers = speakers[:n_train]
val_speakers = speakers[n_train:n_train+n_val]
test_speakers = speakers[n_train+n_val:]

splits = {}
for _, row in df.iterrows():
    sid = row["speaker_id"]
    if sid in train_speakers:
        split = "train"
    elif sid in val_speakers:
        split = "val"
    else:
        split = "test"
    splits[row["utt_id"]] = split

out_path = f"configs/splits/{dataset}_splits.json"
os.makedirs("configs/splits", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Saved split file: {out_path}")
