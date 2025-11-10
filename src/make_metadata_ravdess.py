import os
import csv

root = "data/RAVDESS"
out_csv = "data/RAVDESS/metadata.csv"

emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

rows = []
for actor in os.listdir(root):
    actor_path = os.path.join(root, actor)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")
            emo = emotion_map[parts[2]]
            speaker = actor
            utt_id = f"{actor}_{file[:-4]}"
            wav_path = os.path.join(actor_path, file)
            rows.append([utt_id, wav_path, speaker, emo])

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["utt_id","wav_path","speaker_id","emotion_label"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {out_csv}")
