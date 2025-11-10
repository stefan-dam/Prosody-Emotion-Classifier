import os, csv

root = "data/CREMAD"
out_csv = "data/CREMAD/metadata.csv"

rows = []
for file in os.listdir(root):
    if not file.endswith(".wav"): continue
    parts = file.split("_")
    speaker = parts[0]
    emo = parts[2]
    utt_id = file[:-4]
    wav_path = os.path.join(root, file)
    rows.append([utt_id, wav_path, speaker, emo])

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["utt_id","wav_path","speaker_id","emotion_label"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {out_csv}")
