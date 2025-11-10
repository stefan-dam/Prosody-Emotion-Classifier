import json, pandas as pd, soundfile as sf, numpy as np
from pathlib import Path

def load_split_table(dataset_name):
    split_path = Path(f"configs/splits/{dataset_name}_splits.json")
    return json.loads(split_path.read_text())

def load_label_maps():
    import json
    m1 = json.load(open("configs/label_maps/dataset_to_common_7.json"))
    m2 = json.load(open("configs/label_maps/common7_to_vad.json"))
    return m1, m2

def read_wav_16k(path, target_sr=16000):
    audio, sr = sf.read(path)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=target_sr)
    # mono
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    return audio

def build_table(dataset_name, split):
    df = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    splits = load_split_table(dataset_name)
    df["split"] = df["utt_id"].map(splits)
    df = df[df["split"]==split].copy()
    return df
