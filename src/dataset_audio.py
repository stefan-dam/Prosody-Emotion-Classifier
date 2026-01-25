import json, pandas as pd, soundfile as sf, numpy as np
from pathlib import Path
from scipy.signal import resample_poly

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
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio.astype(float), up=up, down=down)
    # mono
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    return audio

def build_table(dataset_name, split):
    df = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    splits = load_split_table(dataset_name)
    df["split"] = df["utt_id"].map(splits)
    df = df[df["split"]==split].copy()
    return df
