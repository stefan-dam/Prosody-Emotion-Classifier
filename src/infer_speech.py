import json, torch, numpy as np, soundfile as sf
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

SR = 16000
MODEL_DIR = "models/RAVDESS_hubert_cls"
LABELS = ["neutral","happy","angry","sad","disgust","fear","excited"]

def resample_to_16k(x, sr):
    if sr == SR:
        return x
    import librosa
    y = librosa.resample(y=x.astype(np.float32, copy=False), orig_sr=sr, target_sr=SR)
    return y

def read16k(path):
    x, sr = sf.read(path, always_2d=False)

    if hasattr(x, "ndim") and x.ndim > 1:
        x = np.mean(x, axis=1)

    x = x.astype(np.float32, copy=False)

    x = resample_to_16k(x, sr)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    max_len = SR * 6
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = np.pad(x, (0, max_len - len(x)), mode="constant")
    return x.astype(np.float32, copy=False)

extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR, sampling_rate=SR)
model = HubertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

@torch.no_grad()
def infer(wav_path: str):
    x = read16k(wav_path)
    feats = extractor(x, sampling_rate=SR, return_tensors="pt")
    feats = {k: v.to(dtype=torch.float32, device="cpu") for k, v in feats.items()}

    logits = model(**feats).logits
    logits_np = logits.cpu().numpy()[0]

    if not np.isfinite(logits_np).all():
        logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()

    out = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top = LABELS[int(np.argmax(probs))]

    vad_map = json.load(open("configs/label_maps/common7_to_vad.json"))
    vad = vad_map[top]
    return {"pred_label": top, "emotion_probs": out, "vad": vad}

if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 2:
        print("Usage: python src/infer_speech.py <path_to_wav>")
        sys.exit(1)
    pprint.pprint(infer(sys.argv[1]))
