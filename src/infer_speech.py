import argparse, json, torch, numpy as np, soundfile as sf
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.signal import resample_poly

DEFAULT_VAD = "configs/label_maps/common7_to_vad.json"
DEFAULT_MAX_SECONDS = 6
CPU_DEVICE = "cpu"

def resample_audio(x, sr, target_sr):
    if sr == target_sr:
        return x
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(x.astype(np.float32, copy=False), up=up, down=down)

def read_audio(path, target_sr, max_seconds=DEFAULT_MAX_SECONDS):
    x, sr = sf.read(path, always_2d=False)

    if hasattr(x, "ndim") and x.ndim > 1:
        x = np.mean(x, axis=1)

    x = x.astype(np.float32, copy=False)

    x = resample_audio(x, sr, target_sr)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    max_len = int(target_sr * max_seconds)
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = np.pad(x, (0, max_len - len(x)), mode="constant")
    return x.astype(np.float32, copy=False)

def load_labels(model):
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # keys may be str, ensure index order
        labels = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
        return labels
    label2id = getattr(model.config, "label2id", None)
    if label2id:
        ordered = sorted(label2id.items(), key=lambda kv: kv[1])
        return [k for k, _ in ordered]
    raise ValueError("No label mapping found in model config.")

@torch.no_grad()
def infer(wav_path: str, model_dir: str, device: str = CPU_DEVICE, max_seconds: int = DEFAULT_MAX_SECONDS, vad_map_path: str = DEFAULT_VAD):
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    sr = extractor.sampling_rate or 16000
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    labels = load_labels(model)

    x = read_audio(wav_path, target_sr=sr, max_seconds=max_seconds)
    feats = extractor(x, sampling_rate=sr, return_tensors="pt")
    feats = {k: v.to(dtype=torch.float32, device=device) for k, v in feats.items()}

    logits = model(**feats).logits
    logits_np = logits.detach().cpu().numpy()[0]

    if not np.isfinite(logits_np).all():
        logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()

    out = {labels[i]: float(probs[i]) for i in range(len(labels))}
    top = labels[int(np.argmax(probs))]

    vad_map = json.load(open(vad_map_path))
    vad = vad_map.get(top)
    return {"pred_label": top, "emotion_probs": out, "vad": vad}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to WAV file for inference")
    parser.add_argument("--model_dir", default="models/RAVDESS_hubert_cls", help="Path to trained model directory")
    parser.add_argument("--device", default=CPU_DEVICE, help="Device to run inference on (e.g., cpu or cuda)")
    parser.add_argument("--max_seconds", type=int, default=DEFAULT_MAX_SECONDS, help="Pad/truncate audio to this length")
    parser.add_argument("--vad_map", default=DEFAULT_VAD, help="Path to VAD label map JSON")
    args = parser.parse_args()

    wav_path = Path(args.audio_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    result = infer(str(wav_path), model_dir=args.model_dir, device=args.device, max_seconds=args.max_seconds, vad_map_path=args.vad_map)
    import pprint
    pprint.pprint(result)
