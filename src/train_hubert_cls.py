import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments, set_seed

def read_wav_fixed(path, target_sr=16000, max_seconds=3):
    x, sr = sf.read(path, always_2d=False)
    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    T = int(target_sr * max_seconds)
    if len(x) >= T:
        x = x[:T]
    else:
        x = np.pad(x, (0, T - len(x)), mode="constant")
    return x

def load_split_table(dataset_name):
    with open(f"configs/splits/{dataset_name}_splits.json", "r") as f:
        return json.load(f)

def build_table(dataset_name, split):
    df = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    splits = load_split_table(dataset_name)
    df["split"] = df["utt_id"].map(splits)
    df = df[df["split"] == split].copy()
    return df

def load_label_maps():
    with open("configs/label_maps/dataset_to_common_7.json", "r") as f:
        m1 = json.load(f)
    with open("configs/label_maps/common7_to_vad.json", "r") as f:
        m2 = json.load(f)
    return m1, m2

class AudioDS(Dataset):
    def __init__(self, df, extractor, sr, max_seconds, raw2common, label2id):
        self.df = df.reset_index(drop=True)
        self.ext = extractor
        self.sr = sr
        self.max_seconds = max_seconds
        self.raw2common = raw2common
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = read_wav_fixed(row["wav_path"], target_sr=self.sr, max_seconds=self.max_seconds)
        feats = self.ext(x, sampling_rate=self.sr, return_tensors="pt")
        iv = feats["input_values"][0]
        am = feats.get("attention_mask", None)
        if am is None:
            am = torch.ones_like(iv, dtype=torch.long)
        raw = str(row["emotion_label"])
        key = raw if raw in self.raw2common else raw.lower()
        lab = self.raw2common[key]
        y = int(self.label2id[lab])
        return {"input_values": iv, "attention_mask": am, "labels": torch.tensor(y, dtype=torch.long)}

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    y_pred = preds.argmax(-1)
    return {"accuracy": accuracy_score(labels, y_pred), "f1": f1_score(labels, y_pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["RAVDESS","CREMAD","IEMOCAP"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    sr = int(cfg["sample_rate"])
    max_seconds = int(cfg.get("max_seconds", 3))
    label_set = cfg.get("label_set", cfg.get("common_labels"))
    num_labels = len(label_set)

    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_to_common, _ = load_label_maps()
    raw2common = dataset_to_common[args.dataset]
    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_df = build_table(args.dataset, "train")
    val_df = build_table(args.dataset, "val")

    mapped = train_df["emotion_label"].map(lambda r: raw2common[str(r)] if str(r) in raw2common else raw2common[str(r).lower()])
    if mapped.isna().any():
        bad = train_df[mapped.isna()]
        raise ValueError(f"Unmapped labels exist:\n{bad[['utt_id','emotion_label']].to_string(index=False)}")
    miss = set(mapped.unique()) - set(label_set)
    if miss:
        raise ValueError(f"label_set missing classes from mapping: {miss}")

    extractor = AutoFeatureExtractor.from_pretrained(cfg["model_name"])
    model = AutoModelForAudioClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    train_ds = AudioDS(train_df, extractor, sr, max_seconds, raw2common, label2id)
    val_ds = AudioDS(val_df, extractor, sr, max_seconds, raw2common, label2id)

    out_dir = Path(f"models/{args.dataset}_hubert_cls")
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 2)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        num_train_epochs=int(cfg.get("epochs", 3)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=1,
        report_to=[],
        seed=42,
        max_grad_norm=0.5,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[DONE] Saved best model to {out_dir}")

if __name__ == "__main__":
    main()
