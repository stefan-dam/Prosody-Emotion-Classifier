import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from prosody_features import extract_prosody

def read_wav(path, target_sr=16000, max_seconds=None):
    x, sr = sf.read(path, always_2d=False)
    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
    if max_seconds and max_seconds > 0:
        T = int(target_sr * max_seconds)
        if len(x) > T:
            x = x[:T]
    return x

def map_label(raw, ds_map):
    raw_str = str(raw)
    if raw_str in ds_map:
        return ds_map[raw_str]
    raw_lower = raw_str.lower()
    if raw_lower in ds_map:
        return ds_map[raw_lower]
    raw_upper = raw_str.upper()
    if raw_upper in ds_map:
        return ds_map[raw_upper]
    raise KeyError(raw_str)

def load_split_table(dataset_name):
    with open(f"configs/splits/{dataset_name}_splits.json", "r") as f:
        return json.load(f)

def build_table(dataset_name, split):
    df = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    splits = load_split_table(dataset_name)
    df["split"] = df["utt_id"].map(splits)
    df = df[df["split"] == split].copy()
    df["dataset"] = dataset_name
    return df

def load_label_maps():
    with open("configs/label_maps/dataset_to_common_7.json", "r") as f:
        m1 = json.load(f)
    return m1

class ProsodyDS(Dataset):
    def __init__(self, df, sr, cfg, raw2common, label2id, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.cfg = cfg
        self.raw2common = raw2common
        self.label2id = label2id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, dataset, utt_id):
        safe = f"{dataset}_{utt_id}".replace("/", "_")
        return self.cache_dir / f"{safe}.npy"

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ds = row["dataset"]
        raw = row["emotion_label"]
        lab = map_label(raw, self.raw2common[ds])
        y = int(self.label2id[lab])

        feats = None
        if self.cache_dir:
            cache_path = self._cache_path(ds, row["utt_id"])
            if cache_path.exists():
                feats = np.load(cache_path)
        if feats is None:
            x = read_wav(row["wav_path"], target_sr=self.sr, max_seconds=self.cfg.get("max_seconds", None))
            feats = extract_prosody(
                x,
                sr=self.sr,
                hop_ms=float(self.cfg.get("hop_ms", 10.0)),
                frame_ms=float(self.cfg.get("frame_ms", 25.0)),
                f0_frame_ms=float(self.cfg.get("f0_frame_ms", 40.0)),
                fmin=float(self.cfg.get("fmin", 50.0)),
                fmax=float(self.cfg.get("fmax", 500.0)),
                pre_emphasis=float(self.cfg.get("pre_emphasis", 0.97)),
                smooth_energy=bool(self.cfg.get("smooth_energy", True)),
                sg_window=int(self.cfg.get("sg_window", 11)),
                sg_poly=int(self.cfg.get("sg_poly", 2)),
            )
            if self.cache_dir:
                np.save(cache_path, feats)

        feats_t = torch.from_numpy(feats).float()
        return {"features": feats_t, "length": feats_t.shape[0], "labels": torch.tensor(y, dtype=torch.long)}

def collate_batch(batch):
    lengths = [b["length"] for b in batch]
    max_len = int(max(lengths)) if lengths else 1
    feat_dim = int(batch[0]["features"].shape[1])
    feats = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        L = int(item["length"])
        feats[i, :L] = item["features"]
        mask[i, :L] = 1.0
        labels[i] = item["labels"]
    return {"features": feats, "mask": mask, "labels": labels}

def masked_mean_std(x, mask):
    mask = mask.unsqueeze(-1)
    lengths = mask.sum(1).clamp(min=1.0)
    mean = (x * mask).sum(1) / lengths
    var = (x * x * mask).sum(1) / lengths - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-5))
    return torch.cat([mean, std], dim=-1)

class ProsodyClassifier(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.proj = torch.nn.Linear(feat_dim * 2, hidden_dim)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, features, mask):
        pooled = masked_mean_std(features, mask)
        x = self.drop(self.act(self.proj(pooled)))
        return self.classifier(x)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "uar": float(recall_score(y_true, y_pred, average="macro")),
    }

def write_test_report(out_dir, test_df, y_true, y_pred, id2label):
    labels = list(range(len(id2label)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = {}
    for i in labels:
        denom = int(cm[i].sum())
        per_class_recall[id2label[i]] = float(cm[i, i] / denom) if denom > 0 else 0.0

    per_dataset = {}
    for ds in sorted(test_df["dataset"].unique()):
        mask = (test_df["dataset"] == ds).to_numpy()
        if mask.sum() == 0:
            continue
        ds_true = y_true[mask]
        ds_pred = y_pred[mask]
        per_dataset[ds] = {
            "accuracy": float(accuracy_score(ds_true, ds_pred)),
            "f1": float(f1_score(ds_true, ds_pred, average="macro")),
            "uar": float(recall_score(ds_true, ds_pred, average="macro")),
            "n": int(mask.sum()),
        }

    per_speaker = {}
    for speaker, group in test_df.groupby("speaker_id"):
        idx = group.index.to_numpy()
        sp_true = y_true[idx]
        sp_pred = y_pred[idx]
        per_speaker[str(speaker)] = float(accuracy_score(sp_true, sp_pred)) if len(idx) else 0.0

    report = {
        "per_class_recall": per_class_recall,
        "per_dataset": per_dataset,
        "per_speaker_accuracy": per_speaker,
        "confusion_matrix": cm.tolist(),
    }
    out_path = Path(out_dir) / "test_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[TEST] Wrote report to {out_path}")

def run_epoch(model, loader, optimizer, device, criterion, train=True):
    model.train() if train else model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    for batch in loader:
        feats = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(feats, mask)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    metrics = compute_metrics(y_true, y_pred) if len(y_true) else {"accuracy": 0.0, "f1": 0.0, "uar": 0.0}
    metrics["loss"] = total_loss / max(1, len(y_true))
    return metrics, y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=["RAVDESS", "CREMAD", "IEMOCAP"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from best_model.pt if present")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    sr = int(cfg.get("sample_rate", 16000))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_to_common = load_label_maps()

    label_set_cfg = cfg.get("label_set", cfg.get("common_labels"))
    mapped_labels = set()
    for ds in args.datasets:
        if ds not in dataset_to_common:
            raise ValueError(f"Dataset {ds} missing from label map.")
        mapped_labels.update(dataset_to_common[ds].values())
    if label_set_cfg:
        dedup_provided = list(dict.fromkeys(label_set_cfg))
        label_set = [lab for lab in dedup_provided if lab in mapped_labels]
    else:
        label_set = sorted(mapped_labels)
    missing_from_cfg = mapped_labels - set(label_set)
    if missing_from_cfg:
        raise ValueError(f"label_set missing classes from mapping: {missing_from_cfg}")

    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}
    raw2common_map = {ds: dataset_to_common[ds] for ds in args.datasets}

    train_tables, val_tables, test_tables = [], [], []
    for ds in args.datasets:
        train_tables.append(build_table(ds, "train"))
        val_tables.append(build_table(ds, "val"))
        test_tables.append(build_table(ds, "test"))
    train_df = pd.concat(train_tables, ignore_index=True)
    val_df = pd.concat(val_tables, ignore_index=True)
    test_df = pd.concat(test_tables, ignore_index=True)

    # Validate mappings across the full training table
    for ds in args.datasets:
        ds_df = train_df[train_df["dataset"] == ds]
        mapped = ds_df["emotion_label"].map(lambda r: map_label(r, raw2common_map[ds]))
        if mapped.isna().any():
            bad = ds_df[mapped.isna()]
            raise ValueError(f"Unmapped labels exist in {ds}:\n{bad[['utt_id','emotion_label']].to_string(index=False)}")

    cache_dir = cfg.get("cache_dir", None)
    train_ds = ProsodyDS(train_df, sr, cfg, raw2common_map, label2id, cache_dir=cache_dir)
    val_ds = ProsodyDS(val_df, sr, cfg, raw2common_map, label2id, cache_dir=cache_dir)
    test_ds = ProsodyDS(test_df, sr, cfg, raw2common_map, label2id, cache_dir=cache_dir)

    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)

    inferred_dim = int(train_ds[0]["features"].shape[1])
    feat_dim = int(cfg.get("feature_dim", inferred_dim))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    dropout = float(cfg.get("dropout", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProsodyClassifier(feat_dim, hidden_dim, len(label_set), dropout).to(device)

    class_weights = None
    if bool(cfg.get("use_class_weights", True)):
        counts = np.bincount([label2id[map_label(r, raw2common_map[d])] for d, r in zip(train_df["dataset"], train_df["emotion_label"])], minlength=len(label_set))
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset_tag = "+".join(args.datasets)
    out_dir = Path(f"models/{dataset_tag}_prosody_cls")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2))
    (out_dir / "id2label.json").write_text(json.dumps(id2label, indent=2))

    best_path = out_dir / "best_model.pt"
    best_f1 = -1.0
    if args.resume and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Resumed from {best_path}")

    epochs = int(cfg.get("epochs", 50))
    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, device, criterion, train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, optimizer, device, criterion, train=False)
        print(f"[EPOCH {epoch}] train={train_metrics} val={val_metrics}")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, y_true, y_pred = run_epoch(model, test_loader, optimizer, device, criterion, train=False)
    print(f"[TEST] {test_metrics}")
    write_test_report(out_dir, test_df, y_true, y_pred, id2label)

if __name__ == "__main__":
    main()
