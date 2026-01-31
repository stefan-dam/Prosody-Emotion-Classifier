import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

import train_hubert_cls as thc


def _set_low_resource_mode():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def _build_eval_df(datasets, split, label2id, seed, max_samples):
    dataset_to_common, _ = thc.load_label_maps()
    tables = []
    for ds in datasets:
        df = thc.build_table(ds, split)

        def _safe_map(x):
            try:
                return thc.map_label(x, dataset_to_common[ds])
            except KeyError:
                return None

        mapped = df["emotion_label"].map(_safe_map)
        df = df[mapped.notna()].copy()
        df["common_label"] = mapped[mapped.notna()]
        df = df[df["common_label"].isin(label2id)].copy()
        df["dataset"] = ds
        tables.append(df)

    if not tables:
        return pd.DataFrame(), dataset_to_common

    eval_df = pd.concat(tables, ignore_index=True)

    if max_samples and max_samples > 0 and len(eval_df) > max_samples:
        eval_df = eval_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    return eval_df.reset_index(drop=True), dataset_to_common


def _collate(batch):
    if not batch:
        return {}
    input_values = torch.stack([b["input_values"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}


def _compute_metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    total = cm.sum()
    correct = np.trace(cm)
    acc = float(correct / total) if total > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(np.diag(cm), cm.sum(axis=1))
        precision = np.divide(np.diag(cm), cm.sum(axis=0))
        recall = np.nan_to_num(recall, nan=0.0, posinf=0.0, neginf=0.0)
        precision = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)
        f1 = np.divide(2 * precision * recall, precision + recall)
        f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

    uar = float(recall.mean()) if recall.size else 0.0
    macro_f1 = float(f1.mean()) if f1.size else 0.0

    return acc, macro_f1, uar, recall, f1


def _load_feature_extractor(model_dir: Path, override_name: str = None):
    try:
        return AutoFeatureExtractor.from_pretrained(str(model_dir))
    except OSError as e:
        msg = str(e)
        if "preprocessor_config.json" not in msg:
            raise

    fallback_name = override_name
    if not fallback_name:
        try:
            cfg = AutoConfig.from_pretrained(str(model_dir))
            fallback_name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        except Exception:
            fallback_name = None

    if not fallback_name:
        raise OSError(
            "Missing preprocessor_config.json and no fallback model name found. "
            "Pass --feature_extractor with the base model name (e.g., superb/hubert-large-superb-er)."
        )

    extractor = AutoFeatureExtractor.from_pretrained(fallback_name)
    try:
        extractor.save_pretrained(str(model_dir))
    except Exception:
        pass
    return extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, choices=["RAVDESS", "CREMAD", "IEMOCAP"])
    parser.add_argument("--model_dir", required=True, help="Path to a saved model directory or checkpoint")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=50, help="Limit evaluation to N samples for low-resource runs")
    parser.add_argument("--max_seconds", type=int, default=2, help="Pad/truncate audio to this many seconds")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_extractor", default=None, help="Fallback base model name for feature extractor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", default="eval_report.json")
    args = parser.parse_args()

    _set_low_resource_mode()

    device = torch.device(args.device)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    extractor = _load_feature_extractor(model_dir, args.feature_extractor)
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)

    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)
    if label2id is None and id2label is None:
        raise ValueError("Model config missing label mappings (label2id/id2label).")

    if label2id is None and id2label is not None:
        label2id = {label: int(idx) for idx, label in id2label.items()}

    label2id = {str(k): int(v) for k, v in label2id.items()}
    labels = [None] * len(label2id)
    for lab, idx in label2id.items():
        if 0 <= idx < len(labels):
            labels[idx] = lab
    if any(l is None for l in labels):
        raise ValueError("Non-contiguous label2id mapping; cannot infer label order.")

    eval_df, raw2common = _build_eval_df(
        datasets=args.datasets,
        split=args.split,
        label2id=set(labels),
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if eval_df.empty:
        raise ValueError("No evaluation samples found after filtering. Check datasets/splits.")

    sr = int(extractor.sampling_rate or 16000)
    eval_ds = thc.AudioDS(
        eval_df,
        extractor,
        sr,
        int(args.max_seconds),
        raw2common,
        {lab: i for i, lab in enumerate(labels)},
        random_crop=False,
        augment_cfg=None,
        training=False,
    )

    loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=_collate,
    )

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            if not batch:
                continue
            labels_t = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_t = labels_t.to(device)
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)

            y_true = labels_t.detach().cpu().numpy().astype(np.int64)
            y_pred = preds.detach().cpu().numpy().astype(np.int64)

            for t, p in zip(y_true, y_pred):
                if 0 <= t < cm.shape[0] and 0 <= p < cm.shape[1]:
                    cm[t, p] += 1
            total += len(y_true)

    acc, macro_f1, uar, recall, f1 = _compute_metrics_from_cm(cm)

    result = {
        "split": args.split,
        "num_samples": int(total),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "uar": uar,
        "labels": labels,
        "per_class_recall": {labels[i]: float(recall[i]) for i in range(len(labels))},
        "per_class_f1": {labels[i]: float(f1[i]) for i in range(len(labels))},
        "confusion_matrix": cm.tolist(),
    }

    out_path = model_dir / args.out_json
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
