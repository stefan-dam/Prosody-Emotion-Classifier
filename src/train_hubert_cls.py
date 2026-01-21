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
    df["dataset"] = dataset_name
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
        self.raw2common = raw2common  # dict: dataset -> mapping
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
        ds = row["dataset"]
        ds_map = self.raw2common[ds]
        key = raw if raw in ds_map else raw.lower()
        lab = ds_map[key]
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
    ap.add_argument("--datasets", nargs="+", choices=["RAVDESS","CREMAD","IEMOCAP"], help="One or more datasets to train on")
    ap.add_argument("--dataset", choices=["RAVDESS","CREMAD","IEMOCAP"], help="Deprecated: single dataset (use --datasets instead)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from_checkpoint", help="Path to a Trainer checkpoint directory to resume from")
    args = ap.parse_args()

    dataset_arg = args.datasets or ([args.dataset] if args.dataset else None)
    if not dataset_arg:
        raise ValueError("Please specify at least one dataset via --datasets (or --dataset for backward compatibility).")
    dataset_names = dataset_arg

    cfg = json.load(open(args.config))
    sr = int(cfg["sample_rate"])
    max_seconds = int(cfg.get("max_seconds", 3))
    max_steps = int(cfg.get("max_steps", -1))  # -1 means use epochs
    eval_batch_size = int(cfg.get("eval_batch_size", cfg.get("batch_size", 2)))
    eval_accumulation_steps = int(cfg.get("eval_accumulation_steps", 1))
    fp16_flag = bool(cfg.get("fp16", False))
    bf16_flag = bool(cfg.get("bf16", False))
    grad_ckpt = bool(cfg.get("gradient_checkpointing", False))
    save_steps = int(cfg.get("save_steps", 500))
    dataset_to_common, _ = load_label_maps()

    if fp16_flag and not torch.cuda.is_available():
        print("[warn] fp16 requested but CUDA is not available; disabling fp16.")
        fp16_flag = False

    # Build the label set from selected datasets unless explicitly overridden
    label_set_cfg = cfg.get("label_set", cfg.get("common_labels"))
    mapped_labels = set()
    for ds in dataset_names:
        if ds not in dataset_to_common:
            raise ValueError(f"Dataset {ds} missing from label map.")
        mapped_labels.update(dataset_to_common[ds].values())
    if label_set_cfg:
        dedup_provided = list(dict.fromkeys(label_set_cfg))
        label_set = [lab for lab in dedup_provided if lab in mapped_labels]
        unused = set(dedup_provided) - mapped_labels
        if unused:
            print(f"[info] Dropping labels not present in selected datasets: {sorted(unused)}")
    else:
        label_set = sorted(mapped_labels)
    missing_from_cfg = mapped_labels - set(label_set)
    if missing_from_cfg:
        raise ValueError(f"label_set missing classes from mapping: {missing_from_cfg}")
    num_labels = len(label_set)

    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    raw2common_map = {ds: dataset_to_common[ds] for ds in dataset_names}
    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Build concatenated train/val tables across all chosen datasets
    train_tables = []
    val_tables = []
    test_tables = []
    for ds in dataset_names:
        train_df = build_table(ds, "train")
        val_df = build_table(ds, "val")
        test_df = build_table(ds, "test")

        mapped = train_df["emotion_label"].map(lambda r: raw2common_map[ds][str(r)] if str(r) in raw2common_map[ds] else raw2common_map[ds][str(r).lower()])
        if mapped.isna().any():
            bad = train_df[mapped.isna()]
            raise ValueError(f"Unmapped labels exist in {ds}:\n{bad[['utt_id','emotion_label']].to_string(index=False)}")
        miss = set(mapped.unique()) - set(label_set)
        if miss:
            raise ValueError(f"label_set missing classes from mapping for {ds}: {miss}")

        train_tables.append(train_df)
        val_tables.append(val_df)
        test_tables.append(test_df)

    train_df = pd.concat(train_tables, ignore_index=True)
    val_df = pd.concat(val_tables, ignore_index=True)
    test_df = pd.concat(test_tables, ignore_index=True)

    extractor = AutoFeatureExtractor.from_pretrained(cfg["model_name"])
    model = AutoModelForAudioClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    train_ds = AudioDS(train_df, extractor, sr, max_seconds, {**raw2common_map}, label2id)
    val_ds = AudioDS(val_df, extractor, sr, max_seconds, {**raw2common_map}, label2id)

    dataset_tag = "+".join(dataset_names)
    out_dir = Path(f"models/{dataset_tag}_hubert_cls")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = cfg.get("evaluation_strategy", cfg.get("eval_strategy", "epoch"))
    save_strategy_cfg = cfg.get("save_strategy", "epoch")
    save_total_limit = int(cfg.get("save_total_limit", 2))
    load_best_cfg = bool(cfg.get("load_best_model_at_end", True))
    skip_save = bool(cfg.get("skip_save", False))
    skip_eval = bool(cfg.get("skip_eval", False))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        num_train_epochs=int(cfg.get("epochs", 3)),
        max_steps=max_steps,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy_cfg,
        load_best_model_at_end=load_best_cfg and save_strategy_cfg != "no",
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=fp16_flag,
        bf16=bf16_flag,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=int(cfg.get("logging_steps", 50)),
        report_to=[],
        seed=42,
        max_grad_norm=0.5,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
        save_total_limit=save_total_limit,
        save_steps=save_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        gradient_checkpointing=grad_ckpt,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    # Evaluate on test split after training with best model
    if not skip_eval:
        test_ds = AudioDS(test_df, extractor, sr, max_seconds, {**raw2common_map}, label2id)
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(f"[TEST] {test_metrics}")
    else:
        print("[INFO] Skipped evaluation (skip_eval=True)")
    if not skip_save:
        trainer.save_model(str(out_dir))
        print(f"[DONE] Saved best model to {out_dir}")
    else:
        print("[INFO] Skipped saving model (skip_save=True)")

if __name__ == "__main__":
    main()
