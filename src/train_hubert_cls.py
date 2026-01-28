import argparse, json, os, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments, set_seed, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def _apply_gain(x, rng, gain_db_range):
    if gain_db_range <= 0.0:
        return x
    gain_db = rng.uniform(-gain_db_range, gain_db_range)
    gain = 10 ** (gain_db / 20.0)
    return x * gain

def _apply_noise(x, rng, snr_db):
    if snr_db <= 0.0:
        return x
    sig_power = float(np.mean(x * x))
    if sig_power <= 0.0:
        return x
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32, copy=False)
    return x + noise

def read_wav_fixed(path, target_sr=16000, max_seconds=3, random_crop=False, augment_cfg=None, rng=None):
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
    if len(x) > T and random_crop:
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, len(x) - T + 1))
        x = x[start:start + T]
    if augment_cfg:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < float(augment_cfg.get("gain_prob", 0.0)):
            x = _apply_gain(x, rng, float(augment_cfg.get("gain_db_range", 3.0)))
        if rng.random() < float(augment_cfg.get("noise_prob", 0.0)):
            x = _apply_noise(x, rng, float(augment_cfg.get("noise_snr_db", 25.0)))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    valid = len(x)
    if len(x) >= T:
        x = x[:T]
        valid = T
    else:
        x = np.pad(x, (0, T - len(x)), mode="constant")
    return x, int(valid)

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
    if df["split"].isna().any():
        missing = df[df["split"].isna()][["utt_id", "wav_path", "speaker_id"]]
        raise ValueError(f"Missing split assignment for {dataset_name}:\n{missing.to_string(index=False)}")
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
    def __init__(self, df, extractor, sr, max_seconds, raw2common, label2id, random_crop=False, augment_cfg=None, training=False):
        self.df = df.reset_index(drop=True)
        self.ext = extractor
        self.sr = sr
        self.max_seconds = max_seconds
        self.raw2common = raw2common  # dict: dataset -> mapping
        self.label2id = label2id
        self.random_crop = bool(random_crop) and bool(training)
        self.augment_cfg = augment_cfg if training else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x, valid_len = read_wav_fixed(
            row["wav_path"],
            target_sr=self.sr,
            max_seconds=self.max_seconds,
            random_crop=self.random_crop,
            augment_cfg=self.augment_cfg,
            rng=None,
        )
        feats = self.ext(x, sampling_rate=self.sr, return_tensors="pt")
        iv = feats["input_values"][0]
        am = feats.get("attention_mask", None)
        if am is None:
            am = torch.zeros_like(iv, dtype=torch.long)
            valid = min(int(valid_len), int(iv.shape[0]))
            if valid > 0:
                am[:valid] = 1
        else:
            am = am[0]
        raw = str(row["emotion_label"])
        ds = row["dataset"]
        ds_map = self.raw2common[ds]
        lab = map_label(raw, ds_map)
        y = int(self.label2id[lab])
        return {"input_values": iv, "attention_mask": am, "labels": torch.tensor(y, dtype=torch.long)}

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    y_pred = preds.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, y_pred),
        "f1": f1_score(labels, y_pred, average="macro"),
        "uar": recall_score(labels, y_pred, average="macro"),
    }

def _log_line(log_path, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a") as f:
        f.write(line + "\n")

class FileLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_train_begin(self, args, state, control, **kwargs):
        _log_line(self.log_path, "train_begin")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step={state.global_step}"]
        if state.epoch is not None:
            parts.append(f"epoch={state.epoch:.4f}")
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.6f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.6g}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.6f}")
        if "eval_accuracy" in logs:
            parts.append(f"eval_accuracy={logs['eval_accuracy']:.6f}")
        if "eval_f1" in logs:
            parts.append(f"eval_f1={logs['eval_f1']:.6f}")
        if "eval_uar" in logs:
            parts.append(f"eval_uar={logs['eval_uar']:.6f}")
        _log_line(self.log_path, " ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        _log_line(self.log_path, "train_end")

class BestCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, metric_key, keep_best_n, maximize=True, log_path=None):
        self.output_dir = output_dir
        self.metric_key = metric_key
        self.keep_best_n = keep_best_n
        self.maximize = maximize
        self.log_path = log_path
        self.last_metrics = None
        self.last_train_loss = None
        self.best = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.last_metrics = metrics or {}

    def on_save(self, args, state, control, **kwargs):
        if self.metric_key in ("train_loss", "loss"):
            if self.last_train_loss is None:
                return
            metric_val = float(self.last_train_loss)
        else:
            if not self.last_metrics or self.metric_key not in self.last_metrics:
                return
            metric_val = float(self.last_metrics[self.metric_key])
        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self.best.append((metric_val, ckpt_dir))
        self.best = sorted(self.best, key=lambda x: x[0], reverse=self.maximize)
        while len(self.best) > self.keep_best_n:
            _, rm_path = self.best.pop(-1)
            if os.path.isdir(rm_path):
                shutil.rmtree(rm_path)
        if self.log_path:
            _log_line(self.log_path, f"checkpoint={ckpt_dir} {self.metric_key}={metric_val:.6f}")

class LossCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, log_path=None, filename="train_loss.json"):
        self.output_dir = output_dir
        self.log_path = log_path
        self.filename = filename
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])

    def on_save(self, args, state, control, **kwargs):
        if self.last_train_loss is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        meta = {"step": int(state.global_step), "train_loss": float(self.last_train_loss)}
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        (Path(ckpt_dir) / self.filename).write_text(json.dumps(meta, indent=2))
        if self.log_path:
            _log_line(self.log_path, f"checkpoint={ckpt_dir} train_loss={self.last_train_loss:.6f}")

def _freeze_feature_extractor(model):
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        return True
    hubert = getattr(model, "hubert", None)
    if hubert is not None and hasattr(hubert, "feature_extractor"):
        for p in hubert.feature_extractor.parameters():
            p.requires_grad = False
        return True
    return False

def _unfreeze_feature_extractor(model):
    hubert = getattr(model, "hubert", None)
    if hubert is not None and hasattr(hubert, "feature_extractor"):
        for p in hubert.feature_extractor.parameters():
            p.requires_grad = True
        return True
    return False

class FeatureExtractorFreezeCallback(TrainerCallback):
    def __init__(self, model, freeze_steps, log_path=None):
        self.model = model
        self.freeze_steps = int(freeze_steps)
        self.log_path = log_path
        self.frozen = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.freeze_steps > 0 and _freeze_feature_extractor(self.model):
            self.frozen = True
            if self.log_path:
                _log_line(self.log_path, f"feature_extractor_frozen_until_step={self.freeze_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.frozen and state.global_step >= self.freeze_steps:
            if _unfreeze_feature_extractor(self.model):
                self.frozen = False
                if self.log_path:
                    _log_line(self.log_path, f"feature_extractor_unfrozen_at_step={state.global_step}")

def build_common_labels(df, raw2common_map, label2id):
    labels = []
    for _, row in df.iterrows():
        ds = row["dataset"]
        lab = map_label(row["emotion_label"], raw2common_map[ds])
        labels.append(int(label2id[lab]))
    return np.asarray(labels, dtype=np.int64)

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

def validate_tables(train_df, val_df, test_df, raw2common_map, label_set):
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df.empty:
            raise ValueError(f"{split_name} split is empty.")
        if df["wav_path"].isna().any():
            raise ValueError(f"{split_name} split has missing wav_path values.")
        missing_files = df[~df["wav_path"].map(lambda p: Path(p).exists())]
        if not missing_files.empty:
            raise ValueError(f"{split_name} split has missing audio files:\n{missing_files[['utt_id','wav_path']].to_string(index=False)}")
        if df["emotion_label"].isna().any():
            raise ValueError(f"{split_name} split has missing emotion_label values.")
        for ds in df["dataset"].unique():
            ds_map = raw2common_map[ds]
            mapped = df[df["dataset"] == ds]["emotion_label"].map(lambda r: map_label(r, ds_map))
            if mapped.isna().any():
                bad = df[df["dataset"] == ds][mapped.isna()]
                raise ValueError(f"Unmapped labels in {ds} ({split_name}):\n{bad[['utt_id','emotion_label']].to_string(index=False)}")
            miss = set(mapped.unique()) - set(label_set)
            if miss:
                raise ValueError(f"label_set missing classes from mapping for {ds} ({split_name}): {miss}")

    # Verify speaker-disjoint splits per dataset to avoid leakage.
    for ds in sorted(set(train_df["dataset"]).union(val_df["dataset"]).union(test_df["dataset"])):
        tr = set(train_df[train_df["dataset"] == ds]["speaker_id"].unique())
        va = set(val_df[val_df["dataset"] == ds]["speaker_id"].unique())
        te = set(test_df[test_df["dataset"] == ds]["speaker_id"].unique())
        if tr & va or tr & te or va & te:
            raise ValueError(f"Speaker leakage detected within {ds} splits.")

def log_label_distributions(df, raw2common_map, label_set, split_name, log_path):
    overall = {lab: 0 for lab in label_set}
    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        mapped = ds_df["emotion_label"].map(lambda r: map_label(r, raw2common_map[ds]))
        dist = {k: int(v) for k, v in mapped.value_counts().to_dict().items()}
        for lab, cnt in dist.items():
            overall[lab] += int(cnt)
        _log_line(log_path, f"{split_name}_label_dist_{ds}={json.dumps(dist, sort_keys=True)}")
    overall = {k: int(v) for k, v in overall.items()}
    _log_line(log_path, f"{split_name}_label_dist_all={json.dumps(overall, sort_keys=True)}")

def balance_datasets(df, seed):
    rng = np.random.default_rng(seed)
    sizes = df["dataset"].value_counts()
    max_size = int(sizes.max())
    balanced = []
    for ds, size in sizes.items():
        ds_df = df[df["dataset"] == ds]
        if size < max_size:
            sample_idx = rng.choice(ds_df.index, size=max_size, replace=True)
            ds_df = df.loc[sample_idx]
        balanced.append(ds_df)
    return pd.concat(balanced, ignore_index=True)

def _json_safe_metrics(metrics):
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


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
    grad_accum_steps = int(cfg.get("gradient_accumulation_steps", 1))
    seed = int(cfg.get("seed", 42))
    log_path = cfg.get("log_path", "train.log")
    train_fraction = float(cfg.get("train_fraction", 1.0))
    balance_flag = bool(cfg.get("balance_datasets", False))
    keep_best_n = int(cfg.get("keep_best_n", 3))
    checkpoint_metric = cfg.get("checkpoint_metric", "eval_accuracy")
    random_crop = bool(cfg.get("random_crop", False))
    use_augment = bool(cfg.get("use_augment", False))
    augment_cfg = cfg.get("augment", {}) if use_augment else None
    freeze_steps = int(cfg.get("freeze_feature_extractor_steps", 0))
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

    set_seed(seed)
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

        def _safe_map(x):
            try:
                return map_label(x, raw2common_map[ds])
            except KeyError:
                return np.nan
        mapped = train_df["emotion_label"].map(_safe_map)
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
    if balance_flag:
        train_df = balance_datasets(train_df, seed=seed)
    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError("train_fraction must be in (0.0, 1.0].")
    if train_fraction < 1.0:
        sampled = []
        for ds in sorted(train_df["dataset"].unique()):
            ds_df = train_df[train_df["dataset"] == ds]
            n_keep = max(1, int(round(len(ds_df) * train_fraction)))
            sampled.append(ds_df.sample(n=n_keep, random_state=seed))
        train_df = pd.concat(sampled, ignore_index=True)

    validate_tables(train_df, val_df, test_df, raw2common_map, label_set)
    _log_line(log_path, f"train_size={len(train_df)} val_size={len(val_df)} test_size={len(test_df)} balance_datasets={balance_flag} train_fraction={train_fraction}")
    log_label_distributions(train_df, raw2common_map, label_set, "train", log_path)
    log_label_distributions(val_df, raw2common_map, label_set, "val", log_path)
    log_label_distributions(test_df, raw2common_map, label_set, "test", log_path)

    extractor = AutoFeatureExtractor.from_pretrained(cfg["model_name"])
    model_config = AutoConfig.from_pretrained(cfg["model_name"])
    if "use_weighted_layer_sum" in cfg:
        model_config.use_weighted_layer_sum = bool(cfg["use_weighted_layer_sum"])
    if "classifier_proj_size" in cfg:
        model_config.classifier_proj_size = int(cfg["classifier_proj_size"])
    if "apply_spec_augment" in cfg:
        model_config.apply_spec_augment = bool(cfg["apply_spec_augment"])
    for key in [
        "mask_time_prob",
        "mask_time_length",
        "mask_time_min_masks",
        "mask_feature_prob",
        "mask_feature_length",
        "mask_feature_min_masks",
        "layerdrop",
        "hidden_dropout",
        "attention_dropout",
        "activation_dropout",
    ]:
        if key in cfg:
            setattr(model_config, key, cfg[key])

    model_config.num_labels = num_labels
    model_config.label2id = label2id
    model_config.id2label = id2label
    model = AutoModelForAudioClassification.from_pretrained(
        cfg["model_name"],
        config=model_config,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    train_ds = AudioDS(
        train_df,
        extractor,
        sr,
        max_seconds,
        {**raw2common_map},
        label2id,
        random_crop=random_crop,
        augment_cfg=augment_cfg,
        training=True,
    )
    val_ds = AudioDS(val_df, extractor, sr, max_seconds, {**raw2common_map}, label2id)

    dataset_tag = "+".join(dataset_names)
    run_name = cfg.get("run_name")
    out_dir_cfg = cfg.get("output_dir")
    if out_dir_cfg:
        out_dir = Path(out_dir_cfg)
    else:
        base_out_dir = Path(f"models/{dataset_tag}_hubert_cls")
        out_dir = Path(f"{base_out_dir}_{run_name}") if run_name else base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = cfg.get("evaluation_strategy", cfg.get("eval_strategy", "steps"))
    save_strategy_cfg = cfg.get("save_strategy", "steps")
    load_best_cfg = bool(cfg.get("load_best_model_at_end", True))
    skip_save = bool(cfg.get("skip_save", False))
    skip_eval = bool(cfg.get("skip_eval", False))
    eval_steps = int(cfg.get("eval_steps", cfg.get("save_steps", 500)))
    if skip_eval or str(eval_strategy).lower() == "no":
        eval_strategy = "no"
        skip_eval = True
        load_best_cfg = False
        if checkpoint_metric.startswith("eval_"):
            checkpoint_metric = "train_loss"
    if keep_best_n > 0 and save_strategy_cfg == "no":
        raise ValueError("keep_best_n>0 requires save_strategy != 'no'.")
    if keep_best_n > 0:
        if checkpoint_metric not in ("train_loss", "loss") and eval_strategy != save_strategy_cfg:
            raise ValueError("When keep_best_n>0, evaluation_strategy and save_strategy must match.")
        if checkpoint_metric not in ("train_loss", "loss") and eval_strategy == "steps" and save_steps != eval_steps:
            raise ValueError("When keep_best_n>0, save_steps must equal eval_steps for aligned checkpoint metrics.")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        num_train_epochs=int(cfg.get("epochs", 3)),
        max_steps=max_steps,
        gradient_accumulation_steps=grad_accum_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy_cfg,
        load_best_model_at_end=load_best_cfg and save_strategy_cfg != "no",
        metric_for_best_model=checkpoint_metric.replace("eval_", ""),
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=fp16_flag,
        bf16=bf16_flag,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=int(cfg.get("logging_steps", 20)),
        report_to=[],
        seed=seed,
        max_grad_norm=0.5,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
        save_total_limit=None,
        save_steps=save_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        gradient_checkpointing=grad_ckpt,
    )

    callbacks = [FileLoggerCallback(log_path)]
    best_cb = None
    if freeze_steps > 0:
        callbacks.append(FeatureExtractorFreezeCallback(model, freeze_steps, log_path=log_path))
    if not skip_save and save_strategy_cfg != "no":
        callbacks.append(LossCheckpointCallback(str(out_dir), log_path=log_path))
    if keep_best_n > 0:
        maximize = checkpoint_metric not in ("train_loss", "loss")
        best_cb = BestCheckpointCallback(str(out_dir), checkpoint_metric, keep_best_n, maximize=maximize, log_path=log_path)
        callbacks.append(best_cb)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if skip_eval else val_ds,
        compute_metrics=None if skip_eval else compute_metrics,
        callbacks=callbacks,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    best_ckpt = best_cb.best[0][1] if best_cb and best_cb.best else None
    # Evaluate on test split after training with best model
    if not skip_eval:
        test_ds = AudioDS(test_df, extractor, sr, max_seconds, {**raw2common_map}, label2id)
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(f"[TEST] {test_metrics}")
        _log_line(log_path, f"test_metrics={json.dumps(_json_safe_metrics(test_metrics), sort_keys=True)}")
        preds = trainer.predict(test_ds)
        pred_logits = preds.predictions[0] if isinstance(preds.predictions, (tuple, list)) else preds.predictions
        y_pred = pred_logits.argmax(-1)
        y_true = build_common_labels(test_df, raw2common_map, label2id)
        write_test_report(out_dir, test_df, y_true, y_pred, id2label)
    else:
        print("[INFO] Skipped evaluation (skip_eval=True)")
    if not skip_save:
        if best_ckpt and checkpoint_metric in ("train_loss", "loss"):
            print(f"[INFO] Loading best checkpoint by {checkpoint_metric}: {best_ckpt}")
            _log_line(log_path, f"loading_best_checkpoint={best_ckpt} metric={checkpoint_metric}")
            trainer.model = AutoModelForAudioClassification.from_pretrained(best_ckpt)
        trainer.save_model(str(out_dir))
        print(f"[DONE] Saved best model to {out_dir}")
        _log_line(log_path, f"saved_model={out_dir}")
    else:
        print("[INFO] Skipped saving model (skip_save=True)")

if __name__ == "__main__":
    main()
