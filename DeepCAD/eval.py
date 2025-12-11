import os
import json
import numpy as np
import torch
import h5py
import glob
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# === REAL Drawing2CAD imports ===
from config.config import Config
from config.file_utils import ensure_dir
from trainer.trainer import TrainerED
from config.macro import CAD_EOS_IDX


# ================================================================
# USER PATHS (EDIT IF NEEDED)
# ================================================================
JSON_PATH = "C:\\Users\\LEGION\\Desktop\\cad_project\\DeepCAD\\data2\\train_val_test_split.json"

NPY_ROOT = "C:\\Users\\LEGION\\Desktop\\cad_project\\DeepCAD\\data2\\svg_vec"
TRUTH_ROOT = "C:\\Users\\LEGION\\Desktop\\cad_project\\DeepCAD\\data2\\cad_vec"
OUTPUT_ROOT = "C:\\Users\\LEGION\\Desktop\\cad_project\\DeepCAD\\data2\\Converted_H5_Files"
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
EXPECTED_COLUMNS = 17


# ================================================================
# SINGLE FILE INFERENCE
# ================================================================
def run_model_on_file(tr_agent, npy_path, out_path):
    try:
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 1:
            data = np.expand_dims(data, 0)

        inp = torch.from_numpy(data).unsqueeze(0).float().to(tr_agent.device)
        batch = {"input_features": inp}
    except Exception as e:
        print(f"[ERROR] loading npy {npy_path}: {e}")
        return False

    # inference
    with torch.no_grad():
        outputs, _ = tr_agent.forward(batch)
        seq_list = tr_agent.logits2vec(outputs)

    out_vec = seq_list[0].cpu().numpy()

    # apply EOS truncation
    eos_idx = np.where(out_vec[:, 0] == CAD_EOS_IDX)[0]
    if len(eos_idx) > 0:
        out_vec = out_vec[: eos_idx[0] + 1]

    # save
    ensure_dir(os.path.dirname(out_path))
    try:
        with h5py.File(out_path, "w") as f:
            f.create_dataset(GENERATED_DATASET_NAME, data=out_vec, dtype=np.int32)
        return True
    except Exception as e:
        print(f"[ERROR] saving h5 {out_path}: {e}")
        return False


# ================================================================
# LOADING + ALIGNMENT + METRICS
# ================================================================
def pad_to_cols(arr, tgt_cols, dtype):
    if arr.shape[1] == tgt_cols:
        return arr
    if arr.shape[1] > tgt_cols:
        return arr[:, :tgt_cols]
    pad = tgt_cols - arr.shape[1]
    return np.pad(arr, ((0, 0), (0, pad)), mode="constant").astype(dtype)


def load_pair(gen_path, truth_path):
    try:
        truth = h5py.File(truth_path, "r")[TRUTH_DATASET_NAME][:]
        gen = h5py.File(gen_path, "r")[GENERATED_DATASET_NAME][:]
    except:
        return None

    tgt_rows, tgt_cols = truth.shape
    gen = pad_to_cols(gen, tgt_cols, truth.dtype)

    if gen.shape[0] > tgt_rows:
        gen = gen[:tgt_rows]
    else:
        pad = tgt_rows - gen.shape[0]
        gen = np.pad(gen, ((0, pad), (0, 0)))

    if gen.shape != truth.shape:
        return None

    return gen, truth


def compute_metrics(gen, truth):
    metrics = {}

    is_not_pad = (truth[:, 0] != 0)
    valid = is_not_pad.sum()

    # ACCcmd
    cmd_match = (gen[:, 0] == truth[:, 0])
    metrics["ACCcmd"] = (
        100 * np.sum(cmd_match & is_not_pad) / valid if valid > 0 else 0
    )

    # ACCparam
    param_match = np.all(gen[:, 1:EXPECTED_COLUMNS] == truth[:, 1:EXPECTED_COLUMNS], axis=1)
    metrics["ACCparam"] = (
        100 * np.sum(param_match & is_not_pad) / valid if valid > 0 else 0
    )

    # IR
    abs_diff = np.abs(gen - truth)
    metrics["IR"] = np.sum(abs_diff * is_not_pad[:, None]) / (valid if valid else 1)

    # MCD
    sq_diff = (gen - truth) ** 2
    metrics["MCD"] = np.sqrt(
        np.sum(sq_diff * is_not_pad[:, None]) / (valid if valid else 1)
    )

    return metrics


# ================================================================
# PROCESS SPLIT (train/val/test)
# ================================================================
def process_split(split_name, file_list, tr_agent):
    print(f"\n============================")
    print(f"Processing {split_name.upper()}")
    print("============================")

    split_output_dir = os.path.join(OUTPUT_ROOT, split_name)
    ensure_dir(split_output_dir)

    # 1. RUN MODEL FOR ALL FILES
    for item in tqdm(file_list, desc=f"Inference {split_name}"):
        folder, fileid = item.split("/")
        npy_path = os.path.join(NPY_ROOT, folder, fileid + ".npy")
        out_path = os.path.join(split_output_dir, fileid + ".h5")

        run_model_on_file(tr_agent, npy_path, out_path)

    # 2. EVALUATE METRICS
    metrics_list = []
    for item in tqdm(file_list, desc=f"Evaluating {split_name}"):
        folder, fileid = item.split("/")

        gen_path = os.path.join(split_output_dir, fileid + ".h5")
        truth_path = os.path.join(TRUTH_ROOT, folder, fileid + ".h5")

        pair = load_pair(gen_path, truth_path)
        if pair is None:
            continue

        gen, truth = pair
        metrics_list.append(compute_metrics(gen, truth))

    # 3. SUMMARY
    if not metrics_list:
        print(f"No metrics for split {split_name}")
        return

    summary = {}
    keys = metrics_list[0].keys()
    for k in keys:
        summary[k] = np.mean([m[k] for m in metrics_list])

    print(f"\n=== SUMMARY FOR {split_name.upper()} ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


# ================================================================
# MAIN
# ================================================================
def main():

    # Load JSON
    with open(JSON_PATH, "r") as f:
        split_data = json.load(f)

    # Load real model
    cfg = Config("test")
    tr_agent = TrainerED(cfg)
    cfg.ckpt=r'C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\model\latest.pth'
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # Process splits
    # for split in ["train", "val", "test"]:
    for split in ["test"]:
        if split not in split_data:
            print(f"[WARNING] Split {split} missing in JSON")
            continue

        process_split(split, split_data[split], tr_agent)


if __name__ == "__main__":
    main()
