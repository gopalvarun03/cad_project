import os
import json
import numpy as np
import torch
import h5py
from tqdm import tqdm
from typing import Dict, List

# === REAL Drawing2CAD imports ===
from config.config import Config
from config.file_utils import ensure_dir
from trainer.trainer import TrainerED
from dataset.bi_sequence_dataset import BiSequenceDataset
from config.macro import CAD_EOS_IDX, SVG_MAX_TOTAL_LEN, CAD_MAX_TOTAL_LEN


# ================================================================
# USER PATHS (EDIT IF NEEDED)
# ================================================================
OUTPUT_ROOT = "evaluation_results"
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
EXPECTED_COLUMNS = 17


# ================================================================
# SINGLE FILE INFERENCE
# ================================================================
def run_model_on_file(tr_agent, dataset, data_id, out_path):
    try:
        # Get data using the dataset's method
        data = dataset.get_data_by_id(data_id)
        
        # Add batch dimension
        batch = {
            'svg': {
                'view': data['svg']['view'].unsqueeze(0).cuda(),
                'command': data['svg']['command'].unsqueeze(0).cuda(),
                'args': data['svg']['args'].unsqueeze(0).cuda()
            },
            'cad': {
                'command': data['cad']['command'].unsqueeze(0).cuda(),
                'args': data['cad']['args'].unsqueeze(0).cuda()
            },
            'id': [data_id]
        }
    except Exception as e:
        print(f"[ERROR] loading data for {data_id}: {e}")
        return False

    # inference
    with torch.no_grad():
        outputs, _ = tr_agent.forward(batch)
        seq_list = tr_agent.logits2vec(outputs)

    # logits2vec already returns numpy array
    out_vec = seq_list[0]

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
# ================================================================
# LOADING + ALIGNMENT + METRICS
# ================================================================
def load_pair(gen_path, truth_path):
    try:
        with h5py.File(truth_path, "r") as f:
            truth = f[TRUTH_DATASET_NAME][:]
        with h5py.File(gen_path, "r") as f:
            gen = f[GENERATED_DATASET_NAME][:]
    except Exception as e:
        print(f"[ERROR] loading pair: {e}")
        return None

    # Truncate both to the minimum length
    min_len = min(gen.shape[0], truth.shape[0])
    gen = gen[:min_len]
    truth = truth[:min_len]

    # Ensure same number of columns
    if gen.shape[1] != truth.shape[1]:
        min_cols = min(gen.shape[1], truth.shape[1])
        gen = gen[:, :min_cols]
        truth = truth[:, :min_cols]

    return gen, truth


def compute_metrics(gen, truth):
    metrics = {}

    # Only consider non-EOS positions
    is_not_eos = (truth[:, 0] != CAD_EOS_IDX)
    valid = is_not_eos.sum()

    if valid == 0:
        return {"ACCcmd": 0, "ACCparam": 0, "IR": 0, "MCD": 0}

    # ACCcmd - Command accuracy
    cmd_match = (gen[:, 0] == truth[:, 0])
    metrics["ACCcmd"] = 100 * np.sum(cmd_match & is_not_eos) / valid

    # ACCparam - Parameter accuracy (columns 1 onwards)
    param_match = np.all(gen[:, 1:] == truth[:, 1:], axis=1)
    metrics["ACCparam"] = 100 * np.sum(param_match & is_not_eos) / valid

    # IR - Incorrect Rate (L1 distance)
    abs_diff = np.abs(gen.astype(np.float32) - truth.astype(np.float32))
    metrics["IR"] = np.sum(abs_diff * is_not_eos[:, None]) / valid

    # MCD - Mean Command Distance (L2 distance)
    sq_diff = (gen.astype(np.float32) - truth.astype(np.float32)) ** 2
    metrics["MCD"] = np.sqrt(np.sum(sq_diff * is_not_eos[:, None]) / valid)

    return metrics


# ================================================================
# PROCESS SPLIT (train/val/test)
# ================================================================
def process_split(split_name, file_list, tr_agent, dataset, cfg):
    print(f"\n============================")
    print(f"Processing {split_name.upper()}")
    print("============================")

    split_output_dir = os.path.join(cfg.exp_dir, OUTPUT_ROOT, split_name)
    ensure_dir(split_output_dir)

    # 1. RUN MODEL FOR ALL FILES
    for data_id in tqdm(file_list, desc=f"Inference {split_name}"):
        out_path = os.path.join(split_output_dir, data_id.split('/')[-1] + ".h5")
        
        # Skip if already processed
        if os.path.exists(out_path):
            continue
            
        run_model_on_file(tr_agent, dataset, data_id, out_path)

    # 2. EVALUATE METRICS
    metrics_list = []
    for data_id in tqdm(file_list, desc=f"Evaluating {split_name}"):
        file_id = data_id.split('/')[-1]
        gen_path = os.path.join(split_output_dir, file_id + ".h5")
        truth_path = os.path.join(cfg.data_root, "cad_vec", data_id + ".h5")

        if not os.path.exists(gen_path) or not os.path.exists(truth_path):
            continue

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
    
    # Save summary to file
    summary_path = os.path.join(split_output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to: {summary_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    # Load config and model
    cfg = Config("test")
    tr_agent = TrainerED(cfg)
    
    print(f"Loading checkpoint: {cfg.ckpt}")
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # Load split data
    json_path = os.path.join(cfg.data_root, "train_val_test_split.json")
    with open(json_path, "r") as f:
        split_data = json.load(f)

    # Process splits
    for split in ["test"]:  # Start with test, then val, then train
        if split not in split_data:
            print(f"[WARNING] Split {split} missing in JSON")
            continue

        # Create dataset for this split
        dataset = BiSequenceDataset(split, cfg)
        
        process_split(split, split_data[split], tr_agent, dataset, cfg)


if __name__ == "__main__":
    main()
