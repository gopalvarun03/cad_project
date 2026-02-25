# import os
# import json
# import numpy as np
# import torch
# import h5py
# from tqdm import tqdm
# from typing import Dict, List

# # === REAL Drawing2CAD imports ===
# from config.config import Config
# from config.file_utils import ensure_dir
# from trainer.trainer import TrainerED
# from dataset.bi_sequence_dataset import BiSequenceDataset
# from config.macro import CAD_EOS_IDX, SVG_MAX_TOTAL_LEN, CAD_MAX_TOTAL_LEN


# # ================================================================
# # USER PATHS (EDIT IF NEEDED)
# # ================================================================
# OUTPUT_ROOT = "ideal_iso_test_results"
# GENERATED_DATASET_NAME = "out_vec"
# TRUTH_DATASET_NAME = "vec"
# EXPECTED_COLUMNS = 17


# # ================================================================
# # SINGLE FILE INFERENCE
# # ================================================================
# def run_model_on_file(tr_agent, dataset, data_id, out_path):
#     try:
#         # Get data using the dataset's method
#         data = dataset.get_data_by_id(data_id)
        
#         # Add batch dimension
#         batch = {
#             'svg': {
#                 'view': data['svg']['view'].unsqueeze(0).cuda(),
#                 'command': data['svg']['command'].unsqueeze(0).cuda(),
#                 'args': data['svg']['args'].unsqueeze(0).cuda()
#             },
#             'cad': {
#                 'command': data['cad']['command'].unsqueeze(0).cuda(),
#                 'args': data['cad']['args'].unsqueeze(0).cuda()
#             },
#             'id': [data_id]
#         }
#     except Exception as e:
#         print(f"[ERROR] loading data for {data_id}: {e}")
#         return False

#     # inference
#     with torch.no_grad():
#         outputs, _ = tr_agent.forward(batch)
#         seq_list = tr_agent.logits2vec(outputs)

#     # logits2vec already returns numpy array
#     out_vec = seq_list[0]

#     # apply EOS truncation
#     eos_idx = np.where(out_vec[:, 0] == CAD_EOS_IDX)[0]
#     if len(eos_idx) > 0:
#         out_vec = out_vec[: eos_idx[0] + 1]

#     # save
#     ensure_dir(os.path.dirname(out_path))
#     try:
#         with h5py.File(out_path, "w") as f:
#             f.create_dataset(GENERATED_DATASET_NAME, data=out_vec, dtype=np.int32)
#         return True
#     except Exception as e:
#         print(f"[ERROR] saving h5 {out_path}: {e}")
#         return False


# # ================================================================
# # LOADING + ALIGNMENT + METRICS
# # ================================================================
# # ================================================================
# # LOADING + ALIGNMENT + METRICS
# # ================================================================
# def load_pair(gen_path, truth_path):
#     try:
#         with h5py.File(truth_path, "r") as f:
#             truth = f[TRUTH_DATASET_NAME][:]
#         with h5py.File(gen_path, "r") as f:
#             gen = f[GENERATED_DATASET_NAME][:]
#     except Exception as e:
#         print(f"[ERROR] loading pair: {e}")
#         return None

#     # Truncate both to the minimum length
#     min_len = min(gen.shape[0], truth.shape[0])
#     gen = gen[:min_len]
#     truth = truth[:min_len]

#     # Ensure same number of columns
#     if gen.shape[1] != truth.shape[1]:
#         min_cols = min(gen.shape[1], truth.shape[1])
#         gen = gen[:, :min_cols]
#         truth = truth[:, :min_cols]

#     return gen, truth


# def compute_metrics(gen, truth):
#     metrics = {}

#     # Only consider non-EOS positions
#     is_not_eos = (truth[:, 0] != CAD_EOS_IDX)
#     valid = is_not_eos.sum()

#     if valid == 0:
#         return {"ACCcmd": 0, "ACCparam": 0, "IR": 0, "MCD": 0}

#     # ACCcmd - Command accuracy
#     cmd_match = (gen[:, 0] == truth[:, 0])
#     metrics["ACCcmd"] = 100 * np.sum(cmd_match & is_not_eos) / valid

#     # ACCparam - Parameter accuracy (columns 1 onwards)
#     param_match = np.all(gen[:, 1:] == truth[:, 1:], axis=1)
#     metrics["ACCparam"] = 100 * np.sum(param_match & is_not_eos) / valid

#     # IR - Incorrect Rate (L1 distance)
#     abs_diff = np.abs(gen.astype(np.float32) - truth.astype(np.float32))
#     metrics["IR"] = np.sum(abs_diff * is_not_eos[:, None]) / valid

#     # MCD - Mean Command Distance (L2 distance)
#     sq_diff = (gen.astype(np.float32) - truth.astype(np.float32)) ** 2
#     metrics["MCD"] = np.sqrt(np.sum(sq_diff * is_not_eos[:, None]) / valid)

#     return metrics


# # ================================================================
# # PROCESS SPLIT (train/val/test)
# # ================================================================
# def process_split(split_name, file_list, tr_agent, dataset, cfg):
#     print(f"\n============================")
#     print(f"Processing {split_name.upper()}")
#     print("============================")

#     split_output_dir = os.path.join(cfg.exp_dir, OUTPUT_ROOT, split_name)
#     ensure_dir(split_output_dir)

#     # 1. RUN MODEL FOR ALL FILES
#     for data_id in tqdm(file_list, desc=f"Inference {split_name}"):
#         out_path = os.path.join(split_output_dir, data_id.split('/')[-1] + ".h5")
        
#         # Skip if already processed
#         if os.path.exists(out_path):
#             continue
            
#         run_model_on_file(tr_agent, dataset, data_id, out_path)

#     # 2. EVALUATE METRICS
#     metrics_list = []
#     for data_id in tqdm(file_list, desc=f"Evaluating {split_name}"):
#         file_id = data_id.split('/')[-1]
#         gen_path = os.path.join(split_output_dir, file_id + ".h5")
#         truth_path = os.path.join(cfg.data_root, "cad_vec", data_id + ".h5")

#         if not os.path.exists(gen_path) or not os.path.exists(truth_path):
#             continue

#         pair = load_pair(gen_path, truth_path)
#         if pair is None:
#             continue

#         gen, truth = pair
#         metrics_list.append(compute_metrics(gen, truth))

#     # 3. SUMMARY
#     if not metrics_list:
#         print(f"No metrics for split {split_name}")
#         return

#     summary = {}
#     keys = metrics_list[0].keys()
#     for k in keys:
#         summary[k] = np.mean([m[k] for m in metrics_list])

#     print(f"\n=== SUMMARY FOR {split_name.upper()} ===")
#     for k, v in summary.items():
#         print(f"{k}: {v:.4f}")
    
#     # Save summary to file
#     summary_path = os.path.join(split_output_dir, "metrics_summary.json")
#     with open(summary_path, "w") as f:
#         json.dump(summary, f, indent=2)
#     print(f"Metrics saved to: {summary_path}")


# # ================================================================
# # MAIN
# # ================================================================
# def main():
#     # Load config and model
#     cfg = Config("test")
#     tr_agent = TrainerED(cfg)
    
#     print(f"Loading checkpoint: {cfg.ckpt}")
#     tr_agent.load_ckpt(cfg.ckpt)
#     tr_agent.net.eval()

#     # Load split data
#     json_path = os.path.join(cfg.data_root, "train_val_test_split.json")
#     with open(json_path, "r") as f:
#         split_data = json.load(f)

#     # Process splits
#     for split in ["test"]:  # Start with test, then val, then train
#         if split not in split_data:
#             print(f"[WARNING] Split {split} missing in JSON")
#             continue

#         # Create dataset for this split
#         dataset = BiSequenceDataset(split, cfg)
        
#         process_split(split, split_data[split], tr_agent, dataset, cfg)


# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import torch
import h5py
from tqdm import tqdm

# === Drawing2CAD imports ===
from config.config import Config
from config.file_utils import ensure_dir
from trainer.trainer import TrainerED
from dataset.bi_sequence_dataset import BiSequenceDataset
from config.macro import (
    CAD_EOS_IDX,
    CAD_SOL_IDX,
    CAD_EXT_IDX,
    CAD_ARC_IDX,
    CAD_CMD_ARGS_MASK,
    PAD_VAL,
)

# ================================================================
# USER PATHS (EDIT IF NEEDED)
# ================================================================
OUTPUT_ROOT = "evaluation_results"
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance eta in the paper
PAD_PARAM = PAD_VAL   # padded argument marker (from macro.py)


# ================================================================
# MODEL INFERENCE
# ================================================================
def run_model_on_file(tr_agent, dataset, data_id, out_path):
    try:
        data = dataset.get_data_by_id(data_id)

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
        print(f"[ERROR] failed loading {data_id}: {e}")
        return False

    # Model inference
    with torch.no_grad():
        outputs, _ = tr_agent.forward(batch)
        seq_list = tr_agent.logits2vec(outputs)

    out_vec = seq_list[0]

    # Truncate to EOS
    eos = np.where(out_vec[:, 0] == CAD_EOS_IDX)[0]
    if len(eos) > 0:
        out_vec = out_vec[: eos[0] + 1]

    ensure_dir(os.path.dirname(out_path))
    with h5py.File(out_path, "w") as f:
        f.create_dataset(GENERATED_DATASET_NAME, data=out_vec, dtype=np.int32)

    return True


# ================================================================
# SEQUENCE LOADING
# ================================================================
def load_pair(gen_path, truth_path):
    try:
        with h5py.File(truth_path, "r") as f:
            truth = f[TRUTH_DATASET_NAME][:]
        with h5py.File(gen_path, "r") as f:
            gen = f[GENERATED_DATASET_NAME][:]
    except Exception:
        return None

    T = min(len(gen), len(truth))
    gen = gen[:T]
    truth = truth[:T]

    C = min(gen.shape[1], truth.shape[1])
    gen = gen[:, :C]
    truth = truth[:, :C]

    return gen, truth


# ================================================================
# METRIC IMPLEMENTATION (Drawing2CAD / DeepCAD style)
# ================================================================

# ---------------------
# ACCcmd – Command Type Accuracy (Eq. 8)
# ---------------------
def compute_ACCcmd(pred_cmd, gt_cmd):
    """
    Command type accuracy (ACC_cmd) as in Eq. (8) of Drawing2CAD.

    ACC_cmd = (# positions with correct command type) / (sequence length)

    Evaluated over the whole CAD sequence, including EOS / SOL.
    """
    pred_cmd = np.asarray(pred_cmd)
    gt_cmd = np.asarray(gt_cmd)

    assert pred_cmd.shape == gt_cmd.shape
    Nc = len(gt_cmd)
    if Nc == 0:
        return 0.0

    correct = (pred_cmd == gt_cmd).astype(np.float32)
    return float(correct.mean() * 100.0)


# ---------------------
# ACCparam – Parameter Accuracy (Eq. 9)
# ---------------------
def compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=3):
    """
    Parameter accuracy (ACC_param) as in Eq. (9) of Drawing2CAD, matching
    the DeepCAD evaluation logic.

    - Only parameters of correctly predicted commands are evaluated.
    - For each command, only "used" parameters (CAD_CMD_ARGS_MASK) are counted.
    - For most params: |pred - gt| <= eta is considered correct.
    - For some special params of Arc / Extrude, strict equality is required.

    Returns %.
    """
    pred_args = np.asarray(pred_args)
    gt_args = np.asarray(gt_args)
    pred_cmd = np.asarray(pred_cmd)
    gt_cmd = np.asarray(gt_cmd)

    assert pred_args.shape == gt_args.shape
    assert pred_cmd.shape == gt_cmd.shape

    args_mask = CAD_CMD_ARGS_MASK.astype(bool)

    total_param_cnt = 0.0  # K in Eq. (9)
    total_param_correct = 0.0

    for i in range(len(gt_cmd)):
        cmd = int(gt_cmd[i])

        # Skip SOL / EOS: no meaningful parameters
        if cmd == CAD_SOL_IDX or cmd == CAD_EOS_IDX:
            continue

        # Only evaluate parameters if command type is correctly predicted
        if int(pred_cmd[i]) != cmd:
            continue

        # Elementwise tolerance check
        diff = np.abs(pred_args[i] - gt_args[i])
        tole_acc = (diff <= eta).astype(np.float32)

        # Special-case strict params (DeepCAD logic):
        #   - Extrude: last two params (operation, extent type) must be exact
        #   - Arc: parameter index 3 must be exact
        exact = (pred_args[i] == gt_args[i]).astype(np.float32)

        if cmd == CAD_EXT_IDX:
            tole_acc[-2:] = exact[-2:]

        if cmd == CAD_ARC_IDX:
            if pred_args.shape[1] > 3:
                tole_acc[3] = exact[3]

        # Parameters actually used for this command
        valid_param_mask = args_mask[cmd].copy()

        # Ignore pads inside the used region
        valid_param_mask &= (gt_args[i] != PAD_PARAM)

        if not np.any(valid_param_mask):
            continue

        valid_tole = tole_acc[valid_param_mask]

        total_param_correct += float(valid_tole.sum())
        total_param_cnt += float(valid_param_mask.sum())

    if total_param_cnt == 0:
        return 0.0

    return float(total_param_correct / total_param_cnt * 100.0)


# ---------------------------------------------
# INVALIDITY RATIO (IR) – SEQUENCE-LEVEL PROXY
# ---------------------------------------------
def compute_IR(pred_cmd, pred_args):
    """
    A simple sequence-level invalidity proxy (NOT the full CAD-kernel IR).

    A CAD sequence is marked invalid if:
    - EOS missing
    - illegal negative parameters (values < PAD_PARAM)
    - args missing for a command (all params are PAD_PARAM)

    Returns:
        1 if invalid, 0 if valid
    """

    # 1. EOS must appear
    if CAD_EOS_IDX not in pred_cmd:
        return 1

    # 2. Find EOS position and truncate (exclude EOS from validation)
    eos_idx = np.where(pred_cmd == CAD_EOS_IDX)[0]
    if len(eos_idx) > 0:
        eos_idx = eos_idx[0]
        pred_cmd = pred_cmd[:eos_idx]
        pred_args = pred_args[:eos_idx]

    # If no commands before EOS, it's valid (empty sequence)
    if len(pred_cmd) == 0:
        return 0

    # 3. No illegal negative values (except PAD_PARAM which is padding)
    if np.any(pred_args < PAD_PARAM):
        return 1

    # 4. Check for missing args: if ALL args are PAD_PARAM (padding) → invalid
    for i in range(len(pred_cmd)):
        if np.all(pred_args[i] == PAD_PARAM):
            return 1

    # If all checks pass, sequence is valid
    return 0


# ---------------------------------------------
# MEAN CHAMFER DISTANCE (MCD) – TOKEN-SPACE PROXY
# ---------------------------------------------
def compute_MCD(pred_vec, gt_vec):
    """
    Simple token-space RMSE between predicted and GT sequences.

    NOTE: This is NOT the geometry-level Chamfer Distance in the paper,
    which requires reconstructing CAD solids and sampling 3D point clouds.
    """

    pred_cmd = pred_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    # Only compute over valid commands (before EOS in GT)
    valid = (gt_cmd != CAD_EOS_IDX)
    N = valid.sum()

    if N == 0:
        return 0.0

    diff = (pred_vec - gt_vec).astype(np.float32)
    sq = diff * diff

    # Only over valid commands
    sq = sq[valid]

    return float(np.sqrt(sq.mean()))


def compute_metrics(gen, truth):
    """
    Computes all metrics for a single generated-truth pair.
    """
    pred_cmd = gen[:, 0]
    gt_cmd = truth[:, 0]

    pred_args = gen[:, 1:]
    gt_args = truth[:, 1:]

    ACCcmd = compute_ACCcmd(pred_cmd, gt_cmd)
    ACCparam = compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=TOL)
    IR = compute_IR(pred_cmd, pred_args)
    MCD = compute_MCD(gen, truth)

    return {
        "ACCcmd": ACCcmd,
        "ACCparam": ACCparam,
        "IR": IR,
        "MCD": MCD
    }


# ================================================================
# PROCESS SPLIT (train / val / test)
# ================================================================
def process_split(split_name, file_list, tr_agent, dataset, cfg):
    print(f"\n========== PROCESSING {split_name.upper()} ==========")

    split_out_dir = os.path.join(cfg.exp_dir, OUTPUT_ROOT, split_name)
    ensure_dir(split_out_dir)

    # ---------------------
    # 1. Run Model
    # ---------------------
    for data_id in tqdm(file_list, desc=f"Inference {split_name}"):
        out_path = os.path.join(split_out_dir, data_id.split('/')[-1] + ".h5")
        if not os.path.exists(out_path):
            run_model_on_file(tr_agent, dataset, data_id, out_path)

    # ---------------------
    # 2. Evaluate Metrics
    # ---------------------
    metrics_list = []

    for data_id in tqdm(file_list, desc=f"Eval {split_name}"):
        file_id = data_id.split('/')[-1]
        gen_path = os.path.join(split_out_dir, file_id + ".h5")
        truth_path = os.path.join(cfg.data_root, "cad_vec", data_id + ".h5")

        if not os.path.exists(gen_path) or not os.path.exists(truth_path):
            continue

        pair = load_pair(gen_path, truth_path)
        if pair is None:
            continue

        gen, truth = pair
        metrics_list.append(compute_metrics(gen, truth))

    if not metrics_list:
        print("No metrics computed.")
        return
    # -------------------------------
    # FINAL SUMMARY (ALL METRICS)
    # -------------------------------
    summary = {
        "ACCcmd": float(np.mean([m["ACCcmd"] for m in metrics_list])),
        "ACCparam": float(np.mean([m["ACCparam"] for m in metrics_list])),
        "IR": float(np.mean([m["IR"] for m in metrics_list]) * 100.0),   # %
        "MCD": float(np.mean([m["MCD"] for m in metrics_list]) * 100.0), # ×100 proxy
    }

    print(f"\n===== SUMMARY ({split_name.upper()}) =====")
    print(f"ACCcmd: {summary['ACCcmd']:.2f}%")
    print(f"ACCparam: {summary['ACCparam']:.2f}%")
    print(f"IR: {summary['IR']:.2f}%")
    print(f"MCD: {summary['MCD']:.4f}")

    # save metrics
    summary_path = os.path.join(split_out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved metrics → {summary_path}\n")


# ================================================================
# MAIN
# ================================================================
def main():
    # config "test" should correspond to your test-time exp (same as test.py)
    cfg = Config("test")
    tr_agent = TrainerED(cfg)

    print("Loading checkpoint:", cfg.ckpt)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    json_path = os.path.join(cfg.data_root, "train_val_test_split.json")
    with open(json_path, "r") as f:
        split_data = json.load(f)

    # Process in the same order as your script
    # for split in ["test", "val", "train"]:
    #     if split not in split_data:
    #         continue
    for split in ["test", "val"]:
        if split not in split_data:
            continue

        dataset = BiSequenceDataset(split, cfg)
        process_split(split, split_data[split], tr_agent, dataset, cfg)

if __name__ == "__main__":
    main()
