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
from config.macro import CAD_EOS_IDX


# ================================================================
# USER PATHS (EDIT IF NEEDED)
# ================================================================
OUTPUT_ROOT = "evaluation_results"
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance
PAD_PARAM = -1   # padded argument marker


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
    except:
        return None

    T = min(len(gen), len(truth))
    gen = gen[:T]
    truth = truth[:T]

    C = min(gen.shape[1], truth.shape[1])
    gen = gen[:, :C]
    truth = truth[:, :C]

    return gen, truth


# ================================================================
# METRIC IMPLEMENTATION (Drawing2CAD Official)
# ================================================================

# ---------------------
# ACCcmd – Command Type Accuracy
# ---------------------
def compute_ACCcmd(pred_cmd, gt_cmd):
    """
    Computes command type accuracy.
    Only evaluates commands before EOS.
    """
    valid = (gt_cmd != CAD_EOS_IDX)
    Nc = valid.sum()
    if Nc == 0:
        return 0.0
    correct = (pred_cmd == gt_cmd) & valid
    return 100.0 * correct.sum() / Nc


# ---------------------
# ACCparam – Parameter Accuracy
# ---------------------
def compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=3):
    """
    Computes parameter accuracy.
    Only evaluates parameters where:
    1. Command type is correctly predicted
    2. GT parameter is valid (not -1)
    """
    valid_cmd = (gt_cmd != CAD_EOS_IDX)
    correct_cmd = (pred_cmd == gt_cmd) & valid_cmd

    # Only evaluate parameters where GT is valid (not -1)
    valid_param = (gt_args != PAD_PARAM)

    # Mask: only check params where command is correct AND param is valid
    mask = valid_param & correct_cmd[:, None]
    K = mask.sum()

    if K == 0:
        return 0.0

    abs_diff = np.abs(gt_args - pred_args)
    correct_param = (abs_diff <= eta)

    return 100.0 * (correct_param & mask).sum() / K


# ---------------------------------------------
# INVALIDITY RATIO (IR) from Drawing2CAD
# ---------------------------------------------
def compute_IR(pred_cmd, pred_args):
    """
    A CAD sequence is invalid if:
    - EOS missing
    - invalid negative parameters (not -1 pad)
    - args missing for a command (all params are -1)
    
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

    # 3. No illegal negative values (except -1 which is padding)
    if np.any(pred_args < -1):
        return 1

    # 4. Check for missing args: if ALL args are -1 (padding) → invalid
    # Each command should have at least some valid parameters
    for i in range(len(pred_cmd)):
        if np.all(pred_args[i] == PAD_PARAM):
            return 1

    # If all checks pass, sequence is valid
    return 0


# ---------------------------------------------
# MEAN CHAMFER DISTANCE (MCD)
# ---------------------------------------------
def compute_MCD(pred_vec, gt_vec):
    """
    Computes Mean Chamfer Distance between predicted and GT sequences.
    Only computed over valid commands (before EOS).
    """
    pred_cmd = pred_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

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
    # 📌 FINAL SUMMARY (ALL METRICS)
    # -------------------------------
    summary = {
        "ACCcmd": np.mean([m["ACCcmd"] for m in metrics_list]),
        "ACCparam": np.mean([m["ACCparam"] for m in metrics_list]),
        "IR": np.mean([m["IR"] for m in metrics_list]) * 100,  # Convert to percentage
        "MCD": np.mean([m["MCD"] for m in metrics_list]) * 100,  # Multiply by 100 as per paper
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
    cfg = Config("test")
    tr_agent = TrainerED(cfg)

    print("Loading checkpoint:", cfg.ckpt)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    json_path = os.path.join(cfg.data_root, "train_val_test_split.json")
    with open(json_path, "r") as f:
        split_data = json.load(f)

    for split in ["test", "val", "train"]:
        if split not in split_data:
            continue

        dataset = BiSequenceDataset(split, cfg)
        process_split(split, split_data[split], tr_agent, dataset, cfg)#sso ipudu na code ela check cheyali? ikkada cheya local lone kadha visualise chesedhi,,,,, 
if __name__ == "__main__":
    main()