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
from config.macro import CAD_EOS_IDX, SVG_MAX_TOTAL_LEN, CAD_MAX_TOTAL_LEN, CAD_EOS_VEC


# ================================================================
# USER PATHS (EDIT IF NEEDED)
# ================================================================
OUTPUT_ROOT = "evaluation_results_rot1"
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
EXPECTED_COLUMNS = 17

# Path to custom npy folder
CUSTOM_NPY_FOLDER = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_h5_files"


# ================================================================
# HELPER: Build mapping from file_id to data_id
# ================================================================
def build_file_id_to_data_id_map(json_path, split="test"):
    """
    Build a mapping from file_id (e.g., '00000134') to data_id (e.g., '0000/00000134')
    """
    with open(json_path, "r") as f:
        split_data = json.load(f)
    
    mapping = {}
    for data_id in split_data.get(split, []):
        file_id = data_id.split('/')[-1]  # '0000/00000134' -> '00000134'
        mapping[file_id] = data_id
    return mapping


# ================================================================
# HELPER: Load SVG data from custom .npy file
# ================================================================
def load_svg_from_npy(npy_path, input_option="4x"):
    """Load SVG data from a .npy file (same format as BiSequenceDataset)"""
    data = np.load(npy_path)
    
    # 1x
    if input_option == "1x":
        view_vec = data[300:, 0]
        command_vec = data[300:, 1]
        args_vec = data[300:, 2:]
    # 3x
    elif input_option == "3x":
        view_vec = data[:300, 0]
        command_vec = data[:300, 1]
        args_vec = data[:300, 2:]
    # 4x
    else:
        view_vec = data[:, 0]
        command_vec = data[:, 1]
        args_vec = data[:, 2:]

    view_vec = torch.tensor(view_vec, dtype=torch.long)
    command_vec = torch.tensor(command_vec, dtype=torch.long)
    args_vec = torch.tensor(args_vec, dtype=torch.long)
    
    return {"view": view_vec, "command": command_vec, "args": args_vec}


# ================================================================
# HELPER: Load CAD ground truth from h5
# ================================================================
def load_cad_from_h5(h5_path, max_len=CAD_MAX_TOTAL_LEN):
    """Load CAD data from h5 file"""
    with h5py.File(h5_path, "r") as fp:
        cad_vec = fp["vec"][:]

    pad_len = max_len - cad_vec.shape[0]
    if pad_len > 0:
        cad_vec = np.concatenate([cad_vec, CAD_EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
    else:
        cad_vec = cad_vec[:max_len]

    command = cad_vec[:, 0]
    args = cad_vec[:, 1:]
    command = torch.tensor(command, dtype=torch.long)
    args = torch.tensor(args, dtype=torch.long)
    return {"command": command, "args": args}


# ================================================================
# SINGLE FILE INFERENCE (from custom npy)
# ================================================================
def run_model_on_npy_file(tr_agent, npy_path, cad_h5_path, data_id, out_path, input_option="4x"):
    try:
        # Load SVG from custom npy
        svg_data = load_svg_from_npy(npy_path, input_option)
        
        # Load CAD ground truth
        cad_data = load_cad_from_h5(cad_h5_path)
        
        # Add batch dimension
        batch = {
            'svg': {
                'view': svg_data['view'].unsqueeze(0).cuda(),
                'command': svg_data['command'].unsqueeze(0).cuda(),
                'args': svg_data['args'].unsqueeze(0).cuda()
            },
            'cad': {
                'command': cad_data['command'].unsqueeze(0).cuda(),
                'args': cad_data['args'].unsqueeze(0).cuda()
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
# SINGLE FILE INFERENCE (original - from dataset)
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
def compute_ACCcmd(gen_cmd, gt_cmd, eos_token):
    """
    Computes ACCcmd exactly as equation (8) in the DeepCAD paper.
    """
    # Ground truth sequence ends at EOS
    valid_mask = (gt_cmd != eos_token)
    Nc = valid_mask.sum()

    if Nc == 0:
        return 0.0

    correct = (gen_cmd == gt_cmd) & valid_mask

    return 100.0 * correct.sum() / Nc
def compute_ACCparam(gen_args, gt_args, gen_cmd, gt_cmd, eta=3, eos_token=CAD_EOS_IDX):
    """
    Computes ACCparam exactly as equation (9) in the DeepCAD paper.
    """
    # 1. Identify valid command positions (until EOS)
    valid_mask = (gt_cmd != eos_token)

    # 2. Command correctness mask
    cmd_correct = (gen_cmd == gt_cmd) & valid_mask

    # If no commands are correct, ACCparam = 0
    K = cmd_correct.sum() * gt_args.shape[1]   # total evaluated parameters
    if K == 0:
        return 0.0

    # 3. Compute parameter absolute error
    abs_diff = np.abs(gt_args - gen_args)

    # 4. Parameter tolerance mask
    param_correct = (abs_diff < eta)

    # 5. Apply command correctness constraint
    final_mask = param_correct * cmd_correct[:, None]

    # ACCparam = (# of correct parameters) / (total evaluated params)
    return 100.0 * final_mask.sum() / K
def compute_metrics(gen, truth, eta=3):
    gt_cmd = truth[:, 0]
    gen_cmd = gen[:, 0]

    gt_args = truth[:, 1:]
    gen_args = gen[:, 1:]

    # ---- ACCcmd ----
    ACCcmd = compute_ACCcmd(gen_cmd, gt_cmd, CAD_EOS_IDX)

    # ---- ACCparam ----
    ACCparam = compute_ACCparam(gen_args, gt_args, gen_cmd, gt_cmd, eta=eta)

    # ---- IR ----
    valid_mask = (gt_cmd != CAD_EOS_IDX)
    valid_count = valid_mask.sum()

    abs_diff = np.abs(gen - truth)
    IR = np.sum(abs_diff * valid_mask[:, None]) / max(1, valid_count)

    # ---- MCD ----
    sq_diff = (gen - truth) ** 2
    MCD = np.sqrt(np.sum(sq_diff * valid_mask[:, None]) / max(1, valid_count))

    return {
        "ACCcmd": ACCcmd,
        "ACCparam": ACCparam,
        "IR": IR,
        "MCD": MCD
    }
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
# PROCESS CUSTOM NPY FOLDER (test_h5_files)
# ================================================================
def process_custom_npy_folder(tr_agent, cfg, npy_folder, file_id_to_data_id):
    print(f"\n============================")
    print(f"Processing CUSTOM NPY FOLDER: {npy_folder}")
    print("============================")

    split_output_dir = os.path.join(cfg.exp_dir, OUTPUT_ROOT, "test_custom_npy")
    ensure_dir(split_output_dir)

    # Get all .npy files in the folder
    npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
    print(f"Found {len(npy_files)} .npy files")

    # 1. RUN MODEL FOR ALL FILES
    processed = 0
    skipped = 0
    for npy_file in tqdm(npy_files, desc="Inference custom npy"):
        file_id = npy_file.replace('.npy', '')  # e.g., '00000134'
        
        # Find corresponding data_id from JSON mapping
        if file_id not in file_id_to_data_id:
            skipped += 1
            continue
        
        data_id = file_id_to_data_id[file_id]  # e.g., '0000/00000134'
        
        npy_path = os.path.join(npy_folder, npy_file)
        cad_h5_path = os.path.join(cfg.data_root, "cad_vec", data_id + ".h5")
        out_path = os.path.join(split_output_dir, file_id + ".h5")
        
        # Skip if already processed
        if os.path.exists(out_path):
            processed += 1
            continue
        
        # Check if ground truth exists
        if not os.path.exists(cad_h5_path):
            skipped += 1
            continue
            
        if run_model_on_npy_file(tr_agent, npy_path, cad_h5_path, data_id, out_path, cfg.input_option):
            processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")

    # 2. EVALUATE METRICS
    metrics_list = []
    for npy_file in tqdm(npy_files, desc="Evaluating custom npy"):
        file_id = npy_file.replace('.npy', '')
        
        if file_id not in file_id_to_data_id:
            continue
        
        data_id = file_id_to_data_id[file_id]
        
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
        print(f"No metrics computed")
        return

    summary = {}
    keys = metrics_list[0].keys()
    for k in keys:
        summary[k] = np.mean([m[k] for m in metrics_list])

    print(f"\n=== SUMMARY FOR CUSTOM NPY TEST ===")
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

    # Load split data and build file_id -> data_id mapping for test set
    json_path = os.path.join(cfg.data_root, "train_val_test_split.json")
    file_id_to_data_id = build_file_id_to_data_id_map(json_path, split="test")
    print(f"Built mapping for {len(file_id_to_data_id)} test files")

    # Process custom npy folder
    process_custom_npy_folder(tr_agent, cfg, CUSTOM_NPY_FOLDER, file_id_to_data_id)


if __name__ == "__main__":
    main()
