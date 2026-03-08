from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import numpy as np
from config.macro import *

# ------------------------------------------------------------------ #
# Ratio encoding constants (must match trainer.py decode)
# e1 and e2 are re-quantized as e1/s and e2/s during training so the
# model learns the aspect ratio (a/h = s/e1) directly.
# The raw h5 files are NOT changed; the transform is applied on-the-fly.
# ------------------------------------------------------------------ #
ARGS_DIM       = 256           # quantization bins (matches ARGS_DIM in macro.py)
E_RATIO_RANGE  = 4.0           # ratios clamped to [-4, 4]

# Args column indices inside the args array (cad_vec[:, 1:])
_IDX_S  = 11   # sketch_size
_IDX_E1 = 12   # extent_one
_IDX_E2 = 13   # extent_two


def _encode_ratio(e_int, s_int, n=ARGS_DIM):
    """Convert a raw extent token (e_int) into an e/s ratio token.

    Args:
        e_int (np.ndarray, int32): quantized extent value(s) in [0, n-1]
        s_int (np.ndarray, int32): quantized sketch_size in [0, n-1]
        n (int): quantization bins (256)

    Returns:
        np.ndarray (int32): ratio token in [0, n-1]
    """
    # 1. De-quantize to floats
    e_f = e_int.astype(np.float64) / n * 2.0 - 1.0   # e in [-1, 1]
    s_f = s_int.astype(np.float64) / n * 2.0          # s in ( 0, 2]
    s_f = np.where(s_f <= 0, 1e-6, s_f)               # guard div-by-zero
    # 2. Compute ratio, clamp to [-E_RATIO_RANGE, E_RATIO_RANGE]
    ratio = np.clip(e_f / s_f, -E_RATIO_RANGE, E_RATIO_RANGE)
    # 3. Re-quantize ratio to [0, n-1]
    ratio_int = np.round(
        (ratio + E_RATIO_RANGE) / (2.0 * E_RATIO_RANGE) * (n - 1)
    ).clip(0, n - 1).astype(np.int32)
    return ratio_int

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = BiSequenceDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

class BiSequenceDataset(Dataset):
    def __init__(self, phase, config):
        super(BiSequenceDataset, self).__init__()
        self.svg_vec = os.path.join(config.data_root, "svg_vec") # svg_vec data root
        self.cad_vec = os.path.join(config.data_root, "cad_vec") # cad_vec data root
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
    
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.svg_max_total_len = SVG_MAX_TOTAL_LEN
        self.cad_max_total_len = CAD_MAX_TOTAL_LEN
        
        self.input_option = config.input_option

    def __len__(self):
        return len(self.all_data)
    
    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def get_svg_data(self, data_id):
        # npy_path = os.path.join(self.svg_vec, data_id + "_merged.npy")
        npy_path=os.path.join(self.svg_vec, data_id + ".npy")
        data = np.load(npy_path)

        # 1x
        if self.input_option == "1x":
            view_vec = data[300:, 0]
            command_vec = data[300:, 1]
            args_vec = data[300:, 2:]
        # 3x
        if self.input_option == "3x":
            view_vec = data[:300, 0]
            command_vec = data[:300, 1]
            args_vec = data[:300, 2:]
        # 4x
        if self.input_option == "4x":
            view_vec = data[:, 0]
            command_vec = data[:, 1]
            args_vec = data[:, 2:]

        view_vec = torch.tensor(view_vec, dtype=torch.long)
        command_vec = torch.tensor(command_vec, dtype=torch.long)
        args_vec = torch.tensor(args_vec, dtype=torch.long)
        
        return {"view": view_vec, "command": command_vec, "args": args_vec}


    def get_cad_data(self, data_id):
        h5_path = os.path.join(self.cad_vec, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.cad_max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, CAD_EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        # ---- On-the-fly e1/s ratio transform ----
        # Replace e1 and e2 tokens with e1/s and e2/s ratio tokens for EXT rows.
        # This teaches the model the correlated aspect ratio (a/h = s/e1) without
        # changing any h5 file on disk.
        ext_mask = (cad_vec[:, 0] == CAD_EXT_IDX)          # shape (seq_len,)
        if ext_mask.any():
            args = cad_vec[:, 1:]                           # (seq_len, 16)
            s_int  = args[ext_mask, _IDX_S]                # (n_ext,)
            e1_int = args[ext_mask, _IDX_E1]
            e2_int = args[ext_mask, _IDX_E2]
            args[ext_mask, _IDX_E1] = _encode_ratio(e1_int, s_int)
            args[ext_mask, _IDX_E2] = _encode_ratio(e2_int, s_int)
            cad_vec[:, 1:] = args
        # ---- end ratio transform ----

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args}


    def __getitem__(self, index):
        data_id = self.all_data[index]
        cad_data = self.get_cad_data(data_id)
        svg_data = self.get_svg_data(data_id)

        return {"svg": svg_data, "cad": cad_data, "id": data_id}
