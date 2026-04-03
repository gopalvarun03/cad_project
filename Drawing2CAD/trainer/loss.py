import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask_cad, _get_visibility_mask
from config.macro import CAD_CMD_ARGS_MASK, CAD_EXT_IDX


# Args column indices inside the args dimension (0-indexed, 16 total)
#   [0-4]  sketch: x, y, alpha, f, r
#   [5-7]  plane:  theta, phi, gamma
#   [8-11] trans:  p_x, p_y, p_z, s
#   [12-15] ext:   e1, e2, b, u
_IDX_S  = 11   # sketch_size  (s)
_IDX_E1 = 12   # extent_one   (e1)
_IDX_E2 = 13   # extent_two   (e2)

# Clamp range for geometry ratios (e/s). Prevents gradient blowup from
# extreme outliers; a ratio beyond ±4 is already outside the data range.
_RATIO_CLAMP = 4.0


class NewCADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.cad_n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CAD_CMD_ARGS_MASK))

    def forward(self, outputs, cad_data):
        # Target
        tgt_commands = cad_data["command"].cuda()   # (B, S)
        tgt_args     = cad_data["args"].cuda()       # (B, S, N_ARGS=16), int tokens [-1, 255]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask_cad(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        # Prediction
        command_logits = outputs["command_logits"]   # (B, S, N_CMD)
        args_logits    = outputs["args_logits"]      # (B, S, N_ARGS, N_CLASS=257)

        mask = self.cmd_args_mask[tgt_commands.long()]

        # --- Original token losses (unchanged) ---
        loss_cmd  = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long()
        )
        loss_args = gumbel_loss(args_logits, tgt_args, mask)

        loss_cmd  = self.weights["loss_cmd_weight"]  * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        # --- Geometry loss (continuous space, e1/s and e2/s) ---
        loss_geom = geometry_ratio_loss(args_logits, tgt_args, tgt_commands)
        loss_geom = self.weights.get("loss_geom_weight", 0.0) * loss_geom

        res = {
            "loss_cmd":  loss_cmd,
            "loss_args": loss_args,
            "loss_geom": loss_geom,
        }
        return res


def soft_dequantize(logits, mode):
    """Compute expected continuous value from a distribution over token classes.

    The class dimension uses the +1 shift convention from gumbel_loss:
        class 0  -> PAD / unused
        class k  -> token value k-1  (k=1..256 → token 0..255)

    Args:
        logits: (B, S, N_ARGS, N_CLASS=257)
        mode:   's'  -> dequantize as sketch_size:  token / 256 * 2  ∈ (0, 2]
                'e'  -> dequantize as extent:        token / 256 * 2 - 1  ∈ [-1, 1]

    Returns:
        (B, S) float tensor
    """
    N_CLASS = logits.shape[-1]  # 257
    n = N_CLASS - 1             # 256

    probs = F.softmax(logits, dim=-1)                    # (B, S, N_ARGS, 257)

    # Expected class index (weighted sum over class positions 0..256)
    bin_idx = torch.arange(N_CLASS, dtype=torch.float32, device=logits.device)
    soft_class = (probs * bin_idx).sum(dim=-1)           # (B, S, N_ARGS)
    soft_token = soft_class - 1.0                        # shift back to token space ∈ [-1, 255]

    if mode == 's':
        # sketch_size: token in [0, n-1] → s_f = token / n * 2  ∈ (0, 2]
        return soft_token / n * 2.0
    else:  # 'e'
        # extent: token in [0, n-1] → e_f = token / n * 2 - 1  ∈ [-1, 1]
        return soft_token / n * 2.0 - 1.0


def geometry_ratio_loss(args_logits, tgt_args, tgt_commands):
    """L1 loss on e1/s and e2/s in CONTINUOUS geometry space.

    Works entirely in float — no extra quantization, no extra tokens.
    Only applied at EXT command positions where s, e1, e2 are valid.

    Steps:
        1. Soft-dequantize predicted logits for s, e1, e2
        2. Compute predicted ratios: g1 = e1_pred / s_pred
        3. Dequantize GT integer tokens for s, e1, e2
        4. Compute GT ratios: g1_gt = e1_gt / s_gt
        5. L1(g1_pred, g1_gt) restricted to EXT positions

    Returns:
        Scalar loss (0.0 if no EXT positions in batch)
    """
    B, S, N_ARGS, N_CLASS = args_logits.shape
    n = N_CLASS - 1   # 256

    # ---- Predicted continuous values (soft dequantize) ----
    s_pred  = soft_dequantize(args_logits[:, :, _IDX_S,  :], mode='s')  # (B, S)
    e1_pred = soft_dequantize(args_logits[:, :, _IDX_E1, :], mode='e')  # (B, S)
    e2_pred = soft_dequantize(args_logits[:, :, _IDX_E2, :], mode='e')  # (B, S)

    # ---- Ground truth continuous values (hard dequantize from integer tokens) ----
    s_gt_tok  = tgt_args[:, :, _IDX_S].float()    # (B, S)  token ∈ [-1, 255]
    e1_gt_tok = tgt_args[:, :, _IDX_E1].float()
    e2_gt_tok = tgt_args[:, :, _IDX_E2].float()

    s_gt  = s_gt_tok  / n * 2.0           # s   ∈ (0, 2]
    e1_gt = e1_gt_tok / n * 2.0 - 1.0    # e1  ∈ [-1, 1]
    e2_gt = e2_gt_tok / n * 2.0 - 1.0    # e2  ∈ [-1, 1]

    # ---- EXT mask: only supervise at Ext command positions with valid s ----
    ext_mask = (tgt_commands == CAD_EXT_IDX)            # (B, S) bool
    valid_s  = (s_gt > 1e-4) & (s_pred.detach() > 1e-4)
    active   = ext_mask & valid_s                        # (B, S) bool

    if not active.any():
        return args_logits.sum() * 0.0   # keeps graph; returns 0

    # ---- Ratios (clamped to avoid outliers) ----
    g1_pred = (e1_pred / s_pred.clamp(min=1e-6)).clamp(-_RATIO_CLAMP, _RATIO_CLAMP)
    g2_pred = (e2_pred / s_pred.clamp(min=1e-6)).clamp(-_RATIO_CLAMP, _RATIO_CLAMP)

    g1_gt   = (e1_gt / s_gt.clamp(min=1e-6)).clamp(-_RATIO_CLAMP, _RATIO_CLAMP)
    g2_gt   = (e2_gt / s_gt.clamp(min=1e-6)).clamp(-_RATIO_CLAMP, _RATIO_CLAMP)

    loss_e1 = F.l1_loss(g1_pred[active], g1_gt[active])
    loss_e2 = F.l1_loss(g2_pred[active], g2_gt[active])

    return (loss_e1 + loss_e2) / 2.0


def gumbel_loss(pred, target, mask, tolerance=3, alpha=2.0):
    B, S, N_ARGS, N_CLASS = pred.shape
    target += 1

    pred_probs = F.softmax(pred, dim=-1)  # (batchsize, 60, 16, 257)

    target_dist = torch.zeros_like(pred_probs)  # (batchsize, 60, 16, 257)

    for shift in range(-tolerance, tolerance + 1):
        shifted_target = torch.clamp(target + shift, 0, N_CLASS - 1)
        weight = torch.exp(torch.tensor(-alpha * abs(shift), dtype=torch.float32, device='cuda'))
        weight_tensor = weight.unsqueeze(0).expand(B, S, N_ARGS)  # (batchsize, 60, 16)
        target_dist.scatter_(3, shifted_target.unsqueeze(-1), weight_tensor.unsqueeze(-1))

    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)

    loss_per_position = -torch.sum(target_dist * torch.log(pred_probs + 1e-9), dim=-1)  # (batchsize, 60, 16)
    loss_valid = (loss_per_position * mask).sum() / mask.sum()

    return loss_valid