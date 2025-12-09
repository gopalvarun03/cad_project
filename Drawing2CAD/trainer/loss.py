import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask_cad, _get_visibility_mask
from config.macro import CAD_CMD_ARGS_MASK


class NewCADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.cad_n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CAD_CMD_ARGS_MASK))

    def forward(self, outputs, cad_data):
        # Target
        tgt_commands = cad_data["command"].cuda()
        tgt_args = cad_data["args"].cuda()

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask_cad(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        # Prediction
        command_logits = outputs["command_logits"]
        args_logits = outputs["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        loss_args = gumbel_loss(args_logits, tgt_args, mask)

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
    
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