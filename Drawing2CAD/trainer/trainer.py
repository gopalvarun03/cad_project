import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model.model import SVG2CADTransformer
from .base import BaseTrainer
from .loss import NewCADLoss
from .scheduler import GradualWarmupScheduler
from config.macro import *
import torch.nn as nn


class TrainerED(BaseTrainer):
    def build_net(self, cfg):
        self.net = SVG2CADTransformer(cfg).cuda()

        # Total number of model parameters
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"Total parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB with float32)")

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = NewCADLoss(self.cfg).cuda()


    def forward(self, data):
        cad_data = data['cad']
        svg_data = data['svg']
        svg_view = svg_data['view'].cuda() 
        svg_command = svg_data['command'].cuda() 
        svg_args = svg_data['args'].cuda()
        outputs = self.net(svg_view, svg_command, svg_args)

        loss_dict = self.loss_func(outputs, cad_data)

        return outputs, loss_dict
    

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CAD_CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec


    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            cad_data = data['cad']
            svg_data = data['svg']
            with torch.no_grad():
                svg_view = svg_data['view'].cuda()
                svg_command = svg_data['command'].cuda()
                svg_args = svg_data['args'].cuda()
                cad_command = cad_data['command']
                cad_args = cad_data['args']
                outputs = self.net(svg_view, svg_command, svg_args)

            out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
            out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = cad_command.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = cad_args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == CAD_EXT_IDX)
            line_pos = np.where(gt_commands == CAD_LINE_IDX)
            arc_pos = np.where(gt_commands == CAD_ARC_IDX)
            circle_pos = np.where(gt_commands == CAD_CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int32)
            all_ext_args_comp.append(args_comp[ext_pos][:, -CAD_N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :CAD_N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, CAD_N_ARGS_PLANE:CAD_N_ARGS_PLANE+CAD_N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -CAD_N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
