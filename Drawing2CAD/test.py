from tqdm import tqdm
import os
from dataset.bi_sequence_dataset import get_dataloader
from config.config import Config
from config.file_utils import ensure_dir
from trainer.trainer import TrainerED
import torch
import numpy as np
import h5py
import json
from config.macro import *

def main():
    cfg = Config('test')

    tr_agent = TrainerED(cfg)

    # load from checkpoint
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create dataloader
    test_loader = get_dataloader("test", cfg)
    print(f"Total number of test batch: {len(test_loader)}")

    # evaluate
    for i, data in enumerate(test_loader):
        cad_data = data['cad']
        gt_vec = torch.cat([cad_data['command'].unsqueeze(-1), cad_data['args']], dim=-1).squeeze(1).detach().cpu().numpy()
        cad_commands_ = gt_vec[:, :, 0]
        batch_size = cad_data['command'].shape[0]

        with torch.no_grad():
            if cfg.autoregressive:
                # Use autoregressive generation
                out_command, out_args = tr_agent.generate_autoregressive(data)
                batch_outputs = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1).detach().cpu().numpy()
            else:
                # Use parallel generation (original)
                outputs, _ = tr_agent.forward(data)
                batch_outputs = tr_agent.logits2vec(outputs)

        pbar = tqdm(total=batch_size, desc='BATCH[{}]'.format(i))
        for j in range(batch_size):
            out = batch_outputs[j]
            seq_len = cad_commands_[j].tolist().index(CAD_EOS_IDX)

            data_id = data['id'][j].split('/')[-1]
        
            save_dir = os.path.join(cfg.exp_dir, 'test_results')
            ensure_dir(save_dir)
            save_path = os.path.join(save_dir, '{}_vec.h5'.format(data_id))
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('out_vec', data=out[:seq_len], dtype=np.int32)
                f.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int32)
            pbar.update(1)


if __name__ == '__main__':
    main()

