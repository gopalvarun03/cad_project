from tqdm import tqdm
import os
from config.config import Config
from config.file_utils import ensure_dir
from trainer.trainer import TrainerED
import torch
import numpy as np
import h5py
from config.macro import *

def main():
    # --- Configuration and Model Setup (Keep) ---
    cfg = Config('test')
    tr_agent = TrainerED(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()
    
    # --- SPECIFY INPUT PATH HERE ---
    # This is the path to the main folder containing all your input subfolders.
    DATASET_ROOT_PATH = '/Users/bhavithakakkirala/Documents/data2/svg_vec' # <<<< CHANGE THIS PATH
    
    # --- SPECIFY CUSTOM OUTPUT PATH HERE ---
    # The new, separate folder where ALL .h5 results will be saved.
    CUSTOM_OUTPUT_ROOT_PATH = '/Users/bhavithakakkirala/Documents/data2' # <<<< CHANGE THIS PATH
    
    # Ensure the custom output directory exists
    ensure_dir(CUSTOM_OUTPUT_ROOT_PATH)
    
    print(f"Starting processing in input directory: {DATASET_ROOT_PATH}")
    print(f"Saving all outputs to: {CUSTOM_OUTPUT_ROOT_PATH}")
    
    # 1. Get a list of all subdirectories (data points)
    subfolders = [d for d in os.listdir(DATASET_ROOT_PATH) 
                  if os.path.isdir(os.path.join(DATASET_ROOT_PATH, d))]
    
    if not subfolders:
        print(f"Error: No subfolders found in {DATASET_ROOT_PATH}. Exiting.")
        return

    pbar = tqdm(total=len(subfolders), desc='Processing Folders')
    
    for folder_name in subfolders:
        folder_path = os.path.join(DATASET_ROOT_PATH, folder_name)
        
        # 2. Find the .npy file inside the current subfolder
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        if not npy_files:
            print(f"Warning: No .npy file found in {folder_name}. Skipping.")
            pbar.update(1)
            continue
            
        input_npy_path = os.path.join(folder_path, npy_files[0])
        
        # --- NEW OUTPUT PATH CONSTRUCTION ---
        # The output file name will be '{folder_name}_output.h5' and it is placed 
        # inside the CUSTOM_OUTPUT_ROOT_PATH.
        output_h5_path = os.path.join(CUSTOM_OUTPUT_ROOT_PATH, f'{folder_name}_output.h5')

        # --- Load .npy Input Data ---
        try:
            input_data_np = np.load(input_npy_path, allow_pickle=True)
            input_tensor = torch.from_numpy(input_data_np).unsqueeze(0).to(tr_agent.device).float()
            
            # NOTE: 'input_features' is a placeholder key. Change this key if necessary.
            data_for_forward = {
                'input_features': input_tensor,
            }
            
        except Exception as e:
            print(f"Error loading/processing {input_npy_path}: {e}. Skipping.")
            pbar.update(1)
            continue

        # --- Evaluate and Generate Output ---
        with torch.no_grad():
            outputs, _ = tr_agent.forward(data_for_forward) 
            batch_outputs = tr_agent.logits2vec(outputs)

        # --- Process and Save .h5 Output ---
        out_vec = batch_outputs[0]
        
        try:
            seq_end_idx = out_vec[:, 0].tolist().index(CAD_EOS_IDX)
            seq_to_save = out_vec[:seq_end_idx + 1] 
        except ValueError:
            seq_to_save = out_vec

        # Save the output vector to the custom path
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('out_vec', data=seq_to_save, dtype=np.int32)

        pbar.set_postfix(file=folder_name)
        pbar.update(1)

    pbar.close()
    print(f"\nFinished processing all folders. Results saved to {CUSTOM_OUTPUT_ROOT_PATH}")

if __name__ == '__main__':
    main()