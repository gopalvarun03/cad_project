import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from config.configAE import ConfigAE
from trainer.trainerAE import TrainerAE
from dataset.image_cad_dataset import ImageCADDataset
from model.image_encoder import ImageEncoder

def main():
    # 1. Configuration
    # Handle custom arguments that ConfigAE doesn't know about
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='proj_log/pretrained/model/ckpt_epoch1000.pth', help='Path to pretrained AE checkpoint')
    
    # Parse known args (ckpt_path) and keep the rest for ConfigAE
    args, unknown = parser.parse_known_args()
    
    # Modify sys.argv to exclude our custom args so ConfigAE doesn't error
    sys.argv = [sys.argv[0]] + unknown
    
    # Load ConfigAE (parses remaining args like --data_root, --batch_size, --nr_epochs)
    cfg = ConfigAE('train')
    
    # Paths
    # Assuming data2 structure: data2/cad_vec and data2/features
    feature_root = os.path.join(cfg.data_root, "features")
    
    # 2. Load Teacher (Pretrained AE)
    print(f"Loading Teacher from {args.ckpt_path}...")
    teacher_agent = TrainerAE(cfg)
    
    # Load checkpoint
    # Note: TrainerAE.load_ckpt expects a path relative to save_dir or absolute
    # We'll try to load it directly
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path)
        teacher_agent.net.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher loaded successfully.")
    else:
        print(f"Warning: Checkpoint {args.ckpt_path} not found. Using random weights (for testing only).")

    teacher = teacher_agent.net
    teacher.cuda()
    teacher.eval()
    
    # Freeze Teacher
    for param in teacher.parameters():
        param.requires_grad = False
        
    # 3. Initialize Student (Image Encoder)
    print("Initializing Student (ImageEncoder)...")
    # No freeze_backbone arg needed anymore
    student = ImageEncoder(latent_dim=256, feature_dim=384)
    student.cuda()
    student.train()
    
    # Optimizer for Student
    optimizer = optim.Adam(student.parameters(), lr=cfg.lr)
    
    # 4. Data Loading
    print("Setting up DataLoader...")
    # ImageCADDataset handles both vector and feature loading
    train_dataset = ImageCADDataset('train', cfg, feature_root)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # 5. Training Loop
    save_dir = os.path.join("proj_log", cfg.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Starting training for {cfg.nr_epochs} epochs...")
    mse_loss = torch.nn.MSELoss()
    
    for epoch in range(cfg.nr_epochs):
        pbar = tqdm(train_loader)
        total_loss = 0
        
        for i, data in enumerate(pbar):
            # Inputs
            commands = data['command'].cuda()
            args_enc = data['args'].cuda()
            features = data['features'].cuda() # (B, 4, 384)
            
            # Teacher Forward (Get Target Latent)
            with torch.no_grad():
                # CADTransformer.forward with encode_mode=True handles transposition and bottleneck
                # Returns (B, 1, 256)
                target_z = teacher(commands, args_enc, encode_mode=True) 
                target_z = target_z.squeeze(1) # (B, 256)
                
            # Student Forward
            pred_z = student(features) # (B, 256)
            
            # Loss
            loss = mse_loss(pred_z, target_z)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{cfg.nr_epochs} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final model
    final_path = os.path.join(save_dir, "student_final.pth")
    torch.save(student.state_dict(), final_path)
    print("Training complete.")

if __name__ == '__main__':
    main()
