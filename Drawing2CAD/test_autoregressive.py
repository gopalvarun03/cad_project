"""
Quick test script to verify autoregressive generation is working.
Compares parallel vs autoregressive generation on a few samples.
"""

import torch
from dataset.bi_sequence_dataset import get_dataloader
from config.config import Config
from trainer.trainer import TrainerED
import numpy as np

def main():
    cfg = Config('test')
    cfg.batch_size = 4  # Small batch for quick test
    
    # Load model
    tr_agent = TrainerED(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()
    
    # Get a single batch
    test_loader = get_dataloader("test", cfg, shuffle=False)
    data = next(iter(test_loader))
    
    print("=" * 80)
    print("Testing Autoregressive Generation")
    print("=" * 80)
    
    with torch.no_grad():
        # Method 1: Parallel generation (original - fast but ignores history)
        print("\n[1/2] Running parallel generation (original)...")
        outputs_parallel, _ = tr_agent.forward(data)
        commands_parallel = torch.argmax(torch.softmax(outputs_parallel['command_logits'], dim=-1), dim=-1)
        args_parallel = torch.argmax(torch.softmax(outputs_parallel['args_logits'], dim=-1), dim=-1) - 1
        
        # Method 2: Autoregressive generation (new - slow but uses history)
        print("[2/2] Running autoregressive generation (with sliding window history)...")
        commands_auto, args_auto = tr_agent.generate_autoregressive(data)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Compare the two methods
    for i in range(min(cfg.batch_size, 2)):  # Show first 2 samples
        print(f"\n--- Sample {i+1} (ID: {data['id'][i]}) ---")
        
        # Commands
        cmd_par = commands_parallel[i].cpu().numpy()
        cmd_auto = commands_auto[i].cpu().numpy()
        
        # Find sequence length (up to first EOS)
        from config.macro import CAD_EOS_IDX
        try:
            seq_len = np.where(cmd_par == CAD_EOS_IDX)[0][0] + 1
        except:
            seq_len = 10  # Show first 10 if no EOS
        
        print(f"\nCommands (first {seq_len} tokens):")
        print(f"  Parallel:       {cmd_par[:seq_len].tolist()}")
        print(f"  Autoregressive: {cmd_auto[:seq_len].tolist()}")
        
        # Check if identical
        cmd_match = (cmd_par == cmd_auto).sum() / len(cmd_par) * 100
        print(f"  Command match: {cmd_match:.1f}%")
        
        # Args (show first 5 positions)
        args_par = args_parallel[i, :5].cpu().numpy()
        args_aut = args_auto[i, :5].cpu().numpy()
        
        print(f"\nArgs (first 5 positions, showing first 4 params each):")
        for pos in range(min(5, seq_len)):
            print(f"  Pos {pos}:")
            print(f"    Parallel:       {args_par[pos, :4].tolist()}")
            print(f"    Autoregressive: {args_aut[pos, :4].tolist()}")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("""
1. If outputs are IDENTICAL or very similar:
   → Sliding window history has minimal impact (positions are independent)
   → Model might not have learned strong dependencies
   
2. If outputs are DIFFERENT:
   → Autoregressive mode is using the sliding window attention!
   → Generated history influences subsequent predictions
   → This is the expected behavior

3. To properly evaluate quality:
   → Run test.py with --autoregressive flag on full test set
   → Compare CAD outputs visually or with evaluation metrics
   → Check if geometric constraints are better satisfied
    """)
    
    print("\nTo run full test with autoregressive generation:")
    print(f"  python test.py --exp_name {cfg.exp_name} --ckpt {cfg.ckpt} --autoregressive")
    print()

if __name__ == '__main__':
    main()
