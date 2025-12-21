# Autoregressive CAD Generation with Sliding Window Attention

## ⚠️ IMPORTANT UPDATE - Inference Fix Applied

**Date**: December 12, 2025  
**Status**: Training unchanged, Inference corrected

## Summary of Changes

This update implements **true autoregressive inference** while keeping the training process unchanged. The model now properly uses its learned sliding window attention mechanisms at test time by generating tokens sequentially and using previously generated tokens as history.

---

## 🎯 Key Improvements

### 1. **Sliding Window Attention**
- Both CommandDecoder and ArgsDecoder now attend to a sliding window of **previous command-argument pairs**
- Window size: **5 pairs** (configurable via `history_window_size` in config)
- Enables local geometric dependencies (e.g., arcs connecting to previous lines)

### 2. **Teacher Forcing During Training**
- Ground truth CAD tokens are embedded and passed as history during training
- Prevents error accumulation during training
- Controlled by `use_teacher_forcing` flag in config (default: True)

### 3. **Autoregressive Generation at Inference**
- New `generate_autoregressive()` method for sequential token generation
- Each position attends to all previously generated tokens
- Early stopping when EOS is generated
- Can be enabled with `--autoregressive` flag at test time

---

## 📝 Modified Files

### 1. **model/model.py**
- **Added**: `MultiheadAttention` import for sliding window attention
- **Modified**: `CommandDecoder` - added history attention mechanism
- **Modified**: `ArgsDecoder` - added history attention mechanism
- **Added**: `embed_commands()` and `embed_args()` methods to SVG2CADTransformer
- **Modified**: `forward()` - now supports teacher forcing with ground truth tokens
- **Added**: `generate_autoregressive()` - new method for autoregressive inference

### 2. **config/config.py**
- **Added**: `history_window_size` - size of sliding window (default: 5)
- **Added**: `use_teacher_forcing` - enable teacher forcing during training (default: True)
- **Added**: `teacher_forcing_ratio` - ratio of teacher forcing (default: 1.0)
- **Added**: `--autoregressive` - command-line flag for autoregressive test mode

### 3. **trainer/trainer.py**
- **Modified**: `forward()` - passes ground truth CAD tokens when teacher forcing is enabled
- **Added**: `generate_autoregressive()` - wrapper for autoregressive generation with masking

### 4. **test.py**
- **Modified**: Test loop now supports both parallel and autoregressive generation modes
- Use `--autoregressive` flag to enable autoregressive generation at test time

---

## 🚀 Usage

### Training (unchanged)
```bash
python train.py --exp_name my_experiment
# Training continues to use teacher forcing with GT tokens
```

### Quick Test (verify autoregressive works)
```bash
python test_autoregressive.py --exp_name epoch_100_shivank --ckpt latest
# Compares parallel vs autoregressive on a few samples
```

### Testing (parallel generation - fast, for comparison)
```bash
python test.py --exp_name epoch_100_shivank --ckpt latest
# Uses parallel generation (ignores sliding window)
```

### Testing (autoregressive generation - RECOMMENDED)
```bash
python test.py --exp_name epoch_100_shivank --ckpt latest --autoregressive
# Uses true autoregressive with sliding window history
# This is the proper way to test the 170-epoch checkpoint
```

---

## ⚙️ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `history_window_size` | 5 | Number of previous cmd-arg pairs to attend to |
| `use_teacher_forcing` | True | Use ground truth history during training |
| `teacher_forcing_ratio` | 1.0 | Ratio of teacher forcing (for scheduled sampling) |
| `--autoregressive` | False | Use autoregressive generation at test time |

---

## 📊 Performance Impact

### Training
- **Speed**: Similar to original (teacher forcing allows parallelization)
- **Memory**: Slightly increased (stores token embeddings)
- **Quality**: Expected improvement due to sequential dependencies

### Inference
- **Parallel mode** (original): ~3 forward passes → **fast**
- **Autoregressive mode** (new): ~121 forward passes → **~40× slower**
- **Quality**: Autoregressive mode should produce more coherent CAD sequences

---

## 🔍 Architecture Changes

### Before (Parallel Generation)
```
Encoder → z_latent → CommandDecoder → all 60 commands (parallel)
                   ↓
                   → ArgsDecoder → all 60 args (parallel)
```

### After (Autoregressive with Sliding Window)
```
Training (Teacher Forcing):
  Encoder → z_latent → CommandDecoder(z, GT_history) → all 60 commands
                     ↓
                     → ArgsDecoder(z, guidance, GT_history) → all 60 args

Inference (Autoregressive):
  for i in range(60):
    cmd_i = CommandDecoder(z, generated_history[:i])
    args_i = ArgsDecoder(z, cmd_i, generated_history[:i])
    generated_history.append((cmd_i, args_i))
```

---

## 🎓 Benefits

1. **Geometric Coherence**: Commands can now depend on previously generated geometry
2. **Constraint Satisfaction**: Better handling of geometric constraints (tangency, alignment)
3. **Flexible Inference**: Choose between fast parallel or high-quality autoregressive generation
4. **Stable Training**: Teacher forcing prevents error accumulation during training
5. **Local Dependencies**: Sliding window captures relevant nearby geometric relationships

---

## 💡 Future Enhancements

1. **Scheduled Sampling**: Gradually reduce teacher forcing ratio during training
2. **KV Caching**: Optimize autoregressive inference with key-value caching
3. **Beam Search**: Generate multiple candidates and select the best
4. **Variable Window Size**: Adaptive window size based on sequence complexity
5. **Hybrid Mode**: Use parallel for draft, autoregressive for refinement

---

## � What Was Fixed

### Previous Issue (Why 170 epochs didn't converge):
```python
# Training: Used GT tokens in sliding window → model learned attention
# Inference: Passed no history → sliding window was BYPASSED
# Result: Trained features were completely ignored at test time!
```

### Current Fix:
```python
# Training: Unchanged (still uses GT tokens)
# Inference: Now uses GENERATED tokens in sliding window
# Result: Trained attention mechanisms are properly activated!
```

The model spent 170 epochs learning sliding window attention, but inference was ignoring it. Now inference properly uses the generated tokens as history, activating the learned attention patterns.

---

## 📊 Expected Behavior with 170-Epoch Checkpoint

### Why This Should Work:
1. ✅ **Model learned sliding window attention** (170 epochs of training)
2. ✅ **Inference now activates that attention** (uses generated history)
3. ✅ **Same attention mechanism** (just different input quality)
4. ⚠️ **Exposure bias exists** (GT in training, generated in inference)
   - This is standard and acceptable in autoregressive models
   - Model should generalize reasonably well

### What to Expect:
- **Much better than current results** (which ignore history completely)
- **Possible error accumulation** (early mistakes affect later predictions)
- **Local geometric dependencies** (arcs connecting to lines, etc.)
- **More coherent CAD sequences** overall

---

## 🐛 Notes

- **Training is unchanged** - no need to retrain
- **170-epoch checkpoint is usable** - contains trained sliding window attention
- Parallel mode still available (fast but ignores learned features)
- **Autoregressive mode is RECOMMENDED** - uses all trained components
- `test_autoregressive.py` helps verify the fix is working
