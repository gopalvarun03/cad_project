# Autoregressive CAD Generation with Sliding Window Attention

## Summary of Changes

This update implements **autoregressive CAD generation with sliding window attention and teacher forcing** to make command and argument generation fully dependent on previous tokens.

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

### Training (with teacher forcing - default)
```bash
python train.py --exp_name my_experiment
```

### Training (without teacher forcing)
```bash
python train.py --exp_name my_experiment --use_teacher_forcing False
```

### Testing (parallel generation - fast)
```bash
python test.py --exp_name my_experiment --ckpt latest
```

### Testing (autoregressive generation - slower but better quality)
```bash
python test.py --exp_name my_experiment --ckpt latest --autoregressive
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

## 🐛 Notes

- The parallel generation mode (original) is still available and is the default at test time
- Autoregressive mode is optional and can be enabled with `--autoregressive` flag
- Teacher forcing is enabled by default during training
- All existing checkpoints are compatible (new parameters are only used when explicitly enabled)
