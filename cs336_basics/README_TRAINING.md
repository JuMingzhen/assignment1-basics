# Transformer Training Guide

This guide explains how to use the training script to train your Transformer language model.

## Overview

The training system consists of several components:

- **`train.py`**: Main training script with configuration and CLI support
- **`config_template.yaml`**: YAML configuration template
- **`config_template.json`**: JSON configuration template

## Features

- ✅ Memory-efficient data loading with `np.memmap`
- ✅ Checkpoint serialization and resuming
- ✅ Training logging
- ✅ Weights & Biases integration
- ✅ Flexible hyperparameter configuration
- ✅ Both configuration file and command-line argument support
- ✅ Gradient clipping and learning rate scheduling

## Quick Start

### 1. Prepare Your Data

Ensure you have tokenized data in `.npy` format. The data should be a 1D numpy array of token IDs.

```python
import numpy as np
# Your tokenized data should look like this:
# data = np.array([1, 2, 3, 4, 5, ...], dtype=np.uint16)
np.save('data/train_tokens.npy', data)
```

### 2. Train with Configuration File

Create a configuration file based on the template:

```bash
# Copy the template
cp cs336_basics/config_template.yaml my_config.yaml

# Edit the configuration
nano my_config.yaml

# Run training
python -m cs336_basics.train --config my_config.yaml
```

### 3. Train with Command Line Arguments

```bash
python -m cs336_basics.train \
    --data_path data/train_tokens.npy \
    --vocab_size 32000 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 10000
```

## Configuration Options

### Model Architecture

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | 32000 |
| `context_length` | Maximum sequence length | 512 |
| `num_layers` | Number of transformer layers | 6 |
| `d_model` | Model dimension | 512 |
| `num_heads` | Number of attention heads | 8 |
| `d_ff` | Feed-forward dimension | 2048 |
| `eps` | RMSNorm epsilon | 1e-5 |
| `rope_theta` | RoPE theta parameter | 10000 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size | 32 |
| `learning_rate` | Initial learning rate | 1e-4 |
| `weight_decay` | Weight decay for AdamW | 0.1 |
| `beta1` | Adam beta1 parameter | 0.9 |
| `beta2` | Adam beta2 parameter | 0.999 |
| `max_grad_norm` | Maximum gradient norm | 1.0 |
| `warmup_steps` | Number of warmup steps | 1000 |
| `max_steps` | Maximum training steps | 10000 |
| `min_lr` | Minimum learning rate | 1e-6 |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Path to training data (.npy file) | `data/train_tokens.npy` |

### Logging and Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint_dir` | Checkpoint directory | `checkpoints` |
| `checkpoint_interval` | Save checkpoint every N steps | 1000 |
| `log_interval` | Log metrics every N steps | 100 |
| `use_wandb` | Enable W&B logging | false |
| `wandb_project` | W&B project name | `transformer-training` |

## Usage Examples

### Example 1: Basic Training

```bash
# Using default configuration
python -m cs336_basics.train
```

### Example 2: Custom Configuration

```yaml
# my_config.yaml
vocab_size: 50000
context_length: 1024
num_layers: 12
d_model: 768
num_heads: 12
batch_size: 16
learning_rate: 2e-4
max_steps: 50000
data_path: "data/my_tokens.npy"
use_wandb: true
wandb_project: "my-transformer-experiment"
```

```bash
python -m cs336_basics.train --config my_config.yaml
```

### Example 3: Resume Training

```bash
python -m cs336_basics.train \
    --config my_config.yaml \
    --resume_from_checkpoint checkpoints/checkpoint_step_5000.pt
```

### Example 4: Command Line Override

```bash
# Override specific parameters from config file
python -m cs336_basics.train \
    --config my_config.yaml \
    --batch_size 64 \
    --learning_rate 1e-3
```

## Memory-Efficient Training

The training script uses `np.memmap` for memory-efficient data loading, allowing you to train on datasets larger than available RAM:

```python
# The MemoryEfficientDataset class handles this automatically
dataset = MemoryEfficientDataset(
    data_path="data/train_tokens.npy",
    context_length=512,
    device="cuda"
)
```

## Checkpointing

The training script automatically saves checkpoints at regular intervals:

- **Regular checkpoints**: Saved every `checkpoint_interval` steps
- **Best model**: Saved when validation loss improves
- **Resume training**: Use `--resume_from_checkpoint` to continue training

Checkpoint files contain:
- Model state dictionary
- Optimizer state dictionary
- Current training step

## Logging

### Console Logging

The script logs training progress to both console and file (`training.log`):

```
2024-01-15 10:30:15 - INFO - train_loss: 3.2456 | learning_rate: 0.0001 | step: 100 | epoch: 0
2024-01-15 10:30:20 - INFO - val_loss: 3.1234 | best_val_loss: 3.1234
```

### Weights & Biases Integration

To enable W&B logging:

1. Install W&B: `pip install wandb`
2. Login: `wandb login`
3. Set `use_wandb: true` in your config or use `--use_wandb`

```bash
python -m cs336_basics.train --use_wandb --wandb_project "my-experiment"
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `batch_size`
   - Reduce `context_length`
   - Use `dtype: "float16"`

2. **Data file not found**:
   - Ensure data files are tokenized using `prepare_data.py`
   - Check file paths in configuration

3. **Vocabulary mismatch**:
   - Ensure `vocab_size` matches your tokenizer
   - Check `vocab_path` and `merges_path` are correct

### Performance Tips

1. **Use appropriate batch size**: Start with 32 and adjust based on GPU memory
2. **Monitor GPU utilization**: Use `nvidia-smi` to check GPU usage
3. **Use mixed precision**: Set `dtype: "float16"` for faster training
4. **Adjust validation frequency**: Reduce `val_interval` for more frequent validation

## File Structure

```
cs336_basics/
├── train.py                 # Main training script
├── prepare_data.py          # Data preparation script
├── config_template.yaml     # YAML configuration template
├── config_template.json     # JSON configuration template
├── Transformer.py           # Transformer implementation
├── Tokenizer.py             # Tokenizer implementation
├── vocab.pkl               # Vocabulary file
├── merges.pkl              # Merges file
└── README_TRAINING.md      # This file

data/
├── train_tokens.npy        # Tokenized training data
└── val_tokens.npy          # Tokenized validation data

checkpoints/
├── checkpoint_step_1000.pt # Regular checkpoints
└── best_model.pt           # Best model checkpoint
```

## Advanced Usage

### Custom Learning Rate Schedules

The script uses a cosine annealing schedule with warmup. You can modify the `learning_rate_schedule` function in `Transformer.py` for custom schedules.

### Custom Loss Functions

You can modify the `_calculate_loss` method in the `Trainer` class to use different loss functions.

### Multi-GPU Training

For multi-GPU training, you can wrap the model with `torch.nn.DataParallel` or use `torch.nn.parallel.DistributedDataParallel`.

## Support

If you encounter issues:

1. Check the logs in `training.log`
2. Verify your configuration parameters
3. Ensure all data files are properly prepared
4. Check GPU memory usage and adjust batch size accordingly
