#!/usr/bin/env python3
"""
Training script for Transformer Language Model

This script provides a streamlined training pipeline for large language models.
It supports both configuration file and command-line argument modes for flexible hyperparameter control.

Features:
- Memory-efficient data loading with np.memmap
- Checkpoint serialization and resuming
- Training logging
- Support for Weights & Biases integration
- Flexible hyperparameter configuration

Usage:
    # Using configuration file
    python train.py --config config.yaml
    
    # Using command line arguments
    python train.py --data_path data/train_tokens.npy --vocab_size 32000 --batch_size 32
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

# Import our custom modules
from Transformer import (
    transformer_lm, 
    cross_entropy, 
    AdamW, 
    learning_rate_schedule,
    gradient_clipping,
    data_loading,
    checkpoint,
    load_checkpoint
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class MemoryEfficientDataset(Dataset):
    """
    Memory-efficient dataset using np.memmap for large datasets.
    This allows loading datasets larger than available RAM.
    """
    
    def __init__(self, data_path: str, context_length: int, device: str = 'cpu'):
        """
        Initialize the dataset with memory-mapped data.
        
        Args:
            data_path: Path to the tokenized data file (.npy)
            context_length: Length of context window
            device: Device to place tensors on
        """
        self.data_path = data_path
        self.context_length = context_length
        self.device = device
        
        # Load data using memory mapping
        logger.info(f"Loading data from {data_path} using memory mapping...")
        self.data = np.load(data_path, mmap_mode='r')
        self.data_length = len(self.data)
        self.max_start_idx = self.data_length - context_length - 1
        
        logger.info(f"Dataset loaded: {self.data_length} tokens, context length: {context_length}")
    
    def __len__(self) -> int:
        """Return the number of possible sequences."""
        return self.max_start_idx + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target tokens.
        
        Args:
            idx: Starting index for the sequence
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        if idx > self.max_start_idx:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Extract sequence and target
        sequence = self.data[idx:idx + self.context_length]
        target = self.data[idx + 1:idx + self.context_length + 1]
        
        # Convert to tensors and move to device
        sequence_tensor = torch.from_numpy(sequence.astype(np.int64)).to(self.device)
        target_tensor = torch.from_numpy(target.astype(np.int64)).to(self.device)
        
        return sequence_tensor, target_tensor


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        # Model parameters
        self.vocab_size = config_dict.get('vocab_size', 32000)
        self.context_length = config_dict.get('context_length', 512)
        self.num_layers = config_dict.get('num_layers', 6)
        self.d_model = config_dict.get('d_model', 512)
        self.num_heads = config_dict.get('num_heads', 8)
        self.d_ff = config_dict.get('d_ff', 2048)
        self.eps = config_dict.get('eps', 1e-5)
        self.rope_theta = config_dict.get('rope_theta', 10000)
        
        # Training parameters
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.weight_decay = config_dict.get('weight_decay', 0.1)
        self.beta1 = config_dict.get('beta1', 0.9)
        self.beta2 = config_dict.get('beta2', 0.999)
        self.eps_optimizer = config_dict.get('eps_optimizer', 1e-8)
        self.max_grad_norm = config_dict.get('max_grad_norm', 1.0)
        
        # Learning rate schedule
        self.warmup_steps = config_dict.get('warmup_steps', 1000)
        self.max_steps = config_dict.get('max_steps', 10000)
        self.min_lr = config_dict.get('min_lr', 1e-6)
        
        # Data parameters
        self.data_path = config_dict.get('data_path', 'data/train_tokens.npy')
        self.valid_data_path = config_dict.get('valid_data_path', 'data/valid_tokens.npy')

        # Checkpointing and logging
        self.checkpoint_dir = config_dict.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_interval = config_dict.get('checkpoint_interval', 1000)
        self.log_interval = config_dict.get('log_interval', 100)
        
        # Weights & Biases
        self.use_wandb = config_dict.get('use_wandb', False)
        self.wandb_project = config_dict.get('wandb_project', 'transformer-training')
        self.wandb_run_name = config_dict.get('wandb_run_name', None)
        
        # Device
        self.device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = config_dict.get('dtype', 'float32')
        
        # Resume training
        self.resume_from_checkpoint = config_dict.get('resume_from_checkpoint', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class Trainer:
    """Main training class that orchestrates the training process."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize dataset
        self.dataset = self._create_dataset(self.config.data_path)
        self.valid_dataset = self._create_dataset(self.config.valid_data_path)
        # Initialize data loader
        self.data_loader = self._create_dataloader(self.dataset)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize Weights & Biases if enabled
        if config.use_wandb:
            self._init_wandb()
    
    def _create_model(self) -> nn.Module:
        """Create the transformer model."""
        logger.info("Creating transformer model...")
        model = transformer_lm(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            eps=self.config.eps,
            RoPE_theta=self.config.rope_theta,
            device=self.device,
            dtype=self.dtype
        )
        
        # Move model to device
        model = model.to(device=self.device, dtype=self.dtype)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        logger.info("Creating AdamW optimizer...")
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps_optimizer,
            weight_decay=self.config.weight_decay
        )
    
    def _create_dataset(self, dataset_path) -> MemoryEfficientDataset:
        """Create dataset for training."""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        
        return MemoryEfficientDataset(
            data_path=dataset_path,
            context_length=self.config.context_length,
            device=self.device
        )
    
    def _create_dataloader(self, data) -> DataLoader:
        """Create data loader for training."""
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with memmap
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.to_dict()
        )
    
    def _calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate cross-entropy loss."""
        return cross_entropy(logits, targets)
    
    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        sequences, targets = batch
        logits = self.model(sequences)

        # Calculate loss
        loss = self._calculate_loss(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    
    def _update_learning_rate(self):
        """Update learning rate based on schedule."""
        if self.step < self.config.max_steps:
            lr = learning_rate_schedule(
                self.step,
                self.config.learning_rate,
                self.config.min_lr,
                self.config.warmup_steps,
                self.config.max_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{self.step}.pt"
        )
        
        checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=self.step,
            out=checkpoint_path
        )
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.step = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        logger.info(f"Resumed training from step {self.step}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and optionally to W&B."""
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.step)

    def validate(self):
        with torch.no_grad():
            loss = 0.0
            t = 0
            dataloader = self._create_dataloader(self.valid_dataset)
            while t < 10:
                dataloader = iter(dataloader)
                batch = next(dataloader)
                sequences, targets = batch
                logits = self.model(sequences)
                loss += self._calculate_loss(logits, targets)   
                t += 1
            return loss / t           

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Load checkpoint if resuming
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        data_iter = iter(self.data_loader)
        running_loss = 0.0
        
        while self.step < self.config.max_steps:
            try:
                # Get next batch
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator for next epoch
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
                self.epoch += 1
                logger.info(f"Starting epoch {self.epoch}")
            
            # Training step
            loss = self._train_step(batch)
            running_loss += loss
            
            # Update learning rate
            self._update_learning_rate()
            
            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                lr = self.optimizer.param_groups[0]['lr']
                valid_loss = self.validate()
                metrics = {
                    'train_loss': avg_loss,
                    'valid_loss': valid_loss,
                    'learning_rate': lr,
                    'step': self.step,
                    'epoch': self.epoch
                }
                self._log_metrics(metrics)
                running_loss = 0.0
            
            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0 and self.step > 0:
                self._save_checkpoint()
            
            self.step += 1
        
        # Final checkpoint
        self._save_checkpoint()
        logger.info("Training completed!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        # Model parameters
        'vocab_size': 32000,
        'context_length': 512,
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'eps': 1e-5,
        'rope_theta': 10000,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps_optimizer': 1e-8,
        'max_grad_norm': 1.0,
        
        # Learning rate schedule
        'warmup_steps': 1000,
        'max_steps': 10000,
        'min_lr': 1e-6,
        
        # Data parameters
        'data_path': 'data/train_tokens.npy',
        'valid_data_path': 'data/valid_tokens.npy',
        # Checkpointing and logging
        'checkpoint_dir': 'checkpoints',
        'checkpoint_interval': 1000,
        'log_interval': 100,
        
        # Weights & Biases
        'use_wandb': False,
        'wandb_project': 'transformer-training',
        'wandb_run_name': None,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'float32',
        
        # Resume training
        'resume_from_checkpoint': None
    }


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    
    # Configuration file option
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    
    # Command line arguments (will override config file)
    parser.add_argument('--data_path', type=str, help='Path to training data (.npy file)')
    parser.add_argument('--valid_data_path', type=str, help='Path to validation data (.npy file)')
    parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, help='Context length')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_steps', type=int, help='Maximum training steps')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, help='W&B project name')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
    else:
        config_dict = create_default_config()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config_dict[key] = value
    
    # Create training configuration
    config = TrainingConfig(config_dict)
    
    # Print configuration
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
