"""
Configuration loader for prediction_with_action project.
Handles loading YAML configs and merging with command line arguments.
"""

import yaml
import argparse
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Utility class for loading and managing configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        # If path is relative, look in config_dir
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config or {}
    
    def flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested config dictionary for easier access."""
        flat_config = {}
        
        for key, value in config.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self.flatten_config(value, new_key))
            else:
                flat_config[new_key] = value
                
        return flat_config
    
    def create_args_from_config(self, config: Dict[str, Any]) -> argparse.Namespace:
        """Convert config dictionary to argparse.Namespace object."""
        flat_config = self.flatten_config(config)
        
        # Map flattened keys to argument names
        args_dict = {}
        for key, value in flat_config.items():
            # Convert nested keys to argument format
            arg_name = key.replace('_', '-')
            args_dict[key.replace('-', '_')] = value
            
        return argparse.Namespace(**args_dict)
    
    def merge_config_with_args(self, config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
        """Merge YAML config with command line arguments, giving priority to CLI args."""
        flat_config = self.flatten_config(config)
        
        # Start with config values
        merged_dict = {}
        
        # Map config keys to argument names
        for key, value in flat_config.items():
            arg_key = key.replace('-', '_')
            merged_dict[arg_key] = value
        
        # Override with command line arguments (only if they were explicitly set)
        args_dict = vars(args)
        for key, value in args_dict.items():
            # Only override if the argument was explicitly provided
            # This is a simple heuristic - you might want to improve this
            if value is not None:
                merged_dict[key] = value
        
        return argparse.Namespace(**merged_dict)


def load_config(config_file: str = "default.yaml", 
                config_dir: str = "configs",
                args: Optional[argparse.Namespace] = None) -> argparse.Namespace:
    """
    Main function to load configuration.
    
    Args:
        config_file: Name of the YAML config file
        config_dir: Directory containing config files
        args: Command line arguments (optional)
    
    Returns:
        argparse.Namespace with merged configuration
    """
    loader = ConfigLoader(config_dir)
    
    # Load YAML config
    config = loader.load_yaml_config(config_file)
    
    if args is None:
        # If no CLI args provided, just convert config to Namespace
        return loader.create_args_from_config(config)
    else:
        # Merge config with CLI args
        return loader.merge_config_with_args(config, args)


def save_config(config: argparse.Namespace, save_path: str):
    """Save current configuration to YAML file."""
    config_dict = vars(config)
    
    # Reconstruct nested structure
    nested_config = {
        'training': {},
        'components': {},
        'moe': {},
        'wandb': {}
    }
    
    # Mapping of prefixes to sections
    section_mapping = {
        'training': ['feature_path', 'video_path', 'results_dir', 'model', 'image_size', 
                    'num_classes', 'predict_horizon', 'skip_step', 'epochs', 'global_batch_size',
                    'global_seed', 'num_workers', 'without_ema', 'log_every', 'eval_every',
                    'ckpt_every', 'ckpt_wrapper', 'resume', 'auto_resume', 'learning_rate',
                    'weight_decay', 'adam_beta1', 'adam_beta2'],
        'components': ['vae', 'vae_path', 'dit_init', 'rgb_init', 'attn_mask', 'text_cond', 'clip_path',
                      'text_emb_size', 'use_depth', 'd_hidden_size', 'd_patch_size', 'depth_filter',
                      'action_steps', 'action_dim', 'action_scale', 'absolute_action', 'action_condition',
                      'learnable_action_pos', 'action_loss_lambda', 'action_loss_start'],
        'moe': ['use_moe', 'num_experts', 'moe_top_k', 'aux_loss_weight', 'router_z_loss_weight'],
        'wandb': ['use_wandb', 'wandb_project', 'wandb_run_name']
    }
    
    # Organize config into sections
    for key, value in config_dict.items():
        placed = False
        for section, keys in section_mapping.items():
            if key in keys:
                nested_config[section][key] = value
                placed = True
                break
        
        # If key doesn't fit in predefined sections, put it in training
        if not placed:
            nested_config['training'][key] = value
    
    # Save to YAML
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(nested_config, f, default_flow_style=False, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Test the config loader
    try:
        config = load_config("default.yaml")
        print("Successfully loaded config:")
        print(f"Model: {config.model}")
        print(f"Batch size: {config.global_batch_size}")
        print(f"Use MoE: {config.use_moe}")
    except Exception as e:
        print(f"Error loading config: {e}")
