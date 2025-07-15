#!/usr/bin/env python3
"""
🎭 SOCRATIC TUTOR TRAINING SCRIPT 🎭
Main fine-tuning script for creating a Socratic AI tutor using Qwen2.5 + Unsloth.

This script orchestrates the entire training process:
- Loads model and tokenizer with LoRA configuration
- Prepares Socratic dialogue dataset
- Runs fine-tuning with proper logging and checkpointing
- Saves the final model for deployment


Usage:
    python src/training/train.py --config config/training/training_config.yaml
"""

import os
import sys
import argparse
import torch
import wandb
import warnings
from typing import Optional
from datetime import datetime
from src.utils.device_utils import DeviceManager

# Unsloth imports for efficient training
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# HuggingFace dataset utilities
from datasets import load_dataset

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config_loader import ConfigLoader
from src.training.dataset_preparation import ModernSocraticDatasetPreparator

# Suppress some noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SocraticTrainer:
    """
    🧠 Main trainer class for the Socratic tutor fine-tuning process.
    
    This class handles the complete training pipeline from model loading
    to final checkpoint saving, with proper logging and error handling.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the training configuration YAML file
        """
        self.config_loader = ConfigLoader()
        self.training_config = self.config_loader.load_training_config(config_path)
        self.model_config = self.config_loader.load_model_config()

        self.device_manager = DeviceManager()
        self.device_manager.print_device_info()
        # Override model config based on device
        recommended_model = self.device_manager.get_model_name_for_environment(
            self.model_config['model']['name']
        )
        if recommended_model != self.model_config['model']['name']:
            print(f"🔄 Switching model for local testing:")
            print(f"   Original: {self.model_config['model']['name']}")
            print(f"   Using: {recommended_model}")
            self.model_config['model']['name'] = recommended_model
        
        # Check if we should actually train
        if not self.device_manager.should_run_training():
            print("⚠️  WARNING: Current environment not suitable for full training!")
            print("   Will run pipeline validation only.")
        
        
        
        
        # Extract key configuration values
        self.output_dir = self.training_config['training']['output_dir']
        self.max_seq_length = self.training_config['data']['max_seq_length']
        
        # Initialize components (will be set during setup)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        
        print("🎭 Socratic Trainer initialized successfully!")
        
    def setup_model_and_tokenizer(self):
        """
        Load and configure the base model with LoRA for efficient fine-tuning.
        """
        print("🤖 Loading Qwen2.5 model with Unsloth optimizations...")
        
        model_name = self.model_config['model']['name']
        
        # Load model and tokenizer with Unsloth's FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=True,  # 4-bit quantization for memory efficiency
        )
        
        # Configure LoRA for parameter-efficient fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.model_config['lora']['rank'],  # LoRA rank
            target_modules=self.model_config['lora']['target_modules'],
            lora_alpha=self.model_config['lora']['alpha'],
            lora_dropout=self.model_config['lora']['dropout'],
            bias="none",  # No bias adaptation
            use_gradient_checkpointing="unsloth",  # Memory optimization
            random_state=3407,  # Fixed seed for reproducibility
        )
        
        # Apply Qwen2.5 chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5"
        )
        
        print(f"✅ Model loaded: {model_name}")
        print(f"📏 Max sequence length: {self.max_seq_length}")
        print(f"🔧 LoRA rank: {self.model_config['lora']['rank']}")
        
    def prepare_dataset(self, data_path: Optional[str] = None):
        """
        Load and prepare the Socratic dialogue dataset for training.
        
        Args:
            data_path: Optional path to custom dataset, otherwise uses synthetic data
        """
        print("📚 Preparing Socratic dialogue dataset...")
        
        # Initialize dataset preparator
        preparator = SocraticDatasetPreparator(self.config_loader)
        
        if data_path:
            # Load custom dataset
            print(f"📁 Loading custom dataset from: {data_path}")
            raw_dataset = load_dataset('json', data_files=data_path)
        else:
            # Use synthetic Socratic dialogues (you'll need to implement this)
            print("🎲 Generating synthetic Socratic dialogues...")
            # For now, we'll assume you have a dataset ready
            # You could integrate with your synthetic data generation here
            raise NotImplementedError("Synthetic data generation not yet implemented")
        
        # Prepare dataset using our preparator
        self.dataset = preparator.prepare_dataset(
            raw_dataset, 
            self.tokenizer
        )
        
        # Print dataset info
        dataset_info = preparator.get_dataset_info(self.dataset)
        print("📊 Dataset Statistics:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
    
    def setup_training_arguments(self):
        """
        Configure training arguments for the SFTTrainer.
        """
        training_config = self.training_config['training']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            optim=self.training_config['optimizer']['type'],
            lr_scheduler_type=self.training_config['optimizer']['lr_scheduler_type'],
            report_to="wandb" if 'wandb' in self.training_config else "none",
            run_name=f"socratic-tutor-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            seed=42,  # Reproducibility
            data_seed=42,
            remove_unused_columns=False,  # Keep all columns for SFTTrainer
        )
    
    def setup_trainer(self):
        """
        Initialize the SFTTrainer with model, dataset, and training arguments.
        """
        print("🏋️ Setting up SFTTrainer...")
        
        training_args = self.setup_training_arguments()
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            args=training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",  # Column containing the formatted conversations
            packing=False,  # Don't pack multiple conversations into one sequence
        )
        
        print("✅ SFTTrainer configured successfully!")
    
    def setup_wandb(self):
        """
        Initialize Weights & Biases for experiment tracking.
        """
        if 'wandb' in self.training_config:
            wandb_config = self.training_config['wandb']
            
            print(f"📈 Initializing Weights & Biases...")
            print(f"   Project: {wandb_config['project']}")
            print(f"   Run: {wandb_config['run_name']}")
            
            wandb.init(
                project=wandb_config['project'],
                name=wandb_config['run_name'],
                config={
                    **self.training_config,
                    **self.model_config,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            print("📊 Skipping W&B initialization (not configured)")
    
    def train(self):
        """
        Execute the main training loop.
        """
        print("\n" + "="*50)
        print("🎭 STARTING SOCRATIC TUTOR TRAINING 🎭")
        print("="*50)
        
        try:
            # Setup all components
            self.setup_model_and_tokenizer()
            # Note: You'll need to provide a dataset path or implement synthetic generation
            # self.prepare_dataset(data_path="path/to/your/socratic_dialogues.json")
            self.setup_trainer()
            self.setup_wandb()
            
            print("\n🚀 Beginning training process...")
            print(f"📁 Model will be saved to: {self.output_dir}")
            print(f"⏱️  Training for {self.training_config['training']['num_train_epochs']} epochs")
            
            # Start training!
            self.trainer.train()
            
            print("\n🎉 Training completed successfully!")
            
            # Save the final model
            self.save_final_model()
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise
        
        finally:
            # Clean up W&B
            if wandb.run:
                wandb.finish()
    
    def save_final_model(self):
        """
        Save the final trained model and tokenizer.
        """
        print("💾 Saving final model...")
        
        # Save with Unsloth's optimized method
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Also save in HuggingFace format for broader compatibility
        hf_output_dir = f"{self.output_dir}_hf"
        os.makedirs(hf_output_dir, exist_ok=True)
        
        self.model.save_pretrained_merged(
            hf_output_dir,
            tokenizer=self.tokenizer,
            save_method="merged_16bit",  # Save in 16-bit for deployment
        )
        
        print(f"✅ Model saved to:")
        print(f"   Unsloth format: {self.output_dir}")
        print(f"   HuggingFace format: {hf_output_dir}")


def main():
    """
    Main entry point for the training script.
    """
    parser = argparse.ArgumentParser(description="Train Socratic AI Tutor")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="Path to training data (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = SocraticTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()