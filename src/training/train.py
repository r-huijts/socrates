#!/usr/bin/env python3
"""
🎭 SOCRATIC TUTOR TRAINING SCRIPT 🎭
Main fine-tuning script for creating a Socratic AI tutor using Qwen2.5.

Automatically adapts between:
- Unsloth (for NVIDIA/Intel GPU environments like RunPod)
- Standard HuggingFace (for local development on MacBook/CPU)
"""

import os
import sys
import argparse
import torch
import warnings
from typing import Optional, Union, Tuple
from datetime import datetime

# Check device capabilities and import accordingly
DEVICE_CAPABILITIES = {
    "has_cuda": torch.cuda.is_available(),
    "has_mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
}

print("🔍 Detecting compute environment...")
print(f"   CUDA available: {DEVICE_CAPABILITIES['has_cuda']}")
print(f"   MPS available: {DEVICE_CAPABILITIES['has_mps']}")
print(f"   Device count: {DEVICE_CAPABILITIES['device_count']}")

# Conditional imports based on available hardware
try:
    if DEVICE_CAPABILITIES["has_cuda"]:
        print("🔥 CUDA detected - importing Unsloth for optimized GPU training")
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from trl import SFTTrainer
        USE_UNSLOTH = True
        TRAINING_MODE = "unsloth_gpu"
    else:
        print("🍎 No CUDA detected - using standard HuggingFace transformers")
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        USE_UNSLOTH = False
        TRAINING_MODE = "huggingface_cpu_mps"
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Falling back to HuggingFace transformers...")
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    USE_UNSLOTH = False
    TRAINING_MODE = "huggingface_fallback"

# Standard imports that work everywhere
import wandb
from datasets import load_dataset

# Local imports (add path manipulation)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config_loader import ConfigLoader
from src.training.dataset_preparation import SocraticDatasetPreparator

# Suppress some noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

print(f"✅ Using training mode: {TRAINING_MODE}")

class SocraticTrainer:
    """
    🧠 Universal trainer class that adapts to available hardware.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the training configuration YAML file
        """
        self.config_loader = ConfigLoader()
        
        # Load structured configs
        self.training_config = self.config_loader.load_training_config()
        self.model_config, self.lora_config, self.quant_config = self.config_loader.load_model_config()
        
        # Load full YAML for sections not in dataclasses
        full_training_yaml = self.config_loader.load_yaml("config/training/training_config.yaml")
        self.optimizer_config = full_training_yaml['optimizer']
        self.data_config = full_training_yaml['data'] 
        self.wandb_config = full_training_yaml.get('wandb', {})
        
        # Adapt config based on training mode
        self._adapt_config_for_environment()
        
        # Extract key configuration values using proper access methods
        self.output_dir = self.training_config.output_dir
        self.max_seq_length = self.data_config['max_seq_length']
        
        # Initialize components (will be set during setup)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        print(f"🎭 Socratic Trainer initialized in {TRAINING_MODE} mode!")
        
    def _adapt_config_for_environment(self):
        """Automatically adjust config based on detected environment."""
        if TRAINING_MODE == "huggingface_cpu_mps":
            print("🔧 Adapting config for local CPU/MPS environment...")
            
            # Use smaller model for local testing
            original_model = self.model_config.name
            if "7B" in original_model:
                self.model_config.name = "Qwen/Qwen2.5-1.5B-Instruct"
                print(f"   Model: {original_model} → {self.model_config.name}")
            
            # Reduce sequence length
            if self.data_config['max_seq_length'] > 512:
                self.data_config['max_seq_length'] = 512
                print(f"   Max sequence length: → 512")
            
            # Reduce batch size
            if self.training_config.per_device_train_batch_size > 1:
                self.training_config.per_device_train_batch_size = 1
                self.training_config.per_device_eval_batch_size = 1
                print(f"   Batch size: → 1")
            
            # Fewer epochs for testing
            if self.training_config.num_train_epochs > 1:
                self.training_config.num_train_epochs = 1
                print(f"   Epochs: → 1 (testing mode)")
        
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer using the appropriate method."""
        model_name = self.model_config.name
        print(f"🤖 Loading {model_name} using {TRAINING_MODE}...")
        
        if USE_UNSLOTH:
            self._setup_unsloth_model()
        else:
            self._setup_huggingface_model()
            
        print(f"✅ Model loaded successfully!")
        print(f"📏 Max sequence length: {self.max_seq_length}")
    
    def _setup_unsloth_model(self):
        """Setup model using Unsloth (GPU environments)."""
        model_name = self.model_config.name
        
        # Load model and tokenizer with Unsloth's FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=True,  # 4-bit quantization for memory efficiency
        )
        
        # Configure LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        # Apply chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5"
        )
        
        print(f"🔧 LoRA rank: {self.lora_config.r}")
    
    def _setup_huggingface_model(self):
        """Setup model using standard HuggingFace (CPU/MPS environments)."""
        model_name = self.model_config.name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        device_map = "auto" if DEVICE_CAPABILITIES["has_cuda"] else None
        torch_dtype = torch.float16 if DEVICE_CAPABILITIES["has_cuda"] else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        
        # Apply LoRA using PEFT
        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Simple chat template setup (without Unsloth's fancy method)
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Basic Qwen2.5 chat template
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    
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
            # Use synthetic Socratic dialogues from data/synthetic directory
            print("🎲 Loading synthetic Socratic dialogues from data/synthetic/...")
            data_files = {
                'train': [
                    'data/synthetic/general_examples.json',
                    'data/synthetic/learning_examples.json', 
                    'data/synthetic/programming_examples.json'
                ]
            }
            raw_dataset = load_dataset('json', data_files=data_files)
        
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
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_steps=self.training_config.warmup_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            optim=self.optimizer_config['type'],
            lr_scheduler_type=self.optimizer_config['lr_scheduler_type'],
            report_to="wandb" if self.wandb_config else "none",
            run_name=f"socratic-tutor-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            seed=42,  # Reproducibility
            data_seed=42,
            remove_unused_columns=False,  # Keep all columns for SFTTrainer
        )
    
    def setup_trainer(self):
        """Initialize trainer using the appropriate method."""
        print("🏋️ Setting up trainer...")
        
        training_args = self.setup_training_arguments()
        
        if USE_UNSLOTH:
            self._setup_unsloth_trainer(training_args)
        else:
            self._setup_huggingface_trainer(training_args)
        
        print("✅ Trainer configured successfully!")
    
    def _setup_unsloth_trainer(self, training_args):
        """Setup SFTTrainer for Unsloth."""
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            args=training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )
    
    def _setup_huggingface_trainer(self, training_args):
        """Setup standard Trainer for HuggingFace."""
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
    
    def setup_wandb(self):
        """
        Initialize Weights & Biases for experiment tracking.
        """
        if self.wandb_config:
            print(f"📈 Initializing Weights & Biases...")
            print(f"   Project: {self.wandb_config['project']}")
            print(f"   Run: {self.wandb_config['run_name']}")
            
            wandb.init(
                project=self.wandb_config['project'],
                name=self.wandb_config['run_name'],
                config={
                    **self.training_config.__dict__,
                    **self.lora_config.__dict__,
                    **self.data_config,
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
            self.prepare_dataset()
            self.setup_trainer()
            self.setup_wandb()
            
            print("\n🚀 Beginning training process...")
            print(f"📁 Model will be saved to: {self.output_dir}")
            print(f"⏱️  Training for {self.training_config.num_train_epochs} epochs")
            
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
        
        if USE_UNSLOTH:
            self.model.save_pretrained_merged(
                hf_output_dir,
                tokenizer=self.tokenizer,
                save_method="merged_16bit",  # Save in 16-bit for deployment
            )
        
        print(f"✅ Model saved to:")
        print(f"   Unsloth format: {self.output_dir}")
        if USE_UNSLOTH:
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