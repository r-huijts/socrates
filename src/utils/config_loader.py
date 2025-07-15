import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
    max_length: int
    torch_dtype: str
    device_map: str
    
@dataclass
class LoRAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    
@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool
    
@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a YAML file"""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_model_config(self) -> tuple[ModelConfig, LoRAConfig, QuantizationConfig]:
        """Load model configuration from YAML"""
        config_path = self.config_dir / "model" / "qwen_config.yaml"
        config = self.load_yaml(config_path)
        
        model_config = ModelConfig(**config['model'])
        lora_config = LoRAConfig(**config['lora'])
        quant_config = QuantizationConfig(**config['quantization'])
        
        return model_config, lora_config, quant_config
    
    # Add this method to ConfigLoader class
    def load_config_for_environment(self, device_manager, config_type: str = "training"):
        """
        Load the appropriate config based on detected environment.
        
        Args:
            device_manager: DeviceManager instance
            config_type: "training" or "model"
        """
        if device_manager.is_local_machine:
            config_file = f"config/{config_type}/{config_type}_config_local.yaml"
            print(f"🏠 Loading LOCAL config: {config_file}")
        else:
            config_file = f"config/{config_type}/{config_type}_config.yaml"  # Default (RunPod)
            print(f"☁️ Loading CLOUD config: {config_file}")
        
        if config_type == "training":
            return self.load_training_config(config_file)
        else:
            return self.load_model_config(config_file)

    def load_training_config(self) -> TrainingConfig:
        """Load training configuration from YAML"""
        config_path = self.config_dir / "training" / "training_config.yaml"
        config = self.load_yaml(config_path)
        
        return TrainingConfig(**config['training'])