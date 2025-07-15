#!/usr/bin/env python3
"""
🖥️ DEVICE DETECTION AND CONFIGURATION 🖥️
Automatically detect and configure the best available compute device.

Handles the differences between:
- MacBook (CPU only, MPS maybe)
- RunPod (CUDA GPUs)
- Other cloud providers
"""

import torch
import platform
import warnings
from typing import Dict, Any, Optional

class DeviceManager:
    """
    🔧 Automatically configure training for the current hardware environment.
    """
    
    def __init__(self):
        self.device_info = self._detect_device()
        self.is_local_machine = self._is_local_machine()
        
    def _detect_device(self) -> Dict[str, Any]:
        """Detect the best available device and its capabilities."""
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            info.update({
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "compute_capability": torch.cuda.get_device_properties(0).major,
                "recommended_for": "full_training"
            })
        
        # Check for MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            info.update({
                "device_type": "mps",
                "device_name": "Apple Silicon GPU",
                "device_count": 1,
                "memory_gb": "shared",  # Unified memory
                "recommended_for": "light_testing"
            })
        
        # Fallback to CPU
        else:
            info.update({
                "device_type": "cpu",
                "device_name": platform.processor() or "Unknown CPU",
                "device_count": torch.get_num_threads(),
                "memory_gb": "system_ram",
                "recommended_for": "pipeline_testing_only"
            })
        
        return info
    
    def _is_local_machine(self) -> bool:
        """Detect if we're running on a local development machine vs. cloud."""
        # Simple heuristics to detect local vs. cloud
        indicators = [
            self.device_info["device_type"] == "cpu",
            self.device_info["device_type"] == "mps",
            "Darwin" in self.device_info["platform"],  # macOS
            self.device_info.get("memory_gb", 0) < 16 if isinstance(self.device_info.get("memory_gb"), (int, float)) else False
        ]
        
        return any(indicators)
    
    def get_training_config_adjustments(self) -> Dict[str, Any]:
        """
        Get recommended training configuration adjustments based on hardware.
        """
        if self.device_info["device_type"] == "cuda":
            # Full GPU training (RunPod, etc.)
            return {
                "load_in_4bit": True,
                "max_seq_length": 2048,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "fp16": True,
                "dataloader_num_workers": 4,
                "recommended_model_size": "7B",
                "training_mode": "full"
            }
        
        elif self.device_info["device_type"] == "mps":
            # Apple Silicon - light testing
            return {
                "load_in_4bit": False,  # MPS doesn't support 4-bit yet
                "max_seq_length": 512,   # Shorter sequences
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "fp16": False,  # MPS can be finicky with fp16
                "dataloader_num_workers": 2,
                "recommended_model_size": "1.5B",
                "training_mode": "test_pipeline"
            }
        
        else:
            # CPU - pipeline testing only
            return {
                "load_in_4bit": False,
                "max_seq_length": 256,   # Very short sequences
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "fp16": False,
                "dataloader_num_workers": 1,
                "recommended_model_size": "0.5B",
                "training_mode": "validate_only"
            }
    
    def print_device_info(self):
        """Print detailed device information."""
        print("=" * 60)
        print("🖥️  DEVICE DETECTION REPORT")
        print("=" * 60)
        
        print(f"🔧 Platform: {self.device_info['platform']} ({self.device_info['architecture']})")
        print(f"🐍 Python: {self.device_info['python_version']}")
        print(f"🔥 PyTorch: {self.device_info['pytorch_version']}")
        print()
        
        device_type = self.device_info['device_type'].upper()
        print(f"⚡ Device Type: {device_type}")
        print(f"📱 Device Name: {self.device_info['device_name']}")
        print(f"🔢 Device Count: {self.device_info['device_count']}")
        print(f"💾 Memory: {self.device_info['memory_gb']}")
        print(f"🎯 Recommended For: {self.device_info['recommended_for']}")
        print()
        
        if self.is_local_machine:
            print("🏠 DETECTED: Local development machine")
            print("💡 RECOMMENDATION: Use for pipeline testing only")
            print("🚀 For full training, use RunPod or similar GPU cloud")
        else:
            print("☁️  DETECTED: Cloud/remote training environment")
            print("🚀 RECOMMENDATION: Full training ready!")
        
        print("=" * 60)
        
        # Print training adjustments
        adjustments = self.get_training_config_adjustments()
        print("⚙️  RECOMMENDED TRAINING ADJUSTMENTS:")
        for key, value in adjustments.items():
            print(f"   {key}: {value}")
        print("=" * 60)
    
    def get_model_name_for_environment(self, base_model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
        """
        Get the appropriate model size for the current environment.
        """
        adjustments = self.get_training_config_adjustments()
        recommended_size = adjustments["recommended_model_size"]
        
        if recommended_size == "0.5B":
            return "Qwen/Qwen2.5-0.5B-Instruct"
        elif recommended_size == "1.5B":
            return "Qwen/Qwen2.5-1.5B-Instruct"
        elif recommended_size == "7B":
            return base_model  # Use the original
        else:
            return base_model
    
    def should_run_training(self) -> bool:
        """
        Determine if we should actually run training or just test the pipeline.
        """
        training_mode = self.get_training_config_adjustments()["training_mode"]
        return training_mode == "full"