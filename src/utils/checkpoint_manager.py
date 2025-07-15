#!/usr/bin/env python3
"""
🗃️ CHECKPOINT MANAGEMENT UTILITIES 🗃️
Handle saving, loading, and managing model checkpoints during training.

This module provides utilities for:
- Resuming training from interruptions
- Managing multiple checkpoint versions
- Cleaning up old checkpoints to save disk space
- Validating checkpoint integrity
"""

import os
import json
import shutil
import glob
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

class CheckpointManager:
    """
    🔄 Manage model checkpoints throughout the training process.
    
    This class handles the lifecycle of training checkpoints, including
    automatic cleanup, versioning, and resumption capabilities.
    """
    
    def __init__(self, base_output_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            base_output_dir: Base directory where checkpoints are saved
            max_checkpoints: Maximum number of checkpoints to keep (saves disk space)
        """
        self.base_output_dir = Path(base_output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir = self.base_output_dir / "checkpoints"
        self.metadata_file = self.checkpoints_dir / "checkpoint_metadata.json"
        
        # Create directories if they don't exist
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata or create new
        self.metadata = self._load_metadata()
        
        print(f"🗃️ Checkpoint Manager initialized")
        print(f"   📁 Checkpoints directory: {self.checkpoints_dir}")
        print(f"   📊 Max checkpoints to keep: {self.max_checkpoints}")
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "checkpoints": [],
                "best_checkpoint": None,
                "last_checkpoint": None,
                "training_stats": {}
            }
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self, 
                       model: PreTrainedModel, 
                       tokenizer: PreTrainedTokenizer,
                       step: int,
                       epoch: int,
                       loss: float,
                       eval_loss: Optional[float] = None,
                       learning_rate: Optional[float] = None) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            step: Current training step
            epoch: Current training epoch
            loss: Current training loss
            eval_loss: Current validation loss (if available)
            learning_rate: Current learning rate
            
        Returns:
            Path to the saved checkpoint directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint-step-{step}-epoch-{epoch}-{timestamp}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        print(f"💾 Saving checkpoint: {checkpoint_name}")
        
        try:
            # Create checkpoint directory
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save training metadata
            checkpoint_info = {
                "step": step,
                "epoch": epoch,
                "timestamp": timestamp,
                "loss": loss,
                "eval_loss": eval_loss,
                "learning_rate": learning_rate,
                "path": str(checkpoint_path),
                "size_mb": self._get_directory_size(checkpoint_path)
            }
            
            with open(checkpoint_path / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            # Update metadata
            self.metadata["checkpoints"].append(checkpoint_info)
            self.metadata["last_checkpoint"] = checkpoint_info
            
            # Update best checkpoint if this is better
            if (self.metadata["best_checkpoint"] is None or 
                (eval_loss is not None and 
                 eval_loss < self.metadata["best_checkpoint"].get("eval_loss", float('inf')))):
                self.metadata["best_checkpoint"] = checkpoint_info
                print(f"🏆 New best checkpoint! Eval loss: {eval_loss:.4f}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save updated metadata
            self._save_metadata()
            
            print(f"✅ Checkpoint saved successfully")
            print(f"   📍 Step: {step}, Epoch: {epoch}")
            print(f"   📉 Loss: {loss:.4f}" + (f", Eval Loss: {eval_loss:.4f}" if eval_loss else ""))
            print(f"   💽 Size: {checkpoint_info['size_mb']:.1f} MB")
            
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Clean up partial checkpoint if it exists
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Load a checkpoint for resuming training.
        
        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest
            
        Returns:
            Tuple of (checkpoint_path, checkpoint_info)
        """
        if checkpoint_path is None:
            # Load the latest checkpoint
            if not self.metadata["last_checkpoint"]:
                raise ValueError("No checkpoints found to resume from")
            checkpoint_info = self.metadata["last_checkpoint"]
            checkpoint_path = checkpoint_info["path"]
        else:
            # Load specific checkpoint
            checkpoint_path = Path(checkpoint_path)
            with open(checkpoint_path / "checkpoint_info.json", 'r') as f:
                checkpoint_info = json.load(f)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"🔄 Loading checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   📍 Step: {checkpoint_info['step']}, Epoch: {checkpoint_info['epoch']}")
        print(f"   📉 Loss: {checkpoint_info['loss']:.4f}")
        
        return str(checkpoint_path), checkpoint_info
    
    def get_best_checkpoint(self) -> Optional[Tuple[str, Dict]]:
        """
        Get the best checkpoint based on evaluation loss.
        
        Returns:
            Tuple of (checkpoint_path, checkpoint_info) or None if no checkpoints
        """
        if self.metadata["best_checkpoint"]:
            best_info = self.metadata["best_checkpoint"]
            return best_info["path"], best_info
        return None
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.metadata["checkpoints"]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if we exceed the maximum count."""
        if len(self.metadata["checkpoints"]) <= self.max_checkpoints:
            return
        
        # Sort by step (oldest first)
        sorted_checkpoints = sorted(self.metadata["checkpoints"], key=lambda x: x["step"])
        
        # Keep the most recent checkpoints and the best one
        to_keep = sorted_checkpoints[-self.max_checkpoints:]
        best_checkpoint = self.metadata["best_checkpoint"]
        
        # Make sure we don't delete the best checkpoint
        if best_checkpoint and best_checkpoint not in to_keep:
            to_keep.append(best_checkpoint)
        
        # Delete the rest
        to_delete = [cp for cp in sorted_checkpoints if cp not in to_keep]
        
        for checkpoint in to_delete:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                print(f"🗑️ Removing old checkpoint: {checkpoint_path.name}")
                shutil.rmtree(checkpoint_path)
        
        # Update metadata
        self.metadata["checkpoints"] = to_keep
    
    def _get_directory_size(self, path: Path) -> float:
        """Get the size of a directory in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_training_summary(self) -> Dict:
        """
        Get a summary of the training progress.
        
        Returns:
            Dictionary with training statistics and checkpoint info
        """
        if not self.metadata["checkpoints"]:
            return {"status": "No training started"}
        
        latest = self.metadata["last_checkpoint"]
        best = self.metadata["best_checkpoint"]
        
        summary = {
            "total_checkpoints": len(self.metadata["checkpoints"]),
            "latest_step": latest["step"],
            "latest_epoch": latest["epoch"],
            "latest_loss": latest["loss"],
            "latest_eval_loss": latest.get("eval_loss"),
            "best_eval_loss": best.get("eval_loss") if best else None,
            "best_step": best["step"] if best else None,
            "total_size_mb": sum(cp["size_mb"] for cp in self.metadata["checkpoints"]),
            "checkpoints_dir": str(self.checkpoints_dir)
        }
        
        return summary
    
    def cleanup_all_checkpoints(self, confirm: bool = False):
        """
        Delete all checkpoints and reset metadata.
        
        Args:
            confirm: Must be True to actually delete (safety check)
        """
        if not confirm:
            print("⚠️ This will delete ALL checkpoints. Set confirm=True to proceed.")
            return
        
        print("🗑️ Cleaning up all checkpoints...")
        
        if self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset metadata
        self.metadata = {
            "checkpoints": [],
            "best_checkpoint": None,
            "last_checkpoint": None,
            "training_stats": {}
        }
        self._save_metadata()
        
        print("✅ All checkpoints cleaned up")


def resume_training_from_checkpoint(checkpoint_manager: CheckpointManager,
                                  model_class,
                                  tokenizer_class,
                                  checkpoint_path: Optional[str] = None):
    """
    Utility function to resume training from a checkpoint.
    
    Args:
        checkpoint_manager: Initialized CheckpointManager
        model_class: Model class to instantiate
        tokenizer_class: Tokenizer class to instantiate
        checkpoint_path: Specific checkpoint path, or None for latest
        
    Returns:
        Tuple of (model, tokenizer, checkpoint_info)
    """
    # Load checkpoint info
    checkpoint_path, checkpoint_info = checkpoint_manager.load_checkpoint(checkpoint_path)
    
    # Load model and tokenizer from checkpoint
    print(f"🔄 Resuming from step {checkpoint_info['step']}, epoch {checkpoint_info['epoch']}")
    
    model = model_class.from_pretrained(checkpoint_path)
    tokenizer = tokenizer_class.from_pretrained(checkpoint_path)
    
    return model, tokenizer, checkpoint_info