import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from unsloth import FastLanguageModel
from src.utils.config_loader import ConfigLoader, ModelConfig, LoRAConfig, QuantizationConfig

class ModelSetup:
    """
    Handles the setup and configuration of Qwen2.5 model for fine-tuning.
    
    This class provides two approaches for model loading:
    1. Unsloth (recommended) - Optimized for speed and memory efficiency
    2. Standard HuggingFace - Fallback option with more control
    
    Both approaches support:
    - 4-bit quantization for memory savings
    - LoRA (Low-Rank Adaptation) for efficient fine-tuning
    - Proper tokenizer configuration
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize ModelSetup with configuration loader.
        
        Args:
            config_loader: ConfigLoader instance that reads YAML configs
            
        The constructor loads all necessary configurations (model, LoRA, quantization)
        from YAML files and stores them as instance variables for easy access.
        """
        self.config_loader = config_loader
        self.model_config, self.lora_config, self.quant_config = config_loader.load_model_config()
        
    def create_bnb_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytesConfig for 4-bit quantization.
        
        Returns:
            BitsAndBytesConfig: Configuration for 4-bit quantization
            
        4-bit quantization reduces memory usage by ~75% while maintaining performance:
        - load_in_4bit: Store weights in 4-bit format
        - bnb_4bit_compute_dtype: Compute in bfloat16 for stability (even though weights are 4-bit)
        - bnb_4bit_use_double_quant: Quantize the quantization constants (extra memory savings)
        - bnb_4bit_quant_type: "nf4" (Normal Float 4) is the standard quantization method
        
        This is essential for fitting 7B+ models on consumer GPUs (24GB or less).
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.quant_config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.quant_config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.quant_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4"  # Standard quantization type for best quality/speed balance
        )
    
    def load_model_and_tokenizer_unsloth(self):
        """
        Load model and tokenizer using Unsloth (recommended approach).
        
        Returns:
            tuple: (model, tokenizer) ready for training
            
        Unsloth is a specialized library that provides:
        - 2x faster training than standard HuggingFace
        - Lower memory usage through optimized kernels
        - Automatic gradient checkpointing
        - Built-in LoRA support with optimizations
        
        This method handles:
        1. Model loading with quantization
        2. LoRA configuration and application
        3. Optimization for training speed
        
        The resulting model is ready for fine-tuning with minimal memory footprint.
        """
        # Load base model with Unsloth optimizations
        # max_seq_length: Maximum context length (2048 tokens = ~1500 words)
        # dtype: Use bfloat16 for stable mixed precision training
        # load_in_4bit: Enable 4-bit quantization for memory efficiency
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.name,
            max_seq_length=self.model_config.max_length,
            dtype=getattr(torch, self.model_config.torch_dtype),
            load_in_4bit=self.quant_config.load_in_4bit,
        )
        
        # Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
        # LoRA works by freezing the original model and adding small trainable matrices
        # This reduces trainable parameters from ~7B to ~16M while maintaining performance
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.r,  # Rank - higher = more flexible but more memory (16 is sweet spot)
            target_modules=self.lora_config.target_modules,  # Which layers to adapt (attention + MLP)
            lora_alpha=self.lora_config.lora_alpha,  # Scaling factor (usually 2x rank)
            lora_dropout=self.lora_config.lora_dropout,  # Regularization to prevent overfitting
            bias="none",  # Don't adapt bias terms (saves memory, minimal performance impact)
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
            random_state=42,  # For reproducible results
            use_rslora=False,  # RSLoRA is experimental, stick with standard LoRA
        )
        
        return model, tokenizer
    
    def load_model_and_tokenizer_standard(self):
        """
        Load model and tokenizer using standard HuggingFace approach (fallback).
        
        Returns:
            tuple: (model, tokenizer) ready for training
            
        This is the traditional HuggingFace approach that provides:
        - More control over individual components
        - Better debugging capabilities
        - Compatibility with all HuggingFace features
        - Slower training but more predictable behavior
        
        Use this if Unsloth has issues or you need specific HuggingFace features.
        The process involves:
        1. Loading tokenizer and setting pad token
        2. Creating quantization config
        3. Loading model with quantization
        4. Preparing model for k-bit training
        5. Applying LoRA configuration
        """
        # Load tokenizer first
        # Qwen2.5 uses a custom tokenizer, trust_remote_code handles this
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
        
        # Set pad token to EOS token (standard practice for decoder-only models)
        # This is crucial for batch training - without it, sequences can't be padded properly
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create quantization configuration
        # This tells the model how to handle 4-bit quantization during loading
        bnb_config = self.create_bnb_config()
        
        # Load the actual model with all our configurations
        # device_map="auto" automatically distributes model across available GPUs
        # torch_dtype=bfloat16 for stable mixed precision training
        # trust_remote_code=True allows loading of custom model code (needed for Qwen)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            quantization_config=bnb_config,
            device_map=self.model_config.device_map,
            torch_dtype=getattr(torch, self.model_config.torch_dtype),
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        # This function from PEFT handles the setup needed for training quantized models
        # It ensures gradients flow properly through quantized layers
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA using PEFT library
        # This creates the LoRA adapters that will be trained while keeping base model frozen
        peft_config = LoraConfig(
            r=self.lora_config.r,  # Rank of LoRA matrices
            lora_alpha=self.lora_config.lora_alpha,  # Scaling parameter
            lora_dropout=self.lora_config.lora_dropout,  # Dropout for regularization
            target_modules=self.lora_config.target_modules,  # Which layers to adapt
            bias="none",  # Don't adapt bias terms
            task_type="CAUSAL_LM"  # Task type for language modeling
        )
        
        # Apply LoRA configuration to the model
        # This wraps the model with LoRA adapters and makes them trainable
        model = get_peft_model(model, peft_config)
        
        return model, tokenizer
    
    def setup_model(self, use_unsloth: bool = True):
        """
        Main method to set up model and tokenizer for training.
        
        Args:
            use_unsloth: Whether to use Unsloth (True) or standard HuggingFace (False)
            
        Returns:
            tuple: (model, tokenizer) ready for fine-tuning
            
        This is the primary entry point for model setup. It:
        1. Chooses between Unsloth and standard HuggingFace approaches
        2. Loads and configures the model
        3. Prints training parameter information
        4. Returns the ready-to-train model and tokenizer
        
        The resulting model will have:
        - Base weights frozen (saves memory and compute)
        - LoRA adapters trainable (~16M parameters vs ~7B)
        - 4-bit quantization for memory efficiency
        - Proper tokenizer configuration for batch training
        """
        if use_unsloth:
            print("🚀 Loading model with Unsloth (fast & efficient)")
            print("   - 2x faster training")
            print("   - Lower memory usage")
            print("   - Optimized gradient checkpointing")
            model, tokenizer = self.load_model_and_tokenizer_unsloth()
        else:
            print("🐌 Loading model with standard HuggingFace")
            print("   - More control and debugging options")
            print("   - Better compatibility with HF ecosystem")
            print("   - Slower but more predictable")
            model, tokenizer = self.load_model_and_tokenizer_standard()
        
        # Print information about trainable parameters
        # This helps verify that LoRA is working correctly
        # You should see ~16M trainable parameters out of ~7B total
        if hasattr(model, 'print_trainable_parameters'):
            print("\n📊 Model Parameter Information:")
            model.print_trainable_parameters()
        else:
            # Fallback for models without this method
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n📊 Model Parameter Information:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return model, tokenizer