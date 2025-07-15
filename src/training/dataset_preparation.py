import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset, DatasetDict
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from src.utils.config_loader import ConfigLoader

class SocraticDatasetPreparator:
    """
    Modern dataset preparation using official Unsloth and HuggingFace methods.
    
    This class handles loading synthetic Socratic dialogue data and preparing it
    for fine-tuning using the official, recommended approaches from Unsloth
    documentation. No manual dictionary manipulation required!
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the dataset preparator with configuration.
        
        Args:
            config_loader: ConfigLoader instance for accessing data configuration
        """
        self.config_loader = config_loader
        self.training_config = config_loader.load_training_config()
        
        # Extract configuration values
        self.validation_split_ratio = 0.1  # 10% for validation
        self.random_seed = 42  # For reproducible splits
        
    def setup_tokenizer_template(self, tokenizer, chat_template: str = "qwen-2.5"):
        """
        Apply the appropriate chat template to the tokenizer using Unsloth.
        
        Args:
            tokenizer: The base tokenizer to configure
            chat_template: Template name to apply (default: "qwen-2.5")
            
        Returns:
            Configured tokenizer with chat template applied
            
        This method uses Unsloth's official get_chat_template function to ensure
        the tokenizer is properly configured for the target model.
        """
        print(f"🔧 Applying '{chat_template}' chat template to tokenizer...")
        
        # Use official Unsloth method to apply chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template
        )
        
        print(f"✅ Chat template '{chat_template}' applied successfully")
        return tokenizer
    
    def load_raw_dataset(self, data_dir: str = "data/synthetic"):
        """
        Load JSON files using HuggingFace's official load_dataset function.
        
        Args:
            data_dir: Directory containing JSON files
            
        Returns:
            Raw dataset loaded from JSON files
            
        This method uses HuggingFace's load_dataset which automatically handles:
        - File discovery and loading
        - JSON parsing and validation
        - Schema inference
        - Error handling for malformed files
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' not found")
        
        # Find JSON files
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in '{data_dir}'")
        
        print(f"📁 Found {len(json_files)} JSON files in '{data_dir}':")
        for file in json_files:
            print(f"  📄 {file.name}")
        
        # Use HuggingFace's official method to load JSON files
        print("🔄 Loading dataset using HuggingFace load_dataset...")
        
        try:
            # Load all JSON files in the directory
            dataset = load_dataset(
                "json", 
                data_files=str(data_path / "*.json"),
                split="train"
            )
            
            print(f"✅ Successfully loaded {len(dataset)} conversations")
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
    
    def format_for_training(self, dataset, tokenizer):
        """
        Format the dataset for training using the official Unsloth approach.
        
        Args:
            dataset: Raw dataset from load_dataset
            tokenizer: Tokenizer with chat template applied
            
        Returns:
            Formatted dataset with 'text' column ready for training
            
        This method follows the exact pattern recommended in Unsloth documentation
        for preparing conversational data for fine-tuning.
        """
        print("🔄 Formatting conversations for training...")
        
        # Check if we need to standardize format
        # If data uses ShareGPT format (from/value), convert to ChatML (role/content)
        sample_convo = dataset[0]["conversations"]
        if isinstance(sample_convo, list) and len(sample_convo) > 0:
            first_message = sample_convo[0]
            if "from" in first_message and "value" in first_message:
                print("🔄 Converting from ShareGPT to ChatML format...")
                dataset = standardize_sharegpt(dataset)
                print("✅ Format conversion complete")
        
        # Define the official formatting function pattern
        def formatting_prompts_func(examples):
            """
            Official Unsloth formatting function.
            
            This function takes the 'conversations' column and applies the chat
            template to create the 'text' column that the trainer expects.
            """
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=False
                ) 
                for convo in convos
            ]
            return {"text": texts}
        
        # Apply formatting using HuggingFace's map function
        formatted_dataset = dataset.map(
            formatting_prompts_func, 
            batched=True,
            desc="Applying chat template"
        )
        
        print(f"✅ Dataset formatted: {len(formatted_dataset)} conversations")
        
        # Show a sample of the formatted text for verification
        if len(formatted_dataset) > 0:
            print("📝 Sample formatted conversation:")
            sample_text = formatted_dataset[0]["text"]
            # Show first 200 characters for preview
            preview = sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
            print(f"   {preview}")
        
        return formatted_dataset
    
    def create_train_validation_split(self, dataset):
        """
        Create train/validation split using HuggingFace's built-in method.
        
        Args:
            dataset: Formatted dataset with 'text' column
            
        Returns:
            DatasetDict with 'train' and 'validation' splits
            
        Uses HuggingFace's train_test_split which ensures proper randomization
        and maintains data integrity across splits.
        """
        print(f"📊 Creating train/validation split ({int((1-self.validation_split_ratio)*100)}%/{int(self.validation_split_ratio*100)}%)...")
        
        # Use HuggingFace's built-in splitting method
        dataset_split = dataset.train_test_split(
            test_size=self.validation_split_ratio,
            seed=self.random_seed,
            shuffle=True
        )
        
        # Create final DatasetDict with clear naming
        final_dataset = DatasetDict({
            'train': dataset_split['train'],
            'validation': dataset_split['test']  # Rename 'test' to 'validation'
        })
        
        print(f"✅ Dataset split complete:")
        print(f"  🏋️  Training set: {len(final_dataset['train'])} conversations")
        print(f"  📊 Validation set: {len(final_dataset['validation'])} conversations")
        
        return final_dataset
    
    def prepare_dataset(self, tokenizer, data_dir: str = "data/synthetic", 
                       chat_template: str = "qwen-2.5") -> DatasetDict:
        """
        Complete dataset preparation pipeline using official methods.
        
        Args:
            tokenizer: Base tokenizer to configure
            data_dir: Directory containing JSON files
            chat_template: Chat template to apply (default: "qwen-2.5")
            
        Returns:
            DatasetDict ready for training with Unsloth
            
        This is the main entry point that orchestrates the entire modern
        dataset preparation pipeline using official HuggingFace and Unsloth methods.
        """
        print("🚀 Starting modern dataset preparation pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Configure tokenizer with chat template
            configured_tokenizer = self.setup_tokenizer_template(tokenizer, chat_template)
            
            # Step 2: Load raw dataset using HuggingFace
            raw_dataset = self.load_raw_dataset(data_dir)
            
            # Step 3: Format for training using official Unsloth approach
            formatted_dataset = self.format_for_training(raw_dataset, configured_tokenizer)
            
            # Step 4: Create train/validation split
            final_dataset = self.create_train_validation_split(formatted_dataset)
            
            # Final summary
            print("=" * 60)
            print("🎉 Dataset preparation complete!")
            print()
            print("📋 Final Summary:")
            print(f"  📁 Source directory: {data_dir}")
            print(f"  🔤 Chat template: {chat_template}")
            print(f"  📊 Total conversations: {len(raw_dataset)}")
            print(f"  🏋️  Training examples: {len(final_dataset['train'])}")
            print(f"  📊 Validation examples: {len(final_dataset['validation'])}")
            print(f"  ✅ Ready for Unsloth training!")
            
            return final_dataset
            
        except Exception as e:
            print(f"❌ Error during dataset preparation: {e}")
            raise
    
    def get_dataset_info(self, dataset: DatasetDict) -> dict:
        """
        Get information about the prepared dataset.
        
        Args:
            dataset: Prepared DatasetDict
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "total_examples": len(dataset['train']) + len(dataset['validation']),
            "train_examples": len(dataset['train']),
            "validation_examples": len(dataset['validation']),
            "validation_ratio": self.validation_split_ratio,
            "columns": dataset['train'].column_names,
            "features": dataset['train'].features
        }
        
        # Sample text statistics
        if len(dataset['train']) > 0:
            sample_texts = [dataset['train'][i]['text'] for i in range(min(5, len(dataset['train'])))]
            info["avg_text_length"] = sum(len(text) for text in sample_texts) / len(sample_texts)
            info["sample_text_preview"] = sample_texts[0][:100] + "..." if len(sample_texts[0]) > 100 else sample_texts[0]
        
        return info