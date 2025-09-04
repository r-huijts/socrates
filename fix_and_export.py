import torch
from unsloth import FastLanguageModel
import os

# The location of your Unsloth-format trained model
model_dir = "./models/socratic-tutor-qwen2.5"

# The desired output location for the fully merged, standard HF model
output_dir = "./models/socratic-tutor-qwen2.5_fully_merged"

print(f"Loading model from: {model_dir}")

# Load the Unsloth model from the saved checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Load in full precision for merging
)

print("Model loaded. Merging and saving to Hugging Face format...")

# This is the key step: merge_and_unload() fully combines the adapters
# and returns a standard Hugging Face model object.
model = model.merge_and_unload()

# Save the fully merged model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model successfully merged and saved to: {output_dir}")
print("You can now use this directory for GGUF conversion.")