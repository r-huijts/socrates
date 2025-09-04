# 🎭 Socrates: Socratic Tutor Fine-Tuning

This project fine-tunes a Qwen2.5 language model to act as a Socratic tutor. The training pipeline is designed to be adaptable, automatically optimizing for high-performance GPU environments (using Unsloth) while remaining functional for local development on CPU/MPS.

## 🚀 Quick Start

Follow these steps to set up and run the training process.

### 1. Setup Environment

First, ensure you have a Python environment (3.10+ recommended). Then, activate the virtual environment and install the required dependencies.

```bash
# Activate the virtual environment
source socrates_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your Training Run

Before starting, review and adjust the configuration files to match your needs.

-   **Model Configuration (`config/model/qwen_config.yaml`):**
    -   `model.name`: The base Hugging Face model to fine-tune.
    -   `lora.*`: Parameters for Low-Rank Adaptation (LoRA), which allows for efficient training.
    -   `quantization.*`: Settings for 4-bit quantization to reduce memory usage.

-   **Training Configuration (`config/training/training_config.yaml`):**
    -   `training.output_dir`: Where the final model checkpoints will be saved.
    -   `training.num_train_epochs`: The number of passes through the entire dataset.
    -   `data.max_seq_length`: The maximum token length for a training example.
    -   `wandb.*`: Optional configuration for logging metrics to Weights & Biases.

### 3. Prepare Your Data

The training script automatically loads all `*.json` files from the `data/synthetic/` directory. Each JSON file should contain a list of conversations formatted for training.

Example structure of a single record in a `json` file:
```json
[
  {
    "conversations": [
      { "from": "human", "value": "Can you explain the theory of relativity?" },
      { "from": "gpt", "value": "Of course. To begin, what do you already know about it?" }
    ]
  }
]
```

### 4. Run the Training Script

Execute the main training script from the root of the project. The script will automatically detect your hardware and choose the best training mode.

```bash
python src/training/train.py
```

**How it works:**

-   **With a CUDA-enabled GPU:** The script will automatically use the high-performance **Unsloth** library for training.
-   **Without a CUDA GPU (CPU/MPS):** The script will fall back to the standard Hugging Face `Trainer`. It will also use a smaller model (`Qwen/Qwen2.5-1.5B-Instruct`) and reduce batch sizes to ensure it can run on systems with less memory.

#### Command-Line Arguments

You can override the default configurations using command-line arguments:

-   `--config`: Path to a custom training config file.
-   `--data`: Path to a specific JSON data file to use for training, instead of the default directory.

Example with a custom data file:
```bash
python src/training/train.py --data /path/to/your/custom_data.json
```

### 5. Accessing the Trained Model

Once training is complete, the final model will be saved in the directory specified by `output_dir` in your training configuration (default is `./models/socratic-tutor-qwen2.5`).

The script saves the model in two formats for maximum compatibility:
-   **Unsloth format:** The primary output directory.
-   **Hugging Face format:** Saved in a separate directory with an `_hf` suffix (e.g., `socratic-tutor-qwen2.5_hf`), which is ideal for deployment and sharing.
