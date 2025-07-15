# Socratic Tutor - Qwen2.5 Fine-tuning Project

A fine-tuning project to create a Socratic tutor using Qwen2.5 and Unsloth for efficient training.

## Project Structure
socrates/
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned and formatted data
│   └── synthetic/     # Generated training data
├── src/
│   ├── training/      # Training scripts
│   ├── inference/     # Model inference
│   ├── evaluation/    # Evaluation metrics
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── scripts/           # Executable scripts
├── models/            # Saved models
└── tests/             # Unit tests

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Run training: `python scripts/training/train_socratic_tutor.py`