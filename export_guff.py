from unsloth import FastLanguageModel

print("🔄 Loading your trained Socratic tutor...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/socratic-tutor-qwen2.5/",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("📦 Exporting to GGUF format for Ollama...")
model.save_pretrained_gguf("socratic-tutor", tokenizer, quantization_method="q4_k_m")

print("✅ GGUF export complete!")
print("📁 Look for: socratic-tutor-q4_k_m.gguf")