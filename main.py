from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_text, get_answer_from_model_on_gsm8k, evaluate_model_on_gsm8k, TextGenerator

# Load a lightweight LLM (e.g., distilgpt2)
model = TextGenerator("./output")

# Evaluation the model using the GSM8K benchmark
from datasets import load_dataset

# Load the GSM8K dataset (benchmark method)
gsm8k_dataset = load_dataset("gsm8k", "main", split="test")

acc = evaluate_model_on_gsm8k(model, gsm8k_dataset, limit=10)
print(acc)