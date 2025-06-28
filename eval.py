from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_text, get_answer_from_model_on_gsm8k, evaluate_model_on_gsm8k, TextGenerator
from datasets import load_dataset

model = TextGenerator("./output")

# Load the GSM8K dataset (benchmark method)
gsm8k_dataset = load_dataset("gsm8k", "main", split="test")

acc = evaluate_model_on_gsm8k(model, gsm8k_dataset, limit=10)
print(acc)