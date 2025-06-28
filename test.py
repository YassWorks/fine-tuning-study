from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "My favortie movie is the Matrix. I love the part where Neo learns to fly. What is your favorite movie and why?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.strip())
