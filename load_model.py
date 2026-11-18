
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()

question = "What is KYC?"

inputs = tokenizer(
    question,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Question:", question)
print("Model Answer:", answer)

