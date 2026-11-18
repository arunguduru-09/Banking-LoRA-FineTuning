
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()

prompt = """
Answer banking questions using the examples.

Question: What is KYC?
Answer: KYC stands for Know Your Customer.

Question: What is EMI?
Answer: EMI stands for Equated Monthly Instalment.

Question: What is UPI?
Answer:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Few-Shot Model Answer:", answer)
