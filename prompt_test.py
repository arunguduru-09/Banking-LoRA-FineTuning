
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

prompts = [
    "What does KYC stand for in banking?",
    "Expand the banking abbreviation EMI.",
    "What does UPI stand for in Indian banking?",
    "Explain fixed deposit in one sentence."
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

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

    print("\nPrompt:", prompt)
    print("Base Model Answer:", answer)
