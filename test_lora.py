
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

model_name = "google/flan-t5-small"
adapter_path = "banking_lora_adapter"

# Load tokenizer and original model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load our trained LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

model.eval()

questions = [
    "What is KYC?",
    "What is EMI?",
    "What is a fixed deposit?"
]

for question in questions:
    prompt = "Answer this banking question: " + question

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nQuestion:", question)
    print("Fine-Tuned Answer:", answer)
