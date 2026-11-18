
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model = PeftModel.from_pretrained(
    base_model,
    "banking_lora_adapter_v2"
)

model.eval()

questions = [
    "What is KYC?",
    "What is EMI?",
    "What is a fixed deposit?",
    "What is UPI?",
    "What is collateral?"
]

for question in questions:
    prompt = "Answer this banking question: " + question
    inputs = tokenizer(prompt, return_tensors="pt")

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
    print("V2 Fine-Tuned Answer:", answer)
