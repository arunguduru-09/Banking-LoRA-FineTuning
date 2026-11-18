
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

model_name = "google/flan-t5-small"
adapter_path = "banking_lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

fine_tuned_model.eval()

question = "What is KYC?"
prompt = "Answer this banking question: " + question

inputs = tokenizer(prompt, return_tensors="pt")

# Test with LoRA disabled
with fine_tuned_model.disable_adapter():
    with torch.no_grad():
        base_output = fine_tuned_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

# Test with LoRA enabled
with torch.no_grad():
    tuned_output = fine_tuned_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

print("Question:", question)

print(
    "Before Fine-Tuning:",
    tokenizer.decode(base_output[0], skip_special_tokens=True)
)

print(
    "After Fine-Tuning:",
    tokenizer.decode(tuned_output[0], skip_special_tokens=True)
)
