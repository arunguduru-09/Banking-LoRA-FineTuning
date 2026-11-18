
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load dataset
with open("banking_train.json", "r") as f:
    data = json.load(f)

# 2. Load model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Add LoRA
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# 4. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005
)

# 5. Training
model.train()

for epoch in range(3):
    total_loss = 0

    for item in data:
        question = "Answer this banking question: " + item["question"]
        answer = item["answer"]

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        labels = tokenizer(
            text_target=answer,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )["input_ids"]

        outputs = model(
            **inputs,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"Average Loss = {total_loss / len(data):.4f}"
    )

# 6. Save LoRA adapter
model.save_pretrained("banking_lora_adapter")
tokenizer.save_pretrained("banking_lora_adapter")

print("Fine-Tuning Completed Successfully!")
