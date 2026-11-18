
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

# Load V2 dataset
with open("banking_train_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load pre-trained model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Only train LoRA parameters
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=0.0003
)

# Prepare training examples
training_data = []

for item in data:
    prompt = "Answer this banking question: " + item["question"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        text_target=item["answer"],
        return_tensors="pt",
        truncation=True,
        max_length=64
    )["input_ids"]

    training_data.append((inputs, labels))

# Start training
torch.manual_seed(42)
model.train()

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for inputs, labels in training_data:
        optimizer.zero_grad()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_data)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Average Loss: {avg_loss:.4f}",
        flush=True
    )

# Save trained LoRA adapter
model.save_pretrained("banking_lora_adapter_v2")
tokenizer.save_pretrained("banking_lora_adapter_v2")

print("Version 2 Fine-Tuning Completed Successfully!")
