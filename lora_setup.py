
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

model_name = "google/flan-t5-small"

# Load pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Add LoRA adapters
lora_model = get_peft_model(model, lora_config)

# Display trainable parameters
lora_model.print_trainable_parameters()

print("LoRA adapters added successfully!")
