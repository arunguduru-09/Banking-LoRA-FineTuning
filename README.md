
# Banking AI Fine-Tuning Using LoRA

A hands-on Generative AI project demonstrating parameter-efficient fine-tuning of Google's FLAN-T5-Small model for banking question answering using Python, PyTorch, Hugging Face Transformers, and LoRA.

## 1. Project Overview

This project explores how a pre-trained language model can be adapted to answer banking-related questions using Low-Rank Adaptation (LoRA).

The experiment includes dataset preparation, baseline testing, LoRA configuration, model training, inference, and evaluation.

**Project Status:** Experimental / Learning Project

Training completed successfully, but the model still produces inaccurate answers for several banking questions. Further improvements are required.

## 2. Objectives

- Understand pre-trained language models.
- Implement parameter-efficient fine-tuning.
- Freeze original model weights.
- Train additional LoRA parameters.
- Compare base and fine-tuned model responses.
- Analyze training loss and model limitations.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming |
| PyTorch | Deep learning |
| Hugging Face Transformers | Model loading and inference |
| PEFT | LoRA implementation |
| FLAN-T5-Small | Pre-trained language model |
| JSON | Banking Q&A dataset |
| VS Code | Development environment |
| Git & GitHub | Version control |

## 4. Project Architecture

```text
Banking Q&A Dataset
        |
        v
Data Preparation
        |
        v
FLAN-T5-Small Model
        |
        v
LoRA Adapter Configuration
        |
        v
Model Fine-Tuning
        |
        v
Save LoRA Adapters
        |
        v
Load Fine-Tuned Model
        |
        v
Banking Question Answering
        |
        v
Evaluation and Comparison
```

## 5. Dataset

Two datasets were used during experimentation.

| Version | Training Examples |
|---|---:|
| V1 | 5 |
| V2 | 25 |

The dataset contains banking-related questions and reference answers.

Example:

```json
{
  "question": "What is KYC?",
  "answer": "KYC stands for Know Your Customer."
}
```

Topics include:

- KYC and EMI
- Savings and current accounts
- Fixed deposits
- Loans and interest
- Credit scores
- UPI, NEFT and RTGS
- ATM and IFSC
- Collateral

The dataset is small and intended for educational experimentation.

## 6. Base Model

Model: `google/flan-t5-small`

FLAN-T5-Small is an instruction-tuned encoder-decoder language model.

It is used as the foundation for our banking question-answering experiment.

## 7. LoRA Configuration

LoRA allows model adaptation without updating all original pre-trained weights.

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
```

### Parameter Statistics

| Parameter | Value |
|---|---:|
| Total Parameters | 77,305,216 |
| Trainable Parameters | 344,064 |
| Trainable Percentage | 0.4451% |
| Frozen Parameters | Approximately 99.55% |

Only a small fraction of model parameters is updated during LoRA training.

## 8. Training Configuration

| Setting | V2 Value |
|---|---|
| Model | FLAN-T5-Small |
| Dataset | 25 examples |
| Epochs | 5 |
| Learning Rate | 0.0003 |
| Optimizer | AdamW |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Device | CPU |

## 9. Training Results

### Version 2 Training Loss

| Epoch | Average Loss |
|---|---:|
| 1 | 3.5433 |
| 2 | 3.3245 |
| 3 | 3.0770 |
| 4 | 2.8266 |
| 5 | 2.6630 |

The average training loss decreased by approximately 24.8% from epoch 1 to epoch 5.

This indicates improved fitting to the training examples, but does not establish accuracy on unseen questions.

## 10. Model Evaluation

The base model and fine-tuned model were tested using banking questions.

### Example: KYC

**Question:** What is KYC?

**Expected Answer:**
KYC stands for Know Your Customer.

**V2 Fine-Tuned Answer:**
KYC is a regulated banking system.

**Result:** Incorrect

### Example: EMI

**Expected Answer:**
EMI stands for Equated Monthly Instalment.

**V2 Fine-Tuned Answer:**
EMI is a payment system.

**Result:** Incorrect

### Example: UPI

**Expected Answer:**
UPI stands for Unified Payments Interface.

**V2 Fine-Tuned Answer:**
UPI is a payment method used to pay a debt.

**Result:** Incorrect

### Evaluation Summary

Training loss improved, but factual answer quality remained poor.

The tested questions were included in the training dataset, so these tests do not measure generalization to unseen examples.

No formal held-out accuracy metric has been established.

## 11. Project Structure

```text
Banking_LoRA_Project/
|
|-- banking_data.py
|-- banking_data_v2.py
|-- banking_train.json
|-- banking_train_v2.json
|
|-- gpu_check.py
|-- load_model.py
|-- lora_setup.py
|
|-- train_lora.py
|-- train_lora_v2.py
|
|-- test_lora.py
|-- test_lora_v2.py
|-- compare_models.py
|
|-- prompt_test.py
|-- few_shot_test.py
|
|-- README.md
|-- .gitignore
```

The generated adapter folders and local virtual environment are excluded from Git tracking.

## 12. Installation and Setup

### Prerequisites

- Python
- Git
- Internet connection for initial model download
- VS Code or another Python editor

### Clone Repository

```bash
git clone https://github.com/arunguduru-09/Banking-LoRA-FineTuning.git
cd Banking-LoRA-FineTuning
```

### Create Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install torch transformers datasets peft accelerate
```

### Install Dependencies

```bash
python -m pip install torch transformers datasets peft accelerate
```

### Generate V2 Dataset

```bash
python banking_data_v2.py
```

### Train LoRA Model

```bash
python train_lora_v2.py
```

The trained adapter is saved locally in:

`banking_lora_adapter_v2/`

### Test Fine-Tuned Model

```bash
python test_lora_v2.py
```

### Compare Model Responses

```bash
python compare_models.py
```

Note: The current comparison script uses the V1 adapter. The V2 testing script loads the V2 adapter.

## 13. Challenges and Limitations

- Small training dataset.
- Limited domain-specific examples.
- Incorrect factual responses after fine-tuning.
- CPU-based training.
- No independent validation dataset.
- No formal generalization benchmark.
- Adapter weights are not included in the repository.

This model should not be used for production banking decisions or customer advice.

## 14. Future Improvements

- Expand the banking Q&A dataset.
- Create separate training and validation datasets.
- Evaluate accuracy on unseen questions.
- Improve instruction and answer formatting.
- Experiment with learning rates and epochs.
- Compare LoRA configurations.
- Explore larger models when suitable hardware is available.
- Investigate RAG for document-grounded banking answers.
- Build a Streamlit or FastAPI interface.

## 15. Key Learnings

This project provided practical experience in:

- Fine-tuning fundamentals
- Tokenization
- LoRA and trainable parameters
- Model weight freezing
- Training epochs and loss
- Saving and loading adapters
- Model inference
- Prompt engineering
- Evaluating model limitations

## 16. Disclaimer

This is an educational Generative AI experiment.

It does not provide reliable financial or banking advice. Outputs must be independently verified.

## 17. Author

**Guduru Arun Kumar**

GitHub: [arunguduru-09](https://github.com/arunguduru-09)

Project Repository:
https://github.com/arunguduru-09/Banking-LoRA-FineTuning
