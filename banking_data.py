
import json

banking_data = [
    {
        "question": "What is KYC?",
        "answer": "KYC stands for Know Your Customer."
    },
    {
        "question": "What is EMI?",
        "answer": "EMI stands for Equated Monthly Instalment."
    },
    {
        "question": "What is a savings account?",
        "answer": "A savings account helps customers save money."
    },
    {
        "question": "What is a credit score?",
        "answer": "A credit score indicates creditworthiness."
    },
    {
        "question": "What is a fixed deposit?",
        "answer": "A fixed deposit holds money for a fixed period."
    }
]

with open("banking_train.json", "w") as f:
    json.dump(banking_data, f, indent=4)

print("Dataset created successfully!")
print("Total examples:", len(banking_data))
