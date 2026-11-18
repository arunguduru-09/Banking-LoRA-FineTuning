
import json

banking_data = [
    ("What is KYC?", "KYC stands for Know Your Customer."),
    ("What does KYC mean?", "KYC means Know Your Customer."),
    ("Expand KYC.", "KYC stands for Know Your Customer."),
    ("What is EMI?", "EMI stands for Equated Monthly Instalment."),
    ("What does EMI stand for?", "EMI stands for Equated Monthly Instalment."),
    ("Expand EMI.", "EMI means Equated Monthly Instalment."),
    ("What is a fixed deposit?", "A fixed deposit is money deposited for a fixed period."),
    ("Explain fixed deposit.", "A fixed deposit holds money for a fixed period."),
    ("What is FD?", "FD stands for Fixed Deposit."),
    ("What is a savings account?", "A savings account helps customers save money."),
    ("What is a current account?", "A current account supports frequent banking transactions."),
    ("What is a credit score?", "A credit score indicates creditworthiness."),
    ("What is a loan?", "A loan is money borrowed that must be repaid."),
    ("What is interest?", "Interest is the cost of borrowing money."),
    ("What is principal?", "Principal is the original amount borrowed."),
    ("What is collateral?", "Collateral is an asset pledged to secure a loan."),
    ("What is an ATM?", "ATM stands for Automated Teller Machine."),
    ("What is UPI?", "UPI stands for Unified Payments Interface."),
    ("What is NEFT?", "NEFT stands for National Electronic Funds Transfer."),
    ("What is RTGS?", "RTGS stands for Real Time Gross Settlement."),
    ("What is IFSC?", "IFSC stands for Indian Financial System Code."),
    ("What is a debit card?", "A debit card uses funds from a linked bank account."),
    ("What is a credit card?", "A credit card allows borrowing up to an approved limit."),
    ("What is a bank statement?", "A bank statement records account transactions."),
    ("What is a nominee?", "A nominee is a person designated by the account holder.")
]

records = [
    {"question": question, "answer": answer}
    for question, answer in banking_data
]

with open("banking_train_v2.json", "w", encoding="utf-8") as file:
    json.dump(records, file, indent=4, ensure_ascii=False)

print("Version 2 Dataset Created Successfully!")
print("Total Examples:", len(records))
