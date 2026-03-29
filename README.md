# DravidianLangTech-2026 : Abusive-Tamil-Text-Detection-Targeting-Women-on-Social-Media

Our Submission to the tak aims to detect abusive Tamil text targeting women on social media. It involves **preparing raw text data** by cleaning and labeling it, then **tokenizing** this text into a format suitable for machine learning. A *Transformer model* (like IndicBERT or XLM-RoBERTa) is used as the core deep learning architecture, with a *custom PyTorch dataset* efficiently managing the tokenized data. The entire **training and evaluation lifecycle** is **orchestrated** using a `Trainer` class, and the model's effectiveness is rigorously **evaluated and visualized** through various metrics and plots. Finally, the fine-tuned model is used to **predict on unseen test data** and generate a submission file.


## Visual Overview

```mermaid
flowchart TD
    A0["Data Preparation"]
    A1["Text Tokenization"]
    A2["Transformer Model"]
    A3["Custom PyTorch Dataset"]
    A4["Training Orchestration"]
    A5["Performance Evaluation & Visualization"]
    A6["Test Data Prediction and Submission"]
    A0 -- "Provides clean text" --> A1
    A0 -- "Provides labels for" --> A3
    A1 -- "Loads tokenizer for" --> A2
    A1 -- "Provides encodings for" --> A3
    A2 -- "Is trained by" --> A4
    A3 -- "Supplies data to" --> A4
    A4 -- "Generates evaluation data" --> A5
    A6 -- "Reuses preprocessing" --> A0
    A6 -- "Uses tokenizer" --> A1
    A6 -- "Makes predictions" --> A2
```

## Chapters

1. [Transformer Model](readme_files/01_transformer_model_.md)
2. [Data Preparation](readme_files/02_data_preparation_.md)
3. [Text Tokenization](readme_files/03_text_tokenization_.md)
4. [Custom PyTorch Dataset](readme_files/04_custom_pytorch_dataset_.md)
5. [Training](readme_files/05_training_orchestration_.md)
6. [Performance Evaluation & Visualization](readme_files/06_performance_evaluation___visualization_.md)
7. [Test Data Prediction and Submission](readme_files/07_test_data_prediction_and_submission_.md)

---
