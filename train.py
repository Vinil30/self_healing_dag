from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

def fine_tune_model():
    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepare model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Set correct label mapping
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # Train and save
    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
