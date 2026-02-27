import os
import sys
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerLlmConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = os.path.join("artifacts", "llm_results")
    final_model_path: str = os.path.join("artifacts", "sentiment_model")
    batch_size: int = 16
    epochs: int = 1  # 1 epoch is recommended for initial testing on CPU

class ModelTrainerLlm:
    def __init__(self):
        self.llm_config = ModelTrainerLlmConfig()

    def compute_metrics(self, pred):
        """Calculates accuracy and F1 score during training."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def initiate_llm_training(self, train_csv_path):
        try:
            logging.info("LLM Fine-tuning initiated for Review Intelligence")

            # 1. Load and Prepare Data
            df = pd.read_csv(train_csv_path)
            
            # Map stars (1-5) to labels (0: Neg, 1: Neu, 2: Pos)
            def map_sentiment(star):
                if star <= 2: return 0
                if star == 3: return 1
                return 2

            df['label'] = df['stars'].apply(map_sentiment)
            # Use 'review_body' as text and 'label' as target
            df = df[['review_body', 'label']].dropna()

            # Convert to Hugging Face Dataset format
            dataset = Dataset.from_pandas(df)
            dataset = dataset.train_test_split(test_size=0.2)

            # 2. Tokenization
            logging.info(f"Loading tokenizer: {self.llm_config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model_name)

            def tokenize_function(examples):
                return tokenizer(examples["review_body"], truncation=True, padding=True)

            logging.info("Tokenizing datasets...")
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            # 3. Model Setup (3 labels for sentiment)
            logging.info("Initializing Sequence Classification model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.llm_config.model_name, num_labels=3
            )

            # 4. Training Arguments
            # Updated 'eval_strategy' for compatibility with transformers v4.46+
            training_args = TrainingArguments(
                output_dir=self.llm_config.output_dir,
                eval_strategy="epoch",      
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=self.llm_config.batch_size,
                per_device_eval_batch_size=self.llm_config.batch_size,
                num_train_epochs=self.llm_config.epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                logging_dir=os.path.join("logs", "llm_logs"),
                report_to="none" 
            )

            # 5. Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )

            logging.info("Starting LLM Fine-tuning (This may take a while on CPU)...")
            trainer.train()

            # 6. Save final model and tokenizer
            os.makedirs(self.llm_config.final_model_path, exist_ok=True)
            trainer.save_model(self.llm_config.final_model_path)
            tokenizer.save_pretrained(self.llm_config.final_model_path)
            
            logging.info(f"LLM successfully saved to {self.llm_config.final_model_path}")
            return self.llm_config.final_model_path

        except Exception as e:
            raise CustomException(e, sys)