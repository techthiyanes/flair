from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    train_dataset = load_dataset('csv', data_files=['../.flair/datasets/ag_news_csv/train.csv'], column_names=['label', 'header', 'text'], script_version="master", cache_dir="./", split="train")
    test_dataset = load_dataset('csv', data_files=['../.flair/datasets/ag_news_csv/test.csv'], column_names = ['label', 'header', 'text'], script_version = "master", cache_dir="./", split="test")
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir='transformers_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='transformers_logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    trainer.evaluate()

if __name__ == "__main__":
    main()