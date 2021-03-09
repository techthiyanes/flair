from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import flair

def main():
    model_checkpoint = "bert-base-uncased"
    dataset = load_dataset("json", data_files={"train": f"{flair.cache_root}/datasets/fever/train.jsonl", "test": f"{flair.cache_root}/datasets/fever/test.jsonl", "dev": f"{flair.cache_root}/datasets/fever/dev.jsonl"})
    metric_name = "accuracy"
    metric = load_metric('glue', "mnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        return tokenizer(examples["query"], examples["context"], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        "mnli+rte+fever",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()

if __name__ == "__main__":
    main()