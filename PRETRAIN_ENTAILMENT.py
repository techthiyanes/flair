from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import flair
import os

def main():
    """
    model_checkpoint = "bert-base-uncased"
    mnli_dataset = load_dataset("glue", "mnli")
    metric_name = "accuracy"
    metric = load_metric('glue', "mnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        labels = [label if label in [0,1] else 1 for label in examples["label"]]
        examples["label"] = labels
        return tokenizer(examples["hypothesis"], examples["premise"], truncation=True)

    encoded_dataset = mnli_dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        "pretained_mnli",
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
        eval_dataset=encoded_dataset["validation_matched"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("pretained_mnli/best_model")
    tokenizer.save_pretrained("pretained_mnli/best_model")


    model_checkpoint = f"pretained_mnli/best_model"
    rte_dataset = load_dataset("glue", "rte")
    metric_name = "accuracy"
    metric = load_metric('glue', "rte")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        return tokenizer(examples["sentence2"], examples["sentence1"], truncation=True)

    encoded_dataset = rte_dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        "pretained_mnli_rte",
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
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("pretained_mnli_rte/best_model")
    tokenizer.save_pretrained("pretained_mnli_rte/best_model")
    """

    model_checkpoint = f"pretained_mnli_rte/best_model"
    fever_dataset = load_dataset("json", data_files={"train": f"{flair.cache_root}/datasets/fever/train.jsonl",
                                               "test": f"{flair.cache_root}/datasets/fever/test.jsonl",
                                               "dev": f"{flair.cache_root}/datasets/fever/dev.jsonl"})
    metric_name = "accuracy"
    metric = load_metric('glue', "mnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        labels = [1 if x == "SUPPORTS" else 0 for x in examples["label"]]
        examples["label"] = labels
        return tokenizer(examples["context"], examples["query"], truncation=True)

    encoded_dataset = fever_dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        "pretained_mnli_rte_fever",
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
    trainer.save_model("pretained_mnli_rte_fever/best_model")
    tokenizer.save_pretrained("pretained_mnli_rte_fever/best_model")

if __name__ == "__main__":
    main()