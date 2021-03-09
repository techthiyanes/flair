from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import json

def create_datasets(save_files=True):
    data = load_dataset("fever", "v1.0")
    wiki_pages = load_dataset("fever", "wiki_pages")
    wiki_keys = wiki_pages["wikipedia_pages"]["id"]
    wiki_text = wiki_pages["wikipedia_pages"]["text"]
    wiki_ids = np.arange(len(wiki_keys))
    wiki_lookup = {}
    for key, text, id in zip(wiki_keys, wiki_text, wiki_ids):
        wiki_lookup[key] = {"text": text,
                            "id": id}

    train = data["train"]
    train_json = {}
    seen_ids = []
    for data_point in train:
        evidence_ref = data_point["evidence_wiki_url"]
        if evidence_ref != "":
            if train["id"] not in seen_ids:
                train_json["claim"] = train["claim"]
                train_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                unprocessed_label = train["label"]
                if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                    label = 0
                elif unprocessed_label in ["SUPPORTS"]:
                    label = 1
                else:
                    print(unprocessed_label)
                    raise Exception("unknown label.")
                train_json["label"] = label
                train_json["id"] = train["id"]
                seen_ids.append(train["id"])

    with open("train.json", "w") as outfile:
        json.dump(train_json, outfile)

    dev = data["paper_dev"]
    dev_json = {}
    seen_ids = []
    for data_point in dev:
        evidence_ref = data_point["evidence_wiki_url"]
        if evidence_ref != "":
            if dev["id"] not in seen_ids:
                dev_json["claim"] = dev["claim"]
                dev_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                unprocessed_label = dev["label"]
                if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                    label = 0
                elif unprocessed_label in ["SUPPORTS"]:
                    label = 1
                else:
                    raise Exception("unknown label.")
                dev_json["label"] = label
                dev_json["id"] = dev["id"]
                seen_ids.append(dev["id"])

    with open("dev.json", "w") as outfile:
        json.dump(dev_json, outfile)

    test = data["paper_test"]
    test_json = {}
    seen_ids = []
    for data_point in test:
        evidence_ref = data_point["evidence_wiki_url"]
        if evidence_ref != "":
            if test["id"] not in seen_ids:
                test_json["claim"] = test["claim"]
                test_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                unprocessed_label = test["label"]
                if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                    label = 0
                elif unprocessed_label in ["SUPPORTS"]:
                    label = 1
                else:
                    raise Exception("unknown label.")
                test_json["label"] = label
                test_json["id"] = test["id"]
                seen_ids.append(test["id"])

    with open("test.json", "w") as outfile:
        json.dump(test_json, outfile)

def main():
    create_datasets()
    """
    model_checkpoint = "bert-base-uncased"
    sentence1_key, sentence2_key = "claim", "evidence"
    dataset = load_dataset("glue", "mnli")
    metric = load_metric('glue', "mnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        labels = [label if label in [0,1] else 1 for label in examples["label"]]
        examples["label"] = labels
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    args = TrainingArguments(
        "mnli+rte",
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
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()
    """

if __name__ == "__main__":
    main()