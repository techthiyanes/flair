from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import json
import flair

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
    seen_ids = []
    firstLine = True
    for data_point in train:
        train_json = {}
        evidence_ref = data_point["evidence_wiki_url"]
        try:
            if evidence_ref != "":
                if data_point["id"] not in seen_ids:
                    train_json["claim"] = data_point["claim"]
                    train_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                    unprocessed_label = data_point["label"]
                    if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                        label = 0
                    elif unprocessed_label in ["SUPPORTS"]:
                        label = 1
                    else:
                        print(unprocessed_label)
                        raise Exception("unknown label.")
                    train_json["label"] = label
                    train_json["id"] = data_point["id"]
                    seen_ids.append(data_point["id"])

            with open("train.json", "a") as outfile:
                if firstLine:
                    firstLine = False
                else:
                    outfile.write('\n')
                json.dump(train_json, outfile)
        except:
            pass

    dev = data["paper_dev"]
    seen_ids = []
    firstLine = True
    for data_point in dev:
        dev_json = {}
        evidence_ref = data_point["evidence_wiki_url"]
        try:
            if evidence_ref != "":
                if dev["id"] not in seen_ids:
                    dev_json["claim"] = data_point["claim"]
                    dev_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                    unprocessed_label = data_point["label"]
                    if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                        label = 0
                    elif unprocessed_label in ["SUPPORTS"]:
                        label = 1
                    else:
                        raise Exception("unknown label.")
                    dev_json["label"] = label
                    dev_json["id"] = data_point["id"]
                    seen_ids.append(data_point["id"])

            with open("dev.json", "a") as outfile:
                if firstLine:
                    firstLine = False
                else:
                    outfile.write('\n')
                json.dump(dev_json, outfile)
        except:
            pass

    test = data["paper_test"]
    firstLine = True
    seen_ids = []
    for data_point in test:
        test_json = {}
        evidence_ref = data_point["evidence_wiki_url"]
        try:
            if evidence_ref != "":
                if test["id"] not in seen_ids:
                    test_json["claim"] = data_point["claim"]
                    test_json["evidence"] = wiki_lookup[evidence_ref]["text"]
                    unprocessed_label = data_point["label"]
                    if unprocessed_label in ["NOT ENOUGH INFO", "REFUTES"]:
                        label = 0
                    elif unprocessed_label in ["SUPPORTS"]:
                        label = 1
                    else:
                        raise Exception("unknown label.")
                    test_json["label"] = label
                    test_json["id"] = data_point["id"]
                    seen_ids.append(data_point["id"])

            with open("test.json", "a") as outfile:
                if firstLine:
                    firstLine = False
                else:
                    outfile.write('\n')
                json.dump(test_json, outfile)
        except:
            pass

def main():
    #create_datasets()
    model_checkpoint = "mnli+rte/checkpoint-780"
    dataset = load_dataset("json", data_files={"train": f"{flair.cache_root}/datasets/fever/train.json", "test": f"{flair.cache_root}/datasets/fever/test.json", "dev": f"{flair.cache_root}/datasets/fever/dev.json"})
    metric_name = "accuracy"
    metric = load_metric('glue', "mnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    def preprocess_function(examples):
        return tokenizer(examples["claim"], examples["evidence"], truncation=True)

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