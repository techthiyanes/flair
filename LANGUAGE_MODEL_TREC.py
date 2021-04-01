from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flair.datasets import TREC_6, TREC_50

def main():
    num_labels = 50
    model_checkpoint = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def load_trec():
        corpus = TREC_50()
        labels = corpus.make_label_dictionary()
        train_texts = [corpus.train[i].to_plain_string() for i, _ in enumerate(corpus.train)]
        train_labels = [labels.item2idx[corpus.train[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.train)]
        val_texts = [corpus.dev[i].to_plain_string() for i, _ in enumerate(corpus.dev)]
        val_labels = [labels.item2idx[corpus.dev[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.dev)]
        test_texts = [corpus.test[i].to_plain_string() for i, _ in enumerate(corpus.test)]
        test_labels = [labels.item2idx[corpus.test[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.test)]

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_trec()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    class CSVDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CSVDataset(train_encodings, train_labels)
    val_dataset = CSVDataset(val_encodings, val_labels)
    test_dataset = CSVDataset(test_encodings, test_labels)

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
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='transformers_logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    trainer.evaluate()

if __name__ == "__main__":
    main()