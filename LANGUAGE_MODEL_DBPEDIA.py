from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    num_labels = 14
    model_checkpoint = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def read_csv(file):
        texts = []
        labels = []
        with open(file) as f:
            filereader = csv.reader(f, delimiter=',')
            for row in filereader:
                texts.append(row[2])
                labels.append(int(row[0]) - 1)
        return texts, labels

    train_texts, train_labels = read_csv('../.flair/datasets/dbpedia_csv/train.csv')
    test_texts, test_labels = read_csv('../.flair/datasets/dbpedia_csv/test.csv')

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

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