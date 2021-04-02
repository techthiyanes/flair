import csv
import copy
import torch
from torch import nn
import random
from transformers import BertForSequenceClassification, BertTokenizer

def get_model(model_checkpoint, num_labels):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return model, tokenizer

def get_model_with_new_classifier(model_checkpoint, num_labels):
    encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    new_model = copy.deepcopy(encoder)
    decoder = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    new_model.classifier = decoder.classifier
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    return encoder, tokenizer

def read_csv(file, samples = None):
    texts = []
    labels = []
    if samples: class_to_datapoint_mapping = dict()
    with open(file) as f:
        filereader = csv.reader(f, delimiter=',')
        for id, row in enumerate(filereader):
            texts.append(row[2])
            label = int(row[0]) - 1
            labels.append(label)
            if samples:
                if label in class_to_datapoint_mapping:
                    class_to_datapoint_mapping[label].append(id)
                else:
                    class_to_datapoint_mapping[label] = [id]

    if samples:
        texts, labels = sample_datasets(texts, labels, samples, class_to_datapoint_mapping)

    return texts, labels

def sample_datasets(original_texts, original_labels, number_of_samples, class_to_datapoint_mapping):
    sampled_texts = []
    sampled_labels = []
    for cls in class_to_datapoint_mapping.keys():
        ids = random.sample(class_to_datapoint_mapping[cls], number_of_samples)
        for id in ids:
            sampled_texts.append(original_texts[id])
            sampled_labels.append(original_labels[id])

    return sampled_texts, sampled_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)