import csv
import copy
import torch
import random
from flair.datasets import TREC_6, TREC_50
from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def get_model(model_checkpoint, num_labels):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return model, tokenizer

def get_model_with_new_classifier(model_checkpoint, num_labels):
    pretrained_bert = BertForSequenceClassification.from_pretrained(model_checkpoint)
    new_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    pretrained_bert_copy = copy.deepcopy(pretrained_bert)
    new_model.bert = pretrained_bert_copy.bert
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return new_model, tokenizer

def get_bart_model_with_new_classifier(model_checkpoint, num_labels):
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)
    pretrained_bart = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    pretrained_bart.classification_head.out_proj = torch.nn.Linear(in_features=1024, out_features=num_labels, bias=True)
    pretrained_bart.config = config
    pretrained_bart.num_labels = config.num_labels
    torch.nn.init.xavier_uniform_(pretrained_bart.classification_head.out_proj.weight)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return pretrained_bart, tokenizer

def read_csv(file):
    texts = []
    labels = []
    class_to_datapoint_mapping = dict()
    with open(file) as f:
        filereader = csv.reader(f, delimiter=',')
        for id, row in enumerate(filereader):
            if file.__contains__("yelp"):
                row_id = 1
            else:
                row_id = 2
            texts.append(row[row_id])
            label = int(row[0]) - 1
            labels.append(label)
            if label in class_to_datapoint_mapping:
                class_to_datapoint_mapping[label].append(id)
            else:
                class_to_datapoint_mapping[label] = [id]
    return texts, labels, class_to_datapoint_mapping

def read_trec(num_labels=50):
    if num_labels == 50:
        corpus = TREC_50()
    elif num_labels == 6:
        corpus = TREC_6()
    else:
        raise Exception()

    train_texts = [corpus.train[i].to_plain_string() for i, _ in enumerate(corpus.train)]
    dev_texts = [corpus.dev[i].to_plain_string() for i, _ in enumerate(corpus.dev)]
    train_texts = train_texts + dev_texts
    test_texts = [corpus.test[i].to_plain_string() for i, _ in enumerate(corpus.test)]

    label_dict = corpus.make_label_dictionary()
    train_labels = [label_dict.item2idx[corpus.train[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.train)]
    dev_labels = [label_dict.item2idx[corpus.dev[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.dev)]
    train_labels = train_labels + dev_labels
    test_labels = [label_dict.item2idx[corpus.test[i].labels[0].value.encode("utf-8")] for i, _ in enumerate(corpus.test)]

    class_to_datapoint_mapping = dict()
    for id, label in enumerate(train_labels):
        if label in class_to_datapoint_mapping:
            class_to_datapoint_mapping[label].append(id)
        else:
            class_to_datapoint_mapping[label] = [id]

    return train_texts, test_texts, train_labels, test_labels, class_to_datapoint_mapping

def sample_datasets(original_texts, original_labels, number_of_samples, class_to_datapoint_mapping):
    sampled_texts = []
    sampled_labels = []
    for cls in class_to_datapoint_mapping.keys():
        if number_of_samples < len(class_to_datapoint_mapping[cls]):
            ids = random.sample(class_to_datapoint_mapping[cls], number_of_samples)
        else:
            ids = class_to_datapoint_mapping[cls]
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