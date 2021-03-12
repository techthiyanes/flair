import flair
from flair.data import Corpus
from flair.datasets import SentenceDataset, CSVClassificationCorpus
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random
import os

def main():
    label_name_map = {'1': 'World',
                      '2': 'Sports',
                      '3': 'Business',
                      '4': 'Science Technology'
                      }
    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/ag_news_csv"
    whole_corpus: Corpus = CSVClassificationCorpus(corpus_path,
                                                   column_name_map,
                                                   skip_header=False,
                                                   delimiter=',',
                                                   label_name_map=label_name_map
                                                   )

    test_split = [x for x in whole_corpus.test]

    if whole_corpus.__class__.__name__ == "CSVClassificationCorpus":
        corpus_type = "csv"
    else:
        corpus_type = "default"

    label_ids_mapping = extract_label_ids_mapping(whole_corpus, corpus_type)

    path = "experiments/1_bert_entailment/dbpedia_to_agnews/pretrained_model/best-model.pt"

    for no_examples in [1,2]:

        for data_point in test_split:
            data_point.remove_labels("label")

        base_pretrained_tars = init_tars(path)

        few_shot_corpus = create_few_shot_corpus(label_ids_mapping, no_examples, whole_corpus, test_split, corpus_type)

        base_pretrained_tars.add_and_switch_to_new_task("AGNEWS", label_dictionary=few_shot_corpus.make_label_dictionary())

        trainer = ModelTrainer(base_pretrained_tars, few_shot_corpus)

        outpath = f"test{no_examples}"

        trainer.train(base_path=outpath,  # path to store the model artifacts
                      learning_rate=0.02,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,
                      max_epochs=20,
                      embeddings_storage_mode='none')

def init_tars(path):
    #model_path = f"{path}/pretrained_model/best-model.pt"
    model_path = "tars-base"
    tars = TARSClassifier.load(model_path)
    return tars

def extract_label_ids_mapping(corpus, corpus_type):
    if corpus_type == "default":
        id_label_tuples = [(x.labels[0].value, id) for id, x in enumerate(corpus.train.dataset.sentences)]

    elif corpus_type == "csv":
        id_label_tuples = [(x.labels[0].value, id) for id, x in enumerate(corpus.train)]

    label_ids_mapping = {}
    for key, id in id_label_tuples:
        if key not in label_ids_mapping:
            label_ids_mapping[key] = [id]
        else:
            label_ids_mapping[key].append(id)

    return label_ids_mapping

def create_few_shot_corpus(label_ids_mapping, number_examples, corpus, test_sentences, corpus_type = "default"):

        train_ids = []
        dev_ids = []
        for label, ids in label_ids_mapping.items():
            if len(ids) <= (number_examples * 2):
                samples = ids
            else:
                samples = random.sample(ids, number_examples * 2)
            middle_index = len(samples) // 2
            train_ids.extend(samples[:middle_index])
            dev_ids.extend(samples[middle_index:])

        train_sentences = []
        for id in train_ids:
            if corpus_type == "default":
                train_sentences.append(corpus.train.dataset.sentences[id])
            elif corpus_type == "csv":
                train_sentences.append(corpus.train[id])

        dev_sentences = []
        for id in dev_ids:
            if corpus_type == "default":
                dev_sentences.append(corpus.train.dataset.sentences[id])
            elif corpus_type == "csv":
                dev_sentences.append(corpus.train[id])

        # training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
        train = SentenceDataset(
            train_sentences
        )

        dev = SentenceDataset(
            dev_sentences
        )

        test = SentenceDataset(
            test_sentences
        )

        few_shot_corpus = Corpus(train=train, dev=dev, test=test)

        return few_shot_corpus

if __name__ == "__main__":
    main()