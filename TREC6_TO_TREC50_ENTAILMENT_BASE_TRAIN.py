import flair
import os
from flair.data import Corpus
from flair.datasets import TREC_6, TREC_50, SentenceDataset
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random

def train_base_model():
    # 1. define label names in natural language since some datasets come with cryptic set of labels
    label_name_map = {'ENTY':'question about entity',
                      'DESC':'question about description',
                      'ABBR':'question about abbreviation',
                      'HUM':'question about person',
                      'NUM':'question about number',
                      'LOC':'question about location'
                      }

    # 2. get the corpus
    corpus: Corpus = TREC_6(label_name_map=label_name_map)

    # 3. create a TARS classifier
    tars = TARSClassifier(task_name='TREC_6', label_dictionary=corpus.make_label_dictionary(), document_embeddings="facebook/bart-large-mnli")

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # 5. start the training
    trainer.train(base_path='experiments/1_entailment_baseline_mnli/trec6_to_trec50/pretrained_model', # path to store the model artifacts
                  learning_rate=0.02, # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=20, # terminate after 10 epochs
                  embeddings_storage_mode='gpu')

def train_few_shot_model():
    path = "experiments/1_entailment_baseline_mnli/trec6_to_trec50/"
    model = os.path.join(path, "pretrained_model/best-model.pt")
    number_of_seen_examples = 2

    for run_number in range(5):
        tars = TARSClassifier.load(model)

        trec50_label_name_map = {'ENTY:sport': 'question about entity sport',
                                 'ENTY:dismed': 'question about entity diseases medicine',
                                 'LOC:city': 'question about location city',
                                 'DESC:reason': 'question about description reasons',
                                 'NUM:other': 'question about number other',
                                 'LOC:state': 'question about location state',
                                 'NUM:speed': 'question about number speed',
                                 'NUM:ord': 'question about number order ranks',
                                 'ENTY:event': 'question about entity event',
                                 'ENTY:substance': 'question about entity element substance',
                                 'NUM:perc': 'question about number percentage fractions',
                                 'ENTY:product': 'question about entity product',
                                 'ENTY:animal': 'question about entity animal',
                                 'DESC:manner': 'question about description manner of action',
                                 'ENTY:cremat': 'question about entity creative pieces inventions books',
                                 'ENTY:color': 'question about entity color',
                                 'ENTY:techmeth': 'question about entity technique method',
                                 'NUM:dist': 'question about number distance measure',
                                 'NUM:weight': 'question about number weight',
                                 'LOC:mount': 'question about location mountains',
                                 'HUM:title': 'question about person title',
                                 'HUM:gr': 'question about person group organization of persons',
                                 'HUM:desc': 'question about person description',
                                 'ABBR:abb': 'question about abbreviation abbreviation',
                                 'ENTY:currency': 'question about entity currency',
                                 'DESC:def': 'question about description definition',
                                 'NUM:code': 'question about number code',
                                 'LOC:other': 'question about location other',
                                 'ENTY:other': 'question about entity other',
                                 'ENTY:body': 'question about entity body organ',
                                 'ENTY:instru': 'question about entity musical instrument',
                                 'ENTY:termeq': 'question about entity term equivalent',
                                 'NUM:money': 'question about number money prices',
                                 'NUM:temp': 'question about number temperature',
                                 'LOC:country': 'question about location country',
                                 'ABBR:exp': 'question about abbreviation expression',
                                 'ENTY:symbol': 'question about entity symbol signs',
                                 'ENTY:religion': 'question about entity religion',
                                 'HUM:ind': 'question about person individual',
                                 'ENTY:letter': 'question about entity letters characters',
                                 'NUM:date': 'question about number date',
                                 'ENTY:lang': 'question about entity language',
                                 'ENTY:veh': 'question about entity vehicle',
                                 'NUM:count': 'question about number count',
                                 'ENTY:word': 'question about entity word special property',
                                 'NUM:period': 'question about number period lasting time',
                                 'ENTY:plant': 'question about entity plant',
                                 'ENTY:food': 'question about entity food',
                                 'NUM:volsize': 'question about number volume size',
                                 'DESC:desc': 'question about description description'
                                 }
        corpus: Corpus = TREC_50(label_name_map=trec50_label_name_map)

        train_id_label_tuples = [(x.labels[0].value, id) for id, x in enumerate(corpus.train.dataset.sentences)]
        train_label_ids_mapping = {}
        for key, id in train_id_label_tuples:
            if key not in train_label_ids_mapping:
                train_label_ids_mapping[key] = [id]
            else:
                train_label_ids_mapping[key].append(id)

        train_sentences = []
        for label, ids in train_label_ids_mapping.items():
            random.seed(run_number)
            samples = random.sample(ids, number_of_seen_examples)
            for sample in samples:
                train_sentences.append(corpus.train.dataset.sentences[sample])

        test_id_label_tuples = [(x.labels[0].value, id) for id, x in enumerate(corpus.test.sentences)]
        test_label_ids_mapping = {}
        for key, id in test_id_label_tuples:
            if key not in test_label_ids_mapping:
                test_label_ids_mapping[key] = [id]
            else:
                test_label_ids_mapping[key].append(id)

        test_sentences = []
        for label, ids in test_label_ids_mapping.items():
            random.seed(42)
            samples = random.sample(ids, 1)
            for sample in samples:
                test_sentences.append(corpus.test.sentences[sample])

        random.seed(run_number)
        random.shuffle(train_sentences)
        # training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
        train = SentenceDataset(
            train_sentences
        )

        test = SentenceDataset(
            test_sentences
        )

        # make a corpus with train and test split
        few_shot_corpus = Corpus(train=train)

        tars.add_and_switch_to_new_task("TREC_50", label_dictionary=few_shot_corpus.make_label_dictionary())

        trainer = ModelTrainer(tars, few_shot_corpus)

        # 5. start the training
        trainer.train(base_path=f'experiments/1_entailment_baseline_mnli/trec6_to_trec50/pretrained_model_{number_of_seen_examples}_examples/run_{run_number}', # path to store the model artifacts
                      learning_rate=0.02, # use very small learning rate
                      mini_batch_size=16,
                      max_epochs=20, # terminate after 10 epochs
                      embeddings_storage_mode='gpu')

if __name__ == "__main__":
    train_few_shot_model()