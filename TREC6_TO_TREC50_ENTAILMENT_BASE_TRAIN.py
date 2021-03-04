import flair
from flair.data import Corpus
from flair.datasets import TREC_6, TREC_50, SentenceDataset
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random

flair.device = "cuda:0"

def train_base_model(path, document_embeddings):
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
    tars = TARSClassifier(task_name='TREC_6', label_dictionary=corpus.make_label_dictionary(), document_embeddings=document_embeddings)

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # 5. start the training
    trainer.train(base_path=f"{path}/pretrained_model", # path to store the model artifacts
                  learning_rate=0.02, # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=20, # terminate after 10 epochs
                  embeddings_storage_mode='gpu')

def train_few_shot_model(path):
    base_pretrained_model_path = f"{path}/pretrained_model/best-model.pt"
    number_of_seen_examples = [0, 1, 2, 4, 8, 10, 100]

    for no_examples in number_of_seen_examples:
        for run_number in range(5):
            base_pretrained_tars = TARSClassifier.load(base_pretrained_model_path)

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
            whole_corpus: Corpus = TREC_50(label_name_map=trec50_label_name_map)

            if no_examples == 0 and run_number == 0:
                tp = 0
                all = 0
                classes = [key for key in trec50_label_name_map.values()]
                base_pretrained_tars.predict_zero_shot(whole_corpus.test, classes, multi_label=False)
                for sentence in whole_corpus.test:
                    true = sentence.get_labels("class")[0]
                    pred = sentence.get_labels("label")[0]
                    if pred:
                        if pred.value == true.value:
                            tp += 1
                    all += 1

                with open(f"{path}/zeroshot2.log", "w") as file:
                    file.write(f"Accuracy: {tp / all}")
                    file.write(f"Correct predictions: {tp}")
                    file.write(f"Total labels: {all}")

            elif no_examples > 0:
                few_shot_corpus = create_few_shot_corpus(no_examples, whole_corpus)

                base_pretrained_tars.add_and_switch_to_new_task("TREC_50", label_dictionary=few_shot_corpus.make_label_dictionary())

                trainer = ModelTrainer(base_pretrained_tars, few_shot_corpus)

                outpath = f'{path}/fewshot_with_{no_examples}/run_{run_number}'

                trainer.train(base_path=outpath, # path to store the model artifacts
                              learning_rate=0.02, # use very small learning rate
                              mini_batch_size=16,
                              max_epochs=20, # terminate after 10 epochs
                              embeddings_storage_mode='gpu')

def create_few_shot_corpus(number_examples, corpus):

    id_label_tuples = [(x.labels[0].value, id) for id, x in enumerate(corpus.train.dataset.sentences)]
    label_ids_mapping = {}
    for key, id in id_label_tuples:
        if key not in label_ids_mapping:
            label_ids_mapping[key] = [id]
        else:
            label_ids_mapping[key].append(id)

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
        train_sentences.append(corpus.train.dataset.sentences[id])

    dev_sentences = []
    for id in dev_ids:
        dev_sentences.append(corpus.train.dataset.sentences[id])

    # training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
    train = SentenceDataset(
        train_sentences
    )

    dev = SentenceDataset(
        dev_sentences
    )

    few_shot_corpus = Corpus(train=train, dev=dev, test=corpus.test)

    return few_shot_corpus

if __name__ == "__main__":
    path = 'experiments'
    experiment = "1_entailment_baseline_mnli"
    task = "trec6_to_trec50"
    experiment_path = f"{path}/{experiment}/{task}"
    #train_base_model(experiment_path, document_embeddings="facebook/bart-large-mnli")
    train_few_shot_model(experiment_path)