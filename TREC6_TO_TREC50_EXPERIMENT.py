import flair
from flair.data import Corpus
from flair.datasets import SentenceDataset, CSVClassificationCorpus, TREC_6, TREC_50
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random

def train_base_model(corpus, path, document_embeddings):
    # 3. create a TARS classifier
    tars = TARSClassifier(task_name='TREC6', label_dictionary=corpus.make_label_dictionary(),
                          document_embeddings=document_embeddings)

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # 5. start the training
    trainer.train(base_path=path,
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=5,
                  embeddings_storage_mode='none')


def train_few_shot_model(path):
    label_name_map = {'ENTY:sport': 'question about entity sport',
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
    whole_corpus: Corpus = TREC_50(label_name_map=label_name_map)

    number_of_seen_examples = [0, 1, 2, 4, 8, 10, 100]

    test_split = [x for x in whole_corpus.test]

    if whole_corpus.__class__.__name__ == "CSVClassificationCorpus":
        corpus_type = "csv"
    else:
        corpus_type = "default"

    label_ids_mapping = extract_label_ids_mapping(whole_corpus, corpus_type)

    for no_examples in number_of_seen_examples:
        for run_number in range(5):

            for data_point in test_split:
                data_point.remove_labels("label")

            if no_examples == 0 and run_number == 0:
                tp = 0
                all = 0
                classes = [key for key in label_name_map.values()]
                base_pretrained_tars = init_tars(path)
                base_pretrained_tars.predict_zero_shot(test_split, classes, multi_label=False)
                for sentence in test_split:
                    true = sentence.get_labels("class")[0]
                    pred = sentence.get_labels("label")[0]
                    if pred:
                        if pred.value == true.value:
                            tp += 1
                    all += 1

                with open(f"{path}/zeroshot.log", "w") as file:
                    file.write(f"Accuracy: {tp / all} \n")
                    file.write(f"Correct predictions: {tp} \n")
                    file.write(f"Total labels: {all} \n")

            elif no_examples > 0:

                base_pretrained_tars = init_tars(path)

                few_shot_corpus = create_few_shot_corpus(label_ids_mapping, no_examples, whole_corpus, test_split,
                                                         corpus_type)

                base_pretrained_tars.add_and_switch_to_new_task("TREC50",
                                                                label_dictionary=few_shot_corpus.make_label_dictionary())

                trainer = ModelTrainer(base_pretrained_tars, few_shot_corpus)

                # path = experiemnts/1_bert_baseline/trec_to_news
                outpath = f'{path}/fewshot_with_{no_examples}/run_{run_number}'

                trainer.train(base_path=outpath,  # path to store the model artifacts
                              learning_rate=0.02,  # use very small learning rate
                              mini_batch_size=16,
                              max_epochs=20,
                              embeddings_storage_mode='none')


def init_tars(path):
    model_path = f"{path}/pretrained_model/best-model_epoch5.pt"
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


def create_few_shot_corpus(label_ids_mapping, number_examples, corpus, test_sentences, corpus_type="default"):
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
    # TODOS
    # CHECK CUDA ASSIGNMENT
    # CHECK EXPERIMENT
    # CHECK TASK
    # CHECK DOCUMENT EMBEDDINGS
    # CHECK CORPORA + TASK DESCRIPTION
    flair.device = "cuda:3"
    # 1. define label names in natural language since some datasets come with cryptic set of labels
    label_name_map = {'ENTY':'question about entity',
                      'DESC':'question about description',
                      'ABBR':'question about abbreviation',
                      'HUM':'question about person',
                      'NUM':'question about number',
                      'LOC':'question about location'
                      }

    # 2. get the corpus
    trec6: Corpus = TREC_6(label_name_map=label_name_map).downsample(0.1)

    path_model_mapping = {
        "bart-entailment":
            {
                "path": "1_entailment_bart",
                "model": "facebook/bart-large-mnli"
            },
        "bert-entailment-standard":
            {
                "path": "1_entailment_standard_reversed",
                "model": "entailment_label_sep_text/pretrained_mnli/best_model"
            },
        "bert-entailment-advanced":
            {
                "path": "1_entailment_advanced_reversed",
                "model": "entailment_label_sep_text/pretrained_mnli_rte_fever/best_model"
            }
    }

    task = "trec6_to_trec50"
    for model_description, configuration in path_model_mapping.items():
        experiment_path = f"testy/experiments_v2/{configuration['path']}/{task}"
        #train_base_model(trec6, f"{experiment_path}/pretrained_model",
        #                 document_embeddings=f"distilbert-base-uncased")
        train_few_shot_model(experiment_path)