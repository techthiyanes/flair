import flair
from flair.data import Corpus
from flair.datasets import SentenceDataset, CSVClassificationCorpus
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random

flair.device = "cuda:0"

def train_base_model(path, document_embeddings):
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

    # 3. create a TARS classifier
    tars = TARSClassifier(task_name='AG_NEWS', label_dictionary=whole_corpus.make_label_dictionary(), document_embeddings=document_embeddings)

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, whole_corpus)

    # 5. start the training
    trainer.train(base_path=f"{path}/pretrained_model", # path to store the model artifacts
                  learning_rate=0.02, # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=20)

def train_few_shot_model(path):
    base_pretrained_model_path = f"{path}/pretrained_model/best-model.pt"
    number_of_seen_examples = [0, 1, 2, 4, 8, 10, 100]

    for no_examples in number_of_seen_examples:
        for run_number in range(5):
            if no_examples == 0 and run_number == 0:
                base_pretrained_tars = TARSClassifier.load(base_pretrained_model_path)
                label_name_map = {'1': 'Company',
                                  '2': 'Educational Institution',
                                  '3': 'Artist',
                                  '4': 'Athlete',
                                  '5': 'Office Holder',
                                  '6': 'Mean Of Transportation',
                                  '7': 'Building',
                                  '8': 'Natural Place',
                                  '9': 'Village',
                                  '10': 'Animal',
                                  '11': 'Plant',
                                  '12': 'Album',
                                  '13': 'Film',
                                  '14': 'Written Work'
                                  }

                # 2. get the corpus
                column_name_map = {0: "label", 2: "text"}
                corpus_path = f"{flair.cache_root}/datasets/dbpedia_csv"

                whole_corpus: Corpus = CSVClassificationCorpus(corpus_path,
                                                         column_name_map,
                                                         skip_header=False,
                                                         delimiter=',',
                                                         label_name_map=label_name_map
                                                         )
                tp = 0
                all = 0
                classes = [key for key in label_name_map.values()]
                base_pretrained_tars.predict_zero_shot(whole_corpus.test, classes, multi_label=False)
                for sentence in whole_corpus.test:
                    true = sentence.get_labels("class")[0]
                    pred = sentence.get_labels("label")[0]
                    if pred:
                        if pred.value == true.value:
                            tp += 1
                    all += 1

                with open(f"{path}/zeroshot.log", "w") as file:
                    file.write(f"Accuracy: {tp / all}")
                    file.write(f"Correct predictions: {tp}")
                    file.write(f"Total labels: {all}")

            elif no_examples > 0:
                base_pretrained_tars = TARSClassifier.load(base_pretrained_model_path)
                label_name_map = {'1': 'Company',
                                  '2': 'Educational Institution',
                                  '3': 'Artist',
                                  '4': 'Athlete',
                                  '5': 'Office Holder',
                                  '6': 'Mean Of Transportation',
                                  '7': 'Building',
                                  '8': 'Natural Place',
                                  '9': 'Village',
                                  '10': 'Animal',
                                  '11': 'Plant',
                                  '12': 'Album',
                                  '13': 'Film',
                                  '14': 'Written Work'
                                  }

                # 2. get the corpus
                column_name_map = {0: "label", 2: "text"}
                corpus_path = f"{flair.cache_root}/datasets/dbpedia_csv"

                whole_corpus: Corpus = CSVClassificationCorpus(corpus_path,
                                                         column_name_map,
                                                         skip_header=False,
                                                         delimiter=',',
                                                         label_name_map=label_name_map
                                                         )

                few_shot_corpus = create_few_shot_corpus(no_examples, whole_corpus)

                base_pretrained_tars.add_and_switch_to_new_task("DBPEDIA", label_dictionary=few_shot_corpus.make_label_dictionary())

                trainer = ModelTrainer(base_pretrained_tars, few_shot_corpus)

                outpath = f'{path}/fewshot_with_{no_examples}/run_{run_number}'

                trainer.train(base_path=outpath, # path to store the model artifacts
                              learning_rate=0.02, # use very small learning rate
                              mini_batch_size=16,
                              max_epochs=20)

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
    # TODOS
    # CHECK CUDA ASSIGNMENT
    # CHECK EXPERIMENT
    # CHECK TASK
    # CHECK DOCUMENT EMBEDDINGS
    # CHECK CORPORA + TASK DESCRIPTION
    path = 'experiments'
    experiment = "1_bert_baseline"
    task = "agnews_to_dbpedia"
    experiment_path = f"{path}/{experiment}/{task}"
    train_base_model(experiment_path, document_embeddings="bert-base-uncased")
    train_few_shot_model(experiment_path)

