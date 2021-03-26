import flair
from flair.data import Corpus
from flair.datasets import SentenceDataset, CSVClassificationCorpus
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random

def train_base_model(corpus, path, document_embeddings):

    # 3. create a TARS classifier
    tars = TARSClassifier(task_name='YELP', label_dictionary=corpus.make_label_dictionary(), document_embeddings=document_embeddings)

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # 5. start the training
    trainer.train(base_path=path,
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=20,
                  embeddings_storage_mode='none')

def train_few_shot_model(path):
    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/amazon_review_full_csv"
    label_name_map = {'1': 'very negative product sentiment',
                      '2': 'negative product sentiment',
                      '3': 'neutral product sentiment',
                      '4': 'positive product sentiment',
                      '5': 'very positive product sentiment'
                      }
    whole_corpus: Corpus = CSVClassificationCorpus(corpus_path,
                                                   column_name_map,
                                                   skip_header=False,
                                                   delimiter=',',
                                                   label_name_map=label_name_map
                                                   ).downsample(0.05)

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

                few_shot_corpus = create_few_shot_corpus(label_ids_mapping, no_examples, whole_corpus, test_split, corpus_type)

                base_pretrained_tars.add_and_switch_to_new_task("AMAZON", label_dictionary=few_shot_corpus.make_label_dictionary())

                trainer = ModelTrainer(base_pretrained_tars, few_shot_corpus)

                # path = experiemnts/1_bert_baseline/trec_to_news
                outpath = f'{path}/fewshot_with_{no_examples}/run_{run_number}'

                trainer.train(base_path=outpath, # path to store the model artifacts
                              learning_rate=0.02, # use very small learning rate
                              mini_batch_size=16,
                              max_epochs=20,
                              embeddings_storage_mode='none')

def init_tars(path):
    model_path = f"{path}/pretrained_model/best-model.pt"
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
    # TODOS
    # CHECK CUDA ASSIGNMENT
    # CHECK EXPERIMENT
    # CHECK TASK
    # CHECK DOCUMENT EMBEDDINGS
    # CHECK CORPORA + TASK DESCRIPTION
    flair.device = "cuda:1"
    label_name_map = {'1':'very negative restaurant sentiment',
                      '2':'negative restaurant sentiment',
                      '3':'neutral restaurant sentiment',
                      '4':'positive restaurant sentiment',
                      '5':'very positive restaurant sentiment'
                      }
    column_name_map = {0: "label", 1: "text"}
    corpus_path = f"{flair.cache_root}/datasets/yelp_review_full_csv"
    yelp: Corpus = CSVClassificationCorpus(corpus_path,
                                             column_name_map,
                                             skip_header=False,
                                             delimiter=',',
                                             label_name_map=label_name_map
                                             ).downsample(0.25)

    path_model_mapping = {
        "bert-entailment-standard":
            {
                "path": "1_entailment_standard",
                "model": "distilbert_entailment_label_sep_text/pretrained_mnli/best_model"
            },
        "bert-entailment-advanced":
            {
                "path": "1_entailment_advanced",
                "model": "distilbert_entailment_label_sep_text/pretrained_mnli_rte_fever/best_model"
            }
    }

    task = "yelp_to_amazon_NEW"
    for model_description, configuration in path_model_mapping.items():
        experiment_path = f"experiments_v2/{configuration['path']}/{task}"
        train_base_model(yelp, f"{experiment_path}/pretrained_model", document_embeddings=f"{configuration['model']}")
        train_few_shot_model(experiment_path)
