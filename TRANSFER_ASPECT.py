import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoConfig, AutoModel
from flair.datasets import SentenceDataset
import flair
from LANGUAGE_MODEL_FUNCTIONS import read_csv, sample_datasets
from flair.models.tars_tagger_model import TARSTagger
from flair.models.text_classification_model import TARSClassifier
from flair.data import Corpus, Sentence, MultitaskCorpus
from flair.models.multitask_model import MultitaskModel
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.trainers import ModelTrainer


def extract_XML(path):
    data = []
    tree = ET.parse(path)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        text = sentence.find("text").text
        flair_sentence = Sentence(text)
        for token in flair_sentence:
            token.set_label("polarity", "O")
        aspectTerms = sentence.find("aspectTerms")
        if aspectTerms:
            for aspectTerm in aspectTerms:
                _from = int(aspectTerm.get('from'))
                _to = int(aspectTerm.get('to'))
                term = aspectTerm.get("term")
                polarity = f"{aspectTerm.get('polarity')} aspect"
                _curr_from = 0
                _curr_to = 0
                for token in flair_sentence:
                    _curr_to += len(token.text) + 1
                    if _curr_from - len(term) < _from < _curr_from + len(term):
                        if _curr_to - len(term) < _to < _curr_to + len(term):
                            if term.__contains__(token.text):
                                token.set_label("polarity", polarity)
                    _curr_from = _curr_to

        data.append(flair_sentence)

    return data

def get_yelp_dataset():
    label_name_map = {'1':'very negative restaurant sentiment',
                      '2':'negative restaurant sentiment',
                      '3':'neutral restaurant sentiment',
                      '4':'positive restaurant sentiment',
                      '5':'very positive restaurant sentiment'
                      }
    train_texts, train_labels, train_class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/yelp_review_full_csv/train.csv")
    train_texts, train_labels = sample_datasets(original_texts=train_texts,
                                                original_labels=train_labels,
                                                number_of_samples=2000,
                                                class_to_datapoint_mapping=train_class_to_datapoint_mapping)
    train_labels = [x+1 for x in train_labels]

    test_texts, test_labels, test_class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/yelp_review_full_csv/test.csv")
    test_texts, test_labels = sample_datasets(original_texts=test_texts,
                                              original_labels=test_labels,
                                              number_of_samples=2000,
                                              class_to_datapoint_mapping=test_class_to_datapoint_mapping)
    test_labels = [x+1 for x in test_labels]

    train_sentences = []
    for text, label in zip(train_texts, train_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        train_sentences.append(sentence)

    del train_class_to_datapoint_mapping
    del train_labels
    del train_texts

    test_sentences = []
    for text, label in zip(test_texts, test_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        test_sentences.append(sentence)

    del test_class_to_datapoint_mapping
    del label_name_map
    del test_texts
    del test_labels

    yelp_corpus = Corpus(train=SentenceDataset(train_sentences), test=SentenceDataset(test_sentences))

    return yelp_corpus

def get_amazon_dataset():
    label_name_map = {'1': 'very negative product sentiment',
                      '2': 'negative product sentiment',
                      '3': 'neutral product sentiment',
                      '4': 'positive product sentiment',
                      '5': 'very positive product sentiment'
                      }

    train_texts, train_labels, train_class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/amazon_review_full_csv/train.csv")
    train_texts, train_labels = sample_datasets(original_texts=train_texts,
                                                original_labels=train_labels,
                                                number_of_samples=2000,
                                                class_to_datapoint_mapping=train_class_to_datapoint_mapping)
    train_labels = [x+1 for x in train_labels]

    test_texts, test_labels, test_class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/amazon_review_full_csv/test.csv")
    test_texts, test_labels = sample_datasets(original_texts=test_texts,
                                              original_labels=test_labels,
                                              number_of_samples=2000,
                                              class_to_datapoint_mapping=test_class_to_datapoint_mapping)
    test_labels = [x+1 for x in test_labels]

    train_sentences = []
    for text, label in zip(train_texts, train_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        train_sentences.append(sentence)

    del train_class_to_datapoint_mapping
    del train_labels
    del train_texts

    test_sentences = []
    for text, label in zip(test_texts, test_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        test_sentences.append(sentence)

    del test_class_to_datapoint_mapping
    del label_name_map
    del test_texts
    del test_labels

    amazon_corpus = Corpus(train=SentenceDataset(train_sentences), test=SentenceDataset(test_sentences))

    return amazon_corpus

def get_laptop_aspect_dataset():
    laptop_data = extract_XML('aspect_data/Laptop_Train_v2.xml')
    laptop_corpus = Corpus(SentenceDataset(laptop_data))
    return laptop_corpus

def get_restaurant_aspect_dataset():
    restaurant_data = extract_XML('aspect_data/Restaurants_Train_v2.xml')
    restaurant_corpus = Corpus(SentenceDataset(restaurant_data))
    return restaurant_corpus

def tars_classifier():
    for run in range(1,6):
        amazon_corpus = get_amazon_dataset()
        tars_classifier = TARSClassifier(task_name="AMAZON", label_dictionary=amazon_corpus.make_label_dictionary())
        trainer = ModelTrainer(tars_classifier, amazon_corpus)
        trainer.train(base_path=f'experiments_v2/3_results/amazon_classifier/run_{run}',  # path to store the model artifacts
                      learning_rate=0.02,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
                      max_epochs=20,  # terminate after 10 epochs
                      embeddings_storage_mode="none")

    for run in range(1,6):
        yelp_corpus = get_yelp_dataset()
        tars_classifier = TARSClassifier(task_name="YELP", label_dictionary=yelp_corpus.make_label_dictionary())
        trainer = ModelTrainer(tars_classifier, yelp_corpus)
        trainer.train(base_path=f'experiments_v2/3_results/yelp_classifier/run_{run}',  # path to store the model artifacts
                      learning_rate=0.02,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
                      max_epochs=20,  # terminate after 10 epochs
                      embeddings_storage_mode="none")

def joint_model():
    #laptop_label_dict = restaurant_corpus.make_label_dictionary("polarity")

    model_checkpoints = ['bert-base-uncased', 'entailment_label_sep_text/pretrained_mnli/best_model', 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model']
    for model_checkpoint in model_checkpoints:
        if model_checkpoint == 'bert-base-uncased':
            mod = "bert"
        elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli/best_model':
            mod = "mnli_base"
        elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model':
            mod = "mnli_adv"

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=True)
        model = AutoModel.from_pretrained(model_checkpoint, config=config)

        shared_embedding = {"tokenizer": tokenizer, "model": model}

        word_embeddings = TransformerWordEmbeddings(shared_embedding = shared_embedding)
        document_embeddings = TransformerDocumentEmbeddings(shared_embedding = shared_embedding)

        tars_tagger = TARSTagger("laptop", laptop_label_dict, "polarity", embeddings=word_embeddings)
        tars_classifier = TARSClassifier("YELP", amazon_corpus.make_label_dictionary(), document_embeddings=document_embeddings)

        multitask_corpus = MultitaskCorpus(
            {"corpus": restaurant_corpus, "model": tars_tagger},
            {"corpus": amazon_corpus, "model": tars_classifier}
        )

        multitask_model = MultitaskModel(multitask_corpus.models)

        trainer = ModelTrainer(multitask_model, multitask_corpus)

        trainer.train(base_path=f"experiments_v2/3_results/transfer_to_product/{mod}",  # path to store the model artifacts
                      learning_rate=3e-5,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,
                      max_epochs=20,
                      embeddings_storage_mode='none')


if __name__ == "__main__":
    flair.device = "cuda:1"
    tars_classifier()