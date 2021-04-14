import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoConfig, AutoModel

import flair
from LANGUAGE_MODEL_FUNCTIONS import read_csv, sample_datasets
from flair.datasets import SentenceDataset
from flair.models.multitask_model.task_model import RefactoredTARSClassifier
from flair.models.tars_tagger_model import TARSTagger
from flair.models.text_classification_model import TARSClassifier
from flair.data import Corpus, Sentence, TARSCorpus, MultitaskCorpus
from flair.models.multitask_model import MultitaskModel
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.trainers import ModelTrainer
import random

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

def main():
    laptop_data = extract_XML('aspect_data/Laptop_Train_v2.xml')
    #restaurant_data = extract_XML('aspect_data/Restaurants_Train_v2.xml')
    laptop_corpus = Corpus(laptop_data)
    #restaurant_corpus = Corpus(SentenceDataset(restaurant_data))

    laptop_label_dict = laptop_corpus.make_label_dictionary("polarity")

    label_name_map = {'1':'very negative restaurant sentiment',
                      '2':'negative restaurant sentiment',
                      '3':'neutral restaurant sentiment',
                      '4':'positive restaurant sentiment',
                      '5':'very positive restaurant sentiment'
                      }
    train_texts, train_labels, class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/yelp_review_full_csv/train.csv")
    train_texts, train_labels = sample_datasets(original_texts=train_texts,
                                                original_labels=train_labels,
                                                number_of_samples=1000,
                                                class_to_datapoint_mapping=class_to_datapoint_mapping)
    train_labels = [x+1 for x in train_labels]
    sentences = []
    for text, label in zip(train_texts, train_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        sentences.append(sentence)

    yelp_corpus = Corpus(sentences)

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
        tars_classifier = TARSClassifier("YELP", yelp_corpus.make_label_dictionary(), document_embeddings=document_embeddings)

        multitask_corpus = MultitaskCorpus(
            {"corpus": laptop_corpus, "model": tars_tagger},
            {"corpus": yelp_corpus, "model": tars_classifier}
        )

        multitask_model = MultitaskModel(multitask_corpus.models)

        trainer = ModelTrainer(multitask_model, multitask_corpus)

        trainer.train(base_path=f"experiments_v2/3_results/transfer_to_restaurant/{mod}",  # path to store the model artifacts
                      learning_rate=3e-5,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,
                      max_epochs=20,
                      embeddings_storage_mode='none')


if __name__ == "__main__":
    #flair.device = "cuda:0"
    main()