import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoConfig, AutoModel

import flair
from flair.datasets import SentenceDataset, CSVClassificationCorpus
from flair.models.tars_tagger_model import TARSTagger
from flair.data import Corpus, Sentence, TARSCorpus, MultitaskCorpus
from flair.models.multitask_model import MultitaskModel
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models.text_classification_model import TARSClassifier
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
                polarity = aspectTerm.get("polarity")
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

    laptop_corpus = Corpus(SentenceDataset(laptop_data))
    #restaurant_corpus = Corpus(SentenceDataset(restaurant_data))

    laptop_label_dict = laptop_corpus.make_label_dictionary("polarity")

    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/amazon_review_full_csv"
    label_name_map = {'1': 'very negative product sentiment',
                      '2': 'negative product sentiment',
                      '3': 'neutral product sentiment',
                      '4': 'positive product sentiment',
                      '5': 'very positive product sentiment'
                      }
    yelp_corpus: Corpus = CSVClassificationCorpus(corpus_path,
                                                   column_name_map,
                                                   skip_header=False,
                                                   delimiter=',',
                                                   label_name_map=label_name_map
                                                   ).downsample(0.002)
    yelp_label_dict = yelp_corpus.make_label_dictionary()

    model_checkpoints = ['bert-base-uncased','entailment_label_sep_text/pretrained_mnli/best_model', 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model']
    for model_checkpoint in model_checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=True)
        model = AutoModel.from_pretrained(model_checkpoint, config=config)

        shared_embedding = {"tokenizer": tokenizer, "model": model}

        word_embeddings = TransformerWordEmbeddings(shared_embedding = shared_embedding)
        document_embeddings = TransformerDocumentEmbeddings(shared_embedding = shared_embedding)

        tars_tagger = TARSTagger("laptop", laptop_label_dict, "polarity", embeddings=word_embeddings)
        tars_classifier = TARSClassifier(task_name='YELP', label_dictionary=yelp_label_dict, document_embeddings=document_embeddings)

        multitask_corpus = MultitaskCorpus(
            {"corpus": laptop_corpus, "task": "sequence_tagger"},
            {"corpus": yelp_corpus, "task": "text_classifier"}
        )

        multitask_model = MultitaskModel(multitask_corpus.models)

        trainer = ModelTrainer(multitask_model, multitask_corpus)

        trainer.train(base_path="testy",  # path to store the model artifacts
                      learning_rate=0.02,  # use very small learning rate
                      mini_batch_size=16,
                      max_epochs=20,
                      embeddings_storage_mode='none')


if __name__ == "__main__":
    flair.device = "cuda:0"
    main()