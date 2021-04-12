import xml.etree.ElementTree as ET

from LANGUAGE_MODEL_FUNCTIONS import read_csv, sample_datasets
from flair.data import Sentence, Corpus, TARSCorpus
from flair.datasets import TREC_6, AMAZON_REVIEWS
from flair.models.multitask_model.task_model import RefactoredTARSClassifier
from flair.models.tars_tagger_model import TARSTagger
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer


def main():

    label_name_map = {'1':'very negative restaurant sentiment',
                      '2':'negative restaurant sentiment',
                      '3':'neutral restaurant sentiment',
                      '4':'positive restaurant sentiment',
                      '5':'very positive restaurant sentiment'
                      }
    train_texts, train_labels, class_to_datapoint_mapping = read_csv(f"{flair.cache_root}/datasets/yelp_review_full_csv/train.csv")
    train_texts, train_labels = sample_datasets(original_texts=train_texts,
                                                original_labels=train_labels,
                                                number_of_samples=600,
                                                class_to_datapoint_mapping=class_to_datapoint_mapping)
    train_labels = [x+1 for x in train_labels]
    sentences = []
    for text, label in zip(train_texts, train_labels):
        sentence = Sentence(text)
        sentence.add_label("class", label_name_map[str(label)])
        sentences.append(sentence)

    del class_to_datapoint_mapping
    del label_name_map
    del train_labels
    del train_texts

    yelp_corpus = Corpus(sentences)

    tars_classifier = TARSClassifier("YELP", yelp_corpus.make_label_dictionary(), document_embeddings="bert-base-uncased")

    trainer = ModelTrainer(tars_classifier, yelp_corpus)

    trainer.train(base_path=f"testy_classifier",  # path to store the model artifacts
                  learning_rate=0.02,  # use very small learning rate
                  mini_batch_size=16,
                  mini_batch_chunk_size=8,
                  max_epochs=10,
                  embeddings_storage_mode='none')

if __name__ == "__main__":
    import flair
    flair.device = "cuda:1"
    main()