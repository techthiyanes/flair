from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer

def main():
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
    tars = TARSClassifier(task_name='TREC_6', label_dictionary=corpus.make_label_dictionary(), document_embeddings="bart-base-mnli")

    # 4. initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # 5. start the training
    trainer.train(base_path='experiments/1_entailment_baseline/yelp_to_amz/run_1', # path to store the model artifacts
                  learning_rate=0.02, # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=20, # terminate after 10 epochs
                  )

if __name__ == "__main__":
    main()