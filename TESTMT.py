from flair.models.tars_tagger_model import TARSTagger
from flair.datasets import CONLL_03
from flair.trainers import ModelTrainer

def main():

    conll = CONLL_03().downsample(0.01)

    tagger = TARSTagger("conll_ner", conll.make_tag_dictionary("ner"), tag_type="ner")

    trainer = ModelTrainer(tagger, conll)

    trainer.train(base_path="testy",  # path to store the model artifacts
                  learning_rate=0.02,  # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=15,
                  embeddings_storage_mode='none')

if __name__ == "__main__":
    import flair
    flair.device = "cuda:0"
    main()