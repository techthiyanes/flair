import flair
from flair.data import Corpus, Sentence
from flair.models.text_classification_model import TARSClassifier
from flair.tokenization import SegtokTokenizer
from flair.trainers import ModelTrainer
from flair.datasets import TREC_50, CSVClassificationCorpus, SentenceDataset

def make_text(data_point, text_columns):
    return [data_point[0], " ".join(data_point[text_column] for text_column in text_columns)]

def make_sentence(data_point, tokenizer):
    s = Sentence(data_point[1], use_tokenizer=tokenizer)
    s.add_label("class", data_point[0])
    return s

def get_corpora():
    tokenizer = SegtokTokenizer()

    # TREC50 CORPUS
    trec50_label_name_map = {
        'ENTY:sport': 'question about entity sport',
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
    trec_full: Corpus = TREC_50(label_name_map=trec50_label_name_map)
    trec_train = Corpus(train=trec_full.train, dev=trec_full.dev)
    trec_test = trec_full.test

    # AGNEWS CORPUS
    agnews_label_name_map = {
        '1': 'World',
          '2': 'Sports',
          '3': 'Business',
          '4': 'Science Technology'
    }
    column_name_map = {0: "label", 1: "text", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/ag_news_csv"
    agnews_full: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=agnews_label_name_map
    )
    agnews_train = Corpus(train=agnews_full.train, dev=agnews_full.dev)
    text_columns = agnews_full.train.dataset.text_columns
    test_split_sentences = [make_text(data_point, text_columns) for data_point in agnews_full.test.raw_data]
    test_split_sentences = [make_sentence(data_point, tokenizer) for data_point in test_split_sentences]
    agnews_test = SentenceDataset(test_split_sentences)

    # DBPEDIA CORPUS
    dbpedia_label_name_map = {'1': 'Company',
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
    column_name_map = {0: "label", 1: "text", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/dbpedia_csv"
    dbpedia_full: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=dbpedia_label_name_map
    )
    dbpedia_train = Corpus(train=dbpedia_full.train, dev=dbpedia_full.dev).downsample(0.25)
    text_columns = dbpedia_full.train.dataset.text_columns
    test_split_sentences = [make_text(data_point, text_columns) for data_point in dbpedia_full.test.raw_data]
    test_split_sentences = [make_sentence(data_point, tokenizer) for data_point in test_split_sentences]
    dbpedia_test = SentenceDataset(test_split_sentences)

    # AMAZON CORPUS
    amazon_label_name_map = {'1': 'very negative product sentiment',
                      '2': 'negative product sentiment',
                      '3': 'neutral product sentiment',
                      '4': 'positive product sentiment',
                      '5': 'very positive product sentiment'
                      }
    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/amazon_review_full_csv"
    amazon_full: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=amazon_label_name_map
    )
    amazon_train = Corpus(train=amazon_full.train, dev=amazon_full.dev).downsample(0.03)
    text_columns = amazon_full.train.dataset.text_columns
    test_split_sentences = [make_text(data_point, text_columns) for data_point in amazon_full.test.raw_data]
    test_split_sentences = [make_sentence(data_point, tokenizer) for data_point in test_split_sentences]
    amazon_test = SentenceDataset(test_split_sentences)

    # YELP CORPUS
    yelp_label_name_map = {'1': 'very negative restaurant sentiment',
                      '2': 'negative restaurant sentiment',
                      '3': 'neutral restaurant sentiment',
                      '4': 'positive restaurant sentiment',
                      '5': 'very positive restaurant sentiment'
                      }
    column_name_map = {0: "label", 1: "text"}
    corpus_path = f"{flair.cache_root}/datasets/yelp_review_full_csv"
    yelp_full: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=yelp_label_name_map
    )
    yelp_train = Corpus(train=yelp_full.train, dev=yelp_full.dev).downsample(0.1)
    text_columns = yelp_full.train.dataset.text_columns
    test_split_sentences = [make_text(data_point, text_columns) for data_point in yelp_full.test.raw_data]
    test_split_sentences = [make_sentence(data_point, tokenizer) for data_point in test_split_sentences]
    yelp_test = SentenceDataset(test_split_sentences)

    return {"trec50":
                {
                    "train": trec_train,
                    "test": trec_test
                },
            "agnews":
                {
                    "train":agnews_train,
                    "test":agnews_test
                },
            "dbpedia":
                {
                    "train": dbpedia_train,
                    "test": dbpedia_test
                },
            "amazon":
                {
                    "train": amazon_train,
                    "test": amazon_test
                },
            "yelp":
                {
                    "train": yelp_train,
                    "test": yelp_test
                }
            }

def train_sequential_model(corpora, configurations):
    amazon_train = corpora.get("amazon").get("train")
    tars = TARSClassifier(task_name='AMAZON', label_dictionary=amazon_train.make_label_dictionary(),
                          document_embeddings=configurations["model"])
    trainer = ModelTrainer(tars, amazon_train)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/1_after_amazon",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    yelp_train = corpora.get("yelp").get("train")
    tars.add_and_switch_to_new_task("YELP", label_dictionary=yelp_train.make_label_dictionary())
    trainer = ModelTrainer(tars, yelp_train)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/2_after_yelp",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    dbpedia_train = corpora.get("dbpedia").get("train")
    tars.add_and_switch_to_new_task("DBPEDIA", label_dictionary=dbpedia_train.make_label_dictionary())
    trainer = ModelTrainer(tars, dbpedia_train)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/3_after_dbpedia",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    agnews_train = corpora.get("agnews").get("train")
    tars.add_and_switch_to_new_task("AGNEWS", label_dictionary=agnews_train.make_label_dictionary())
    trainer = ModelTrainer(tars, agnews_train)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/4_after_agnews",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    trec_train = corpora.get("trec50").get("train")
    tars.add_and_switch_to_new_task("TREC50", label_dictionary=trec_train.make_label_dictionary())
    trainer = ModelTrainer(tars, trec_train)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/5_after_trec",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

def eval_sequential_model(corpora, configurations):
    best_model_path = f"{configurations['path']}/sequential_model/5_after_trec/best-model.pt"
    out_path = f"{configurations['path']}/sequential_model/evaluation"
    best_model = TARSClassifier.load(best_model_path)
    amazon_test = corpora.get("amazon").get("test")
    best_model.evaluate(amazon_test, out_path=f"{out_path}/amazon.log")
    yelp_test = corpora.get("yelp").get("test")
    best_model.evaluate(yelp_test, out_path=f"{out_path}/yelp.log")
    dbpedia_test = corpora.get("dbpedia").get("test")
    best_model.evaluate(dbpedia_test, out_path=f"{out_path}/dbpedia.log")
    agnews_test = corpora.get("agnews").get("test")
    best_model.evaluate(agnews_test, out_path=f"{out_path}/agnews.log")
    trec_test = corpora.get("trec").get("test")
    best_model.evaluate(trec_test, out_path=f"{out_path}/trec.log")

def train_multitask_model():
    pass

if __name__ == "__main__":
    flair.device = "cuda:0"
    path_model_mapping = {
        "bert-base-uncased":
            {
                "path" : "2_bert_baseline",
                "model": "distilbert-base-uncased"
            },
        "bert-entailment-standard":
            {
                "path": "2_entailment_standard",
                "model": "distilbert_entailment_label_sep_text/pretrained_mnli/best_model"
            },
        "bert-entailment-advanced":
            {
                "path": "2_entailment_advanced",
                "model": "distilbert_entailment_label_sep_text/pretrained_mnli_rte_fever/best_model"
            }
    }
    corpora = get_corpora()
    for key, configurations in path_model_mapping.items():
        train_sequential_model(corpora, configurations)
        eval_sequential_model(corpora, configurations)
        #train_multitask_model()