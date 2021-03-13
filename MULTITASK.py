import flair
from flair.data import Corpus
from flair.datasets import TREC_50, CSVClassificationCorpus

def get_corpora():
    trec50_label_name_map = {'ENTY:sport': 'question about entity sport',
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
    trec50: Corpus = TREC_50(label_name_map=trec50_label_name_map)

agnews_label_name_map = {'1': 'World',
                          '2': 'Sports',
                          '3': 'Business',
                          '4': 'Science Technology'
                          }
column_name_map = {0: "label", 2: "text"}
corpus_path = f"{flair.cache_root}/datasets/ag_news_csv"
agnews: Corpus = CSVClassificationCorpus(
    corpus_path,
    column_name_map,
    skip_header=False,
    delimiter=',',
    label_name_map=agnews_label_name_map
)

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
column_name_map = {0: "label", 2: "text"}
corpus_path = f"{flair.cache_root}/datasets/dbpedia_csv"
dbpedia: Corpus = CSVClassificationCorpus(
    corpus_path,
    column_name_map,
    skip_header=False,
    delimiter=',',
    label_name_map=dbpedia_label_name_map
).downsample(0.25)


if __name__ == "__main__":
    flair.device = "cuda:3"
    corpora = get_corpora()
    train_sequential_model()
    train_multitask_model()