from flair.data import Corpus
from flair.datasets import TREC_50
from flair.models.text_classification_model import TARSClassifier
import os
import logging

#flair.device = "cuda:0"

def main():
    path = "experiments/1_entailment_baseline_mnli/trec6_to_trec50/"
    number_of_seen_examples = 0
    experiment = f"{number_of_seen_examples}_examples"
    logging_path = os.path.join(path, experiment)
    model = os.path.join(path, "pretrained_model/best-model.pt")
    logging.basicConfig(filename=f'{logging_path}/result.log', filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')

    for run_number in range(5):

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
        corpus: Corpus = TREC_50(label_name_map=trec50_label_name_map)

        """
        step1 = [(x.labels[0].value, id) for id, x in enumerate(corpus.train.dataset.sentences)]
        step2 = {}
        for key, id in step1:
            if key not in step2:
                step2[key] = [id]
            else:
                step2[key].append(id)
    
        for label, ids in step2.items():
            random.seed(42)
            random.sample(ids, 3)
        """

        tars = TARSClassifier.load(model)

        tp = 0
        all = 0
        classes = [key for key in trec50_label_name_map.values()]

        for sentence in corpus.train:
            tars.predict_zero_shot(sentence, classes, multi_label=False)
            true = sentence.get_labels("class")[0]
            pred = sentence.get_labels("label")
            if pred:
                if pred.value == true.value:
                    tp += 1
            all += 1

        for sentence in corpus.test:
            tars.predict_zero_shot(sentence, classes, multi_label=False)
            true = sentence.get_labels("class")[0]
            pred = sentence.get_labels("label")
            if pred:
                if pred.value == true.value:
                    tp += 1
            all += 1

        for sentence in corpus.dev:
            tars.predict_zero_shot(sentence, classes, multi_label=False)
            true = sentence.get_labels("class")[0]
            pred = sentence.get_labels("label")
            if pred:
                if pred.value == true.value:
                    tp += 1
            all += 1

        logging.warning(50*'-')
        logging.warning(f"Run {run_number}")
        logging.warning(f"Seen examples from TREC50: {number_of_seen_examples}")
        logging.warning(f"Accuracy: {tp / all}")
        logging.warning(f"TP:{tp} out of {all}.")

    logging.warning(50 * '-')

if __name__ == "__main__":
    main()