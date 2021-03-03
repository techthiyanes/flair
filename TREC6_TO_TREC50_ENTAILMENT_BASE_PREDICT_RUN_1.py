from sklearn import metrics
from torch.utils.data import Dataset

import flair
from flair.data import Corpus
from flair.datasets import TREC_50, SentenceDataset, DataLoader
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
import random
import torch
import numpy as np
from flair.training_utils import store_embeddings, Result

flair.device = "cuda:0"

def main():
    # 2. get the corpus
    corpus: Corpus = TREC_50()
    label_dictionary = corpus.make_label_dictionary()
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

    # 3. create a TARS classifier
    #tars = TARSClassifier.load("experiments/1_entailment_baseline/trec6_to_trec50/run_1/0_examples")
    tars = TARSClassifier.load("tars-base")
    tars.add_and_switch_to_new_task("TREC_50", label_dictionary=corpus.make_label_dictionary())

    if not isinstance(corpus.train, Dataset):
        sentences = SentenceDataset(corpus.train)
    else:
        sentences = corpus.train
    data_loader = DataLoader(sentences, num_workers=6)

    # use scikit-learn to evaluate
    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in data_loader:

            # remove previously predicted labels
            [sentence.remove_labels('predicted') for sentence in batch]

            # get the gold labels
            true_values_for_batch = [sentence.get_labels("class") for sentence in batch]

            # predict for batch
            tars.predict(batch,
                        embedding_storage_mode='gpu',
                        label_name='predicted')

            # get the predicted labels
            predictions = [sentence.get_labels('predicted') for sentence in batch]

            for predictions_for_sentence, true_values_for_sentence in zip(
                    predictions, true_values_for_batch
            ):

                true_values_for_sentence = [label.value for label in true_values_for_sentence]
                predictions_for_sentence = [label.value for label in predictions_for_sentence]

                y_true_instance = np.zeros(len(label_dictionary), dtype=int)
                for i in range(len(label_dictionary)):
                    if label_dictionary.get_item_for_index(i) in true_values_for_sentence:
                        y_true_instance[i] = 1
                y_true.append(y_true_instance.tolist())

                y_pred_instance = np.zeros(len(label_dictionary), dtype=int)
                for i in range(len(label_dictionary)):
                    if label_dictionary.get_item_for_index(i) in predictions_for_sentence:
                        y_pred_instance[i] = 1
                y_pred.append(y_pred_instance.tolist())

            store_embeddings(batch, 'gpu')

        # remove predicted labels
        for sentence in sentences:
            sentence.annotation_layers['predicted'] = []

        # make "classification report"
        target_names = []
        for i in range(len(label_dictionary)):
            target_names.append(label_dictionary.get_item_for_index(i))
        classification_report = metrics.classification_report(y_true, y_pred, digits=4,
                                                              target_names=target_names, zero_division=0)

        # get scores
        micro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=1.0, average='micro', zero_division=0),
                              4)
        accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)
        macro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=1.0, average='macro', zero_division=0),
                              4)
        precision_score = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
        recall_score = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0), 4)

        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {micro_f_score}"
                f"\n- F-score (macro) {macro_f_score}"
                f"\n- Accuracy {accuracy_score}"
                '\n\nBy class:\n' + classification_report
        )

        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" \
                   f"{recall_score}\t" \
                   f"{macro_f_score}\t" \
                   f"{accuracy_score}"

        result = Result(
            main_score=micro_f_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
        )

        print(accuracy_score)


if __name__ == "__main__":
    main()