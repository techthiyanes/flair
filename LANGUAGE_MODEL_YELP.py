import itertools
from transformers import Trainer, TrainingArguments
from LANGUAGE_MODEL_FUNCTIONS import get_model, read_csv, Dataset, get_model_with_new_classifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train(model_checkpoint, samples, run):
    num_labels = 5
    if model_checkpoint == 'bert-base-uncased':
        mod = "bert"
    elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli/best_model':
        mod = "mnli_base"
    elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model':
        mod = "mnli_adv"
    else:
        mod = "unknown"

    if mod == "bert":
        model, tokenizer = get_model(model_checkpoint, num_labels)
    else:
        model, tokenizer = get_model_with_new_classifier(model_checkpoint, num_labels)

    train_texts, train_labels = read_csv('../.flair/datasets/yelp_review_full_csv/train.csv', samples=samples)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = Dataset(train_encodings, train_labels)

    test_texts, test_labels = read_csv('../.flair/datasets/yelp_review_full_csv/test.csv')
    test_encodings = tokenizer(test_texts, padding=True)
    test_dataset = Dataset(test_encodings, test_labels)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir='transformers_results_yelp',
        num_train_epochs=20,
        logging_dir='transformers_logs_yelp',
        learning_rate=3e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    scores = trainer.evaluate()

    with open(f"experiments_v2/0_bert_baseline/yelp/not_finetuned/{mod}-trained_on_{samples}-run_{run}.log", 'w') as f:
        f.write(model_checkpoint + "\n")
        f.write(f"Number of seen examples: {samples} \n")
        for metric, score in scores.items():
            f.write(f"{metric}: {score} \n")

if __name__ == "__main__":
    model_checkpoints = ['bert-base-uncased','entailment_label_sep_text/pretrained_mnli/best_model', 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model']
    number_data_points = [1,2,4,8,10,100]
    runs = [1,2,3,4,5]
    for model_checkpoint, number_of_samples, run in itertools.product(model_checkpoints, number_data_points, runs):
        train(model_checkpoint, number_of_samples, run)