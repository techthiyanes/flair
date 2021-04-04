import itertools
from transformers import Trainer, TrainingArguments
from LANGUAGE_MODEL_FUNCTIONS import get_model, read_trec, Dataset, get_model_with_new_classifier, sample_datasets, get_bart_model_with_new_classifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train(model_checkpoint, run, samples, train_texts, train_labels, test_texts, test_labels):

    if model_checkpoint == 'experiments_v2/0_bert_baseline/trec/finetuned/bert/best_model':
        mod = "bert"
    elif model_checkpoint == 'experiments_v2/0_bert_baseline/trec/finetuned/mnli_base/best_model':
        mod = "mnli_base"
    elif model_checkpoint == 'experiments_v2/0_bert_baseline/trec/finetuned/mnli_adv/best_model':
        mod = "mnli_adv"
    elif model_checkpoint == 'facebook/bart-large-mnli':
        mod = "bart"
    else:
        mod = "unknown"

    if mod == "bert":
        model, tokenizer = get_model(model_checkpoint, num_labels)
    elif mod == "bart":
        model, tokenizer = get_bart_model_with_new_classifier(model_checkpoint, num_labels)
    else:
        model, tokenizer = get_model_with_new_classifier(model_checkpoint, num_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = Dataset(train_encodings, train_labels)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
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
        output_dir='transformers_results_trec',
        num_train_epochs=20,
        logging_dir='transformers_logs_trec',
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

    with open(f"experiments_v2/0_bert_baseline/trec/finetuned/{mod}-trained_on_{samples}-run_{run}.log", 'w') as f:
        f.write(model_checkpoint + "\n")
        f.write(f"Number of seen examples: {samples} \n")
        for metric, score in scores.items():
            f.write(f"{metric}: {score} \n")

if __name__ == "__main__":
    model_checkpoints = ['experiments_v2/0_bert_baseline/trec/finetuned/bert/best_model', 'experiments_v2/0_bert_baseline/trec/finetuned/mnli_base/best_model', 'experiments_v2/0_bert_baseline/trec/finetuned/mnli_adv/best_model']
    number_data_points = [1,2,4,8,10,100]
    runs = [1,2,3,4,5]
    num_labels = 50
    train_texts, test_texts, train_labels, test_labels, class_to_datapoint_mapping = read_trec(num_labels=num_labels)
    for model_checkpoint, number_of_samples, run in itertools.product(model_checkpoints, number_data_points, runs):
        sampled_train_texts, sampled_train_labels = sample_datasets(original_texts=train_texts, original_labels=train_labels, number_of_samples=number_of_samples, class_to_datapoint_mapping=class_to_datapoint_mapping)
        train(model_checkpoint, run, number_of_samples, sampled_train_texts, sampled_train_labels, test_texts, test_labels)