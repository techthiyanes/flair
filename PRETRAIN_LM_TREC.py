from transformers import Trainer, TrainingArguments
from LANGUAGE_MODEL_FUNCTIONS import get_model, read_trec, Dataset, get_model_with_new_classifier, get_bart_model_with_new_classifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train(model_checkpoint, train_texts, train_labels, test_texts, test_labels):
    num_labels = 6
    if model_checkpoint == 'bert-base-uncased':
        mod = "bert"
    elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli/best_model':
        mod = "mnli_base"
    elif model_checkpoint == 'entailment_label_sep_text/pretrained_mnli_rte_fever/best_model':
        mod = "mnli_adv"
    else:
        mod = "bart"

    if mod == "bert":
        model, tokenizer = get_model(model_checkpoint, num_labels)
    elif mod == "bart":
        model, tokenizer = get_bart_model_with_new_classifier(model_checkpoint, num_labels)
    else:
        model, tokenizer = get_model_with_new_classifier(model_checkpoint, num_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    train_dataset = Dataset(train_encodings, train_labels)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
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
        output_dir='transformers_results_yelplm',
        num_train_epochs=20,
        logging_dir='transformers_logs_yelplm',
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

    trainer.evaluate()

    trainer.save_model(f"experiments_v2/0_bert_baseline/trec/finetuned/{mod}/best_model")
    tokenizer.save_pretrained(f"experiments_v2/0_bert_baseline/trec/finetuned/{mod}/best_model")

if __name__ == "__main__":
    model_checkpoints = ['facebook/bart-large-mnli']
    train_texts, test_texts, train_labels, test_labels, class_to_datapoint_mapping = read_trec(num_labels=6)
    for model_checkpoint in model_checkpoints:
        train(model_checkpoint, train_texts, train_labels, test_texts, test_labels)