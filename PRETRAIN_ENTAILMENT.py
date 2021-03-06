from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    actual_task = "mnli"
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    task_to_keys = {"mnli": ("premise", "hypothesis"),"rte": ("sentence1", "sentence2")}
    sentence1_key, sentence2_key = task_to_keys[actual_task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)


if __name__ == "__main__":
    main()