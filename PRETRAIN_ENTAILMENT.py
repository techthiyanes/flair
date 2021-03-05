import transformers

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW

class MNLIDataBert(Dataset):

    def __init__(self, train_df, val_df):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        self.train_df = train_df
        self.val_df = val_df

        self.base_path = '/content/'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.train_data = None
        self.val_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)

    def load_data(self, df):
        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premise_list = df['sentence1'].to_list()
        hypothesis_list = df['sentence2'].to_list()
        label_list = df['gold_label'].to_list()

        for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [
                self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
            premise_len = len(premise_id)
            hypothesis_len = len(hypothesis_id)

            segment_ids = torch.tensor(
                [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            y.append(self.label_dict[label])

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        print(len(dataset))
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=shuffle,
            batch_size=batch_size
        )

        return train_loader, val_loader


def main():
    x = load_dataset("multi_nli")

if __name__ == "__main__":
    main()