import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class SummaryDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index=0, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep="\t")
        self.len = self.docs.shape[0]
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def __getitem__(self, idx):
        # instance = self.docs.iloc[idx]
        # input_ids = self.tok.encode(instance["text"])
        # input_ids = self.add_padding_data(input_ids)

        # label_ids = self.tok.encode(instance["summary"])
        # label_ids.append(self.tok.eos_token_id)
        # dec_input_ids = [self.pad_index]
        # dec_input_ids += label_ids[:-1]
        # dec_input_ids = self.add_padding_data(dec_input_ids)
        # label_ids = self.add_ignored_data(label_ids)
        instance = self.docs.iloc[idx]
        inputs = self.tok(
            instance["text"], padding="max_length", truncation=True, max_length=self.max_len
        )
        outputs = self.tok(
            instance["summary"], padding="max_length", truncation=True, max_length=self.max_len
        )
        labels_ids = outputs.input_ids.copy()
        labels = self.add_ignored_data(labels_ids)

        return {
            "input_ids": torch.tensor(inputs.input_ids),
            "attention_mask": torch.tensor(inputs.attention_mask),
            "decoder_input_ids": torch.tensor(labels_ids),
            "decoder_attention_mask": torch.tensor(outputs.attention_mask),
            "labels": torch.tensor(labels).long(),
        }

    def __len__(self):
        return self.len
