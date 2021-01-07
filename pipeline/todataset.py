#!/usr/bin/env python3

import torch
from torchtext.data import Dataset, Example, Field
import spacy

class DataFrameDataset(Dataset):
    def __init__(self, df, fields, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label
            text = row.titlecontent
            examples.append(Example.fromlist([label, text], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.titlecontent)

    @classmethod
    def splits(cls, fields, train_df, valid_df=None, test_df=None, **kwargs):
        train_data, valid_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if valid_df is not None:
            valid_data = cls(valid_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, **kwargs)

        return tuple(d for d in (train_data, valid_data, test_data) if d is not None)

def todataset(train_df, valid_df, test_df, max_seq_length, max_vocab_size=25000):
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize=spacy.load("en_core_web_sm"), include_lengths=True, batch_first=True, fix_length=max_seq_length)
    fields = [('label', label_field), ('titlecontent', text_field)]

    train_ds, valid_ds, test_ds = DataFrameDataset.splits(fields, train_df=train_df, valid_df=valid_df, test_df=test_df)
    text_field.build_vocab(train_ds, max_size=max_vocab_size, min_freq=3)

    return train_ds, valid_ds, test_ds, text_field
