#!/usr/bin/env python3

from torchtext.data import BucketIterator
from . import device

def iterator(train_ds, valid_ds, test_ds, batch_size):
    train_iter, valid_iter, test_iter = BucketIterator.splits((train_ds, valid_ds, test_ds), 
                                                              batch_size=batch_size, 
                                                              sort_within_batch=True,
                                                              device=device)

    return train_iter, valid_iter, test_iter
