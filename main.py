#!/usr/bin/env python3

import sys
import time
from pipeline import *

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def main(infile,
         max_length,
         batch_size,
         num_epochs,
         learning_rate,
         model_outdir):
    print()
    print(get_time() + " Loading data ...")
    df_train, df_valid, df_test = preprocessing(infile)
    print()
    print(get_time() + " Transforming data ...")
    train_ds, valid_ds, test_ds, field = todataset(df_train, df_valid, df_test, max_length)
    train_iter, valid_iter, test_iter = iterator(train_ds, valid_ds, test_ds, batch_size)
    print()
    print(get_time() + " Start training LSTM model ...")
    training(train_iter, valid_iter, field, num_epochs, learning_rate, model_outdir)
    print()
    print(get_time() + " Evaluate model on test data ...")
    evaluation(test_iter, field, model_outdir, learning_rate)

main(infile = sys.argv[1],
     max_length = int(sys.argv[2]),
     batch_size = int(sys.argv[3]),
     num_epochs = int(sys.argv[4]),
     learning_rate = float(sys.argv[5]),
     model_outdir = sys.argv[6])
