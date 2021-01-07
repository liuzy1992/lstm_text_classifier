#!/usr/bin/env python3

import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from .model import LSTM
from .savingandloading import load_model
from . import device

def evaluation(test_iter, field, file_path, lr):
    model = LSTM(field=field).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    load_model(file_path + '/model.pt', model, optimizer)

    y_pred = []
    y_true = []

    model.eval()
    for (labels, (titlecontent, titlecontent_len)), _ in test_iter:
        labels = labels.to(device)
        titlecontent = titlecontent.to(device)
        titlecontent_len = titlecontent_len.to(device)

        with torch.no_grad():
            output = model(titlecontent, titlecontent_len)
            output = (output > 0.5).int()

        y_pred.extend(output.tolist())
        y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
