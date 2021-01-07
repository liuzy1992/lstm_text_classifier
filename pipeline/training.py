#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from .model import LSTM
from .savingandloading import save_model
from . import device

def cal_acc(y_pred, y_true):
    y_pred = torch.round(y_pred)
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)

    return acc

def training(train_iter,
             valid_iter,
             field,
             num_epochs,
             lr,
             file_path,
             criterion=nn.BCELoss()):
    model = LSTM(field=field).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-7)
    
    train_loss_list = []
    valid_loss_list = []
#    train_r2_list = []
#    valid_r2_list = []
    train_acc_list = []
    valid_acc_list = []   
    epoch_list = []

    best_valid_loss = float("Inf")
    total_t0 = time.time()
    
    for epoch in range(0, num_epochs):
        t0 = time.time()
        total_train_loss = 0
#        total_train_r2 = 0
        total_train_acc = 0

        model.train()

        for (labels, (titlecontent, titlecontent_len)), _ in train_iter:
            labels = labels.to(device)
            titlecontent = titlecontent.to(device)
            titlecontent_len = titlecontent_len.to(device)            
            output = model(titlecontent, titlecontent_len)
            loss = criterion(output, labels)

            total_train_loss += loss.item()
#            total_train_r2 += r2_score(labels.tolist(), output.tolist())
            total_train_acc += cal_acc(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_iter)
#        avg_train_r2 = total_train_r2 / len(train_iter)
        avg_train_acc = total_train_acc / len(train_iter)

        model.eval()

        total_valid_loss = 0
#        total_valid_r2 = 0
        total_valid_acc = 0

        for (labels, (titlecontent, titlecontent_len)), _ in valid_iter:
            labels = labels.to(device)
            titlecontent = titlecontent.to(device)
            titlecontent_len = titlecontent_len.to(device)

            with torch.no_grad():
                output = model(titlecontent, titlecontent_len)
                loss = criterion(output, labels)

            total_valid_loss += loss.item()
#            total_valid_r2 += r2_score(labels.tolist(), output.tolist())
            total_valid_acc += cal_acc(output, labels)

            scheduler.step(loss)
            
        avg_valid_loss = total_valid_loss / len(valid_iter)
#        avg_valid_r2 = total_valid_r2 / len(valid_iter)
        avg_valid_acc = total_valid_acc / len(valid_iter)

        print("## Epoch {:}/{:} ==> Train Loss: {:.5f}, Train ACC: {:.5f}; Valid Loss: {:.5f}, Valid ACC: {:.5f}; Elapsed Time: {:.2f} s".format(epoch + 1,
                         num_epochs,
                         avg_train_loss,
                         avg_train_acc,
                         avg_valid_loss,
                         avg_valid_acc,
                         time.time() - t0))

        train_loss_list.append(avg_train_loss)
#        train_r2_list.append(avg_train_r2)
        train_acc_list.append(avg_train_acc)
        valid_loss_list.append(avg_valid_loss)
#        valid_r2_list.append(avg_valid_r2)
        valid_acc_list.append(avg_valid_acc)
        epoch_list.append(epoch + 1)

        if best_valid_loss > avg_valid_loss:
            best_valid_loss = avg_valid_loss
            save_model(file_path + '/' + 'model.pt', model, optimizer, best_valid_loss)

    print("Training Complete! Total Elapsed Time: {:.2f} s.".format(time.time() - total_t0))

    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, train_acc_list, label='Train ACC')
    plt.plot(epoch_list, valid_loss_list, label='Valid Loss')
    plt.plot(epoch_list, valid_acc_list, label='Valid ACC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
