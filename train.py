# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:24:53 2019

@author: rehakomoon
"""

from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import VRCDataset
from model import Model


if __name__ == "__main__":

    dataset_dir_train = Path("E:/vrc_rotation/dataset/resized/")
    dataset_dir_test = Path("E:/vrc_rotation/dataset/resized_validation_aoinu/")
    log_dir = Path("E:/vrc_rotation/log/")
    logfile_path = Path("E:/vrc_rotation/log/log.txt")
    
    log_dir.mkdir(exist_ok=True)
    
    #with open(logfile_path, "w") as fout:
    #    pass
    
    batch_size = 128
    num_epoch = 10000
    initial_epoch = 0
    
    dataset_train = VRCDataset(dataset_dir_train, (256, 256))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dataset_test = VRCDataset(dataset_dir_test, (256, 256))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
    
    larning_rate = 0.001
    weight_decay = 0.0
    
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=larning_rate, weight_decay=weight_decay)
    
    model_epoch_list = [int(str(s)[-10:-4]) for s in log_dir.glob("model_*.pth")]
    if (len(model_epoch_list) > 0):
        latest_model_path = log_dir / f"model_{max(model_epoch_list):06}.pth"
        print(f"load {latest_model_path}...")
        model_params = torch.load(str(latest_model_path))
        model.load_state_dict(model_params)
        initial_epoch = max(model_epoch_list) + 1
    
    model = model.cuda()
    criterion = criterion.cuda()
    
    for epoch in range(initial_epoch, num_epoch):
        sum_loss = 0
        sum_correct = 0
        num_seen = 0
        
        model.train()
        for i, (batch_x, batch_y) in enumerate(dataloader_train):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            
            batch_size = batch_x.shape[0]
    
            optimizer.zero_grad()
            
            batch_y = batch_y.view(-1)
            batch_y_pred = model(batch_x)
            loss = criterion(batch_y_pred, batch_y)
            
            loss.backward()
            optimizer.step()
    
            _, estimated = batch_y_pred.max(dim=1)
            correct = (batch_y == estimated).sum()
    
            sum_loss += loss.item()
            sum_correct += correct.item()
            num_seen += batch_size

        if True:
            print(f'e: {epoch},\t loss: {sum_loss/num_seen},\t acc: {sum_correct/num_seen}')
            with open(logfile_path, "a") as fout:
                fout.write(f"t, {epoch}, {sum_loss/num_seen}, {sum_correct/num_seen}\n")

        if epoch%10 == 0:
            sum_loss = 0
            sum_correct = 0
            num_seen = 0
            model.eval()
            
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(dataloader_test):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    
                    batch_size = batch_x.shape[0]
                    
                    batch_y = batch_y.view(-1)
                    batch_y_pred = model(batch_x)
                    loss = criterion(batch_y_pred, batch_y)
                    
                    _, estimated = batch_y_pred.max(dim=1)
                    correct = (batch_y == estimated).sum()
            
                    sum_loss += loss.item()
                    sum_correct += correct.item()
                    num_seen += batch_size
            print(f"test e: {epoch},\t loss: {sum_loss/num_seen},\t acc: {sum_correct/num_seen}")
            with open(logfile_path, "a") as fout:
                fout.write(f"e, {epoch}, {sum_loss/num_seen}, {sum_correct/num_seen}\n")

        if epoch%10 == 0:
            model_save_path = log_dir / f"model_{epoch:06}.pth"
            torch.save(model.state_dict(), str(model_save_path))
