# -*- coding: utf-8 -*-

import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

class Network(torch.nn.Module):
    def __init__(self, context=0):
        super(Network, self).__init__()
        neurons = [300, 2048, 2048, 1024, 1024, 512, 128, 32, 1]
        layers = []
        for i in range(len(neurons)-2):
          layers.append(nn.Linear(neurons[i], neurons[i+1]))
          layers.append(nn.BatchNorm1d(neurons[i+1]))
          layers.append(nn.SiLU())
        layers.append(nn.Dropout(0.4))
        layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.layers = nn.Sequential(*layers)
        # self.criterion = nn.MSELoss()

    def forward(self, A0):
        x = self.layers(A0)
        return x
    
    # def cal_Loss(self,y_hat,y):
    #     return self.criterion(y_hat,y)

class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= 'train', shuffle=False): 

        self.X_dir = data_path + '/' + partition + '/'
        
        self.X_files = os.listdir(self.X_dir)
        self.shuffle = shuffle


    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):
        
        df = pd.read_csv(self.X_dir + self.X_files[ind])
        features = [f'f_{i}' for i in range(300)]
        
        if self.shuffle:
            df_shuffle=pd.DataFrame()
            for i in df["time_id"].unique():
                df_subset = df[df['time_id'] == i]
                df_subset = df_subset.sample(frac=1).reset_index(drop=True)
                df_shuffle = pd.concat([df_shuffle, df_subset])
            df = df_shuffle
    
        X = torch.FloatTensor(df[features].to_numpy())
        Y = torch.FloatTensor(df['target'].to_numpy())

        return X, Y
    def collate_fn(batch):

        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        return batch_x, batch_y    

# class LibriItems(torch.utils.data.Dataset):
#     def __init__(self, X, Y, context = 0):
        
#         self.length  = X.shape[0]
#         self.X, self.Y = X, Y

        
#     def __len__(self):
#         return self.length
        
#     def __getitem__(self, i):

#         xx = self.X[i]
#         yy = self.Y[i]

#         return xx, yy
    
#     def collate_fn(batch):

#         batch_x = [x for x,y in batch]
#         batch_y = [y for x,y in batch]

#         return batch_x, batch_y

def get_corr(y,target):
    y,target = y.reshape(-1),target.reshape(-1)
    ymean,targetmean = torch.mean(y),torch.mean(target)
    
    vy = y-ymean
    vt = target-targetmean
    
    corr = torch.sum(vy*vt)/(torch.sqrt(torch.sum(vy**2))* torch.sqrt(torch.sum(vt**2)))
    return corr

batch_size=128
epochs=30
save_path = '/home/ubuntu/efs/mlp.pth'
train_data = LibriSamples('/home/ubuntu/data','train',shuffle=True)
val_data = LibriSamples('/home/ubuntu/data','val')
train_loader = DataLoader(train_data, batch_size=batch_size,collate_fn=LibriSamples.collate_fn,shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size,collate_fn=LibriSamples.collate_fn)

model = Network().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.02, weight_decay=1e-4, amsgrad=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-8,verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
scaler = torch.cuda.amp.GradScaler()

# for i in range(len(train_samples)):
#     X, Y = train_samples[i]
#     train_items = LibriItems(X, Y)
#     train_loader = DataLoader(train_items, batch_size=batch_size,collate_fn=LibriItems.collate_fn)

torch.cuda.empty_cache()
best_loss = 1000
for epoch in range(epochs):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    total_loss = 0
    total_corr = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        x, y = data
        x = x.cuda()
        y = y.cuda()

        
        with torch.cuda.amp.autocast():
            y_hat = model(x)
            loss = criterion(y_hat, y)

        total_loss += float(loss)
        total_corr += float(get_corr(y_hat,y))
        # total_dist += float(dist)
        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss/(i+1))),
            corr="{:.04f}".format(float(total_corr/(i+1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16       
        scheduler.step()
        
        batch_bar.update() # Update tqdm bar
    batch_bar.close()
    print("Epoch {}/{}: Train Loss {:.04f},Train corr {:.04f}, Learning Rate {:.04f}".format(
        epoch + 1,
        epochs,
        float(total_loss / len(train_loader)),
        float(total_corr / len(train_loader)),
        float(optimizer.param_groups[0]['lr'])))
    
    ##### validation####
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    val_loss=0
    val_corr=0
    for i, data in enumerate(val_loader):
        x, y = data
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            y_hat = model(x)
        
        val_loss += float(criterion(y_hat, y))
        val_corr += float(get_corr(y_hat,y))

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(val_loss="{:.04f}".format(val_loss/(i+1)),val_corr="{:.04f}".format(float(val_corr/(i+1))))
        batch_bar.update() # Update tqdm bar
    batch_bar.close()
    avg_val_loss = float(val_loss / len(val_loader))
    print("Validation: Loss {:.04f}, corr {:.04f} ".format(
        avg_val_loss,
        float(val_corr / len(val_loader))))
    
    if(val_loss<best_loss):
        best_loss = val_loss
        torch.save(model.state_dict(),save_path)
        print("successfully save model")



