import torch
import os
import random
import csv
import sys
sys.path.append("../config")
from config import *
from preprocess.preprocess import *
import numpy as np
import pandas as pd
import slugify
class LibriSamples(torch.utils.data.Dataset):
    '''
    read data from CSVs and return data
    '''
    def __init__(self, full_data=False, download=False, data_path=None, sample=20000, shuffle=True,  csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample
        self.full_data = full_data
        self.download = download

        # 如果full_data==true && download==false, 则默认~/kaggle_data路径下已包含train.csv和splitted data
        if download:
            # 直接从kaggle下载zip到~，再解压，得到~/train.csv
            os.system("pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8")
            os.system("mkdir /root/.kaggle")
            with open("kaggle.json", "w+") as f:
                f.write('{"username":"zhuyuhe16","key":"6176383558b0033d473ee3714a957b7f"}')
            os.system("mkdir ~/kaggle_data")
            os.system("sudo cp kaggle.json ~/.kaggle/kaggle.json")
            os.system("chmod 600 /root/.kaggle/kaggle.json")
            os.system("kaggle competitions download -c ubiquant-market-prediction")
            os.system("mv ubiquant-market-prediction.zip " + FULL_DATA_DIR + "/ubiquant-market-prediction.zip")
            os.system("unzip -oq ~/ubiquant-market-prediction.zip -d " + FULL_DATA_DIR)
            os.system("mkdir " + FULL_DATA_DIR + "/train")

            process_train_csv(FULL_DATA_DIR + "/train.csv", FULL_DATA_DIR + "/train")

        self.X_dir = data_path

        self.X_names = os.listdir(self.X_dir)

        # using a small part of the dataset to debug
        if csvpath:
            subset = self.parse_csv(csvpath)
            self.X_names = [i for i in self.X_names if i in subset]

        # full_data==true时，指定splitted data目录
        if full_data:
            self.X_names = os.listdir(FULL_DATA_DIR + "/train")

        # if shuffle == True:


        self.length = len(self.X_names)


    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))

    def __getitem__(self, i):
        sample_range = range(i * self.sample, min((i + 1) * self.sample, self.length))
        # TODO rewrite getitem code, return X and Y
        X, Y = [], []
        for j in sample_range:
            # append jth csv to X&Y, read sample # of csv each time
            X_path = self.X_dir + '\\'+self.X_names[j]
            if self.full_data:
                X_path = '~/kaggle_data/train' + '/' + self.X_names[j]

            # X:features
            data = pd.read_csv(X_path)
            X_data=data.drop(['Unnamed: 0','target'],axis=1).to_numpy()
            Y_data=data['target'].to_numpy()
            #X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
            X.append(X_data)
            Y.append(Y_data)

        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y


class LibriItems(torch.utils.data.Dataset):
    '''
    generate training DataSet for DataLoader
    '''
    #FIXME: 这边我们也许可以保留context？
    def __init__(self, X, Y, context=0):
        assert (X.shape[0] == Y.shape[0])

        self.length = X.shape[0]
        self.context = context

        if context == 0:
            self.X, self.Y = X, Y
        else:
            X = np.pad(X, ((context, context), (0, 0)), 'constant', constant_values=(0, 0))
            self.X, self.Y = X, Y

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # TODO: return X & Y
        # X shape:(batch_length,feature_length)
        if self.context == 0:
            xx = self.X[i].flatten().astype(np.float32)
            yy = self.Y[i]
        else:
            xx = self.X[i:(i + 2 * self.context + 1)].flatten().astype(np.float32)
            yy = self.Y[i]
        return xx, yy


if __name__=="__main__":
    librisample_test=LibriSamples(DEBUG_PATH)
    for i in range(len(librisample_test)):
        X,Y=librisample_test[i]
        train_items = LibriItems(X, Y )
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True)#num_workers=1
        for batch_idx, (data, target) in enumerate(train_loader):
            input()
