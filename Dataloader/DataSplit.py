# -*- coding: utf-8 -*-

import pandas as pd 


with pd.read_csv('train.csv', chunksize=20000) as reader:
    for counter, chunk in enumerate(reader):
        if counter < 110:
            chunk.to_csv('/home/ubuntu/data/train/'+str(counter)+'.csv')
        else:
            chunk.to_csv('/home/ubuntu/data/val/'+str(counter)+'.csv')

print(counter)
