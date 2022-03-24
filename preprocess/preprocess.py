# 按横截面split，因为metric是correlation coefficient，估计是ic法的metrics，可能会按照横截面计算correlation coefficient
import pandas as pd
from config.config import *
import os
def process_train_csv(in_path=ORIGINAL_FILE_PATH, data_path=TRAINING_PATH):
    data = pd.read_csv(in_path)
    time_id=data['time_id'].unique()
    for i in time_id:
      temp=data[data['time_id'].isin([i])]
      path=os.path.join(data_path,str(i)+'.csv')
      temp.to_csv(path)