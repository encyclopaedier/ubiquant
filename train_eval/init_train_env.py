from Model import *
from config.config import *
import torch.optim as optim
import torch
from Model.Models import *

def init_training_env():
    model=Model()
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    batch_size=BATCH_SIZE
    path=TRAINING_PATH
    epoch=EPOCHS
    num_workers=NUM_WORKERS
    return model,optimizer,criterion,scaler,batch_size,path,epoch,num_workers