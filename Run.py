from train_eval.train import Train
from train_eval.init_train_env import *
def run():
    train=Train(*init_training_env())
    train.train()

if __name__=="__main__":
    run()