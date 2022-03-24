import os

# =============================================================================
# PATH
DATA_DIR=r'G:\dataset\ubiquant\data'
ORIGINAL_FILE_NAME='train.csv'
ORIGINAL_FILE_PATH=os.path.join(DATA_DIR,ORIGINAL_FILE_NAME)

DEBUG_PATH=r'G:\dataset\ubiquant\data\debug'
TRAINING='train'
VALIDATION='validation'

TRAINING_PATH=os.path.join(DATA_DIR,TRAINING)
VALIDATION_PATH=os.path.join(DATA_DIR,VALIDATION)

FULL_DATA_DIR = '~/kaggle_data'
# =============================================================================
# training config
EPOCHS = 30
BATCH_SIZE=10
LEARNING_RATE=0.0001

#==============================================================================
# dataLoader
NUM_WORKERS=4