# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data_standard
from modules.text_preprocessor import *
from modules.networks import network_qpidgram
from modules.training import train_model


#### hyper-parameters
learning_rate = 0.0008
batch_size = 32
num_epochs = 64
dropout_rate = 0.34
####

token_types = [
        # {'method': 'char', 'ngram': 1},
        #{'method': 'char', 'ngram': 2},
        {'method': 'char', 'ngram': 3},
        {'method': 'word', 'ngram': 0}
    ]

train_df, full_df = import_data_standard()

