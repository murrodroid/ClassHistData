# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data_random
from modules.text_preprocessor import *
from modules.networks import individualized_network
from modules.training import train_model
from modules.tools import return_device, undersampling
from sklearn.preprocessing import StandardScaler


#### hyper-parameters
learning_rate = 0.0008
batch_size = 64
num_epochs = 16
dropout_rate = 0.34
retain_pct = 0.4
####

device = return_device()

df, _ = import_data_random(retain_pct)
print('Data imported successfully.')

df = df[df.icd10h_random.notna()]
