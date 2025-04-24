# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data_standard
from modules.text_preprocessor import *
from modules.networks import network_qpidgram
from modules.training import train_model
from modules.tools import undersampling


#### hyper-parameters
learning_rate = 0.0008
batch_size = 32
num_epochs = 64
dropout_rate = 0.34
undersampling_scale = 0.4
undersampling_lower_bound = 150
####

token_types = [
        # {'method': 'char', 'ngram': 1},
        #{'method': 'char', 'ngram': 2},
        {'method': 'char', 'ngram': 3},
        {'method': 'word', 'ngram': 0}
    ]

train_df, full_df = import_data_standard()

X_tensor,vocab = prepare_deathcauses_tensors(train_df,column='tidy_cod',token_types=token_types)
y_tensor, label_encoder = encode_labels(train_df, column='icd10h_category')

# train_df = undersampling(df=train_df, target_col='icd10h_category', scale=undersampling_scale, lower_bound=undersampling_lower_bound)

X_train, X_test, y_train, y_test = train_test_split_tensors(X_tensor, y_tensor)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

vocab_size = len(vocab)
output_dim = train_df['icd10h_code'].nunique()

model = network_qpidgram(input_dim=vocab_size, len_output=output_dim, dropout_rate=dropout_rate)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)