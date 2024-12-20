# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data
from modules.text_preprocessor import *
from modules.networks import network_qpidgram
from modules.training import train_model


#### hyper-parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 64
dropout_rate = 0.2
####

token_types = [
        {'method': 'char', 'ngram': 1, 'name': 'char_unigram'},
        {'method': 'char', 'ngram': 2, 'name': 'char_bigram'},
        {'method': 'char', 'ngram': 3, 'name': 'char_trigram'},
        {'method': 'word', 'ngram': 0, 'name': 'word_gram'}
    ]

train_df,full_df = import_data()
X_tensor,vocab = prepare_combined_tensors(train_df,column='tidy_cod',token_types=token_types)
y_tensor, label_encoder = encode_labels(train_df, column='icd10h_code')

X_train, X_test, y_train, y_test = train_test_split_tensors(X_tensor, y_tensor)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

vocab_size = len(vocab)
output_dim = train_df['icd10h_code'].nunique()

model = network_qpidgram(input_dim=vocab_size, len_output=output_dim, dropout_rate=dropout_rate)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)