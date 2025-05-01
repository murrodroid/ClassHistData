# packages
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# modules
from modules.import_data import import_data_standard
from modules.text_preprocessor import *
from modules.networks import CategoryClassifier
from modules.training import train_model
from modules.tools import undersampling, return_device


#### hyper-parameters
learning_rate = 0.0004
batch_size = 32
num_epochs = 64
dropout_rate = 0.55
undersampling_scale = 0.4
undersampling_lower_bound = 150
####

save_path = 'trained_models/icd10h_upper_category.pth'

device = return_device()

token_types = [
        # {'method': 'char', 'ngram': 1},
        # {'method': 'char', 'ngram': 2},
        {'method': 'char', 'ngram': 3},
        # {'method': 'word', 'ngram': 0}
    ]

train_df, full_df = import_data_standard(target='icd10h_category')

X_tensor,vocab = prepare_deathcauses_tensors(train_df,column='tidy_cod',token_types=token_types)
y_tensor, label_encoder = encode_labels(train_df, transform_column='icd10h_category')
X_train, X_test, y_train, y_test = train_test_split_tensors(X_tensor, y=y_tensor, test_size=0.1)

vocab_size = len(vocab)
output_dim = train_df['icd10h_category'].nunique()

train_loader,test_loader = create_dataloaders(train=[X_train,y_train],test=[X_test,y_test],batch_size=batch_size)
print('Dataloaders prepared and ready.')

model = CategoryClassifier(vocab_size=vocab_size, 
                           num_classes=output_dim, 
                           embed_dim=256,
                           dropout=dropout_rate,
                           ).to(device)
print('Network initialized.')

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, device=device, save_path=save_path)