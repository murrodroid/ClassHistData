# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data_individualized
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
undersampling_scale = 0.15
####

device = return_device()

df, _ = import_data_individualized()
print('Data imported successfully.')

val_sample = df.sample(n=round(df.shape[0]*0.002),random_state=333)
df = df.drop(val_sample.index)
train_df = undersampling(df=df[df.icd10h.notna()], target_col='icd10h', scale=undersampling_scale, lower_bound=70, verbose=True)

token_types = [
    #{'method': 'char', 'ngram': 1},
    {'method': 'char', 'ngram': 2},
    {'method': 'char', 'ngram': 3},
    {'method': 'word', 'ngram': 0}
]

# should we train on deathcause_mono or deathcauses?
X_cause, vocab = prepare_deathcauses_tensors(df=train_df,column='deathcause_mono',token_types=token_types)
print(f'Deathcauses tensors are prepared with format: {token_types}')
X_age  = torch.tensor(StandardScaler().fit_transform(train_df['age'].to_numpy().reshape(-1,1)),dtype=torch.float).to(device)
X_sex = torch.tensor(LabelEncoder().fit_transform(train_df['sex']),dtype=torch.long).to(device)
y_tensor, y_label_encoder = encode_labels(train_df,column='icd10h')
num_classes = train_df['icd10h'].nunique()

X_cause_train, X_cause_test, X_age_train, X_age_test, X_sex_train, X_sex_test, y_train, y_test = train_test_split_tensors(X_cause, X_age, X_sex, y=y_tensor, test_size=0.1)

train_loader,test_loader = create_dataloaders(train=[X_cause_train,X_age_train,X_sex_train,y_train],test=[X_cause_test,X_age_test,X_sex_test,y_test],batch_size=batch_size)
print('Dataloaders prepared and ready.')

model = individualized_network(
    vocab_size_cause=len(vocab),
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    emb_dim_cause=128,
    emb_dim_sex=8,
    emb_dim_age=16
).to(device)
print('Network initialized.')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)