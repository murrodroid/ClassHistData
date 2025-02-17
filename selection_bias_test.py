# packages
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import os
# modules
from modules.import_data import import_data_random
from modules.text_preprocessor import *
from modules.networks import individualized_network
from modules.training import train_model
from modules.tools import return_device, undersampling


#### hyper-parameters
learning_rate = 0.0008
batch_size = 64
num_epochs = 16
dropout_rate = 0.34
retain_pct = 0.4
k_folds = 5
####

device = return_device()

df, _ = import_data_random(retain_pct)
print('Data imported successfully.')

model_names = ['random_df','ordered_df']
random_df, test_random_df = df[df.icd10h_random.notna()], df[df.icd10h_random.isna()]
ordered_df, test_ordered_df = df[df.icd10h_ordered.notna()], df[df.icd10h_ordered.isna()]


token_types = [
    {'method': 'char', 'ngram': 2},
    {'method': 'char', 'ngram': 3},
    {'method': 'word', 'ngram': 0}
]

for i, train_df in enumerate([random_df, ordered_df]):
    model_folder = f'trained_models/{model_names[i]}_{k_folds}Folds'
    os.makedirs(model_folder, exist_ok=True)
    
    X_cause, vocab = prepare_deathcauses_tensors(df=train_df, column='deathcause_mono', token_types=token_types)
    X_age = torch.tensor(StandardScaler().fit_transform(train_df['age'].to_numpy().reshape(-1, 1)), dtype=torch.float).to(device)
    X_sex = torch.tensor(LabelEncoder().fit_transform(train_df['sex']), dtype=torch.long).to(device)
    
    y_column = 'icd10h_random' if i == 0 else 'icd10h_ordered'
    y_tensor, y_label_encoder = encode_labels(train_df, column=y_column)
    num_classes = train_df['icd10h'].nunique()
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_model_paths = []
    
    for fold, (train_idx, _) in enumerate(skf.split(X_cause, y_tensor.cpu().numpy())):
        X_cause_train = X_cause[train_idx]
        X_age_train = X_age[train_idx]
        X_sex_train = X_sex[train_idx]
        y_train = y_tensor[train_idx]
        
        train_loader, _ = create_dataloaders(train=[X_cause_train, X_age_train, X_sex_train, y_train], batch_size=batch_size)
        model = individualized_network(vocab_size_cause=len(vocab), num_classes=num_classes, dropout_rate=dropout_rate,
                                       emb_dim_cause=128, emb_dim_sex=8, emb_dim_age=16).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_model(model, train_loader, None, criterion, optimizer, num_epochs, device=device)
        
        fold_model_path = os.path.join(model_folder, f"{model_names[i]}_fold{fold+1}.pth")
        torch.save(model.state_dict(), fold_model_path)
        fold_model_paths.append(fold_model_path)
    
    test_df = test_random_df if i == 0 else test_ordered_df
    X_cause_test, _ = prepare_deathcauses_tensors(df=test_df, column='deathcause_mono', token_types=token_types)
    X_age_test = torch.tensor(StandardScaler().fit_transform(test_df['age'].to_numpy().reshape(-1, 1)),
                              dtype=torch.float).to(device)
    X_sex_test = torch.tensor(LabelEncoder().fit_transform(test_df['sex']), dtype=torch.long).to(device)
    y_test, _ = encode_labels(test_df, column='icd10h')
    _, test_loader = create_dataloaders(train=[X_cause_test, X_age_test, X_sex_test, y_test], batch_size=batch_size)
    
    for fold_model_path in fold_model_paths:
        model = individualized_network(vocab_size_cause=len(vocab), num_classes=num_classes, dropout_rate=dropout_rate,
                                       emb_dim_cause=128, emb_dim_sex=8, emb_dim_age=16).to(device)
        model.load_state_dict(torch.load(fold_model_path))
        evaluate_model(model, test_loader, criterion, device=device)