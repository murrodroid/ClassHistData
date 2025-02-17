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

    # prepares training
    model_folder = f'trained_models/{model_names[i]}_fullTrain'
    os.makedirs(model_folder,exist_ok=True)
    
    X_cause, vocab = prepare_deathcauses_tensors(df=train_df, column='deathcause_mono', token_types=token_types)
    print(f'Deathcauses tensors are prepared with format: {token_types}')
    
    X_age = torch.tensor(StandardScaler().fit_transform(train_df['age'].to_numpy().reshape(-1, 1)), dtype=torch.float).to(device)
    X_sex = torch.tensor(LabelEncoder().fit_transform(train_df['sex']), dtype=torch.long).to(device)
    
    y_tensor, y_label_encoder = encode_labels(train_df, column='cd10h')
    num_classes = train_df['icd10h'].nunique()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cause, y_tensor.cpu().numpy())):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")
        # Create training and validation splits for this fold
        X_cause_train, X_cause_val = X_cause[train_idx], X_cause[val_idx]
        X_age_train, X_age_val = X_age[train_idx], X_age[val_idx]
        X_sex_train, X_sex_val = X_sex[train_idx], X_sex[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
        # Create dataloaders for training and validation
        train_loader, val_loader = create_dataloaders(
            train=[X_cause_train, X_age_train, X_sex_train, y_train],
            test=[X_cause_val, X_age_val, X_sex_val, y_val],
            batch_size=batch_size
        )
        print(f"Training and validation dataloaders prepared for fold {fold+1}.")
        
        # Initialize a new model for this fold
        model = individualized_network(
            vocab_size_cause=len(vocab),
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            emb_dim_cause=128,
            emb_dim_sex=8,
            emb_dim_age=16
        ).to(device)
        print(f"Network initialized for fold {fold+1}.")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model on the training split and validate on the validation split
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device=device)
        
        # Optionally evaluate on the validation set (using a function such as evaluate_model)
        # metrics = evaluate_model(model, val_loader, criterion, device=device)
        # fold_metrics.append(metrics)
        
        # Save the model for this fold
        fold_model_path = os.path.join(model_folder, f"{model_names[i]}_fold{fold+1}.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"{model_names[i]} for fold {fold+1} saved to {fold_model_path}.")
    
    # prepares testing
    if i == 0:
        test_df = test_random_df
    else:
        test_df = test_ordered_df
    
    X_cause_test, _ = prepare_deathcauses_tensors(df=test_df, column='deathcause_mono', token_types=token_types)
    X_age_test = torch.tensor(StandardScaler().fit_transform(
        test_df['age'].to_numpy().reshape(-1, 1)
    ), dtype=torch.float).to(device)
    X_sex_test = torch.tensor(LabelEncoder().fit_transform(
        test_df['sex']
    ), dtype=torch.long).to(device)
    # For testing, always use the column 'icd10h'
    y_test, _ = encode_labels(test_df, column='icd10h')
    
    # Create test dataloader
    _, test_loader = create_dataloaders(
        train=[X_cause, X_age, X_sex, y_tensor],
        test=[X_cause_test, X_age_test, X_sex_test, y_test],
        batch_size=batch_size
    )
    print(f'Test dataloader prepared for model: {model_names[i]}.')


    # init the model
    model = individualized_network(
        vocab_size_cause=len(vocab),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        emb_dim_cause=128,
        emb_dim_sex=8,
        emb_dim_age=16
    ).to(device)
    print(f'Network initialized for model: {model_names[i]}.')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model on the full training set and evaluate on the external test set
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(model_folder, f"{model_names[i]}_fullTrain.pth"))
    print(f"Model for {model_names[i]} saved successfully.\n")