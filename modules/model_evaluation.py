import torch
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
from text_preprocessor import *
from networks import individualized_network
from training import train_model


import os
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix



def evaluate_model(model, dataloader, criterion, device, folder_location, file_name):
    """
    Evaluate the model on a given dataloader and save evaluation metrics and confusion matrix
    as two separate .csv files.

    The results are saved in:
        - 'metrics_{file_name}.csv' containing accuracy and loss.
        - 'cm_{file_name}.csv' containing the confusion matrix with labeled rows & columns.

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): The device to perform computation on (e.g. 'cpu' or 'cuda').
        folder_location (str): The folder where the result files will be saved.
        file_name (str): The base name for the result files.

    Returns:
        avg_loss (float): The average loss over the evaluation dataset.
        accuracy (float): The accuracy of the model on the evaluation dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for X_cause, X_age, X_sex, y in dataloader:
            X_cause, X_age, X_sex, y = X_cause.to(device), X_age.to(device), X_sex.to(device), y.to(device)
            outputs = model(X_cause, X_age, X_sex)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
            # Accumulate targets and predictions for confusion matrix
            all_targets.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Evaluation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Define file paths
    metrics_path = os.path.join(folder_location, f"metrics_{file_name}.csv")
    cm_path = os.path.join(folder_location, f"cm_{file_name}.csv")

    # Save metrics as a CSV file
    metrics_df = pd.DataFrame({"Metric": ["Accuracy", "Average Loss"], "Value": [accuracy, avg_loss]})
    metrics_df.to_csv(metrics_path, index=False)

    # Save confusion matrix with labeled rows & columns
    cm_df = pd.DataFrame(cm, 
                         index=["Actual Positive", "Actual Negative"], 
                         columns=["Predicted Positive", "Predicted Negative"])
    cm_df.to_csv(cm_path)

    model.train()
    return avg_loss, accuracy


def validate_models(val_df, fold_model_paths, scaler_age, le_sex, y_label_encoder, token_types, batch_size, num_classes, vocab, dropout_rate, criterion, model_folder, device):
        X_cause_val, _ = prepare_deathcauses_tensors(df=val_df, column='deathcause_mono', token_types=token_types, pretrained_vocab=vocab)
        X_age_val = torch.tensor(scaler_age.transform(val_df['age'].to_numpy().reshape(-1, 1)),
                            dtype=torch.float).to(device)
        X_sex_val = torch.tensor(le_sex.transform(val_df['sex']),
                            dtype=torch.long).to(device)

        y_val, _ = encode_labels(val_df, transform_column='icd10h', label_encoder=y_label_encoder)
        val_loader = create_dataloaders(
            train=[X_cause_val, X_age_val, X_sex_val, y_val],
            batch_size=batch_size
        )

        for i,fold_model_path in enumerate(fold_model_paths):
            print(f'Initiating evaluation of fold {i+1}.')
            model = individualized_network(
                vocab_size_cause=len(vocab),
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                emb_dim_cause=128,
                emb_dim_sex=8,
                emb_dim_age=16
            ).to(device)
            model.load_state_dict(torch.load(fold_model_path))
            loss,acc = evaluate_model(model, val_loader, criterion, device, model_folder, f'fold{i+1}.csv')
            print(f'Accuracy: {acc} | Loss: {loss}')
        