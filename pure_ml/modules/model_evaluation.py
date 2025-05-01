import torch
import os
import os
import torch
import pandas as pd
from sklearn.metrics import classification_report

from modules.text_preprocessor import *
from modules.networks import individualized_network


def evaluate_model(model, dataloader, criterion, device, folder_location, file_name):
    """
    Evaluate the model on a given dataloader and save evaluation metrics and classification report
    as two separate .csv files.

    The results are saved in:
        - 'metrics_{file_name}.csv' containing accuracy, loss, precision, recall, and f1-score.
        - 'cr_{file_name}.csv' containing the classification report for each class.

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

            all_targets.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Evaluation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    # Use only the labels that appear in targets or predictions
    available_labels = sorted(set(all_targets) | set(all_preds))
    # Get the classification report as a dictionary
    cr_dict = classification_report(
        all_targets, all_preds, 
        labels=available_labels, 
        zero_division=0, 
        output_dict=True
    )
    
    # Extract overall metrics (using weighted average for precision/recall/f1)
    overall_accuracy = cr_dict['accuracy']
    precision = cr_dict['weighted avg']['precision']
    recall = cr_dict['weighted avg']['recall']
    f1_score = cr_dict['weighted avg']['f1-score']
    
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    
    # Save overall metrics to CSV
    metrics_path = os.path.join(folder_location, f"metrics_{file_name}.csv")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Average Loss", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, avg_loss, precision, recall, f1_score]
    })
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save the full classification report to a separate CSV file
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_path = os.path.join(folder_location, f"cr_{file_name}.csv")
    cr_df.to_csv(cr_path)
    
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
            model.load_state_dict(torch.load(fold_model_path,weights_only=True))
            loss,acc = evaluate_model(model, val_loader, criterion, device, model_folder, f'fold{i+1}')
            print(f'Accuracy: {acc} | Loss: {loss}')
        