import torch
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix

def evaluate_model(model, dataloader, criterion, device, folder_location, file_name):
    """
    Evaluate the model on a given dataloader and save evaluation metrics along with the confusion matrix.
    
    The results are saved into a single .txt file with the format:
        {accuracy}\n
        {loss}\n
        {confusion_matrix}
    
    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): The device to perform computation on (e.g. 'cpu' or 'cuda').
        folder_location (str): The folder where the result file will be saved.
        file_name (str): The name of the file to save the evaluation results.
    
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
    
    # Save the metrics and confusion matrix to a single .txt file.
    output_path = os.path.join(folder_location, file_name)
    with open(output_path, 'w') as f:
        f.write(f"{accuracy}\n{avg_loss}\n{cm}")
    
    model.train()
    return avg_loss, accuracy