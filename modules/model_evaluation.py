import torch
import torch.nn

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataloader.

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): The device to perform computation on (e.g. 'cpu' or 'cuda').

    Returns:
        avg_loss (float): The average loss over the evaluation dataset.
        accuracy (float): The accuracy of the model on the evaluation dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_cause, X_age, X_sex, y in dataloader:
            X_cause, X_age, X_sex, y = X_cause.to(device), X_age.to(device), X_sex.to(device), y.to(device)
            outputs = model(X_cause, X_age, X_sex)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Evaluation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    model.train()
    return avg_loss, accuracy