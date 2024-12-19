import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=16):
    """Trains the model and evaluates accuracy on both training and validation sets.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): Number of epochs to train.
    
    Returns:
        None
    """
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_output = model(X_val)
                val_loss += criterion(val_output, y_val).item()
                val_predictions = val_output.argmax(dim=1)
                val_correct_predictions += (val_predictions == y_val).sum().item()
                val_total_samples += y_val.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct_predictions / val_total_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}%")