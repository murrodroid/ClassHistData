import torch
import torch.nn as nn

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Trains the model using the DataLoader."""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")