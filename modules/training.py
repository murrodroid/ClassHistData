import torch

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=16, device='cpu'):
    """
    Trains the model while dynamically adapting to the input format.

    Each batch from the DataLoader is assumed to be a tuple or list where:
      - The last element is the labels.
      - All preceding elements are input tensors.
      
    This function will call the model accordingly:
      - For one input: model(input)
      - For multiple inputs: model(*inputs)

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the validation/test set.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        None
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            # Unpack the batch: assume the last element is the labels.
            if not isinstance(batch, (list, tuple)):
                raise ValueError("Each batch must be a tuple or list.")
            if len(batch) < 2:
                raise ValueError("Each batch must contain at least one input and one label.")
            
            # Unpack: all elements except the last are inputs; the last is labels.
            *inputs, labels = batch
            
            # Move labels to device.
            labels = labels.to(device)
            # Move inputs to device: if there's one input, it remains a tensor;
            # if multiple, each tensor is moved to the device.
            if len(inputs) == 1:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
            else:
                inputs = [inp.to(device) for inp in inputs]
                outputs = model(*inputs)
            
            # Compute loss and update.
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation phase.
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in test_loader:
                if not isinstance(batch, (list, tuple)):
                    raise ValueError("Each batch must be a tuple or list.")
                if len(batch) < 2:
                    raise ValueError("Each batch must contain at least one input and one label.")
                
                *inputs, labels = batch
                labels = labels.to(device)
                if len(inputs) == 1:
                    inputs = inputs[0].to(device)
                    outputs = model(inputs)
                else:
                    inputs = [inp.to(device) for inp in inputs]
                    outputs = model(*inputs)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(test_loader)
        val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")