from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import StratifiedKFold

from modules.text_preprocessor import *

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs: int = 16, device: str = "cpu", save_path: str | None = None):
    """
    Trains `model` for the specified number of epochs, evaluates on `test_loader`
    after every epoch, prints progress/loss/accuracy, and (optionally) saves the
    trained weights.

    Args
    ----
    model (nn.Module)              : The neural-network model to train.
    train_loader (DataLoader)      : DataLoader yielding the training batches.
    test_loader (DataLoader)       : DataLoader yielding the validation/test batches.
    criterion (nn.Module)          : Loss function.
    optimizer (torch.optim.Optimizer): Optimizer.
    num_epochs (int), default 16   : Training epochs.
    device (str), default "cpu"    : "cuda" or "cpu".
    save_path (str | None)         : If given, `model.state_dict()` is written here
                                     when training completes.

    Returns
    -------
    None
    """
    # ─────────────────── setup ───────────────────
    model.to(device)
    print("Training has begun.")

    # ─────────────────── training ───────────────────
    for epoch in tqdm(range(num_epochs), desc="Total Epoch Progress", unit="epoch"):
        model.train()
        total_loss = correct_train = total_train = 0

        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
        )

        for *inputs, labels in train_iterator:
            labels = labels.to(device)
            outputs = model(inputs[0].to(device)) if len(inputs) == 1 else model(
                *[x.to(device) for x in inputs]
            )
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)
            train_iterator.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # ─────────────────── validation ───────────────────
        model.eval()
        total_val_loss = correct_val = total_val = 0
        with torch.no_grad():
            for *inputs, labels in test_loader:
                labels = labels.to(device)
                outputs = model(inputs[0].to(device)) if len(inputs) == 1 else model(
                    *[x.to(device) for x in inputs]
                )
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(test_loader)
        val_acc = correct_val / total_val

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}]: "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Test Loss: {avg_val_loss:.4f}, Test Acc: {val_acc*100:.2f}%"
        )

    # ─────────────────── save model ───────────────────
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nModel weights saved to {save_path}")

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

        # Display summary for each epoch
        print(f"\nEpoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Test Loss: {avg_val_loss:.4f}, Test Acc: {val_acc*100:.2f}%")
        


def train_k_folds(i,X_cause,X_age,X_sex,y_tensor,model_names,model_folder,vocab,num_classes,dropout_rate,learning_rate,batch_size,num_epochs,network_architecture,k_folds,criterion,device,verbose=True):
    fold_model_paths = [] 
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, fold_test_idx) in enumerate(skf.split(X_cause, y_tensor.cpu().numpy())):
        print(f'Fold {fold+1} for {model_names[i]} initiated.')
        
        X_cause_train = X_cause[train_idx]
        X_age_train = X_age[train_idx]
        X_sex_train = X_sex[train_idx]
        y_train = y_tensor[train_idx]
        
        X_cause_fold_test = X_cause[fold_test_idx]
        X_age_fold_test = X_age[fold_test_idx]
        X_sex_fold_test = X_sex[fold_test_idx]
        y_fold_test = y_tensor[fold_test_idx]
        
        train_loader, fold_test_loader = create_dataloaders(
            train=[X_cause_train, X_age_train, X_sex_train, y_train],
            test=[X_cause_fold_test, X_age_fold_test, X_sex_fold_test, y_fold_test],
            batch_size=batch_size
        )

        model = network_architecture(
            vocab_size_cause=len(vocab),
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            emb_dim_cause=128,
            emb_dim_sex=8,
            emb_dim_age=16
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f'Training of {model_names[i]} in Fold {fold+1} has begun.')
        train_model(model, train_loader, fold_test_loader, criterion, optimizer, num_epochs, device=device)
        
        fold_model_path = os.path.join(model_folder, f"{model_names[i]}_fold{fold+1}.pth")
        torch.save(model.state_dict(), fold_model_path)
        fold_model_paths.append(fold_model_path)

    if verbose: print(f'Training for all {k_folds} folds completed.')
    
    return fold_model_paths