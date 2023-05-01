import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(model, X_train, Y_train, X_val, Y_val, optimizer, criterion, device, n_epochs=50, batch_size=128):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Define data loaders
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, device=device, dtype=torch.float32),
                                                torch.tensor(Y_train, device=device, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, device=device, dtype=torch.float32),
                                              torch.tensor(Y_val, device=device, dtype=torch.long))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch_x, batch_y in tqdm(train_loader):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{n_epochs}, '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, '
              f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc*100:.2f}%')

    return train_losses, train_accs, val_losses, val_accs