
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.permute(0, 2, 1, 3) # reshape for LSTM
        batch_size, timesteps, channels, h, w = x.size()
        x = x.view(batch_size * timesteps, channels, h * w)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, optimizer, criterion, X_train, Y_train):
    model.train()
    train_loss = 0
    correct = 0
    for i in range(len(X_train)):
        optimizer.zero_grad()
        output = model(X_train[i])
        loss = criterion(output.unsqueeze(0), Y_train[i].unsqueeze(0))
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(Y_train[i].view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= len(X_train)
    train_acc = 100. * correct / len(X_train)
    return train_loss, train_acc

def validate(X, Y):
    with torch.no_grad():
        model.eval()
        loss = criterion(model(X), Y)
        predictions = torch.argmax(model(X), dim=1)
        accuracy = torch.mean((predictions == Y).float())
    return loss.item(), accuracy.item() * 100, predictions