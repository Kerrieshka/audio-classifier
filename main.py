import dataset
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Создание датасетов
train_dataset = CustomDataset('train.txt')
val_dataset = CustomDataset('val.txt')

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Создание модели и оптимизатора
model = CustomModel()
optimizer = torch.optim.Adam(model.parameters())

# Функция потерь
criterion = nn.CrossEntropyLoss()


# Обучение модели
def train():
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss


# Валидация модели
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


# Запуск обучения и валидации модели
if name == '__main__':
    for epoch in range(10):
        train_loss = train()
        val_loss, val_acc = validate()
        print(
            f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
