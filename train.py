import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from model import my_CNN
from custom_dataloader import MyImageDataset, my_transforms


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_weights.pth')

def main(lr=0.001, epochs=40, BATCH_SIZE=4, train_dir, test_dir, val_dir):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = MyImageDataset(root_dir=train_dir, transform=my_transforms)
    test_dataset = MyImageDataset(root_dir=test_dir, transform=my_transforms)
    val_dataset = MyImageDataset(root_dir=val_dir, transform=my_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, criterion, optimizer, scheduler
    model = my_CNN(in_channels=3, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    train_losses_his = []
    val_losses_his = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        train_losses_his.append(train_loss)
        val_losses_his.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs}: Learning rate = {current_lr:.6f}, Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}")

if __name__ == "__main__":
    # Directory paths
    train_dir = 'train/'
    test_dir = 'test/'
    val_dir = 'val/'

    LR = 0.001
    epochs = 40
    batchsize = 4
    
    main(LR, epochs, batchsize, train_dir, test_dir, val_dir)
