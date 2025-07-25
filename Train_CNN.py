import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from models import CNN
from dataloader import CustomDataset
from tqdm import tqdm

# Set paths
datapath = './data/CNN_data'
save_dir = './checkpoints/CNN'
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters
epochs = 500
batch_size = 64
lr = 0.0002
weight_decay = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training and validation datasets
train_dataset = CustomDataset(
    os.path.join(datapath, 'data_training_HD.csv'),
    os.path.join(datapath, 'data_training_label_CNN.csv'),
    mode='cnn'
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

valid_dataset = CustomDataset(
    os.path.join(datapath, 'data_valid_HD.csv'),
    os.path.join(datapath, 'data_valid_label_CNN.csv'),
    mode='cnn'
)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, optimizer, loss function, and scheduler
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, final_label in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        data = data.to(device)
        final_label = final_label.to(device)  # label: 0, 1, 2

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, final_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == final_label).sum().item()
        total += final_label.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch} [Train]: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_probs = []
    all_true = []
    all_preds = []

    with torch.no_grad():
        for data, final_label in tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]"):
            data = data.to(device)
            final_label = final_label.to(device)

            logits = model(data)
            loss = criterion(logits, final_label)

            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            val_loss += loss.item()
            val_correct += (pred == final_label).sum().item()
            val_total += final_label.size(0)

            all_probs.append(probs.cpu().numpy())
            all_true.append(final_label.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch} [Valid]: Loss = {val_loss:.4f}, Accuracy = {val_acc:.2f}%")

    # Step the learning rate scheduler
    scheduler.step()

    # Save prediction probabilities and model weights every 50 epochs
    if epoch % 50 == 0:
        all_probs = np.concatenate(all_probs, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        df = pd.DataFrame(all_probs, columns=['P_Healthy', 'P_Lung', 'P_Gastric'])
        df['TrueLabel'] = all_true
        df['PredLabel'] = all_preds
        df.to_csv(os.path.join(save_dir, f'val_probs_epoch{epoch}.csv'), index=False)

        # ✅ 모델 가중치 저장 추가
        checkpoint_path = os.path.join(save_dir, f'cnn_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss / len(valid_loader),
            'val_accuracy': val_acc
        }, checkpoint_path)
        print(f"✅ Saved CNN model checkpoint: {checkpoint_path}")
