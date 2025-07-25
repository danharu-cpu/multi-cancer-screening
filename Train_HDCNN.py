import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from models import HD_CNN
from dataloader import CustomDataset
from tqdm import tqdm

datapath = './data/HD_CNN_data'
save_dir = './checkpoints/HDCNN'
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters
epochs = 500
batch_size = 64
lr = 0.0002
weight_decay = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert coarse and fine labels to final label (0: Healthy, 1: Lung, 2: Gastric)
def to_final_label(coarse_label, fine_label):
    final_label = coarse_label.clone()
    for i in range(len(coarse_label)):
        if coarse_label[i] == 0:
            final_label[i] = 0  # Healthy
        else:
            final_label[i] = fine_label[i] + 1  # Lung = 1, Gastric = 2
    return final_label

# Load training and validation datasets
train_dataset = CustomDataset(
    os.path.join(datapath, 'data_training.csv'),
    os.path.join(datapath, 'data_training_label_HDCNN.csv'),
    mode='hdcnn'
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

valid_dataset = CustomDataset(
    os.path.join(datapath, 'data_valid.csv'),
    os.path.join(datapath, 'data_valid_label_HDCNN.csv'),
    mode='hdcnn'
)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, optimizer, loss function, and learning rate scheduler
model = HD_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, coarse_label, fine_label in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        data = data.to(device)
        coarse_label = coarse_label.to(device)
        fine_label = fine_label.to(device)

        final_label = to_final_label(coarse_label, fine_label).to(device)

        optimizer.zero_grad()
        final_logits, _, _ = model(data)
        loss = criterion(final_logits, final_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = final_logits.argmax(dim=1)
        correct += (pred == final_label).sum().item()
        total += final_label.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch} [Train]: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, coarse_label, fine_label in tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]"):
            data = data.to(device)
            coarse_label = coarse_label.to(device)
            fine_label = fine_label.to(device)

            final_label = to_final_label(coarse_label, fine_label).to(device)

            final_logits, _, _ = model(data)
            loss = criterion(final_logits, final_label)

            val_loss += loss.item()
            pred = final_logits.argmax(dim=1)
            val_correct += (pred == final_label).sum().item()
            val_total += final_label.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch} [Valid]: Loss = {val_loss:.4f}, Accuracy = {val_acc:.2f}%")

    # Step the learning rate scheduler
    scheduler.step()

    # Save model and prediction probabilities every 50 epochs
    if epoch % 50 == 0:
        all_probs = []
        all_true = []
        all_preds = []

        model.eval()
        with torch.no_grad():
            for data, coarse_label, fine_label in valid_loader:
                data = data.to(device)
                final_label = to_final_label(coarse_label, fine_label).to(device)

                final_logits, _, _ = model(data)
                probs = torch.softmax(final_logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_true.append(final_label.cpu().numpy())
                all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        df = pd.DataFrame(all_probs, columns=['P_Healthy', 'P_Lung', 'P_Gastric'])
        df['TrueLabel'] = all_true
        df['PredLabel'] = all_preds

        df.to_csv(os.path.join(save_dir, f'val_probs_epoch{epoch}.csv'), index=False)

        # ✅ Save model weights and optimizer state
        checkpoint_path = os.path.join(save_dir, f'hdcnn_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss / len(valid_loader),
            'val_accuracy': val_acc
        }, checkpoint_path)
        print(f"✅ Saved model checkpoint: {checkpoint_path}")

