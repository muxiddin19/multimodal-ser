import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random

from training.CustomizedDataset import CustomizedDataset
from training.BERT_ECAPA import FlexibleMMSER

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility Functions
def calculate_accuracy(y_pred, y_true):
    wa = accuracy_score(y_true, y_pred)  # Weighted Accuracy
    ua = balanced_accuracy_score(y_true, y_pred)  # Unweighted Accuracy (balanced)
    return wa, ua

def train_step(model, dataloader, optim, loss_fn):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []
    
    for text_embed, audio_embed, label in dataloader:
        text_embed = text_embed.to(device)
        audio_embed = audio_embed.to(device)
        label = label.to(device)
        
        y_logits, y_softmax = model(text_embed, audio_embed)
        loss = loss_fn(y_logits, label)

        optim.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()

        train_loss += loss.item()
        all_preds.extend(y_softmax.argmax(dim=1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    wa, ua = calculate_accuracy(all_preds, all_labels)
    return train_loss / len(dataloader), wa, ua

def eval_step(model, dataloader, loss_fn):
    model.eval()
    eval_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for text_embed, audio_embed, label in dataloader:
            text_embed = text_embed.to(device)
            audio_embed = audio_embed.to(device)
            label = label.to(device)
            
            y_logits, y_softmax = model(text_embed, audio_embed)
            loss = loss_fn(y_logits, label)

            eval_loss += loss.item()
            all_preds.extend(y_softmax.argmax(dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    wa, ua = calculate_accuracy(all_preds, all_labels)
    return eval_loss / len(dataloader), wa, ua

def get_class_weights(dataset):
    """Calculate class weights for imbalanced data."""
    labels = [dataset[i][2].item() if torch.is_tensor(dataset[i][2]) else dataset[i][2] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))], dtype=torch.float32)
    return weights.to(device)

def train_and_evaluate(model, train_loader, val_loader, train_dataset, num_epochs, lr=1e-4, weight_decay=1e-4, save_path=None):
    
    # Class-weighted loss for imbalanced data
    class_weights = get_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    train_loss_hist, val_loss_hist = [], []
    train_wa_hist, val_wa_hist = [], []
    train_ua_hist, val_ua_hist = [], []

    best_val_ua = 0.0
    best_val_wa = 0.0
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_wa, train_ua = train_step(model, train_loader, optimizer, criterion)
        val_loss, val_wa, val_ua = eval_step(model, val_loader, criterion)

        scheduler.step()

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_wa_hist.append(train_wa * 100)
        val_wa_hist.append(val_wa * 100)
        train_ua_hist.append(train_ua * 100)
        val_ua_hist.append(val_ua * 100)

        # Save best model based on UA (more important for emotion recognition)
        if val_ua > best_val_ua:
            best_val_ua = val_ua
            best_val_wa = val_wa
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ua': val_ua,
                    'val_wa': val_wa,
                }, save_path)
                print(f"  -> New best model saved! UA: {val_ua:.4f}")
        else:
            patience_counter += 1

        print(f"Train Loss: {train_loss:.4f}, Train WA: {train_wa:.4f}, Train UA: {train_ua:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val WA: {val_wa:.4f}, Val UA: {val_ua:.4f}")
        print("-" * 50)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nBest Val WA: {best_val_wa:.4f}, Best Val UA: {best_val_ua:.4f}")
    return train_loss_hist, val_loss_hist, train_wa_hist, val_wa_hist, train_ua_hist, val_ua_hist

def plot_metrics(epochs, train_hist, val_hist, metric_name, save_name=None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_hist, label=f'Train {metric_name}')
    plt.plot(epochs, val_hist, label=f'Validation {metric_name}')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Training and Validation {metric_name} Over Epochs")
    plt.grid(True, alpha=0.3)
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()

# Paths
train_metadata = r"features/IEMOCAP_BERT_ECAPA_train.pkl"
val_metadata = r"features/IEMOCAP_BERT_ECAPA_val.pkl"

# Datasets and Dataloaders
BATCH_SIZE = 32  # Smaller batch size helps regularization
train_dataset = CustomizedDataset(train_metadata)
val_dataset = CustomizedDataset(val_metadata)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Model Training
model = FlexibleMMSER(num_classes=4, dropout=0.5).to(device)  # Add dropout parameter

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

save_path = r"saved_models/IEMOCAP_ENG_CMN_BERT_ECAPA.pt"

train_hist, val_hist, train_wa_hist, val_wa_hist, train_ua_hist, val_ua_hist = train_and_evaluate(
    model, 
    train_dataloader, 
    val_dataloader,
    train_dataset,
    num_epochs=200,
    lr=5e-5,  # Lower learning rate
    weight_decay=1e-3,  # Stronger regularization
    save_path=save_path
)

# Plot Metrics
epochs = list(range(1, len(train_hist) + 1))
plot_metrics(epochs, train_hist, val_hist, "Loss", "loss_plot.png")
plot_metrics(epochs, train_wa_hist, val_wa_hist, "Weighted Accuracy", "wa_plot.png")
plot_metrics(epochs, train_ua_hist, val_ua_hist, "Unweighted Accuracy", "ua_plot.png")