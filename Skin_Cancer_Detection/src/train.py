import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np

from model_def import HybridSkinModel
from dataset import HAM10000
from utils import get_data_splits

def train_model():
  
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    METADATA_PATH = os.path.join(BASE_DIR, "Data", "HAM10000_metadata.csv")
    IMG_DIR = os.path.join(BASE_DIR, "Data", "all_images")
    SAVE_PATH = os.path.join(BASE_DIR, "model", "hybrid_model.pth")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
  
    train_df, val_df = get_data_splits(METADATA_PATH)
    train_ds = HAM10000(train_df, IMG_DIR, transform=train_transform)
    val_ds = HAM10000(val_df, IMG_DIR, transform=val_transform)
    
    y_train = [train_ds.label_map[x] for x in train_df['dx']]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in y_train])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
  
    model = HybridSkinModel(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
   
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    
    EPOCHS = 15
    best_val_acc = 0.0

    

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

       
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"📈 Validation Accuracy: {val_acc:.2f}%")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ New Best Model Saved!")

if __name__ == "__main__":
    train_model()