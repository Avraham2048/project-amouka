import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

# --- Configuration ---
PARQUET_FILE_PATHS = ['dataocr/train2.parquet'] 

# Training parameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2 # 20% of data for validation
RANDOM_SEED = 42

# --- Step 1: Create a Custom PyTorch Dataset ---
class HebrewHandwrittenDataset(Dataset):
    """Custom Dataset for loading Hebrew handwritten characters from .parquet files."""
    def __init__(self, parquet_paths, transform=None):
        """
        Args:
            parquet_paths (list of string): A list of paths to the .parquet files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print("Loading data from multiple parquet files...")
        all_dfs = []
        for path in parquet_paths:
            try:
                print(f" -> Loading data from {path}...")
                df = pd.read_parquet(path)
                all_dfs.append(df)
                print(f"    ...loaded {len(df)} samples.")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not all_dfs:
            raise ValueError("No data could be loaded. Please check file paths.")

        print("Concatenating all dataframes...")
        self.dataframe = pd.concat(all_dfs, ignore_index=True)
        self.dataframe = self.dataframe.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        self.transform = transform
        print(f"Data loaded successfully. Total samples: {len(self.dataframe)}")
        
        self.num_classes = self.dataframe['label'].nunique()
        print(f"Found {self.num_classes} unique classes.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample_row = self.dataframe.iloc[idx]
        image_bytes = sample_row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        label = int(sample_row['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- Step 2: Define Image Transformations ---

# NEW: Augmentation pipeline for the training set
train_transform = transforms.Compose([
    # Add augmentations that mimic handwriting variations
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    
    # Standard transformations
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# NEW: A separate, simpler pipeline for the validation set (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --- Step 3: Define the Convolutional Neural Network (CNN) ---
class HebrewOCRNet(nn.Module):
    """A simple CNN for handwritten character recognition."""
    def __init__(self, num_classes=27):
        super(HebrewOCRNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 4: Load Data and Create DataLoaders ---
    
    # MODIFIED: Load the dataset WITHOUT any transforms initially
    full_dataset = HebrewHandwrittenDataset(parquet_paths=PARQUET_FILE_PATHS, transform=None)
    
    dataset_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset first
    train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    # NEW: Apply the respective transforms to the split datasets
    # To do this, we need to access the underlying dataset object from the Subset
    train_dataset_raw.dataset.transform = train_transform
    val_dataset_raw.dataset.transform = val_transform

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {len(train_dataset_raw)}")
    print(f"Validation set size: {len(val_dataset_raw)}")
    
    # The datasets are now ready with their respective transforms
    train_loader = DataLoader(dataset=train_dataset_raw, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset_raw, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 5: Initialize Model, Loss, and Optimizer ---
    
    model = HebrewOCRNet(num_classes=full_dataset.num_classes).to(device)
    print("\nModel Architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Step 6: Training and Validation Loop ---
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        # Make sure to set the transform for the training part of the dataset
        train_dataset_raw.dataset.transform = train_transform
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for i, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress.set_postfix({'loss': running_loss / (i + 1)})

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # Make sure to set the transform for the validation part of the dataset
        val_dataset_raw.dataset.transform = val_transform
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_progress.set_postfix({'val_loss': val_loss / len(val_loader), 'val_acc': f"{100 * correct / total:.2f}%"})

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

    print("\nTraining finished.")
    
    # --- Step 7: Save the Trained Model ---
    model_path = "hebrew_ocr_model_augmented.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")