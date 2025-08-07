import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import io
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split # Used for robust data splitting

# --- Configuration ---
PARQUET_FILE_PATHS = ['dataocr/train2.parquet'] 
MODEL_SAVE_PATH = "hebrew_ocr_model_v2.pth"

# Training parameters
NUM_EPOCHS = 25 # Increased epochs slightly for the more complex model
BATCH_SIZE = 64 # Larger batch size can help stabilize training
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# --- Step 1: Create a Custom PyTorch Dataset ---
# MODIFIED: Now accepts a DataFrame directly to allow for pre-splitting.
class HebrewHandwrittenDataset(Dataset):
    """Custom Dataset for loading Hebrew handwritten characters."""
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the image data and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = self.dataframe['label'].nunique()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Fetches the sample at the given index."""
        sample_row = self.dataframe.iloc[idx]
        image_bytes = sample_row['image']['bytes']
        # Convert to 'L' for grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        label = int(sample_row['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- Step 2: Define Image Transformations (with Augmentation) ---
# NEW: We now have separate transforms for training (with augmentation) and validation (without).
# This helps the model generalize better.

# Augmentations for the training set
train_transform = transforms.Compose([
    transforms.Resize((32, 32)), # Slightly larger size for more detail
    # RandomAffine applies random rotations, translations, and shearing.
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize for grayscale
])

# No augmentations for the validation set, just resizing and normalization.
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --- Step 3: Define the Improved Convolutional Neural Network (CNN) ---
# NEW: This network is deeper and uses residual blocks and batch normalization for
#      better performance and more stable training.

class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers and a shortcut connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The "magic" of residual connections
        out = F.relu(out)
        return out

class ComplexHebrewOCRNet(nn.Module):
    """A more complex CNN using Residual Blocks for character recognition."""
    def __init__(self, num_classes=27):
        super(ComplexHebrewOCRNet, self).__init__()
        self.in_channels = 64
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        
        # Adaptive pooling is more flexible than a fixed flatten operation
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Initial: 32x32 -> 32x32
        out = self.layer1(out) # Stays 32x32
        out = self.layer2(out) # Downsamples to 16x16
        out = self.layer3(out) # Downsamples to 8x8
        out = self.layer4(out) # Downsamples to 4x4
        out = self.avg_pool(out) # Pools to 1x1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 4: Load and Split Data ---
    # NEW: Load all data first, then split the DataFrame before creating Datasets.
    # This allows us to assign different transforms to the train and validation sets.
    
    print("Loading and concatenating all dataframes...")
    all_dfs = [pd.read_parquet(path) for path in PARQUET_FILE_PATHS]
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total samples loaded: {len(full_df)}")
    
    # Extract labels for stratified splitting
    labels = full_df['label']
    
    # Split the DataFrame into training and validation sets
    train_df, val_df = train_test_split(
        full_df, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED,
        stratify=labels # Ensures train/val sets have similar class distributions
    )
    
    # Create Dataset objects with their respective transformations
    train_dataset = HebrewHandwrittenDataset(dataframe=train_df, transform=train_transform)
    val_dataset = HebrewHandwrittenDataset(dataframe=val_df, transform=val_transform)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Step 5: Initialize Model, Loss, and Optimizer ---
    num_classes = full_df['label'].nunique()
    model = ComplexHebrewOCRNet(num_classes=num_classes).to(device)
    print("\nModel Architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # NEW: Learning rate scheduler reduces LR when validation loss plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, verbose=True)

    # --- Step 6: Training and Validation Loop ---
    best_val_accuracy = 0.0
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
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
        
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🎉 New best model saved to {MODEL_SAVE_PATH} with accuracy: {val_accuracy:.2f}%")

    print("\nTraining finished.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")
    print(f"Best model saved to {MODEL_SAVE_PATH}")