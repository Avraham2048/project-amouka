import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms # <-- CORRECT IMPORT
import pandas as pd
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# File path for your downloaded .parquet file
PARQUET_FILE_PATH = 'dataocr/train.parquet' 
# You might need to change this if your file has a different name

# Training parameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2 # 20% of data for validation
RANDOM_SEED = 42

# --- Step 1: Create a Custom PyTorch Dataset ---
# This class will load and preprocess the data from your .parquet file.

class HebrewHandwrittenDataset(Dataset):
    """Custom Dataset for loading Hebrew handwritten characters from a .parquet file."""
    def __init__(self, parquet_path, transform=None):
        """
        Args:
            parquet_path (string): Path to the .parquet file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the entire dataset from the parquet file into a pandas DataFrame
        print(f"Loading data from {parquet_path}...")
        self.dataframe = pd.read_parquet(parquet_path)
        self.transform = transform
        print("Data loaded successfully.")
        
        # The dataset has 27 unique characters (labels 0-26)
        self.num_classes = self.dataframe['label'].nunique()
        print(f"Found {self.num_classes} unique classes.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Args:
            idx (int): The index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the integer label.
        """
        # Get the row for the given index
        sample_row = self.dataframe.iloc[idx]
        
        # The image data is stored in a dictionary-like structure under the 'image' column.
        # We need to access the 'bytes' key to get the raw image data.
        image_bytes = sample_row['image']['bytes']
        
        # Convert the raw bytes into a PIL Image object.
        # The image is grayscale ('L' mode).
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Get the corresponding label
        label = int(sample_row['label'])
        
        # Apply transformations if any are provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- Step 2: Define Image Transformations ---
# We need to convert images to tensors and normalize them.

# Transformation pipeline
# 1. Resize() ensures all images are the same size (28x28), which is required by the network.
# 2. ToTensor() converts the PIL Image into a PyTorch Tensor and scales pixel values to [0.0, 1.0].
# 3. Normalize() adjusts the tensor values to have a mean of 0.5 and a standard deviation of 0.5.
transform = transforms.Compose([
    transforms.Resize((28, 28)), # <-- FIX: Resize all images to a standard 28x28 size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --- Step 3: Define the Convolutional Neural Network (CNN) ---

class HebrewOCRNet(nn.Module):
    """A simple CNN for handwritten character recognition."""
    def __init__(self, num_classes=27):
        super(HebrewOCRNet, self).__init__()
        # The input is a 1x28x28 grayscale image.
        
        # First convolutional block
        # Takes 1 input channel (grayscale), produces 32 output channels (feature maps)
        # Kernel size is 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Max pooling layer with a 2x2 window
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # Second convolutional block
        # Takes 32 input channels, produces 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        
        # After two pooling layers, a 28x28 image becomes 7x7.
        # The number of features is 64 (from conv2) * 7 * 7.
        # This is the input size for our fully connected layer.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Dropout layer to prevent overfitting (randomly zeros 50% of elements)
        self.dropout = nn.Dropout(0.5)
        
        # Final fully connected layer that outputs the scores for each class.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Defines the forward pass of the network."""
        # Apply first conv block: conv -> relu -> pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Apply second conv block: conv -> relu -> pool
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the feature maps into a vector
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply first fully connected layer with relu activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply final fully connected layer to get class scores
        x = self.fc2(x)
        
        return x

# --- Main Execution Block ---
if __name__ == '__main__':
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 4: Load Data and Create DataLoaders ---
    
    # Instantiate the dataset
    full_dataset = HebrewHandwrittenDataset(parquet_path=PARQUET_FILE_PATH, transform=transform)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 5: Initialize Model, Loss, and Optimizer ---
    
    model = HebrewOCRNet(num_classes=full_dataset.num_classes).to(device)
    print("\nModel Architecture:")
    print(model)
    
    # Loss function: CrossEntropyLoss is suitable for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam is a popular and effective choice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Step 6: Training and Validation Loop ---
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        # Use tqdm for a progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for i, (images, labels) in enumerate(train_progress):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Update progress bar
            train_progress.set_postfix({'loss': running_loss / (i + 1)})

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad(): # No need to calculate gradients for validation
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
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
    
    model_path = "hebrew_ocr_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
