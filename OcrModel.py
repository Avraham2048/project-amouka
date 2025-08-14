import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm
import unicodedata

# --- Configuration ---
# You can adjust these hyperparameters
IMAGE_DIR = r'C:\Users\julia\OneDrive - Technion\Desktop\semestre7\deeplearning\ocrpython3.10\output1' # IMPORTANT: Update this path
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
IMAGE_HEIGHT = 64 # Resize all images to this height
IMAGE_WIDTH = 256 # Resize all images to this width
NUM_WORKERS = 4 # For DataLoader
TRAIN_VAL_SPLIT = 0.9 # 90% for training, 10% for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Loading and Preprocessing ---

def parse_filename(filename):
    """
    Extracts the Hebrew word from a filename like '1466_להטהר.jpg'.
    It removes the initial number, underscore, and file extension.
    """
    match = re.search(r'_(.+)\.jpg$', filename)
    if match:
        # Normalize to handle different Unicode representations if necessary
        return unicodedata.normalize('NFC', match.group(1))
    # Handle filenames that might not have a number prefix
    match_no_prefix = re.search(r'(.+)\.jpg$', filename)
    if match_no_prefix:
        # Check if it contains non-Hebrew characters to avoid matching '1466_word' incorrectly
        if not re.search(r'[0-9_]', match_no_prefix.group(1)):
             return unicodedata.normalize('NFC', match_no_prefix.group(1))
    return None

class OcrDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the OCR image data.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print(f"Loading images from: {self.image_dir}")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"The directory '{self.image_dir}' was not found. Please update the IMAGE_DIR variable.")

        # Load image paths and parse labels
        for filename in os.listdir(image_dir):
            if filename.lower().endswith('.jpg'):
                label = parse_filename(filename)
                if label:
                    self.image_paths.append(os.path.join(image_dir, filename))
                    self.labels.append(label)

        if not self.image_paths:
            raise ValueError(f"No valid image files found in '{self.image_dir}'. Check your files and the parsing logic.")

        print(f"Found {len(self.image_paths)} images.")

        # Create character vocabulary
        self.all_chars = sorted(list(set("".join(self.labels))))
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.all_chars)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.all_chars)}
        
        # Add a blank token for CTC loss (index 0)
        self.vocab_size = len(self.all_chars) + 1
        print(f"Vocabulary Size: {self.vocab_size} (including blank token)")
        print(f"Characters: {''.join(self.all_chars)}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]

        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # Return a dummy sample which will be filtered by collate_fn
            return None, None

        if self.transform:
            image = self.transform(image)
        
        # Encode label into integers
        label_encoded = [self.char_to_int[char] for char in label_str]
        
        return image, torch.tensor(label_encoded, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function to handle batches of images and labels of varying lengths.
    It filters out None values that might result from loading errors.
    """
    # Filter out failed samples
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    images, labels = zip(*batch)
    
    # All images are resized to the same size by the transform, so we can stack them
    images = torch.stack(images, 0)
    
    # The ResNet backbone downsamples the width by a factor of 32.
    # For an input width of 256, the feature map width will be 256 / 32 = 8.
    # This is the sequence length for the RNN and must be passed to the CTC loss.
    downsample_factor = 32
    seq_len = images.size(3) // downsample_factor
    input_lengths = torch.full(size=(len(images),), fill_value=seq_len, dtype=torch.long)
    
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    # Pad labels
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    return images, labels_padded, input_lengths, target_lengths


# --- 2. Model Architecture (CRNN) ---

class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, rnn_layers=2):
        super(CRNN, self).__init__()

        # --- CNN Backbone (Feature Extractor) ---
        # Using a pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # We take all layers except the last two (avgpool and fc)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # The output of ResNet18 will have 512 channels.
        # The height will be downsampled. For an input height of 64, it becomes 2.
        cnn_output_channels = 512
        cnn_output_height = 2 # Calculated based on ResNet architecture for 64px input height
        
        # --- Map-to-Sequence ---
        # The CNN output is (batch, channels, height, width). We want (batch, width, features) for the RNN.
        self.map_to_seq = nn.Linear(cnn_output_channels * cnn_output_height, rnn_hidden_size)

        # --- RNN (Sequence Modeling) ---
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        # --- Transcription Layer ---
        self.fc = nn.Linear(rnn_hidden_size * 2, num_chars) # *2 for bidirectional

    def forward(self, x):
        # CNN forward pass
        features = self.cnn(x) # -> (batch, 512, H/32, W/32) -> (batch, 512, 2, 8 for 64x256 input)
        
        # Reshape for RNN
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2) # -> (batch, width, channels, height)
        features = features.reshape(b, w, c * h) # -> (batch, width, channels * height)
        
        # Map-to-Sequence
        features = self.map_to_seq(features) # -> (batch, width, rnn_hidden_size)

        # RNN forward pass
        rnn_output, _ = self.rnn(features) # -> (batch, seq_len, rnn_hidden_size * 2)
        
        # Transcription
        output = self.fc(rnn_output) # -> (batch, seq_len, num_chars)
        
        # For CTC loss, we need (seq_len, batch, num_chars)
        output = output.permute(1, 0, 2)
        
        return output

# --- 3. Training and Evaluation ---

def decode_predictions(preds, int_to_char_map):
    """Greedy decoding of the model's predictions."""
    preds = preds.permute(1, 0, 2) # -> (batch, seq_len, num_chars)
    preds = torch.argmax(preds, dim=2)
    
    decoded_preds = []
    for p in preds:
        # --- FIX WAS APPLIED HERE ---
        # Manual implementation to collapse repeated characters, replacing torch.groupby
        p_collapsed = []
        last_char = None
        for char_tensor in p:
            char_val = char_tensor.item()
            # Add character if it's not a blank (0) and not the same as the last character
            if char_val != 0 and char_val != last_char:
                p_collapsed.append(char_val)
            last_char = char_val
        
        decoded_preds.append("".join([int_to_char_map.get(c, '') for c in p_collapsed]))
        
    return decoded_preds

def calculate_accuracy(preds_decoded, labels_str):
    """Calculates exact match accuracy."""
    correct = 0
    for p, l in zip(preds_decoded, labels_str):
        if p == l:
            correct += 1
    return correct / len(labels_str) if len(labels_str) > 0 else 0


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, input_lengths, target_lengths in tqdm(dataloader, desc="Training"):
        # Skip empty batches that might result from the collate_fn filtering
        if images.nelement() == 0:
            continue
            
        images, labels = images.to(device), labels.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
        
        optimizer.zero_grad()
        
        preds = model(images)
        log_probs = nn.functional.log_softmax(preds, dim=2)
        
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        # Handle potential inf or nan loss
        if torch.isinf(loss) or torch.isnan(loss):
            print("Warning: Skipping batch due to inf/nan loss.")
            continue

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def evaluate(model, dataloader, criterion, dataset, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, input_lengths, target_lengths in tqdm(dataloader, desc="Validating"):
            # Skip empty batches
            if images.nelement() == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

            preds = model(images)
            log_probs = nn.functional.log_softmax(preds, dim=2)
            
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode for accuracy calculation
            preds_decoded = decode_predictions(preds, dataset.int_to_char)
            # To decode labels, we must ignore padding (0)
            labels_str = []
            for l in labels:
                filtered_l = l[l != 0] # Filter out padding
                labels_str.append("".join([dataset.int_to_char.get(c.item(), '') for c in filtered_l]))


            all_preds.extend(preds_decoded)
            all_labels.extend(labels_str)

    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    val_acc = calculate_accuracy(all_preds, all_labels)
    
    # Print a few examples
    print("\n--- Validation Examples ---")
    for i in range(min(5, len(all_preds))):
        print(f"Label: {all_labels[i]:<20} | Predicted: {all_preds[i]}")
    print("-------------------------\n")

    return val_loss, val_acc


# --- 4. Main Execution ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    try:
        full_dataset = OcrDataset(image_dir=IMAGE_DIR, transform=transform)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # Split dataset
    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Initialize model, loss, and optimizer
    model = CRNN(num_chars=full_dataset.vocab_size).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create folder for saved models
    os.makedirs("saved_models", exist_ok=True)

    best_val_loss = float("inf")  # initial best loss

    # Training loop
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, full_dataset, DEVICE)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"saved_models/crnn_best_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved at {save_path}")

    print("\n--- Training Finished ---")