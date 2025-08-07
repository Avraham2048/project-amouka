import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- Part 1: Copy Model Class Definitions ---
# You MUST copy the class definitions for your network and its blocks
# from your training script so PyTorch knows how to load the model.

class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers and a shortcut connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ComplexHebrewOCRNet(nn.Module):
    """A more complex CNN using Residual Blocks for character recognition."""
    def __init__(self, num_classes=27):
        super(ComplexHebrewOCRNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- Part 2: Create the Label Map ---
# ⚠️ IMPORTANT: This map must match the labels from your dataset.
# If your dataset labeled 'א' as 0, 'ב' as 1, etc., this will work.
# You may need to adjust this based on your specific 'train2.parquet' file.
HEBREW_LABEL_MAP = {
    0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
    9: 'י', 10: 'כ', 11: 'ך', 12: 'ל', 13: 'מ', 14: 'ם' , 15: 'נ', 16: 'ן' ,
    17: 'ס', 18: 'ע', 19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר',
    25: 'ש', 26: 'ת', 27: ','
}

# --- Part 3: Prediction Function ---
def predict_character(model_path, image_path, label_map):
    """Loads a trained model and predicts the character in an image."""
    
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transformations as the validation set
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Instantiate the model architecture
    num_classes = len(label_map)
    model = ComplexHebrewOCRNet(num_classes=num_classes)
    
    # Load the trained weights
    # Use map_location to ensure it works whether you are on CPU or GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Set the model to evaluation mode
    # This is crucial as it disables layers like Dropout
    model.eval()

    # Load and process the image
    try:
        # 'L' converts the image to grayscale, matching the training data
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return None, None

    # Apply transformations and add a batch dimension (B, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)
        # Get the index of the highest score
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
    
    predicted_char = label_map.get(predicted_idx, "Unknown")
    
    return predicted_char, predicted_idx

# --- Main Execution Block ---
if __name__ == '__main__':
    MODEL_PATH = "hebrew_ocr_model_v2.pth"
    IMAGE_TO_PREDICT = "ain.png" # The image you saved in Step 1

    predicted_char, predicted_index = predict_character(MODEL_PATH, IMAGE_TO_PREDICT, HEBREW_LABEL_MAP)

    if predicted_char:
        print(f"The model predicts the character is: '{predicted_char}' (Index: {predicted_index})")