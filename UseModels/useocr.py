import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import unicodedata

# --- 1. CRITICAL: Define Model Architecture ---
# This class MUST be identical to the one used for training.
# I have copied it directly from your 'ocrmodel.py' file.
class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, rnn_layers=2):
        super(CRNN, self).__init__()
        resnet = models.resnet18(weights=None) # Set weights to None for inference if not needed
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        cnn_output_channels = 512
        cnn_output_height = 2
        self.map_to_seq = nn.Linear(cnn_output_channels * cnn_output_height, rnn_hidden_size)
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(rnn_hidden_size * 2, num_chars)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2)
        features = features.reshape(b, w, c * h)
        features = self.map_to_seq(features)
        rnn_output, _ = self.rnn(features)
        output = self.fc(rnn_output)
        output = output.permute(1, 0, 2)
        return output

# --- 2. CRITICAL: Define Vocabulary and Mappings ---
# This vocabulary MUST MATCH the one used during training.
# Please verify and add any other characters (numbers, symbols) if they were in your labels.
VOCAB = "-אבגדהוזחטיכךלמםנןסעפףצץקרשת" # Standard Hebrew alphabet. 

# Create the character-to-integer and integer-to-character mappings
# Note: We add 1 to the index to reserve 0 for the CTC blank token.
char_to_int = {char: i + 1 for i, char in enumerate(VOCAB)}
int_to_char = {i + 1: char for i, char in enumerate(VOCAB)}
vocab_size = len(VOCAB) + 1 # Add 1 for the CTC blank token

# --- 3. Define the Decoding Function ---

def decode_prediction(preds, int_to_char_map):
    """Greedy decoding of the model's predictions."""
    preds = preds.permute(1, 0, 2)
    preds = torch.argmax(preds, dim=2)

    decoded_preds = []
    for p in preds:
        p_collapsed = []
        last_char = None
        for char_tensor in p:
            char_val = char_tensor.item()
            if char_val != 0 and char_val != last_char: # Not a blank and not a repeat
                p_collapsed.append(char_val)
            last_char = char_val

        # --- CHANGE IS HERE ---
        # Join the characters into a string and then reverse it for correct RTL display.
        decoded_string = "".join([int_to_char_map.get(c, '') for c in p_collapsed])
        decoded_preds.append(decoded_string[::-1]) # Reverses the string

    return decoded_preds[0] # Return the single, corrected prediction string

# --- 4. Main Prediction Function ---
def predict_from_image(model_path, image_path):
    """
    Loads the trained model and predicts the text from a single image file.
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((64, 256)), # Same as IMAGE_HEIGHT, IMAGE_WIDTH
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model with the correct vocabulary size
    model = CRNN(num_chars=vocab_size).to(device)

    # Load the saved weights (the state_dict)
    # This is the correct way to load a file saved with torch.save(model.state_dict(), ...)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    # Load and process the image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"

    image = transform(image)
    # Add a batch dimension (the model expects a batch of images)
    image = image.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        preds = model(image)

    # Decode the result
    predicted_text = decode_prediction(preds, int_to_char)
    
    return predicted_text

# --- 5. Run the Script ---
if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE PATHS ---
    CHECKPOINT_PATH = r'C:\Users\julia\OneDrive - Technion\Desktop\semestre7\deeplearning\ocrpython3.10\saved_models\crnn_best_epoch_13.pth'
    IMAGE_TO_PREDICT = r'C:\Users\julia\OneDrive - Technion\Desktop\semestre7\deeplearning\ocrpython3.10\messi.jpg' # <-- CHANGE THIS to your JPG or PNG image

    if IMAGE_TO_PREDICT == r'path/to/your/image.jpg':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE the IMAGE_TO_PREDICT path !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # Get the prediction
        result = predict_from_image(CHECKPOINT_PATH, IMAGE_TO_PREDICT)
        
        print("\n--- OCR Prediction ---")
        print(f"Image Path: {IMAGE_TO_PREDICT}")
        print(f"Predicted Word: {result}")
        print("----------------------")
