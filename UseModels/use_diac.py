# use_model.py
import torch
import torch.nn as nn
import argparse
import os

# --- Step 1: Copy necessary definitions from the training script ---
# It's best practice to make the inference script self-contained.
# We copy the model class and the inference function here so this script
# doesn't need to import anything from the original training file.

class BiLSTMPunctuator(nn.Module):
    """
    The exact same model class definition used during training.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, 
                              num_layers=num_layers,
                              bidirectional=True, 
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size) # *2 for bidirectional

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.bilstm(emb)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

def punctuate(model, text, char2idx, idx2tag, device):
    """
    The exact same inference function used during training.
    """
    model.eval()
    
    # Preprocess the input text
    text_chars = list(text)
    # Use <UNK> for any characters not in the training vocabulary
    x = [char2idx.get(c, char2idx['<UNK>']) for c in text_chars]
    
    # Convert to tensor and add batch dimension
    x_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        # Get the most likely tag index for each character
        predictions = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    # Reconstruct the punctuated text
    result = ""
    for char, tag_idx in zip(text_chars, predictions):
        tag = idx2tag.get(tag_idx, '')
        if tag == 'O' or tag == '<PAD>': # Check for 'O' (no diacritic) or padding
            result += char
        else:
            result += char + tag
            
    return result

# --- Step 2: Main script logic ---

def main():
    parser = argparse.ArgumentParser(description="Use a trained Hebrew Punctuation model.")
    parser.add_argument("model_path", help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("text", help="The unpunctuated Hebrew text to process.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Use CPU for inference as it's often sufficient for single sentences
    device = torch.device('cpu')

    # 1. Load the checkpoint
    # We use weights_only=False because we need to load the saved args and vocab mappings.
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model file: {e}")
        print("Please ensure you are using a model file saved from the improved training script.")
        return

    # 2. Extract everything needed from the checkpoint
    model_args = checkpoint['args']
    char2idx = checkpoint['char2idx']
    tag2idx = checkpoint['tag2idx']
    model_state = checkpoint['model_state']
    
    # Create an inverted mapping for tags to convert indices back to diacritics
    idx2tag = {i: t for t, i in tag2idx.items()}

    # 3. Re-create the model with the *exact* same architecture as when it was trained
    # This is more robust than hardcoding dimensions.
    model = BiLSTMPunctuator(
        vocab_size=len(char2idx),
        embedding_dim=model_args.embed_dim,
        hidden_dim=model_args.hidden_dim,
        tagset_size=len(tag2idx),
        num_layers=model_args.num_layers,
        dropout=model_args.dropout
    ).to(device)

    # 4. Load the learned weights
    model.load_state_dict(model_state)
    model.eval() # Set the model to evaluation mode

    # 5. Punctuate the user's text
    punctuated_text = punctuate(model, args.text, char2idx, idx2tag, device)

    # 6. Print the results
    print("Input Text:    ", args.text)
    print("Punctuated Text:", punctuated_text)


if __name__ == '__main__':
    main()
