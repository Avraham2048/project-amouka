# punctuator_with_accuracy.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import glob

# ---------------------------------------------
# 1. Define Hebrew Character Sets
# ---------------------------------------------
HEBREW_LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
DIACRITICS = "ְֱֲֳִֵֶַָׁׂ"
ALL_CHARS = HEBREW_LETTERS + DIACRITICS
TOKEN_RE = re.compile(f'([{HEBREW_LETTERS}])([{DIACRITICS}]*)')

# ---------------------------------------------
# 2. Improved Dataset and Preprocessing
# ---------------------------------------------

def load_and_align_data(no_diac_pattern, with_diac_pattern):
    no_diac_files = sorted(glob.glob(no_diac_pattern))
    with_diac_files = sorted(glob.glob(with_diac_pattern))
    all_input_sequences = []
    all_target_sequences = []
    print(f"Found {len(with_diac_files)} files with diacritics to process.")
    for with_file in with_diac_files:
        print(f"Processing: {os.path.basename(with_file)}")
        with open(with_file, 'r', encoding='utf-8') as f_with_diac:
            for line_with_diac in f_with_diac:
                line_with_diac = line_with_diac.strip()
                if not line_with_diac:
                    continue
                input_seq, target_seq = [], []
                tokens = TOKEN_RE.findall(line_with_diac)
                for letter, diac in tokens:
                    input_seq.append(letter)
                    target_seq.append(diac if diac else 'O')
                reconstructed_no_diac = "".join(input_seq)
                if reconstructed_no_diac and target_seq:
                    all_input_sequences.append(reconstructed_no_diac)
                    all_target_sequences.append(target_seq)
    print(f"Loaded {len(all_input_sequences)} aligned sequences.")
    return all_input_sequences, all_target_sequences

def build_mappings(input_texts, target_sequences):
    all_input_chars = set(c for text in input_texts for c in text)
    char2idx = {char: i + 2 for i, char in enumerate(sorted(list(all_input_chars)))}
    char2idx['<PAD>'] = 0
    char2idx['<UNK>'] = 1
    all_tags = set(tag for seq in target_sequences for tag in seq)
    tag2idx = {tag: i + 1 for i, tag in enumerate(sorted(list(all_tags)))}
    tag2idx['<PAD>'] = 0
    return char2idx, tag2idx

class HebrewPunctuationDataset(Dataset):
    def __init__(self, input_texts, target_sequences, char2idx, tag2idx, max_len):
        self.input_texts = input_texts
        self.target_sequences = target_sequences
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        text, tags = self.input_texts[idx], self.target_sequences[idx]
        x = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
        y = [self.tag2idx.get(t, 0) for t in tags]
        assert len(x) == len(y), f"Mismatch at index {idx}: len(x)={len(x)}, len(y)={len(y)}"
        x = x[:self.max_len] + [self.char2idx['<PAD>']] * (self.max_len - len(x))
        y = y[:self.max_len] + [self.tag2idx['<PAD>']] * (self.max_len - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------------------------------------------
# 3. Enhanced Model Definition
# ---------------------------------------------
class BiLSTMPunctuator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.bilstm(emb)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

# ---------------------------------------------
# 4. Training & Evaluation Loops with Accuracy
# ---------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, pad_idx):
    """
    Evaluates the model on a dataset, returning both loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)

            # Calculate loss (on non-padded tokens, handled by ignore_index)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            
            # Create a mask to exclude padding tokens from accuracy calculation
            non_pad_mask = (y_batch != pad_idx)
            
            # Count correct predictions on the actual, non-padded characters
            total_correct += ((predictions == y_batch) & non_pad_mask).sum().item()
            
            # Count total number of non-padded characters to get the correct average
            total_tokens += non_pad_mask.sum().item()

    avg_loss = total_loss / len(loader)
    # Ensure we don't divide by zero if the validation set is empty
    accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    return avg_loss, accuracy

# ---------------------------------------------
# 5. Inference Function
# ---------------------------------------------
def punctuate(model, text, char2idx, idx2tag, device):
    model.eval()
    text_chars = list(text)
    x = [char2idx.get(c, char2idx['<UNK>']) for c in text_chars]
    x_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        predictions = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    result = ""
    for char, tag_idx in zip(text_chars, predictions):
        tag = idx2tag.get(tag_idx, '')
        if tag == 'O' or tag == '<PAD>':
            result += char
        else:
            result += char + tag
    return result

# ---------------------------------------------
# 6. Main Function
# ---------------------------------------------
def main(args):
    inputs, targets = load_and_align_data(args.input_no_diac, args.input_with_diac)
    if not inputs:
        print("No data loaded. Please check your input file patterns.")
        return

    char2idx, tag2idx = build_mappings(inputs, targets)
    idx2tag = {i: t for t, i in tag2idx.items()}
    print(f"Vocabulary size: {len(char2idx)} characters")
    print(f"Tagset size: {len(tag2idx)} unique diacritic combinations")

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.15, random_state=42)
    print(f"Training set size: {len(train_inputs)}")
    print(f"Validation set size: {len(val_inputs)}")

    train_dataset = HebrewPunctuationDataset(train_inputs, train_targets, char2idx, tag2idx, args.max_len)
    val_dataset = HebrewPunctuationDataset(val_inputs, val_targets, char2idx, tag2idx, args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BiLSTMPunctuator(
        vocab_size=len(char2idx), embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, tagset_size=len(tag2idx),
        num_layers=args.num_layers, dropout=args.dropout
    ).to(device)

    pad_idx = tag2idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # The evaluate function now returns both loss and accuracy
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, pad_idx)
        
        # Updated print statement to include the new accuracy metric
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'punctuator_best.pt')
            torch.save({
                'model_state': model.state_dict(), 'char2idx': char2idx,
                'tag2idx': tag2idx, 'args': args
            }, save_path)
            print(f"Validation loss improved. Model saved to {save_path}")

    print("\n--- Inference ---")
    save_path = os.path.join(args.save_dir, 'punctuator_best.pt')
    if not os.path.exists(save_path):
        print("No model was saved. Ending.")
        return
        
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    
    saved_args = checkpoint['args']
    inference_model = BiLSTMPunctuator(
        vocab_size=len(checkpoint['char2idx']), embedding_dim=saved_args.embed_dim,
        hidden_dim=saved_args.hidden_dim, tagset_size=len(checkpoint['tag2idx']),
        num_layers=saved_args.num_layers, dropout=saved_args.dropout
    ).to(device)
    
    inference_model.load_state_dict(checkpoint['model_state'])
    
    sample = args.sample_text.strip() or "שלום עולם"
    output = punctuate(inference_model, sample, checkpoint['char2idx'], {i: t for t, i in checkpoint['tag2idx'].items()}, device)
    print("Input:     ", sample)
    print("Punctuated:", output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved Hebrew Punctuation Trainer')
    parser.add_argument('--input_no_diac', required=True, help='Glob pattern for no-diacritics text files (e.g., "data/*_no_diac.txt")')
    parser.add_argument('--input_with_diac', required=True, help='Glob pattern for diacritics text files (e.g., "data/*_with_diac.txt")')
    parser.add_argument('--save_dir', default='./model_improved', help='Directory to save the best model')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of character embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Size of LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--sample_text', default='שלום עולם', help='Sample unpointed text for testing after training')
    args = parser.parse_args()
    main(args)
