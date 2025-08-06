# punctuator_advanced.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint # Import for Gradient Checkpointing
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import for Dynamic LR
from sklearn.model_selection import train_test_split
import re
import glob
import time 

# ---------------------------------------------
# 1. Define Hebrew Character Sets
# ---------------------------------------------
HEBREW_LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
DIACRITICS = "ְֱֲֳִֵֶַָׁׂ"
TOKEN_RE = re.compile(f'([{HEBREW_LETTERS}])([{DIACRITICS}]*)')

# ---------------------------------------------
# 2. Dataset and Preprocessing (No changes here)
# ---------------------------------------------

def load_and_align_data(with_diac_pattern):
    with_diac_files = sorted(glob.glob(with_diac_pattern))
    all_input_sequences, all_target_sequences = [], []
    print(f"Found {len(with_diac_files)} files with diacritics to process.")
    for with_file in with_diac_files:
        print(f"Processing: {os.path.basename(with_file)}")
        with open(with_file, 'r', encoding='utf-8') as f_with_diac:
            for line_with_diac in f_with_diac:
                line_with_diac = line_with_diac.strip()
                if not line_with_diac: continue
                input_seq, target_seq = [], []
                tokens = TOKEN_RE.findall(line_with_diac)
                for letter, diac in tokens:
                    input_seq.append(letter)
                    target_seq.append(diac if diac else 'O')
                if input_seq and target_seq:
                    all_input_sequences.append("".join(input_seq))
                    all_target_sequences.append(target_seq)
                #   elif line_with_diac:
                    # print(f"  [Warning] Skipped line: '{line_with_diac[:50]}...'")
    print(f"Loaded {len(all_input_sequences)} aligned sequences.")
    return all_input_sequences, all_target_sequences

def build_mappings(input_texts, target_sequences):
    all_input_chars = set(c for text in input_texts for c in text)
    char2idx = {char: i + 2 for i, char in enumerate(sorted(list(all_input_chars)))}
    char2idx['<PAD>'], char2idx['<UNK>'] = 0, 1
    all_tags = set(tag for seq in target_sequences for tag in seq)
    tag2idx = {tag: i + 1 for i, tag in enumerate(sorted(list(all_tags)))}
    tag2idx['<PAD>'] = 0
    return char2idx, tag2idx

class HebrewPunctuationDataset(Dataset):
    def __init__(self, texts, sequences, char2idx, tag2idx, max_len):
        self.texts, self.sequences = texts, sequences
        self.char2idx, self.tag2idx, self.max_len = char2idx, tag2idx, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text, tags = self.texts[idx], self.sequences[idx]
        x = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
        y = [self.tag2idx.get(t, 0) for t in tags]
        x = x[:self.max_len] + [0] * (self.max_len - len(x))
        y = y[:self.max_len] + [0] * (self.max_len - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------------------------------------------
# 3. Model with Gradient Checkpointing
# ---------------------------------------------
class BiLSTMPunctuator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers, dropout, use_checkpointing=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        emb = self.embedding(x)
        if self.use_checkpointing and self.training:
            lstm_out, _ = checkpoint(self.bilstm, emb)
        else:
            lstm_out, _ = self.bilstm(emb)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

# ---------------------------------------------
# 4. Training & Evaluation Loops (No changes here)
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
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            non_pad_mask = (y_batch != pad_idx)
            total_correct += ((predictions == y_batch) & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    return avg_loss, accuracy

# ---------------------------------------------
# 5. Inference Function (No changes here)
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
        if tag in ('O', '<PAD>'): result += char
        else: result += char + tag
    return result

# ---------------------------------------------
# 6. Main Function with Advanced Features
# ---------------------------------------------
def main(args):
    start = time.time()
    inputs, targets = load_and_align_data(args.input_with_diac)
    if not inputs: return

    char2idx, tag2idx = build_mappings(inputs, targets)
    print(f"Vocabulary size: {len(char2idx)} chars, Tagset size: {len(tag2idx)} tags")

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.15, random_state=42)
    print(f"Training set: {len(train_inputs)}, Validation set: {len(val_inputs)}")

    train_ds = HebrewPunctuationDataset(train_inputs, train_targets, char2idx, tag2idx, args.max_len)
    val_ds = HebrewPunctuationDataset(val_inputs, val_targets, char2idx, tag2idx, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BiLSTMPunctuator(
        vocab_size=len(char2idx), embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, tagset_size=len(tag2idx),
        num_layers=args.num_layers, dropout=args.dropout,
        use_checkpointing=args.use_checkpointing
    ).to(device)
    
    if args.use_checkpointing:
        print("Gradient Checkpointing is ENABLED.")

    pad_idx = tag2idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- DYNAMIC LEARNING RATE SCHEDULER (verbose parameter removed) ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, pad_idx)
        
        # --- Manually get the current learning rate ---
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- Updated print statement to show the current LR ---
        print(f"Epoch {epoch:02d}/{args.epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # The scheduler monitors the validation loss to decide when to change the LR
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'punctuator_best.pt')
            torch.save({'model_state': model.state_dict(), 'char2idx': char2idx,
                        'tag2idx': tag2idx, 'args': args}, save_path)
            print(f"Validation loss improved. Model saved to {save_path}")

    print("\n--- Inference ---")
    # Load and run inference as before
    save_path = os.path.join(args.save_dir, 'punctuator_best.pt')
    if not os.path.exists(save_path):
        print("No model was saved. Ending.")
        return
        
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    saved_args = checkpoint['args']
    idx2tag = {i: t for t, i in checkpoint['tag2idx'].items()}
    inference_model = BiLSTMPunctuator(
        vocab_size=len(checkpoint['char2idx']), embedding_dim=saved_args.embed_dim,
        hidden_dim=saved_args.hidden_dim, tagset_size=len(checkpoint['tag2idx']),
        num_layers=saved_args.num_layers, dropout=saved_args.dropout
    ).to(device)
    inference_model.load_state_dict(checkpoint['model_state'])
    sample = args.sample_text.strip() or "שלום עולם"
    output = punctuate(inference_model, sample, checkpoint['char2idx'], idx2tag, device)
    print("Input:     ", sample)
    print("Punctuated:", output)
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Hebrew Punctuation Trainer')
    parser.add_argument('--input_with_diac', required=True, help='Glob pattern for diacritics text files (e.g., "data/*_with_diac.txt")')
    parser.add_argument('--save_dir', default='./model_advanced', help='Directory to save the best model')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--use_checkpointing', action='store_true', help='Enable gradient checkpointing to save memory.')
    parser.add_argument('--sample_text', default='שלום לך צדיק. איך אתה מרגיש היום? המלכות שבבינה והגבורה')
    args = parser.parse_args()
    
    if not glob.glob(args.input_with_diac):
        print(f"Error: No files found matching the pattern '{args.input_with_diac}'")
    else:
        main(args)
