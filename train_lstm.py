# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm_model import LSTMClassifier
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Sample dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor([vocab.get(w, 0) for w in text.split()]) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def pad_batch(batch):
    texts, labels = zip(*batch)
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded_texts, torch.tensor(labels)

# Dummy vocab and data
vocab = {"this": 1, "is": 2, "fake": 3, "news": 4, "real": 5}
texts = ["this is fake news", "this is real news", "fake fake news"]
labels = [0, 1, 0]  # 0 = Fake, 1 = Real

dataset = FakeNewsDataset(texts, labels, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_batch)

# Hyperparameters
vocab_size = len(vocab) + 1
embed_dim = 10
hidden_dim = 16
output_dim = 2

# Model, Loss, Optimizer
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

print("Training complete.")
