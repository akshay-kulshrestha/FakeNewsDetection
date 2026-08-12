import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.lstm_model import LSTMClassifier


# -----------------------------
# Dataset
# -----------------------------

class FakeNewsDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def encode_text(self, text):

        tokens = [
            self.vocab.get(word.lower(), self.vocab["<UNK>"])
            for word in text.split()
        ]

        tokens = tokens[:self.max_length]

        if len(tokens) < self.max_length:
            tokens += [self.vocab["<PAD>"]] * (
                self.max_length - len(tokens)
            )

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.encode_text(self.texts[idx])

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return text, label


# -----------------------------
# Vocabulary
# -----------------------------

vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "this": 2,
    "is": 3,
    "fake": 4,
    "news": 5,
    "real": 6
}

vocab_size = len(vocab)


# -----------------------------
# Dataset
# -----------------------------

texts = [
    "this is fake news",
    "this is real news",
    "fake fake news",
    "this is real news",
]

# 0 = Fake
# 1 = Real

labels = [
    0,
    1,
    0,
    1
]


dataset = FakeNewsDataset(
    texts,
    labels,
    vocab
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# -----------------------------
# Model
# -----------------------------

embedding_dim = 64
hidden_dim = 128
output_dim = 2

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)


# -----------------------------
# Loss + optimizer
# -----------------------------

criterion = nn.NLLLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

num_epochs = 20

print("Starting training...")

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {average_loss:.4f}"
    )


print("Training complete.")


# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "fake_news_model.pth"
)

print("Model saved to fake_news_model.pth")


# -----------------------------
# Save vocabulary
# -----------------------------

torch.save(
    vocab,
    "vocab.pth"
)

print("Vocabulary saved to vocab.pth")
