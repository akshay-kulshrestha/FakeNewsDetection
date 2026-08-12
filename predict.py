import torch

from models.lstm_model import LSTMClassifier


# -----------------------------
# Configuration
# -----------------------------

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 2
MAX_LENGTH = 100


# -----------------------------
# Load vocabulary
# -----------------------------

vocab = torch.load(
    "vocab.pth",
    weights_only=False
)

vocab_size = len(vocab)


# -----------------------------
# Load model
# -----------------------------

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
)

model.load_state_dict(
    torch.load(
        "fake_news_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# -----------------------------
# Preprocess text
# -----------------------------

def preprocess_text(text):

    tokens = [
        vocab.get(
            word.lower(),
            vocab["<UNK>"]
        )
        for word in text.split()
    ]

    tokens = tokens[:MAX_LENGTH]

    if len(tokens) < MAX_LENGTH:
        tokens += [
            vocab["<PAD>"]
        ] * (MAX_LENGTH - len(tokens))

    return torch.tensor(
        [tokens],
        dtype=torch.long
    )


# -----------------------------
# Prediction
# -----------------------------

def predict(text):

    input_tensor = preprocess_text(text)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.exp(output)

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

    return prediction, confidence


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    text = input(
        "Enter a news article or headline: "
    )

    prediction, confidence = predict(text)

    if prediction == 1:
        print(
            f"🟢 Likely REAL news "
            f"({confidence:.2%} confidence)"
        )
    else:
        print(
            f"🔴 Likely FAKE news "
            f"({confidence:.2%} confidence)"
        )
