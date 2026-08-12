import streamlit as st
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
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# Load vocabulary
# -----------------------------

@st.cache_resource
def load_vocab():

    return torch.load(
        "vocab.pth",
        weights_only=False
    )


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    vocab = load_vocab()

    model = LSTMClassifier(
        vocab_size=len(vocab),
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

    return model


vocab = load_vocab()
model = load_model()


# -----------------------------
# Text preprocessing
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
# UI
# -----------------------------

st.title("📰 Fake News Detection")

st.write(
    "LSTM-based text classification using PyTorch"
)

st.divider()


news_text = st.text_area(
    "Enter a news article or headline:",
    height=220,
    placeholder=(
        "Paste a news headline or article here..."
    )
)


if st.button(
    "🔍 Analyze News",
    use_container_width=True
):

    if not news_text.strip():

        st.warning(
            "Please enter some news text first."
        )

    else:

        prediction, confidence = predict(
            news_text
        )

        st.subheader("Result")

        if prediction == 1:

            st.success(
                "🟢 Likely REAL news"
            )

        else:

            st.error(
                "🔴 Likely FAKE news"
            )

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}"
        )


st.divider()

st.caption(
    "⚠️ Research/demo model. "
    "This prediction should not be treated "
    "as definitive fact-checking."
)
