import streamlit as st
import torch
from models.lstm_model import LSTMClassifier


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    model = LSTMClassifier(
        vocab_size=5000,
        embedding_dim=64,
        hidden_dim=128,
        output_dim=1
    )

    model.load_state_dict(
        torch.load(
            "fake_news_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Tokenizer
# -----------------------------
def simple_tokenizer(text):

    tokens = [
        ord(c) % 256
        for c in text.lower()
    ]

    # Limit sequence length
    tokens = tokens[:100]

    # Pad sequence
    if len(tokens) < 100:
        tokens += [0] * (100 - len(tokens))

    return tokens


# -----------------------------
# Prediction
# -----------------------------
def predict_news(text):

    input_data = simple_tokenizer(text)

    input_tensor = torch.tensor(
        [input_data],
        dtype=torch.long
    )

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output).item()

    return probability


# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection")

st.write(
    "An LSTM-based PyTorch model for "
    "text classification."
)

st.divider()

news_text = st.text_area(
    "Enter a news article or headline:",
    height=200,
    placeholder="Paste news text here..."
)


if st.button("🔍 Analyze News", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter some news text.")

    else:

        probability = predict_news(news_text)

        st.subheader("Prediction")

        if probability > 0.5:

            st.success("🟢 Likely REAL news")

        else:

            st.error("🔴 Likely FAKE news")

        st.metric(
            "Model Score",
            f"{probability:.2%}"
        )


st.divider()

st.caption(
    "⚠️ Research/demo model. "
    "Predictions should not be treated as definitive fact-checking."
)
