import torch
from models.lstm_model import LSTMClassifier

# Load the saved model
model = LSTMClassifier(vocab_size=5000, embedding_dim=64, hidden_dim=128, output_dim=1)
model.load_state_dict(torch.load("fake_news_model.pth"))
model.eval()

# Fake example text (you can update this with real test input)
example_input = "This news article claims something very controversial."

# Dummy tokenizer simulation (for now)
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text.lower()][:100]  # truncate or pad to length

# Preprocess input
input_data = simple_tokenizer(example_input)
input_tensor = torch.tensor([input_data], dtype=torch.long)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    print(f"Prediction score: {prediction.item():.4f}")
    if prediction.item() > 0.5:
        print("🟢 Likely REAL news.")
    else:
        print("🔴 Likely FAKE news.")
