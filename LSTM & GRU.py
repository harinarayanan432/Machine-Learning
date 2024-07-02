import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def create_ngrams(text, n):
    ngrams = []
    for i in range(len(text) - n):
        ngrams.append(text[i:i+n])
    return ngrams

def preprocess_data(text, n):
    ngrams = create_ngrams(text, n)
    X = [gram[:-1] for gram in ngrams]
    y = [gram[-1] for gram in ngrams]
    return X, y

# Build LSTM model
def build_lstm_model(seq_length, vocab_size):
    model = Sequential([
        Embedding(vocab_size, 50, input_length=seq_length),
        LSTM(128),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build GRU model
def build_gru_model(seq_length, vocab_size):
    model = Sequential([
        Embedding(vocab_size, 50, input_length=seq_length),
        GRU(128),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model
def train_model(model, X_train, y_train, epochs=10):
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, verbose=1)
    end_time = time.time()
    training_time = end_time - start_time
    return history, training_time

# Generate prediction
def generate_text(model, start_seed, char_to_idx, idx_to_char, seq_length, num_chars):
    generated_text = start_seed
    for _ in range(num_chars):
        encoded = [char_to_idx[char] for char in generated_text]
        encoded = np.array(encoded).reshape(1, -1)
        predicted_idx = np.argmax(model.predict(encoded), axis=1)
        predicted_char = idx_to_char[predicted_idx[0]]
        generated_text += predicted_char
    return generated_text

# Plot training history
def plot_training_history(history_lstm, history_gru):
    plt.plot(history_lstm.history['loss'], label='LSTM Loss')
    plt.plot(history_gru.history['loss'], label='GRU Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Load and preprocess data
text = load_data('D:\\shakespeare.txt')
n = 5
X, y = preprocess_data(text, n)

# Convert characters to indices
char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
X_idx = [[char_to_idx[char] for char in gram] for gram in X]
y_idx = [char_to_idx[char] for char in y]

# Pad sequences to have the same length
max_len = max(len(gram) for gram in X_idx)
X_padded = [gram + [0] * (max_len - len(gram)) for gram in X_idx]

# Build models
lstm_model = build_lstm_model(max_len, len(char_to_idx))
gru_model = build_gru_model(max_len, len(char_to_idx))

# Train models
history_lstm, lstm_training_time = train_model(lstm_model, np.array(X_padded), np.array(y_idx))
history_gru, gru_training_time = train_model(gru_model, np.array(X_padded), np.array(y_idx))

# Generate predictions
start_seed = "Project Gutenberg-tm eBooks are often created from several printed editions"
generated_text_lstm = generate_text(lstm_model, start_seed, char_to_idx, idx_to_char, max_len, num_chars=500)
generated_text_gru = generate_text(gru_model, start_seed, char_to_idx, idx_to_char, max_len, num_chars=500)

# Plot training history
plot_training_history(history_lstm, history_gru)

# Compare performance
print("LSTM Training Time:", lstm_training_time)
print("GRU Training Time:", gru_training_time)

# Evaluation
y_pred_lstm = np.argmax(lstm_model.predict(np.array(X_padded)), axis=1)
y_pred_gru = np.argmax(gru_model.predict(np.array(X_padded)), axis=1)

print("LSTM Classification Report:")
print(classification_report(y_idx, y_pred_lstm))

print("GRU Classification Report:")
print(classification_report(y_idx, y_pred_gru))
