# Rnn Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Describe the process of implementing an RNN with TensorFlow or PyTorch.**

### Answer

**Process Overview:**

```
1. Define Model Architecture
2. Prepare Data (sequences + padding)
3. Create DataLoader
4. Define Loss & Optimizer
5. Training Loop
6. Evaluation
```

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============ STEP 1: Define Model ============
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, h0)
        # out shape: (batch, seq_len, hidden_size)
        
        # Use last time step output
        out = self.fc(out[:, -1, :])
        return out

# ============ STEP 2: Prepare Data ============
# Example: sequence classification
X = torch.randn(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
y = torch.randint(0, 3, (100,))  # 3 classes

# ============ STEP 3: Create DataLoader ============
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ============ STEP 4: Define Loss & Optimizer ============
model = SimpleRNN(input_size=5, hidden_size=32, output_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============ STEP 5: Training Loop ============
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ============ STEP 6: Evaluation ============
model.eval()
with torch.no_grad():
    predictions = model(X)
    predicted_classes = predictions.argmax(dim=1)
    accuracy = (predicted_classes == y).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

**TensorFlow/Keras Implementation:**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ============ STEP 1: Define Model ============
model = Sequential([
    SimpleRNN(32, input_shape=(10, 5), return_sequences=False),
    Dense(3, activation='softmax')
])

# ============ STEP 2: Compile ============
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============ STEP 3: Train ============
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# ============ STEP 4: Evaluate ============
loss, accuracy = model.evaluate(X, y)
```

**Key Differences:**

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|------------------|
| Model definition | Class with forward() | Sequential/Functional API |
| Training loop | Manual | model.fit() |
| Gradients | Manual zero_grad, backward | Automatic |
| Flexibility | More control | Simpler API |

---

## Question 2
- [ ] Done

**Implement a basic RNN to classify sequential data in Python using a library of your choice.**

### Answer

**Task:** Sentiment Classification on text sequences

**Pipeline:**
```
Text → Tokenize → Pad → Embed → RNN → Classify
```

**Complete Implementation (PyTorch):**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# ========== STEP 1: Text Preprocessing ==========
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=50):
        self.labels = labels
        self.max_len = max_len
        
        # Build vocabulary from training data
        if vocab is None:
            word_counts = Counter()
            for text in texts:
                word_counts.update(text.lower().split())
            # word2idx: PAD=0, UNK=1, then frequent words
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            for word, _ in word_counts.most_common(5000):
                self.word2idx[word] = len(self.word2idx)
        else:
            self.word2idx = vocab
        
        # Convert texts to indices
        self.sequences = []
        for text in texts:
            tokens = text.lower().split()
            indices = [self.word2idx.get(w, 1) for w in tokens]  # 1 = UNK
            self.sequences.append(indices)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Pad or truncate
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))  # Pad
        else:
            seq = seq[:self.max_len]  # Truncate
        return torch.tensor(seq), torch.tensor(self.labels[idx])

# ========== STEP 2: RNN Model ==========
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextRNN, self).__init__()
        
        # Embedding layer: word indices → vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # Classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) - word indices
        
        # Embed: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # RNN: (batch, seq_len, hidden_dim)
        output, hidden = self.rnn(embedded)
        
        # Use last hidden state: (batch, hidden_dim)
        last_hidden = output[:, -1, :]
        
        # Classify: (batch, num_classes)
        out = self.fc(last_hidden)
        return out

# ========== STEP 3: Training Function ==========
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Accuracy={correct/total:.4f}")

# ========== STEP 4: Main Execution ==========
# Sample data (replace with real data)
texts = [
    "I love this movie it is great",
    "This film is terrible I hate it",
    "Amazing performance by the actors",
    "Worst movie ever do not watch",
    "Highly recommend this film",
    "Boring and dull waste of time"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Create dataset and dataloader
dataset = TextDataset(texts, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Create model
model = TextRNN(
    vocab_size=len(dataset.word2idx),
    embed_dim=32,
    hidden_dim=64,
    num_classes=2
)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=10)

# ========== STEP 5: Inference ==========
def predict(model, text, word2idx, max_len=50):
    model.eval()
    tokens = text.lower().split()
    indices = [word2idx.get(w, 1) for w in tokens]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    with torch.no_grad():
        input_tensor = torch.tensor([indices])
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# Test
print(predict(model, "This is a fantastic movie", dataset.word2idx))
```

**Output Flow:**
```
Input: "I love this movie" 
  → Tokenize: ["i", "love", "this", "movie"]
  → Index: [23, 45, 12, 78, 0, 0, ...] (padded)
  → Embed: (1, 50, 32) 
  → RNN: (1, 50, 64) 
  → Last hidden: (1, 64)
  → FC: (1, 2) → softmax → [0.2, 0.8]
  → Prediction: Positive
```

---

## Question 3
- [ ] Done

**Write Python code using TensorFlow/Keras to build and train an LSTM network on a text dataset.**

### Answer

**Task:** Text Classification with LSTM

**Pipeline:**
```
Text → Tokenize → Pad → Embedding → LSTM → Dense → Output
```

**Complete Implementation:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ========== STEP 1: Prepare Data ==========
# Sample dataset (replace with real data)
texts = [
    "I love this product it works great",
    "Terrible quality do not buy",
    "Best purchase I have ever made",
    "Waste of money very disappointed",
    "Highly recommend to everyone",
    "Poor customer service awful experience",
    "Amazing results exceeded expectations",
    "Broke after one week useless"
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

# ========== STEP 2: Tokenization ==========
vocab_size = 1000
max_length = 20

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
# Output: [[12, 45, 23, ...], [8, 92, 15, ...], ...]

# Pad sequences to same length
padded = pad_sequences(sequences, maxlen=max_length, 
                       padding='post', truncating='post')
# Output shape: (num_samples, max_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# ========== STEP 3: Build LSTM Model ==========
embedding_dim = 32
lstm_units = 64

model = Sequential([
    # Embedding: word indices → dense vectors
    Embedding(input_dim=vocab_size, 
              output_dim=embedding_dim, 
              input_length=max_length),
    
    # LSTM layer
    LSTM(units=lstm_units, return_sequences=False),
    
    # Dropout for regularization
    Dropout(0.3),
    
    # Output layer
    Dense(1, activation='sigmoid')  # Binary classification
])

# ========== STEP 4: Compile Model ==========
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# ========== STEP 5: Train Model ==========
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=2,
    validation_data=(X_test, y_test),
    verbose=1
)

# ========== STEP 6: Evaluate ==========
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# ========== STEP 7: Inference Function ==========
def predict_sentiment(text):
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_length, 
                               padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded_seq, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

# Test predictions
test_texts = [
    "This is absolutely wonderful",
    "I hate this so much"
]

for text in test_texts:
    sentiment, conf = predict_sentiment(text)
    print(f"'{text}' → {sentiment} ({conf:.2%})")
```

**Model Architecture:**
```
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
embedding (Embedding)       (None, 20, 32)            32000
lstm (LSTM)                 (None, 64)                24832
dropout (Dropout)           (None, 64)                0
dense (Dense)               (None, 1)                 65
=================================================================
Total params: 56,897
```

**Flow Explanation:**
```
Input: "I love this product"
  ↓
Tokenizer: [15, 42, 8, 67, 0, 0, ...] (padded to 20)
  ↓
Embedding: (1, 20, 32) - each word → 32-dim vector
  ↓
LSTM: (1, 64) - processes sequence, outputs final state
  ↓
Dropout: (1, 64) - regularization
  ↓
Dense + Sigmoid: (1, 1) - probability [0, 1]
  ↓
Output: 0.87 → "Positive"
```

**Multi-class Variation:**
```python
# For multi-class (e.g., 5 categories)
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64),
    Dense(5, activation='softmax')  # 5 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Integer labels
    metrics=['accuracy']
)
```

---

## Question 4
- [ ] Done

**Create a GRU-based neural network in PyTorch for predicting the next item in a sequence.**

### Answer

**Task:** Next Item Prediction (e.g., next word, next product)

**Pipeline:**
```
Sequence [item1, item2, ..., item_n] → GRU → Predict item_{n+1}
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ========== STEP 1: Dataset Class ==========
class SequenceDataset(Dataset):
    """Creates sequences for next-item prediction"""
    
    def __init__(self, data, seq_length):
        """
        data: list of item indices [1, 5, 3, 8, 2, ...]
        seq_length: how many items to use for prediction
        """
        self.seq_length = seq_length
        self.X = []  # Input sequences
        self.y = []  # Next item (target)
        
        # Create sliding window sequences
        for i in range(len(data) - seq_length):
            self.X.append(data[i:i+seq_length])
            self.y.append(data[i+seq_length])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ========== STEP 2: GRU Model ==========
class GRUPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(GRUPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer: predict next item
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        
        # Embedding: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # GRU forward: output (batch, seq_len, hidden_dim)
        output, hidden = self.gru(embedded, hidden)
        
        # Use last output for prediction
        last_output = output[:, -1, :]  # (batch, hidden_dim)
        
        # Predict next item: (batch, vocab_size)
        prediction = self.fc(last_output)
        
        return prediction, hidden

# ========== STEP 3: Training Function ==========
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, targets in train_loader:
            # Forward pass
            outputs, _ = model(sequences)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

# ========== STEP 4: Main Execution ==========
# Example: Product sequence prediction
# Items: [1, 2, 3, ...] representing products

# Sample data: user purchase sequences
data = [1, 3, 5, 2, 8, 3, 5, 9, 1, 4, 7, 2, 5, 3, 8, 6, 1, 9, 4, 2, 
        5, 7, 3, 8, 1, 6, 4, 9, 2, 5, 7, 3, 1, 8, 6, 4]

# Parameters
vocab_size = 10   # Number of unique items (0-9)
seq_length = 5    # Use 5 items to predict 6th
embed_dim = 16
hidden_dim = 32

# Create dataset and dataloader
dataset = SequenceDataset(data, seq_length)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create model
model = GRUPredictor(vocab_size, embed_dim, hidden_dim)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
train(model, train_loader, criterion, optimizer, epochs=50)

# ========== STEP 5: Prediction Function ==========
def predict_next(model, sequence, top_k=3):
    """Predict next item given a sequence"""
    model.eval()
    
    with torch.no_grad():
        input_seq = torch.tensor([sequence])
        output, _ = model(input_seq)
        
        # Get probabilities
        probs = torch.softmax(output[0], dim=0)
        
        # Top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append((idx.item(), prob.item()))
        
        return predictions

# Test prediction
test_sequence = [1, 3, 5, 2, 8]
predictions = predict_next(model, test_sequence)
print(f"\nGiven sequence: {test_sequence}")
print("Top predictions for next item:")
for item, prob in predictions:
    print(f"  Item {item}: {prob:.2%}")
```

**Output Example:**
```
Given sequence: [1, 3, 5, 2, 8]
Top predictions for next item:
  Item 3: 45.2%
  Item 5: 22.1%
  Item 9: 15.8%
```

**Flow:**
```
Input: [1, 3, 5, 2, 8]
  ↓
Embedding: (1, 5, 16)
  ↓
GRU: processes sequence → hidden state encodes pattern
  ↓
Last hidden: (1, 32)
  ↓
Linear: (1, 10) → logits for each possible item
  ↓
Softmax: probabilities
  ↓
Output: Item 3 most likely
```

---

## Question 5
- [ ] Done

**Develop an RNN model with attention that translates sentences from English to French.**

### Answer

**Task:** Sequence-to-Sequence Translation with Attention

**Pipeline:**
```
English → Encoder (BiLSTM) → Attention → Decoder (LSTM) → French
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== STEP 1: Encoder ==========
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        # x: (batch, src_len)
        embedded = self.embedding(x)  # (batch, src_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, src_len, hidden_dim * 2) - all hidden states
        # hidden: (2, batch, hidden_dim) - final states (forward, backward)
        
        # Combine bidirectional hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # (batch, hidden_dim * 2)
        cell = torch.cat((cell[0], cell[1]), dim=1)
        
        return outputs, hidden, cell

# ========== STEP 2: Attention ==========
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        # Project encoder outputs and decoder state to same dim
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, decoder_dim)
        # encoder_outputs: (batch, src_len, encoder_dim)
        
        src_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden for each source position
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_dim)
        
        # Concatenate and compute attention scores
        energy = torch.tanh(self.attn(
            torch.cat((decoder_hidden, encoder_outputs), dim=2)
        ))  # (batch, src_len, decoder_dim)
        
        attention_scores = self.v(energy).squeeze(2)  # (batch, src_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # (batch, 1, encoder_dim) → squeeze → (batch, encoder_dim)
        
        return context.squeeze(1), attention_weights

# ========== STEP 3: Decoder with Attention ==========
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim)
        
        # LSTM input: embedding + context
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        # Output projection
        self.fc = nn.Linear(decoder_dim + encoder_dim + embed_dim, vocab_size)
        
    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (batch,) - current target token
        # hidden, cell: (batch, decoder_dim)
        # encoder_outputs: (batch, src_len, encoder_dim)
        
        # Embed input token
        embedded = self.embedding(input_token)  # (batch, embed_dim)
        
        # Compute attention
        context, attn_weights = self.attention(hidden, encoder_outputs)
        # context: (batch, encoder_dim)
        
        # LSTM step
        lstm_input = torch.cat((embedded, context), dim=1)
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Output prediction
        output = self.fc(torch.cat((hidden, context, embedded), dim=1))
        # (batch, vocab_size)
        
        return output, hidden, cell, attn_weights

# ========== STEP 4: Seq2Seq with Attention ==========
class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)
        
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode source
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First decoder input is <SOS> token
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else output.argmax(1)
        
        return outputs

# ========== STEP 5: Usage Example ==========
# Vocabulary setup (simplified)
SRC_VOCAB_SIZE = 1000  # English vocabulary
TRG_VOCAB_SIZE = 1200  # French vocabulary
EMBED_DIM = 64
ENCODER_DIM = 128  # BiLSTM: 64*2 = 128
DECODER_DIM = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model components
encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, 64)  # 64 per direction
decoder = Decoder(TRG_VOCAB_SIZE, EMBED_DIM, ENCODER_DIM, DECODER_DIM)
model = Seq2SeqAttention(encoder, decoder, device).to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example training step
def train_step(model, src, trg, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    output = model(src, trg)  # (batch, trg_len, vocab_size)
    
    # Reshape for loss: ignore first token (SOS)
    output = output[:, 1:, :].reshape(-1, TRG_VOCAB_SIZE)
    trg = trg[:, 1:].reshape(-1)
    
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# Inference
def translate(model, src_sentence, src_vocab, trg_vocab, max_len=50):
    model.eval()
    
    with torch.no_grad():
        # Encode source
        src_tensor = torch.tensor([src_sentence]).to(device)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Start with <SOS>
        input_token = torch.tensor([trg_vocab['<SOS>']]).to(device)
        
        translation = []
        for _ in range(max_len):
            output, hidden, cell, attn = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            
            next_token = output.argmax(1)
            if next_token.item() == trg_vocab['<EOS>']:
                break
            translation.append(next_token.item())
            input_token = next_token
    
    return translation
```

**Flow:**
```
English: "I love cats"
  ↓
Encoder (BiLSTM): captures source semantics
  ↓
At each decoding step:
  - Attention: which source words to focus on
  - Context: weighted sum of encoder outputs
  - Decoder: generates French word
  ↓
French: "J'aime les chats"
```

---

## Question 6
- [ ] Done

**Code a function that visualizes the hidden state dynamics of an RNN during sequence processing.**

### Answer

**Task:** Visualize how RNN hidden states evolve as it processes a sequence

**Pipeline:**
```
Input Sequence → RNN (capture hidden states) → Visualize dynamics
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== STEP 1: RNN with Hidden State Tracking ==========
class RNNWithStateTracking(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNWithStateTracking, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        
    def forward(self, x, return_all_states=True):
        # x: (batch, seq_len, input_size)
        output, final_hidden = self.rnn(x)
        
        if return_all_states:
            # output contains all hidden states: (batch, seq_len, hidden_size)
            return output, final_hidden
        return final_hidden

# ========== STEP 2: Visualization Functions ==========
def visualize_hidden_states(hidden_states, title="Hidden State Dynamics"):
    """
    Visualize hidden states as a heatmap
    
    Args:
        hidden_states: (seq_len, hidden_size) numpy array
        title: plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Heatmap of hidden states
    plt.subplot(1, 2, 1)
    sns.heatmap(hidden_states.T, cmap='RdBu_r', center=0)
    plt.xlabel('Time Step')
    plt.ylabel('Hidden Unit')
    plt.title(f'{title} - Heatmap')
    
    # Line plot of selected hidden units
    plt.subplot(1, 2, 2)
    num_units_to_plot = min(5, hidden_states.shape[1])
    for i in range(num_units_to_plot):
        plt.plot(hidden_states[:, i], label=f'Unit {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Activation')
    plt.title('Hidden Unit Activations Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_hidden_state_pca(hidden_states, labels=None):
    """
    Visualize hidden states using PCA
    
    Args:
        hidden_states: (seq_len, hidden_size) numpy array
        labels: optional labels for each time step
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(hidden_states)
    
    plt.figure(figsize=(10, 6))
    
    # Color by time step
    colors = np.arange(len(hidden_states))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                         c=colors, cmap='viridis', s=100)
    
    # Draw trajectory
    plt.plot(reduced[:, 0], reduced[:, 1], 'k-', alpha=0.3)
    
    # Mark start and end
    plt.scatter(reduced[0, 0], reduced[0, 1], c='green', s=200, 
               marker='o', label='Start', zorder=5)
    plt.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=200, 
               marker='X', label='End', zorder=5)
    
    # Add labels if provided
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                        fontsize=8, ha='center')
    
    plt.colorbar(scatter, label='Time Step')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Hidden State Trajectory (PCA)')
    plt.legend()
    plt.show()

def visualize_gate_activations(model, x, gate_names=['Forget', 'Input', 'Output']):
    """
    Visualize LSTM gate activations
    
    Args:
        model: LSTM model
        x: input tensor (1, seq_len, input_size)
        gate_names: names of gates
    """
    # This requires hooking into LSTM internals - simplified version
    # For full gate visualization, use a custom LSTM implementation
    
    model.eval()
    with torch.no_grad():
        hidden_states, _ = model(x, return_all_states=True)
        hidden_states = hidden_states[0].numpy()  # Remove batch dim
    
    visualize_hidden_states(hidden_states, "LSTM Hidden States")

# ========== STEP 3: Main Visualization Pipeline ==========
def analyze_rnn_dynamics(text_sequence, word2idx, model):
    """
    Complete pipeline to analyze RNN dynamics for a text sequence
    """
    # Convert text to indices
    tokens = text_sequence.lower().split()
    indices = [word2idx.get(w, 0) for w in tokens]
    
    # Create input tensor (one-hot or embedding)
    vocab_size = len(word2idx)
    x = torch.zeros(1, len(indices), vocab_size)
    for t, idx in enumerate(indices):
        x[0, t, idx] = 1.0
    
    # Get hidden states
    model.eval()
    with torch.no_grad():
        hidden_states, _ = model(x, return_all_states=True)
        hidden_states = hidden_states[0].numpy()  # (seq_len, hidden_size)
    
    # Visualizations
    print(f"Analyzing: '{text_sequence}'")
    print(f"Sequence length: {len(tokens)}")
    print(f"Hidden state shape: {hidden_states.shape}")
    
    # 1. Heatmap and line plot
    visualize_hidden_states(hidden_states)
    
    # 2. PCA trajectory with word labels
    visualize_hidden_state_pca(hidden_states, labels=tokens)
    
    return hidden_states

# ========== STEP 4: Example Usage ==========
# Create simple vocabulary
sample_text = "the cat sat on the mat and the dog ran away"
words = sample_text.split()
word2idx = {w: i for i, w in enumerate(set(words))}
vocab_size = len(word2idx)

# Create model
model = RNNWithStateTracking(input_size=vocab_size, hidden_size=16)

# Analyze dynamics
hidden_states = analyze_rnn_dynamics(sample_text, word2idx, model)

# ========== STEP 5: Compare Different Sequences ==========
def compare_sequences(sequences, word2idx, model):
    """Compare hidden state trajectories for multiple sequences"""
    from sklearn.decomposition import PCA
    
    all_states = []
    all_labels = []
    
    for seq in sequences:
        tokens = seq.lower().split()
        indices = [word2idx.get(w, 0) for w in tokens]
        
        vocab_size = len(word2idx)
        x = torch.zeros(1, len(indices), vocab_size)
        for t, idx in enumerate(indices):
            x[0, t, idx] = 1.0
        
        model.eval()
        with torch.no_grad():
            hidden_states, _ = model(x, return_all_states=True)
            final_state = hidden_states[0, -1, :].numpy()
            all_states.append(final_state)
            all_labels.append(seq[:20] + '...' if len(seq) > 20 else seq)
    
    # PCA on final states
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(all_states))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100)
    for i, label in enumerate(all_labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Final Hidden States for Different Sequences')
    plt.show()

# Compare multiple sentences
sequences = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat ran on the mat"
]
compare_sequences(sequences, word2idx, model)
```

**Visualization Outputs:**

1. **Heatmap**: Shows all hidden units (y-axis) across time steps (x-axis)
   - Reveals which neurons activate for different inputs
   - Shows temporal patterns

2. **Line Plot**: Selected hidden unit activations over time
   - Shows how individual neurons respond
   - Useful for finding pattern-detecting neurons

3. **PCA Trajectory**: Hidden states projected to 2D
   - Shows how state evolves through "state space"
   - Similar sequences should follow similar paths

**Interview Tip:**
Key insight: Hidden state visualization reveals what patterns the RNN learns. Similar inputs should produce similar trajectories, and you can identify which neurons respond to specific features.

---
