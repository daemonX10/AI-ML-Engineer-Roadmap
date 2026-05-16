# Deep Learning Interview Questions - Scenario-Based Questions

## Question 1: Architecture and applications of LSTMs
- [ ] Done

### Architecture
LSTM (Long Short-Term Memory) addresses vanishing gradients with gating mechanisms.

### Gates

| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget** | $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ | What to forget from cell state |
| **Input** | $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ | What new info to store |
| **Output** | $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ | What to output |

### Cell State Update
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

### Applications

| Application | Why LSTM |
|-------------|----------|
| **Language Modeling** | Long-range word dependencies |
| **Speech Recognition** | Sequential audio processing |
| **Time Series** | Capture temporal patterns |
| **Machine Translation** | Encoder-decoder architectures |

---

## Question 2: Role of learning rate and its impact
- [ ] Done

### Impact

| Learning Rate | Effect |
|---------------|--------|
| **Too High** | Overshoots minimum, unstable |
| **Too Low** | Very slow convergence, stuck |
| **Just Right** | Steady progress to minimum |

### Strategies

| Strategy | Description |
|----------|-------------|
| **Learning Rate Schedule** | Decrease over time |
| **Warmup** | Start low, increase, then decrease |
| **Adaptive (Adam)** | Per-parameter adjustment |
| **Cyclical** | Oscillate between bounds |

### Best Practice
- Start with 0.001 for Adam
- Use learning rate finder
- Monitor training curves
- Reduce on plateau

---

## Question 3: Importance of data augmentation in deep learning
- [ ] Done

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **More Training Data** | Virtually increases dataset size |
| **Regularization** | Prevents overfitting |
| **Invariance** | Model learns to ignore irrelevant transformations |
| **Robustness** | Handles real-world variations |

### Techniques by Domain

| Domain | Augmentations |
|--------|---------------|
| **Images** | Flip, rotate, crop, color jitter |
| **Text** | Synonym replacement, back-translation |
| **Audio** | Time stretch, pitch shift, noise |

### When Critical
- Small datasets
- Complex models
- Overfitting observed

---

## Question 4: Style transfer in deep learning
- [ ] Done

### Concept
Style transfer combines the content of one image with the artistic style of another using neural networks.

### How It Works
1. Use pretrained CNN (VGG) to extract features
2. **Content Loss**: Match feature maps at deep layer
3. **Style Loss**: Match Gram matrices across layers
4. Optimize generated image to minimize both losses

### Loss Function
$$L_{total} = \alpha L_{content} + \beta L_{style}$$

### Applications
- Artistic filters
- Photo enhancement
- Game graphics
- Design tools

---

## Question 5: Deep learning in NLP
- [ ] Done

### Evolution

| Era | Approach | Models |
|-----|----------|--------|
| **Pre-2017** | RNN/LSTM | Seq2Seq |
| **2017-2018** | Attention | Transformer |
| **2018-Present** | Pretrained LMs | BERT, GPT |

### Key Applications

| Task | Model |
|------|-------|
| **Classification** | Fine-tuned BERT |
| **Generation** | GPT family |
| **Translation** | Transformer |
| **QA** | BERT variants |
| **Summarization** | T5, BART |

### Impact
- Eliminated feature engineering
- Transfer learning possible
- State-of-the-art across all tasks

---

## Question 6: Deep learning for self-driving cars
- [ ] Done

### Architecture Components

| Module | Deep Learning Role |
|--------|-------------------|
| **Perception** | Object detection (YOLO), segmentation |
| **Localization** | Visual odometry, SLAM |
| **Prediction** | Trajectory forecasting |
| **Planning** | End-to-end learning or separate module |

### Design Considerations
- **Real-time**: Must run at 30+ FPS
- **Redundancy**: Multiple sensors (camera, LiDAR, radar)
- **Safety**: Uncertainty estimation
- **Weather**: Robust to conditions

### Sensor Fusion
- Multi-modal networks combining camera + LiDAR
- Late fusion vs early fusion approaches

---

## Question 7: Deep learning for medical image diagnosis
- [ ] Done

### Strategy

| Step | Implementation |
|------|----------------|
| **Data** | Partner with hospitals, handle privacy |
| **Preprocessing** | Standardize, augment heavily |
| **Model** | CNN (ResNet, DenseNet) with transfer learning |
| **Output** | Confidence scores, not just labels |
| **Validation** | Extensive clinical validation |

### Considerations
- **Imbalanced Data**: Disease cases rare
- **Interpretability**: Doctors need to understand
- **Regulation**: FDA approval needed
- **Integration**: Into clinical workflow

### Techniques
- Grad-CAM for visualization
- Ensemble for reliability
- Uncertainty quantification

---

## Question 8: Neural network for stock price prediction
- [ ] Done

### Approach

| Component | Choice |
|-----------|--------|
| **Features** | OHLCV, technical indicators, sentiment |
| **Model** | LSTM, Transformer, or hybrid |
| **Output** | Next day price, direction, or returns |
| **Validation** | Walk-forward (not random split) |

### Architecture
```python
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
```

### Important Caveats
- Markets are efficient (hard to beat)
- Avoid look-ahead bias
- Transaction costs matter
- Use walk-forward validation only

---

## Question 9: Real-time object detection in videos
- [ ] Done

### Approach

| Aspect | Choice |
|--------|--------|
| **Model** | YOLO (v5, v8) for speed |
| **Tracking** | DeepSORT for continuity |
| **Hardware** | GPU required |
| **Optimization** | TensorRT, lower resolution |

### Pipeline
```
Frame → Resize → YOLO → NMS → Tracker → Display
```

### Speed Optimization
- Batch processing if possible
- Frame skipping
- Model quantization
- Lower input resolution

### Key Metrics
- FPS (need 30+ for real-time)
- mAP for accuracy
- Latency (end-to-end)

---

## Question 10: Deep learning for chatbots
- [ ] Done

### Architecture Options

| Approach | Description |
|----------|-------------|
| **Retrieval** | Match query to predefined responses |
| **Generative** | Generate responses (GPT-based) |
| **Hybrid** | Combine both |

### Key Components
- **Intent Classification**: What user wants
- **Entity Extraction**: Key information
- **Dialogue Management**: Context tracking
- **Response Generation**: Natural responses

### Modern Approach
Fine-tune large language model (GPT, LLaMA) on dialogue data with:
- Instruction tuning
- RLHF for safety/quality

---

## Question 11: Handling high variance in models
- [ ] Done

### Definition
High variance = overfitting = model memorizes training data

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **More Data** | Better coverage of distribution |
| **Data Augmentation** | Artificial diversity |
| **Regularization** | L1/L2, dropout |
| **Early Stopping** | Stop before overfitting |
| **Simpler Model** | Reduce capacity |
| **Ensemble** | Average predictions |
| **Cross-Validation** | Robust model selection |

### Diagnosis
- Large gap between train and validation accuracy
- Validation loss increases while training loss decreases

---

## Question 12: Precision-Recall curves and their importance
- [ ] Done

### Why Use PR Curves

| Scenario | Preference |
|----------|------------|
| **Imbalanced Data** | PR curve better than ROC |
| **Focus on Positives** | When negatives are abundant |
| **Different Thresholds** | Visualize trade-off |

### Key Metrics

| Metric | Formula |
|--------|---------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **AP** | Area under PR curve |

### When to Use
- Fraud detection (few positives)
- Disease screening
- Information retrieval
- When false positives and false negatives have different costs

### Interpretation
- High precision: When positive, usually correct
- High recall: Finds most actual positives
- Trade-off: Can't maximize both

---

## Question 13: Steps to create a recommendation system using deep learning
- [ ] Done

### Architecture Approaches

| Approach | Description |
|----------|-------------|
| **Neural Collaborative Filtering** | Learn user-item interactions |
| **Content-Based** | Encode item features |
| **Two-Tower** | Separate user and item encoders |
| **Sequential** | Model user history with RNN/Transformer |

### Pipeline
1. **Data**: User-item interactions, item features
2. **Embedding**: Learn user and item embeddings
3. **Interaction**: Combine embeddings (dot product, MLP)
4. **Training**: Binary cross-entropy or ranking loss
5. **Serving**: Approximate nearest neighbor for retrieval

### Python Code Example
```python
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
    def forward(self, user_id, item_id):
        user_vec = self.user_embed(user_id)
        item_vec = self.item_embed(item_id)
        return (user_vec * item_vec).sum(dim=1)  # Dot product
```

---

## Question 14: Approach for sentiment analysis on social media using deep learning
- [ ] Done

### Pipeline

| Step | Implementation |
|------|----------------|
| **Preprocessing** | Handle mentions, hashtags, emojis |
| **Tokenization** | Subword tokenization (BPE) |
| **Model** | Fine-tuned BERT or DistilBERT |
| **Output** | Positive/Negative/Neutral |

### Preprocessing
```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'http\S+', '<URL>', text)
    return text
```

### Model Approach
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This product is amazing!")
# {'label': 'POSITIVE', 'score': 0.99}
```

---
