# Neural Networks Interview Questions - Scenario-Based Questions

## Question 1: Importance of data augmentation in training neural networks
- [ ] Done

### Why It's Important
Data augmentation artificially increases training data diversity by applying transformations, helping neural networks generalize better and reducing overfitting.

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Reduces Overfitting** | More diverse training examples |
| **Improves Generalization** | Model sees variations it might encounter |
| **Cost-Effective** | No need to collect more data |
| **Regularization Effect** | Acts as implicit regularizer |

### Common Techniques (Images)
- Rotation, flipping, cropping
- Color jittering, brightness adjustment
- Random erasing, cutout
- Mixup, CutMix

### When to Use
- Limited training data
- Model overfitting
- Want more robust predictions

---

## Question 2: GANs and their typical applications
- [ ] Done

### Architecture
- **Generator**: Creates fake samples from noise
- **Discriminator**: Distinguishes real from fake
- Trained adversarially until generator fools discriminator

### Applications

| Application | Description |
|-------------|-------------|
| **Image Generation** | StyleGAN for realistic faces |
| **Image-to-Image** | Pix2Pix, CycleGAN |
| **Super-Resolution** | SRGAN for image upscaling |
| **Data Augmentation** | Generate synthetic training data |
| **Art/Design** | Creative content generation |

### Challenges
- Training instability
- Mode collapse
- Hard to evaluate quality

---

## Question 3: SGD vs Mini-batch Gradient Descent
- [ ] Done

### Comparison

| Aspect | SGD (1 sample) | Mini-batch (32-256 samples) |
|--------|----------------|----------------------------|
| **Update Frequency** | After each sample | After each batch |
| **Noise** | Very high | Moderate |
| **Convergence** | Noisy, can escape local minima | More stable |
| **GPU Utilization** | Poor | Good |
| **Memory** | Low | Higher |

### Recommendation
Mini-batch is the default for deep learning:
- Balances noise (helps escape) with stability
- Efficient GPU utilization
- Typical sizes: 32, 64, 128, 256

---

## Question 4: Tackling overfitting in deep neural networks
- [ ] Done

### Strategy (Priority Order)

| Step | Action |
|------|--------|
| **1** | Collect more data (if possible) |
| **2** | Data augmentation |
| **3** | Early stopping |
| **4** | Dropout (0.2-0.5) |
| **5** | L2 regularization (weight decay) |
| **6** | Reduce model capacity |
| **7** | Batch normalization |

### Diagnosis
- Training accuracy >> Validation accuracy
- Validation loss increases while training loss decreases

### Code Example
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
```

---

## Question 5: Implications of batch size on training
- [ ] Done

### Effects

| Batch Size | Effect |
|------------|--------|
| **Small (16-32)** | More noise, better generalization, slower |
| **Large (256+)** | Less noise, faster training, may generalize worse |

### Trade-offs

| Aspect | Small Batch | Large Batch |
|--------|-------------|-------------|
| **Gradient Estimate** | Noisy | Accurate |
| **Training Speed** | Slower | Faster |
| **Memory** | Less | More |
| **Generalization** | Often better | May need tuning |

### Practical Tips
- Start with 32 or 64
- Increase if GPU memory allows
- Scale learning rate with batch size
- Large batch may need learning rate warmup

---

## Question 6: Gradient tracking loss and solutions
- [ ] Done

### Problem: Vanishing/Exploding Gradients
Gradients become too small or too large during backpropagation, preventing learning.

### Solutions

| Issue | Solution |
|-------|----------|
| **Vanishing Gradients** | ReLU activation, skip connections, batch norm |
| **Exploding Gradients** | Gradient clipping, proper initialization |
| **Both** | LSTM/GRU for sequences, careful architecture |

### Code Example
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Proper initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

---

## Question 7: CNNs vs Capsule Networks
- [ ] Done

### Comparison

| Aspect | CNN | Capsule Network |
|--------|-----|-----------------|
| **Unit Output** | Scalar (feature presence) | Vector (pose information) |
| **Spatial Hierarchy** | Learned implicitly | Explicit part-whole |
| **Translation** | Invariance (pooling) | Equivariance |
| **Maturity** | Well-established | Still experimental |
| **Compute** | Efficient | More expensive |

### Key Difference
Capsules preserve spatial relationships between features (e.g., knowing a face has eyes above nose, not just that eyes and nose exist).

---

## Question 8: Challenges of sequence modeling
- [ ] Done

### Main Challenges

| Challenge | Description |
|-----------|-------------|
| **Long-range Dependencies** | Remembering distant information |
| **Variable Length** | Different sequence lengths |
| **Vanishing Gradients** | RNNs struggle with long sequences |
| **Sequential Processing** | Slow (can't parallelize) |
| **Alignment** | For tasks like translation |

### Solutions
- LSTM/GRU for vanishing gradients
- Attention mechanisms for long-range
- Transformers for parallelization
- Padding/masking for variable length

---

## Question 9: Sequence padding importance and effects
- [ ] Done

### Why Padding is Needed
Batching requires same-length sequences; padding fills shorter sequences to match the longest.

### Effects on Performance

| Effect | How to Handle |
|--------|---------------|
| **Wasted Computation** | Use masking to ignore pad tokens |
| **Corrupted Statistics** | Mask in batch norm/attention |
| **Memory** | Truncate very long sequences |

### Implementation
```python
# Padding with masking
padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
# Create mask
mask = (padded != 0)
```

---

## Question 10: Reducing inference time in production
- [ ] Done

### Strategies

| Technique | Speedup |
|-----------|---------|
| **Quantization** | 2-4x (INT8) |
| **Pruning** | Remove unimportant weights |
| **Knowledge Distillation** | Train smaller student model |
| **TensorRT/ONNX Runtime** | Optimized inference |
| **Batch Inference** | Process multiple inputs |
| **Hardware** | GPU, TPU, specialized accelerators |

### Steps
1. Profile to find bottlenecks
2. Try quantization first (easy wins)
3. Consider model architecture changes
4. Use optimized serving frameworks

---

## Question 11: Neural networks in NLP
- [ ] Done

### Applications

| Task | Model Type |
|------|------------|
| **Text Classification** | BERT, DistilBERT |
| **Named Entity Recognition** | BiLSTM-CRF, BERT |
| **Machine Translation** | Transformer (Seq2Seq) |
| **Question Answering** | BERT, RoBERTa |
| **Summarization** | T5, BART |

### Evolution
1. RNNs/LSTMs (sequence modeling)
2. Attention mechanisms
3. Transformers (BERT, GPT)
4. Large Language Models

---

## Question 12: Real-time object detection in video
- [ ] Done

### Architecture Choice
**YOLO (You Only Look Once)** - single-pass detection for speed

### Design Considerations

| Aspect | Implementation |
|--------|----------------|
| **Speed** | Need 30+ FPS |
| **Model** | YOLOv5/v8, SSD |
| **Tracking** | SORT, DeepSORT for continuity |
| **Hardware** | GPU required |

### Pipeline
```
Video Frame → Resize → YOLO → NMS → Tracking → Output
```

### Optimizations
- Lower resolution for speed
- Skip frames if needed
- Use TensorRT for inference

---

## Question 13: Architecture for automatic speech recognition
- [ ] Done

### Modern Pipeline

| Stage | Implementation |
|-------|----------------|
| **Input** | Raw waveform or spectrogram |
| **Encoder** | CNN + Transformer (Wav2Vec) |
| **Decoder** | CTC or attention-based |
| **Output** | Text transcription |

### Architecture Options
- **End-to-end**: Wav2Vec 2.0, Whisper
- **Hybrid**: CNN + RNN + CTC
- **Attention**: Listen-Attend-Spell

### Key Components
- Mel spectrogram features
- Subword tokenization (BPE)
- Language model integration (beam search)

---

## Question 14: Recent advances in neural network architectures
- [ ] Done

### Notable Advances (2020-2024)

| Advance | Description |
|---------|-------------|
| **Vision Transformers (ViT)** | Transformers for images |
| **Mixture of Experts (MoE)** | Sparse, conditional computation |
| **State Space Models (Mamba)** | Alternative to transformers |
| **Diffusion Models** | DALL-E, Stable Diffusion |
| **Multimodal Models** | GPT-4V, CLIP |

### Impact
- ViT shows attention works for images
- MoE enables massive models efficiently
- Diffusion models dominate image generation

---

## Question 15: Neural Architecture Search (NAS)
- [ ] Done

### Definition
Automated process of designing neural network architectures using algorithms rather than manual trial-and-error.

### Approaches

| Method | Description |
|--------|-------------|
| **Reinforcement Learning** | Controller learns to propose architectures |
| **Evolutionary** | Mutate and select best architectures |
| **Differentiable** | Make architecture choices differentiable |
| **Weight Sharing** | Share weights across candidates |

### Significance
- Discovers novel architectures
- Reduces human bias
- EfficientNet, NASNet from NAS

---

## Question 16: Energy-efficient neural networks
- [ ] Done

### Why Important
- Environmental concerns (carbon footprint)
- Edge deployment (battery-powered devices)
- Cost reduction (cloud compute)

### Techniques

| Technique | Energy Savings |
|-----------|----------------|
| **Quantization** | Reduce precision (FP32→INT8) |
| **Pruning** | Remove unnecessary weights |
| **Efficient Architectures** | MobileNet, EfficientNet |
| **Early Exit** | Stop computation when confident |
| **Knowledge Distillation** | Smaller student models |

---

## Question 17: Neural networks in autonomous vehicle systems
- [ ] Done

### Applications

| Task | Neural Network Role |
|------|---------------------|
| **Perception** | Object detection, segmentation |
| **Localization** | Visual odometry, mapping |
| **Prediction** | Trajectory forecasting |
| **Planning** | Route planning, decision making |

### Architecture
Multi-task network processing:
- Multiple camera views
- LiDAR point clouds
- Radar data

### Key Requirements
- Real-time inference
- Robustness to weather/lighting
- Uncertainty quantification
- Redundancy for safety

---
