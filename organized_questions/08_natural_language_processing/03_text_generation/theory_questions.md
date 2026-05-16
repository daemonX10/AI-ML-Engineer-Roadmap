# Text Generation (Seq2Seq, Translation, QA, Summarization) - Theory Questions

## Core Questions

## Question 1
- [ ] Done
**What is the encoder-decoder architecture and how does it enable sequence-to-sequence learning?**

**Answer:**

**Definition:**
Encoder-decoder is a neural architecture where an encoder processes input sequence into a fixed context representation, and a decoder generates output sequence from that context. Enables variable-length input to variable-length output mapping for translation, summarization, etc.

**Architecture:**

```
Input: "Hello world" → [Encoder] → Context Vector → [Decoder] → "Bonjour monde"
```

| Component | Role |
|-----------|------|
| Encoder | Process input, create context representation |
| Context | Fixed-size summary of input |
| Decoder | Generate output token by token |

**How It Works:**
1. Encoder (LSTM/Transformer) reads input tokens
2. Final hidden state becomes context vector
3. Decoder uses context to generate output one token at a time
4. Each decoder step conditions on previous output + context

**Python Code Example:**
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell  # Context

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs = []
        for t in range(trg.size(1)):
            inp = trg[:, t:t+1]
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs.append(output)
        return torch.cat(outputs, dim=1)
```

**Interview Tips:**
- Bottleneck: fixed-size context loses information for long sequences
- Attention mechanism solves this bottleneck
- Modern: Transformers are encoder-decoder without recurrence
- BART, T5, mBART are pre-trained encoder-decoder models

---

## Question 2
- [ ] Done
**Explain the attention mechanism in seq2seq models. Why was it a breakthrough?**

**Answer:**

**Definition:**
Attention allows the decoder to dynamically focus on relevant parts of the input at each generation step, instead of relying on a single fixed context vector. It computes weighted sum of encoder states, where weights depend on decoder's current state.

**Why It's a Breakthrough:**
- Solves information bottleneck of fixed context
- Handles long sequences without degradation
- Provides interpretable alignment (which input → which output)
- Foundation for Transformer architecture

**How It Works:**

$$\alpha_{t,i} = \frac{\exp(score(s_t, h_i))}{\sum_j \exp(score(s_t, h_j))}$$
$$c_t = \sum_i \alpha_{t,i} h_i$$

where $s_t$ = decoder state, $h_i$ = encoder states, $c_t$ = context for step t.

**Score Functions:**

| Type | Formula |
|------|--------|
| Dot | $s_t^T h_i$ |
| General | $s_t^T W h_i$ |
| Concat (Bahdanau) | $v^T \tanh(W[s_t; h_i])$ |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]
        
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concat decoder state with each encoder output
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Softmax to get weights
        weights = F.softmax(attention, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.attention = Attention(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
        context, weights = self.attention(hidden.squeeze(0), encoder_outputs)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell, weights
```

**Interview Tips:**
- Bahdanau (2014) introduced attention for NMT
- Attention weights are interpretable - show alignment
- "Attention is All You Need" removed recurrence entirely
- Cross-attention: decoder attending to encoder (seq2seq)
- Self-attention: sequence attending to itself (Transformer)

---

## Question 3
- [ ] Done
**What is the Transformer architecture? Explain self-attention and multi-head attention.**

**Answer:**

**Definition:**
Transformer is a neural architecture using only attention mechanisms (no recurrence or convolution). Self-attention lets each token attend to all other tokens. Multi-head attention runs multiple attention functions in parallel to capture different relationship types.

**Key Components:**

| Component | Purpose |
|-----------|--------|
| Self-attention | Token-to-token relationships |
| Multi-head attention | Multiple attention patterns |
| Feed-forward | Position-wise transformations |
| Layer norm | Stabilize training |
| Positional encoding | Inject position information |

**Self-Attention (Scaled Dot-Product):**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q (Query): what to look for
- K (Key): what's available
- V (Value): what to retrieve
- Scale by $\sqrt{d_k}$ to prevent large dot products

**Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can learn different relationship patterns.

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)
```

**Interview Tips:**
- Transformer enables parallelization (no sequential dependency)
- 8 heads typical for base models, 16 for large
- BERT = encoder-only, GPT = decoder-only, T5/BART = encoder-decoder
- Positional encoding needed since no inherent position info

---

## Question 4
- [ ] Done
**What is beam search decoding and how does it compare to greedy and sampling methods?**

**Answer:**

**Definition:**
Beam search maintains top-k candidates (beams) at each step, exploring multiple paths simultaneously. Greedy picks only the best token at each step. Sampling introduces randomness for diversity. Choice affects quality vs diversity vs speed.

**Comparison:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Greedy | Pick argmax at each step | Fast | Misses global optimum |
| Beam Search | Keep top-k candidates | Better quality | Repetitive, slow |
| Sampling | Sample from distribution | Diverse | Less coherent |
| Top-k Sampling | Sample from top-k tokens | Balanced | Needs tuning |
| Nucleus (Top-p) | Sample from top cumulative prob | Adaptive diversity | Needs tuning |

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

def greedy_decode(model, src, max_len, sos_token, eos_token):
    """Greedy: always pick highest probability token"""
    encoder_out = model.encode(src)
    decoder_input = torch.tensor([[sos_token]])
    
    for _ in range(max_len):
        logits = model.decode(decoder_input, encoder_out)
        next_token = logits[:, -1, :].argmax(dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_token:
            break
    
    return decoder_input

def beam_search(model, src, beam_size=5, max_len=50):
    """Beam search: maintain top-k candidates"""
    encoder_out = model.encode(src)
    
    # Initialize with SOS token
    beams = [(torch.tensor([[sos_token]]), 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_token:
                new_beams.append((seq, score))
                continue
            
            logits = model.decode(seq, encoder_out)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            top_probs, top_indices = log_probs.topk(beam_size)
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_beams.append((new_seq, score + prob.item()))
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return beams[0][0]  # Return best sequence

def sample_decode(model, src, temperature=1.0, top_p=0.9):
    """Nucleus (top-p) sampling"""
    encoder_out = model.encode(src)
    decoder_input = torch.tensor([[sos_token]])
    
    for _ in range(max_len):
        logits = model.decode(decoder_input, encoder_out)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        
        # Sample
        idx = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[0, idx]
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
    
    return decoder_input
```

**Interview Tips:**
- Beam search for translation (quality matters)
- Sampling for creative generation (diversity matters)
- Temperature controls randomness (higher = more random)
- Beam size 4-6 typical for translation
- Length penalty prevents short outputs

---

## Question 5
- [ ] Done
**Explain BLEU score and its limitations for evaluating machine translation.**

**Answer:**

**Definition:**
BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated translation and reference translations. Scores 0-1, where 1 is perfect match. Includes brevity penalty. Standard metric for machine translation evaluation.

**Formula:**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$ = modified n-gram precision
- $BP$ = brevity penalty (penalizes short outputs)
- $w_n$ = weights (typically uniform 1/4 for n=1..4)

**How It Works:**

| Component | Purpose |
|-----------|--------|
| N-gram precision | Count matching n-grams |
| Modified precision | Clip counts to reference max |
| Brevity penalty | Penalize short translations |
| Geometric mean | Combine 1-4 gram precisions |

**Limitations:**

| Limitation | Description |
|------------|-------------|
| Ignores meaning | Synonyms get no credit |
| Order insensitive | Within n-gram only |
| Requires exact match | "car" vs "automobile" = 0 |
| Multiple references needed | Single reference insufficient |
| Doesn't correlate well | At sentence level |

**Python Code Example:**
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from sacrebleu import corpus_bleu as sacre_bleu

# Simple example
reference = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

# Sentence-level BLEU
score = sentence_bleu(reference, candidate)
print(f"Sentence BLEU: {score:.4f}")

# Corpus-level BLEU (more reliable)
references = [[['the', 'quick', 'brown', 'fox']],
              [['jumps', 'over', 'the', 'lazy', 'dog']]]
hypotheses = [['the', 'fast', 'brown', 'fox'],
              ['leaps', 'over', 'the', 'lazy', 'dog']]

corpus_score = corpus_bleu(references, hypotheses)
print(f"Corpus BLEU: {corpus_score:.4f}")

# SacreBLEU (standardized)
refs = [["The quick brown fox jumps over the lazy dog."]]
hyps = ["The fast brown fox leaps over the lazy dog."]
sacre_score = sacre_bleu(hyps, refs)
print(f"SacreBLEU: {sacre_score.score:.2f}")
```

**Alternatives:**

| Metric | Advantage |
|--------|----------|
| chrF | Character-level, better for morphological languages |
| METEOR | Includes synonyms, stems |
| BERTScore | Semantic similarity via embeddings |
| COMET | Learned metric, best correlation |

**Interview Tips:**
- Always use corpus-level BLEU, not sentence-level
- SacreBLEU for reproducible scores
- COMET correlates better with human judgment
- BLEU of 30+ is decent, 40+ is strong
- Different tokenization = different BLEU scores

---

## Question 6
- [ ] Done
**What is the difference between extractive and abstractive summarization?**

**Answer:**

**Definition:**
Extractive summarization selects and combines existing sentences from source document. Abstractive summarization generates new sentences, potentially paraphrasing and synthesizing information. Abstractive is more flexible but harder and prone to hallucination.

**Comparison:**

| Aspect | Extractive | Abstractive |
|--------|------------|-------------|
| Output | Original sentences | New sentences |
| Approach | Selection/ranking | Generation |
| Faithfulness | High (copies source) | Risk of hallucination |
| Compression | Limited | High |
| Fluency | May be choppy | More natural |
| Models | BERT + classifier | BART, T5, Pegasus |

**Python Code Example:**
```python
# Extractive: TextRank-based
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def extractive_summarize(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join([str(sentence) for sentence in summary])

# Extractive with BERT: sentence scoring
class BertExtractiveSummarizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def summarize(self, text, ratio=0.3):
        sentences = sent_tokenize(text)
        embeddings = self.model.encode(sentences)
        
        # Score by similarity to document centroid
        centroid = embeddings.mean(axis=0)
        scores = [np.dot(emb, centroid) for emb in embeddings]
        
        # Select top sentences
        n_select = max(1, int(len(sentences) * ratio))
        top_indices = np.argsort(scores)[-n_select:]
        top_indices = sorted(top_indices)  # Maintain order
        
        return ' '.join([sentences[i] for i in top_indices])

# Abstractive with transformers
from transformers import pipeline

abstractive_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"  # or "t5-base", "google/pegasus-xsum"
)

text = "Long article text here..."
summary = abstractive_summarizer(text, max_length=130, min_length=30)
print(summary[0]['summary_text'])
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Legal/medical (accuracy critical) | Extractive |
| News headlines | Abstractive |
| Multi-document | Abstractive |
| Limited compute | Extractive |

**Interview Tips:**
- Extractive is safer (no hallucination)
- Abstractive produces more natural summaries
- Pegasus is state-of-art for abstractive
- Hybrid: extract then abstract
- Always check for factual consistency in abstractive

---

## Question 7
- [ ] Done
**How does reading comprehension QA (SQuAD-style) differ from open-domain QA?**

**Answer:**

**Definition:**
Reading comprehension QA extracts answers from a given passage (context provided). Open-domain QA must first find relevant passages from a large corpus, then extract answers. Open-domain is much harder, requiring retrieval + reading.

**Comparison:**

| Aspect | Reading Comprehension | Open-Domain |
|--------|----------------------|-------------|
| Context | Given | Must retrieve |
| Corpus | Single passage | Millions of documents |
| Pipeline | Reader only | Retriever + Reader |
| Example | SQuAD | Natural Questions, TriviaQA |
| Complexity | Easier | Much harder |

**Architecture:**

```
Reading Comprehension:
Question + Passage → [Reader Model] → Answer span

Open-Domain:
Question → [Retriever] → Top-k passages → [Reader] → Answer
```

**Python Code Example:**
```python
# Reading Comprehension (SQuAD-style)
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
question = "When was the Eiffel Tower built?"

result = qa_pipeline(question=question, context=context)
print(result)  # {'answer': '1889', 'start': 62, 'end': 66, 'score': 0.98}

# Open-Domain QA with Retriever + Reader
from transformers import DPRQuestionEncoder, DPRContextEncoder
import faiss

class OpenDomainQA:
    def __init__(self):
        # Retriever: encode questions and passages
        self.q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.reader = pipeline("question-answering")
        self.index = None  # FAISS index of passage embeddings
        self.passages = []  # Corpus
    
    def build_index(self, passages):
        self.passages = passages
        # Encode all passages (in practice, use passage encoder)
        embeddings = self.encode_passages(passages)
        self.index = faiss.IndexFlatIP(768)  # Inner product
        self.index.add(embeddings)
    
    def answer(self, question, top_k=5):
        # Retrieve relevant passages
        q_emb = self.encode_question(question)
        scores, indices = self.index.search(q_emb, top_k)
        
        # Read from each passage
        answers = []
        for idx in indices[0]:
            passage = self.passages[idx]
            result = self.reader(question=question, context=passage)
            answers.append(result)
        
        # Return best answer
        return max(answers, key=lambda x: x['score'])
```

**Interview Tips:**
- SQuAD: answer always in passage, extractive
- Open-domain: may need to say "I don't know"
- DPR (Dense Passage Retrieval) for neural retrieval
- BM25 still strong baseline for retrieval
- RAG combines retriever with generator

---

## Question 8
- [ ] Done
**What is retrieval-augmented generation (RAG) and why is it important for QA?**

**Answer:**

**Definition:**
RAG combines retrieval (finding relevant documents) with generation (synthesizing answers). Instead of relying only on model's parameters, it retrieves external knowledge at inference time. Reduces hallucination, enables knowledge updates without retraining.

**Why It's Important:**

| Benefit | Description |
|---------|-------------|
| Factual grounding | Answers based on retrieved docs |
| Knowledge updates | Update docs, not model |
| Reduces hallucination | Can cite sources |
| Scalable | Knowledge in index, not parameters |
| Interpretable | Show retrieved context |

**RAG Architecture:**

```
Query → [Retriever] → Top-k Documents → [Generator] → Answer
              ↑                              ↑
         Document Index              LLM (GPT, BART, T5)
```

**Python Code Example:**
```python
# RAG with LangChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Build vector store from documents
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",  # Combine all docs
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# Query
answer = qa_chain.run("What is the capital of France?")

# Manual RAG implementation
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = None  # Your LLM
        self.docs = []
        self.doc_embeddings = None
    
    def index(self, documents):
        self.docs = documents
        self.doc_embeddings = self.encoder.encode(documents)
    
    def retrieve(self, query, k=3):
        query_emb = self.encoder.encode([query])
        scores = np.dot(self.doc_embeddings, query_emb.T).flatten()
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [self.docs[i] for i in top_k_idx]
    
    def answer(self, question):
        contexts = self.retrieve(question)
        
        prompt = f"""Answer based on the following contexts:

{chr(10).join(contexts)}

Question: {question}
Answer:"""
        
        return self.llm.generate(prompt)
```

**Interview Tips:**
- RAG reduces hallucination by grounding in retrieved docs
- Vector databases: FAISS, Pinecone, Weaviate, Chroma
- Chunk documents appropriately (not too long/short)
- Retriever quality crucial - bad retrieval = bad answers
- Can combine with web search for real-time info

---

## Question 9
- [ ] Done
**Explain the teacher forcing technique in training seq2seq models.**

**Answer:**

**Definition:**
Teacher forcing feeds ground truth tokens as decoder input during training, instead of model's own predictions. Speeds up training convergence but creates mismatch with inference (exposure bias) where model uses its own predictions.

**How It Works:**

| Mode | Decoder Input |
|------|---------------|
| Teacher forcing | Ground truth previous token |
| Autoregressive | Model's previous prediction |

**Training:**
```
Target: "I love NLP"

With teacher forcing:
  Step 1: Input=<SOS>, Target="I"
  Step 2: Input="I" (ground truth), Target="love"
  Step 3: Input="love" (ground truth), Target="NLP"

Without:
  Step 1: Input=<SOS>, Predict="I"
  Step 2: Input="I" (predicted), Predict="love"
  Step 3: Input="love" (predicted), Predict="NLP"
```

**Python Code Example:**
```python
import torch
import random

def train_step(model, src, trg, teacher_forcing_ratio=0.5):
    """
    Train with scheduled teacher forcing
    """
    encoder_outputs, hidden = model.encoder(src)
    
    # First input is SOS token
    decoder_input = trg[:, 0:1]
    outputs = []
    
    for t in range(1, trg.size(1)):
        output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs.append(output)
        
        # Teacher forcing: use ground truth or prediction
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:
            decoder_input = trg[:, t:t+1]  # Ground truth
        else:
            decoder_input = output.argmax(dim=-1)  # Prediction
    
    outputs = torch.cat(outputs, dim=1)
    loss = criterion(outputs.view(-1, vocab_size), trg[:, 1:].view(-1))
    return loss

# Scheduled sampling: decrease teacher forcing over time
class ScheduledSampling:
    def __init__(self, initial_ratio=1.0, decay=0.99, min_ratio=0.1):
        self.ratio = initial_ratio
        self.decay = decay
        self.min_ratio = min_ratio
    
    def step(self):
        self.ratio = max(self.min_ratio, self.ratio * self.decay)
    
    def get_ratio(self):
        return self.ratio

# Usage
scheduler = ScheduledSampling()
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, src, trg, scheduler.get_ratio())
    scheduler.step()  # Decrease teacher forcing ratio
```

**Exposure Bias Problem:**
- Training: always sees correct previous tokens
- Inference: sees its own (possibly wrong) predictions
- Error compounds - wrong token leads to more wrong tokens

**Solutions:**
- Scheduled sampling: gradually reduce teacher forcing
- Curriculum learning: start easy, increase difficulty
- Reinforcement learning: train with own outputs

**Interview Tips:**
- Teacher forcing speeds training significantly
- Without it, early training is very slow
- Exposure bias is a real problem in practice
- Scheduled sampling is common compromise
- Modern LLMs still use teacher forcing (masked LM objectives)

---

## Question 10
- [ ] Done
**What are the key evaluation metrics for summarization (ROUGE, BERTScore, factual consistency)?**

**Answer:**

**Definition:**
ROUGE measures n-gram overlap with reference summaries. BERTScore uses semantic similarity via embeddings. Factual consistency metrics check if generated summary is factually accurate relative to source. Each captures different quality aspects.

**Key Metrics:**

| Metric | Measures | How |
|--------|----------|-----|
| ROUGE-1 | Unigram overlap | Recall/Precision of words |
| ROUGE-2 | Bigram overlap | Captures phrase similarity |
| ROUGE-L | Longest common subsequence | Word order |
| BERTScore | Semantic similarity | Embedding cosine similarity |
| Factual consistency | Correctness | NLI-based or QA-based |

**ROUGE Formula:**

$$\text{ROUGE-N Recall} = \frac{\text{Matching n-grams}}{\text{Total n-grams in reference}}$$

$$\text{ROUGE-N Precision} = \frac{\text{Matching n-grams}}{\text{Total n-grams in summary}}$$

$$\text{ROUGE-N F1} = \frac{2 \cdot P \cdot R}{P + R}$$

**Python Code Example:**
```python
# ROUGE score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat."
prediction = "The cat is sitting on the mat."

scores = scorer.score(reference, prediction)
print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# BERTScore - semantic similarity
from bert_score import score

references = ["The cat sat on the mat."]
candidates = ["A feline rested on the rug."]  # Same meaning, different words

P, R, F1 = score(candidates, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")  # High score despite different words

# Factual Consistency with NLI
from transformers import pipeline

nli_model = pipeline("text-classification", model="roberta-large-mnli")

def check_factual_consistency(source, summary):
    """Check if summary is entailed by source"""
    result = nli_model(f"{source} [SEP] {summary}")
    # Labels: entailment, neutral, contradiction
    return result[0]['label'] == 'ENTAILMENT'

# QA-based factual consistency (QuestEval)
def qa_factual_check(source, summary):
    """
    1. Generate questions from summary
    2. Answer questions using source
    3. Answer questions using summary
    4. Compare answers
    """
    # If answers match, summary is factually consistent
    pass
```

**When to Use Each:**

| Metric | Best For |
|--------|----------|
| ROUGE | Quick evaluation, comparing systems |
| BERTScore | Paraphrase-heavy summaries |
| Factual consistency | Abstractive summarization |

**Interview Tips:**
- ROUGE-2 most commonly reported
- BERTScore captures paraphrasing that ROUGE misses
- Factual consistency is crucial for abstractive
- Human evaluation still gold standard
- Report multiple metrics, not just one

---

## Interview Questions

## Question 11
- [ ] Done
**How do you handle sequence generation for tasks with extreme input/output length disparities?**

**Answer:**

**Definition:**
Length disparity means input is much longer/shorter than output. Examples: long document → short summary (compression), keyword → paragraph (expansion). Challenges include information selection, context preservation, and generation control.

**Strategies by Type:**

| Disparity | Challenge | Strategies |
|-----------|-----------|------------|
| Long → Short | Information selection | Hierarchical, chunking |
| Short → Long | Content expansion | Prompting, retrieval |
| Variable | Flexibility | Length conditioning |

**Python Code Example:**
```python
# Long input → Short output (Summarization)
def hierarchical_summarize(long_text, model, max_chunk=1024, final_length=100):
    """Hierarchical approach for very long documents"""
    # Split into chunks
    words = long_text.split()
    chunks = [' '.join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = model.generate(chunk, max_length=200)
        chunk_summaries.append(summary)
    
    # Combine and summarize again
    combined = ' '.join(chunk_summaries)
    final_summary = model.generate(combined, max_length=final_length)
    return final_summary

# Alternatively: Longformer for long inputs
from transformers import LongformerForConditionalGeneration

model = LongformerForConditionalGeneration.from_pretrained('allenai/led-base-16384')
# Handles up to 16K tokens

# Short input → Long output (Expansion)
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

def expand_with_retrieval(keywords, retrieved_context):
    prompt = f"""Based on these keywords: {keywords}

Context: {retrieved_context}

Write a detailed paragraph:"""
    return generator(prompt, max_length=500)

# Length conditioning
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Control output length via special tokens or max_length
def generate_with_length_control(text, target_length):
    # Some models support length control tokens
    input_text = f"summarize in {target_length} words: {text}"
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=target_length * 2)
    return tokenizer.decode(outputs[0])
```

**Interview Tips:**
- LED, Longformer handle long inputs (16K+ tokens)
- Hierarchical approaches for very long documents
- Length control tokens in some models
- Short → long often needs retrieval for content
- Consider sliding window with overlap for context

---

## Question 12
- [ ] Done
**What techniques prevent repetition and improve diversity in text generation?**

**Answer:**

**Definition:**
Repetition is a common failure mode where models repeat phrases or sentences. Diversity techniques encourage varied outputs. Solutions include n-gram blocking, penalty terms, and sampling strategies.

**Techniques:**

| Technique | Description |
|-----------|-------------|
| N-gram blocking | Prevent repeating n-grams |
| Repetition penalty | Reduce probability of repeated tokens |
| Top-k sampling | Sample from top-k tokens |
| Nucleus (top-p) | Sample from top cumulative probability |
| Temperature | Control randomness |
| Diverse beam search | Encourage different beams |

**Python Code Example:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors='pt')

# Without repetition handling (may repeat)
output_repetitive = model.generate(
    **inputs,
    max_length=100,
    do_sample=False  # Greedy
)

# With repetition penalty
output_no_repeat = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    repetition_penalty=1.2,  # Penalize repeated tokens
    no_repeat_ngram_size=3,  # Block repeating 3-grams
    top_p=0.9,               # Nucleus sampling
    temperature=0.8          # Some randomness
)

# Diverse beam search
output_diverse = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,
    num_beam_groups=5,
    diversity_penalty=0.5,  # Encourage different beams
    num_return_sequences=3  # Return multiple diverse outputs
)

# Custom n-gram blocking
def generate_with_ngram_block(model, input_ids, max_len, n=3):
    generated = input_ids.clone()
    seen_ngrams = set()
    
    for _ in range(max_len):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Block n-grams we've seen
        if generated.size(1) >= n - 1:
            last_ngram = tuple(generated[0, -(n-1):].tolist())
            for token_id in range(next_token_logits.size(-1)):
                potential_ngram = last_ngram + (token_id,)
                if potential_ngram in seen_ngrams:
                    next_token_logits[0, token_id] = float('-inf')
        
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        # Record new n-gram
        if generated.size(1) >= n:
            new_ngram = tuple(generated[0, -n:].tolist())
            seen_ngrams.add(new_ngram)
    
    return generated
```

**Interview Tips:**
- `no_repeat_ngram_size=3` is simple and effective
- `repetition_penalty=1.1-1.3` works well
- Top-p sampling better than top-k generally
- Temperature: 0.7-0.9 for balanced diversity
- Beam search tends toward repetition

---

## Question 13
- [ ] Done
**How do you implement controllable text generation (style, length, tone)?**

**Answer:**

**Definition:**
Controllable generation steers model output toward desired attributes (formal/casual tone, specific length, positive/negative sentiment). Approaches include control codes, prompting, fine-tuning, and steering during decoding.

**Control Dimensions:**

| Attribute | Example |
|-----------|--------|
| Style | Formal, casual, poetic |
| Tone | Positive, negative, neutral |
| Length | Short, medium, long |
| Topic | Sports, technology, politics |
| Persona | Expert, friendly, professional |

**Approaches:**

| Method | Description |
|--------|-------------|
| Control codes | Prepend tokens like "<FORMAL>" |
| Prompting | Describe desired style in prompt |
| CTRL model | Trained with control prefixes |
| PPLM | Perturb hidden states toward attribute |
| Fine-tuning | Train on style-specific data |

**Python Code Example:**
```python
# Approach 1: Prompt-based control (simplest)
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2-medium')

# Control via prompt
prompts = {
    'formal': "Write a formal business email: Dear Sir or Madam,",
    'casual': "Write a casual message: Hey! What's up,",
    'positive': "Write a positive review: This product is",
    'negative': "Write a negative review: I'm disappointed because"
}

for style, prompt in prompts.items():
    output = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    print(f"{style}: {output[:100]}...")

# Approach 2: Control codes (CTRL model style)
# During training, prepend control token
# At inference, use desired control token

class ControlledGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.control_tokens = {
            'formal': '<FORMAL>',
            'casual': '<CASUAL>',
            'short': '<SHORT>',
            'long': '<LONG>'
        }
    
    def generate(self, text, controls):
        # Prepend control tokens
        control_str = ' '.join([self.control_tokens[c] for c in controls])
        input_text = f"{control_str} {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])

# Approach 3: Length control via max_length and length penalty
output = model.generate(
    input_ids,
    max_length=50,          # Short output
    min_length=30,          # Ensure minimum
    length_penalty=1.5,     # Encourage brevity
    early_stopping=True
)

# Approach 4: Classifier-guided decoding
def guided_decoding(model, tokenizer, prompt, attribute_classifier, target_class):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    for _ in range(50):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # Score each possible continuation
        best_score = float('-inf')
        best_token = 0
        
        for token_id in logits.topk(50).indices[0]:
            candidate = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            text = tokenizer.decode(candidate[0])
            
            # Check attribute score
            attr_score = attribute_classifier(text)[target_class]
            combined_score = logits[0, token_id] + attr_score * 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_token = token_id
        
        input_ids = torch.cat([input_ids, best_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_ids[0])
```

**Interview Tips:**
- Prompting is simplest for modern LLMs
- Control codes require training/fine-tuning
- Length penalty in beam search for brevity
- PPLM for post-hoc attribute control
- InstructGPT/ChatGPT follow style instructions well

---

## Question 14
- [ ] Done
**What are the challenges of machine translation for low-resource language pairs?**

**Answer:**

**Definition:**
Low-resource translation involves language pairs with little parallel training data (e.g., English-Swahili, French-Yoruba). Challenges include limited data, domain mismatch, and lack of evaluation resources. Solutions involve transfer learning, multilingual models, and data augmentation.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Limited parallel data | Few translated sentence pairs |
| No pre-trained models | Language-specific models missing |
| Evaluation | Few test sets, no native speakers |
| Domain mismatch | Available data is religious/news |
| Morphological complexity | Many low-resource languages are morphologically rich |

**Solutions:**

| Solution | Approach |
|----------|----------|
| Multilingual models | M2M-100, mBART, NLLB |
| Transfer learning | Pre-train on high-resource, adapt |
| Back-translation | Generate synthetic parallel data |
| Cross-lingual transfer | Train on related language |
| Unsupervised NMT | Use monolingual data only |

**Python Code Example:**
```python
# Use multilingual model (works for 100+ languages)
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer

# NLLB (No Language Left Behind) - 200 languages
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate_low_resource(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example: English to Yoruba (low-resource)
translation = translate_low_resource(
    "Hello, how are you?",
    src_lang="eng_Latn",
    tgt_lang="yor_Latn"  # Yoruba
)

# Back-translation for data augmentation
def back_translate(texts, src_lang, pivot_lang):
    """
    Augment low-resource data via back-translation
    src -> pivot -> src creates paraphrased version
    """
    augmented = []
    for text in texts:
        # Forward translation
        pivot_text = translate(text, src_lang, pivot_lang)
        # Back translation
        back_text = translate(pivot_text, pivot_lang, src_lang)
        augmented.append(back_text)
    return augmented

# Cross-lingual transfer: train on related high-resource pair
# E.g., Train on Spanish-Portuguese, adapt to Portuguese-Galician
```

**Interview Tips:**
- NLLB from Meta supports 200 languages
- Multilingual models enable zero-shot for unseen pairs
- Back-translation is crucial data augmentation
- Related language transfer helps (Hindi → Urdu)
- Quality degrades but is better than nothing

---

## Question 15
- [ ] Done
**How do you handle domain adaptation in neural machine translation?**

**Answer:**

**Definition:**
Domain adaptation adjusts a general translation model for specific domains (medical, legal, technical). General models trained on news fail on domain-specific terminology. Techniques include fine-tuning, domain tags, and terminology injection.

**Domain Challenges:**

| Challenge | Example |
|-----------|--------|
| Terminology | "Stat" = immediately (medical) |
| Style | Legal formality, technical precision |
| Coverage | Rare terms not in training data |
| Consistency | Same term should translate consistently |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Fine-tuning | Continue training on domain data |
| Mixed training | Combine general + domain data |
| Domain tags | "<medical>" prefix |
| Terminology injection | Force glossary terms |
| Retrieval-augmented | Retrieve similar translations |

**Python Code Example:**
```python
# Approach 1: Fine-tune on domain data
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments

# Start with general model
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Fine-tune on medical parallel data
training_args = TrainingArguments(
    output_dir='./medical-mt',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Lower LR to preserve general knowledge
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=medical_dataset  # Domain-specific pairs
)
trainer.train()

# Approach 2: Terminology-constrained decoding
def translate_with_glossary(text, model, tokenizer, glossary):
    """Force glossary terms in translation"""
    # Replace terms before translation
    for src_term, tgt_term in glossary.items():
        text = text.replace(src_term, f"[TERM]{tgt_term}[/TERM]")
    
    # Translate
    inputs = tokenizer(text, return_tensors='pt')
    output = model.generate(**inputs)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Restore glossary terms
    # (In practice, use constrained beam search)
    return translation

glossary = {
    'myocardial infarction': 'Herzinfarkt',
    'blood pressure': 'Blutdruck'
}

# Approach 3: Domain tags
def domain_translate(text, domain, model, tokenizer):
    """Use domain tag prefix"""
    tagged_input = f"<{domain}> {text}"
    inputs = tokenizer(tagged_input, return_tensors='pt')
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
medical_translation = domain_translate(
    "The patient presents with acute chest pain.",
    domain="medical",
    model=domain_tagged_model,
    tokenizer=tokenizer
)
```

**Interview Tips:**
- Fine-tuning is most common approach
- Use low learning rate to avoid catastrophic forgetting
- Domain data is often limited - augment with back-translation
- Terminology consistency critical for professional translation
- Consider training domain classifier to route requests

---

## Question 16
- [ ] Done
**What techniques work best for real-time/simultaneous machine translation?**

**Answer:**

**Definition:**
Simultaneous translation translates while speaker is still talking, with minimal delay. Unlike offline translation, it must decide when to commit to translation before seeing full sentence. Key challenge: wait-k policy balancing latency vs quality.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Incomplete input | Must translate before sentence ends |
| Word order | Source/target may have different order |
| Latency requirement | Real-time constraint |
| Revision | May need to correct earlier outputs |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Wait-k | Wait for k source tokens, then translate |
| Adaptive wait | Learn when to read vs write |
| Incremental | Output partial translations |
| Re-translation | Revise previous output |

**Python Code Example:**
```python
import torch

class WaitKTranslator:
    """
    Wait-k policy: wait for k source tokens before translating
    """
    def __init__(self, model, tokenizer, k=3):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
    
    def translate_stream(self, source_stream):
        """Translate from streaming source"""
        source_buffer = []
        translations = []
        
        for token in source_stream:
            source_buffer.append(token)
            
            # Wait-k policy: start translating after k tokens
            if len(source_buffer) >= self.k:
                # Translate current buffer
                partial_src = ' '.join(source_buffer)
                inputs = self.tokenizer(partial_src, return_tensors='pt')
                
                # Generate one target token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    prefix_allowed_tokens_fn=self.get_prefix_fn(translations)
                )
                
                new_tokens = outputs[0, len(translations):]
                translations.extend(new_tokens.tolist())
                
                yield self.tokenizer.decode(new_tokens)
        
        # Finish remaining
        # ...

# Adaptive policy with reinforcement learning
class AdaptiveSimultaneousNMT:
    """
    Learn when to read source vs write target
    Actions: READ (get next source), WRITE (emit target)
    """
    def __init__(self, translator, policy):
        self.translator = translator
        self.policy = policy  # RL agent
    
    def translate(self, source_tokens):
        source_idx = 0
        outputs = []
        state = self.get_initial_state()
        
        while source_idx < len(source_tokens) or self.can_write(state):
            action = self.policy.predict(state)
            
            if action == 'READ' and source_idx < len(source_tokens):
                state = self.update_state(state, source_tokens[source_idx])
                source_idx += 1
            elif action == 'WRITE':
                next_token = self.translator.generate_next(state)
                outputs.append(next_token)
                state = self.update_state_output(state, next_token)
        
        return outputs

# Quality-latency tradeoff
# Higher k = better quality, higher latency
# Lower k = faster, lower quality
```

**Interview Tips:**
- Wait-k is simple and effective baseline
- k=3-5 typical tradeoff
- Adaptive policies learned via RL
- German-English harder than French-English (word order)
- Evaluation: BLEU + latency metrics

---

## Question 17
- [ ] Done
**How do you preserve named entities and technical terms during translation?**

**Answer:**

**Definition:**
Named entities (names, companies) and technical terms often shouldn't be translated or need specific translations. NMT may corrupt them. Solutions include copying mechanisms, terminology constraints, and placeholder approaches.

**Challenges:**

| Issue | Example |
|-------|--------|
| Name corruption | "Michael Jordan" → "Michel Jourdain" |
| Technical terms | "API" mistranslated |
| Brand names | "iPhone" should stay "iPhone" |
| Inconsistency | Same entity translated differently |

**Solutions:**

| Solution | Description |
|----------|-------------|
| Placeholder | Replace entities with tokens, restore after |
| Copy mechanism | Model can copy source tokens |
| Constrained decoding | Force specific translations |
| Terminology injection | Pre/post-process with glossary |
| Fine-tuning | Train with entity-aware data |

**Python Code Example:**
```python
import spacy
import re

nlp = spacy.load('en_core_web_sm')

def translate_with_entity_preservation(text, translator, entity_dict=None):
    """
    Preserve named entities during translation
    1. Extract entities
    2. Replace with placeholders
    3. Translate
    4. Restore entities
    """
    doc = nlp(text)
    
    # Extract entities and create placeholders
    entity_map = {}
    modified_text = text
    
    for i, ent in enumerate(reversed(doc.ents)):  # Reverse to maintain positions
        placeholder = f"__ENTITY_{i}__"
        entity_map[placeholder] = ent.text
        modified_text = modified_text[:ent.start_char] + placeholder + modified_text[ent.end_char:]
    
    # Translate text with placeholders
    translated = translator(modified_text)
    
    # Restore entities
    for placeholder, entity in entity_map.items():
        # Use glossary if available
        if entity_dict and entity in entity_dict:
            translated_entity = entity_dict[entity]
        else:
            translated_entity = entity  # Keep original
        translated = translated.replace(placeholder, translated_entity)
    
    return translated

# Example usage
glossary = {
    'Machine Learning': 'Machine Learning',  # Keep as-is
    'TensorFlow': 'TensorFlow',
    'New York': 'Nueva York'  # Specific translation
}

text = "Michael works at Google in New York using TensorFlow."
translated = translate_with_entity_preservation(
    text, 
    translator=my_translator,
    entity_dict=glossary
)

# Constrained beam search (force specific outputs)
from transformers import MarianMTModel

def translate_with_constraints(text, model, tokenizer, forced_terms):
    """
    Force specific term translations
    """
    inputs = tokenizer(text, return_tensors='pt')
    
    # Define allowed token sequences
    force_words_ids = [
        tokenizer(term, add_special_tokens=False).input_ids
        for term in forced_terms.values()
    ]
    
    outputs = model.generate(
        **inputs,
        force_words_ids=force_words_ids  # Require these terms
    )
    return tokenizer.decode(outputs[0])
```

**Interview Tips:**
- Placeholder approach is simple and effective
- Copy mechanism built into some seq2seq models
- Enterprise systems use terminology databases
- Test with important entities before deployment
- Case sensitivity matters for detection

---

## Question 18
- [ ] Done
**What is back-translation and how does it help with data augmentation for NMT?**

**Answer:**

**Definition:**
Back-translation generates synthetic parallel data by translating monolingual target text back to source. Provides diverse source expressions for same target, augmenting training data. Critical for low-resource scenarios.

**How It Works:**

```
Step 1: Have monolingual target data (German sentences)
Step 2: Train reverse model (German → English)
Step 3: Back-translate to get synthetic source (English)
Step 4: Use (synthetic English, real German) for training
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| More data | Leverage abundant monolingual text |
| Diversity | Same target, varied source expressions |
| Domain adaptation | Use domain-specific monolingual data |
| Low-resource | Critical when parallel data scarce |

**Python Code Example:**
```python
from transformers import MarianMTModel, MarianTokenizer

# Step 1: Train/load reverse translation model
reverse_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
reverse_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')

def back_translate_batch(target_texts):
    """Translate target language back to source"""
    inputs = reverse_tokenizer(target_texts, return_tensors='pt', padding=True)
    outputs = reverse_model.generate(**inputs)
    synthetic_source = [reverse_tokenizer.decode(out, skip_special_tokens=True) 
                       for out in outputs]
    return synthetic_source

# Step 2: Generate synthetic parallel data
monolingual_german = [
    "Das Wetter ist heute schön.",
    "Ich arbeite gerne mit Python.",
    # ... millions of German sentences
]

synthetic_english = back_translate_batch(monolingual_german)

# Step 3: Create augmented dataset
augmented_data = []
for src, tgt in zip(synthetic_english, monolingual_german):
    augmented_data.append({'source': src, 'target': tgt})

# Combine with real parallel data
full_training_data = real_parallel_data + augmented_data

# Step 4: Train forward model on augmented data
from transformers import Trainer, TrainingArguments

forward_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Training with mixed data
trainer = Trainer(
    model=forward_model,
    train_dataset=full_training_data,
    # Tag synthetic data with lower weight optionally
)
trainer.train()

# Sampling strategies for back-translation
def back_translate_with_sampling(texts, model, tokenizer, num_samples=3):
    """Generate multiple diverse back-translations per text"""
    all_synthetic = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        
        # Sample multiple translations
        outputs = model.generate(
            **inputs,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )
        
        translations = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        all_synthetic.append(translations)
    
    return all_synthetic
```

**Interview Tips:**
- Back-translation is standard practice in NMT
- Sampling > beam search for diversity
- Tag synthetic data optionally (helps model distinguish)
- Iterate: train → back-translate → train
- Noisy back-translation can help (add noise to synthetic source)

---

## Question 19
- [ ] Done
**How do multilingual translation models (mBART, M2M-100) work?**

**Answer:**

**Definition:**
Multilingual translation models translate between many language pairs with a single model. mBART pre-trains via denoising autoencoder on multiple languages. M2M-100 directly trains on 100 languages without English pivoting. Enable zero-shot translation for unseen pairs.

**Key Models:**

| Model | Languages | Approach |
|-------|-----------|----------|
| mBART | 25-50 | Denoising pre-training + fine-tune |
| M2M-100 | 100 | Direct many-to-many training |
| NLLB | 200 | Focus on low-resource languages |
| Google's | 100+ | Multilingual with language tags |

**How They Work:**

**mBART:**
1. Pre-train: denoise sentences in 25 languages (mask + shuffle)
2. Fine-tune: on parallel data for specific pairs
3. Enables zero-shot for unseen pairs via shared representations

**M2M-100:**
1. Train directly on many-to-many parallel data
2. No English-centric approach (direct fr→de without en)
3. Uses language tokens to specify direction

**Python Code Example:**
```python
# M2M-100 translation
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

def translate_any_pair(text, src_lang, tgt_lang):
    """Translate between any of 100 language pairs"""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors='pt')
    
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# French to German (no English pivot)
print(translate_any_pair("Bonjour le monde", "fr", "de"))
# Chinese to Spanish
print(translate_any_pair("你好世界", "zh", "es"))

# mBART for pre-training + fine-tuning
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

mbart = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
mbart_tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

def mbart_translate(text, src_lang, tgt_lang):
    mbart_tokenizer.src_lang = src_lang
    inputs = mbart_tokenizer(text, return_tensors='pt')
    
    generated = mbart.generate(
        **inputs,
        forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_lang]
    )
    return mbart_tokenizer.decode(generated[0], skip_special_tokens=True)

# Language codes differ by model
# M2M-100: "fr", "de", "zh"
# mBART-50: "fr_XX", "de_DE", "zh_CN"

# NLLB for low-resource
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nllb_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
nllb_tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')

# 200 languages including many low-resource
# Language codes: "eng_Latn", "fra_Latn", "zho_Hans", "yor_Latn"
```

**Interview Tips:**
- M2M-100 avoids English-centric bias
- mBART enables zero-shot via shared representations
- NLLB best for low-resource languages
- Larger models = better quality but slower
- Language code format varies by model

---

## Question 20
- [ ] Done
**What approaches handle idiomatic expressions and cultural references in translation?**

**Answer:**

**Definition:**
Idioms and cultural references don't translate literally. "It's raining cats and dogs" means heavy rain, not animals. Approaches include idiom databases, paraphrase before translation, and models trained on idiom-aware data.

**Challenges:**

| Type | Example |
|------|--------|
| Idioms | "Break a leg" ≠ Break your leg |
| Metaphors | "Heart of gold" = very kind |
| Cultural refs | "Super Bowl" - meaningless in some cultures |
| Humor | Puns don't translate |
| Sayings | "Every cloud has a silver lining" |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Idiom dictionary | Map to target equivalent |
| Paraphrase first | Convert idiom to literal meaning |
| Culture adaptation | Replace with local equivalent |
| Literal + note | Translate literally, add explanation |
| LLM prompting | Explain meaning, then translate |

**Python Code Example:**
```python
# Approach 1: Idiom dictionary lookup
idiom_dict = {
    "en": {
        "break a leg": {
            "meaning": "good luck",
            "de": "Hals- und Beinbruch",
            "fr": "merde",
            "es": "mucha mierda"
        },
        "it's raining cats and dogs": {
            "meaning": "heavy rain",
            "de": "Es regnet in Strömen",
            "fr": "Il pleut des cordes"
        }
    }
}

def translate_with_idioms(text, src_lang, tgt_lang, translator):
    """Check for idioms before translation"""
    text_lower = text.lower()
    
    for idiom, translations in idiom_dict.get(src_lang, {}).items():
        if idiom in text_lower:
            if tgt_lang in translations:
                # Replace with target idiom equivalent
                text = text.replace(idiom, f"[IDIOM]{translations[tgt_lang]}[/IDIOM]")
    
    # Translate rest normally
    translated = translator(text)
    
    # Clean up markers
    translated = translated.replace("[IDIOM]", "").replace("[/IDIOM]", "")
    return translated

# Approach 2: LLM-based paraphrase + translate
def translate_idiom_aware(text, src_lang, tgt_lang, llm):
    # Step 1: Identify and paraphrase idioms
    paraphrase_prompt = f"""Identify any idioms in this text and replace them with their literal meaning:

Text: {text}

Paraphrased:"""
    
    paraphrased = llm.generate(paraphrase_prompt)
    
    # Step 2: Translate the paraphrased text
    translate_prompt = f"""Translate this {src_lang} text to {tgt_lang}:

{paraphrased}

Translation:"""
    
    return llm.generate(translate_prompt)

# Approach 3: Cultural adaptation
def adapt_cultural_reference(text, source_culture, target_culture, llm):
    prompt = f"""Adapt this text from {source_culture} culture for a {target_culture} audience.
Replace cultural references with locally appropriate equivalents while preserving the meaning.

Original: {text}

Adapted:"""
    return llm.generate(prompt)

# Example
text = "He hit it out of the park with that presentation."
# Baseball metaphor - needs adaptation for non-US audiences
adapted = adapt_cultural_reference(text, "American", "British")
# Could become: "He absolutely nailed that presentation."
```

**Interview Tips:**
- Idiom dictionaries exist but aren't complete
- LLMs handle idioms better than older NMT
- Cultural adaptation > literal translation
- Some things genuinely untranslatable
- Professional translators use localization (adaptation)

---

## Question 21
- [ ] Done
**How do you design QA systems that handle multi-hop reasoning across documents?**

**Answer:**

**Definition:**
Multi-hop QA requires reasoning across multiple documents or passages to answer a question. Example: "Where was the director of Inception born?" needs to find (1) director = Nolan, (2) Nolan's birthplace. Requires chaining evidence.

**Example:**
```
Question: "What university did the founder of Microsoft attend?"
Hop 1: Microsoft founder = Bill Gates
Hop 2: Bill Gates attended Harvard
Answer: Harvard
```

**Approaches:**

| Approach | Description |
|----------|-------------|
| Iterative retrieval | Retrieve → extract → retrieve more |
| Graph-based | Build entity graph, traverse |
| Chain-of-thought | LLM reasons step by step |
| Multi-hop attention | Attend across documents |

**Python Code Example:**
```python
# Iterative retrieval for multi-hop
class MultiHopQA:
    def __init__(self, retriever, reader, max_hops=3):
        self.retriever = retriever
        self.reader = reader
        self.max_hops = max_hops
    
    def answer(self, question):
        context = ""
        current_query = question
        reasoning_chain = []
        
        for hop in range(self.max_hops):
            # Retrieve relevant passages
            passages = self.retriever.retrieve(current_query)
            context += " ".join(passages)
            
            # Try to answer
            answer, confidence = self.reader.answer(question, context)
            
            if confidence > 0.8:
                return answer, reasoning_chain
            
            # Extract intermediate answer for next hop
            intermediate = self.reader.extract_entity(question, context)
            reasoning_chain.append(intermediate)
            
            # Form new query for next hop
            current_query = f"{question} {intermediate}"
        
        return answer, reasoning_chain

# Chain-of-thought with LLM
def multi_hop_cot(question, context, llm):
    prompt = f"""Answer the question using step-by-step reasoning.

Context:
{context}

Question: {question}

Let's think step by step:
1."""
    
    response = llm.generate(prompt)
    return response

# Graph-based approach
import networkx as nx

class GraphMultiHopQA:
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_graph(self, documents):
        """Build entity-relation graph from documents"""
        for doc in documents:
            entities = extract_entities(doc)
            relations = extract_relations(doc)
            
            for e1, relation, e2 in relations:
                self.graph.add_edge(e1, e2, relation=relation, source=doc)
    
    def answer(self, question):
        # Find question entities
        q_entities = extract_entities(question)
        
        # Find paths between entities
        for start in q_entities:
            if start in self.graph:
                paths = nx.single_source_shortest_path(self.graph, start, cutoff=3)
                for target, path in paths.items():
                    if is_answer_type(target, question):
                        return target, path
        
        return None, None
```

**Interview Tips:**
- HotpotQA, 2WikiMultiHopQA are standard benchmarks
- Chain-of-thought prompting helps LLMs
- Iterative retrieval is practical approach
- Graph-based good for structured domains
- Multi-hop harder than single-hop

---

## Question 22
- [ ] Done
**What techniques help QA systems know when they don't know the answer?**

**Answer:**

**Definition:**
Knowing "I don't know" (unanswerable detection) is crucial for reliable QA. Systems should abstain rather than hallucinate. Techniques include confidence thresholding, training on unanswerable examples (SQuAD 2.0), and verification steps.

**Why It Matters:**
- Hallucinated answers can be harmful
- Users need to know reliability
- Critical for high-stakes domains

**Approaches:**

| Approach | Description |
|----------|-------------|
| Confidence threshold | Reject if score < threshold |
| No-answer training | Train on unanswerable examples |
| Verification | Verify answer against source |
| Calibration | Ensure confidence matches accuracy |

**Python Code Example:**
```python
from transformers import pipeline
import torch

# SQuAD 2.0 style: model outputs no-answer score
qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"  # Trained on SQuAD 2.0
)

def answer_with_abstention(question, context, threshold=0.5):
    result = qa_model(question=question, context=context)
    
    # Check confidence
    if result['score'] < threshold:
        return {
            'answer': "I don't have enough information to answer this.",
            'confidence': result['score'],
            'abstained': True
        }
    
    return {
        'answer': result['answer'],
        'confidence': result['score'],
        'abstained': False
    }

# Verification-based approach
def verified_answer(question, context, qa_model, nli_model):
    # Get answer
    result = qa_model(question=question, context=context)
    answer = result['answer']
    
    # Verify: does context entail "The answer is {answer}"?
    claim = f"The {question.replace('?', '')} is {answer}."
    nli_result = nli_model(f"{context}</s></s>{claim}")
    
    if nli_result[0]['label'] == 'CONTRADICTION':
        return "I cannot verify this answer.", 0.0
    
    return answer, result['score']

# Ensemble abstention
class EnsembleQA:
    def __init__(self, models):
        self.models = models
    
    def answer(self, question, context):
        answers = []
        for model in self.models:
            result = model(question=question, context=context)
            answers.append(result)
        
        # Check agreement
        answer_texts = [a['answer'] for a in answers]
        if len(set(answer_texts)) > 1:
            # Models disagree - abstain
            return "Models disagree, unable to provide confident answer.", 0.0
        
        avg_confidence = sum(a['score'] for a in answers) / len(answers)
        return answer_texts[0], avg_confidence
```

**Interview Tips:**
- SQuAD 2.0 adds unanswerable questions
- Calibration ensures 80% confidence ≈ 80% accuracy
- Ensemble disagreement signals uncertainty
- Verification catches hallucinations
- Always expose confidence to users

---

## Question 23
- [ ] Done
**How do you implement conversational QA with context tracking across turns?**

**Answer:**

**Definition:**
Conversational QA maintains context across dialog turns. Questions like "What about his wife?" reference previous entities. Requires coreference resolution, conversation history tracking, and context-aware retrieval.

**Challenges:**

| Challenge | Example |
|-----------|--------|
| Coreference | "he", "she", "it" → previous entity |
| Ellipsis | "And the year?" (implicit: "What year was X") |
| Topic shift | When conversation changes topic |
| Context window | Long conversations exceed limits |

**Python Code Example:**
```python
from transformers import pipeline

class ConversationalQA:
    def __init__(self, qa_model, coref_model=None):
        self.qa = qa_model
        self.history = []  # List of (question, answer) tuples
        self.context = ""  # Document context
        self.entities = {}  # Track mentioned entities
    
    def resolve_question(self, question):
        """Resolve pronouns and ellipsis"""
        # Simple pronoun resolution
        pronouns = {'he': 'male_entity', 'she': 'female_entity', 'it': 'thing_entity'}
        
        resolved = question
        for pronoun, entity_type in pronouns.items():
            if pronoun in question.lower():
                if entity_type in self.entities:
                    resolved = resolved.replace(pronoun, self.entities[entity_type])
        
        return resolved
    
    def build_context_with_history(self):
        """Include conversation history in context"""
        history_text = ""
        for q, a in self.history[-3:]:  # Last 3 turns
            history_text += f"Q: {q}\nA: {a}\n"
        
        return f"{self.context}\n\nConversation:\n{history_text}"
    
    def answer(self, question):
        # Resolve references
        resolved_question = self.resolve_question(question)
        
        # Build full context with history
        full_context = self.build_context_with_history()
        
        # Get answer
        result = self.qa(question=resolved_question, context=full_context)
        answer = result['answer']
        
        # Update history
        self.history.append((question, answer))
        
        # Track entities from answer
        self.update_entities(answer)
        
        return answer
    
    def update_entities(self, text):
        # Extract and track entities for coreference
        # Simplified: use NER
        pass

# Using LLM for conversational QA (better coreference)
def llm_conversational_qa(question, history, context, llm):
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
    
    prompt = f"""You are answering questions based on the following context.

Context:
{context}

Conversation history:
{history_text}

User: {question}
Assistant:"""
    
    return llm.generate(prompt)

# QuAC and CoQA are conversational QA benchmarks
```

**Interview Tips:**
- QuAC, CoQA are conversational QA datasets
- LLMs handle coreference naturally
- Limit history window to avoid context overflow
- Track salient entities explicitly
- Consider conversation state machine for complex dialogs

---

## Question 24
- [ ] Done
**What is the role of knowledge graphs in question answering systems?**

**Answer:**

**Definition:**
Knowledge graphs (KGs) store structured facts as (entity, relation, entity) triples. In QA, they enable precise factual lookup, multi-hop reasoning via graph traversal, and grounding answers in verified knowledge. Examples: Wikidata, Freebase, DBpedia.

**Benefits for QA:**

| Benefit | Description |
|---------|-------------|
| Factual accuracy | Facts are curated/verified |
| Multi-hop reasoning | Follow graph edges |
| Structured answers | Precise entities, not text spans |
| Explainability | Show reasoning path |

**KGQA Approaches:**

| Approach | Description |
|----------|-------------|
| SPARQL generation | Convert NL to graph query |
| Entity linking + traversal | Find entities, navigate graph |
| Embedding-based | Embed question and graph |
| LLM + KG | Use KG to ground LLM answers |

**Python Code Example:**
```python
from SPARQLWrapper import SPARQLWrapper, JSON

# Approach 1: SPARQL-based KGQA (Wikidata example)
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def query_wikidata(question):
    """Convert question to SPARQL and query"""
    # Example: "What is the capital of France?"
    sparql_query = """
    SELECT ?capital ?capitalLabel WHERE {
      wd:Q142 wdt:P36 ?capital .  # Q142 = France, P36 = capital
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    return results['results']['bindings'][0]['capitalLabel']['value']

# Approach 2: Entity linking + graph traversal
class KGQASystem:
    def __init__(self, kg):
        self.kg = kg  # Knowledge graph
        self.entity_linker = EntityLinker()  # NER + entity linking
    
    def answer(self, question):
        # Step 1: Extract entities from question
        entities = self.entity_linker.link(question)
        
        # Step 2: Determine relation (classify question type)
        relation = self.classify_relation(question)
        # "capital" → P36, "born" → P19
        
        # Step 3: Query graph
        for entity in entities:
            result = self.kg.query(entity, relation)
            if result:
                return result
        
        return None

# Approach 3: LLM grounded with KG
def kg_grounded_qa(question, kg_retriever, llm):
    # Retrieve relevant facts from KG
    entities = extract_entities(question)
    facts = []
    for entity in entities:
        entity_facts = kg_retriever.get_facts(entity)
        facts.extend(entity_facts)
    
    # Format facts for LLM
    facts_text = "\n".join([f"{s} {p} {o}" for s, p, o in facts])
    
    prompt = f"""Answer based on these facts:
{facts_text}

Question: {question}
Answer:"""
    
    return llm.generate(prompt)

# Example KG facts:
# (Barack_Obama, born_in, Honolulu)
# (Honolulu, located_in, Hawaii)
# (Hawaii, part_of, United_States)
```

**Interview Tips:**
- Wikidata has 100M+ entities, good for general QA
- SPARQL generation is hard - often use templates
- Entity linking accuracy is crucial
- Combine KG facts with LLM for fluent answers
- KGs provide explainable reasoning paths

---

## Question 25
- [ ] Done
**How do you handle questions requiring numerical reasoning or table understanding?**

**Answer:**

**Definition:**
Numerical QA requires math operations (count, compare, average) on extracted values. Table QA answers questions about structured tables. Both require beyond-text reasoning. Models: TAPAS, TaBERT, Program synthesis approaches.

**Challenges:**

| Challenge | Example |
|-----------|--------|
| Aggregation | "How many players scored > 10?" |
| Comparison | "Who is taller?" |
| Arithmetic | "What's the total revenue?" |
| Table structure | Multi-column, hierarchical |

**Approaches:**

| Approach | Description |
|----------|-------------|
| TAPAS | BERT extended for tables |
| TaBERT | Pre-trained on table + text |
| Program synthesis | Generate SQL/Python code |
| LLM + code | LLM writes calculation code |

**Python Code Example:**
```python
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

# TAPAS for table QA
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

# Example table
table = {
    "Player": ["Jordan", "LeBron", "Kobe"],
    "Points": ["32292", "38652", "33643"],
    "Rings": ["6", "4", "5"]
}

question = "Who has the most points?"

inputs = tokenizer(table=table, queries=[question], return_tensors="pt")
outputs = model(**inputs)

# Get predicted cells
predicted_answer = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)
print(predicted_answer)  # "LeBron"

# Numerical reasoning with code generation
def numerical_qa_with_code(question, data, llm):
    prompt = f"""Data:
{data}

Question: {question}

Write Python code to calculate the answer:
```python
"""
    
    code = llm.generate(prompt)
    
    # Execute code safely
    result = safe_execute(code, data)
    return result

# Example: "What is the average points per player?"
# Generated code: sum(data['Points']) / len(data['Points'])

# SQL generation for database QA
def text_to_sql(question, schema, llm):
    prompt = f"""Database schema:
{schema}

Question: {question}

SQL query:"""
    
    sql = llm.generate(prompt)
    return sql

# Example
schema = """
Table: sales
- product_id (int)
- amount (float)
- date (date)
"""
question = "What was the total sales in January?"
# Generated: SELECT SUM(amount) FROM sales WHERE MONTH(date) = 1
```

**Interview Tips:**
- TAPAS handles simple table operations
- Code generation better for complex calculations
- Text2SQL converts NL to database queries
- WikiTableQuestions, Spider are benchmarks
- Execution is more reliable than extraction for numbers

---

## Question 26
- [ ] Done
**What is dense passage retrieval (DPR) and how does it improve open-domain QA?**

**Answer:**

**Definition:**
DPR uses dual encoders to embed questions and passages in the same vector space. Retrieval is similarity search in this space, replacing keyword-based BM25. Better at semantic matching ("dog" matches "canine"), enabling more accurate passage retrieval.

**DPR vs BM25:**

| Aspect | BM25 | DPR |
|--------|------|-----|
| Matching | Lexical (exact words) | Semantic (meaning) |
| "dog"/"canine" | No match | Matches |
| Index | Inverted index | Vector index |
| Training | None | Contrastive learning |
| Speed | Very fast | Fast (with ANN) |

**Architecture:**

```
Question: "What is the capital of France?"
    ↓ [Question Encoder]
    ↓ Question Vector (768-dim)
    ↓ Cosine similarity search
    ↓ Top-k passages

Passages indexed:
Passage → [Passage Encoder] → Passage Vector → FAISS Index
```

**Python Code Example:**
```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import faiss
import numpy as np

# Load DPR models
q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Encode passages (do once, store in index)
def encode_passages(passages):
    embeddings = []
    for passage in passages:
        inputs = ctx_tokenizer(passage, return_tensors='pt', truncation=True, max_length=256)
        embedding = ctx_encoder(**inputs).pooler_output.detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Build FAISS index
passages = ["Paris is the capital of France.", "Berlin is in Germany.", ...]
passage_embeddings = encode_passages(passages)

index = faiss.IndexFlatIP(768)  # Inner product
index.add(passage_embeddings)

# Retrieve at query time
def retrieve(question, k=5):
    # Encode question
    inputs = q_tokenizer(question, return_tensors='pt')
    q_embedding = q_encoder(**inputs).pooler_output.detach().numpy()
    
    # Search
    scores, indices = index.search(q_embedding, k)
    
    return [(passages[i], scores[0][j]) for j, i in enumerate(indices[0])]

# Usage
results = retrieve("What is the capital of France?")
# Returns: [("Paris is the capital of France.", 0.95), ...]
```

**Interview Tips:**
- DPR is foundation for modern open-domain QA
- FAISS enables fast billion-scale search
- Dual encoder = separate question and passage encoders
- Training: contrastive loss with hard negatives
- ColBERT uses late interaction for better quality

---

## Question 27
- [ ] Done
**How do you balance faithfulness vs fluency in abstractive summarization?**

**Answer:**

**Definition:**
Faithfulness means summary accurately reflects source facts. Fluency means summary reads naturally. These can conflict: fluent paraphrasing may introduce inaccuracies. Balance requires careful training, decoding, and evaluation.

**The Tradeoff:**

| Priority | Risk |
|----------|------|
| High faithfulness | Choppy, extractive-like output |
| High fluency | Hallucination, factual errors |

**Strategies:**

| Strategy | How It Helps |
|----------|-------------|
| Copy mechanism | Forces source words when uncertain |
| Coverage loss | Penalizes ignoring source content |
| Factual consistency training | Reward faithful outputs |
| Constrained decoding | Limit vocabulary to source |
| Post-hoc filtering | Remove unfaithful summaries |

**Python Code Example:**
```python
import torch.nn as nn

# Loss with coverage penalty
class CoverageLoss(nn.Module):
    """
    Penalize if attention doesn't cover source
    Encourages looking at all parts of source
    """
    def __init__(self, lambda_cov=1.0):
        super().__init__()
        self.lambda_cov = lambda_cov
    
    def forward(self, attn_weights, coverage):
        # attn_weights: [batch, tgt_len, src_len]
        # coverage: sum of past attention weights
        
        # Penalize attending to already-attended positions too much
        coverage_loss = torch.sum(torch.min(attn_weights, coverage), dim=-1)
        
        return self.lambda_cov * coverage_loss.mean()

# Factual consistency reward (RLHF-style)
class FactualConsistencyReward:
    def __init__(self, nli_model):
        self.nli = nli_model
    
    def compute_reward(self, source, summary):
        # Check if source entails summary
        result = self.nli(f"{source} [SEP] {summary}")
        
        if result['label'] == 'entailment':
            return 1.0
        elif result['label'] == 'contradiction':
            return -1.0
        else:
            return 0.0

# Post-hoc filtering
def filter_unfaithful(source, summaries, nli_model, threshold=0.8):
    """Filter out unfaithful summaries"""
    faithful = []
    for summary in summaries:
        result = nli_model(f"{source} [SEP] {summary}")
        if result['label'] == 'entailment' and result['score'] > threshold:
            faithful.append(summary)
    return faithful

# Controlled generation with faithfulness constraint
def generate_faithful(model, tokenizer, source, max_length):
    # Extract source vocabulary
    source_tokens = set(tokenizer.tokenize(source))
    
    # Bias toward source tokens during generation
    def restrict_vocabulary(batch_id, sent):
        # Increase probability of source tokens
        # Decrease probability of unseen tokens
        pass
    
    outputs = model.generate(
        tokenizer(source, return_tensors='pt').input_ids,
        max_length=max_length,
        prefix_allowed_tokens_fn=restrict_vocabulary
    )
    return tokenizer.decode(outputs[0])
```

**Interview Tips:**
- Modern abstractive models prone to hallucination
- Copy mechanism helps faithfulness
- NLI-based consistency checking is common
- RLHF can train for faithfulness reward
- Extractive-abstractive hybrid is pragmatic solution

---

## Question 28
- [ ] Done
**What is the hallucination problem in summarization and how do you mitigate it?**

**Answer:**

**Definition:**
Hallucination occurs when summaries contain facts not in the source or contradict the source. Types: intrinsic (misrepresents source) and extrinsic (adds outside info). Major problem for abstractive models.

**Types of Hallucination:**

| Type | Example |
|------|--------|
| Intrinsic | Source: "Sales grew 10%" → Summary: "Sales grew 20%" |
| Extrinsic | Source: "Apple released iPhone" → Summary: "Apple released iPhone in California" (not stated) |
| Entity error | Wrong names, dates, numbers |
| Relation error | Wrong connections between entities |

**Mitigation Strategies:**

| Strategy | Description |
|----------|-------------|
| Factual consistency loss | Penalize unfaithful outputs during training |
| Copy mechanism | Force copying from source |
| Constrained decoding | Limit to source vocabulary |
| Post-hoc detection | Filter unfaithful outputs |
| RAG approach | Ground in retrieved facts |

**Python Code Example:**
```python
from transformers import pipeline

# Hallucination detection with NLI
nli_model = pipeline("text-classification", model="roberta-large-mnli")

def detect_hallucination(source, summary):
    """Check if summary is faithful to source"""
    # Split summary into sentences for fine-grained check
    sentences = summary.split('. ')
    
    hallucinations = []
    for sent in sentences:
        if not sent.strip():
            continue
        
        result = nli_model(f"{source}</s></s>{sent}")
        label = result[0]['label']
        
        if label == 'CONTRADICTION':
            hallucinations.append({'sentence': sent, 'type': 'contradiction'})
        elif label == 'NEUTRAL':
            # Potentially extrinsic hallucination
            hallucinations.append({'sentence': sent, 'type': 'possibly_extrinsic'})
    
    return hallucinations

# QA-based factual consistency (SummaC)
def qa_based_consistency(source, summary):
    """Generate questions from summary, answer from source"""
    qa_model = pipeline("question-answering")
    qg_model = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    
    # Generate questions from summary
    questions = qg_model(f"generate questions: {summary}")
    
    consistency_score = 0
    for q in questions:
        # Answer from source
        source_answer = qa_model(question=q, context=source)['answer']
        # Answer from summary
        summary_answer = qa_model(question=q, context=summary)['answer']
        
        # Compare answers
        if source_answer.lower() == summary_answer.lower():
            consistency_score += 1
    
    return consistency_score / len(questions)

# Training with factual consistency reward
class FactualSummarizer:
    def __init__(self, model, nli_model):
        self.model = model
        self.nli = nli_model
    
    def compute_consistency_reward(self, source, summary):
        result = self.nli(f"{source}</s></s>{summary}")
        if result[0]['label'] == 'ENTAILMENT':
            return 1.0
        elif result[0]['label'] == 'CONTRADICTION':
            return -1.0
        return 0.0
    
    def train_step(self, batch):
        # Generate summary
        summary = self.model.generate(batch['source'])
        
        # Compute reward
        reward = self.compute_consistency_reward(batch['source'], summary)
        
        # Policy gradient update
        # ...
```

**Interview Tips:**
- Hallucination rate can be 20-30% in abstractive models
- NLI-based detection is current standard
- QA-based metrics (QuestEval, SummaC) more fine-grained
- Extractive summaries don't hallucinate
- Critical issue for news, medical, legal domains

---

## Question 29
- [ ] Done
**How do you implement query-focused summarization for specific information needs?**

**Answer:**

**Definition:**
Query-focused summarization generates summaries relevant to a specific query/question, not just general summaries. Selects and emphasizes information relevant to user's need. Useful for search, research, and personalized content.

**Difference from Generic:**

| Generic | Query-Focused |
|---------|---------------|
| Summarize entire document | Focus on query-relevant parts |
| No user input | User provides query |
| General overview | Specific information |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Query-biased extraction | Weight sentences by query relevance |
| Query-conditioned generation | Condition decoder on query |
| Retrieval + summarization | Retrieve relevant passages, then summarize |
| QA-based | Treat as long-form QA |

**Python Code Example:**
```python
from sentence_transformers import SentenceTransformer, util
import nltk

# Approach 1: Query-biased extractive
class QueryFocusedExtractor:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def summarize(self, document, query, top_k=3):
        sentences = nltk.sent_tokenize(document)
        
        # Encode query and sentences
        query_emb = self.encoder.encode(query)
        sent_embs = self.encoder.encode(sentences)
        
        # Score by relevance to query
        scores = util.cos_sim(query_emb, sent_embs).flatten()
        
        # Select top-k relevant sentences
        top_indices = scores.argsort(descending=True)[:top_k]
        top_indices = sorted(top_indices.tolist())  # Maintain order
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary

# Approach 2: Query-conditioned abstractive
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def query_focused_abstractive(document, query):
    # Condition on query
    input_text = f"summarize for query '{query}': {document}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=4
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Approach 3: Retrieve then summarize
class RetrieveThenSummarize:
    def __init__(self, retriever, summarizer):
        self.retriever = retriever
        self.summarizer = summarizer
    
    def summarize(self, document, query):
        # Split document into chunks
        chunks = split_into_chunks(document, chunk_size=200)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(query, chunks, top_k=5)
        
        # Summarize relevant content
        context = ' '.join(relevant_chunks)
        summary = self.summarizer.generate(context)
        
        return summary

# Usage
extractor = QueryFocusedExtractor()

document = """Long article about climate change, covering causes, effects, 
solutions, political aspects, economic impacts..."""

query = "economic impacts of climate change"
summary = extractor.summarize(document, query)
# Returns sentences most relevant to economic impacts
```

**Interview Tips:**
- DUC datasets include query-focused tasks
- Query relevance weighting is key
- Can fine-tune LLMs for query-conditioned generation
- Similar to long-form QA
- User intent understanding important

---

## Question 30
- [ ] Done
**What techniques work for multi-document summarization with conflicting information?**

**Answer:**

**Definition:**
Multi-document summarization (MDS) synthesizes information from multiple sources. Challenges include redundancy, conflicting claims, and different perspectives. Must aggregate, reconcile, and present coherent summary.

**Challenges:**

| Challenge | Example |
|-----------|--------|
| Redundancy | Same fact in multiple docs |
| Contradiction | "Sales grew 10%" vs "Sales grew 15%" |
| Complementary info | Different aspects of same topic |
| Ordering | Which info comes first |
| Attribution | Which source said what |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Cluster then summarize | Group similar content, summarize clusters |
| Hierarchical | Summarize each doc, then combine |
| Graph-based | Build concept graph, extract key nodes |
| Fusion | Merge and deduplicate |
| LLM with citations | Generate with source attribution |

**Python Code Example:**
```python
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

class MultiDocSummarizer:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def summarize(self, documents, summary_length=5):
        # Extract all sentences with source
        all_sentences = []
        for doc_id, doc in enumerate(documents):
            for sent in nltk.sent_tokenize(doc):
                all_sentences.append({'text': sent, 'source': doc_id})
        
        # Encode sentences
        embeddings = self.encoder.encode([s['text'] for s in all_sentences])
        
        # Cluster to find redundancy
        n_clusters = min(summary_length * 2, len(all_sentences))
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select representative from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Pick sentence closest to cluster center
            cluster_embeddings = embeddings[cluster_indices]
            center = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            best_idx = cluster_indices[distances.argmin()]
            
            selected.append(all_sentences[best_idx])
        
        # Sort and return
        return selected[:summary_length]

# Handling contradictions
def detect_contradictions(sentences, nli_model):
    """Find contradicting claims"""
    contradictions = []
    
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences[i+1:], i+1):
            result = nli_model(f"{s1['text']}</s></s>{s2['text']}")
            
            if result[0]['label'] == 'CONTRADICTION':
                contradictions.append({
                    'claim1': s1,
                    'claim2': s2,
                    'confidence': result[0]['score']
                })
    
    return contradictions

def summarize_with_conflicts(documents, summarizer, nli_model):
    """Summarize and report conflicts"""
    summary = summarizer.summarize(documents)
    contradictions = detect_contradictions(summary, nli_model)
    
    # Present summary with conflict notes
    output = "Summary:\n"
    for s in summary:
        output += f"- {s['text']} [Source {s['source']}]\n"
    
    if contradictions:
        output += "\nConflicting claims found:\n"
        for c in contradictions:
            output += f"- '{c['claim1']['text']}' vs '{c['claim2']['text']}'\n"
    
    return output
```

**Interview Tips:**
- Clustering handles redundancy
- NLI detects contradictions
- Attribution ("according to...") helps transparency
- LLMs can synthesize but may not handle conflicts well
- DUC, TAC have multi-doc benchmarks

---

## Question 31
- [ ] Done
**How do you control summary length while maintaining coherence and coverage?**

**Answer:**

**Definition:**
Length control ensures summaries meet specific constraints (word count, sentence count) while remaining coherent and covering key information. Challenges: truncation loses info, padding adds fluff. Approaches include length tokens, constrained decoding, and iterative refinement.

**Challenges:**

| Constraint | Risk |
|------------|------|
| Too short | Missing key information |
| Too long | Redundancy, off-topic content |
| Fixed length | May cut mid-sentence |

**Approaches:**

| Approach | Description |
|----------|-------------|
| max/min_length | Hard constraints in decoding |
| Length tokens | Train with length control tokens |
| Length penalty | Bias toward shorter/longer |
| Iterative | Generate, then expand/compress |

**Python Code Example:**
```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Basic length control with min/max
def summarize_with_length(text, target_words=50, tolerance=10):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    
    # Estimate tokens from words (roughly 1.3 tokens per word)
    min_tokens = int((target_words - tolerance) * 1.3)
    max_tokens = int((target_words + tolerance) * 1.3)
    
    outputs = model.generate(
        **inputs,
        min_length=min_tokens,
        max_length=max_tokens,
        length_penalty=1.0,  # Neutral
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Length penalty effects
# length_penalty > 1.0: favors longer sequences
# length_penalty < 1.0: favors shorter sequences
# length_penalty = 1.0: neutral

# Length token approach (requires training)
class LengthControlledSummarizer:
    """Model trained with length bins: <SHORT>, <MEDIUM>, <LONG>"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def summarize(self, text, length='medium'):
        length_tokens = {
            'short': '<SHORT>',    # ~30 words
            'medium': '<MEDIUM>',  # ~80 words  
            'long': '<LONG>'       # ~150 words
        }
        
        input_text = f"{length_tokens[length]} {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])

# Iterative refinement
def compress_to_length(summary, target_words, model):
    """Iteratively compress until target length"""
    current = summary
    
    while len(current.split()) > target_words:
        prompt = f"Compress this to {target_words} words: {current}"
        current = model.generate(prompt)
    
    return current

def expand_to_length(summary, target_words, source, model):
    """Expand short summary with more detail"""
    while len(summary.split()) < target_words:
        prompt = f"""Expand this summary to {target_words} words using the source:
        
Source: {source}
Summary: {summary}

Expanded:"""
        summary = model.generate(prompt)
    
    return summary
```

**Interview Tips:**
- min/max_length are simplest controls
- Length penalty affects beam search ranking
- Length tokens require training but work well
- Iterative refinement for precise control
- Balance: coverage vs length constraint

---

## Question 32
- [ ] Done
**What is copy mechanism in seq2seq models and when is it useful?**

**Answer:**

**Definition:**
Copy mechanism (pointer-generator) allows the decoder to either generate words from vocabulary OR copy words directly from source. Useful when output should include source words verbatim: names, technical terms, rare words.

**When It Helps:**

| Scenario | Why Copy Helps |
|----------|---------------|
| Named entities | "Barack Obama" should be exact |
| Technical terms | "TensorFlow" shouldn't change |
| Rare words | Not in vocabulary, must copy |
| Quotes | Preserve exact wording |
| Numbers | "$1.5 million" should be exact |

**Architecture:**

$$P(w) = p_{gen} \cdot P_{vocab}(w) + (1-p_{gen}) \cdot \sum_{i:w_i=w} a_i$$

where $p_{gen}$ is generate probability, $a_i$ is attention weight on source position $i$.

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerGeneratorNetwork(nn.Module):
    """
    Pointer-Generator for summarization
    Can copy from source or generate from vocabulary
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 3, 1)
        
        # Vocabulary distribution
        self.vocab_proj = nn.Linear(hidden_dim * 3, vocab_size)
        
        # Pointer switch (generate vs copy)
        self.p_gen = nn.Linear(hidden_dim * 3 + embed_dim, 1)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, src_ids, tgt_ids, src_extended_ids):
        # Encode source
        src_emb = self.embedding(src_ids)
        encoder_outputs, (h, c) = self.encoder(src_emb)
        
        outputs = []
        for t in range(tgt_ids.size(1)):
            tgt_emb = self.embedding(tgt_ids[:, t:t+1])
            decoder_output, (h, c) = self.decoder(tgt_emb, (h, c))
            
            # Attention over source
            attn_weights = self.compute_attention(decoder_output, encoder_outputs)
            context = (attn_weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
            
            # Combined features
            combined = torch.cat([decoder_output.squeeze(1), context], dim=-1)
            
            # Vocabulary distribution
            vocab_dist = F.softmax(self.vocab_proj(combined), dim=-1)
            
            # Copy distribution (attention over source)
            copy_dist = attn_weights
            
            # Generate probability
            p_gen_input = torch.cat([combined, tgt_emb.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.p_gen(p_gen_input))
            
            # Final distribution: weighted combination
            # Scatter copy distribution to vocabulary space
            final_dist = p_gen * vocab_dist
            final_dist.scatter_add_(1, src_extended_ids, (1 - p_gen) * copy_dist)
            
            outputs.append(final_dist)
        
        return torch.stack(outputs, dim=1)
    
    def compute_attention(self, decoder_output, encoder_outputs):
        # Simplified attention computation
        scores = torch.bmm(encoder_outputs, decoder_output.transpose(1, 2)).squeeze(-1)
        return F.softmax(scores, dim=-1)

# Usage in modern transformers
# BART, T5 implicitly learn to copy through attention
# For explicit copy, use specialized models or add copy loss
```

**Interview Tips:**
- Pointer-Generator Networks (See et al., 2017) for summarization
- Essential for preserving factual accuracy
- Modern transformers learn soft copying via attention
- Especially important for abstractive summarization
- Reduces OOV (out-of-vocabulary) problems

---

## Question 33
- [ ] Done
**How do you implement knowledge distillation for compressing large generation models?**

**Answer:**

**Definition:**
Distillation for generation trains a small "student" model to mimic a large "teacher" model's generation behavior. Challenges: sequence-level outputs, exposure bias, and quality-speed tradeoff. Approaches include sequence-level distillation and word-level KD.

**Approaches:**

| Approach | Description |
|----------|-------------|
| Word-level KD | Match token probability distributions |
| Sequence-level KD | Train on teacher-generated outputs |
| Intermediate layers | Match hidden representations |
| Hint-based | Use teacher attention as hints |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Word-level knowledge distillation
class WordLevelDistillation(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss (cross-entropy with ground truth)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Soft label loss (KL divergence with teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Sequence-level distillation
class SequenceLevelDistillation:
    """
    1. Generate outputs from teacher
    2. Train student on (source, teacher_output) pairs
    """
    def __init__(self, teacher, student, tokenizer):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
    
    def create_distillation_data(self, sources):
        """Generate pseudo-labels from teacher"""
        self.teacher.eval()
        pairs = []
        
        with torch.no_grad():
            for source in sources:
                inputs = self.tokenizer(source, return_tensors='pt')
                teacher_output = self.teacher.generate(**inputs, num_beams=4)
                target = self.tokenizer.decode(teacher_output[0])
                pairs.append((source, target))
        
        return pairs
    
    def train_student(self, distillation_data):
        """Train student on teacher-generated outputs"""
        # Standard seq2seq training
        # Source -> Student -> Teacher's output
        pass

# DistilBART example (pre-distilled model)
from transformers import BartForConditionalGeneration

# Full model
teacher = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Distilled model (fewer layers)
student = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6')
# 6 encoder + 6 decoder layers vs 12+12
# ~2x faster, similar quality

# Training loop with distillation
def distill_train_step(student, teacher, batch, optimizer, kd_loss_fn):
    teacher.eval()
    student.train()
    
    # Get teacher predictions
    with torch.no_grad():
        teacher_outputs = teacher(**batch, output_hidden_states=True)
    
    # Get student predictions
    student_outputs = student(**batch, output_hidden_states=True)
    
    # Compute distillation loss
    loss = kd_loss_fn(
        student_outputs.logits,
        teacher_outputs.logits,
        batch['labels']
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

**Interview Tips:**
- DistilBART, DistilGPT2 are pre-distilled
- Sequence-level distillation often better for generation
- Temperature 2-4 works well
- Typically 2-4x speedup with 1-5% quality drop
- Can distill ensemble of teachers

---

## Question 34
- [ ] Done
**What is nucleus (top-p) sampling and how does it improve generation quality?**

**Answer:**

**Definition:**
Nucleus sampling (top-p) samples from the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.9). Unlike top-k which uses fixed k, it adapts to the probability distribution shape, including more tokens when uncertain, fewer when confident.

**Comparison:**

| Method | How It Works | Adaptivity |
|--------|--------------|------------|
| Greedy | Pick highest prob | None |
| Top-k | Sample from top k tokens | Fixed k |
| Top-p | Sample from top p% probability mass | Adapts to distribution |

**Why Top-p Is Better:**

```
Confident: P = [0.9, 0.05, 0.03, ...]  → top-p=0.9 uses 1 token
Uncertain: P = [0.1, 0.1, 0.1, ...]   → top-p=0.9 uses 9 tokens

Top-k=5 would use 5 tokens in both cases (suboptimal)
```

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling
    Sample from smallest set of tokens with cumulative prob > p
    """
    # Apply temperature
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Sort by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff: smallest set with cumsum > p
    # Keep tokens up to and including first token that exceeds p
    cutoff_mask = cumsum_probs > p
    # Shift mask to include the first token that exceeds threshold
    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
    cutoff_mask[..., 0] = False
    
    # Zero out tokens beyond cutoff
    sorted_probs[cutoff_mask] = 0.0
    
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    
    # Map back to original indices
    return sorted_indices.gather(-1, sampled_idx)

# Combined top-k and top-p
def top_k_top_p_sampling(logits, k=50, p=0.9, temperature=1.0):
    """Apply both top-k and top-p filtering"""
    logits = logits / temperature
    
    # Top-k filtering
    if k > 0:
        top_k_logits, _ = torch.topk(logits, k)
        min_top_k = top_k_logits[..., -1:]
        logits = torch.where(logits < min_top_k, 
                            torch.full_like(logits, float('-inf')),
                            logits)
    
    # Top-p filtering
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        
        mask = cumsum - probs > p
        sorted_logits[mask] = float('-inf')
        
        # Unsort
        logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Using transformers library
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

output = model.generate(
    tokenizer("The future of AI is", return_tensors='pt').input_ids,
    max_length=100,
    do_sample=True,
    top_p=0.9,          # Nucleus sampling
    top_k=0,            # Disable top-k
    temperature=0.8,    # Slight randomness
    no_repeat_ngram_size=3
)
```

**Interview Tips:**
- p=0.9 or 0.95 commonly used
- Temperature + top-p often combined
- Top-p adapts to context better than top-k
- Used in GPT-3, ChatGPT, etc.
- For factual tasks, use lower p (more deterministic)

---

## Question 35
- [ ] Done
**How do you evaluate factual consistency in generated text?**

**Answer:**

**Definition:**
Factual consistency evaluation checks if generated text accurately represents source facts. Critical for summarization, QA, and data-to-text. Methods include NLI-based, QA-based, and learned metrics.

**Evaluation Methods:**

| Method | Approach |
|--------|----------|
| NLI-based | Check entailment: source → generated |
| QA-based | Generate Qs from output, answer from source |
| Entity matching | Verify entities mentioned correctly |
| Learned metrics | Train model to predict consistency |

**Key Metrics:**

| Metric | How It Works |
|--------|-------------|
| FactCC | Trained classifier for consistency |
| SummaC | NLI-based, sentence-level |
| QuestEval | QA-based consistency |
| DAE | Dependency arc entailment |

**Python Code Example:**
```python
from transformers import pipeline
import nltk

# Approach 1: NLI-based consistency (SummaC-style)
nli_model = pipeline("text-classification", model="roberta-large-mnli")

def nli_consistency_score(source, generated):
    """Check if source entails generated text"""
    # Sentence-level evaluation
    generated_sents = nltk.sent_tokenize(generated)
    
    scores = []
    for sent in generated_sents:
        result = nli_model(f"{source}</s></s>{sent}")
        
        # Score based on entailment probability
        for r in result:
            if r['label'] == 'ENTAILMENT':
                scores.append(r['score'])
            elif r['label'] == 'CONTRADICTION':
                scores.append(-r['score'])  # Penalize contradictions
            else:
                scores.append(0)  # Neutral
    
    return sum(scores) / len(scores) if scores else 0

# Approach 2: QA-based consistency (QuestEval-style)
qa_model = pipeline("question-answering")
qg_model = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def qa_consistency_score(source, generated):
    """Generate questions from output, verify answers in source"""
    # Generate questions from generated text
    questions = qg_model(f"generate questions: {generated}", max_length=128)
    
    consistency_scores = []
    for q_text in questions:
        q = q_text['generated_text']
        
        try:
            # Answer from source
            source_answer = qa_model(question=q, context=source)
            # Answer from generated
            gen_answer = qa_model(question=q, context=generated)
            
            # Compare answers (simple string match)
            if source_answer['answer'].lower() == gen_answer['answer'].lower():
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
        except:
            consistency_scores.append(0.5)  # Uncertainty
    
    return sum(consistency_scores) / len(consistency_scores)

# Approach 3: Entity-level verification
import spacy
nlp = spacy.load('en_core_web_sm')

def entity_consistency(source, generated):
    """Check if entities in generated are in source"""
    source_doc = nlp(source)
    gen_doc = nlp(generated)
    
    source_entities = set([ent.text.lower() for ent in source_doc.ents])
    gen_entities = set([ent.text.lower() for ent in gen_doc.ents])
    
    # Check what fraction of generated entities are in source
    if not gen_entities:
        return 1.0
    
    correct = len(gen_entities & source_entities)
    return correct / len(gen_entities)

# Combined evaluation
def evaluate_factual_consistency(source, generated):
    return {
        'nli_score': nli_consistency_score(source, generated),
        'qa_score': qa_consistency_score(source, generated),
        'entity_score': entity_consistency(source, generated)
    }
```

**Interview Tips:**
- NLI-based is fast but coarse-grained
- QA-based catches more subtle errors
- Entity matching good for names/numbers
- Combine multiple methods for robustness
- Human evaluation still gold standard

---

## Question 36
- [ ] Done
**What is curriculum learning and how does it help train seq2seq models?**

**Answer:**

**Definition:**
Curriculum learning trains models on easier examples first, gradually increasing difficulty. For seq2seq, this means starting with short sequences, then longer ones; or simple patterns before complex ones. Improves convergence and final performance.

**Curriculum Dimensions:**

| Dimension | Easy → Hard |
|-----------|-------------|
| Sequence length | Short → Long |
| Vocabulary | Common words → Rare words |
| Sentence complexity | Simple → Complex syntax |
| Noise level | Clean → Noisy |

**Benefits:**

| Benefit | Description |
|---------|-------------|
| Faster convergence | Learns basic patterns first |
| Better generalization | Progressive complexity |
| Stability | Avoids early training collapse |
| Long sequence handling | Learns incrementally |

**Python Code Example:**
```python
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class CurriculumSampler(Sampler):
    """
    Sample easier (shorter) examples first, gradually include harder ones
    """
    def __init__(self, dataset, difficulty_scores, num_epochs, current_epoch):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores  # e.g., sequence lengths
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
    
    def __iter__(self):
        # Compute difficulty threshold based on epoch progress
        progress = (self.current_epoch + 1) / self.num_epochs
        threshold = np.percentile(self.difficulty_scores, progress * 100)
        
        # Include examples up to current difficulty
        valid_indices = [i for i, d in enumerate(self.difficulty_scores) 
                        if d <= threshold]
        
        # Shuffle valid indices
        np.random.shuffle(valid_indices)
        return iter(valid_indices)
    
    def __len__(self):
        return len(self.dataset)

# Length-based curriculum for seq2seq
class LengthCurriculumTrainer:
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data
        
        # Compute difficulty (sequence length)
        self.difficulties = [len(example['source']) + len(example['target']) 
                            for example in train_data]
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Update curriculum sampler
            sampler = CurriculumSampler(
                self.train_data, 
                self.difficulties, 
                num_epochs, 
                epoch
            )
            dataloader = DataLoader(self.train_data, sampler=sampler, batch_size=32)
            
            # Train epoch
            for batch in dataloader:
                loss = self.train_step(batch)
            
            print(f"Epoch {epoch}: trained on {len(sampler)} examples")

# Competence-based curriculum (dynamic)
class CompetenceCurriculum:
    """
    Platanios et al. (2019) - adaptive competence
    c(t) = min(1, sqrt(t * (1-c0^2) / T + c0^2))
    """
    def __init__(self, c0=0.01, T=10000):
        self.c0 = c0  # Initial competence
        self.T = T    # Steps to full competence
        self.step = 0
    
    def get_competence(self):
        competence = min(1, np.sqrt(
            self.step * (1 - self.c0**2) / self.T + self.c0**2
        ))
        return competence
    
    def get_difficulty_threshold(self, difficulties):
        c = self.get_competence()
        return np.percentile(difficulties, c * 100)
    
    def update(self):
        self.step += 1
```

**Interview Tips:**
- Curriculum especially helps for long sequences
- Sort by length is simplest curriculum
- Can also use loss-based difficulty (train on low-loss examples first)
- Transformer training often uses length-based batching
- Related: self-paced learning (model decides difficulty)

---

## Question 37
- [ ] Done
**How do you handle code generation and code summarization tasks?**

**Answer:**

**Definition:**
Code generation produces code from natural language descriptions. Code summarization produces natural language from code. Both require understanding code structure, syntax, and semantics. Models: Codex, CodeT5, StarCoder.

**Tasks:**

| Task | Input | Output |
|------|-------|--------|
| Code generation | "Sort a list in Python" | `sorted(list)` |
| Code summarization | Function code | "Sorts a list ascending" |
| Code completion | Partial code | Completed code |
| Code translation | Python code | JavaScript code |

**Key Challenges:**

| Challenge | Description |
|-----------|-------------|
| Syntax correctness | Generated code must parse |
| Semantic correctness | Code must do what's asked |
| Context | Function calls, imports |
| Testing | Verify code works |

**Python Code Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Code generation with StarCoder
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.2,  # Low temp for code
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
prompt = """# Python function to check if a number is prime
def is_prime(n):"""
code = generate_code(prompt)
print(code)

# Code summarization with CodeT5
from transformers import T5ForConditionalGeneration, RobertaTokenizer

model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')

def summarize_code(code):
    inputs = tokenizer(code, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

code_snippet = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
summary = summarize_code(code_snippet)
print(summary)  # "Calculates the factorial of a number recursively"

# Execution-based evaluation
def evaluate_generated_code(generated_code, test_cases):
    """Run generated code against test cases"""
    try:
        exec(generated_code, globals())
        
        passed = 0
        for inputs, expected in test_cases:
            result = eval(f"solution({inputs})")
            if result == expected:
                passed += 1
        
        return passed / len(test_cases)
    except Exception as e:
        return 0.0  # Syntax error or runtime error

# OpenAI Codex via API
import openai

def codex_generate(prompt):
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0,
        stop=["\n\n"]
    )
    return response.choices[0].text
```

**Interview Tips:**
- Use low temperature for code (0.1-0.3)
- Execution-based evaluation (pass@k) is key metric
- Code models trained on GitHub data
- Context window important for imports, classes
- HumanEval, MBPP are standard benchmarks

---

## Question 38
- [ ] Done
**What techniques improve translation quality for morphologically rich languages?**

**Answer:**

**Definition:**
Morphologically rich languages (Turkish, Finnish, Hungarian, Arabic) have many word forms through inflection, creating large vocabularies and data sparsity. Techniques include subword tokenization, morphological analysis, and character-level models.

**Challenges:**

| Challenge | Example |
|-----------|--------|
| Large vocabulary | One lemma → many surface forms |
| Data sparsity | Rare word forms |
| Agreement | Gender, case, number matching |
| Long words | Agglutinative languages |

**Techniques:**

| Technique | Description |
|-----------|-------------|
| BPE/WordPiece | Subword tokenization |
| Morphological segmentation | Split into morphemes |
| Character-level | Bypass tokenization entirely |
| Lemmatization + tags | Separate lemma from features |

**Python Code Example:**
```python
# Approach 1: BPE handles morphology implicitly
from tokenizers import Tokenizer, models, trainers

# Train BPE on morphologically rich language
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
)
tokenizer.train(files=["turkish_corpus.txt"], trainer=trainer)

# BPE naturally segments morphemes
# "evlerimizde" (in our houses) -> ["ev", "ler", "imiz", "de"]

# Approach 2: Explicit morphological segmentation
def morphological_segment(word, analyzer):
    """
    Split word into morphemes
    Turkish: "okuyacaklardı" -> "oku" + "yacak" + "lar" + "dı"
    (read + will + they + past)
    """
    analysis = analyzer.analyze(word)
    return analysis.morphemes

# Approach 3: Character-level model for morphology
import torch.nn as nn

class CharLevelEncoder(nn.Module):
    """
    Encode words character by character
    Better for unseen word forms
    """
    def __init__(self, char_vocab_size, char_embed_dim, hidden_dim):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        self.lstm = nn.LSTM(char_embed_dim, hidden_dim, bidirectional=True)
    
    def forward(self, char_ids):
        # char_ids: [batch, word_len]
        embedded = self.char_embed(char_ids)
        outputs, (h, c) = self.lstm(embedded)
        # Use final hidden state as word representation
        word_repr = torch.cat([h[-2], h[-1]], dim=-1)
        return word_repr

# Approach 4: Factored translation (lemma + features)
class FactoredNMT:
    """
    Separate lemma from morphological features
    Input: "running" -> lemma:"run" + features:"VBG"
    """
    def encode(self, word):
        lemma = self.lemmatizer(word)
        features = self.get_morph_features(word)
        return lemma, features
    
    def translate(self, source):
        # Translate lemmas
        # Transfer/generate appropriate target morphology
        pass

# SentencePiece for language-agnostic tokenization
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='spm_morph',
    vocab_size=32000,
    model_type='unigram'  # Unigram better for morphology
)
```

**Interview Tips:**
- BPE/SentencePiece handle morphology implicitly
- Unigram LM segmentation often better than BPE for morphology
- Character-level helps generalize to unseen forms
- Morphological analyzers available for major languages
- Agglutinative languages need larger BPE vocabulary

---

## Question 39
- [ ] Done
**How do you implement hybrid extractive-abstractive summarization?**

**Answer:**

**Definition:**
Hybrid summarization combines extractive (select sentences) and abstractive (generate new text) approaches. Extract important content first, then paraphrase/compress. Balances faithfulness (extractive) with fluency (abstractive).

**Hybrid Strategies:**

| Strategy | Description |
|----------|-------------|
| Extract-then-abstract | Select sentences, then rewrite |
| Bottom-up attention | Mask then generate |
| Sentence fusion | Combine extracted sentences |
| Guided generation | Extractive output guides decoder |

**Benefits:**

| Benefit | How |
|---------|-----|
| Faithfulness | Extraction ensures source coverage |
| Fluency | Abstraction polishes output |
| Efficiency | Reduce input to abstractive model |
| Control | Can tune extraction ratio |

**Python Code Example:**
```python
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk

class HybridSummarizer:
    def __init__(self):
        # Extractive: sentence embeddings
        self.sent_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Abstractive: BART
        self.abstractive = pipeline('summarization', model='facebook/bart-large-cnn')
    
    def extract_top_sentences(self, text, ratio=0.3):
        """Extract most important sentences"""
        sentences = nltk.sent_tokenize(text)
        n_select = max(1, int(len(sentences) * ratio))
        
        # Encode sentences
        embeddings = self.sent_encoder.encode(sentences)
        
        # Compute importance: similarity to document centroid
        centroid = embeddings.mean(axis=0)
        scores = [util.cos_sim(emb, centroid).item() for emb in embeddings]
        
        # Select top sentences
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_select]
        top_indices = sorted(top_indices)  # Maintain order
        
        return ' '.join([sentences[i] for i in top_indices])
    
    def summarize(self, text, extract_ratio=0.4, final_length=100):
        """Hybrid: extract then abstract"""
        # Step 1: Extract important content
        extracted = self.extract_top_sentences(text, ratio=extract_ratio)
        
        # Step 2: Abstract (compress and polish)
        abstract_result = self.abstractive(
            extracted,
            max_length=final_length,
            min_length=30,
            do_sample=False
        )
        
        return abstract_result[0]['summary_text']

# Bottom-up attention approach
class BottomUpSummarizer:
    """
    Gehrmann et al. (2018)
    1. Extract content words/phrases
    2. Constrain decoder to use extracted content
    """
    def __init__(self, extractor, generator):
        self.extractor = extractor  # Content selector
        self.generator = generator  # Constrained decoder
    
    def summarize(self, text):
        # Extract important phrases
        important_phrases = self.extractor.extract(text)
        
        # Generate with copy constraint
        summary = self.generator.generate(
            text,
            copy_phrases=important_phrases  # Bias toward these
        )
        
        return summary

# Sentence fusion
def fuse_sentences(sentences, fusion_model):
    """
    Combine related sentences into one
    "John went to store. He bought milk." -> "John went to store and bought milk."
    """
    prompt = f"Combine these sentences into one: {' '.join(sentences)}"
    return fusion_model.generate(prompt)

# Usage
hybrid = HybridSummarizer()

long_article = """...(long news article)..."""
summary = hybrid.summarize(long_article)
print(summary)
```

**Interview Tips:**
- Extract-then-abstract is most common hybrid
- Reduces hallucination vs pure abstractive
- Extraction ratio is key hyperparameter
- Bottom-up attention constrains generation
- Good for long documents (reduce input length)

---

## Question 40
- [ ] Done
**What is cross-lingual summarization and what challenges does it present?**

**Answer:**

**Definition:**
Cross-lingual summarization generates a summary in language B from a document in language A (e.g., French document → English summary). Combines translation and summarization challenges. Approaches: translate-then-summarize, summarize-then-translate, or end-to-end.

**Pipeline Options:**

| Approach | Pipeline | Pros | Cons |
|----------|----------|------|------|
| Translate-Summarize | Doc(Fr) → Doc(En) → Summary(En) | Use best En summarizer | Error propagation |
| Summarize-Translate | Doc(Fr) → Summary(Fr) → Summary(En) | Summarize in source | Summary translation errors |
| End-to-end | Doc(Fr) → Summary(En) directly | No error propagation | Needs parallel data |

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Parallel data | Few cross-lingual summarization datasets |
| Cultural context | Different relevance across cultures |
| Named entities | May need transliteration |
| Length | Different languages have different verbosity |

**Python Code Example:**
```python
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast

# Approach 1: Translate then summarize
def translate_then_summarize(doc, src_lang, tgt_lang):
    # Translate
    translator = pipeline('translation', model=f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    translated = translator(doc, max_length=1024)[0]['translation_text']
    
    # Summarize
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(translated, max_length=130)[0]['summary_text']
    
    return summary

# Approach 2: End-to-end with mBART
class CrossLingualSummarizer:
    def __init__(self):
        self.model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
        self.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
    
    def summarize(self, doc, src_lang, tgt_lang):
        # Set source language
        self.tokenizer.src_lang = src_lang
        
        inputs = self.tokenizer(doc, return_tensors='pt', max_length=1024, truncation=True)
        
        # Generate summary in target language
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=150
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Approach 3: Zero-shot with multilingual LLM
def zero_shot_cross_lingual_summary(doc, src_lang, tgt_lang, llm):
    prompt = f"""Summarize the following {src_lang} document in {tgt_lang}.

Document:
{doc}

{tgt_lang} Summary:"""
    
    return llm.generate(prompt)

# Example usage
french_article = """La Tour Eiffel est un monument emblématique de Paris..."""

# Translate then summarize
english_summary = translate_then_summarize(french_article, 'fr', 'en')

# End-to-end
cls = CrossLingualSummarizer()
english_summary = cls.summarize(french_article, 'fr_XX', 'en_XX')

print(english_summary)
```

**Interview Tips:**
- mBART, mT5 enable end-to-end cross-lingual
- Translate-then-summarize is simpler, often works well
- Error propagation is main issue with pipeline
- XL-Sum dataset for cross-lingual evaluation
- Zero-shot with LLMs increasingly effective

---

## Question 41
- [ ] Done
**How do you handle long document summarization that exceeds context limits?**

**Answer:**

**Definition:**
Long documents exceed transformer context limits (512-4096 tokens). Solutions: chunking with aggregation, hierarchical models, sliding window attention (Longformer), or retrieve-then-summarize approaches.

**Strategies:**

| Strategy | Description |
|----------|-------------|
| Chunking | Split doc, summarize chunks, merge |
| Hierarchical | Encode sentences → encode document |
| Sparse attention | Longformer, BigBird |
| Retrieve + summarize | Select relevant chunks first |

**Chunking Approaches:**

| Approach | Description |
|----------|-------------|
| Recursive | Summarize chunks → summarize summaries |
| MapReduce | Parallel chunk summaries → combine |
| Refine | Iteratively refine summary with new chunks |
| Stuff | If fits, just use all (baseline) |

**Python Code Example:**
```python
from transformers import pipeline
import nltk

class LongDocSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn', max_chunk_tokens=1024):
        self.summarizer = pipeline('summarization', model=model_name)
        self.max_chunk = max_chunk_tokens
    
    def chunk_document(self, text, chunk_size=800, overlap=100):
        """Split into overlapping chunks"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last sentences for overlap
                current_chunk = current_chunk[-2:] if overlap else []
                current_len = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sent)
            current_len += sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_mapreduce(self, text, final_length=200):
        """MapReduce: summarize chunks in parallel, then combine"""
        chunks = self.chunk_document(text)
        
        # Map: summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=150, min_length=30)[0]['summary_text']
            chunk_summaries.append(summary)
        
        # Reduce: combine and summarize again
        combined = ' '.join(chunk_summaries)
        
        if len(combined.split()) > 500:
            return self.summarize_mapreduce(combined, final_length)
        
        final_summary = self.summarizer(combined, max_length=final_length)[0]['summary_text']
        return final_summary

# Using LED for longer context
from transformers import LEDForConditionalGeneration, LEDTokenizer

class LongformerSummarizer:
    def __init__(self):
        # LED: Longformer Encoder-Decoder (up to 16K tokens)
        self.model = LEDForConditionalGeneration.from_pretrained('allenai/led-large-16384-arxiv')
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384-arxiv')
    
    def summarize(self, long_doc, max_length=256):
        inputs = self.tokenizer(long_doc, return_tensors='pt', max_length=16384, truncation=True)
        
        # Set global attention on first token
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1
        
        outputs = self.model.generate(
            **inputs,
            global_attention_mask=global_attention_mask,
            max_length=max_length
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Interview Tips:**
- MapReduce most common for chunking
- Overlap helps maintain context
- Longformer/LED handle 16K+ tokens natively
- LLMs with 100K+ context changing landscape
- Hierarchical attention is research direction

---

## Question 42
- [ ] Done
**What is the exposure bias problem in seq2seq training and how do you address it?**

**Answer:**

**Definition:**
Exposure bias: mismatch between training (uses ground truth previous tokens) and inference (uses model's own predictions). At inference, errors compound because model never saw its own mistakes during training.

**The Problem:**

| Phase | Previous Token Source |
|-------|----------------------|
| Training | Ground truth (teacher forcing) |
| Inference | Model's own prediction |
| Gap | Model doesn't handle own errors |

**Solutions:**

| Solution | Description |
|----------|-------------|
| Scheduled sampling | Gradually use model outputs during training |
| Beam search | Explore multiple hypotheses |
| RAML | Reward augmented maximum likelihood |
| RL fine-tuning | Optimize on actual outputs |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import random

# Scheduled Sampling (Bengio et al., 2015)
class ScheduledSamplingDecoder(nn.Module):
    def __init__(self, embedding, decoder_cell, vocab_size):
        super().__init__()
        self.embedding = embedding
        self.decoder_cell = decoder_cell
        self.output_proj = nn.Linear(decoder_cell.hidden_size, vocab_size)
    
    def forward(self, encoder_output, target, teacher_forcing_ratio=0.5):
        batch_size, seq_len = target.shape
        outputs = []
        
        current_input = target[:, 0]
        hidden = encoder_output
        
        for t in range(1, seq_len):
            embedded = self.embedding(current_input)
            output, hidden = self.decoder_cell(embedded, hidden)
            logits = self.output_proj(output)
            outputs.append(logits)
            
            # Scheduled sampling: use teacher forcing probabilistically
            if random.random() < teacher_forcing_ratio:
                current_input = target[:, t]  # Ground truth
            else:
                current_input = logits.argmax(dim=-1)  # Model prediction
        
        return torch.stack(outputs, dim=1)

# Curriculum scheduler
class CurriculumScheduler:
    def __init__(self, init_ratio=1.0, min_ratio=0.1, decay=0.99):
        self.ratio = init_ratio
        self.min_ratio = min_ratio
        self.decay = decay
    
    def step(self):
        self.ratio = max(self.min_ratio, self.ratio * self.decay)
        return self.ratio

# RL Fine-tuning (REINFORCE)
class RLFineTuning:
    def __init__(self, model, reward_fn):
        self.model = model
        self.reward_fn = reward_fn
    
    def compute_loss(self, source, reference):
        # Sample from model (not teacher forcing)
        with torch.no_grad():
            sampled_output, log_probs = self.model.sample(source)
        
        # Compute reward (e.g., BLEU, ROUGE)
        reward = self.reward_fn(sampled_output, reference)
        
        # REINFORCE loss
        baseline = self.get_baseline(source)
        loss = -log_probs * (reward - baseline)
        
        return loss.mean()
```

**Interview Tips:**
- Scheduled sampling is most common solution
- Linear/exponential decay schedules for teacher forcing ratio
- RL approaches (REINFORCE) are more complex but effective
- Beam search at inference partially mitigates
- Modern LLMs with RLHF address this differently

---

## Question 43
- [ ] Done
**How do you implement real-time QA with low latency requirements?**

**Answer:**

**Definition:**
Real-time QA requires sub-second response times. Optimize: model size (distillation), caching, efficient retrieval (ANN), batching, and hardware (GPU, TPU). Trade accuracy for speed where acceptable.

**Latency Components:**

| Component | Typical Latency | Optimization |
|-----------|-----------------|---------------|
| Retrieval | 50-200ms | ANN, caching |
| Model inference | 100-500ms | Distillation, quantization |
| Network | 10-50ms | CDN, edge |
| Total budget | <500ms | Parallelize |

**Optimization Strategies:**

| Strategy | Speedup | Trade-off |
|----------|---------|----------|
| Model distillation | 5-10x | Slight accuracy loss |
| Quantization (INT8) | 2-4x | Minimal loss |
| Caching | 10-100x | Memory cost |
| Early exit | 2-3x | Variable quality |

**Python Code Example:**
```python
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import faiss
from functools import lru_cache

class LowLatencyQA:
    def __init__(self):
        # Use distilled model (smaller, faster)
        self.model = DistilBertForQuestionAnswering.from_pretrained(
            'distilbert-base-cased-distilled-squad'
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-cased-distilled-squad'
        )
        self.model.eval()
        
        # Quantize for faster inference
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    @lru_cache(maxsize=10000)
    def answer_cached(self, question, context):
        """Cache frequent Q&A pairs"""
        return self._answer(question, context)
    
    def _answer(self, question, context):
        inputs = self.tokenizer(
            question, context,
            return_tensors='pt',
            max_length=384,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start = outputs.start_logits.argmax()
        end = outputs.end_logits.argmax()
        
        answer = self.tokenizer.decode(inputs['input_ids'][0][start:end+1])
        return answer

# Efficient retrieval with FAISS HNSW
class FastRetriever:
    def __init__(self, dimension=768):
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efSearch = 64
        self.documents = []
    
    def add_documents(self, embeddings, docs):
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(docs)
    
    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1), k
        )
        return [self.documents[i] for i in indices[0]]

# ONNX for faster inference
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained(
    'distilbert-base-cased-distilled-squad',
    export=True
)
# ONNX typically 2-3x faster
```

**Interview Tips:**
- DistilBERT is 60% faster, 97% accuracy of BERT
- INT8 quantization: 2-4x speedup, minimal loss
- FAISS HNSW: millisecond retrieval
- Cache common queries (80/20 rule)
- GPU batching for throughput, not latency

---

## Question 44
- [ ] Done
**What role do pre-trained language models (T5, BART, GPT) play in text generation tasks?**

**Answer:**

**Definition:**
Three paradigms: T5 (encoder-decoder, text-to-text), BART (encoder-decoder, denoising), GPT (decoder-only, autoregressive). Each suited for different tasks.

**Architecture Comparison:**

| Model | Architecture | Pre-training | Best For |
|-------|--------------|--------------|----------|
| T5 | Encoder-Decoder | Text-to-text | Translation, summarization |
| BART | Encoder-Decoder | Denoising | Summarization, generation |
| GPT | Decoder-only | Autoregressive LM | Open-ended generation |

**Pre-training Objectives:**

| Model | Objective |
|-------|----------|
| T5 | Span corruption (fill masked spans) |
| BART | Multiple noising (delete, permute, mask) |
| GPT | Next token prediction |

**Key Differences:**

| Aspect | T5 | BART | GPT |
|--------|----|----|-----|
| Input format | "translate: ..." | Raw text | Raw text + prompt |
| Bidirectional encoder | Yes | Yes | No (causal) |
| Task framing | All as text-to-text | Seq2seq | Continuation |

**Python Code Example:**
```python
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)

# T5: Text-to-Text approach
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def t5_generate(task, text):
    input_text = f"{task}: {text}"
    inputs = t5_tokenizer(input_text, return_tensors='pt')
    outputs = t5_model.generate(**inputs, max_length=150)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

print(t5_generate("summarize", "Long article..."))
print(t5_generate("translate English to French", "Hello world"))

# BART: Best for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def bart_summarize(text):
    inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = bart_model.generate(**inputs, max_length=150, num_beams=4)
    return bart_tokenizer.decode(outputs[0], skip_special_tokens=True)

# GPT: Open-ended generation
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def gpt_generate(prompt):
    inputs = gpt_tokenizer(prompt, return_tensors='pt')
    outputs = gpt_model.generate(
        **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
        pad_token_id=gpt_tokenizer.eos_token_id
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Interview Tips:**
- T5: most flexible (any task as text-to-text)
- BART: best for summarization (denoising pretraining)
- GPT: best for open-ended, creative generation
- Encoder-decoder better when input ≠ output structure
- Decoder-only scales better (GPT-3, GPT-4)

---

## Question 45
- [ ] Done
**How do you fine-tune large language models for specific generation tasks efficiently (LoRA, adapters)?**

**Answer:**

**Definition:**
Parameter-efficient fine-tuning (PEFT): update small portion of parameters instead of full model. LoRA (Low-Rank Adaptation) adds trainable low-rank matrices. Adapters insert small modules. Both reduce memory and enable multi-task with shared base.

**Comparison:**

| Method | Trainable Params | Memory | Quality |
|--------|------------------|--------|--------|
| Full fine-tuning | 100% | Very high | Baseline |
| LoRA | 0.1-1% | Low | ~Full FT |
| Adapters | 1-5% | Low | ~Full FT |
| Prefix tuning | 0.1% | Very low | Good |

**How LoRA Works:**
- Original weight: $W \in \mathbb{R}^{d \times k}$
- LoRA decomposition: $W' = W + BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$
- Only train $A$ and $B$ (freeze $W$), where $r << d, k$

**Python Code Example:**
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_lora_model(model_name, rank=8):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,              # Low-rank dimension
        lora_alpha=32,       # Scaling factor
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'],
        bias='none'
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # "trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.062"
    
    return peft_model

# Training with LoRA
from transformers import TrainingArguments, Trainer

def train_lora(peft_model, train_dataset):
    training_args = TrainingArguments(
        output_dir='./lora_model',
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Higher LR for LoRA
        num_train_epochs=3,
        fp16=True
    )
    
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    
    # Save only LoRA weights (~10MB vs 10GB+)
    peft_model.save_pretrained('./lora_weights')

# QLoRA: LoRA + 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config
)
# Then apply LoRA on top
```

**Interview Tips:**
- LoRA: rank 8-64 typically sufficient
- QLoRA: 4-bit base + LoRA = fine-tune 65B on 1 GPU
- Multiple LoRA adapters can share one base model
- Merge weights for deployment (no overhead)
- Alpha/rank ratio affects learning dynamics

---

## Question 46
- [ ] Done
**What is RLHF (Reinforcement Learning from Human Feedback) and how does it improve generation?**

**Answer:**

**Definition:**
RLHF trains language models using human preferences instead of just maximum likelihood. Process: 1) Pre-train on text, 2) Fine-tune supervised on demonstrations, 3) Train reward model from human comparisons, 4) Optimize policy with RL (PPO).

**RLHF Pipeline:**

| Step | Description |
|------|-------------|
| 1. Pre-training | Standard LM training on large corpus |
| 2. SFT | Supervised fine-tuning on demonstrations |
| 3. Reward Model | Train from human preference comparisons |
| 4. RL Optimization | PPO to maximize reward |

**Why RLHF:**

| Problem | How RLHF Helps |
|---------|---------------|
| Toxicity | Penalize harmful outputs |
| Helpfulness | Reward useful responses |
| Hallucination | Penalize factual errors |
| Following instructions | Reward adherence |

**Python Code Example:**
```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Step 1: Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward

# Training reward model from preferences
def train_reward_model(model, preferences):
    """preferences: [(prompt, chosen, rejected), ...]"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for prompt, chosen, rejected in preferences:
        reward_chosen = model(chosen)
        reward_rejected = model(rejected)
        
        # Bradley-Terry loss: P(chosen > rejected) = sigmoid(r_c - r_r)
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Step 2: PPO Training (using TRL library)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def setup_ppo():
    config = PPOConfig(
        model_name='gpt2',
        learning_rate=1.41e-5,
        batch_size=256,
        mini_batch_size=16
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
    
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    return ppo_trainer

def ppo_step(ppo_trainer, reward_model, prompts):
    # Generate responses
    responses = ppo_trainer.generate(prompts)
    
    # Get rewards
    rewards = reward_model(responses)
    
    # PPO update with KL penalty to stay close to ref model
    stats = ppo_trainer.step(prompts, responses, rewards)
    return stats
```

**Interview Tips:**
- RLHF key to ChatGPT, Claude success
- Reward model trained on preference pairs
- KL penalty prevents reward hacking
- PPO most common, but DPO (direct preference) emerging
- Constitutional AI: AI feedback instead of human

---

## Question 47
- [ ] Done
**How do you handle bias and fairness issues in text generation systems?**

**Answer:**

**Definition:**
Text generation models can produce biased, stereotypical, or harmful content reflecting training data. Address through: data curation, debiasing techniques, output filtering, RLHF, and ongoing monitoring.

**Types of Bias:**

| Bias Type | Example |
|-----------|--------|
| Gender | "Doctor" → assumes male |
| Racial | Stereotypical associations |
| Occupational | Linking gender to jobs |
| Religious | Negative associations |

**Mitigation Strategies:**

| Strategy | Description |
|----------|-------------|
| Data filtering | Remove biased training examples |
| Counterfactual augmentation | Balance with diverse examples |
| Output filtering | Block harmful outputs |
| RLHF | Train to avoid bias |
| Prompt engineering | Instruct for neutrality |

**Python Code Example:**
```python
# Bias detection
def detect_gender_bias(model, tokenizer):
    templates = [
        "The doctor said {} would help the patient.",
        "The nurse said {} would care for the patient.",
        "The engineer said {} would fix the problem."
    ]
    
    pronouns = ['he', 'she', 'they']
    
    for template in templates:
        probs = {}
        for pronoun in pronouns:
            text = template.format(pronoun)
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                loss = model(**inputs, labels=inputs['input_ids']).loss
            probs[pronoun] = torch.exp(-loss).item()
        
        print(f"{template}: {probs}")

# Counterfactual data augmentation
def counterfactual_augment(text, word_lists):
    """Swap gendered terms to create balanced data"""
    swaps = {
        'he': 'she', 'she': 'he',
        'him': 'her', 'her': 'him',
        'his': 'her', 'man': 'woman', 'woman': 'man'
    }
    
    words = text.split()
    augmented = [swaps.get(w.lower(), w) for w in words]
    return ' '.join(augmented)

# Output filtering
class BiasFilter:
    def __init__(self):
        self.blocked_patterns = [
            r'\b(stereotype|slur)\b',  # Example patterns
        ]
        self.toxicity_classifier = pipeline('text-classification', 
            model='unitary/toxic-bert')
    
    def filter(self, text):
        toxicity = self.toxicity_classifier(text)[0]
        if toxicity['label'] == 'toxic' and toxicity['score'] > 0.7:
            return "[Content filtered]"
        return text

# Debiased word embeddings
from gensim.models import Word2Vec
import numpy as np

def debias_embeddings(model, gender_words, target_words):
    """Remove gender direction from neutral words"""
    he_vec = model.wv['he']
    she_vec = model.wv['she']
    gender_direction = he_vec - she_vec
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    
    for word in target_words:
        if word in model.wv:
            vec = model.wv[word]
            # Remove gender component
            projection = np.dot(vec, gender_direction) * gender_direction
            model.wv[word] = vec - projection
    
    return model

# Fair generation with prompting
def fair_generation(model, prompt):
    fairness_instruction = """Generate a response that is:
    - Gender neutral when profession/role is not specified
    - Avoids stereotypes
    - Inclusive and respectful
    
    """
    return model.generate(fairness_instruction + prompt)
```

**Interview Tips:**
- Bias comes from training data
- Multiple intervention points: data, model, output
- Measure bias with benchmarks (WinoBias, StereoSet)
- RLHF can reduce but not eliminate bias
- Ongoing monitoring essential in production

---

## Question 48
- [ ] Done
**What are the key differences between autoregressive and non-autoregressive generation?**

**Answer:**

**Definition:**
Autoregressive (AR): generate tokens one at a time, each conditioned on previous. Non-autoregressive (NAR): generate all tokens in parallel. AR is higher quality but slower; NAR is faster but handles dependencies poorly.

**Comparison:**

| Aspect | Autoregressive | Non-Autoregressive |
|--------|---------------|--------------------|
| Generation | Sequential (O(n)) | Parallel (O(1)) |
| Speed | Slow | 10-15x faster |
| Quality | Higher | Lower |
| Dependencies | Models well | Conditional independence |

**NAR Challenges:**

| Challenge | Description |
|-----------|-------------|
| Multimodality | Multiple valid outputs |
| Token repetition | Common failure mode |
| Length prediction | Must predict length first |
| Dependency modeling | Tokens generated independently |

**Python Code Example:**
```python
import torch
import torch.nn as nn

# Autoregressive generation
class AutoregressiveDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerDecoder(...)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def generate(self, encoder_output, max_len):
        batch_size = encoder_output.size(0)
        tokens = torch.full((batch_size, 1), BOS_TOKEN)
        
        for _ in range(max_len):
            # Each step depends on all previous
            embedded = self.embed(tokens)
            output = self.transformer(embedded, encoder_output)
            logits = self.output(output[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if (next_token == EOS_TOKEN).all():
                break
        
        return tokens

# Non-autoregressive generation (simplified)
class NonAutoregressiveDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_len):
        super().__init__()
        self.length_predictor = nn.Linear(hidden_dim, max_len)
        self.transformer = nn.TransformerDecoder(...)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def generate(self, encoder_output):
        # Predict length
        length_logits = self.length_predictor(encoder_output.mean(dim=1))
        predicted_length = length_logits.argmax(dim=-1)
        
        # Generate all tokens in parallel
        positions = torch.arange(predicted_length.max())
        # All positions attend to encoder simultaneously
        decoder_input = self.create_decoder_input(predicted_length)
        output = self.transformer(decoder_input, encoder_output)
        logits = self.output(output)  # All positions at once
        tokens = logits.argmax(dim=-1)
        
        return tokens

# Iterative refinement NAR (improves quality)
class IterativeNAR(nn.Module):
    """CMLM: Conditionally Masked Language Model"""
    def generate(self, encoder_output, num_iterations=10):
        # Initial: all masked
        tokens = torch.full((batch_size, max_len), MASK_TOKEN)
        
        for iteration in range(num_iterations):
            # Predict all positions
            logits = self.forward(tokens, encoder_output)
            probs = torch.softmax(logits, dim=-1)
            
            # Unmask most confident predictions
            confidence = probs.max(dim=-1).values
            n_unmask = int((iteration + 1) / num_iterations * max_len)
            _, top_indices = confidence.topk(n_unmask, dim=-1)
            
            # Update tokens at confident positions
            new_tokens = logits.argmax(dim=-1)
            tokens.scatter_(1, top_indices, new_tokens.gather(1, top_indices))
        
        return tokens
```

**Interview Tips:**
- AR: GPT, BART. NAR: Mask-Predict, CMLM
- NAR useful for real-time translation
- Iterative refinement bridges quality gap
- Knowledge distillation helps NAR training
- Hybrid: AR for some tokens, NAR for rest

---

## Question 49
- [ ] Done
**How do you build production-ready generation systems (caching, batching, optimization)?**

**Answer:**

**Definition:**
Production systems need: low latency, high throughput, cost efficiency, and reliability. Optimize through caching, batching, model optimization (quantization, distillation), and infrastructure (GPU serving, auto-scaling).

**Key Optimizations:**

| Optimization | Benefit |
|--------------|--------|
| KV-cache | Avoid recomputing attention |
| Continuous batching | Higher GPU utilization |
| Quantization | Smaller, faster models |
| Speculative decoding | Faster AR generation |

**Architecture Components:**

| Component | Purpose |
|-----------|--------|
| Load balancer | Distribute requests |
| Request queue | Handle bursts |
| Model server | GPU inference |
| Response cache | Avoid recomputation |

**Python Code Example:**
```python
import torch
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor
import redis

# 1. Prompt caching with Redis
class PromptCache:
    def __init__(self, redis_host='localhost'):
        self.redis = redis.Redis(host=redis_host)
        self.ttl = 3600  # 1 hour
    
    def get_cache_key(self, prompt, params):
        content = f"{prompt}:{params}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt, params):
        key = self.get_cache_key(prompt, params)
        cached = self.redis.get(key)
        return cached.decode() if cached else None
    
    def set(self, prompt, params, response):
        key = self.get_cache_key(prompt, params)
        self.redis.setex(key, self.ttl, response)

# 2. Continuous batching (vLLM-style)
class ContinuousBatcher:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.pending_requests = []
    
    def add_request(self, request):
        self.pending_requests.append(request)
        if len(self.pending_requests) >= self.max_batch_size:
            return self.process_batch()
    
    def process_batch(self):
        # Batch inference
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        # Pad to same length for batching
        padded_inputs = self.pad_inputs([r.input_ids for r in batch])
        outputs = self.model.generate(padded_inputs)
        
        return outputs

# 3. KV-cache optimization
class KVCacheModel:
    def __init__(self, model):
        self.model = model
        self.kv_cache = None
    
    def generate_with_cache(self, input_ids, max_new_tokens):
        generated = input_ids
        
        for _ in range(max_new_tokens):
            if self.kv_cache is None:
                # First pass: compute full attention
                outputs = self.model(generated, use_cache=True)
            else:
                # Subsequent: only compute for new token
                outputs = self.model(
                    generated[:, -1:],
                    past_key_values=self.kv_cache,
                    use_cache=True
                )
            
            self.kv_cache = outputs.past_key_values
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

# 4. Model serving with TorchServe
class GenerationHandler:
    def initialize(self, context):
        self.model = AutoModelForCausalLM.from_pretrained(
            'model_path',
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained('model_path')
    
    def inference(self, data):
        prompt = data[0]['body']['prompt']
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0])

# 5. Speculative decoding (draft + verify)
class SpeculativeDecoder:
    def __init__(self, target_model, draft_model):
        self.target = target_model  # Large, accurate
        self.draft = draft_model    # Small, fast
    
    def generate(self, input_ids, k=4):
        # Draft generates k tokens quickly
        draft_tokens = self.draft.generate(input_ids, max_new_tokens=k)
        
        # Target verifies in parallel
        target_logits = self.target(draft_tokens)
        
        # Accept matching tokens, reject divergent
        # Potentially accept all k tokens in one target forward pass
        return accepted_tokens
```

**Interview Tips:**
- vLLM, TGI are production serving frameworks
- KV-cache essential for autoregressive models
- Continuous batching: 2-10x throughput
- Speculative decoding: 2-3x speedup
- Monitor latency percentiles (p50, p95, p99)

---

## Question 50
- [ ] Done
**What is prompt engineering for generation tasks and how do in-context learning approaches work?**

**Answer:**

**Definition:**
Prompt engineering: crafting input text to elicit desired outputs from LLMs without fine-tuning. In-context learning: providing examples in the prompt so model learns task pattern. Zero-shot (no examples), few-shot (some examples).

**Prompt Types:**

| Type | Description | Example |
|------|-------------|--------|
| Zero-shot | Just instruction | "Translate to French: Hello" |
| Few-shot | Examples + query | "cat→chat, dog→chien, bird→" |
| Chain-of-thought | Step-by-step reasoning | "Let's think step by step..." |
| Self-consistency | Multiple reasoning paths | Sample + majority vote |

**Prompt Engineering Techniques:**

| Technique | Purpose |
|-----------|--------|
| Clear instructions | Reduce ambiguity |
| Role assignment | "You are an expert..." |
| Output format | "Respond in JSON..." |
| Examples | Demonstrate task |
| Constraints | Limit response |

**Python Code Example:**
```python
from openai import OpenAI

client = OpenAI()

# Zero-shot prompting
def zero_shot_classify(text):
    prompt = f"""Classify the sentiment of this text as positive, negative, or neutral.

Text: {text}
Sentiment:"""
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

# Few-shot prompting
def few_shot_classify(text):
    prompt = f"""Classify the sentiment of each text.

Text: I love this product!
Sentiment: positive

Text: This is terrible, waste of money.
Sentiment: negative

Text: It arrived on time.
Sentiment: neutral

Text: {text}
Sentiment:"""
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

# Chain-of-thought prompting
def chain_of_thought(question):
    prompt = f"""Solve this step by step.

Question: {question}

Let's think through this carefully:
1."""
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

# Self-consistency: sample multiple times, majority vote
def self_consistency(question, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[{'role': 'user', 'content': question}],
            temperature=0.7  # Enable diversity
        )
        answers.append(response.choices[0].message.content)
    
    # Majority vote
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]

# Structured output with format instructions
def structured_extraction(text):
    prompt = f"""Extract entities from this text and return as JSON.

Text: {text}

Output format:
{{
    "people": ["name1", "name2"],
    "organizations": ["org1"],
    "locations": ["loc1"]
}}

JSON:"""
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

# System prompts for behavior control
def with_system_prompt(user_query):
    messages = [
        {'role': 'system', 'content': 'You are a helpful coding assistant. Provide concise code examples.'},
        {'role': 'user', 'content': user_query}
    ]
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=messages
    )
    return response.choices[0].message.content
```

**Interview Tips:**
- Few-shot examples should be diverse, representative
- Chain-of-thought helps reasoning tasks
- Temperature 0 for deterministic, >0 for diversity
- Order of examples can matter
- Prompt engineering is iterative experimentation

---


---

# --- Summarization & Text Generation Questions (from 08_nlp/10_summarization_text_generation) ---

# Summarization/Text Generation - Theory Questions

## Question 1
- [ ] Done
**How do you balance faithfulness and fluency in abstractive text summarization?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What techniques work best for controlling summary length and level of detail?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you handle summarization for documents with multiple topics or themes?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies help with generating summaries that preserve important factual information?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you implement domain-specific summarization for technical or specialized texts?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What approaches work best for multi-document summarization with conflicting information?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle summarization quality assessment without gold standard summaries?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What techniques help with explaining summarization decisions and content selection?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement query-focused summarization for specific information needs?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What strategies work best for summarization in different languages and cultural contexts?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle summarization bias and ensure balanced representation of viewpoints?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What approaches help with summarization robustness against input variations?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement knowledge distillation for compressing large summarization models?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques work best for real-time summarization with low latency requirements?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle summarization for streaming or continuously updated content?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What strategies help with summarization consistency across different content types?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement online learning for summarization models adapting to user preferences?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What approaches work best for summarization in conversational or interactive contexts?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle summarization optimization for specific downstream applications?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help with summarization for content requiring expert domain knowledge?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement fairness-aware summarization to reduce demographic bias?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What strategies work best for hierarchical or structured summarization?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle summarization quality control with subjective content evaluation?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What approaches help with summarization for low-resource or minority languages?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving summarization for sensitive documents?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques work best for summarization with external knowledge integration?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle summarization adaptation to emerging content formats?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What strategies help with summarization requiring factual accuracy verification?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement robust error handling when summarization fails or produces poor output?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What approaches work best for combining summarization with other text processing tasks?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle summarization for content with varying importance levels?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What techniques help with summarization consistency in distributed processing scenarios?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement efficient batch processing for large-scale summarization applications?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for summarization with regulatory compliance requirements?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle summarization for content requiring high accuracy and reliability?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with summarization customization for different user needs?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement monitoring and quality drift detection for summarization systems?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for summarization of multimedia or multi-modal content?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle summarization optimization when balancing conciseness and completeness?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with summarization for emerging content types and platforms?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement transfer learning for summarization across different domains?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for summarization with minimal computational resources?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle summarization integration with information management workflows?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with summarization for content requiring creative or editorial judgment?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement controllable summarization with style and perspective adjustment?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for summarization in high-volume content processing scenarios?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle summarization quality benchmarking across different model types?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with summarization for evolving information landscapes?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you implement efficient memory management for large-scale summarization inference?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What techniques work best for balancing summarization automation with human oversight?**
**Answer:** _To be filled_

---


---

# --- Prompt Engineering Questions (from 08_nlp/11_prompt_engineering) ---

# Prompt Engineering - Theory Questions

## Question 1
- [ ] Done
**How do you design prompts that consistently elicit desired behaviors from different LLM architectures?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What techniques work best for chain-of-thought prompting in complex reasoning tasks?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement few-shot prompting strategies with optimal example selection?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies help with prompt optimization for domain-specific applications?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle prompt engineering for multilingual or cross-cultural contexts?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What approaches work best for reducing prompt sensitivity and improving robustness?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you design prompts that minimize harmful or biased outputs?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What techniques help with measuring and evaluating prompt effectiveness?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement automated prompt generation and optimization techniques?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What strategies work best for prompt engineering in conversational AI systems?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle prompt versioning and management in production systems?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What approaches help with prompt engineering for specific output formats and structures?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement meta-prompting strategies for complex multi-step tasks?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques work best for prompt engineering with limited computational resources?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle prompt adaptation to different user expertise levels?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What strategies help with prompt consistency across different model versions?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement dynamic prompting that adapts to context and user needs?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What approaches work best for prompt engineering in specialized domains?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle prompt optimization for specific quality metrics?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help with prompt engineering for creative and open-ended tasks?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement fairness-aware prompting to reduce demographic bias?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What strategies work best for prompt engineering with safety and alignment constraints?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle prompt quality assessment with subjective or creative outputs?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What approaches help with prompt engineering for low-resource languages?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving prompt engineering for sensitive applications?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques work best for prompt engineering with external tool integration?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle prompt adaptation to emerging model capabilities?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What strategies help with prompt engineering requiring domain expert validation?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement robust error handling when prompts produce unexpected outputs?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What approaches work best for combining prompt engineering with fine-tuning strategies?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle prompt engineering for tasks with multiple valid approaches?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What techniques help with prompt consistency in distributed AI applications?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement efficient prompt testing and validation pipelines?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for prompt engineering with regulatory compliance needs?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle prompt engineering for applications requiring high reliability?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with prompt customization for different user preferences?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement monitoring and performance tracking for prompt-based systems?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for prompt engineering with structured data inputs?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle prompt optimization when balancing performance and interpretability?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with prompt engineering for emerging AI capabilities?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement transfer learning for prompts across different models?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for prompt engineering with resource constraints?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle prompt integration with broader AI system architectures?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with prompt engineering for tasks requiring factual accuracy?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement controllable prompting with adjustable parameters?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for prompt engineering in high-volume applications?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle prompt quality benchmarking across different LLMs?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with prompt engineering for evolving use cases?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you implement efficient prompt caching and optimization?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What techniques work best for balancing prompt complexity with model capabilities?**
**Answer:** _To be filled_

---


---

# --- RAG Questions (from 08_nlp/12_rag) ---

# Retrieval-Augmented Generation (RAG) - Theory Questions

## Question 1
- [ ] Done
**How do you design RAG systems that effectively balance retrieval relevance and generation quality?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What techniques work best for chunking and indexing documents in RAG knowledge bases?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement dense retrieval methods that capture semantic similarity effectively?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies help with handling multi-modal RAG systems combining text, images, and structured data?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you design RAG architectures that maintain factual accuracy and reduce hallucination?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What approaches work best for real-time RAG with low latency retrieval and generation?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle RAG system evaluation when ground truth answers aren't available?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What techniques help with explaining RAG decisions and source attribution?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement dynamic retrieval strategies that adapt to query complexity?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What strategies work best for RAG in specialized domains with technical knowledge bases?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle RAG quality control and confidence scoring for generated answers?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What approaches help with RAG robustness against noisy or outdated knowledge sources?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement knowledge distillation for compressing large RAG models?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques work best for multilingual RAG systems across different languages?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle RAG knowledge base updates and maintaining retrieval consistency?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What strategies help with RAG personalization for different user contexts and preferences?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement online learning for RAG systems adapting to new information?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What approaches work best for conversational RAG with context tracking?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle RAG optimization for specific enterprise applications?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help with RAG for questions requiring temporal or current information?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement fairness-aware RAG to reduce bias in retrieved information?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What strategies work best for RAG with structured knowledge graphs and databases?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle RAG quality assessment with conflicting information sources?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What approaches help with RAG for low-resource domains with limited knowledge bases?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving RAG for sensitive organizational knowledge?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques work best for RAG systems requiring high factual precision?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle RAG adaptation to emerging topics and evolving knowledge?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What strategies help with RAG requiring domain expert validation and oversight?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement robust error handling when RAG retrieval fails or returns poor results?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What approaches work best for combining RAG with other AI system components?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle RAG for queries requiring multiple information sources?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What techniques help with RAG consistency in distributed deployment scenarios?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement efficient batch processing for large-scale RAG applications?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for RAG with regulatory compliance and audit requirements?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle RAG for applications requiring high accuracy and reliability?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with RAG customization for different organizational knowledge structures?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement monitoring and performance tracking for RAG systems?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for RAG handling semi-structured and unstructured data?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle RAG optimization when balancing retrieval quality and generation fluency?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with RAG for emerging enterprise AI applications?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement transfer learning for RAG across different knowledge domains?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for RAG with minimal computational and storage resources?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle RAG integration with existing enterprise search and knowledge systems?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with RAG for applications requiring creative or analytical outputs?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement controllable RAG with adjustable retrieval scope and generation style?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for RAG in high-concurrency and multi-user environments?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle RAG quality benchmarking across different system architectures?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with RAG for rapidly evolving information landscapes?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you implement efficient vector database management for RAG knowledge bases?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What techniques work best for balancing RAG system complexity with maintainability?**
**Answer:** _To be filled_

---
