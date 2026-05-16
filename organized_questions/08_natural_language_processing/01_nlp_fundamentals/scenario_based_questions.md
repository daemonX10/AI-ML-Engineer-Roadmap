# Nlp Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Define 'sentiment analysis' and discuss its applications.**

**Answer:**

**Definition:**
Sentiment analysis (opinion mining) automatically determines the emotional tone or attitude expressed in text — positive, negative, or neutral. It can be extended to fine-grained emotions (joy, anger, sadness) or aspect-level opinions.

**Levels of Sentiment Analysis:**

| Level | Description | Example |
|-------|-------------|---------|
| Document-level | Overall sentiment of entire text | "This movie is great" → Positive |
| Sentence-level | Sentiment per sentence | Each review sentence rated |
| Aspect-level | Sentiment per feature/aspect | "Battery great, screen bad" → Battery: +, Screen: − |
| Emotion detection | Specific emotions | "I'm furious about the delay" → Anger |

**Applications:**

| Domain | Application |
|--------|-------------|
| Business | Brand monitoring, customer feedback analysis |
| Finance | Stock market prediction from news/social media |
| Healthcare | Patient satisfaction, mental health monitoring |
| Politics | Public opinion tracking, election prediction |
| E-commerce | Product review summarization, recommendation |
| Social media | Trend detection, crisis management |

**Python Code Example:**
```python
# Pipeline: Text -> Preprocess -> Feature extraction -> Classify sentiment

from transformers import pipeline

# Using pre-trained model
sentiment_analyzer = pipeline("sentiment-analysis")

texts = [
    "I absolutely love this product, it exceeded my expectations!",
    "Terrible experience, worst customer service ever.",
    "The food was okay, nothing special but not bad either."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text[:50]}... -> {result['label']} ({result['score']:.3f})")

# Aspect-based sentiment (simplified)
def extract_aspects(review, aspects):
    """Extract sentiment per aspect"""
    results = {}
    sentences = review.split('.')
    for aspect in aspects:
        for sent in sentences:
            if aspect.lower() in sent.lower():
                result = sentiment_analyzer(sent.strip())[0]
                results[aspect] = result['label']
    return results

review = "The battery life is excellent. The camera quality is disappointing. The design looks premium."
aspects = ['battery', 'camera', 'design']
print(extract_aspects(review, aspects))
# {'battery': 'POSITIVE', 'camera': 'NEGATIVE', 'design': 'POSITIVE'}
```

**Interview Tips:**
- Mention the three levels: document, sentence, aspect-based
- Challenges: sarcasm, negation ("not bad"), context-dependent sentiment
- Traditional approaches: lexicon-based (VADER), ML (SVM + TF-IDF), Deep Learning (BERT)
- VADER is good for social media; BERT-based models are state-of-the-art

---

## Question 2
- [ ] Done

**Discuss the role ofSupport Vector Machines (SVM)intext classification.**

**Answer:**

**Definition:**
SVMs find the optimal hyperplane that maximally separates classes in feature space. For text classification, SVMs work exceptionally well with high-dimensional TF-IDF/BoW features because they handle sparse, high-dimensional data effectively and generalize well even with limited training data.

**Why SVMs Excel at Text Classification:**

| Property | Benefit for Text |
|----------|-----------------|
| High-dimensional handling | TF-IDF creates 10K-100K features; SVM handles this natively |
| Sparse data tolerance | Most words are zero in any document; SVM thrives here |
| Margin maximization | Good generalization with limited labeled data |
| Kernel trick | Captures non-linear patterns (RBF, polynomial) |
| Regularization | Prevents overfitting on noisy text data |

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| Hyperplane | Decision boundary: $\mathbf{w} \cdot \mathbf{x} + b = 0$ |
| Support vectors | Data points closest to the hyperplane |
| Margin | Distance between hyperplane and support vectors |
| Kernel | Maps data to higher dimensions (linear, RBF, polynomial) |
| C parameter | Trade-off between margin width and misclassification |

**Python Code Example:**
```python
# Pipeline: Text corpus -> TF-IDF vectorize -> SVM train -> Classify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Sample data
texts = [
    "stock market gains profits trading", "economy GDP growth rates",
    "quarterback touchdown football game", "basketball court slam dunk",
    "neural network deep learning AI", "machine learning algorithm model",
    "election vote president campaign", "congress senate legislation law",
]
labels = ['finance', 'finance', 'sports', 'sports', 'tech', 'tech', 'politics', 'politics']

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# LinearSVC (faster for text, equivalent to SVM with linear kernel)
svm_model = LinearSVC(C=1.0, max_iter=10000)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'max_iter': [5000]}
grid_search = GridSearchCV(LinearSVC(), param_grid, cv=3, scoring='f1_macro')
grid_search.fit(X_train, y_train)
print(f"Best C: {grid_search.best_params_['C']}")
```

**SVM vs Other Classifiers for Text:**

| Classifier | When to Use |
|-----------|-------------|
| LinearSVC | Default choice; fast, accurate on text |
| SVM (RBF) | When non-linear boundaries needed (slower) |
| Naive Bayes | Very small data; fast baseline |
| Logistic Regression | When probability estimates needed |
| BERT | When accuracy matters most; large data available |

**Interview Tips:**
- LinearSVC is preferred over SVC for text (scales better to large datasets)
- SVM with linear kernel often matches or beats non-linear kernels for text
- TF-IDF + SVM was state-of-the-art before deep learning (still a strong baseline)
- Key hyperparameter: C (regularization); lower C = wider margin, more misclassification allowed
- For multi-class: SVM uses one-vs-rest (OVR) or one-vs-one (OVO) strategies

---

## Question 3
- [ ] Done

**Discuss the concept ofsemantic similarityand its computational approaches in NLP.**

**Answer:**

**Definition:**
Semantic similarity measures how close two text units are in meaning, even if they use different words. "Car" and "automobile" are semantically similar despite being different strings. This enables search, paraphrase detection, question answering, and deduplication.

**Approaches:**

| Approach | Method | Quality | Speed |
|----------|--------|---------|-------|
| Lexical | Exact/fuzzy string matching | Low | Fast |
| Knowledge-based | WordNet path/similarity | Medium | Fast |
| Distributional | Word2Vec, GloVe cosine distance | Good | Fast |
| Contextual | BERT, Sentence-BERT embeddings | Best | Slower |
| Hybrid | Combine multiple signals | Best | Varies |

**Key Methods:**

| Method | Description |
|--------|-------------|
| Jaccard similarity | $\frac{\|A \cap B\|}{\|A \cup B\|}$ (word overlap) |
| WordNet Wu-Palmer | Path-based similarity in taxonomy |
| Word Mover's Distance | Minimum transport between word embeddings |
| Sentence-BERT | Dense sentence embeddings + cosine similarity |
| Cross-encoder | Jointly encodes both sentences (most accurate) |

**Python Code Example:**
```python
# Pipeline: Two texts -> Multiple similarity methods -> Compare results

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Method 1: Sentence embeddings (best general-purpose approach)
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """Compute semantic similarity using sentence embeddings"""
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

pairs = [
    ("The cat sits on the mat", "A feline is resting on the rug"),
    ("The cat sits on the mat", "The stock market crashed today"),
    ("How old are you?", "What is your age?"),
]

for t1, t2 in pairs:
    sim = semantic_similarity(t1, t2)
    print(f"{sim:.3f}: '{t1}' vs '{t2}'")
# ~0.75, ~0.05, ~0.85

# Method 2: WordNet similarity (knowledge-based)
from nltk.corpus import wordnet as wn

def wordnet_similarity(word1, word2):
    """Wu-Palmer similarity using WordNet"""
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if not synsets1 or not synsets2:
        return 0.0
    return max(
        s1.wup_similarity(s2) or 0
        for s1 in synsets1 for s2 in synsets2
    )

print(f"car-automobile: {wordnet_similarity('car', 'automobile'):.3f}")
print(f"car-bicycle: {wordnet_similarity('car', 'bicycle'):.3f}")
```

**Interview Tips:**
- Sentence-BERT (bi-encoder) is best for large-scale similarity search
- Cross-encoder is more accurate but O(n²) — use for re-ranking
- Word Mover's Distance is elegant but slow (Earth Mover's Distance on embeddings)
- Cosine similarity on TF-IDF is a fast baseline
- Semantic similarity != textual entailment (similarity is symmetric, entailment is not)

---

## Question 4
- [ ] Done

**Discuss the use ofConditional Random Fields (CRF)in sequence modeling for NLP.**

**Answer:**

**Definition:**
CRFs are discriminative probabilistic models for labeling sequences. Unlike HMMs (generative), CRFs model $P(Y|X)$ directly, considering the full input context. They're widely used for NER, POS tagging, and chunking, often as a layer on top of BiLSTM or BERT.

**CRF vs Other Sequence Models:**

| Model | Type | Context | Strengths |
|-------|------|---------|-----------|
| HMM | Generative | Limited | Simple, fast |
| MEMM | Discriminative | One direction | Rich features |
| CRF | Discriminative | Full sequence | Global optimization |
| BiLSTM-CRF | Neural + CRF | Full + learned | State-of-the-art |

**Why CRF for NER/POS:**

| Advantage | Explanation |
|-----------|-------------|
| Label dependencies | Models transitions (B-PER → I-PER valid, B-PER → I-LOC invalid) |
| Global normalization | Avoids label bias problem of MEMMs |
| Rich features | Uses arbitrary features of input |
| Sequence-level optimization | Optimizes entire label sequence, not individual tokens |

**Mathematical Formulation:**
$$P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^{T} \sum_k \lambda_k f_k(y_{t-1}, y_t, X, t)\right)$$

Where $f_k$ are feature functions and $Z(X)$ is the partition function.

**Python Code Example:**
```python
# Pipeline: Tokens -> Feature extraction -> CRF train -> Sequence labeling

import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word_to_features(sent, i):
    """Extract features for a word in context"""
    word = sent[i]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        prev_word = sent[i - 1]
        features.update({'-1:word.lower()': prev_word.lower(), '-1:word.istitle()': prev_word.istitle()})
    else:
        features['BOS'] = True  # Beginning of sentence
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({'+1:word.lower()': next_word.lower(), '+1:word.istitle()': next_word.istitle()})
    else:
        features['EOS'] = True  # End of sentence
    return features

def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

# Training data: (sentence, labels) pairs
train_sents = [
    (['John', 'lives', 'in', 'New', 'York'], ['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']),
    (['Apple', 'released', 'a', 'new', 'iPhone'], ['B-ORG', 'O', 'O', 'O', 'B-PROD']),
]

X_train = [sent_to_features(s) for s, _ in train_sents]
y_train = [labels for _, labels in train_sents]

# Train CRF
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,    # L1 regularization
    c2=0.1,    # L2 regularization
    max_iterations=100
)
crf.fit(X_train, y_train)

# Predict
test_sent = ['Mary', 'works', 'at', 'Google']
features = sent_to_features(test_sent)
prediction = crf.predict([features])[0]
print(list(zip(test_sent, prediction)))
# [('Mary', 'B-PER'), ('works', 'O'), ('at', 'O'), ('Google', 'B-ORG')]
```

**Interview Tips:**
- BiLSTM-CRF was NER standard before BERT; BERT+CRF still adds 1-2% F1
- CRF layers enforce valid label transitions (no "I-PER" after "B-LOC")
- Viterbi algorithm finds the best label sequence in O(T × K²) time
- CRF addresses the "label bias" problem that affects MEMMs
- Feature engineering matters for CRF (context windows, prefixes, capitalization)

---

## Question 5
- [ ] Done

**Discuss strategies for dealing withslangandabbreviationsin text processing.**

**Answer:**

**Definition:**
Slang ("gonna", "srsly") and abbreviations ("NLP", "brb") deviate from standard language, causing OOV issues, tokenization errors, and reduced model accuracy. Strategies include normalization, custom dictionaries, and robust model training.

**Types of Non-Standard Text:**

| Type | Examples | Challenge |
|------|----------|-----------|
| Abbreviations | NLP, API, brb, ASAP | Expansion needed |
| Slang | gonna, wanna, lit, salty | Informal vocabulary |
| Acronyms | LOL, FWIW, IMO | Context-dependent meaning |
| Elongation | sooooo, nooooo | Normalization needed |
| Phonetic spelling | u, r, 2, 4 | Character-level patterns |

**Strategies:**

| Strategy | Description |
|----------|-------------|
| Dictionary lookup | Map slang → standard form |
| Regex normalization | Collapse repeated chars, expand contractions |
| Subword tokenization | BPE handles unseen words naturally |
| Data augmentation | Include noisy text in training data |
| Pre-trained embeddings | Models trained on social media (BERTweet) |

**Python Code Example:**
```python
# Pipeline: Raw noisy text -> Normalize -> Clean -> Process

import re

class TextNormalizer:
    def __init__(self):
        self.slang_dict = {
            'gonna': 'going to', 'wanna': 'want to', 'gotta': 'got to',
            'u': 'you', 'r': 'are', 'ur': 'your', 'b4': 'before',
            'brb': 'be right back', 'lol': 'laughing out loud',
            'imo': 'in my opinion', 'tbh': 'to be honest',
            'smh': 'shaking my head', 'fwiw': 'for what it is worth',
        }
        self.contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "i'm": "i am", "it's": "it is", "he's": "he is",
            "they're": "they are", "we're": "we are",
        }
    
    def normalize(self, text):
        text = text.lower()
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        # Reduce elongation: "soooo" -> "so"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        # Replace slang
        words = text.split()
        words = [self.slang_dict.get(w, w) for w in words]
        return ' '.join(words)

normalizer = TextNormalizer()

texts = [
    "u r gonna love this!! sooooo good lol",
    "brb, gotta go. tbh this was amazing",
    "can't believe this... smh it's terrible"
]

for t in texts:
    print(f"Original: {t}")
    print(f"Cleaned:  {normalizer.normalize(t)}\n")

# For social media: use BERTweet (pre-trained on tweets)
# from transformers import AutoModel, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
# model = AutoModel.from_pretrained("vinai/bertweet-base")
```

**Interview Tips:**
- Subword tokenizers (BPE/WordPiece) handle many slang words naturally
- BERTweet and TweetEval are pre-trained on social media text
- Normalization can lose sentiment signals ("AMAZING" vs "amazing")
- Context matters: "lit" means different things in different contexts
- Build domain-specific dictionaries for your use case

---

## Question 6
- [ ] Done

**How would you build a chatbot using NLP principles?**

**Answer:**

**Definition:**
An NLP chatbot understands user intent, extracts relevant entities, manages conversation context, and generates appropriate responses. Modern chatbots use intent classification + entity extraction (task-oriented) or generative LLMs (open-domain).

**Chatbot Architecture:**

| Component | Purpose | Technology |
|-----------|---------|------------|
| NLU (Understanding) | Intent + entity extraction | BERT, Rasa NLU |
| Dialog Manager | Track state, decide next action | Rule-based / RL |
| NLG (Generation) | Produce response text | Templates / LLM |
| Context Manager | Track conversation history | Memory / database |

**Types of Chatbots:**

| Type | Approach | Best For |
|------|----------|----------|
| Rule-based | Pattern matching, decision trees | FAQs, simple flows |
| Retrieval-based | Match query to response database | Customer support |
| Generative | Seq2seq / LLM generation | Open-domain conversation |
| Hybrid | Intent + retrieval + generation | Production systems |

**Python Code Example:**
```python
# Pipeline: User input -> Intent classification -> Entity extraction -> Response generation

from transformers import pipeline

class SimpleNLPChatbot:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")
        self.ner = pipeline("ner", grouped_entities=True)
        self.intents = ["greeting", "booking", "complaint", "information", "goodbye"]
        self.context = {}
    
    def understand(self, user_input):
        """NLU: Extract intent and entities"""
        intent_result = self.classifier(user_input, candidate_labels=self.intents)
        intent = intent_result['labels'][0]
        confidence = intent_result['scores'][0]
        entities = self.ner(user_input)
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'raw_text': user_input
        }
    
    def generate_response(self, nlu_result):
        """Generate response based on intent"""
        intent = nlu_result['intent']
        responses = {
            'greeting': "Hello! How can I help you today?",
            'booking': "I'd be happy to help with a booking. What date works for you?",
            'complaint': "I'm sorry to hear that. Let me escalate this for you.",
            'information': "Let me find that information for you.",
            'goodbye': "Thank you! Have a great day!",
        }
        return responses.get(intent, "I'm not sure I understand. Could you rephrase?")
    
    def chat(self, user_input):
        """Full chatbot pipeline"""
        nlu = self.understand(user_input)
        response = self.generate_response(nlu)
        self.context['last_intent'] = nlu['intent']
        return response

bot = SimpleNLPChatbot()
test_inputs = ["Hello!", "I want to book a table for Friday", "The service was terrible"]
for inp in test_inputs:
    print(f"User: {inp}")
    print(f"Bot: {bot.chat(inp)}\n")
```

**Production Chatbot Stack:**

| Layer | Tools |
|-------|-------|
| NLU | Rasa, Dialogflow, LUIS, custom BERT |
| Dialog Management | Rasa Core, state machines |
| Backend | REST APIs, databases |
| Frontend | Web widget, Slack, WhatsApp |
| Analytics | Conversation logs, intent accuracy |

**Interview Tips:**
- Start with intent classification + entity extraction (task-oriented approach)
- Use Rasa for open-source production chatbots
- LLM-based chatbots (GPT-4) excel at open-domain but need guardrails
- Key metrics: intent accuracy, task completion rate, user satisfaction
- Always have a fallback/handoff-to-human mechanism

---

## Question 7
- [ ] Done

**Propose an NLP solution for detectingfake news articles.**

**Answer:**

**Definition:**
Fake news detection classifies news articles as real or fabricated. It combines content analysis (linguistic features, writing style), source credibility, claim verification, and propagation patterns to identify misinformation.

**Multi-Signal Approach:**

| Signal | Features | Method |
|--------|----------|--------|
| Linguistic | Writing style, sensationalism, emotion | Text classification |
| Content | Factual claims, contradictions | Fact-checking NLP |
| Source | Author credibility, domain reputation | Knowledge base |
| Propagation | Spread patterns, bot detection | Network analysis |
| Temporal | Timing relative to events | Time series |

**NLP Features for Detection:**

| Feature Category | Examples |
|-----------------|----------|
| Lexical | Sensational words, hedging phrases, emotional language |
| Syntactic | Sentence complexity, passive voice usage |
| Semantic | Claim consistency, factual accuracy |
| Stylistic | Clickbait patterns, ALL CAPS, excessive punctuation |
| Source | Author history, publication bias |

**Python Code Example:**
```python
# Pipeline: Article -> Feature extraction -> Classification -> Credibility score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import re
import numpy as np

class FakeNewsDetector:
    def __init__(self):
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
            ('clf', GradientBoostingClassifier(n_estimators=200))
        ])
    
    def extract_style_features(self, text):
        """Extract linguistic style features"""
        return {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'num_sentences': len(re.split(r'[.!?]', text)),
            'has_clickbait': int(bool(re.search(
                r'you won\'t believe|shocking|breaking|exclusive', text, re.IGNORECASE
            ))),
        }
    
    def train(self, articles, labels):
        """Train on labeled articles (0=fake, 1=real)"""
        self.text_pipeline.fit(articles, labels)
    
    def predict(self, article):
        """Predict credibility"""
        prediction = self.text_pipeline.predict([article])[0]
        probability = self.text_pipeline.predict_proba([article])[0]
        style = self.extract_style_features(article)
        
        return {
            'label': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'style_flags': {k: v for k, v in style.items() if v > 0.5 or isinstance(v, int) and v > 2}
        }

# Claim verification with NLI (Natural Language Inference)
from transformers import pipeline as hf_pipeline

def verify_claim(claim, evidence):
    """Check if evidence supports or contradicts claim"""
    nli = hf_pipeline("text-classification", model="roberta-large-mnli")
    result = nli(f"{evidence} </s></s> {claim}")[0]
    return result  # ENTAILMENT, CONTRADICTION, or NEUTRAL
```

**Interview Tips:**
- Multi-modal approach is essential (text alone is insufficient)
- BERT + style features outperform text-only models
- Natural Language Inference (NLI) can verify claims against evidence
- Real-world challenges: satire vs fake news, evolving tactics
- Datasets: LIAR, FakeNewsNet, FEVER for fact verification

---

## Question 8
- [ ] Done

**How would you design a voice-activated assistant like Siri or Alexa with NLP technology?**

**Answer:**

**Definition:**
A voice assistant combines Automatic Speech Recognition (ASR), Natural Language Understanding (NLU), Dialog Management, and Text-to-Speech (TTS) to enable spoken interaction. The pipeline converts speech to text, understands intent, performs actions, and responds verbally.

**Architecture Pipeline:**

| Stage | Component | Technology |
|-------|-----------|------------|
| 1. Wake word | "Hey Siri" detection | Small keyword model (always on) |
| 2. ASR | Speech → Text | Whisper, DeepSpeech, cloud APIs |
| 3. NLU | Intent + Entity extraction | BERT, Rasa, slot filling |
| 4. Dialog Manager | Track context, decide action | State machine / RL |
| 5. Action Executor | Call APIs, query databases | Backend services |
| 6. NLG | Generate response text | Templates / LLM |
| 7. TTS | Text → Speech | WaveNet, VITS, cloud TTS |

**Key NLP Components:**

| Component | Challenge | Solution |
|-----------|-----------|----------|
| Intent classification | "Set alarm for 7 AM" → SetAlarm intent | Fine-tuned BERT |
| Slot filling | Extract "7 AM" as time entity | Sequence labeling (NER) |
| Coreference | "Play it again" → resolve "it" | Context tracking |
| Multi-turn dialog | Follow-up questions | Dialog state tracking |
| Error recovery | Handle misrecognition | Confirmation strategies |

**Python Code Example:**
```python
# Pipeline: Audio -> ASR -> NLU -> Action -> TTS

import whisper
from transformers import pipeline

class VoiceAssistant:
    def __init__(self):
        # ASR: Speech to Text
        self.asr_model = whisper.load_model("base")
        # NLU: Intent + Entity
        self.classifier = pipeline("zero-shot-classification")
        self.ner = pipeline("ner", grouped_entities=True)
        
        self.intents = [
            "set_alarm", "play_music", "weather_query",
            "send_message", "search_web", "control_device"
        ]
        self.dialog_state = {}
    
    def transcribe(self, audio_path):
        """ASR: Convert speech to text"""
        result = self.asr_model.transcribe(audio_path)
        return result["text"]
    
    def understand(self, text):
        """NLU: Extract intent and entities"""
        intent = self.classifier(text, candidate_labels=self.intents)
        entities = self.ner(text)
        
        return {
            'intent': intent['labels'][0],
            'confidence': intent['scores'][0],
            'entities': [{'text': e['word'], 'type': e['entity_group']} for e in entities],
            'slots': self._extract_slots(text, intent['labels'][0])
        }
    
    def _extract_slots(self, text, intent):
        """Extract intent-specific slots"""
        import re
        slots = {}
        if intent == 'set_alarm':
            time_match = re.search(r'\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?', text)
            if time_match:
                slots['time'] = time_match.group()
        elif intent == 'play_music':
            slots['query'] = text.replace('play', '').strip()
        return slots
    
    def execute(self, nlu_result):
        """Execute action based on NLU result"""
        intent = nlu_result['intent']
        slots = nlu_result['slots']
        
        actions = {
            'set_alarm': f"Setting alarm for {slots.get('time', 'unknown time')}",
            'play_music': f"Playing {slots.get('query', 'music')}",
            'weather_query': "Fetching weather forecast...",
        }
        return actions.get(intent, "I'm not sure how to help with that.")
    
    def process(self, audio_path):
        """Full pipeline"""
        text = self.transcribe(audio_path)
        nlu = self.understand(text)
        response = self.execute(nlu)
        return {'transcription': text, 'nlu': nlu, 'response': response}
```

**Design Considerations:**

| Aspect | Detail |
|--------|--------|
| Latency | ASR + NLU should complete in <500ms for natural interaction |
| Privacy | On-device processing (wake word), minimal cloud data |
| Multi-language | Multilingual ASR + NLU models |
| Personalization | Learn user preferences, voice profiles |
| Error handling | "Did you mean...?" confirmation dialogs |

**Interview Tips:**
- Wake word detection runs locally (low-power, always-on model)
- ASR: Whisper (OpenAI) is state-of-the-art open-source
- Streaming ASR processes audio chunks for real-time response
- On-device NLU (DistilBERT, TFLite) for privacy-sensitive applications
- Key challenge: noisy environments, accents, ambiguous commands

---
