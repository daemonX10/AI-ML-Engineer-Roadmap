# NLP Fundamentals (Tokenization, NER, POS Tagging) - Theory Questions

## Core Questions

## Question 1
- [ ] Done
**What is tokenization and why is it the fundamental first step in NLP pipelines?**

**Answer:**

**Definition:**
Tokenization is the process of breaking raw text into smaller meaningful units called tokens (words, subwords, or characters). It is the fundamental first step because all downstream NLP models operate on numerical representations of tokens, not raw text strings.

**Core Concepts:**
- Token: A discrete unit of text (word, subword, character, or symbol)
- Vocabulary: The set of all unique tokens the model recognizes
- Token ID: Integer mapping of each token for model input
- Whitespace tokenization: Splitting by spaces (simplest approach)
- Subword tokenization: Breaking words into smaller frequent units (BPE, WordPiece)

**Why Fundamental:**
- Converts unstructured text → structured numerical input
- Defines the vocabulary size and model's understanding capacity
- Affects OOV handling, sequence length, and memory usage
- Poor tokenization = poor model performance downstream

**Practical Relevance:**
- Every NLP pipeline starts with tokenization (classification, NER, translation, QA)
- Choice of tokenizer affects model accuracy, inference speed, and memory
- Pre-trained models require exact same tokenizer used during training

**Python Code Example:**
```python
# Pipeline: Raw text -> Tokens -> Token IDs -> Model input

from transformers import AutoTokenizer

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 2: Tokenize text
text = "Machine learning is amazing!"
tokens = tokenizer.tokenize(text)
# Output: ['machine', 'learning', 'is', 'amazing', '!']

# Step 3: Convert to IDs
token_ids = tokenizer.encode(text)
# Output: [101, 3698, 4083, 2003, 6429, 999, 102]
# 101=[CLS], 102=[SEP]

# Step 4: Full encoding for model
encoded = tokenizer(text, return_tensors="pt")
# Output: {'input_ids': tensor, 'attention_mask': tensor}
```

**Interview Tips:**
- Always mention that tokenization choice depends on task and language
- Know difference between word-level, subword, and character-level
- BERT uses WordPiece, GPT uses BPE - know this distinction

---

## Question 2
- [ ] Done
**Explain the difference between stemming and lemmatization with examples.**

**Answer:**

**Definition:**
Stemming reduces words to their root form by chopping off suffixes using crude heuristic rules. Lemmatization reduces words to their dictionary base form (lemma) using vocabulary and morphological analysis. Lemmatization is linguistically accurate; stemming is faster but less precise.

**Core Concepts:**
| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Approach | Rule-based suffix stripping | Dictionary + morphological analysis |
| Output | May not be valid word | Always valid dictionary word |
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| POS needed | No | Often yes |

**Examples:**
| Word | Stemming (Porter) | Lemmatization |
|------|-------------------|---------------|
| running | runn | run |
| studies | studi | study |
| better | better | good |
| caring | car | care |
| wolves | wolv | wolf |

**When to Use:**
- **Stemming:** Information retrieval, search engines, when speed matters
- **Lemmatization:** Sentiment analysis, chatbots, when meaning matters

**Python Code Example:**
```python
# Pipeline: Word -> Apply Stemmer/Lemmatizer -> Normalized form

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "wolves", "caring"]

# Stemming
stemmed = [stemmer.stem(w) for w in words]
# Output: ['run', 'studi', 'better', 'wolv', 'care']

# Lemmatization (default: noun)
lemmatized = [lemmatizer.lemmatize(w) for w in words]
# Output: ['running', 'study', 'better', 'wolf', 'caring']

# Lemmatization with POS tag (verb)
lemmatized_verb = [lemmatizer.lemmatize(w, pos='v') for w in words]
# Output: ['run', 'study', 'better', 'wolf', 'care']
```

**Interview Tips:**
- Porter Stemmer is most common; Snowball is improved version
- Lemmatization needs POS tag for accuracy ("better" → "good" only if adjective)
- Modern deep learning models often skip both - subword tokenizers handle morphology

---

## Question 3
- [ ] Done
**What are BPE, WordPiece, and SentencePiece? How do subword tokenization algorithms work?**

**Answer:**

**Definition:**
Subword tokenization algorithms split text into units between characters and words, learning frequent subword patterns from data. BPE (Byte Pair Encoding), WordPiece, and SentencePiece are the three dominant algorithms that balance vocabulary size with representation power.

**Core Concepts:**

| Algorithm | Used By | Approach |
|-----------|---------|----------|
| BPE | GPT, RoBERTa | Merge most frequent adjacent pairs iteratively |
| WordPiece | BERT, DistilBERT | Merge pairs that maximize likelihood |
| SentencePiece | T5, XLNet, mBERT | Language-agnostic, treats text as raw stream |

**How BPE Works (Algorithm):**
1. Start with character-level vocabulary + end-of-word symbol
2. Count frequency of all adjacent symbol pairs
3. Merge the most frequent pair into new symbol
4. Repeat until desired vocabulary size reached

**Example:**
```
Corpus: "low lower lowest"
Initial: ['l', 'o', 'w', '</w>', 'l', 'o', 'w', 'e', 'r', '</w>', ...]
Step 1: Merge 'l'+'o' → 'lo' (most frequent)
Step 2: Merge 'lo'+'w' → 'low'
Step 3: Merge 'low'+'e' → 'lowe'
Final vocab: ['low</w>', 'low', 'er</w>', 'est</w>', ...]
```

**Key Differences:**
- **BPE:** Frequency-based merging
- **WordPiece:** Uses likelihood (probability) instead of frequency
- **SentencePiece:** No pre-tokenization, works directly on raw text (good for CJK)

**Mathematical Formulation (WordPiece):**
$$\text{Score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$$
Merge pair with highest score (maximizes likelihood gain).

**Python Code Example:**
```python
# Pipeline: Train BPE -> Tokenize new text

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Step 1: Initialize BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Step 2: Train on corpus
trainer = BpeTrainer(vocab_size=1000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Step 3: Tokenize
output = tokenizer.encode("unhappiness")
print(output.tokens)
# Output: ['un', 'happi', 'ness'] (subwords)
```

**Interview Tips:**
- BPE is greedy (frequency), WordPiece is probabilistic
- SentencePiece is a library that can implement BPE or Unigram
- Subword tokenizers solve OOV by breaking unknown words into known subwords

---

## Question 4
- [ ] Done
**What is Named Entity Recognition (NER) and what are the common entity types?**

**Answer:**

**Definition:**
Named Entity Recognition (NER) is a sequence labeling task that identifies and classifies named entities in text into predefined categories like person, organization, location, and date. It extracts structured information from unstructured text for downstream applications like knowledge graphs and information extraction.

**Core Concepts:**
- **Entity:** A real-world object that can be denoted with a proper name
- **Sequence Labeling:** Assigning a label to each token in a sequence
- **Entity Span:** Start and end positions of entity in text
- **Entity Linking:** Connecting extracted entities to knowledge base entries

**Common Entity Types (Standard):**

| Type | Tag | Examples |
|------|-----|----------|
| Person | PER | "Elon Musk", "Albert Einstein" |
| Organization | ORG | "Google", "United Nations" |
| Location | LOC | "Paris", "Mount Everest" |
| Geo-Political | GPE | "France", "New York City" |
| Date/Time | DATE/TIME | "January 2024", "3:00 PM" |
| Money | MONEY | "$500", "10 million euros" |
| Percentage | PERCENT | "25%", "fifty percent" |
| Miscellaneous | MISC | "Nobel Prize", "World Cup" |

**Domain-Specific Types:**
- **Medical:** Disease, Drug, Gene, Symptom
- **Legal:** Court, Law, Judge, Case
- **Finance:** Ticker, Company, Currency

**Python Code Example:**
```python
# Pipeline: Text -> NER Model -> Entity spans with labels

import spacy

# Step 1: Load pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Step 2: Process text
text = "Apple Inc. was founded by Steve Jobs in California."
doc = nlp(text)

# Step 3: Extract entities
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
    
# Output:
# Apple Inc. -> ORG
# Steve Jobs -> PERSON
# California -> GPE
```

**Interview Tips:**
- NER is foundation for knowledge extraction and QA systems
- Nested NER handles entities within entities ("Bank of America" → ORG contains LOC)
- Fine-grained NER may have 100+ entity types

---

## Question 5
- [ ] Done
**Explain the BIO/IOB tagging scheme used in NER and sequence labeling tasks.**

**Answer:**

**Definition:**
BIO (Beginning-Inside-Outside) tagging scheme is a labeling format where each token gets a tag indicating whether it begins an entity (B), continues inside an entity (I), or is outside any entity (O). It enables unambiguous representation of entity boundaries in sequence labeling.

**Core Concepts:**
- **B-TAG:** Beginning of an entity of type TAG
- **I-TAG:** Inside/continuation of entity of type TAG
- **O:** Outside any entity (not part of named entity)

**Why BIO is Needed:**
Without BIO, consecutive entities of same type would merge incorrectly.

**Example:**
```
Text:    "John  Smith  works  at  New   York   Times"
BIO:      B-PER I-PER  O      O   B-ORG I-ORG  I-ORG

Text:    "John  met   Mary  in  Paris"
BIO:      B-PER O     B-PER O   B-LOC
```

**Tagging Variants:**

| Scheme | Tags | Use Case |
|--------|------|----------|
| BIO | B, I, O | Standard, most common |
| BIOES | B, I, O, E, S | E=End, S=Single-token entity |
| BILOU | B, I, L, O, U | L=Last, U=Unit (single) |

**BIOES Example:**
```
Text:    "New   York   Times"
BIOES:    B-ORG I-ORG  E-ORG

Text:    "IBM"
BIOES:    S-ORG
```

**Why BIOES > BIO:**
- More explicit boundary information
- Helps model learn entity boundaries better
- Slightly improves NER accuracy

**Python Code Example:**
```python
# Pipeline: Tokens + Labels -> BIO format -> Model training

# Sample data in BIO format
sentence = ["John", "Smith", "works", "at", "Apple"]
bio_tags = ["B-PER", "I-PER", "O", "O", "B-ORG"]

# Convert to training format
training_data = list(zip(sentence, bio_tags))
# [('John', 'B-PER'), ('Smith', 'I-PER'), ('works', 'O'), 
#  ('at', 'O'), ('Apple', 'B-ORG')]

# Extract entities from BIO tags
def extract_entities(tokens, tags):
    entities = []
    current_entity = []
    current_type = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith('I-') and current_type:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = []
            current_type = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_type))
    
    return entities

print(extract_entities(sentence, bio_tags))
# Output: [('John Smith', 'PER'), ('Apple', 'ORG')]
```

**Interview Tips:**
- Always use B- prefix even for single-token entities (consistency)
- BIOES/BILOU often gives 1-2% F1 improvement over BIO
- During prediction, enforce valid transitions (I-PER cannot follow B-ORG)

---

## Question 6
- [ ] Done
**What is Part-of-Speech (POS) tagging? Why is it important for downstream NLP tasks?**

**Answer:**

**Definition:**
Part-of-Speech (POS) tagging assigns grammatical categories (noun, verb, adjective, etc.) to each word in a sentence based on its definition and context. It provides syntactic information essential for parsing, NER, sentiment analysis, and understanding sentence structure.

**Core Concepts:**
- **POS Tag:** Grammatical category of a word
- **Context-dependent:** Same word can have different POS ("run" as verb vs noun)
- **Tag Set:** Predefined set of tags (Penn Treebank has 45 tags)

**Common POS Tags (Penn Treebank):**

| Tag | Description | Example |
|-----|-------------|---------|
| NN | Noun, singular | "dog", "city" |
| NNS | Noun, plural | "dogs", "cities" |
| VB | Verb, base form | "run", "eat" |
| VBD | Verb, past tense | "ran", "ate" |
| VBG | Verb, gerund | "running", "eating" |
| JJ | Adjective | "happy", "fast" |
| RB | Adverb | "quickly", "very" |
| DT | Determiner | "the", "a" |
| IN | Preposition | "in", "on", "at" |
| PRP | Personal pronoun | "he", "she", "it" |

**Why Important for Downstream Tasks:**

| Task | How POS Helps |
|------|---------------|
| NER | Entities are typically nouns (NN, NNP) |
| Sentiment | Adjectives carry sentiment weight |
| Parsing | Determines phrase structure |
| Lemmatization | Need POS for correct lemma |
| Information Extraction | Filter by grammatical role |

**Python Code Example:**
```python
# Pipeline: Text -> Tokenize -> POS Tagger -> Tagged tokens

import nltk
nltk.download('averaged_perceptron_tagger')

# Method 1: NLTK
text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), 
#  ('fox', 'NN'), ('jumps', 'VBZ'), ...]

# Method 2: spaCy (more accurate)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
# The: DET (DT)
# quick: ADJ (JJ)
# fox: NOUN (NN)
# jumps: VERB (VBZ)
```

**Interview Tips:**
- Universal POS tags (17 tags) vs Penn Treebank (45 tags) - know the difference
- Accuracy: ~97% for English with neural models
- Ambiguity example: "book" can be NN (noun) or VB (verb)

---

## Question 7
- [ ] Done
**How do CRF (Conditional Random Fields) work for sequence labeling tasks like NER and POS tagging?**

**Answer:**

**Definition:**
Conditional Random Fields (CRF) are discriminative probabilistic models that predict a sequence of labels given an input sequence, modeling the conditional probability P(Y|X) while considering dependencies between adjacent labels. They excel at sequence labeling by capturing transition patterns between tags.

**Core Concepts:**
- **Discriminative model:** Models P(Y|X) directly, not P(X,Y)
- **Global normalization:** Normalizes over entire sequence, not per-token
- **Feature functions:** Capture relationships between input and labels
- **Transition matrix:** Learns valid label transitions (B-PER → I-PER is valid)

**Mathematical Formulation:**

$$P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^{T} \sum_{k} \lambda_k f_k(y_{t-1}, y_t, X, t)\right)$$

Where:
- $Z(X)$ = partition function (normalization)
- $f_k$ = feature functions
- $\lambda_k$ = learned weights
- $y_t$ = label at position $t$

**Why CRF > Softmax for Sequence Labeling:**

| Approach | Issue |
|----------|-------|
| Per-token Softmax | Ignores label dependencies; can predict "I-PER" after "B-ORG" |
| CRF Layer | Learns transition scores; enforces valid sequences |

**CRF Components:**
1. **Emission scores:** Score for each label at each position (from LSTM/BERT)
2. **Transition scores:** Learned matrix A where A[i,j] = score of transitioning from label i to j
3. **Viterbi decoding:** Find best label sequence efficiently

**Algorithm Steps (Training):**
1. Compute emission scores from neural network
2. Apply transition matrix for all label pairs
3. Use forward algorithm to compute Z(X) (partition function)
4. Compute negative log-likelihood loss
5. Backpropagate and update weights

**Algorithm Steps (Inference - Viterbi):**
1. Initialize scores for first position
2. For each position: score = emission + max(previous_score + transition)
3. Backtrack to find best path

**Python Code Example:**
```python
# Pipeline: Tokens -> BERT embeddings -> CRF layer -> Predicted tags

import torch
from torchcrf import CRF

# Step 1: Define model with CRF
class NERModel(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = torch.nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
    
    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:  # Training
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:  # Inference
            return self.crf.decode(emissions, mask=mask)

# Step 2: Training
model = NERModel(vocab_size=1000, tagset_size=9, embed_dim=100, hidden_dim=128)
# Loss = negative log-likelihood from CRF

# Step 3: Inference (Viterbi decoding)
# predictions = model(input_ids)  # Returns best tag sequence
```

**Interview Tips:**
- CRF adds ~1-2% F1 improvement for NER
- Modern BERT+CRF is standard for production NER
- Transition matrix size = num_tags × num_tags
- Viterbi is O(T × K²) where T=sequence length, K=num tags

---

## Question 8
- [ ] Done
**What is the difference between rule-based, statistical, and neural approaches to NER?**

**Answer:**

**Definition:**
NER approaches have evolved from handcrafted rules to statistical models to neural networks. Rule-based uses pattern matching, statistical uses features with ML classifiers like CRF, and neural uses deep learning to automatically learn representations and patterns.

**Core Comparison:**

| Aspect | Rule-Based | Statistical | Neural |
|--------|-----------|-------------|--------|
| Approach | Regex, gazetteers, patterns | Feature engineering + CRF/HMM | End-to-end deep learning |
| Features | Handcrafted | Manual feature extraction | Automatic representation |
| Training Data | None needed | Moderate | Large amounts |
| Adaptability | Low (needs rule rewrite) | Moderate | High (fine-tuning) |
| Accuracy | Domain-specific high | Good | State-of-the-art |
| Interpretability | High | Moderate | Low |

**Rule-Based Approach:**
```
Pattern: [A-Z][a-z]+ (Inc|Corp|Ltd)  → ORG
Gazetteer: {"New York", "London", "Paris"} → LOC
Context: "Dr. [Name]" → PERSON
```
- Pros: No training data, interpretable, precise for known patterns
- Cons: Doesn't generalize, maintenance nightmare

**Statistical Approach (CRF/HMM):**
- Features: Capitalization, word shape, prefix/suffix, POS tag, gazetteer lookup
- Model: CRF learns feature weights and transition probabilities
- Pros: Generalizes better, principled probability framework
- Cons: Requires feature engineering expertise

**Neural Approach (LSTM/Transformer):**
- Architecture: BERT → Linear → CRF
- Learns: Character patterns, contextual embeddings, entity boundaries
- Pros: State-of-the-art, handles unseen entities, transfer learning
- Cons: Needs data, compute-intensive, less interpretable

**Python Code Example:**
```python
# Comparing three approaches

# 1. Rule-based (simple regex)
import re
def rule_based_ner(text):
    entities = []
    # Pattern for organizations
    orgs = re.findall(r'[A-Z][a-z]+ (Inc|Corp|Ltd)\.?', text)
    # Pattern for dates
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)
    return {"ORG": orgs, "DATE": dates}

# 2. Statistical (sklearn-crfsuite)
# Feature extraction + CRF training (conceptual)
def extract_features(word, pos):
    return {
        'word': word.lower(),
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'pos': pos,
        'prefix_2': word[:2],
        'suffix_2': word[-2:]
    }

# 3. Neural (transformers)
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
result = ner("Apple Inc was founded by Steve Jobs")
# [{'entity': 'B-ORG', 'word': 'Apple'}, ...]
```

**When to Use Each:**
- **Rule-based:** Quick prototype, specific domain, no training data
- **Statistical:** Moderate data, need interpretability, limited compute
- **Neural:** Best accuracy needed, sufficient data and compute

**Interview Tips:**
- Hybrid approaches combine all three (rules for post-processing neural output)
- Statistical CRF is still used as final layer on top of BERT
- Gazetteers remain useful even in neural systems

---

## Question 9
- [ ] Done
**How do modern transformers like BERT handle tokenization differently from traditional methods?**

**Answer:**

**Definition:**
BERT uses WordPiece subword tokenization which splits words into frequent subword units, adds special tokens ([CLS], [SEP], [PAD]), and maintains a fixed vocabulary. Unlike traditional word-level tokenization, it handles OOV words by breaking them into known subwords, enabling open vocabulary modeling.

**Core Differences:**

| Aspect | Traditional (Word-level) | BERT (WordPiece) |
|--------|-------------------------|------------------|
| OOV Handling | [UNK] token | Splits into subwords |
| Vocabulary | Very large or limited | Fixed size (~30K) |
| Morphology | Lost | Partially captured |
| Special Tokens | None | [CLS], [SEP], [PAD], [MASK] |
| Subword prefix | N/A | ## for continuation |

**BERT Tokenization Process:**
1. Add special tokens: [CLS] at start, [SEP] at end
2. Apply WordPiece algorithm to split text
3. Convert tokens to IDs using vocabulary
4. Generate attention mask (1 for real tokens, 0 for padding)
5. Generate token type IDs (for sentence A vs B)

**Example:**
```
Input: "unhappiness is contagious"

Word-level:  ["unhappiness", "is", "contagious"]  # 3 tokens
WordPiece:   ["un", "##hap", "##pi", "##ness", "is", "con", "##ta", "##gious"]

Full BERT:   ["[CLS]", "un", "##hap", "##pi", "##ness", "is", 
              "con", "##ta", "##gious", "[SEP]"]
```

**Special Tokens Purpose:**

| Token | Purpose |
|-------|---------|
| [CLS] | Classification token; pooled for sentence-level tasks |
| [SEP] | Separator between sentences |
| [PAD] | Padding for batch processing |
| [MASK] | Masked token for pre-training |
| [UNK] | Unknown (rarely used due to subwords) |

**Python Code Example:**
```python
# Pipeline: Text -> BERT Tokenizer -> Model-ready tensors

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "unhappiness is contagious"

# Step 1: Basic tokenization
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', '##happ', '##iness', 'is', 'con', '##tag', '##ious']

# Step 2: Full encoding with special tokens
encoded = tokenizer(text, return_tensors='pt', padding=True)
print(encoded)
# {'input_ids': tensor([[101, 4895, 18223, 11231, 2003, 9530, 16259, 6313, 102]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])}

# Step 3: Decode back
decoded = tokenizer.decode(encoded['input_ids'][0])
# '[CLS] unhappiness is contagious [SEP]'

# For NER: Track subword alignment
tokens_with_offsets = tokenizer(text, return_offsets_mapping=True)
# Offset mapping helps align subword predictions to original words
```

**Key Implications for NER:**
- Subword tokens must be realigned to original words
- First subword gets the label; rest get special label or are ignored
- Handle ## prefix tokens during post-processing

**Interview Tips:**
- BERT vocab size = 30,522 (base) with WordPiece
- GPT uses BPE (no ## prefix, uses Ġ for space)
- [CLS] embedding is used for classification, not for token-level tasks

---

## Question 10
- [ ] Done
**What is the out-of-vocabulary (OOV) problem and how do subword tokenizers solve it?**

**Answer:**

**Definition:**
The Out-of-Vocabulary (OOV) problem occurs when a model encounters words not present in its training vocabulary, typically mapped to a useless [UNK] token. Subword tokenizers solve this by breaking unknown words into smaller known subword units, ensuring any word can be represented.

**Core Concepts:**
- **Closed vocabulary:** Fixed word set; unseen words → [UNK]
- **Open vocabulary:** Can represent any word via subword composition
- **[UNK] token:** Placeholder for unknown words; loses all semantic information
- **Subword composition:** Build unknown words from known pieces

**The OOV Problem:**
```
Vocabulary: ["happy", "sad", "the", ...]
New word: "unhappiness"
Word-level: "unhappiness" → [UNK]  ❌ Information lost!

Subword vocab: ["un", "happy", "ness", ...]  
Subword: "unhappiness" → ["un", "happy", "ness"]  ✓ Meaning preserved!
```

**Why OOV is Problematic:**
- Rare words, typos, new terms all become [UNK]
- [UNK] has no useful semantics
- Critical for names, technical terms, multilingual text
- High OOV rate = poor model performance

**How Subword Tokenizers Solve OOV:**

| Scenario | Word-Level | Subword (BPE) |
|----------|-----------|---------------|
| Rare word: "cryptocurrency" | [UNK] | ["crypt", "o", "currency"] |
| Typo: "hapyy" | [UNK] | ["hap", "y", "y"] |
| New name: "ChatGPT" | [UNK] | ["Chat", "G", "PT"] |
| Morphology: "running" | [UNK] or separate | ["run", "ning"] |

**Trade-off:**
- Smaller vocab + longer sequences vs Larger vocab + shorter sequences
- Subwords balance this: ~30K vocab handles most text efficiently

**Python Code Example:**
```python
# Pipeline: Demonstrate OOV handling

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example words (some rare/unusual)
words = ["happy", "unhappiness", "cryptocurrency", "xyzabc123"]

for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"{word} -> {tokens}")

# Output:
# happy -> ['happy']
# unhappiness -> ['un', '##happ', '##iness']
# cryptocurrency -> ['cry', '##pt', '##oc', '##ur', '##ren', '##cy']
# xyzabc123 -> ['x', '##yz', '##ab', '##c', '##12', '##3']

# Even gibberish gets tokenized (no [UNK])!

# Check UNK usage
unk_id = tokenizer.unk_token_id
encoded = tokenizer.encode("xyzabc123 unhappiness")
print(f"Contains UNK: {unk_id in encoded}")  # False
```

**Vocabulary Size Comparison:**

| Tokenizer | Vocabulary Size | OOV Rate |
|-----------|----------------|----------|
| Word-level (top 50K) | 50,000 | ~5-10% |
| BERT WordPiece | 30,522 | ~0% |
| GPT-2 BPE | 50,257 | ~0% |

**Interview Tips:**
- Subword tokenizers virtually eliminate OOV
- Character-level is extreme subword (always 0% OOV but very long sequences)
- OOV was major problem in word2vec/GloVe era
- FastText partially solved OOV with character n-grams

---

## Interview Questions

## Question 11
- [ ] Done
**How do you choose between word-level, subword, and character-level tokenization for different NLP tasks?**

**Answer:**

**Definition:**
The choice depends on vocabulary size constraints, language characteristics, and task requirements. Subword is the default for most modern NLP; character-level for morphologically rich languages or noisy text; word-level for simple tasks with clean, limited vocabulary.

**Decision Framework:**

| Factor | Word-Level | Subword | Character-Level |
|--------|-----------|---------|-----------------|
| Vocabulary size | Very large | Moderate (~30K) | Tiny (~100) |
| Sequence length | Short | Moderate | Very long |
| OOV handling | Poor | Excellent | Perfect |
| Training speed | Fast | Moderate | Slow |
| Morphology capture | None | Partial | Full |

**Choose Word-Level When:**
- Clean, well-defined vocabulary (e.g., sentiment on product reviews)
- Pre-trained word embeddings available (Word2Vec, GloVe)
- Simple classification with limited vocabulary
- Legacy systems or resource constraints

**Choose Subword When:**
- Using transformer models (BERT, GPT) - this is the default
- Multilingual or cross-lingual tasks
- Need to handle rare words, typos, new terms
- Most modern NLP applications

**Choose Character-Level When:**
- Extremely noisy text (OCR, social media with typos)
- Morphologically rich languages (Turkish, Finnish)
- Text with no clear word boundaries
- Tasks requiring character patterns (spelling correction)

**Task-Specific Recommendations:**

| Task | Recommended | Reason |
|------|-------------|--------|
| Text Classification | Subword | Balance of vocab and semantics |
| NER | Subword + alignment | Handles rare entity names |
| Machine Translation | Subword (BPE) | Cross-lingual, handles morphology |
| Spelling Correction | Character | Need character-level patterns |
| Social Media Analysis | Subword or Char | Handles hashtags, typos, slang |
| Code Generation | Byte-level BPE | Handles any character/token |

**Python Code Example:**
```python
# Comparison of tokenization approaches

text = "I loveeee #MachineLearning!!! 🚀"

# 1. Word-level (NLTK)
import nltk
word_tokens = nltk.word_tokenize(text)
# ['I', 'loveeee', '#', 'MachineLearning', '!', '!', '!', '🚀']

# 2. Subword (BERT)
from transformers import BertTokenizer
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
subword_tokens = bert_tok.tokenize(text)
# ['i', 'love', '##ee', '##e', '#', 'machine', '##lea', '##rning', '!', ...]

# 3. Character-level
char_tokens = list(text)
# ['I', ' ', 'l', 'o', 'v', 'e', 'e', 'e', 'e', ...]

print(f"Word: {len(word_tokens)} tokens")      # ~8
print(f"Subword: {len(subword_tokens)} tokens") # ~12
print(f"Char: {len(char_tokens)} tokens")       # ~35
```

**Interview Tips:**
- Default answer: "Subword tokenization with modern transformers"
- Know trade-off: vocab size ↔ sequence length
- Character-level is rarely used standalone; often combined with CNN/RNN

---

## Question 12
- [ ] Done
**What are the trade-offs between stemming and lemmatization for information retrieval?**

**Answer:**

**Definition:**
Stemming offers faster processing and higher recall by aggressively matching word variants, while lemmatization provides higher precision with linguistically correct forms. For information retrieval, stemming is often preferred for recall-oriented systems; lemmatization for precision-critical applications.

**Trade-off Comparison:**

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Speed | Faster (rule-based) | Slower (dictionary lookup) |
| Recall | Higher (aggressive matching) | Lower (exact forms) |
| Precision | Lower (over-stemming) | Higher (correct forms) |
| Storage | Smaller index | Larger index |
| Errors | Over/under-stemming | Few errors |

**Information Retrieval Impact:**

**Stemming Benefits:**
- Query "running" matches documents with "run", "runs", "runner"
- Reduces index size (fewer unique terms)
- Increases recall: finds more relevant documents

**Stemming Problems (Over-stemming):**
```
"universal" → "univers"
"university" → "univers"
"universe" → "univers"
# All conflated incorrectly!
```

**Lemmatization Benefits:**
- "better" → "good" (correct normalization)
- No false conflation
- Better for semantic search

**When to Choose:**

| Use Case | Choice | Reason |
|----------|--------|--------|
| Web search | Stemming | Recall matters, speed critical |
| Legal document search | Lemmatization | Precision critical |
| Question answering | Lemmatization | Semantic accuracy needed |
| Log analysis | Stemming | Fast, recall-oriented |
| Modern neural IR | Neither | Embeddings handle variations |

**Python Code Example:**
```python
# IR comparison: Stemming vs Lemmatization

from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Build simple inverted index
documents = [
    "The runner is running fast",
    "She runs every morning",
    "Run for better health"
]

def build_index(docs, normalizer):
    index = {}
    for doc_id, doc in enumerate(docs):
        for word in doc.lower().split():
            normalized = normalizer(word)
            if normalized not in index:
                index[normalized] = set()
            index[normalized].add(doc_id)
    return index

# Compare indexes
stem_index = build_index(documents, stemmer.stem)
lemma_index = build_index(documents, lambda w: lemmatizer.lemmatize(w, pos='v'))

# Search for "running"
query = "running"
stem_query = stemmer.stem(query)  # "run"
lemma_query = lemmatizer.lemmatize(query, pos='v')  # "run"

print(f"Stemming index entries: {len(stem_index)}")
print(f"Lemma index entries: {len(lemma_index)}")
# Stemming typically produces fewer unique entries
```

**Interview Tips:**
- Modern neural retrievers (DPR, ColBERT) often skip both - embeddings handle morphology
- Hybrid approach: lemmatize for indexing, stem for query expansion
- Language-specific: stemming works well for English, less so for morphologically rich languages

---

## Question 13
- [ ] Done
**How do you handle tokenization for languages without clear word boundaries (Chinese, Japanese, Thai)?**

**Answer:**

**Definition:**
Languages like Chinese, Japanese, and Thai don't use spaces between words, requiring specialized segmentation approaches. Solutions include dictionary-based methods, statistical models (HMM, CRF), or character/subword tokenization that bypasses explicit word segmentation.

**Core Challenge:**
```
English: "I love machine learning"  → Spaces define words
Chinese: "我喜欢机器学习"           → No spaces; "机器学习" = "machine learning"
Japanese: "私は機械学習が好きです"    → Mixed scripts, no spaces
Thai: "ฉันรักการเรียนรู้เครื่อง"      → No spaces
```

**Approaches:**

| Method | Description | Example Tools |
|--------|-------------|---------------|
| Dictionary-based | Maximum matching with word dictionary | Jieba (Chinese) |
| Statistical (CRF/HMM) | Learn segmentation from annotated data | Stanford Segmenter |
| Character-level | Each character is a token | Simple but loses word semantics |
| Subword (SentencePiece) | Learn segments from raw text | mBERT, XLM-R |

**Chinese Word Segmentation Example:**
```
Input: "我喜欢机器学习"
Segmented: "我 / 喜欢 / 机器 / 学习"
English: "I / like / machine / learning"
```

**Algorithm: Maximum Matching (Forward):**
1. Start from beginning of string
2. Find longest word in dictionary that matches
3. Segment that word, move forward
4. Repeat until end

**Ambiguity Problem:**
```
"研究生命起源" can be:
- "研究 / 生命 / 起源" (study origin of life)
- "研究生 / 命 / 起源" (graduate student's fate origin) ❌
```
Requires context and statistical methods to resolve.

**Python Code Example:**
```python
# Pipeline: Raw text -> Segmenter -> Tokenized text

# Method 1: Jieba for Chinese
import jieba

chinese_text = "我喜欢机器学习和深度学习"

# Default segmentation
tokens = jieba.lcut(chinese_text)
print(tokens)
# ['我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Search mode (finer granularity for search engines)
tokens_search = jieba.lcut_for_search(chinese_text)
# ['我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Method 2: SentencePiece (language-agnostic)
# Works directly on raw text, learns subwords without pre-segmentation
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
tokens = tokenizer.tokenize(chinese_text)
print(tokens)
# ['▁我', '喜欢', '机器', '学习', '和', '深度', '学习']

# Method 3: Character-level (simple baseline)
char_tokens = list(chinese_text)
# ['我', '喜', '欢', '机', '器', '学', '习', '和', '深', '度', '学', '习']
```

**Modern Solution:**
- Use SentencePiece-based tokenizers (mBERT, XLM-R)
- They learn subwords directly from raw text
- No need for language-specific segmenters
- Works for 100+ languages

**Interview Tips:**
- Jieba is standard for Chinese NLP
- MeCab is standard for Japanese
- SentencePiece is language-agnostic solution used by multilingual models
- Character-level works but loses semantic grouping

---

## Question 14
- [ ] Done
**What techniques work best for tokenization of social media text with informal language, hashtags, and emojis?**

**Answer:**

**Definition:**
Social media text requires specialized tokenization to handle hashtags, mentions, emojis, slang, elongation ("sooooo"), and non-standard spelling. Techniques include specialized pre-processing, social media-trained tokenizers, and emoji/hashtag-aware segmentation.

**Challenges in Social Media Text:**
```
Tweet: "OMG I loveeeee #MachineLearning 🔥🔥🔥 @elonmusk is amazingggg!!!"

Issues: 
- Elongation: "loveeeee", "amazingggg"
- Hashtag: #MachineLearning (compound word)
- Mentions: @elonmusk
- Emojis: 🔥🔥🔥
- Slang: OMG
- Punctuation: !!! (emphasis)
```

**Techniques:**

| Technique | Purpose | Example |
|-----------|---------|---------|
| Elongation normalization | "loveeeee" → "love" | Regex: reduce repeated chars |
| Hashtag segmentation | #MachineLearning → "Machine Learning" | WordNinja, CamelCase split |
| Emoji handling | Keep, remove, or convert to text | emoji library |
| Mention handling | @user → [USER] token | Anonymization |
| Slang expansion | "OMG" → "Oh My God" | Slang dictionary |

**Pre-processing Pipeline:**
1. Normalize URLs → [URL]
2. Normalize mentions → [USER]
3. Segment hashtags
4. Normalize elongation
5. Handle emojis (keep or convert to description)
6. Apply tokenizer

**Python Code Example:**
```python
# Pipeline: Raw tweet -> Preprocess -> Tokenize

import re
import emoji

def preprocess_tweet(text):
    # Step 1: Replace URLs
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    
    # Step 2: Replace mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Step 3: Normalize elongation (keep max 2 repeated chars)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # "loveeeee" → "lovee"
    
    # Step 4: Segment hashtags
    def segment_hashtag(match):
        tag = match.group(1)
        # CamelCase split
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        return words
    text = re.sub(r'#(\w+)', segment_hashtag, text)
    
    # Step 5: Convert emojis to text (optional)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 🔥 → " fire "
    
    return text

# Test
tweet = "OMG I loveeeee #MachineLearning 🔥🔥 @elonmusk https://t.co/xyz"
cleaned = preprocess_tweet(tweet)
print(cleaned)
# "OMG I lovee Machine Learning fire fire [USER] [URL]"

# Now apply standard tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
tokens = tokenizer.tokenize(cleaned)
print(tokens)
```

**Specialized Tokenizers:**
- **TweetTokenizer (NLTK):** Basic social media handling
- **BERTweet:** BERT pre-trained on 850M tweets
- **RoBERTa-Twitter:** Twitter-specific language model

**Interview Tips:**
- Use pre-trained social media models (BERTweet) when possible
- Hashtag segmentation is critical for semantic understanding
- Emojis carry sentiment - don't blindly remove them
- Normalize but don't over-clean (lose information)

---

## Question 15
- [ ] Done
**How do you handle NER for nested or overlapping entities?**

**Answer:**

**Definition:**
Nested NER occurs when entities contain other entities (e.g., "Bank of America" contains location "America"). Traditional BIO tagging can only assign one label per token. Solutions include span-based models, multi-layer tagging, or hypergraph representations.

**The Problem:**
```
Text: "Bank of America headquarters in New York City"

Flat NER:   [Bank of America]_ORG ... [New York City]_LOC

Nested NER: [Bank of [America]_LOC]_ORG ... [[New York]_LOC City]_LOC
            └── ORG ──────────────┘     └── LOC ─────────────────┘
                  └── LOC ─┘                    └── LOC ──┘
```

**Why Standard BIO Fails:**
- Each token gets exactly one tag
- Cannot represent "America" as both part of ORG and as LOC simultaneously

**Solutions:**

| Approach | Description | Complexity |
|----------|-------------|------------|
| Span-based | Enumerate all spans, classify each | O(n²) spans |
| Layered/Stacked | Multiple BIO layers for nesting levels | Multiple passes |
| Hypergraph | Represent all possible nestings | Complex inference |
| Biaffine | Score all (start, end, type) tuples | O(n²) |

**Span-Based Approach (Most Common):**
1. Generate all possible spans up to max length
2. Encode each span using [start_token; end_token; span_representation]
3. Classify each span into entity type or "none"
4. Handle overlaps in post-processing

**Algorithm (Span Enumeration):**
```
Text: "Bank of America" (3 tokens)
Spans: 
  (0,0): "Bank"
  (0,1): "Bank of"  
  (0,2): "Bank of America"  ← ORG
  (1,1): "of"
  (1,2): "of America"
  (2,2): "America"          ← LOC (nested!)
```

**Python Code Example:**
```python
# Pipeline: Text -> Generate spans -> Classify spans -> Extract nested entities

import torch
import torch.nn as nn

class SpanBasedNER(nn.Module):
    def __init__(self, encoder, hidden_dim, num_labels, max_span_len=10):
        super().__init__()
        self.encoder = encoder  # BERT or similar
        self.max_span_len = max_span_len
        self.span_classifier = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Step 1: Encode tokens
        outputs = self.encoder(input_ids, attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Step 2: Generate span representations
        batch_size, seq_len, hidden_dim = hidden.shape
        spans = []
        
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_len, seq_len)):
                # Span representation: concat of start and end tokens
                span_repr = torch.cat([hidden[:, start], hidden[:, end]], dim=-1)
                spans.append((start, end, span_repr))
        
        # Step 3: Classify each span
        span_reprs = torch.stack([s[2] for s in spans], dim=1)
        logits = self.span_classifier(span_reprs)
        
        return logits, spans

# Inference: Extract nested entities
def extract_nested_entities(spans, predictions, tokens, id2label):
    entities = []
    for (start, end, _), pred in zip(spans, predictions):
        label = id2label[pred]
        if label != "O":  # Not "none"
            entity_text = " ".join(tokens[start:end+1])
            entities.append({
                "text": entity_text,
                "start": start,
                "end": end,
                "label": label
            })
    return entities  # Can have overlapping entities!
```

**Post-Processing for Nested Entities:**
- Keep all predicted entities (unlike flat NER where you'd choose one)
- May need to filter low-confidence overlapping predictions
- Build entity hierarchy if needed

**Interview Tips:**
- Span-based is current state-of-the-art for nested NER
- ACE, GENIA datasets have nested annotations for benchmarking
- Computational cost: O(n²) spans vs O(n) for flat NER
- In production, limit max span length for efficiency

---

## Question 16
- [ ] Done
**What strategies help with NER in low-resource languages with limited training data?**

**Answer:**

**Definition:**
Low-resource NER faces scarcity of labeled data. Strategies include cross-lingual transfer from high-resource languages, data augmentation, multilingual pre-training, active learning, and distant supervision using knowledge bases.

**Core Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Cross-lingual transfer | Train on English, apply to target language | Multilingual model available |
| Data augmentation | Generate synthetic training data | Some labeled data exists |
| Distant supervision | Auto-label using knowledge bases | KB coverage for target entities |
| Active learning | Strategically select samples to annotate | Limited annotation budget |
| Zero-shot transfer | Use multilingual embeddings directly | No target language data |

**1. Cross-Lingual Transfer:**
```
Train: English NER data (abundant)
Model: mBERT or XLM-R (multilingual)
Apply: Target low-resource language

Why it works: Shared multilingual representation space
```

**2. Data Augmentation Techniques:**
- **Mention replacement:** Replace entities with other entities of same type
- **Back-translation:** Translate to another language and back
- **Synonym replacement:** Replace words with synonyms
- **Entity swapping:** Swap entity positions in sentence

**3. Distant Supervision:**
```
Knowledge Base: Wikidata, DBpedia
Process:
1. Get entity list from KB (e.g., all person names)
2. Match entities in unlabeled text
3. Auto-generate noisy labels
4. Train with noise-aware methods
```

**Python Code Example:**
```python
# Strategy 1: Cross-lingual transfer with XLM-R

from transformers import AutoModelForTokenClassification, AutoTokenizer

# Train on English, apply to Hindi/Swahili/etc.
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fine-tune on English NER data
# ... training code ...

# Apply directly to target language (zero-shot)
hindi_text = "नरेंद्र मोदी भारत के प्रधानमंत्री हैं"
# Model transfers knowledge across languages

# Strategy 2: Simple data augmentation
import random

def augment_ner_data(sentence, entities, entity_dict):
    """Replace entities with other entities of same type"""
    augmented = sentence
    for entity in entities:
        if entity['type'] in entity_dict:
            replacement = random.choice(entity_dict[entity['type']])
            augmented = augmented.replace(entity['text'], replacement)
    return augmented

entity_dict = {
    'PER': ['John Smith', 'Maria Garcia', 'Chen Wei'],
    'ORG': ['Google', 'Microsoft', 'Amazon'],
    'LOC': ['Paris', 'Tokyo', 'Mumbai']
}

original = "Barack Obama visited New York"
entities = [{'text': 'Barack Obama', 'type': 'PER'}, 
            {'text': 'New York', 'type': 'LOC'}]

augmented = augment_ner_data(original, entities, entity_dict)
# "John Smith visited Mumbai"
```

**Interview Tips:**
- mBERT/XLM-R are go-to for cross-lingual transfer
- Distant supervision is noisy - use noise-robust training
- 100-500 annotated sentences can bootstrap reasonable NER with transfer learning
- Active learning can reduce annotation needs by 50-80%

---

## Question 17
- [ ] Done
**How do you implement domain adaptation for NER models (e.g., from news to biomedical text)?**

**Answer:**

**Definition:**
Domain adaptation bridges the gap between source domain (e.g., news with abundant data) and target domain (e.g., biomedical with limited data). Techniques include continued pre-training on target domain text, gradual fine-tuning, and domain-specific feature integration.

**Core Challenge:**
```
News NER:      "Apple Inc. announced a new product"
               [Apple Inc.]_ORG

Biomedical:    "Apple extract inhibits tumor growth"
               Same word "Apple" but different context (not ORG!)
               [tumor]_DISEASE, [growth]_PROCESS
```

**Domain Shift Types:**
- **Vocabulary shift:** New terms (gene names, drug names)
- **Entity type shift:** Different entity categories
- **Distribution shift:** Different writing styles, sentence structures

**Domain Adaptation Strategies:**

| Strategy | Description | Data Required |
|----------|-------------|---------------|
| Continued pre-training | Pre-train LM on target domain text | Unlabeled target text |
| Fine-tuning | Fine-tune on small target labeled data | Labeled target data |
| Multi-task learning | Joint training on source + target | Both datasets |
| Feature augmentation | Add domain-specific features | Domain gazetteers |

**Step-by-Step Approach:**
1. **Continued Pre-training:** Pre-train BERT on unlabeled biomedical text (PubMed)
2. **Domain-specific vocabulary:** Add domain terms to tokenizer
3. **Fine-tune on target NER:** Use limited biomedical NER data
4. **Add gazetteers:** Include drug/disease dictionaries as features

**Python Code Example:**
```python
# Pipeline: Base model -> Domain pre-train -> Fine-tune NER

from transformers import (
    AutoModelForMaskedLM, 
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)

# Step 1: Continued pre-training on domain text
def domain_pretrain(base_model, domain_corpus):
    """
    Continue MLM pre-training on biomedical text
    """
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Train on domain corpus with MLM objective
    # ... training loop ...
    
    model.save_pretrained("domain-adapted-bert")
    return model

# Step 2: Fine-tune for NER on target domain
def finetune_ner(domain_model_path, ner_dataset, num_labels):
    model = AutoModelForTokenClassification.from_pretrained(
        domain_model_path,
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir="./ner-model",
        num_train_epochs=5,
        learning_rate=2e-5,
        # Lower learning rate for domain adaptation
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ner_dataset
    )
    trainer.train()
    return model

# Step 3: Use domain-specific pre-trained models
# BioBERT, PubMedBERT, ClinicalBERT, SciBERT
from transformers import AutoModel

biomedical_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
# Already pre-trained on PubMed + PMC
```

**Pre-trained Domain Models:**
| Domain | Model | Pre-training Data |
|--------|-------|-------------------|
| Biomedical | BioBERT, PubMedBERT | PubMed abstracts |
| Clinical | ClinicalBERT | MIMIC clinical notes |
| Scientific | SciBERT | Semantic Scholar papers |
| Legal | LegalBERT | Legal documents |
| Finance | FinBERT | Financial news |

**Interview Tips:**
- Use existing domain-specific BERT when available
- Continued pre-training helps even with 10-50MB of domain text
- Gradual unfreezing: fine-tune top layers first, then all layers
- Domain gazetteers significantly boost NER on specialized entities

---

## Question 18
- [ ] Done
**What approaches work best for NER in noisy text like social media or OCR output?**

**Answer:**

**Definition:**
Noisy text NER requires robustness to spelling errors, missing punctuation, informal language, and OCR artifacts. Approaches include noise-aware training, character-level representations, spelling correction pre-processing, and models trained on noisy data.

**Types of Noise:**

| Source | Noise Type | Example |
|--------|-----------|---------|
| Social media | Informal spelling | "u", "2morrow", "gonna" |
| OCR | Character confusion | "rn" → "m", "0" → "O" |
| ASR | Phonetic errors | "there" vs "their" |
| Typos | Missing/swapped chars | "Gogle" instead of "Google" |

**Strategies:**

| Approach | Description | Effectiveness |
|----------|-------------|---------------|
| Noise injection training | Add synthetic noise during training | High |
| Character embeddings | Capture sub-word patterns | Medium-High |
| Spelling correction | Pre-process to correct errors | Medium |
| Robust tokenization | Handle noisy tokens gracefully | Medium |
| Pre-trained on noisy data | BERTweet, etc. | High |

**1. Noise Injection Training:**
```python
def inject_noise(text, noise_prob=0.1):
    """Add synthetic noise for robust training"""
    noisy = []
    for char in text:
        if random.random() < noise_prob:
            noise_type = random.choice(['delete', 'swap', 'insert', 'replace'])
            if noise_type == 'delete':
                continue
            elif noise_type == 'swap' and len(noisy) > 0:
                noisy[-1], char = char, noisy[-1]
            elif noise_type == 'insert':
                noisy.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif noise_type == 'replace':
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
        noisy.append(char)
    return ''.join(noisy)
```

**2. Character-Level Representations:**
- CNN or LSTM over characters captures morphology
- Robust to OOV and spelling variations
- Combine with word embeddings

**Python Code Example:**
```python
# Pipeline: Noisy text -> Normalize -> Character+Word features -> NER

import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Step 1: Noise normalization
def normalize_noisy_text(text):
    # Normalize repeated characters
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # "sooooo" -> "soo"
    
    # Common OCR fixes
    ocr_fixes = {'0': 'O', '1': 'l', '|': 'l', 'rn': 'm'}
    for error, fix in ocr_fixes.items():
        text = text.replace(error, fix)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

# Step 2: Use noise-robust model
# BERTweet is trained on noisy Twitter data
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForTokenClassification.from_pretrained(
    "vinai/bertweet-base",
    num_labels=9
)

# Step 3: Character-level CNN (conceptual)
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, num_filters):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        self.conv = nn.Conv1d(char_embed_dim, num_filters, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, char_ids):
        # char_ids: (batch, word_len)
        embedded = self.char_embed(char_ids)  # (batch, word_len, embed)
        embedded = embedded.permute(0, 2, 1)   # (batch, embed, word_len)
        conv_out = self.conv(embedded)
        pooled = self.pool(conv_out).squeeze(-1)
        return pooled  # Character-level word representation
```

**Practical Pipeline:**
1. Light normalization (don't over-correct)
2. Use character-level features
3. Train with noise injection
4. Use pre-trained models on similar noisy data

**Interview Tips:**
- Don't over-normalize - may lose information
- Character CNNs help with spelling variations
- BERTweet/RoBERTa-Twitter for social media
- OCR post-correction is a separate research area

---

## Question 19
- [ ] Done
**How do you handle entity disambiguation and linking NER results to knowledge bases?**

**Answer:**

**Definition:**
Entity Linking (EL) connects extracted entity mentions to unique entries in a knowledge base (Wikipedia, Wikidata). It resolves ambiguity when the same name refers to different entities (e.g., "Apple" → fruit or company). The process involves candidate generation, candidate ranking, and NIL prediction.

**The Problem:**
```
"Michael Jordan was a great basketball player."
"Michael Jordan is a famous ML professor."

Both "Michael Jordan" mentions → different KB entries!
- Basketball player: Q41421 (Wikidata)
- ML Professor: Q3308285 (Wikidata)
```

**Entity Linking Pipeline:**

| Step | Description | Output |
|------|-------------|--------|
| 1. Mention Detection | Find entity spans (NER) | "Michael Jordan" |
| 2. Candidate Generation | Get potential KB matches | [Q41421, Q3308285, ...] |
| 3. Candidate Ranking | Score candidates using context | Q41421: 0.95, Q3308285: 0.03 |
| 4. NIL Prediction | Detect if entity not in KB | "new_entity" or linked ID |

**Candidate Generation Methods:**
- **Alias table:** Pre-built mapping of surface forms to candidates
- **Fuzzy matching:** Handle typos, variations
- **Prior probability:** P(entity | mention) from Wikipedia anchor text

**Candidate Ranking Features:**
- Textual similarity (mention vs entity description)
- Context compatibility (surrounding text vs entity context)
- Coherence (other entities in same document)
- Prior probability (popularity)

**Python Code Example:**
```python
# Pipeline: Mention -> Generate candidates -> Rank -> Link

# Method 1: Using spaCy EntityLinker
import spacy

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entity_linker", after="ner")

text = "Apple was founded by Steve Jobs"
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text} -> {ent.kb_id_}")
# Apple -> Q312 (Apple Inc.)
# Steve Jobs -> Q19837 (Steve Jobs)

# Method 2: Custom Entity Linker
from sentence_transformers import SentenceTransformer, util

class SimpleEntityLinker:
    def __init__(self, kb_descriptions):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb = kb_descriptions  # {entity_id: description}
        self.kb_embeddings = self.model.encode(list(kb_descriptions.values()))
        self.kb_ids = list(kb_descriptions.keys())
    
    def link(self, mention, context, top_k=5):
        # Step 1: Generate candidates (simplified: use all KB)
        # In practice: use alias table for efficient candidate generation
        
        # Step 2: Create query from mention + context
        query = f"{mention}: {context}"
        query_emb = self.model.encode(query)
        
        # Step 3: Rank candidates by similarity
        scores = util.cos_sim(query_emb, self.kb_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = [(self.kb_ids[i], float(scores[i])) for i in top_indices]
        return results

# Usage
kb = {
    "Q41421": "Michael Jordan, American basketball player for Chicago Bulls",
    "Q3308285": "Michael I. Jordan, American computer scientist, machine learning",
    "Q312": "Apple Inc., American technology company founded by Steve Jobs"
}

linker = SimpleEntityLinker(kb)
results = linker.link("Michael Jordan", "great basketball player who won 6 championships")
# [('Q41421', 0.85), ('Q3308285', 0.32), ...]
```

**NIL Prediction:**
When entity is not in knowledge base:
- Set threshold on ranking score
- Train binary classifier for in-KB vs out-of-KB
- Cluster NIL mentions for new entity discovery

**Interview Tips:**
- Entity linking is harder than NER - same mention, different entities
- Wikipedia is most common KB; Wikidata for structured data
- Prior probability (popularity) is a strong baseline feature
- Modern approach: bi-encoder for candidate generation, cross-encoder for ranking

---

## Question 20
- [ ] Done
**What are the challenges of POS tagging for morphologically rich languages?**

**Answer:**

**Definition:**
Morphologically rich languages (Turkish, Finnish, Hungarian, Arabic) have complex word structures with many inflections, cases, and affixes. This creates massive vocabulary, high OOV rates, and ambiguity that simple POS taggers struggle with.

**Core Challenges:**

| Challenge | Description | Example (Turkish) |
|-----------|-------------|-------------------|
| Vocabulary explosion | One lemma → many forms | "ev" (house) → evler, evde, evden, evlerin, ... |
| OOV rate | Unseen word forms | Up to 50% OOV in test data |
| Ambiguity | Same suffix, different meanings | "-ler" = plural OR 3rd person |
| Agglutination | Multiple morphemes per word | "evlerinden" = house+PLURAL+from+their |
| Long dependencies | Affixes interact with each other | Root properties affect suffix choices |

**Example - Turkish Agglutination:**
```
Word: "Avrupalılaştıramadıklarımızdanmışsınızcasına"
Meaning: "As if you were one of those we could not Europeanize"
One word = entire sentence!
```

**Solutions:**

| Solution | Description | Effectiveness |
|----------|-------------|---------------|
| Subword tokenization | BPE/WordPiece handles morphemes | High |
| Morphological analysis | Explicit morpheme segmentation | High |
| Character-level models | Learn morphological patterns | Medium-High |
| Joint morphology+POS | Predict both together | High |
| Universal Dependencies | Cross-lingual annotation standards | Enables transfer |

**Python Code Example:**
```python
# Pipeline: Morphologically rich text -> Subword/Morphological analysis -> POS

# Approach 1: Subword tokenization (BPE handles morphology implicitly)
from transformers import AutoTokenizer

# Turkish BERT uses subword tokenization
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

turkish_word = "evlerinden"  # from their houses
tokens = tokenizer.tokenize(turkish_word)
print(tokens)
# ['ev', '##ler', '##in', '##den'] - morphemes separated!

# Approach 2: Morphological analyzer (language-specific)
# For Turkish: Zemberek, TRMorph
# Conceptual example:
def morphological_analysis(word):
    """
    Returns morpheme breakdown
    Example output for 'evlerinden':
    {
        'root': 'ev',
        'morphemes': ['ev', 'ler', 'i', 'n', 'den'],
        'tags': ['NOUN', 'PLURAL', 'POSS.3SG', 'ABL'],
        'pos': 'NOUN'
    }
    """
    pass

# Approach 3: Character-level CNN for morphology
import torch.nn as nn

class MorphAwareTagger(nn.Module):
    def __init__(self, char_vocab, word_vocab, num_tags):
        super().__init__()
        # Character-level representation
        self.char_embed = nn.Embedding(char_vocab, 50)
        self.char_cnn = nn.Conv1d(50, 100, kernel_size=3)
        
        # Word-level (subword) representation
        self.word_embed = nn.Embedding(word_vocab, 300)
        
        # Combine and classify
        self.lstm = nn.LSTM(400, 200, bidirectional=True)
        self.classifier = nn.Linear(400, num_tags)
    
    def forward(self, word_ids, char_ids):
        # Get character-level features (captures morphology)
        char_emb = self.char_embed(char_ids)
        char_features = self.char_cnn(char_emb.permute(0, 2, 1))
        char_repr = char_features.max(dim=-1)[0]
        
        # Combine with word embeddings
        word_emb = self.word_embed(word_ids)
        combined = torch.cat([word_emb, char_repr], dim=-1)
        
        # Sequence labeling
        lstm_out, _ = self.lstm(combined)
        logits = self.classifier(lstm_out)
        return logits
```

**Best Practices:**
1. Use multilingual models (mBERT, XLM-R) pre-trained on target language
2. Character-level features are essential
3. Consider joint morphological analysis + POS tagging
4. Universal Dependencies provides consistent cross-lingual tagset

**Interview Tips:**
- Subword tokenization is key breakthrough for morphologically rich languages
- Finnish/Hungarian can have 10,000+ forms per lemma
- Turkish BERT, Arabic BERT exist for major morphologically rich languages
- POS accuracy typically 5-10% lower than English for these languages

---

## Question 21
- [ ] Done
**How do you handle ambiguous words that can have multiple POS tags depending on context?**

**Answer:**

**Definition:**
POS ambiguity occurs when the same word can function as different parts of speech (e.g., "book" as noun or verb). Resolution requires contextual understanding using surrounding words, syntactic patterns, and sequential models that consider the entire sentence.

**Common Ambiguous Words:**

| Word | POS 1 | POS 2 | Example |
|------|-------|-------|---------|
| book | NN (noun) | VB (verb) | "Read the book" vs "Book a flight" |
| run | VB (verb) | NN (noun) | "Run fast" vs "A morning run" |
| that | DT (determiner) | IN (conjunction) | "That book" vs "I know that..." |
| light | JJ (adjective) | NN (noun) | "Light weight" vs "Turn on the light" |

**Resolution Approaches:**

| Method | How It Resolves Ambiguity |
|--------|---------------------------|
| N-gram context | Look at surrounding word POS patterns |
| HMM | Transition probabilities between tags |
| CRF | Global sequence optimization |
| BiLSTM | Bidirectional context encoding |
| Transformer | Full sentence attention |

**Contextual Clues:**
```
"I will book a table" 
- "will" + [WORD] → [WORD] is likely VB (verb after modal)
- "a" + [WORD] → Unlikely (article before verb is rare)
- Sequence context: VB wins

"I read the book"
- "the" + [WORD] → [WORD] is likely NN (article before noun)
- NN wins
```

**Python Code Example:**
```python
# Pipeline: Sentence -> Context-aware tagger -> Resolved POS

import spacy

nlp = spacy.load("en_core_web_sm")

# Same word, different contexts
sentences = [
    "I need to book a flight",      # book = VERB
    "I read an interesting book",   # book = NOUN
    "The light is too bright",      # light = NOUN
    "This bag is very light"        # light = ADJ
]

for sent in sentences:
    doc = nlp(sent)
    for token in doc:
        if token.text.lower() in ['book', 'light']:
            print(f"'{sent}' -> {token.text}: {token.pos_}")

# Output:
# 'I need to book a flight' -> book: VERB
# 'I read an interesting book' -> book: NOUN
# 'The light is too bright' -> light: NOUN
# 'This bag is very light' -> light: ADJ

# How context helps (simplified demonstration)
def get_context_features(tokens, idx):
    """Extract context features for disambiguation"""
    features = {
        'word': tokens[idx].lower(),
        'prev_word': tokens[idx-1].lower() if idx > 0 else '<START>',
        'next_word': tokens[idx+1].lower() if idx < len(tokens)-1 else '<END>',
        'prev_pos': None,  # From previous prediction
    }
    
    # Pattern-based hints
    if features['prev_word'] in ['the', 'a', 'an']:
        features['likely_noun'] = True
    if features['prev_word'] in ['will', 'can', 'should', 'to']:
        features['likely_verb'] = True
        
    return features
```

**Why Transformers Excel:**
- Self-attention sees entire sentence at once
- Can capture long-range dependencies
- Pre-training learns grammatical patterns
- Contextual embeddings differ for same word in different contexts

**Interview Tips:**
- ~11% of English words are ambiguous in POS
- Contextual embeddings (BERT) give different vectors for same word in different contexts
- HMM/CRF use transition probabilities: P(VB|MD) > P(NN|MD)
- Accuracy for ambiguous words is typically lower than overall accuracy

---

## Question 22
- [ ] Done
**What is the role of context window in POS tagging and how do neural models capture it?**

**Answer:**

**Definition:**
Context window refers to the surrounding words considered when predicting a token's POS tag. Larger context enables better disambiguation. Neural models capture context through recurrent connections (LSTM), convolutional filters (CNN), or self-attention (Transformer).

**Context Window Importance:**
```
"The old man the boats"
Without context: "man" → likely NOUN (most common)
With context: "old man" → "man" is VERB (old people operate the boats)

Full context needed to resolve this garden-path sentence.
```

**Context Window by Model Type:**

| Model | Context Window | Mechanism |
|-------|---------------|-----------|
| N-gram | Fixed (2-5 words) | Look at adjacent n-grams |
| HMM | Limited (Markov) | Previous state only |
| CRF | Fixed feature window | Handcrafted features |
| LSTM | Entire sequence | Hidden state propagation |
| BiLSTM | Entire sequence (both directions) | Forward + backward states |
| Transformer | Entire sequence | Self-attention over all tokens |

**How Neural Models Capture Context:**

**1. BiLSTM:**
```
Forward:  →  h₁ → h₂ → h₃ → h₄  (past context)
Backward: ←  h₁ ← h₂ ← h₃ ← h₄  (future context)
Combined: [h_forward; h_backward] for each position
```

**2. Transformer Self-Attention:**
```
Every token attends to every other token
Attention(Q, K, V) = softmax(QK^T / √d) V

Token "man" attends to:
- "old" (helps determine subject/verb)
- "boats" (object → "man" is verb)
```

**Python Code Example:**
```python
# Pipeline: Demonstrate context capture in different models

import torch
import torch.nn as nn

# 1. Fixed Window (Traditional)
class WindowTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, window_size, num_tags):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Context = current word + window on each side
        self.fc = nn.Linear(embed_dim * (2 * window_size + 1), num_tags)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embed(x)  # (batch, seq_len, embed)
        # Create windows (simplified - would need padding)
        # Each position sees [-window, ..., 0, ..., +window]
        pass

# 2. BiLSTM (Full sequence context)
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
    
    def forward(self, x):
        embedded = self.embed(x)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out contains context from entire sequence
        logits = self.fc(lstm_out)
        return logits

# 3. Transformer (Self-attention)
# Each position attends to all other positions
class TransformerTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_tags):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_tags)
    
    def forward(self, x):
        embedded = self.embed(x).permute(1, 0, 2)  # (seq, batch, embed)
        # Self-attention: every token sees every other token
        attended, _ = self.attention(embedded, embedded, embedded)
        logits = self.fc(attended.permute(1, 0, 2))
        return logits
```

**Context Window Trade-offs:**

| Aspect | Small Window | Large Window |
|--------|--------------|--------------|
| Speed | Faster | Slower |
| Memory | Less | More |
| Long dependencies | Misses | Captures |
| Local patterns | Good | Good |

**Interview Tips:**
- Transformers have unlimited context (within sequence length)
- BiLSTM captures context but may forget distant information
- BERT uses 512 token context window (sufficient for most POS tasks)
- Context is why neural POS taggers achieve 97%+ accuracy

---

## Question 23
- [ ] Done
**How do you evaluate NER systems? Explain precision, recall, F1, and entity-level vs token-level metrics.**

**Answer:**

**Definition:**
NER evaluation uses precision (correct predictions / all predictions), recall (correct predictions / all ground truth), and F1 (harmonic mean). Entity-level evaluation requires exact span and type match; token-level evaluates each token independently.

**Core Metrics:**

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Entity-Level vs Token-Level:**

| Metric Type | What Counts as Correct | Strictness |
|-------------|------------------------|------------|
| Entity-level (Exact) | Exact span + correct type | Strict |
| Entity-level (Partial) | Overlap + correct type | Medium |
| Token-level | Each token independently | Lenient |

**Example:**
```
Ground Truth: [New York City]_LOC
Prediction:   [New York]_LOC [City]_LOC

Entity-level (Exact): 
  - 0 TP (spans don't match exactly)
  - 2 FP (predicted entities)
  - 1 FN (missed ground truth)

Token-level:
  - 3 TP (New, York, City all tagged LOC)
  - 0 FP, 0 FN
```

**Evaluation Variants:**

| Variant | Requirement |
|---------|-------------|
| Strict | Exact boundary + exact type |
| Type-only | Correct type (any overlap) |
| Boundary-only | Exact boundary (any type) |
| Partial | Some overlap + correct type |

**Python Code Example:**
```python
# Pipeline: Predictions + Ground Truth -> Compute metrics

from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import classification_report as token_report

# Entity-level evaluation using seqeval
y_true = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'I-LOC']]
y_pred = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O']]

# Entity-level F1 (seqeval handles BIO correctly)
print(classification_report(y_true, y_pred))
# Note: "New York City" with last token wrong = entity missed

# Token-level evaluation (each token independently)
y_true_flat = [tag for seq in y_true for tag in seq]
y_pred_flat = [tag for seq in y_pred for tag in seq]
print(token_report(y_true_flat, y_pred_flat))

# Manual entity extraction for exact match
def extract_entities(tags):
    entities = []
    current = None
    start = None
    
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if current:
                entities.append((start, i, current))
            current = tag[2:]
            start = i
        elif tag.startswith('I-') and current:
            continue
        else:
            if current:
                entities.append((start, i, current))
            current = None
    
    if current:
        entities.append((start, len(tags), current))
    
    return set(entities)

# Compute entity-level metrics manually
true_entities = extract_entities(y_true[0])
pred_entities = extract_entities(y_pred[0])

tp = len(true_entities & pred_entities)
fp = len(pred_entities - true_entities)
fn = len(true_entities - pred_entities)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Entity P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
```

**Micro vs Macro Averaging:**
- **Micro:** Aggregate TP, FP, FN across all classes, then compute
- **Macro:** Compute per-class, then average (treats rare classes equally)

**Interview Tips:**
- Always report entity-level F1 for NER (industry standard)
- CoNLL-2003 uses entity-level exact match
- seqeval library is standard for NER evaluation
- Token-level inflates scores (partial entities count as correct)

---

## Question 24
- [ ] Done
**What is few-shot NER and how can you identify new entity types with minimal examples?**

**Answer:**

**Definition:**
Few-shot NER identifies entities of new types using only a handful of examples (1-10), without retraining the model from scratch. Approaches include prompt-based methods, prototype networks, and in-context learning with large language models.

**Core Challenge:**
```
Traditional NER: Train on 10,000+ examples per entity type
Few-shot NER: New entity type "PRODUCT" with only 5 examples

Examples provided:
- "iPhone 15" -> PRODUCT
- "Tesla Model 3" -> PRODUCT  
- "MacBook Pro" -> PRODUCT

Now identify: "The Galaxy S24 has great features"
                    ↓
             [Galaxy S24]_PRODUCT
```

**Approaches:**

| Method | Description | LLM Required |
|--------|-------------|--------------|
| Prototype Networks | Learn entity type embeddings | No |
| Prompt-based | Frame as text generation/QA | Yes |
| In-context Learning | Provide examples in prompt | Yes (GPT-3+) |
| Meta-learning | Learn to learn from few examples | No |

**1. Prototype Networks:**
- Compute average embedding for each entity type from examples
- Classify new mentions by nearest prototype
- Works with small encoder models

**2. In-Context Learning (LLMs):**
```
Prompt:
"Identify PRODUCT entities in text. Examples:
'I bought an iPhone 15' -> 'iPhone 15' is PRODUCT
'Tesla Model 3 is fast' -> 'Tesla Model 3' is PRODUCT

Now identify PRODUCT in: 'The Galaxy S24 has great features'"

Output: "Galaxy S24"
```

**Python Code Example:**
```python
# Approach 1: Prototype-based Few-shot NER

import torch
from transformers import AutoTokenizer, AutoModel

class PrototypeFewShotNER:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.prototypes = {}  # entity_type -> embedding
    
    def get_span_embedding(self, text, start, end):
        """Get embedding for entity span"""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        # Average token embeddings in span
        span_emb = outputs.last_hidden_state[0, start:end].mean(dim=0)
        return span_emb
    
    def add_examples(self, entity_type, examples):
        """
        examples: list of (text, entity_text) tuples
        """
        embeddings = []
        for text, entity_text in examples:
            start = text.find(entity_text)
            end = start + len(entity_text)
            emb = self.get_span_embedding(text, start, end)
            embeddings.append(emb)
        
        # Prototype = average of example embeddings
        self.prototypes[entity_type] = torch.stack(embeddings).mean(dim=0)
    
    def predict(self, text, candidate_spans):
        """Classify candidate spans"""
        results = []
        for start, end in candidate_spans:
            span_emb = self.get_span_embedding(text, start, end)
            
            # Find nearest prototype
            best_type = None
            best_sim = -1
            for etype, proto in self.prototypes.items():
                sim = torch.cosine_similarity(span_emb, proto, dim=0)
                if sim > best_sim:
                    best_sim = sim
                    best_type = etype
            
            results.append((start, end, best_type, best_sim.item()))
        return results

# Usage
ner = PrototypeFewShotNER()

# Few-shot examples for new entity type
ner.add_examples('PRODUCT', [
    ("I bought an iPhone 15", "iPhone 15"),
    ("Tesla Model 3 is fast", "Tesla Model 3"),
    ("Love my MacBook Pro", "MacBook Pro")
])

# Predict on new text
# candidates = get_candidate_spans(text)  # From NP chunker or sliding window

# Approach 2: LLM In-Context Learning
from openai import OpenAI

def few_shot_ner_llm(examples, text, entity_type):
    prompt = f"Identify {entity_type} entities in text.\n\nExamples:\n"
    for ex_text, entity in examples:
        prompt += f"Text: '{ex_text}' -> {entity_type}: '{entity}'\n"
    prompt += f"\nNow identify {entity_type} in: '{text}'"
    
    # Call LLM API
    # response = client.completions.create(...)
    return prompt  # Would return actual entities from LLM
```

**Interview Tips:**
- Few-shot NER is hot research area
- GPT-3/4 with in-context learning is surprisingly effective
- Prototype networks work without large LLMs
- SetFit and sentence-transformers enable efficient few-shot learning

---

## Question 25
- [ ] Done
**How do you handle tokenization for domain-specific texts like legal, medical, or code?**

**Answer:**

**Definition:**
Domain-specific tokenization requires handling specialized vocabulary (medical terms, legal jargon, code syntax) that general tokenizers fragment poorly. Solutions include domain-specific pre-trained tokenizers, vocabulary expansion, and custom tokenization rules.

**Domain Challenges:**

| Domain | Challenge | Example |
|--------|-----------|---------|
| Medical | Complex terms, abbreviations | "COVID-19", "mg/dL", "t.i.d." |
| Legal | Latin phrases, citations | "habeas corpus", "42 U.S.C. § 1983" |
| Code | Syntax, identifiers, operators | "getUserById()", "!=", "//comment" |
| Scientific | Chemical formulas, equations | "H₂SO₄", "E=mc²" |

**General Tokenizer Problems:**
```
Medical: "acetaminophen" → ['ace', '##tam', '##ino', '##phen']  # 4 tokens
Code: "getUserById" → ['get', '##User', '##By', '##Id']  # Loses semantics
Legal: "§1983" → ['§', '##19', '##83']  # Splits legal reference
```

**Solutions:**

| Solution | Description | When to Use |
|----------|-------------|-------------|
| Domain pre-trained model | Use BioBERT, LegalBERT, CodeBERT | Best option if available |
| Vocabulary extension | Add domain terms to tokenizer | Limited new terms |
| Custom pre-tokenization | Domain-aware splitting rules | Complex syntax (code) |
| Train custom tokenizer | Train BPE on domain corpus | Full control needed |

**Python Code Example:**
```python
# Pipeline: Domain text -> Domain-aware tokenization

# Solution 1: Use domain-specific models
from transformers import AutoTokenizer

# Medical
bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
tokens = bio_tokenizer.tokenize("acetaminophen dosage 500mg")
# Better handling of medical terms

# Code
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokens = code_tokenizer.tokenize("def getUserById(user_id):")
# Understands code syntax

# Legal
legal_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Solution 2: Add tokens to vocabulary
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add domain-specific tokens
new_tokens = ['covid-19', 'mrna', 'acetaminophen', '§', 'getUserById']
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens")

# Now these tokenize as single tokens
tokens = tokenizer.tokenize("covid-19 vaccine")
# ['covid-19', 'vaccine'] instead of ['co', '##vid', '-', '19', ...]

# Remember to resize model embeddings!
# model.resize_token_embeddings(len(tokenizer))

# Solution 3: Custom pre-tokenization for code
import re

def code_pretokenize(code):
    """Split code into meaningful units before subword tokenization"""
    # Split camelCase: getUserById -> get User By Id
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    # Split snake_case: user_id -> user id
    code = re.sub(r'_', ' ', code)
    # Keep operators together
    code = re.sub(r'([!=<>]=)', r' \1 ', code)
    return code

code = "getUserById(userId) != null"
pretokenized = code_pretokenize(code)
# "get User By Id(user Id) != null"

# Solution 4: Train custom BPE tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_domain_tokenizer(corpus_file, vocab_size=30000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    tokenizer.train([corpus_file], trainer)
    return tokenizer

# Train on medical corpus for medical-specific BPE
```

**Best Practices:**
1. First check if domain-specific model exists (BioBERT, CodeBERT, etc.)
2. If fine-tuning, add frequent domain terms to vocabulary
3. For code: use dedicated code models (CodeBERT, GraphCodeBERT)
4. Never split important domain identifiers

**Interview Tips:**
- Domain-specific models are pre-trained on domain corpora (PubMed, legal docs, GitHub)
- Adding tokens requires resizing model embeddings and additional training
- Code tokenization is unique - needs to handle syntax meaningfully
- Medical: BioBERT, PubMedBERT; Legal: LegalBERT; Code: CodeBERT

---

## Question 26
- [ ] Done
**What is the impact of tokenization choices on model vocabulary size and training efficiency?**

**Answer:**

**Definition:**
Tokenization directly determines vocabulary size, sequence length, and memory usage. Larger vocabularies mean more embedding parameters but shorter sequences; smaller vocabularies mean longer sequences but fewer parameters. This trade-off affects training speed, memory, and model capacity.

**Core Trade-offs:**

| Tokenization | Vocab Size | Sequence Length | Parameters | Memory |
|--------------|------------|-----------------|------------|--------|
| Character-level | ~100 | Very long | Few embeddings | High (long sequences) |
| Subword (BPE) | ~30K-50K | Moderate | Moderate | Balanced |
| Word-level | 100K+ | Short | Many embeddings | High (large vocab) |

**Impact Analysis:**

**1. Vocabulary Size Impact:**
```
Embedding layer size = vocab_size × embedding_dim

Word-level:  100,000 × 768 = 76.8M parameters (embeddings alone!)
Subword:     30,000 × 768 = 23M parameters
Character:   256 × 768 = 0.2M parameters
```

**2. Sequence Length Impact:**
```
Text: "internationalization"

Word-level:    ["internationalization"]           → 1 token
Subword (BPE): ["international", "ization"]       → 2 tokens
Character:     ['i','n','t','e','r',...,'n']      → 20 tokens

Transformer complexity: O(n²) where n = sequence length
Longer sequences = slower training + more memory
```

**3. Training Efficiency Equation:**

$$\text{Compute} \propto \text{Batch Size} \times \text{Seq Length}^2 \times \text{Vocab Size}$$

**Python Code Example:**
```python
# Pipeline: Compare tokenization choices and their impacts

from transformers import AutoTokenizer
import time

text = "The internationalization of artificial intelligence research continues."

tokenizers = {
    'bert': 'bert-base-uncased',
    'gpt2': 'gpt2',
    'char': None  # Manual character-level
}

# Compare token counts
print("Tokenization comparison:")
print("-" * 50)

# BERT (WordPiece ~30K vocab)
bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tok.tokenize(text)
print(f"BERT WordPiece: {len(bert_tokens)} tokens")
print(f"  Vocab size: {bert_tok.vocab_size}")
print(f"  Tokens: {bert_tokens[:5]}...")

# GPT-2 (BPE ~50K vocab)
gpt_tok = AutoTokenizer.from_pretrained('gpt2')
gpt_tokens = gpt_tok.tokenize(text)
print(f"GPT-2 BPE: {len(gpt_tokens)} tokens")
print(f"  Vocab size: {gpt_tok.vocab_size}")

# Character-level
char_tokens = list(text)
print(f"Character: {len(char_tokens)} tokens")
print(f"  Vocab size: ~256")

# Memory/compute estimation
def estimate_memory(seq_len, vocab_size, hidden_dim=768, batch_size=32):
    """Rough memory estimate for transformer forward pass"""
    # Embedding lookup
    embed_params = vocab_size * hidden_dim
    
    # Attention matrices (simplified)
    attention_mem = batch_size * seq_len * seq_len * 4  # float32
    
    # Sequence processing
    hidden_mem = batch_size * seq_len * hidden_dim * 4
    
    total_mb = (embed_params + attention_mem + hidden_mem) / (1024 * 1024)
    return total_mb

print("\nMemory estimates (batch=32):")
print(f"BERT (seq={len(bert_tokens)}): {estimate_memory(len(bert_tokens), 30522):.1f} MB")
print(f"Char (seq={len(char_tokens)}): {estimate_memory(len(char_tokens), 256):.1f} MB")
```

**Practical Guidelines:**

| Scenario | Recommendation |
|----------|----------------|
| General NLP | Subword (30K-50K vocab) |
| Multilingual | Larger vocab (100K+) |
| Memory constrained | Smaller vocab, accept longer sequences |
| Speed critical | Balanced vocab (30K) |
| Rare vocabulary domain | Smaller subword vocab |

**Interview Tips:**
- BERT: 30K vocab, GPT-2: 50K vocab, XLM-R: 250K vocab (multilingual)
- Doubling vocab doubles embedding parameters
- Sequence length has O(n²) impact on attention computation
- Subword is the sweet spot - that's why everyone uses it

---

## Question 27
- [ ] Done
**How do multilingual models like mBERT handle tokenization across different scripts and languages?**

**Answer:**

**Definition:**
Multilingual BERT (mBERT) uses a shared WordPiece vocabulary trained on 104 languages, enabling cross-lingual transfer. It handles multiple scripts (Latin, Cyrillic, Chinese, Arabic) by including script-specific characters and subwords in a single unified vocabulary.

**Core Mechanism:**

| Aspect | How mBERT Handles It |
|--------|---------------------|
| Vocabulary | Shared 110K WordPiece tokens across all languages |
| Scripts | Includes characters from all scripts in vocab |
| Training | Masked LM on Wikipedia for each language |
| Representation | Shared embedding space across languages |

**Vocabulary Composition:**
```
mBERT Vocabulary (~110K tokens):
- English subwords: "the", "ing", "tion", ...
- Chinese characters: "的", "是", "我", ...
- Arabic subwords: "ال", "من", ...
- Cyrillic: "и", "не", "это", ...
- Shared subwords: Numbers, punctuation, common patterns
```

**Cross-lingual Transfer Intuition:**
```
English: "I love cats"
French:  "J'aime les chats"
German:  "Ich liebe Katzen"

Shared structure → Similar representations → Transfer possible
```

**Vocabulary Distribution Challenge:**
```
High-resource languages (English) → More vocabulary slots
Low-resource languages → Fewer slots, more character-level splitting

"internationalization" (English) → 2-3 tokens
Same length word in low-resource language → 5-10 tokens
```

**Python Code Example:**
```python
# Pipeline: See how mBERT tokenizes different languages

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

texts = {
    'English': "Machine learning is powerful",
    'French': "L'apprentissage automatique est puissant",
    'Chinese': "机器学习很强大",
    'Arabic': "التعلم الآلي قوي",
    'Hindi': "मशीन लर्निंग शक्तिशाली है",
    'Russian': "Машинное обучение мощное"
}

print("mBERT Tokenization across languages:")
print("-" * 60)

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"{lang}: {text}")
    print(f"  Tokens ({len(tokens)}): {tokens}")
    print()

# Output shows:
# - English: Fewer subwords (well-represented in vocab)
# - Chinese: Character-level + some common phrases
# - Low-resource: More fragmentation

# Vocabulary analysis
print(f"Total vocabulary size: {tokenizer.vocab_size}")

# Check language-specific tokens
sample_tokens = ['machine', '机器', 'обучение', 'التعلم']
for tok in sample_tokens:
    if tok in tokenizer.vocab:
        print(f"'{tok}' is in vocabulary")
    else:
        subtokens = tokenizer.tokenize(tok)
        print(f"'{tok}' splits to: {subtokens}")
```

**XLM-RoBERTa Improvements:**
- 250K vocabulary (larger than mBERT's 110K)
- Trained on more languages (100+)
- Better coverage for low-resource languages
- Uses SentencePiece (language-agnostic)

**Challenges:**
1. **Curse of multilinguality:** Single model capacity shared across languages
2. **Vocabulary imbalance:** High-resource languages dominate
3. **Script mixing:** Code-switching text is challenging

**Interview Tips:**
- mBERT vocab: 110K, XLM-R vocab: 250K
- Shared vocabulary enables zero-shot cross-lingual transfer
- Low-resource languages get fragmented tokenization
- For best results on single language: use language-specific model

---

## Question 28
- [ ] Done
**What techniques help preserve named entities during tokenization (e.g., preventing "New York" from splitting)?**

**Answer:**

**Definition:**
Entity-preserving tokenization ensures multi-word entities stay together or are properly tracked during subword tokenization. Techniques include pre-tokenization with NER, adding entities to vocabulary, and maintaining token-to-entity alignment for downstream processing.

**The Problem:**
```
Standard tokenization:
"New York Stock Exchange" → ["New", "York", "Stock", "Exchange"]
                           or ["New", "York", "Stock", "Ex", "##change"]

Issue: Entity boundaries lost; "New York" split from "Stock Exchange"
```

**Solutions:**

| Technique | Description | Trade-off |
|-----------|-------------|-----------|
| Add to vocabulary | Add entity as single token | Increases vocab size |
| Pre-tokenization | Run NER first, mark entities | Two-pass processing |
| Entity markers | Wrap entities with special tokens | Extra tokens in sequence |
| Alignment tracking | Map subwords back to entities | Post-processing overhead |

**Approach 1: Add Entities to Vocabulary**
```python
# Add multi-word entities as single tokens
tokenizer.add_tokens(["New York", "Machine Learning", "United States"])
```

**Approach 2: Entity Markers**
```
Original: "I visited New York last week"
Marked:   "I visited [ENT]New York[/ENT] last week"
```

**Approach 3: Subword-to-Entity Alignment**
```
Text: "New York"
Subwords: ["New", "York"]
Alignment: [(0, 'B-LOC'), (1, 'I-LOC')]

During inference: Aggregate subword predictions
```

**Python Code Example:**
```python
# Pipeline: Text with entities -> Preserve entities -> Tokenize

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Approach 1: Add entities to vocabulary
entities = ["new york", "machine learning", "united states", "world war ii"]
tokenizer.add_tokens(entities)
print(f"Vocab size after: {len(tokenizer)}")

text = "I studied machine learning in new york"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['i', 'studied', 'machine learning', 'in', 'new york']
# Entities preserved as single tokens!

# Approach 2: Subword alignment for NER
def align_labels_with_subwords(text, entities, tokenizer):
    """
    Align entity labels with subword tokens
    entities: list of (start_char, end_char, label)
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    # Initialize labels
    labels = ['O'] * len(tokens)
    
    for start, end, label in entities:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start is None:
                continue
            # Check if token overlaps with entity
            if tok_start >= start and tok_end <= end:
                if tok_start == start:
                    labels[idx] = f'B-{label}'
                else:
                    labels[idx] = f'I-{label}'
    
    return list(zip(tokens, labels))

text = "Steve Jobs founded Apple in California"
entities = [(0, 10, 'PER'), (19, 24, 'ORG'), (28, 38, 'LOC')]

aligned = align_labels_with_subwords(text, entities, tokenizer)
for token, label in aligned:
    if label != 'O':
        print(f"{token}: {label}")

# Approach 3: Entity marker tokens
def add_entity_markers(text, entities):
    """Add markers around entities"""
    # Sort by position (reverse to preserve indices)
    sorted_ents = sorted(entities, key=lambda x: x[0], reverse=True)
    
    for start, end, label in sorted_ents:
        text = text[:end] + f" [/{label}]" + text[end:]
        text = text[:start] + f"[{label}] " + text[start:]
    
    return text

marked = add_entity_markers("Apple was founded in California", 
                            [(0, 5, 'ORG'), (21, 31, 'LOC')])
print(marked)
# "[ORG] Apple [/ORG] was founded in [LOC] California [/LOC]"
```

**Best Practice for NER:**
1. Don't modify vocabulary (pretrained weights won't exist for new tokens)
2. Use offset mapping for alignment
3. Take first subword's prediction for entity label
4. Or use special handling (max pooling, voting) for multi-subword entities

**Interview Tips:**
- Adding tokens requires retraining/finetuning embeddings
- Offset mapping is standard way to handle alignment
- First-subword labeling is common convention for NER
- Entity markers help models learn entity boundaries

---

## Question 29
- [ ] Done
**How do you implement real-time NER for streaming text applications?**

**Answer:**

**Definition:**
Real-time NER processes text streams with low latency requirements (milliseconds). Implementation requires model optimization (quantization, distillation), efficient batching, caching, and handling partial/incomplete sentences from continuous streams.

**Core Challenges:**

| Challenge | Description |
|-----------|-------------|
| Latency | Must respond in milliseconds |
| Throughput | Handle high volume of requests |
| Context | Entities may span message boundaries |
| Partial text | Incomplete sentences in streams |
| Resource usage | Limited CPU/memory budget |

**Optimization Strategies:**

| Strategy | Latency Reduction | Trade-off |
|----------|-------------------|-----------|
| Model distillation | 3-5x faster | Slight accuracy drop |
| Quantization (INT8) | 2-4x faster | Minimal accuracy drop |
| ONNX Runtime | 1.5-2x faster | Export complexity |
| Smaller model | 5-10x faster | Accuracy drop |
| Batching | Better throughput | Higher latency |
| Caching | Near-instant for repeats | Memory usage |

**Architecture for Streaming:**
```
Input Stream → Buffer → Sentence Detector → NER Model → Output Stream
                ↓
         Context Window (for cross-boundary entities)
```

**Python Code Example:**
```python
# Pipeline: Text stream -> Buffer -> Batch NER -> Emit entities

import time
from collections import deque
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch

# Step 1: Load optimized model
class RealtimeNER:
    def __init__(self, model_name="dslim/bert-base-NER", device="cpu"):
        # Use smaller/distilled model for speed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Optimization: Use ONNX or TorchScript
        # self.model = torch.jit.script(self.model)
        
        self.device = device
        self.model.to(device)
        
        # Buffer for streaming
        self.buffer = deque(maxlen=5)  # Keep last 5 messages for context
        
        # Cache for repeated queries
        self.cache = {}
    
    @torch.no_grad()
    def predict(self, text):
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
        
        # Extract entities
        entities = self._extract_entities(tokens, labels)
        
        # Cache result
        self.cache[text] = entities
        
        return entities
    
    def _extract_entities(self, tokens, labels):
        entities = []
        current_entity = []
        current_label = None
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if label.startswith('B-'):
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = [token.replace('##', '')]
                current_label = label[2:]
            elif label.startswith('I-') and current_label:
                current_entity.append(token.replace('##', ''))
            else:
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = []
                current_label = None
        
        if current_entity:
            entities.append((' '.join(current_entity), current_label))
        
        return entities
    
    def process_stream(self, text_generator):
        """Process streaming text"""
        for text in text_generator:
            start = time.time()
            entities = self.predict(text)
            latency = (time.time() - start) * 1000
            yield {
                'text': text,
                'entities': entities,
                'latency_ms': latency
            }

# Step 2: Simulate streaming
def simulate_stream():
    messages = [
        "Breaking: Elon Musk announces new Tesla factory",
        "Apple Inc reports record Q4 earnings",
        "President Biden visits Berlin tomorrow"
    ]
    for msg in messages:
        yield msg
        time.sleep(0.1)  # Simulate stream delay

# Step 3: Run real-time NER
ner = RealtimeNER()

for result in ner.process_stream(simulate_stream()):
    print(f"Text: {result['text']}")
    print(f"Entities: {result['entities']}")
    print(f"Latency: {result['latency_ms']:.1f}ms\n")
```

**Production Optimizations:**
1. Use DistilBERT (40% smaller, 60% faster)
2. ONNX Runtime for inference
3. Batch multiple messages when possible
4. GPU with TensorRT for high throughput
5. Async processing with message queues

**Interview Tips:**
- DistilBERT-NER is good for real-time (60% faster than BERT)
- Target latency: <50ms for real-time applications
- Consider spaCy for simpler/faster NER if BERT accuracy not needed
- Caching helps with repeated/similar text patterns

---

## Question 30
- [ ] Done
**What is cross-lingual transfer learning for NER? How does it work?**

**Answer:**

**Definition:**
Cross-lingual transfer trains NER on one language (usually high-resource like English) and applies it to another language (low-resource) without target language training data. It works through multilingual representations that map semantically similar words across languages to nearby embeddings.

**Core Mechanism:**
```
Training: English NER data
         "Apple Inc. is in California"
         [Apple Inc.]_ORG    [California]_LOC

Transfer: German (zero-shot)
         "Apple Inc. ist in Kalifornien"
         [Apple Inc.]_ORG    [Kalifornien]_LOC

Why it works: Multilingual embeddings place "California" and "Kalifornien" 
             in similar representation space.
```

**Transfer Approaches:**

| Approach | Description | Requirements |
|----------|-------------|--------------|
| Zero-shot | Train on source, apply directly to target | Multilingual model |
| Translate-train | Translate source training data | Translation system |
| Translate-test | Translate target to source at inference | Translation system |
| Few-shot | Add small target language data | Limited target labels |

**How Multilingual Representations Enable Transfer:**

$$\text{embed}(\text{"king"}_{\text{EN}}) \approx \text{embed}(\text{"roi"}_{\text{FR}}) \approx \text{embed}(\text{"König"}_{\text{DE}})$$

Semantically similar words across languages have similar embeddings.

**Python Code Example:**
```python
# Pipeline: Train on English -> Apply to other languages (zero-shot)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load multilingual model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=9  # BIO tags for 4 entity types + O
)

# Step 2: Fine-tune on English NER (CoNLL-2003)
# Assume english_dataset is prepared
# trainer = Trainer(model=model, train_dataset=english_dataset, ...)
# trainer.train()

# Step 3: Zero-shot inference on other languages
def predict_ner(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    id2label = model.config.id2label
    results = []
    for token, pred, offset in zip(tokens, predictions, offset_mapping[0]):
        if offset[0] == offset[1]:  # Skip special tokens
            continue
        results.append((token, id2label[pred.item()]))
    
    return results

# Test on multiple languages (zero-shot transfer)
test_sentences = {
    'English': "Barack Obama visited Paris last week",
    'German': "Angela Merkel besuchte Berlin gestern",
    'French': "Emmanuel Macron est à Londres",
    'Spanish': "El presidente visitó Madrid ayer",
    'Chinese': "习近平访问了北京"
}

print("Zero-shot Cross-lingual NER:")
print("-" * 50)
for lang, text in test_sentences.items():
    result = predict_ner(text, model, tokenizer)
    entities = [(t, l) for t, l in result if l != 'O']
    print(f"{lang}: {text}")
    print(f"  Entities: {entities}\n")

# Step 4: Translate-train approach (alternative)
def translate_train_data(source_data, source_lang, target_lang, translator):
    """
    Translate training data to target language
    Note: Entity alignment is challenging
    """
    translated_data = []
    for text, entities in source_data:
        # Translate text
        translated_text = translator.translate(text, source_lang, target_lang)
        # Align entities (complex - may use word alignment)
        # translated_entities = align_entities(...)
        # translated_data.append((translated_text, translated_entities))
    return translated_data
```

**Performance (Typical F1 Scores):**

| Target Language | Zero-shot mBERT | Zero-shot XLM-R |
|-----------------|-----------------|-----------------|
| German | 65-70% | 70-75% |
| Spanish | 70-75% | 75-80% |
| Dutch | 70-75% | 75-80% |
| Chinese | 50-55% | 55-65% |

**Best Practices:**
1. Use XLM-RoBERTa over mBERT for better cross-lingual transfer
2. Few-shot (10-50 examples) significantly boosts performance
3. Related languages transfer better (English → German > English → Chinese)
4. Translate-train often outperforms zero-shot if good translation available

**Interview Tips:**
- XLM-RoBERTa is current state-of-the-art for cross-lingual NER
- Zero-shot works surprisingly well for related languages
- Transfer degrades for distant language pairs (different scripts)
- Shared vocabulary tokens (proper nouns, numbers) transfer best

---

## Question 31
- [ ] Done
**How do you handle NER for entities that change over time (new celebrities, companies, products)?**

**Answer:**

**Definition:**
Temporal entity drift occurs when new entities emerge and existing ones become obsolete. Solutions include periodic retraining, gazetteer updates, continual learning, and leveraging external knowledge bases that are regularly updated.

**The Problem:**
```
Model trained in 2020:
- Knows: "Tesla", "iPhone 11", "Trump"
- Doesn't know: "ChatGPT", "iPhone 15", "Threads app"

Real-world: ~1000s of new notable entities emerge monthly
```

**Strategies:**

| Strategy | Description | Effort |
|----------|-------------|--------|
| Periodic retraining | Retrain on updated data periodically | High |
| Gazetteer updates | Add new entities to lookup lists | Low |
| Continual learning | Update model incrementally | Medium |
| Knowledge base linking | Link to updated KB (Wikidata) | Medium |
| LLM zero-shot | Use LLMs with recent knowledge | Low |

**Approach 1: Gazetteer-Enhanced NER**
```
Static model + Dynamic gazetteers

Gazetteer (updated weekly):
- NEW_PRODUCTS: ["iPhone 15", "ChatGPT", "Vision Pro"]
- NEW_COMPANIES: ["OpenAI", "Anthropic", "Threads"]
- NEW_PERSONS: [trending celebrities]

Combine: Model predictions + Gazetteer matches
```

**Approach 2: Continual Learning**
- Collect new entity examples over time
- Fine-tune model incrementally
- Use elastic weight consolidation to prevent forgetting

**Python Code Example:**
```python
# Pipeline: Updatable NER with dynamic gazetteers

import re
from datetime import datetime

class TemporalNER:
    def __init__(self, base_model):
        self.model = base_model  # Pre-trained NER model
        self.gazetteers = {
            'PRODUCT': set(),
            'ORG': set(),
            'PER': set()
        }
        self.last_updated = None
    
    def update_gazetteers(self, new_entities):
        """
        Update gazetteers with new entities
        Called periodically (daily/weekly)
        """
        for entity_type, entities in new_entities.items():
            self.gazetteers[entity_type].update(entities)
        self.last_updated = datetime.now()
    
    def predict(self, text):
        # Step 1: Get model predictions
        model_entities = self.model.predict(text)
        
        # Step 2: Add gazetteer matches
        gazetteer_entities = self._match_gazetteers(text)
        
        # Step 3: Merge (gazetteer overrides for new entities)
        merged = self._merge_entities(model_entities, gazetteer_entities)
        
        return merged
    
    def _match_gazetteers(self, text):
        """Find entities from gazetteers in text"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, entity_set in self.gazetteers.items():
            for entity in entity_set:
                # Case-insensitive search
                pattern = re.escape(entity.lower())
                for match in re.finditer(pattern, text_lower):
                    entities.append({
                        'text': text[match.start():match.end()],
                        'start': match.start(),
                        'end': match.end(),
                        'type': entity_type,
                        'source': 'gazetteer'
                    })
        
        return entities
    
    def _merge_entities(self, model_ents, gaz_ents):
        """Merge model and gazetteer predictions"""
        # Prefer gazetteer for overlapping spans
        all_entities = gaz_ents.copy()
        
        for ment in model_ents:
            overlap = False
            for gent in gaz_ents:
                if self._spans_overlap(ment, gent):
                    overlap = True
                    break
            if not overlap:
                all_entities.append(ment)
        
        return all_entities
    
    def _spans_overlap(self, e1, e2):
        return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

# Usage
ner = TemporalNER(base_model=None)  # Replace with actual model

# Weekly update with new entities
ner.update_gazetteers({
    'PRODUCT': ['iPhone 15', 'ChatGPT', 'Claude', 'Gemini'],
    'ORG': ['OpenAI', 'Anthropic', 'Mistral AI'],
    'PER': ['Sam Altman', 'Dario Amodei']
})

# Now recognizes new entities even if model was trained in 2020
```

**Production Best Practices:**
1. Maintain curated entity lists by domain
2. Subscribe to news/trend APIs for automatic entity discovery
3. Use Wikidata SPARQL queries to get new entities
4. Quarterly model retraining with accumulated data
5. A/B test updated models before deployment

**Interview Tips:**
- Gazetteers are simple but effective for entity drift
- Continual learning is research area - catastrophic forgetting is a challenge
- LLMs (GPT-4) have more recent knowledge but may hallucinate
- Hybrid: use model for common entities, gazetteers for trending/new ones

---

## Question 32
- [ ] Done
**What are gazetteer features in NER and when should you use them?**

**Answer:**

**Definition:**
Gazetteers are curated lists of known entities (names, locations, organizations) used as lookup features for NER. They provide high-precision recognition for listed entities and are especially valuable for domain-specific or rare entities that models struggle with.

**Core Concept:**
```
Gazetteers = Dictionary lookup for entities

PERSON gazetteer: {"Barack Obama", "Elon Musk", "Taylor Swift", ...}
LOCATION gazetteer: {"New York", "Paris", "Tokyo", ...}
COMPANY gazetteer: {"Apple Inc", "Google", "Microsoft", ...}

Text: "Elon Musk visited New York"
Lookup: "Elon Musk" ∈ PERSON, "New York" ∈ LOCATION
Result: High-confidence entity detection
```

**Gazetteer Features for ML:**

| Feature | Description |
|---------|-------------|
| in_person_gaz | Is word/phrase in person list? |
| in_org_gaz | Is word/phrase in organization list? |
| in_loc_gaz | Is word/phrase in location list? |
| gaz_prefix | Does phrase start with gazetteer entry? |
| gaz_suffix | Does phrase end with gazetteer entry? |

**When to Use Gazetteers:**

| Scenario | Gazetteer Value |
|----------|-----------------|
| Domain-specific entities | Very High (medical terms, legal terms) |
| Rare/unusual names | High (foreign names, brand names) |
| High-precision required | High (no false positives acceptable) |
| Well-curated lists exist | High |
| General newswire NER | Medium (model usually sufficient) |
| Noisy/informal text | Medium-Low (variations not in list) |

**Python Code Example:**
```python
# Pipeline: Text -> Gazetteer lookup -> Feature extraction -> NER

class GazetteerFeatures:
    def __init__(self):
        self.gazetteers = {
            'PER': self._load_gazetteer('persons.txt'),
            'ORG': self._load_gazetteer('organizations.txt'),
            'LOC': self._load_gazetteer('locations.txt'),
            'DRUG': self._load_gazetteer('drug_names.txt')
        }
        # Build prefix tree for efficient matching
        self.max_ngram = 5
    
    def _load_gazetteer(self, filepath):
        """Load gazetteer from file"""
        # In practice: load from file
        sample = {
            'persons.txt': {'elon musk', 'barack obama', 'taylor swift'},
            'organizations.txt': {'apple inc', 'google', 'microsoft', 'openai'},
            'locations.txt': {'new york', 'san francisco', 'paris', 'london'},
            'drug_names.txt': {'aspirin', 'ibuprofen', 'acetaminophen'}
        }
        return sample.get(filepath, set())
    
    def extract_features(self, tokens):
        """
        Extract gazetteer features for each token
        """
        features = []
        n = len(tokens)
        
        for i in range(n):
            token_features = {
                'token': tokens[i],
                'in_PER_gaz': False,
                'in_ORG_gaz': False,
                'in_LOC_gaz': False,
                'in_DRUG_gaz': False,
                'gaz_B': None,  # Beginning of gazetteer match
                'gaz_I': None   # Inside gazetteer match
            }
            
            # Check n-grams starting at this position
            for length in range(1, min(self.max_ngram + 1, n - i + 1)):
                ngram = ' '.join(tokens[i:i+length]).lower()
                
                for gaz_type, gaz_set in self.gazetteers.items():
                    if ngram in gaz_set:
                        key = f'in_{gaz_type}_gaz'
                        token_features[key] = True
                        if length == 1:
                            token_features['gaz_B'] = gaz_type
                        else:
                            # Mark all tokens in match
                            token_features['gaz_B'] = gaz_type
            
            features.append(token_features)
        
        return features

# Usage
gaz = GazetteerFeatures()

text = "Elon Musk announced that OpenAI released a new model"
tokens = text.lower().split()

features = gaz.extract_features(tokens)

for f in features:
    if any(f[k] for k in ['in_PER_gaz', 'in_ORG_gaz', 'in_LOC_gaz']):
        print(f"{f['token']}: {f}")

# Combining with neural NER
def combine_gazetteer_with_model(text, model_preds, gaz_features):
    """
    Combine model predictions with gazetteer matches
    """
    final_preds = []
    for pred, gaz in zip(model_preds, gaz_features):
        # If gazetteer match with high confidence
        if gaz['gaz_B'] and pred == 'O':
            final_preds.append(f'B-{gaz["gaz_B"]}')
        else:
            final_preds.append(pred)
    return final_preds
```

**Best Practices:**
1. Keep gazetteers clean and curated
2. Handle case-insensitivity and normalization
3. Use prefix trees (trie) for efficient multi-word matching
4. Combine gazetteer with model (not replace)
5. Update gazetteers regularly for temporal entities

**Interview Tips:**
- Gazetteers boost precision significantly for known entities
- They don't generalize - unseen entities won't be matched
- Common sources: Wikipedia, Wikidata, domain databases
- Neural + gazetteer hybrid is common in production NER

---

## Question 33
- [ ] Done
**How do you balance precision vs recall in NER for different business use cases?**

**Answer:**

**Definition:**
Precision measures accuracy of predictions (how many detected entities are correct), while recall measures coverage (how many actual entities were found). The optimal balance depends on business cost of false positives vs false negatives.

**Core Trade-off:**

$$\text{Precision} = \frac{TP}{TP + FP}$$ (Of what I found, how much is correct?)

$$\text{Recall} = \frac{TP}{TP + FN}$$ (Of what exists, how much did I find?)

**Business Case Decision Matrix:**

| Use Case | Priority | Why |
|----------|----------|-----|
| Medical diagnosis | High Recall | Missing a disease mention is dangerous |
| Legal contract review | High Recall | Missing a clause could be costly |
| Customer-facing chatbot | High Precision | Wrong answers hurt trust |
| Spam/fraud detection | High Precision | False positives block legitimate users |
| Search/retrieval | High Recall | Users want comprehensive results |
| Data anonymization (PII) | High Recall | Missing PII is a privacy violation |
| Knowledge graph population | Balanced | Both quality and coverage matter |

**How to Adjust Balance:**

| Method | Effect |
|--------|--------|
| Confidence threshold | ↑ threshold = ↑ precision, ↓ recall |
| Training data balance | More examples = better recall for that class |
| Post-processing rules | Filter predictions = ↑ precision |
| Model ensemble | Multiple models = ↑ recall |
| Loss function weighting | Weight FN higher = ↑ recall |

**Python Code Example:**
```python
# Pipeline: Adjust NER threshold based on business requirements

import numpy as np
from sklearn.metrics import precision_recall_curve

class BusinessTunedNER:
    def __init__(self, model, business_case='balanced'):
        self.model = model
        self.business_case = business_case
        self.thresholds = self._set_thresholds()
    
    def _set_thresholds(self):
        """Set confidence thresholds based on business case"""
        if self.business_case == 'high_precision':
            # Higher threshold = fewer but more accurate predictions
            return {'PER': 0.9, 'ORG': 0.85, 'LOC': 0.85}
        elif self.business_case == 'high_recall':
            # Lower threshold = more predictions, some may be wrong
            return {'PER': 0.3, 'ORG': 0.3, 'LOC': 0.3}
        else:  # balanced
            return {'PER': 0.5, 'ORG': 0.5, 'LOC': 0.5}
    
    def predict(self, text):
        # Get model predictions with confidence scores
        raw_predictions = self.model.predict_with_confidence(text)
        
        # Filter based on thresholds
        filtered = []
        for entity in raw_predictions:
            threshold = self.thresholds.get(entity['type'], 0.5)
            if entity['confidence'] >= threshold:
                filtered.append(entity)
        
        return filtered

# Finding optimal threshold using precision-recall curve
def find_optimal_threshold(y_true, y_scores, target='f1'):
    """
    Find threshold that optimizes target metric
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    if target == 'f1':
        # Optimize F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
    elif target == 'precision_at_recall':
        # Maximize precision while maintaining recall >= 0.9
        valid = recalls >= 0.9
        if valid.any():
            best_idx = np.argmax(precisions[valid])
        else:
            best_idx = 0
    elif target == 'recall_at_precision':
        # Maximize recall while maintaining precision >= 0.9
        valid = precisions >= 0.9
        if valid.any():
            best_idx = np.argmax(recalls[valid])
        else:
            best_idx = 0
    
    return thresholds[best_idx]

# Business case examples
def configure_for_business(use_case):
    """
    Configure NER based on business requirements
    """
    configs = {
        'pii_detection': {
            'threshold': 0.3,  # Low threshold - catch all PII
            'post_filter': False,
            'priority': 'recall',
            'acceptable_precision': 0.7
        },
        'customer_chatbot': {
            'threshold': 0.85,  # High threshold - avoid wrong answers
            'post_filter': True,
            'priority': 'precision',
            'acceptable_recall': 0.6
        },
        'medical_ner': {
            'threshold': 0.4,
            'post_filter': False,
            'priority': 'recall',
            'secondary_review': True  # Human review for low-confidence
        }
    }
    return configs.get(use_case, configs['customer_chatbot'])
```

**Scenario-Based Decision:**
```
Scenario: Building NER for medical records to identify drug mentions

Analysis:
- False Negative: Miss drug interaction → patient harm (HIGH COST)
- False Positive: Flag non-drug as drug → extra review (LOW COST)

Decision: Optimize for HIGH RECALL
- Use lower confidence threshold
- Accept more false positives
- Add human review for flagged entities
```

**Interview Tips:**
- Always ask about business context before recommending precision/recall balance
- PII detection: recall is critical (privacy violations are expensive)
- User-facing: precision matters (bad UX from wrong predictions)
- Use F2 score (weights recall) or F0.5 (weights precision) as alternatives to F1

---

## Question 34
- [ ] Done
**What is the Universal Dependencies project and how does it standardize POS tagging across languages?**

**Answer:**

**Definition:**
Universal Dependencies (UD) is a framework for consistent grammatical annotation across 100+ languages. It provides standardized POS tags, morphological features, and dependency relations, enabling cross-lingual NLP research and multilingual model development.

**Core Components:**

| Component | Description | Example |
|-----------|-------------|---------|
| UPOS | Universal POS tags (17 tags) | NOUN, VERB, ADJ, ADV |
| XPOS | Language-specific POS | NN, VB, JJ (Penn Treebank) |
| Features | Morphological features | Number=Plur, Tense=Past |
| DEPREL | Dependency relations | nsubj, dobj, amod |

**Universal POS Tags (17 Categories):**

| Tag | Description | Example |
|-----|-------------|---------|
| NOUN | Noun | cat, freedom |
| VERB | Verb | run, think |
| ADJ | Adjective | big, old |
| ADV | Adverb | quickly, very |
| PROPN | Proper noun | Paris, John |
| PRON | Pronoun | he, she, it |
| DET | Determiner | the, a, this |
| ADP | Adposition | in, on, of |
| NUM | Numeral | one, 2024 |
| CONJ | Conjunction | and, or |
| AUX | Auxiliary | is, was, will |
| PUNCT | Punctuation | . , ! |
| PART | Particle | not, 's |
| INTJ | Interjection | oh, wow |
| SYM | Symbol | $, % |
| X | Other | foreign words |
| SCONJ | Subordinating conj. | if, because |

**Why Universal Dependencies Matters:**

| Benefit | Description |
|---------|-------------|
| Cross-lingual consistency | Same annotation for same concept across languages |
| Model transfer | Train on one language, apply to another |
| Linguistic research | Compare syntax across languages |
| Tool interoperability | Same format for all UD tools |

**Comparison: Penn Treebank vs Universal Dependencies**
```
Penn Treebank (45 tags, English-specific):
NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, ...

Universal Dependencies (17 tags, language-agnostic):
NOUN, VERB, ADJ, ADV, PROPN, ...

Mapping:
NN, NNS → NOUN
NNP, NNPS → PROPN
VB, VBD, VBG, VBN, VBP, VBZ → VERB
```

**Python Code Example:**
```python
# Pipeline: Text -> UD POS tagging -> Universal tags

import spacy

# Load model with UD tagset
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

print("Token | UPOS | XPOS | Features")
print("-" * 50)
for token in doc:
    print(f"{token.text:10} | {token.pos_:6} | {token.tag_:5} | {token.morph}")

# Output:
# The        | DET    | DT    | Definite=Def|PronType=Art
# quick      | ADJ    | JJ    | Degree=Pos
# brown      | ADJ    | JJ    | Degree=Pos
# fox        | NOUN   | NN    | Number=Sing
# jumps      | VERB   | VBZ   | Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
# over       | ADP    | IN    | 
# the        | DET    | DT    | Definite=Def|PronType=Art
# lazy       | ADJ    | JJ    | Degree=Pos
# dog        | NOUN   | NN    | Number=Sing
# .          | PUNCT  | .     | PunctType=Peri

# Cross-lingual UD tagging
from stanza import Pipeline

# German with UD
de_nlp = Pipeline('de', processors='tokenize,pos')
de_doc = de_nlp("Der schnelle braune Fuchs springt.")

for sent in de_doc.sentences:
    for word in sent.words:
        print(f"{word.text:15} | {word.upos:6} | {word.xpos}")

# German uses same UPOS tags as English!
# Der            | DET    | ART
# schnelle       | ADJ    | ADJA
# braune         | ADJ    | ADJA
# Fuchs          | NOUN   | NN
# springt        | VERB   | VVFIN
```

**UD Treebanks Available:**
- 200+ treebanks for 100+ languages
- Consistent format (CoNLL-U)
- Enables training multilingual models

**Interview Tips:**
- UD has 17 universal POS tags vs 45 in Penn Treebank
- spaCy, Stanza, UDPipe all support UD format
- UPOS is universal, XPOS is language-specific (for fine-grained analysis)
- UD enables cross-lingual transfer for syntactic tasks

---

## Question 35
- [ ] Done
**How do you handle POS tagging for code-mixed or transliterated text?**

**Answer:**

**Definition:**
Code-mixed text alternates between two or more languages in a single utterance. Transliterated text writes one language in another's script (e.g., Hindi in Latin). Both challenge POS taggers trained on monolingual, standard text.

**Examples:**

| Type | Example |
|------|---------|
| Code-mixing | "I went to the market aur maine sabzi li" (English + Hindi) |
| Transliteration | "Main bahut khush hoon" (Hindi in Latin script) |
| Mixed both | "That movie was bahut accha" (English + Romanized Hindi) |

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Language identification | Which language is each word? |
| Script mismatch | Standard model expects native script |
| No standard spelling | "bahut" vs "bohot" vs "bhot" |
| Grammar mixing | Which language's rules apply? |
| Limited training data | Code-mixed datasets are scarce |

**Approaches:**

| Approach | Description |
|----------|-------------|
| Language ID first | Identify language per word, route to appropriate tagger |
| Joint model | Train single model on code-mixed data |
| Transliteration norm | Normalize to standard script before tagging |
| Multilingual model | Use mBERT/XLM-R that handles multiple languages |

**Python Code Example:**
```python
# Pipeline: Code-mixed text -> Language ID -> Unified POS tagging

import re

class CodeMixedPOSTagger:
    def __init__(self, en_tagger, hi_tagger, transliterator=None):
        self.en_tagger = en_tagger
        self.hi_tagger = hi_tagger
        self.transliterator = transliterator
        
        # Simple language detection patterns
        self.hi_romanized_words = {
            'aur', 'hai', 'hoon', 'hain', 'kya', 'nahi', 'bahut', 
            'accha', 'maine', 'mera', 'sabzi', 'khush'
        }
    
    def detect_language(self, word):
        """Simple per-word language detection"""
        word_lower = word.lower()
        
        # Check if Devanagari script
        if re.match(r'[\u0900-\u097F]+', word):
            return 'hi'
        
        # Check romanized Hindi vocabulary
        if word_lower in self.hi_romanized_words:
            return 'hi-roman'
        
        # Check if English word
        if re.match(r'^[a-zA-Z]+$', word):
            return 'en'
        
        return 'unknown'
    
    def tag(self, text):
        """Tag code-mixed text"""
        words = text.split()
        results = []
        
        for word in words:
            lang = self.detect_language(word)
            
            if lang == 'en':
                # Use English tagger
                pos = self._get_english_pos(word)
            elif lang == 'hi':
                # Use Hindi tagger
                pos = self._get_hindi_pos(word)
            elif lang == 'hi-roman':
                # Transliterate then tag
                if self.transliterator:
                    hi_word = self.transliterator.romanized_to_devanagari(word)
                    pos = self._get_hindi_pos(hi_word)
                else:
                    pos = 'UNK'
            else:
                pos = 'X'  # Unknown
            
            results.append({
                'word': word,
                'lang': lang,
                'pos': pos
            })
        
        return results
    
    def _get_english_pos(self, word):
        # Simplified - would use actual tagger
        return 'EN-POS'
    
    def _get_hindi_pos(self, word):
        # Simplified - would use actual tagger
        return 'HI-POS'

# Using multilingual model (better approach)
from transformers import AutoTokenizer, AutoModelForTokenClassification

def multilingual_pos_tagger(text):
    """
    Use mBERT for unified code-mixed POS tagging
    """
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    # Decode predictions
    predictions = outputs.logits.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return list(zip(tokens, predictions[0].tolist()))

# Handling transliteration
def normalize_transliterated(text):
    """
    Normalize common transliteration variations
    """
    normalizations = {
        'bohot': 'bahut',
        'bhot': 'bahut',
        'acha': 'accha',
        'achha': 'accha',
        'hai': 'hai',
        'he': 'hai',
        'h': 'hai'
    }
    
    words = text.split()
    normalized = [normalizations.get(w.lower(), w) for w in words]
    return ' '.join(normalized)
```

**Best Practices:**
1. Use multilingual models (mBERT, XLM-R) as base
2. Fine-tune on code-mixed datasets if available
3. Normalize transliterations before processing
4. Build per-word language ID as first step
5. Use Universal Dependencies tags for consistency

**Interview Tips:**
- Code-mixing is common in multilingual societies (India, Singapore)
- GLUECoS benchmark has code-mixed datasets
- mBERT handles code-mixing reasonably well out-of-box
- Spelling normalization is crucial for transliterated text

---

## Question 36
- [ ] Done
**What role does dependency parsing play after POS tagging?**

**Answer:**

**Definition:**
Dependency parsing analyzes grammatical structure by identifying relationships between words, creating a tree where each word depends on another. POS tags inform parsing by constraining which dependency relations are valid (e.g., verbs govern subjects, nouns take adjective modifiers).

**Core Concept:**
```
Sentence: "The quick fox jumps over the fence"

Dependency Tree:
        jumps (ROOT/VERB)
       /     \
     fox      over
    / \         \
  The quick    fence
                 |
                the

Relations:
- "fox" is subject (nsubj) of "jumps"
- "quick" is adjective modifier (amod) of "fox"  
- "fence" is object of preposition (pobj)
```

**How POS Helps Parsing:**

| POS Tag | Likely Dependency Role |
|---------|------------------------|
| VERB | ROOT, head of clause |
| NOUN | Subject, object, complement |
| ADJ | Modifier of noun (amod) |
| ADV | Modifier of verb (advmod) |
| DET | Determiner of noun (det) |
| ADP | Introduces prepositional phrase |

**Common Dependency Relations:**

| Relation | Description | Example |
|----------|-------------|---------|
| nsubj | Nominal subject | "**John** runs" |
| dobj | Direct object | "I saw **him**" |
| amod | Adjectival modifier | "**big** dog" |
| det | Determiner | "**the** book" |
| prep | Prepositional modifier | "ran **to** store" |
| pobj | Object of preposition | "to **store**" |
| ROOT | Root of sentence | Usually main verb |

**Parsing Algorithms:**
1. **Transition-based:** Stack-based, builds tree left-to-right
2. **Graph-based:** Scores all possible edges, finds best tree

**Python Code Example:**
```python
# Pipeline: Text -> POS tagging -> Dependency Parsing -> Tree structure

import spacy

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Display dependency structure
print("Token | POS | Head | Dependency")
print("-" * 50)
for token in doc:
    print(f"{token.text:10} | {token.pos_:5} | {token.head.text:10} | {token.dep_}")

# Output:
# The        | DET   | fox        | det
# quick      | ADJ   | fox        | amod
# brown      | ADJ   | fox        | amod
# fox        | NOUN  | jumps      | nsubj
# jumps      | VERB  | jumps      | ROOT
# over       | ADP   | jumps      | prep
# the        | DET   | dog        | det
# lazy       | ADJ   | dog        | amod
# dog        | NOUN  | over       | pobj
# .          | PUNCT | jumps      | punct

# Practical applications
def extract_subject_verb_object(doc):
    """Extract SVO triples from parsed sentence"""
    triples = []
    
    for token in doc:
        if token.dep_ == 'ROOT':
            verb = token
            subject = None
            obj = None
            
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject = child
                elif child.dep_ in ['dobj', 'pobj']:
                    obj = child
            
            if subject and obj:
                triples.append((subject.text, verb.text, obj.text))
    
    return triples

# Extract relationships
doc = nlp("John loves Mary")
svo = extract_subject_verb_object(doc)
print(svo)  # [('John', 'loves', 'Mary')]

# Use for information extraction
doc = nlp("Apple acquired Beats for $3 billion")
for token in doc:
    if token.dep_ == 'dobj':
        print(f"Acquisition target: {token.text}")
```

**Downstream Applications:**
- Information extraction (who did what to whom)
- Relation extraction (extract facts)
- Semantic role labeling
- Question generation
- Machine translation

**Interview Tips:**
- POS tagging is prerequisite for parsing - provides word categories
- Two main paradigms: transition-based (fast) vs graph-based (accurate)
- Universal Dependencies standardizes both POS and dependency relations
- spaCy, Stanza provide integrated POS+parsing pipelines

---

## Question 37
- [ ] Done
**How do transformer-based models perform joint NER and relation extraction?**

**Answer:**

**Definition:**
Joint NER and relation extraction identifies entities and their relationships simultaneously, rather than as separate pipeline stages. Transformers enable this by producing rich contextual representations that can be used for both entity detection and pairwise relation classification in a single forward pass.

**Pipeline vs Joint Approach:**

| Aspect | Pipeline | Joint |
|--------|----------|-------|
| Architecture | NER → RE (separate models) | Single model, shared encoder |
| Error propagation | NER errors cascade to RE | End-to-end training reduces this |
| Efficiency | Two forward passes | One forward pass |
| Interaction | RE can't help NER | Tasks inform each other |

**Joint Model Architecture:**
```
Input: "Steve Jobs founded Apple in California"

BERT Encoder
     ↓
[Shared Representations]
    ↙         ↘
NER Head      Relation Head
   ↓              ↓
Entities      Relations

Output:
- Entities: [Steve Jobs]_PER, [Apple]_ORG, [California]_LOC
- Relations: (Steve Jobs, founded, Apple), (Apple, located_in, California)
```

**Common Approaches:**

| Approach | Description |
|----------|-------------|
| Table filling | Predict entity tags + relation for all token pairs |
| Span-based | Enumerate spans, classify entities and relations |
| Set prediction | Predict set of (subject, relation, object) tuples |
| Sequence labeling | Special tags encode relations |

**Python Code Example:**
```python
# Pipeline: Text -> BERT -> Joint NER + RE

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class JointNERRE(nn.Module):
    def __init__(self, model_name, num_entity_labels, num_relation_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # NER head (token classification)
        self.ner_classifier = nn.Linear(hidden_size, num_entity_labels)
        
        # Relation head (entity pair classification)
        # Input: concat of two entity representations
        self.relation_classifier = nn.Linear(hidden_size * 2, num_relation_labels)
    
    def forward(self, input_ids, attention_mask, entity_spans=None):
        # Step 1: Encode with BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Step 2: NER predictions (per token)
        ner_logits = self.ner_classifier(sequence_output)
        
        # Step 3: Relation predictions (for entity pairs)
        relation_logits = None
        if entity_spans is not None:
            # Get entity representations (average pool over span)
            entity_reprs = []
            for start, end in entity_spans:
                entity_repr = sequence_output[:, start:end, :].mean(dim=1)
                entity_reprs.append(entity_repr)
            
            # Classify all entity pairs
            relation_logits = []
            for i, repr_i in enumerate(entity_reprs):
                for j, repr_j in enumerate(entity_reprs):
                    if i != j:
                        pair_repr = torch.cat([repr_i, repr_j], dim=-1)
                        rel_logit = self.relation_classifier(pair_repr)
                        relation_logits.append((i, j, rel_logit))
        
        return ner_logits, relation_logits

# Training loop (simplified)
def train_joint_model(model, dataloader, optimizer):
    model.train()
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        relation_labels = batch['relation_labels']
        entity_spans = batch['entity_spans']
        
        # Forward pass
        ner_logits, relation_logits = model(
            input_ids, attention_mask, entity_spans
        )
        
        # Compute joint loss
        ner_loss = nn.CrossEntropyLoss()(
            ner_logits.view(-1, ner_logits.size(-1)),
            ner_labels.view(-1)
        )
        
        rel_loss = 0
        if relation_logits:
            for i, j, logit in relation_logits:
                rel_loss += nn.CrossEntropyLoss()(
                    logit, 
                    relation_labels[i, j]
                )
        
        # Joint loss (weighted)
        loss = ner_loss + 0.5 * rel_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference
def extract_entities_and_relations(text, model, tokenizer):
    # Step 1: Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Step 2: Get NER predictions
    ner_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
    ner_preds = ner_logits.argmax(dim=-1)
    
    # Step 3: Extract entity spans from NER
    entity_spans = extract_spans_from_bio(ner_preds)
    
    # Step 4: Get relation predictions for entity pairs
    _, relation_logits = model(
        inputs['input_ids'], 
        inputs['attention_mask'],
        entity_spans
    )
    
    return entity_spans, relation_logits
```

**State-of-the-Art Models:**
- **SpERT:** Span-based entity and relation transformer
- **PL-Marker:** Packing levitated markers for entity/relation
- **TPLinker:** Token pair linking for joint extraction

**Interview Tips:**
- Joint models outperform pipelines by 2-5% F1
- Key insight: relation extraction can help NER (and vice versa)
- Computational cost: O(n²) for all entity pairs
- Negative relations (no relation) dominate - handle class imbalance

---

## Question 38
- [ ] Done
**What is byte-level BPE and why do models like GPT use it?**

**Answer:**

**Definition:**
Byte-level BPE operates on raw bytes (256 possible values) rather than Unicode characters, enabling the tokenizer to handle any text input including unknown characters, emojis, and any script without [UNK] tokens. GPT-2/3/4 use it for truly open-vocabulary language modeling.

**Standard BPE vs Byte-Level BPE:**

| Aspect | Standard BPE | Byte-level BPE |
|--------|--------------|----------------|
| Base units | Unicode characters | Raw bytes (0-255) |
| [UNK] tokens | Yes, for unknown chars | Never needed |
| Vocabulary | Language-specific | Universal |
| Emoji/symbols | May fail | Always works |
| Base vocab | 1000s of characters | 256 bytes |

**How Byte-Level BPE Works:**
```
Text: "Hello 👋"

Standard: ['Hello', '👋'] or ['Hello', '[UNK]'] if emoji unknown

Byte-level:
1. Convert to bytes: [72, 101, 108, 108, 111, 32, 240, 159, 145, 139]
                     H   e    l    l    o   space  👋 in UTF-8 bytes
2. Apply BPE on bytes: Merge frequent byte pairs
3. Result: ['Hello', ' ', 'Ġ', bytes for 👋...]
```

**Why GPT Uses Byte-Level:**
1. **No UNK tokens:** Can encode literally any input
2. **Multilingual:** Works for all languages without preprocessing
3. **Robust:** Handles typos, code, unusual characters
4. **Simple:** No language-specific tokenization rules

**Byte-Level Encoding Details:**
- UTF-8 encodes characters as 1-4 bytes
- ASCII characters = 1 byte (unchanged)
- Emojis = 4 bytes
- Non-ASCII characters = 2-4 bytes

**Python Code Example:**
```python
# Pipeline: Understand byte-level tokenization

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Regular text
text = "Hello World!"
tokens = tokenizer.tokenize(text)
print(f"Regular: {tokens}")
# ['Hello', 'ĠWorld', '!']
# Note: Ġ represents space (byte 32)

# Text with emojis
text_emoji = "I love 🤖 robots!"
tokens_emoji = tokenizer.tokenize(text_emoji)
print(f"With emoji: {tokens_emoji}")
# Works without [UNK]!

# Non-Latin script (works without special handling)
chinese = "你好世界"
tokens_cn = tokenizer.tokenize(chinese)
print(f"Chinese: {tokens_cn}")
# Each character becomes byte sequences

# Why it works: UTF-8 encoding
def show_byte_encoding(text):
    """Show how text is encoded as bytes"""
    utf8_bytes = text.encode('utf-8')
    print(f"Text: {text}")
    print(f"UTF-8 bytes: {list(utf8_bytes)}")
    print(f"Byte count: {len(utf8_bytes)}")

show_byte_encoding("A")      # [65] - 1 byte
show_byte_encoding("é")      # [195, 169] - 2 bytes
show_byte_encoding("你")     # [228, 189, 160] - 3 bytes
show_byte_encoding("🤖")     # [240, 159, 164, 150] - 4 bytes

# Vocabulary mapping in GPT-2
# GPT-2 maps bytes 0-255 to printable characters to avoid issues
# Byte 32 (space) -> 'Ġ'
# Byte 10 (newline) -> 'Ċ'
print(f"Space byte mapping: {tokenizer.byte_encoder[ord(' ')]}")
# 'Ġ'

# Compare vocab sizes
print(f"GPT-2 vocab size: {tokenizer.vocab_size}")  # 50,257
```

**Trade-offs:**

| Advantage | Disadvantage |
|-----------|--------------|
| Universal - any input works | Multi-byte chars = multiple tokens |
| No preprocessing needed | Chinese/emoji uses more tokens |
| Robust to noise/typos | Slight efficiency loss for non-ASCII |
| Simple implementation | Less interpretable tokens |

**Models Using Byte-Level BPE:**
- GPT-2, GPT-3, GPT-4
- RoBERTa
- BLOOM

**Interview Tips:**
- GPT-2 uses byte-level BPE with 50,257 tokens
- The "Ġ" character you see is the space byte (encoded for visibility)
- Byte-level is why GPT can handle code, emojis, any language
- Trade-off: Unicode characters may become multiple tokens (less efficient)

---

## Question 39
- [ ] Done
**How do you debug tokenization issues affecting downstream model performance?**

**Answer:**

**Definition:**
Tokenization bugs can silently degrade model performance. Debugging involves inspecting token sequences, checking alignment with labels, verifying special token handling, and comparing tokenization between training and inference.

**Common Tokenization Issues:**

| Issue | Symptom | Cause |
|-------|---------|-------|
| Train/inference mismatch | Accuracy drop in production | Different tokenizer versions |
| Label misalignment | Poor NER performance | Subwords not aligned with labels |
| Truncation | Information loss | Sequence too long |
| Special token issues | Weird predictions at boundaries | [CLS], [SEP] handling |
| Vocabulary mismatch | High loss, poor output | Wrong tokenizer for model |

**Debugging Checklist:**

1. **Inspect Tokenization Output**
2. **Check Token-Label Alignment**
3. **Verify Special Tokens**
4. **Compare Train vs Inference**
5. **Check for Truncation**
6. **Validate Vocabulary Match**

**Python Code Example:**
```python
# Pipeline: Debug tokenization step by step

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def debug_tokenization(text, tokenizer):
    """
    Comprehensive tokenization debugging
    """
    print("=" * 60)
    print(f"Input text: {text}")
    print("=" * 60)
    
    # 1. Basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"\n1. Tokens ({len(tokens)}): {tokens}")
    
    # 2. Full encoding with special tokens
    encoded = tokenizer(text, return_tensors='pt')
    all_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    print(f"\n2. With special tokens: {all_tokens}")
    
    # 3. Token IDs
    print(f"\n3. Token IDs: {encoded['input_ids'][0].tolist()}")
    
    # 4. Check for UNK tokens
    unk_id = tokenizer.unk_token_id
    ids = encoded['input_ids'][0].tolist()
    if unk_id in ids:
        print(f"\n⚠️  WARNING: UNK token found at positions: {[i for i, x in enumerate(ids) if x == unk_id]}")
    else:
        print("\n✓ No UNK tokens")
    
    # 5. Offset mapping (for label alignment)
    encoded_with_offsets = tokenizer(text, return_offsets_mapping=True)
    print(f"\n5. Offset mapping: {encoded_with_offsets['offset_mapping']}")
    
    # 6. Check for truncation
    max_length = tokenizer.model_max_length
    if len(tokens) > max_length - 2:  # -2 for [CLS] and [SEP]
        print(f"\n⚠️  WARNING: Text will be truncated! Tokens: {len(tokens)}, Max: {max_length - 2}")
    
    # 7. Decode back
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"\n7. Decoded: {decoded}")
    
    # 8. Check round-trip
    if decoded.replace('[CLS]', '').replace('[SEP]', '').strip() != text.lower():
        print("\n⚠️  Round-trip mismatch!")

# Debug example
debug_tokenization("The café serves açaí bowls", tokenizer)

# Check NER label alignment
def check_ner_alignment(text, entities, tokenizer):
    """
    Verify NER labels align with subword tokens
    entities: list of (start_char, end_char, label)
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    print("Token | Offset | Entity")
    print("-" * 40)
    
    for token, (start, end) in zip(tokens, offsets):
        entity_label = 'O'
        for ent_start, ent_end, label in entities:
            if start >= ent_start and end <= ent_end:
                if start == ent_start:
                    entity_label = f'B-{label}'
                else:
                    entity_label = f'I-{label}'
        print(f"{token:15} | ({start:2}, {end:2}) | {entity_label}")

text = "John Smith works at Apple"
entities = [(0, 10, 'PER'), (20, 25, 'ORG')]
check_ner_alignment(text, entities, tokenizer)

# Verify tokenizer-model compatibility
def verify_tokenizer_model_match(tokenizer, model):
    """
    Verify tokenizer vocabulary matches model embeddings
    """
    vocab_size = len(tokenizer)
    embed_size = model.get_input_embeddings().num_embeddings
    
    if vocab_size != embed_size:
        print(f"⚠️  MISMATCH: Tokenizer vocab={vocab_size}, Model embeddings={embed_size}")
        return False
    print(f"✓ Tokenizer and model vocabulary sizes match: {vocab_size}")
    return True
```

**Common Fixes:**

| Issue | Fix |
|-------|-----|
| Version mismatch | Pin tokenizer version in requirements |
| Label alignment | Use offset_mapping, ignore [CLS]/[SEP] |
| Truncation | Chunk long documents, use stride |
| Special tokens | Mask them during loss computation |
| OOV issues | Use subword tokenizer or add tokens |

**Interview Tips:**
- Always save tokenizer with model to ensure consistency
- Offset mapping is essential for token-level tasks like NER
- Check for UNK tokens - high UNK rate indicates vocabulary issues
- Test with edge cases: emojis, non-ASCII, very long text

---

## Question 40
- [ ] Done
**What are the memory and latency considerations for different tokenization approaches in production?**

**Answer:**

**Definition:**
Production tokenization must balance speed, memory footprint, and accuracy. Considerations include vocabulary loading time, tokenization throughput, batching efficiency, and memory usage for large vocabularies.

**Performance Comparison:**

| Approach | Speed | Memory | Vocab Load Time |
|----------|-------|--------|-----------------|
| Whitespace split | Fastest | Minimal | None |
| Regex-based | Fast | Low | Low |
| WordPiece/BPE | Moderate | Moderate | Moderate |
| SentencePiece | Moderate | Low | Low |
| Full tokenizer pipeline | Slower | Higher | Higher |

**Key Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Latency | Time to tokenize single request | <5ms |
| Throughput | Requests per second | 1000+ |
| Memory | RAM for tokenizer | <100MB |
| Cold start | Time to load tokenizer | <1s |

**Memory Considerations:**

```
Vocabulary memory = vocab_size × avg_token_length × 2 (Unicode)

BERT (30K vocab):   ~1MB for vocabulary
GPT-2 (50K vocab):  ~2MB for vocabulary
Multilingual (250K): ~10MB for vocabulary

Plus:
- Trie/hash table for lookup: 2-3x vocabulary size
- Merge rules (BPE): ~vocab_size entries
- Special token mappings: Minimal
```

**Python Code Example:**
```python
# Pipeline: Benchmark tokenization performance

import time
import sys
from transformers import AutoTokenizer

def benchmark_tokenizer(tokenizer_name, texts, num_iterations=100):
    """
    Benchmark tokenization speed and memory
    """
    # Cold start time
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    load_time = time.time() - start
    
    # Memory estimate
    vocab_size = len(tokenizer)
    
    # Tokenization speed
    start = time.time()
    for _ in range(num_iterations):
        for text in texts:
            _ = tokenizer(text, return_tensors='pt')
    total_time = time.time() - start
    
    avg_latency = (total_time / (num_iterations * len(texts))) * 1000  # ms
    throughput = (num_iterations * len(texts)) / total_time
    
    return {
        'load_time_s': load_time,
        'vocab_size': vocab_size,
        'avg_latency_ms': avg_latency,
        'throughput_per_s': throughput
    }

# Sample texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require careful tokenization.",
    "Natural language processing is a fascinating field."
]

# Benchmark different tokenizers
results = {}
for name in ['bert-base-uncased', 'gpt2', 'xlm-roberta-base']:
    results[name] = benchmark_tokenizer(name, texts)
    print(f"\n{name}:")
    for k, v in results[name].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# Production optimizations
class OptimizedTokenizer:
    def __init__(self, tokenizer_name, use_fast=True, max_length=128):
        # Use fast tokenizer (Rust-based)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            use_fast=use_fast
        )
        self.max_length = max_length
        
        # Pre-compute common patterns
        self._cache = {}
        self._cache_size = 10000
    
    def tokenize(self, text):
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Tokenize
        result = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        # Cache if space available
        if len(self._cache) < self._cache_size:
            self._cache[text] = result
        
        return result
    
    def batch_tokenize(self, texts):
        """Batch tokenization is more efficient"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

# Compare fast vs slow tokenizer
def compare_fast_slow(tokenizer_name, text):
    # Slow (Python-based)
    slow_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    
    # Fast (Rust-based)
    fast_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Benchmark
    iterations = 1000
    
    start = time.time()
    for _ in range(iterations):
        slow_tok.tokenize(text)
    slow_time = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        fast_tok.tokenize(text)
    fast_time = time.time() - start
    
    print(f"Slow tokenizer: {slow_time:.3f}s")
    print(f"Fast tokenizer: {fast_time:.3f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")
```

**Production Best Practices:**

| Practice | Benefit |
|----------|---------|
| Use fast tokenizers | 3-10x speedup |
| Batch tokenization | Better throughput |
| Cache common inputs | Avoid redundant work |
| Limit max_length | Reduce memory/compute |
| Pre-load tokenizer | Avoid cold start |
| Use ONNX export | Faster inference |

**Interview Tips:**
- HuggingFace fast tokenizers are Rust-based, 3-10x faster
- Batching amortizes overhead, always batch when possible
- Vocabulary size impacts embedding memory linearly
- For real-time: cache tokenizer instance, use fast tokenizers

---

## Question 41
- [ ] Done
**How do you handle NER confidence scoring and when to abstain from predictions?**

**Answer:**

**Definition:**
NER confidence scoring quantifies prediction certainty, enabling systems to abstain from low-confidence predictions or flag them for human review. This is crucial for high-stakes applications where wrong predictions are costly.

**Why Confidence Matters:**

| Scenario | Low Confidence Action |
|----------|----------------------|
| Medical NER | Flag for doctor review |
| Legal NER | Abstain, request human verification |
| Chatbot | Ask user for clarification |
| Data pipeline | Log for later review |

**Confidence Scoring Methods:**

| Method | Description | Quality |
|--------|-------------|---------|
| Softmax probability | max(softmax(logits)) | Poor calibration |
| Entropy | Higher entropy = less confident | Better |
| CRF score | Sequence-level probability | Good for BIO |
| MC Dropout | Run multiple passes, measure variance | Best, but slow |
| Temperature scaling | Post-hoc calibration | Improves softmax |

**Mathematical Formulation:**

**Softmax Confidence:**
$$\text{confidence} = \max_i \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Entropy (Uncertainty):**
$$H = -\sum_i p_i \log(p_i)$$

Low entropy = high confidence, High entropy = uncertain

**Python Code Example:**
```python
# Pipeline: NER predictions -> Confidence scores -> Abstention decision

import torch
import torch.nn.functional as F
import numpy as np

class ConfidenceNER:
    def __init__(self, model, tokenizer, abstain_threshold=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.abstain_threshold = abstain_threshold
    
    def predict_with_confidence(self, text):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Predictions and confidence
        max_probs, predictions = probs.max(dim=-1)
        
        # Entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        results = []
        for i, (token, pred, conf, ent) in enumerate(zip(
            tokens, predictions[0], max_probs[0], entropy[0]
        )):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.model.config.id2label[pred.item()]
            
            # Abstention decision
            should_abstain = conf.item() < self.abstain_threshold
            
            results.append({
                'token': token,
                'label': label if not should_abstain else 'ABSTAIN',
                'confidence': conf.item(),
                'entropy': ent.item(),
                'abstain': should_abstain
            })
        
        return results
    
    def mc_dropout_confidence(self, text, n_samples=10):
        """
        Monte Carlo Dropout for better uncertainty estimation
        """
        self.model.train()  # Enable dropout
        
        inputs = self.tokenizer(text, return_tensors='pt')
        
        all_probs = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs)
        
        # Stack and compute statistics
        all_probs = torch.stack(all_probs)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Confidence = mean, Uncertainty = std
        max_probs, predictions = mean_probs.max(dim=-1)
        max_std = std_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
        
        self.model.eval()
        
        return max_probs, max_std  # Low std = confident

# Calibration with temperature scaling
class TemperatureScaler:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def calibrate(self, logits, labels):
        """
        Find optimal temperature using validation set
        """
        # Grid search for best temperature
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            scaled_probs = F.softmax(logits / temp, dim=-1)
            ece = self._compute_ece(scaled_probs, labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
    
    def _compute_ece(self, probs, labels, n_bins=10):
        """Expected Calibration Error"""
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)
        
        ece = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].float().mean()
                ece += prop_in_bin * abs(avg_conf - avg_acc)
        
        return ece
```

**Abstention Strategies:**

| Strategy | Description |
|----------|-------------|
| Threshold-based | Abstain if confidence < threshold |
| Entropy-based | Abstain if entropy > threshold |
| Multi-class | Abstain if top-2 predictions are close |
| Selective prediction | Guarantee precision at cost of coverage |

**Interview Tips:**
- Softmax probabilities are often overconfident - use calibration
- MC Dropout gives better uncertainty but is slower (multiple passes)
- Temperature scaling is simple post-hoc calibration
- For critical applications, always include abstention mechanism

---

## Question 42
- [ ] Done
**What is span-based NER and how does it differ from token classification approaches?**

**Answer:**

**Definition:**
Span-based NER classifies text spans (contiguous sequences of tokens) rather than individual tokens. Instead of BIO tagging each token, it enumerates candidate spans and classifies each as an entity type or "none". This naturally handles nested entities and avoids BIO decoding issues.

**Comparison:**

| Aspect | Token Classification (BIO) | Span-based |
|--------|---------------------------|------------|
| Unit | Individual tokens | Contiguous spans |
| Output | Sequence of BIO tags | Set of (span, type) |
| Nested entities | Cannot handle | Handles naturally |
| Decoding | BIO → entities | Direct span output |
| Complexity | O(n) predictions | O(n²) candidate spans |

**Token Classification Approach:**
```
Input:  "Steve Jobs founded Apple"
Tags:   B-PER I-PER  O       B-ORG
Decode: [Steve Jobs]_PER, [Apple]_ORG
```

**Span-based Approach:**
```
Input: "Steve Jobs founded Apple"

Candidate spans (up to length 3):
(0,0): "Steve"        → classify
(0,1): "Steve Jobs"   → PER ✓
(0,2): "Steve Jobs founded" → None
(1,1): "Jobs"         → classify
(1,2): "Jobs founded" → None
(2,2): "founded"      → None
(3,3): "Apple"        → ORG ✓

Output: {(0,1): PER, (3,3): ORG}
```

**Why Span-based Works Better for Nested NER:**
```
Text: "Bank of America"

Nested entities:
- [Bank of America]_ORG
- [America]_LOC (inside ORG)

Token BIO: Each token gets ONE tag → Can't represent both
Span-based: Multiple spans can be entities → Both detected
```

**Python Code Example:**
```python
# Pipeline: Text -> Enumerate spans -> Classify -> Extract entities

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SpanNER(nn.Module):
    def __init__(self, model_name, num_labels, max_span_len=8):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.max_span_len = max_span_len
        
        # Span representation: start + end + width embedding
        self.width_embedding = nn.Embedding(max_span_len, 50)
        
        # Classifier: 2*hidden + width_embed -> num_labels
        self.classifier = nn.Linear(hidden_size * 2 + 50, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Step 1: Encode tokens
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        batch_size, seq_len, hidden_size = hidden.shape
        
        # Step 2: Enumerate all spans
        span_reprs = []
        span_indices = []
        
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_len, seq_len)):
                # Span representation
                start_repr = hidden[:, start, :]
                end_repr = hidden[:, end, :]
                width = end - start
                width_repr = self.width_embedding(
                    torch.tensor([width]).to(hidden.device)
                ).expand(batch_size, -1)
                
                span_repr = torch.cat([start_repr, end_repr, width_repr], dim=-1)
                span_reprs.append(span_repr)
                span_indices.append((start, end))
        
        # Stack all span representations
        span_reprs = torch.stack(span_reprs, dim=1)  # (batch, num_spans, repr_dim)
        
        # Step 3: Classify each span
        logits = self.classifier(span_reprs)  # (batch, num_spans, num_labels)
        
        return logits, span_indices

def predict_entities(model, tokenizer, text, id2label, threshold=0.5):
    """Extract entities from span predictions"""
    inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping')
    
    with torch.no_grad():
        logits, span_indices = model(**inputs)
        probs = torch.softmax(logits, dim=-1)
    
    entities = []
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    for idx, (start, end) in enumerate(span_indices):
        # Get prediction (argmax)
        pred_label_id = probs[0, idx].argmax().item()
        pred_prob = probs[0, idx, pred_label_id].item()
        
        label = id2label[pred_label_id]
        
        # Filter: not "None" and above threshold
        if label != 'O' and pred_prob > threshold:
            span_text = tokenizer.decode(inputs['input_ids'][0, start:end+1])
            entities.append({
                'text': span_text,
                'start': start,
                'end': end,
                'label': label,
                'confidence': pred_prob
            })
    
    # Handle overlapping predictions (keep highest confidence)
    entities = remove_overlaps(entities)
    
    return entities

def remove_overlaps(entities):
    """Keep only highest confidence for overlapping spans"""
    entities = sorted(entities, key=lambda x: -x['confidence'])
    
    kept = []
    for ent in entities:
        overlaps = False
        for kept_ent in kept:
            if (ent['start'] <= kept_ent['end'] and 
                ent['end'] >= kept_ent['start']):
                overlaps = True
                break
        if not overlaps:
            kept.append(ent)
    
    return kept
```

**Span-based Advantages:**
1. Natural handling of nested/overlapping entities
2. No BIO decoding errors
3. Span length features can help
4. Easier to add span-level features

**Span-based Disadvantages:**
1. O(n²) complexity (vs O(n) for BIO)
2. Most spans are negative (class imbalance)
3. Needs max span length hyperparameter

**Interview Tips:**
- SpERT and SpanBERT are popular span-based NER models
- O(n²) is acceptable for short texts; chunk long documents
- Span-based is preferred for nested NER (ACE, GENIA datasets)
- Use span width embeddings - they help!

---

## Question 43
- [ ] Done
**How do you implement active learning for efficient NER annotation?**

**Answer:**

**Definition:**
Active learning iteratively selects the most informative unlabeled samples for human annotation, reducing labeling costs by 50-80% while maintaining model quality. For NER, it selects sentences where the model is most uncertain or would learn the most from.

**Core Concept:**
```
Traditional: Randomly annotate 1000 sentences
Active Learning: Strategically select 300 most valuable sentences

Both achieve similar F1, but active learning uses 70% less annotation!
```

**Active Learning Loop:**
1. Train initial model on small labeled set
2. Predict on large unlabeled pool
3. Score samples by uncertainty/informativeness
4. Select top-K most informative samples
5. Send to human for annotation
6. Add to training set, retrain
7. Repeat until budget exhausted or performance satisfied

**Uncertainty Sampling Strategies:**

| Strategy | Description | For NER |
|----------|-------------|---------|
| Least Confidence | Lowest max probability | Min(max(P(tag))) per token |
| Margin Sampling | Smallest gap between top-2 | Top1 - Top2 probability |
| Entropy | Highest entropy | Sum of token entropies |
| Sequence Entropy | Sentence-level uncertainty | CRF sequence probability |
| Query-by-Committee | Disagreement between models | Ensemble variance |

**Python Code Example:**
```python
# Pipeline: Train -> Score unlabeled -> Select -> Annotate -> Repeat

import numpy as np
import torch
import torch.nn.functional as F

class ActiveLearningNER:
    def __init__(self, model, tokenizer, strategy='entropy'):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
    
    def compute_uncertainty(self, texts):
        """
        Compute uncertainty scores for each text
        """
        scores = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]  # (seq_len, num_labels)
                
                if self.strategy == 'least_confidence':
                    # Lowest maximum probability
                    max_probs = probs.max(dim=-1).values
                    score = 1 - max_probs.mean().item()
                
                elif self.strategy == 'margin':
                    # Smallest margin between top-2
                    sorted_probs = probs.sort(dim=-1, descending=True).values
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                    score = 1 - margins.mean().item()
                
                elif self.strategy == 'entropy':
                    # Token-level entropy averaged
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    score = entropy.mean().item()
                
                elif self.strategy == 'total_token_entropy':
                    # Sum of entropies (longer = higher, more entities = higher)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    score = entropy.sum().item()
                
                scores.append(score)
        
        return np.array(scores)
    
    def select_samples(self, unlabeled_texts, n_samples):
        """
        Select most informative samples for annotation
        """
        scores = self.compute_uncertainty(unlabeled_texts)
        
        # Select top-n highest uncertainty
        top_indices = np.argsort(scores)[-n_samples:]
        
        selected = [unlabeled_texts[i] for i in top_indices]
        selected_scores = scores[top_indices]
        
        return selected, selected_scores, top_indices

def active_learning_loop(
    model, 
    tokenizer,
    labeled_data,
    unlabeled_data,
    annotation_budget,
    samples_per_round=50
):
    """
    Full active learning loop
    """
    history = []
    
    while annotation_budget > 0:
        # Step 1: Train on current labeled data
        train_model(model, labeled_data)
        
        # Step 2: Evaluate current performance
        f1 = evaluate_model(model, test_data)
        history.append({
            'labeled_size': len(labeled_data),
            'f1': f1
        })
        print(f"Labeled: {len(labeled_data)}, F1: {f1:.3f}")
        
        # Step 3: Select samples for annotation
        al = ActiveLearningNER(model, tokenizer, strategy='entropy')
        samples_to_annotate, _, indices = al.select_samples(
            unlabeled_data, 
            min(samples_per_round, annotation_budget)
        )
        
        # Step 4: Get annotations (simulated or human)
        # In practice: send to annotation platform
        annotations = get_human_annotations(samples_to_annotate)
        
        # Step 5: Add to labeled, remove from unlabeled
        labeled_data.extend(zip(samples_to_annotate, annotations))
        unlabeled_data = [u for i, u in enumerate(unlabeled_data) 
                         if i not in indices]
        
        annotation_budget -= len(samples_to_annotate)
    
    return history

# Diversity sampling (avoid redundant samples)
def diversity_sampling(embeddings, selected_indices, n_samples):
    """
    Select diverse samples using k-means or max-min distance
    """
    from sklearn.cluster import KMeans
    
    # Cluster unlabeled embeddings
    kmeans = KMeans(n_clusters=n_samples)
    kmeans.fit(embeddings)
    
    # Select sample closest to each cluster center
    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest = np.argmin(distances)
        selected.append(closest)
    
    return selected
```

**Hybrid Strategy (Uncertainty + Diversity):**
1. Filter top-K uncertain samples
2. Among K, select diverse subset using clustering
3. Best of both worlds

**Interview Tips:**
- Active learning can reduce annotation needs by 50-80%
- Entropy is simple and effective baseline
- Combine uncertainty + diversity for best results
- Start with 50-100 random samples, then switch to active learning

---

## Question 44
- [ ] Done
**What preprocessing steps are essential before tokenization (normalization, cleaning)?**

**Answer:**

**Definition:**
Preprocessing prepares raw text for tokenization by standardizing formats, removing noise, and normalizing variations. Essential steps include Unicode normalization, case handling, whitespace normalization, and domain-specific cleaning.

**Preprocessing Pipeline:**
```
Raw Text → Unicode Normalize → Clean → Normalize → Tokenize
```

**Essential Steps:**

| Step | Purpose | Example |
|------|---------|---------|
| Unicode normalization | Standardize character encoding | é (composed) → é (decomposed) |
| Whitespace normalization | Consistent spacing | Multiple spaces → single space |
| Case normalization | Optional lowercasing | "Apple" → "apple" |
| Punctuation handling | Standardize quotes, dashes | "smart quotes" → "straight quotes" |
| Number normalization | Optional | "123" → [NUM] |
| URL/email handling | Replace or remove | http://... → [URL] |
| HTML/XML stripping | Remove tags | \<p>text\</p> → text |

**Unicode Normalization Forms:**

| Form | Description | Use Case |
|------|-------------|----------|
| NFC | Composed (single codepoint) | Most common |
| NFD | Decomposed (base + combining) | Text search |
| NFKC | Compatibility composed | Aggressive normalization |
| NFKD | Compatibility decomposed | Aggressive normalization |

**Python Code Example:**
```python
# Pipeline: Raw text -> Full preprocessing -> Clean text

import re
import unicodedata
import html

class TextPreprocessor:
    def __init__(self, 
                 lowercase=False,
                 normalize_unicode=True,
                 remove_urls=True,
                 remove_emails=True,
                 normalize_whitespace=True,
                 strip_html=True,
                 normalize_quotes=True):
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.strip_html = strip_html
        self.normalize_quotes = normalize_quotes
    
    def preprocess(self, text):
        # Step 1: Handle None/empty
        if not text:
            return ""
        
        # Step 2: Decode HTML entities
        text = html.unescape(text)  # &amp; → &
        
        # Step 3: Strip HTML tags
        if self.strip_html:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Step 4: Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Step 5: Replace URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Step 6: Replace emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        
        # Step 7: Normalize quotes and dashes
        if self.normalize_quotes:
            # Smart quotes → straight quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            # Em/en dash → hyphen
            text = text.replace('—', '-').replace('–', '-')
        
        # Step 8: Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Step 9: Lowercase (optional)
        if self.lowercase:
            text = text.lower()
        
        return text

# Usage
preprocessor = TextPreprocessor(lowercase=False)

samples = [
    'Check out https://example.com for more info!',
    'Contact us at info@company.com',
    '<p>Hello <b>World</b></p>',
    '"Smart   quotes"   and — dashes',
    'Café and naïve have accents',
]

for sample in samples:
    cleaned = preprocessor.preprocess(sample)
    print(f"Original: {sample}")
    print(f"Cleaned:  {cleaned}\n")

# Domain-specific preprocessing
def preprocess_medical_text(text):
    """Medical text preprocessing"""
    preprocessor = TextPreprocessor()
    text = preprocessor.preprocess(text)
    
    # Expand common abbreviations
    abbreviations = {
        'pt': 'patient',
        'dx': 'diagnosis',
        'hx': 'history',
        'rx': 'prescription',
        'yo': 'year old'
    }
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
    
    return text

def preprocess_social_media(text):
    """Social media preprocessing"""
    preprocessor = TextPreprocessor()
    text = preprocessor.preprocess(text)
    
    # Normalize elongation
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Keep hashtags but segment them
    def segment_hashtag(m):
        tag = m.group(1)
        return '#' + re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
    text = re.sub(r'#(\w+)', segment_hashtag, text)
    
    return text
```

**When NOT to Preprocess:**

| Scenario | Reason |
|----------|--------|
| Using BERT-like models | They handle raw text well |
| Case matters (NER) | "apple" (fruit) vs "Apple" (company) |
| Sentiment analysis | Punctuation carries sentiment (!!!) |
| Code analysis | Syntax matters |

**Interview Tips:**
- Modern transformers need less preprocessing than traditional methods
- Always do Unicode normalization (NFKC is safe default)
- Don't lowercase for NER - case is a strong feature
- Domain-specific preprocessing can significantly help (medical, legal, social)

---

## Question 45
- [ ] Done
**How do you handle special tokens ([CLS], [SEP], [PAD]) in transformer tokenizers?**

**Answer:**

**Definition:**
Special tokens are reserved tokens with specific meanings: [CLS] for classification pooling, [SEP] for sentence separation, [PAD] for batch padding, and [MASK] for masked language modeling. Proper handling is crucial for correct model behavior.

**Special Tokens Overview:**

| Token | BERT ID | Purpose | When Added |
|-------|---------|---------|------------|
| [CLS] | 101 | Classification; start of sequence | Automatically |
| [SEP] | 102 | Separator; end of sequence/between sentences | Automatically |
| [PAD] | 0 | Padding for batch uniformity | During batching |
| [MASK] | 103 | Masked token for pre-training | MLM training |
| [UNK] | 100 | Unknown token | Rare in subword tokenizers |

**Token Sequence Structure:**
```
Single sentence:  [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD]
Two sentences:    [CLS] sent1_tokens [SEP] sent2_tokens [SEP] [PAD]
```

**Attention Mask:**
```
Tokens:         [CLS] Hello World [SEP] [PAD] [PAD]
Attention mask:   1     1    1     1     0     0
                 ↑ Real tokens     ↑ Padding (ignored)
```

**Token Type IDs (Segment IDs):**
```
[CLS] Sentence A [SEP] Sentence B [SEP]
  0       0        0       1        1
  └── Segment 0 ──┘  └── Segment 1 ─┘
```

**Python Code Example:**
```python
# Pipeline: Understanding and handling special tokens

from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1. Inspect special tokens
print("Special tokens:")
print(f"  [CLS]: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"  [SEP]: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"  [PAD]: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  [MASK]: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
print(f"  [UNK]: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

# 2. Single sentence encoding
text = "Hello World"
encoded = tokenizer(text, return_tensors='pt')

tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
print(f"\nSingle sentence: {tokens}")
# ['[CLS]', 'hello', 'world', '[SEP]']

# 3. Sentence pair encoding
sent1 = "What is AI?"
sent2 = "AI stands for Artificial Intelligence"
encoded_pair = tokenizer(sent1, sent2, return_tensors='pt')

tokens_pair = tokenizer.convert_ids_to_tokens(encoded_pair['input_ids'][0])
print(f"\nSentence pair: {tokens_pair}")
# ['[CLS]', 'what', 'is', 'ai', '?', '[SEP]', 'ai', 'stands', 'for', ...]

print(f"Token type IDs: {encoded_pair['token_type_ids'][0].tolist()}")
# [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 4. Batch with padding
texts = ["Short text", "This is a longer piece of text", "Hi"]
batch = tokenizer(texts, padding=True, return_tensors='pt')

print(f"\nBatch shape: {batch['input_ids'].shape}")
for i, text in enumerate(texts):
    tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
    mask = batch['attention_mask'][i].tolist()
    print(f"  {text}: {tokens}")
    print(f"  Attention mask: {mask}")

# 5. Handling special tokens in NER
def get_ner_labels_with_special_tokens(tokens, bio_labels, tokenizer):
    """
    Add special token labels for NER
    """
    # Special tokens get -100 (ignored in loss)
    full_labels = [-100]  # [CLS]
    full_labels.extend([label2id[l] for l in bio_labels])
    full_labels.append(-100)  # [SEP]
    
    return full_labels

# 6. Extracting [CLS] for classification
def get_cls_embedding(model, input_ids, attention_mask):
    """
    Get [CLS] token embedding for classification
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
    
    # [CLS] is always at position 0
    cls_embedding = last_hidden[:, 0, :]  # (batch, hidden)
    
    return cls_embedding

# 7. Masking special tokens in attention for some tasks
def create_special_token_mask(input_ids, tokenizer):
    """
    Create mask that's 0 for special tokens, 1 for real tokens
    """
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id
    }
    
    mask = torch.ones_like(input_ids)
    for special_id in special_ids:
        mask[input_ids == special_id] = 0
    
    return mask
```

**Key Points for Different Tasks:**

| Task | [CLS] Usage | [SEP] Usage | Padding |
|------|-------------|-------------|---------|
| Classification | Pool for prediction | End marker | Mask with attention_mask |
| NER | Ignore (label=-100) | Ignore | Ignore |
| QA | Not used directly | Separates question/context | Mask |
| MLM | Not masked | Not masked | Mask |

**Interview Tips:**
- [CLS] embedding is used for sentence-level classification
- Always use attention_mask to handle [PAD] tokens
- NER: label special tokens with -100 (PyTorch ignores in loss)
- Token type IDs distinguish sentence A from sentence B
- [SEP] appears between sentences AND at the end

---

## Question 46
- [ ] Done
**What is the difference between greedy and optimal tokenization algorithms?**

**Answer:**

**Definition:**
Greedy tokenization picks the locally best token at each step (longest match), while optimal tokenization finds the globally best segmentation considering all possible splits. BPE uses greedy merging; Unigram LM uses optimal (Viterbi) decoding.

**Comparison:**

| Aspect | Greedy | Optimal |
|--------|--------|---------|
| Approach | Longest match at each step | Best overall segmentation |
| Speed | Faster O(n) | Slower O(n²) |
| Consistency | Deterministic | Deterministic |
| Quality | May miss better splits | Globally optimal |
| Examples | BPE, WordPiece | Unigram LM |

**Greedy Example (BPE):**
```
Word: "lowest"
Vocabulary: ["low", "er", "est", "low", "e", "s", "t"]

Greedy (left-to-right, longest match):
Step 1: "low" matches → ["low"]
Step 2: "est" matches → ["low", "est"]
Result: ["low", "est"]

Alternative (not chosen by greedy):
["lo", "we", "st"] - greedy doesn't explore this
```

**Optimal Example (Unigram):**
```
Word: "lowest"
Vocabulary with probabilities:
  P("low") = 0.1, P("est") = 0.05
  P("lo") = 0.08, P("west") = 0.02
  P("l") = 0.3, P("o") = 0.2, ...

Optimal: Find segmentation that maximizes:
  ∏ P(token) for all tokens

Uses Viterbi algorithm to find global optimum:
  P("low" + "est") = 0.1 × 0.05 = 0.005
  P("lo" + "west") = 0.08 × 0.02 = 0.0016
  
Best: ["low", "est"] (happens to match greedy here)
```

**Viterbi Algorithm for Optimal Tokenization:**
```
1. Create lattice of all possible tokens at each position
2. Compute cumulative probability for each path
3. Backtrack to find highest probability segmentation
```

**Python Code Example:**
```python
# Pipeline: Compare greedy vs optimal tokenization

import math

class GreedyTokenizer:
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.max_word_len = max(len(w) for w in vocab)
    
    def tokenize(self, text):
        """Greedy left-to-right longest match"""
        tokens = []
        i = 0
        
        while i < len(text):
            # Find longest matching token
            match = None
            for length in range(min(self.max_word_len, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in self.vocab:
                    match = candidate
                    break
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                # Character fallback
                tokens.append(text[i])
                i += 1
        
        return tokens

class UnigramTokenizer:
    def __init__(self, vocab_with_probs):
        self.vocab = vocab_with_probs  # {token: log_prob}
        self.max_word_len = max(len(w) for w in vocab_with_probs.keys())
    
    def tokenize(self, text):
        """Optimal tokenization using Viterbi"""
        n = len(text)
        
        # best_score[i] = best log prob to tokenize text[0:i]
        best_score = [-float('inf')] * (n + 1)
        best_score[0] = 0
        
        # backpointer[i] = position where best tokenization to i starts last token
        backpointer = [0] * (n + 1)
        
        # Dynamic programming
        for i in range(1, n + 1):
            for length in range(1, min(self.max_word_len + 1, i + 1)):
                start = i - length
                token = text[start:i]
                
                if token in self.vocab:
                    score = best_score[start] + self.vocab[token]
                    if score > best_score[i]:
                        best_score[i] = score
                        backpointer[i] = start
        
        # Backtrack to get tokens
        tokens = []
        i = n
        while i > 0:
            start = backpointer[i]
            tokens.append(text[start:i])
            i = start
        
        tokens.reverse()
        return tokens

# Compare
vocab = ['low', 'lower', 'lowest', 'est', 'er', 'l', 'o', 'w', 'e', 's', 't']
greedy = GreedyTokenizer(vocab)

vocab_probs = {
    'low': math.log(0.1), 'lower': math.log(0.05), 'lowest': math.log(0.03),
    'est': math.log(0.08), 'er': math.log(0.1),
    'l': math.log(0.2), 'o': math.log(0.15), 'w': math.log(0.1),
    'e': math.log(0.2), 's': math.log(0.1), 't': math.log(0.1)
}
optimal = UnigramTokenizer(vocab_probs)

word = "lowest"
print(f"Greedy:  {greedy.tokenize(word)}")
print(f"Optimal: {optimal.tokenize(word)}")
```

**Real-World Tokenizers:**

| Tokenizer | Algorithm | Used By |
|-----------|-----------|---------|
| BPE | Greedy merge | GPT-2, RoBERTa |
| WordPiece | Greedy (likelihood-based) | BERT |
| Unigram | Optimal (Viterbi) | T5, XLNet, ALBERT |
| SentencePiece | Either BPE or Unigram | Many models |

**Interview Tips:**
- BPE/WordPiece are greedy - fast but may not be optimal
- Unigram uses Viterbi - optimal but slightly slower
- In practice, difference in downstream performance is minimal
- Unigram allows multiple valid segmentations (subword regularization)

---

## Question 47
- [ ] Done
**How do you implement privacy-preserving NER for sensitive documents (PII detection)?**

**Answer:**

**Definition:**
Privacy-preserving NER identifies and protects Personally Identifiable Information (PII) like names, SSNs, credit cards, and addresses. It requires high recall (missing PII is costly), pattern matching for structured data, and secure handling of detected entities.

**PII Entity Types:**

| Type | Examples | Detection Method |
|------|----------|------------------|
| Names | John Smith | NER model |
| SSN | 123-45-6789 | Regex pattern |
| Credit Card | 4111-1111-1111-1111 | Regex + Luhn check |
| Phone | (555) 123-4567 | Regex pattern |
| Email | john@example.com | Regex pattern |
| Address | 123 Main St, City, ST 12345 | NER + patterns |
| DOB | 01/15/1990 | Date pattern + context |
| Medical IDs | MRN: 12345 | Domain patterns |

**PII Detection Pipeline:**
```
Document → Pattern Matching → NER Model → Merge Results → Redact/Mask
              ↓                   ↓
         Structured PII      Unstructured PII
         (SSN, CC, Phone)    (Names, Addresses)
```

**Python Code Example:**
```python
# Pipeline: Document -> Detect PII -> Mask/Redact

import re
from transformers import pipeline

class PIIDetector:
    def __init__(self, ner_model="dslim/bert-base-NER"):
        self.ner_pipeline = pipeline("ner", model=ner_model, grouped_entities=True)
        
        # Regex patterns for structured PII
        self.patterns = {
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'PHONE': r'\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'DATE_OF_BIRTH': r'\b(?:DOB|Date of Birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        }
    
    def detect_pii(self, text):
        """Detect all PII in text"""
        pii_entities = []
        
        # 1. Pattern-based detection (structured PII)
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pii_entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': pii_type,
                    'source': 'regex',
                    'confidence': 1.0  # Pattern matches are certain
                })
        
        # 2. NER-based detection (names, organizations, locations)
        ner_results = self.ner_pipeline(text)
        for entity in ner_results:
            if entity['entity_group'] in ['PER', 'PERSON']:
                pii_entities.append({
                    'text': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'type': 'PERSON_NAME',
                    'source': 'ner',
                    'confidence': entity['score']
                })
        
        # 3. Additional validation
        pii_entities = self._validate_entities(pii_entities)
        
        return pii_entities
    
    def _validate_entities(self, entities):
        """Validate detected PII (e.g., Luhn check for credit cards)"""
        validated = []
        for entity in entities:
            if entity['type'] == 'CREDIT_CARD':
                # Luhn algorithm validation
                if self._luhn_check(entity['text']):
                    validated.append(entity)
            else:
                validated.append(entity)
        return validated
    
    def _luhn_check(self, card_number):
        """Validate credit card number using Luhn algorithm"""
        digits = [int(d) for d in card_number if d.isdigit()]
        if len(digits) < 13:
            return False
        
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0
    
    def redact(self, text, entities, method='mask'):
        """Redact PII from text"""
        # Sort by position (reverse to maintain indices)
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        redacted = text
        for entity in entities:
            if method == 'mask':
                replacement = f"[{entity['type']}]"
            elif method == 'hash':
                import hashlib
                h = hashlib.sha256(entity['text'].encode()).hexdigest()[:8]
                replacement = f"[{entity['type']}:{h}]"
            elif method == 'generalize':
                replacement = self._generalize(entity)
            
            redacted = redacted[:entity['start']] + replacement + redacted[entity['end']:]
        
        return redacted
    
    def _generalize(self, entity):
        """Generalize PII instead of removing"""
        if entity['type'] == 'PERSON_NAME':
            return "[PERSON]"
        elif entity['type'] == 'EMAIL':
            # Keep domain, mask username
            parts = entity['text'].split('@')
            return f"***@{parts[1]}"
        elif entity['type'] == 'PHONE':
            return "***-***-" + entity['text'][-4:]
        return f"[{entity['type']}]"

# Usage
detector = PIIDetector()

text = """
Dear John Smith,
Your SSN 123-45-6789 has been verified.
Please contact us at john.smith@email.com or (555) 123-4567.
Credit card ending in 4111-1111-1111-1111 was charged.
"""

pii = detector.detect_pii(text)
print("Detected PII:")
for p in pii:
    print(f"  {p['type']}: {p['text']} (confidence: {p['confidence']:.2f})")

redacted = detector.redact(text, pii, method='mask')
print(f"\nRedacted:\n{redacted}")
```

**Privacy Considerations:**

| Consideration | Implementation |
|---------------|----------------|
| High recall | Lower confidence thresholds |
| Audit logging | Log detections (not values) |
| Secure storage | Don't store raw PII |
| Re-identification | Consider context attacks |
| Differential privacy | Add noise to aggregates |

**Interview Tips:**
- PII detection requires HIGH RECALL - missing PII is a privacy violation
- Combine regex (structured) + NER (unstructured) for best coverage
- Credit card validation with Luhn algorithm reduces false positives
- HIPAA, GDPR, CCPA have specific PII requirements
- Tools: Microsoft Presidio, AWS Comprehend PII, spaCy with custom NER

---

## Question 48
- [ ] Done
**What are the best practices for fine-tuning pre-trained models on custom NER datasets?**

**Answer:**

**Definition:**
Fine-tuning adapts pre-trained models (BERT, RoBERTa) to custom NER tasks. Best practices include proper data formatting, learning rate selection, handling class imbalance, using appropriate loss functions, and preventing overfitting.

**Fine-tuning Pipeline:**
```
Custom NER Data → Format to BIO → Align with subwords → 
Configure training → Fine-tune → Evaluate → Deploy
```

**Best Practices Checklist:**

| Category | Practice |
|----------|----------|
| Data | Convert to BIO/BIOES format |
| Data | Align labels with subwords |
| Data | Use train/val/test split |
| Learning Rate | Use 2e-5 to 5e-5 (BERT default) |
| Epochs | 3-5 epochs (rarely more) |
| Batch Size | 16-32 (increase with gradient accumulation) |
| Scheduler | Linear warmup + decay |
| Loss | CrossEntropy with class weights for imbalance |
| Regularization | Dropout (already in BERT) |
| Evaluation | Entity-level F1 (not token-level) |

**Python Code Example:**
```python
# Pipeline: Custom NER data -> Fine-tune BERT -> Evaluate

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

# Step 1: Prepare data in correct format
def prepare_ner_data(examples, tokenizer, label2id):
    """
    Tokenize and align labels with subwords
    """
    tokenized = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of word gets the label
                label_ids.append(label2id[label[word_idx]])
            else:
                # Subsequent tokens of same word
                # Option 1: Same label (common)
                # Option 2: -100 (ignore)
                label_ids.append(label2id[label[word_idx]])
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized['labels'] = labels
    return tokenized

# Step 2: Define metrics
def compute_metrics(eval_preds, id2label):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label strings (ignore -100)
    true_labels = []
    pred_labels = []
    
    for pred, label in zip(predictions, labels):
        true_seq = []
        pred_seq = []
        for p, l in zip(pred, label):
            if l != -100:
                true_seq.append(id2label[l])
                pred_seq.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels)
    }

# Step 3: Configure training
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label mappings
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training arguments (best practices)
training_args = TrainingArguments(
    output_dir="./ner-model",
    
    # Learning rate: 2e-5 to 5e-5 for BERT
    learning_rate=3e-5,
    
    # Epochs: 3-5 is usually enough
    num_train_epochs=4,
    
    # Batch size (adjust based on GPU memory)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    
    # Gradient accumulation for larger effective batch
    gradient_accumulation_steps=2,
    
    # Learning rate schedule
    warmup_ratio=0.1,  # 10% warmup
    lr_scheduler_type="linear",
    
    # Regularization
    weight_decay=0.01,
    
    # Evaluation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # Logging
    logging_steps=100,
    
    # Mixed precision (faster training)
    fp16=True,
)

# Data collator handles padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 4: Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, id2label)
)

trainer.train()

# Step 5: Evaluate on test set
results = trainer.evaluate(test_dataset)
print(f"Test F1: {results['eval_f1']:.4f}")
```

**Common Mistakes to Avoid:**

| Mistake | Fix |
|---------|-----|
| Wrong label alignment | Use word_ids() for subword alignment |
| Too high learning rate | Use 2e-5 to 5e-5 |
| Too many epochs | 3-5 epochs, use early stopping |
| Token-level evaluation | Use seqeval for entity-level F1 |
| Ignoring class imbalance | Weight loss or oversample rare entities |
| Not freezing layers | For small data, freeze lower layers |

**Interview Tips:**
- Always align labels with subwords using word_ids()
- seqeval library for proper entity-level evaluation
- Learning rate 3e-5 is safe default for BERT fine-tuning
- Add CRF layer for 1-2% F1 improvement
- Use fp16 for faster training on modern GPUs

---

## Question 49
- [ ] Done
**How do you handle NER for hierarchical or multi-level entity types?**

**Answer:**

**Definition:**
Hierarchical NER handles entities with type hierarchies (e.g., PERSON → ATHLETE → FOOTBALLER). Multi-level tagging assigns both coarse and fine-grained types. Approaches include hierarchical classification, multi-task learning, and type embeddings.

**Hierarchy Example:**
```
ENTITY
├── PERSON
│   ├── POLITICIAN
│   ├── ATHLETE
│   │   ├── FOOTBALLER
│   │   └── TENNIS_PLAYER
│   └── ARTIST
├── ORGANIZATION
│   ├── COMPANY
│   ├── GOVERNMENT
│   └── SPORTS_TEAM
└── LOCATION
    ├── CITY
    ├── COUNTRY
    └── FACILITY
```

**Why Hierarchical NER:**
- More informative predictions (not just "PERSON" but "FOOTBALLER")
- Consistency constraints (FOOTBALLER implies ATHLETE implies PERSON)
- Better generalization (share knowledge across related types)
- Business needs (fine-grained vs coarse-grained requirements)

**Approaches:**

| Approach | Description |
|----------|-------------|
| Flat multi-class | All leaf types as separate classes |
| Hierarchical loss | Penalize less for hierarchically related errors |
| Multi-task | Predict at each level separately |
| Type embeddings | Encode hierarchy in type vectors |
| Constraint decoding | Enforce valid parent-child relations |

**Python Code Example:**
```python
# Pipeline: Text -> Hierarchical NER -> Multi-level entity types

import torch
import torch.nn as nn
from transformers import BertModel

class HierarchicalNER(nn.Module):
    def __init__(self, model_name, type_hierarchy):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Store hierarchy
        self.hierarchy = type_hierarchy
        self.levels = self._get_levels()
        
        # Separate classifier for each level
        self.level_classifiers = nn.ModuleDict()
        for level, types in self.levels.items():
            self.level_classifiers[level] = nn.Linear(hidden_size, len(types))
    
    def _get_levels(self):
        """Extract types at each hierarchy level"""
        return {
            'level_0': ['O', 'PERSON', 'ORGANIZATION', 'LOCATION'],
            'level_1': ['O', 'POLITICIAN', 'ATHLETE', 'ARTIST', 
                       'COMPANY', 'GOVERNMENT', 'CITY', 'COUNTRY'],
            'level_2': ['O', 'FOOTBALLER', 'TENNIS_PLAYER', 
                       'SPORTS_TEAM', 'FACILITY']
        }
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Predict at each level
        level_logits = {}
        for level, classifier in self.level_classifiers.items():
            level_logits[level] = classifier(sequence_output)
        
        return level_logits

def hierarchical_loss(level_logits, level_labels, hierarchy, alpha=0.5):
    """
    Loss that considers hierarchy:
    - Full penalty for wrong prediction
    - Reduced penalty if prediction is ancestor/descendant of true type
    """
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    
    for level, logits in level_logits.items():
        labels = level_labels[level]
        
        # Standard CE loss at each level
        loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss
    
    return total_loss

# Simpler approach: Multi-label with type embeddings
class TypeEmbeddingNER(nn.Module):
    def __init__(self, model_name, num_types, hierarchy_adj):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Type embeddings that capture hierarchy
        self.type_embeddings = nn.Embedding(num_types, 128)
        
        # Initialize type embeddings based on hierarchy
        self._init_type_embeddings(hierarchy_adj)
        
        # Bilinear scoring
        self.scorer = nn.Bilinear(hidden_size, 128, 1)
    
    def _init_type_embeddings(self, adj):
        """Initialize similar embeddings for related types"""
        # Graph-based initialization
        pass
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        token_reprs = outputs.last_hidden_state  # (batch, seq, hidden)
        
        # Score each token against each type
        batch_size, seq_len, _ = token_reprs.shape
        num_types = self.type_embeddings.num_embeddings
        
        all_type_embs = self.type_embeddings.weight  # (num_types, type_dim)
        
        # Compute scores
        scores = []
        for t in range(num_types):
            type_emb = all_type_embs[t].unsqueeze(0).unsqueeze(0)
            type_emb = type_emb.expand(batch_size, seq_len, -1)
            score = self.scorer(token_reprs, type_emb).squeeze(-1)
            scores.append(score)
        
        scores = torch.stack(scores, dim=-1)  # (batch, seq, num_types)
        
        return scores

# Inference with hierarchy constraints
def constrained_decode(predictions, hierarchy):
    """
    Ensure predictions are consistent with hierarchy
    e.g., if FOOTBALLER predicted, parent types should also be true
    """
    # Get parent types
    parent_map = {
        'FOOTBALLER': ['ATHLETE', 'PERSON'],
        'TENNIS_PLAYER': ['ATHLETE', 'PERSON'],
        'ATHLETE': ['PERSON'],
        'COMPANY': ['ORGANIZATION'],
        # ... etc
    }
    
    # Propagate to ancestors
    consistent = predictions.copy()
    for pred in predictions:
        if pred in parent_map:
            for parent in parent_map[pred]:
                if parent not in consistent:
                    consistent.add(parent)
    
    return consistent
```

**Evaluation for Hierarchical NER:**
- Exact match: Full type path must match
- Partial match: Credit for correct ancestors
- Level-wise F1: Evaluate each level separately

**Interview Tips:**
- Flat classification ignores hierarchy - wastes structure
- Multi-task learning at each level is common approach
- Type embeddings can encode semantic similarity
- Constraint decoding ensures consistency at inference
- Datasets: OntoNotes (18 types), FIGER (113 types), TypeNet (~1000 types)

---

## Question 50
- [ ] Done
**What metrics and techniques help evaluate tokenization quality?**

**Answer:**

**Definition:**
Tokenization quality affects downstream model performance. Evaluation metrics include vocabulary coverage, fertility (tokens per word), token boundary quality, and downstream task performance. Good tokenization balances vocabulary size, sequence length, and semantic preservation.

**Key Metrics:**

| Metric | Description | Good Value |
|--------|-------------|------------|
| Vocabulary coverage | % words not going to [UNK] | >99% |
| Fertility | Avg tokens per word | 1.2-1.5 |
| Sequence length ratio | Output length / input words | ~1.2-1.5 |
| Subword quality | Meaningful subword splits | Subjective |
| OOV rate | % unknown tokens | <1% |
| Compression ratio | Bits per character | Lower = better |

**Fertility Definition:**
$$\text{Fertility} = \frac{\text{Number of tokens}}{\text{Number of words}}$$

High fertility = more fragmentation = longer sequences

**Python Code Example:**
```python
# Pipeline: Evaluate tokenization quality

from transformers import AutoTokenizer
from collections import Counter
import numpy as np

class TokenizationEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate(self, texts):
        """Comprehensive tokenization evaluation"""
        results = {
            'fertility': [],
            'unk_rate': [],
            'seq_length_ratio': [],
            'vocab_coverage': [],
            'subword_counts': Counter()
        }
        
        unk_id = self.tokenizer.unk_token_id
        
        for text in texts:
            words = text.split()
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Fertility: tokens per word
            fertility = len(tokens) / max(len(words), 1)
            results['fertility'].append(fertility)
            
            # UNK rate
            if unk_id:
                unk_count = token_ids.count(unk_id)
                unk_rate = unk_count / max(len(token_ids), 1)
                results['unk_rate'].append(unk_rate)
            
            # Sequence length ratio
            ratio = len(tokens) / max(len(words), 1)
            results['seq_length_ratio'].append(ratio)
            
            # Count subword patterns
            for token in tokens:
                if token.startswith('##') or token.startswith('Ġ'):
                    results['subword_counts']['continuation'] += 1
                else:
                    results['subword_counts']['word_start'] += 1
        
        # Aggregate
        summary = {
            'avg_fertility': np.mean(results['fertility']),
            'std_fertility': np.std(results['fertility']),
            'avg_unk_rate': np.mean(results['unk_rate']) if results['unk_rate'] else 0,
            'avg_seq_ratio': np.mean(results['seq_length_ratio']),
            'subword_ratio': results['subword_counts']['continuation'] / 
                           max(sum(results['subword_counts'].values()), 1)
        }
        
        return summary
    
    def compare_tokenizers(self, texts, tokenizer_names):
        """Compare multiple tokenizers"""
        comparison = {}
        
        for name in tokenizer_names:
            tok = AutoTokenizer.from_pretrained(name)
            evaluator = TokenizationEvaluator(tok)
            comparison[name] = evaluator.evaluate(texts)
        
        return comparison

def evaluate_subword_quality(tokenizer, words):
    """
    Check if tokenization produces meaningful subwords
    """
    quality_checks = {
        'morphological': 0,  # Splits at morpheme boundaries
        'arbitrary': 0       # Random-looking splits
    }
    
    # Known good splits
    good_patterns = ['ing', 'tion', 'ness', 'able', 'ment', 'er', 'ed', 'ly']
    
    for word in words:
        tokens = tokenizer.tokenize(word)
        
        if len(tokens) > 1:
            # Check if splits are meaningful
            for token in tokens[1:]:  # Skip first token
                clean = token.replace('##', '').replace('Ġ', '')
                if clean in good_patterns:
                    quality_checks['morphological'] += 1
                else:
                    quality_checks['arbitrary'] += 1
    
    total = sum(quality_checks.values())
    if total > 0:
        morphological_ratio = quality_checks['morphological'] / total
        return {'morphological_ratio': morphological_ratio}
    return {'morphological_ratio': None}

# Usage
tokenizers = ['bert-base-uncased', 'gpt2', 'xlm-roberta-base']

sample_texts = [
    "The internationalization of artificial intelligence continues.",
    "Unprecedented breakthroughs in biotechnology research.",
    "Cryptocurrency markets experienced unprecedented volatility."
]

for tok_name in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    evaluator = TokenizationEvaluator(tokenizer)
    
    results = evaluator.evaluate(sample_texts)
    print(f"\n{tok_name}:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Avg fertility: {results['avg_fertility']:.2f}")
    print(f"  UNK rate: {results['avg_unk_rate']:.4f}")
    print(f"  Subword ratio: {results['subword_ratio']:.2f}")

# Check downstream impact
def evaluate_downstream_impact(tokenizer1, tokenizer2, model, test_data):
    """
    Compare tokenization impact on downstream task
    """
    # Tokenize with each tokenizer
    # Train/evaluate model
    # Compare performance
    pass
```

**What Makes Good Tokenization:**

| Good Sign | Bad Sign |
|-----------|----------|
| Low fertility (1.2-1.5) | High fertility (>2.0) |
| Near-zero UNK rate | High UNK rate |
| Morphological splits | Arbitrary splits |
| Consistent behavior | Inconsistent |
| Works cross-lingually | Language-specific issues |

**Interview Tips:**
- Fertility is key metric - lower is better (fewer tokens = faster inference)
- Zero UNK rate with subword tokenizers
- For multilingual, check fertility across languages
- Best evaluation: downstream task performance
- Common issue: tokenizer-model mismatch (different vocab)

---


---

# --- Tokenization/Lemmatization/Stemming Questions (from 08_nlp/01_tokenization_lemmatization_stemming) ---

# Tokenization/Lemmatization/Stemming - Theory Questions

## Question 1
- [ ] Done
**How do you choose between different tokenization strategies (word-level, subword, character-level) for specific NLP tasks?**

**Answer:**

Tokenization strategy selection depends on the task's vocabulary requirements, language characteristics, and computational constraints. **Word-level** tokenization splits on whitespace/punctuation and works well when the vocabulary is closed and well-defined. **Subword** methods (BPE, WordPiece, Unigram) split rare words into frequent subunits, balancing vocabulary size with coverage. **Character-level** tokenization uses individual characters as tokens, yielding the smallest vocabulary but longest sequences.

**Core Concepts — Strategy Comparison:**

| Strategy | Vocab Size | Sequence Length | OOV Handling | Best For |
|---|---|---|---|---|
| Word-level | Large (50K–200K) | Short | Poor (UNK tokens) | Closed-domain NLP, BoW models, IR |
| Subword (BPE/WP) | Medium (30K–50K) | Medium | Good (splits unseen words) | LLMs, MT, most modern NLP |
| Character-level | Tiny (< 300) | Very long | None (all chars known) | Spell-check, morphological tasks |
| Hybrid | Configurable | Medium | Good | Domain-specific, noisy text |

**Decision Framework:**
1. **Language morphology** — Agglutinative languages (Turkish, Finnish) benefit from subword; isolating languages (Chinese) need character/word segmentation.
2. **Vocabulary openness** — Social media or biomedical text with neologisms needs subword; formal legal text may work with word-level.
3. **Downstream task** — Generation tasks prefer subword for fluent decoding; classification can tolerate word-level.
4. **Compute budget** — Character-level sequences are 4–5× longer, increasing attention cost quadratically.

```python
# Pipeline: Compare tokenization strategies on sample text
from transformers import AutoTokenizer

text = "Hydroxychloroquine showed antiviral properties in vitro #COVID19"

# Word-level (whitespace)
word_tokens = text.split()
print(f"Word-level ({len(word_tokens)} tokens): {word_tokens}")

# Subword (BPE via GPT-2 tokenizer)
tokenizer_bpe = AutoTokenizer.from_pretrained("gpt2")
bpe_tokens = tokenizer_bpe.tokenize(text)
print(f"BPE ({len(bpe_tokens)} tokens): {bpe_tokens}")

# Subword (WordPiece via BERT tokenizer)
tokenizer_wp = AutoTokenizer.from_pretrained("bert-base-uncased")
wp_tokens = tokenizer_wp.tokenize(text)
print(f"WordPiece ({len(wp_tokens)} tokens): {wp_tokens}")

# Character-level
char_tokens = list(text)
print(f"Character ({len(char_tokens)} tokens): {char_tokens[:20]}...")

# Fertility (tokens per word) — lower is better
num_words = len(text.split())
print(f"\nFertility — BPE: {len(bpe_tokens)/num_words:.2f}, "
      f"WordPiece: {len(wp_tokens)/num_words:.2f}, "
      f"Char: {len(char_tokens)/num_words:.2f}")
```

**Interview Tips:**
- Default to subword tokenization (BPE/WordPiece) — it's the industry standard for transformer models
- Mention **fertility** (tokens per word) as the key metric: lower = faster inference
- Word-level is still valid for classical ML (TF-IDF, BoW) and closed-vocabulary tasks
- Character-level is niche: useful for typo-robust models and morphological analysis
- Always check tokenizer-model compatibility — mismatched tokenizers degrade performance

---

## Question 2
- [ ] Done
**What are the trade-offs between stemming and lemmatization for information retrieval applications?**

**Answer:**

**Stemming** reduces words to a root form by stripping suffixes via heuristic rules (e.g., "running" → "run", "studies" → "studi"), while **lemmatization** maps words to their dictionary base form using morphological analysis and POS context (e.g., "better" → "good", "studies" → "study"). For information retrieval (IR), this choice affects recall, precision, index size, and processing speed.

**Core Concepts — Comparison Table:**

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Approach | Rule-based suffix stripping | Dictionary + morphological analysis |
| Speed | Very fast (regex rules) | Slower (needs POS tagging, lookup) |
| Output | May not be a real word ("studi") | Always a valid lemma ("study") |
| Recall | Higher (aggressive conflation) | Moderate (precise grouping) |
| Precision | Lower (over-stemming errors) | Higher (fewer false merges) |
| Index size | Smaller (more conflation) | Slightly larger |
| Language support | Limited (English-centric) | Better multilingual support |
| Common tools | Porter, Snowball, Lancaster | WordNet, spaCy, Stanza |

**Key Trade-offs for IR:**
- **Over-stemming** — Stemming may merge unrelated words: "university" and "universe" → "univers", hurting precision.
- **Under-stemming** — Lemmatization may keep related forms separate if POS tags differ, reducing recall.
- **Speed vs. quality** — Stemming handles millions of documents/sec; lemmatization requires NLP pipeline overhead.
- **Query expansion** — Stemming naturally expands queries; lemmatization needs explicit synonym handling.

```python
# Pipeline: Compare stemming vs lemmatization for IR
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["studies", "studying", "better", "running", "geese", "universities", "universal"]

# Stemming
stems = {w: stemmer.stem(w) for w in words}
print("Stemming results:")
for w, s in stems.items():
    print(f"  {w:15s} -> {s}")

# Lemmatization (with POS mapping)
def get_wordnet_pos(tag):
    from nltk.corpus import wordnet
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

print("\nLemmatization results:")
for w in words:
    tag = pos_tag([w])[0][1]
    lemma = lemmatizer.lemmatize(w, get_wordnet_pos(tag))
    print(f"  {w:15s} -> {lemma}")

# IR demo: query-document matching
query = "studying universities"
doc = "The university study showed improved results"
q_stems = {stemmer.stem(w) for w in query.split()}
d_stems = {stemmer.stem(w) for w in doc.split()}
print(f"\nStem overlap: {q_stems & d_stems}")
```

**Interview Tips:**
- For high-recall IR (web search), stemming is preferred — speed and broad matching outweigh occasional errors
- For high-precision IR (legal/medical), lemmatization is safer — avoids conflating unrelated terms
- Modern neural IR (dense retrieval) largely bypasses this choice via subword tokenization
- Mention specific stemmer trade-offs: Porter (light) vs Lancaster (aggressive)
- Real systems often use stemming at index time + lemmatization for query understanding

---

## Question 3
- [ ] Done
**How do you handle tokenization for multilingual texts with mixed scripts and languages?**

**Answer:**

Multilingual tokenization must handle texts containing multiple scripts (Latin, Cyrillic, CJK, Arabic, Devanagari) and languages interleaved within sentences. The core challenge is that different scripts have different word boundary conventions — spaces in Latin scripts, no spaces in Chinese/Japanese, and morphological complexity in Arabic/Hindi.

**Core Concepts:**

| Challenge | Example | Solution |
|---|---|---|
| Mixed scripts | "I love 東京 and Paris" | Unicode-aware segmentation |
| No word boundaries | "今日は天気がいい" (Japanese) | CJK-specific segmenters (MeCab, jieba) |
| Code-switching | "Yeh bahut achha hai bro" | Language-ID + per-language tokenizer |
| Shared subwords | Cognates across languages | Multilingual BPE with shared vocab |
| Script detection | Identifying script boundaries | Unicode block detection |

**Key Strategies:**
1. **Multilingual subword models** — SentencePiece trained on multilingual corpora (used by mBERT, XLM-R) handles mixed scripts natively with a shared vocabulary.
2. **Script-based routing** — Detect Unicode script blocks and route segments to specialized tokenizers (MeCab for Japanese, jieba for Chinese, whitespace for Latin).
3. **Unicode normalization** — Apply NFC/NFKC normalization to handle equivalent character representations.
4. **Balanced vocabulary sampling** — Prevent high-resource languages from dominating the shared vocabulary using temperature-based sampling.

```python
# Pipeline: Multilingual tokenization with script detection and routing
import unicodedata
import re

def detect_script_segments(text):
    """Split text into segments by Unicode script."""
    segments = []
    current_script = None
    current_text = ""
    for char in text:
        if char.isspace():
            current_text += char
            continue
        script = unicodedata.name(char, "").split()[0] if unicodedata.name(char, "") else "UNKNOWN"
        if script != current_script and current_text.strip():
            segments.append((current_script, current_text.strip()))
            current_text = ""
        current_script = script
        current_text += char
    if current_text.strip():
        segments.append((current_script, current_text.strip()))
    return segments

# Multilingual text: English + Japanese + Hindi
text = "I visited 東京タワー and it was अद्भुत experience"
segments = detect_script_segments(text)
for script, segment in segments:
    print(f"  [{script:12s}] {segment}")

# SentencePiece-based multilingual tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokens = tokenizer.tokenize(text)
print(f"\nXLM-R tokens ({len(tokens)}): {tokens}")

# Check fertility across languages
for lang, sample in [("English", "The weather is nice today"),
                      ("Japanese", "今日は天気がいいです"),
                      ("Hindi", "आज मौसम अच्छा है")]:
    toks = tokenizer.tokenize(sample)
    print(f"  {lang}: {len(toks)} tokens (fertility: {len(toks)/len(sample.split()):.2f})")
```

**Interview Tips:**
- Default recommendation: use SentencePiece-based multilingual models (XLM-R) for mixed-script text
- Mention the **curse of multilinguality** — shared vocab dilutes per-language representation quality
- Vocabulary balancing via temperature sampling (T=0.7 in XLM-R) is critical for fair coverage
- For production: script detection → language ID → per-language tokenizer pipeline gives best quality
- Unicode NFKC normalization is a must before any tokenization

---

## Question 4
- [ ] Done
**What techniques work best for tokenization of social media text with informal language and emojis?**

**Answer:**

Social media text presents unique tokenization challenges: abbreviations ("u r gr8"), hashtags (#MachineLearning), @mentions, emojis (😂🔥), elongations ("sooooo"), missing punctuation, and creative spelling. Standard tokenizers trained on formal text fail on these patterns, requiring specialized preprocessing and tokenization strategies.

**Core Concepts — Social Media Challenges:**

| Challenge | Example | Solution |
|---|---|---|
| Hashtag segmentation | #DeepLearningIsFun | CamelCase splitting, WordNinja |
| Emojis/emoticons | 😂❤️ :) | Emoji-aware tokenizer, emoji-to-text |
| Elongation | "soooooo goood" | Regex normalization (3+ → 2 chars) |
| Abbreviations | "u r gr8 lol" | Lookup normalization dictionary |
| @mentions/URLs | @user http://t.co/x | Regex replacement with placeholders |
| Missing spaces | "idk why lol" | Already separate; "idkwhy" needs segmentation |
| Mixed case/slang | "OMG thats AMAZINGG" | Case-aware normalization |

**Recommended Approach:**
1. **Pre-normalization** — Replace URLs, @mentions with tokens; normalize elongations; segment hashtags.
2. **Specialized tokenizers** — TweetTokenizer (NLTK), ekphrasis, or BERTweet tokenizer handle social media conventions.
3. **Emoji handling** — Keep emojis as tokens (they carry sentiment) or convert to text with `emoji` library.
4. **Domain-adapted subword models** — BERTweet, TwHIN-BERT are pretrained on tweets with social-media-aware vocabulary.

```python
# Pipeline: Social media text tokenization
import re
from nltk.tokenize import TweetTokenizer

tweet = "OMG 😂😂 @elonmusk's #SpaceXLaunch was sooooo amazing!!! "\
        "check it out https://t.co/abc123 u wont believe it lol 🚀🔥"

# Step 1: Pre-normalization
def normalize_social_text(text):
    # Replace URLs
    text = re.sub(r'https?://\S+', '<URL>', text)
    # Replace @mentions (keep for context or replace)
    text = re.sub(r'@\w+', '<USER>', text)
    # Normalize elongations (3+ repeated chars → 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Segment CamelCase hashtags
    text = re.sub(r'#(\S+)', lambda m: '# ' + re.sub(r'([a-z])([A-Z])', r'\1 \2', m.group(1)), text)
    return text

normalized = normalize_social_text(tweet)
print(f"Normalized: {normalized}")

# Step 2: Tweet-aware tokenization
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
tokens = tokenizer.tokenize(tweet)
print(f"\nTweetTokenizer ({len(tokens)} tokens): {tokens}")

# Step 3: BERTweet tokenization (social-media pretrained)
from transformers import AutoTokenizer
bertweet_tok = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
bt_tokens = bertweet_tok.tokenize(tweet)
print(f"\nBERTweet ({len(bt_tokens)} tokens): {bt_tokens}")

# Step 4: Emoji handling
import emoji
text_with_emoji_desc = emoji.demojize(tweet)
print(f"\nEmoji→text: {text_with_emoji_desc}")
```

**Interview Tips:**
- Always mention BERTweet or TwHIN-BERT for tweet/social media NLP tasks
- Emojis are valuable features for sentiment — don't strip them blindly
- Hashtag segmentation (#MachineLearning → Machine Learning) improves downstream performance
- Elongation normalization is simple but high-impact: reduces vocab size significantly
- TweetTokenizer preserves emoticons like `:)` as single tokens, unlike standard tokenizers

---

## Question 5
- [ ] Done
**How do you implement subword tokenization algorithms like BPE, WordPiece, and SentencePiece?**

**Answer:**

Subword tokenization algorithms solve the vocabulary size vs. coverage trade-off by splitting rare words into frequent subunits while keeping common words intact. **BPE** (Byte Pair Encoding) iteratively merges the most frequent adjacent character pairs. **WordPiece** uses a likelihood-based scoring criterion instead of raw frequency. **SentencePiece** is a language-agnostic framework that treats the input as a raw byte stream, handling any language without pre-tokenization.

**Core Concepts — Algorithm Comparison:**

| Aspect | BPE | WordPiece | Unigram (SentencePiece) |
|---|---|---|---|
| Merge criterion | Most frequent pair | Max likelihood gain | Remove lowest-loss subwords |
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) |
| Training | Greedy, deterministic | Greedy, deterministic | EM-based optimization |
| Tokenization | Deterministic | Greedy left-to-right | Probabilistic (Viterbi) |
| Pre-tokenization | Needed (whitespace) | Needed (whitespace) | None (raw text) |
| Used by | GPT-2, RoBERTa, LLaMA | BERT, DistilBERT | T5, XLM-R, ALBERT |
| Subword prefix | `Ġ` (space before) | `##` (continuation) | `▁` (sentence start) |

**How BPE Works:**
1. Start with character-level vocabulary + end-of-word marker
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until desired vocabulary size is reached

```python
# Pipeline: Implement BPE from scratch + compare with HuggingFace tokenizers
from collections import Counter, defaultdict

def train_bpe(corpus, num_merges=10):
    """Minimal BPE implementation."""
    # Initialize: split words into characters
    word_freqs = Counter(corpus)
    splits = {word: list(word) + ['</w>'] for word in word_freqs}
    merges = []
    
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for j in range(len(symbols) - 1):
                pair_counts[(symbols[j], symbols[j+1])] += freq
        
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        
        # Merge best pair in all words
        for word in splits:
            symbols = splits[word]
            new_symbols = []
            j = 0
            while j < len(symbols):
                if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                    new_symbols.append(symbols[j] + symbols[j+1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1
            splits[word] = new_symbols
        
        print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} -> {best_pair[0]+best_pair[1]}")
    
    return merges, splits

# Train BPE on a toy corpus
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
merges, splits = train_bpe(corpus, num_merges=10)
print("\nFinal splits:")
for word, toks in splits.items():
    print(f"  {word} -> {toks}")

# Compare with HuggingFace tokenizers
from transformers import AutoTokenizer
text = "unbelievably"
for name in ["gpt2", "bert-base-uncased", "t5-small"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(f"\n{name}: {tok.tokenize(text)}")
```

**Interview Tips:**
- BPE is the most commonly asked — know the merge algorithm step by step
- Key difference: BPE uses frequency, WordPiece uses likelihood gain `P(ab)/P(a)P(b)`
- SentencePiece is a *framework* (not algorithm) — it supports both BPE and Unigram modes
- Unigram goes top-down (starts large, prunes), opposite of BPE's bottom-up merging
- Mention that `tokenizers` library (HuggingFace) lets you train custom BPE/WordPiece in minutes

---

## Question 6
- [ ] Done
**What strategies help with handling out-of-vocabulary words during tokenization?**

**Answer:**

**Out-of-vocabulary (OOV)** words are tokens not present in the tokenizer's vocabulary, causing them to be mapped to a generic `<UNK>` token that discards all semantic information. OOV handling is critical because unseen words (typos, neologisms, domain terms, names) are inevitable in real-world text. The goal is to minimize information loss while maintaining a manageable vocabulary size.

**Core Concepts — OOV Handling Strategies:**

| Strategy | How It Works | Pros | Cons |
|---|---|---|---|
| Subword tokenization | Split OOV into known subwords | Zero UNK rate, preserves morphology | Longer sequences |
| Character fallback | Fall back to char-level for OOV | Handles any string | Very long, loses semantics |
| Hash-based embedding | Hash OOV to fixed embedding buckets | Bounded memory, no UNK | Hash collisions |
| Morphological analysis | Decompose OOV into root + affixes | Linguistically motivated | Language-specific |
| Dynamic vocabulary | Expand vocab during fine-tuning | Full coverage for new domains | Requires retraining embeddings |
| Spell correction | Correct OOV to nearest known word | Reduces noise | May change meaning |
| Copy mechanism | Copy OOV directly from input | Preserves names/entities | Only for generation tasks |

**Best Practices:**
1. **Use subword tokenizers** — BPE/WordPiece eliminate OOV entirely by construction.
2. **Domain vocabulary augmentation** — Add domain-specific tokens and continue pretraining.
3. **Byte-level fallback** — GPT-2's byte-level BPE can tokenize *any* UTF-8 string with zero UNK.
4. **FastText-style embeddings** — Compose word vectors from character n-grams, providing embeddings even for unseen words.

```python
# Pipeline: Demonstrate OOV handling strategies
from transformers import AutoTokenizer

oov_words = ["covfefe", "CRISPR-Cas9", "doomscrolling", "üñîcödé"]

# Strategy 1: Word-level (OOV = UNK)
word_vocab = {"the", "is", "a", "good", "word"}
for w in oov_words:
    result = w if w.lower() in word_vocab else "<UNK>"
    print(f"Word-level: {w} -> {result}")

# Strategy 2: Subword (BPE) — no UNK possible
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("\nBPE subword (GPT-2):")
for w in oov_words:
    tokens = tokenizer.tokenize(w)
    print(f"  {w} -> {tokens}")

# Strategy 3: Hash-based OOV embedding
import hashlib
def hash_oov_embedding(word, num_buckets=1000, embed_dim=4):
    bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % num_buckets
    return f"hash_bucket_{bucket}"

print("\nHash-based:")
for w in oov_words:
    print(f"  {w} -> {hash_oov_embedding(w)}")

# Strategy 4: Dynamic vocab addition
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"\nBefore adding tokens: {tok.tokenize('CRISPR-Cas9')}")
tok.add_tokens(["crispr-cas9"])
print(f"After adding token:  {tok.tokenize('crispr-cas9')}")
print(f"New vocab size: {len(tok)}")
```

**Interview Tips:**
- "Use BPE/byte-level BPE" is the modern answer — zero OOV by design
- For domain adaptation, mention `tokenizer.add_tokens()` + resize model embeddings
- Hash-based embeddings (like in FastText) are asked about in feature engineering contexts
- Copy mechanism / pointer networks handle OOV in seq2seq tasks (summarization, translation)
- Byte-level BPE (GPT-2) vs byte-fallback (LLaMA's SentencePiece) are both valid zero-OOV solutions

---

## Question 7
- [ ] Done
**How do you design tokenization schemes that preserve important linguistic information?**

**Answer:**

Standard tokenizers often destroy linguistic information critical for downstream tasks: morphological boundaries (un-believe-able), multi-word expressions ("New York"), entity spans, and syntactic cues (punctuation, case). **Linguistically-informed tokenization** explicitly preserves these signals by incorporating morphological rules, named entity awareness, and configurable splitting policies.

**Core Concepts — Linguistic Information to Preserve:**

| Linguistic Feature | What Gets Lost | Preservation Strategy |
|---|---|---|
| Morpheme boundaries | "unhappiness" → random BPE splits | Morphological pre-segmentation (Morfessor) |
| Multi-word expressions | "New York" → ["New", "York"] | MWE-aware tokenizer, compound tokens |
| Named entities | "O'Brien" → ["O", "'", "Brien"] | Entity-aware pre-tokenization |
| Case information | "URGENT" vs "urgent" | Case-sensitive vocab or case features |
| Punctuation semantics | "Dr." → ["Dr", "."] | Abbreviation-aware rules |
| Negation scope | "don't" → ["don", "'", "t"] | Contraction handling ("do", "n't") |
| Hyphenated compounds | "state-of-the-art" | Keep as single token or split meaningfully |

**Design Principles:**
1. **Morphological pre-segmentation** — Use Morfessor or morphological analyzers to split words at linguistically valid boundaries before subword tokenization.
2. **Pre-tokenization rules** — Define regex rules that handle contractions, abbreviations, and special patterns before the subword model runs.
3. **Special tokens for structure** — Add tokens for sentence boundaries, paragraph breaks, and discourse markers.
4. **Alignment with annotations** — Ensure token boundaries align with NER/POS annotation spans.

```python
# Pipeline: Linguistically-informed tokenization design
import re
from transformers import AutoTokenizer

text = "Dr. O'Brien couldn't believe the state-of-the-art AI in New York."

# Problem: naive splitting destroys linguistic info
naive_tokens = text.split()
print(f"Naive split: {naive_tokens}")

# Step 1: Linguistic pre-tokenization rules
def linguistic_pretokenize(text):
    """Apply linguistically-informed rules before subword tokenization."""
    rules = [
        # Protect abbreviations (Dr., Mr., U.S.)
        (r'\b(Dr|Mr|Mrs|Ms|Jr|Sr|U\.S)\.', r'\1<DOT>'),
        # Handle contractions linguistically
        (r"(\w)(n't)", r'\1 \2'),       # can't -> ca n't
        (r"(\w)('ll|'re|'ve|'d|'s)", r'\1 \2'),  # I'll -> I 'll
        # Protect multi-word expressions (simplified NER)
        (r'New York', 'New_York'),
        (r'state-of-the-art', 'state_of_the_art'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text.replace('<DOT>', '.')

pretokenized = linguistic_pretokenize(text)
print(f"\nPre-tokenized: {pretokenized}")

# Step 2: Compare subword tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
original_tokens = tokenizer.tokenize(text)
print(f"\nBERT on original:      {original_tokens}")

# Step 3: Morphological segmentation concept
def morphological_split(word):
    """Simple rule-based morphological segmentation."""
    prefixes = ['un', 'dis', 're', 'pre', 'mis']
    suffixes = ['ness', 'ment', 'able', 'tion', 'ing', 'ly', 'ed']
    parts = []
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            parts.append(p); word = word[len(p):]; break
    for s in suffixes:
        if word.endswith(s) and len(word) > len(s) + 2:
            parts.append(word[:-len(s)]); parts.append(s); word = ""; break
    if word: parts.append(word)
    return parts

for word in ["unhappiness", "unbelievable", "preprocessing"]:
    print(f"  {word} -> {morphological_split(word)}")
```

**Interview Tips:**
- Key insight: subword BPE splits are *statistically* motivated, not linguistically — this can split "unhappy" as ["unh", "appy"]
- Mention Morfessor for unsupervised morphological segmentation as a pre-tokenization step
- Contraction handling ("don't" → "do" + "n't") is standard in spaCy and Stanford tokenizers
- For NER tasks, ensure token boundaries align with entity spans — misalignment causes label errors
- Multi-word expression handling is important for machine translation and information extraction

---

## Question 8
- [ ] Done
**What approaches work best for tokenization of domain-specific texts like legal or medical documents?**

**Answer:**

Domain-specific texts contain specialized vocabulary ("myocardial infarction", "habeas corpus"), acronyms ("ECG", "HIPAA"), structured patterns (citations, dosages), and long compound terms that general-purpose tokenizers handle poorly. Effective domain tokenization requires **vocabulary augmentation**, **domain-adapted pretraining**, and **specialized pre-tokenization rules**.

**Core Concepts — Domain Challenges and Solutions:**

| Domain | Challenge | Example | Solution |
|---|---|---|---|
| Medical | Latin/Greek terms | "electroencephalography" | Domain BPE (PubMedBERT vocab) |
| Medical | Dosage patterns | "500mg b.i.d." | Regex pre-tokenization rules |
| Legal | Long formal phrases | "notwithstanding the foregoing" | MWE-aware tokenization |
| Legal | Citation formats | "42 U.S.C. § 1983" | Citation-preserving rules |
| Scientific | Chemical formulas | "H₂SO₄", "CRISPR-Cas9" | Entity-aware special tokens |
| Financial | Numeric expressions | "$1.2M USD", "Q3 FY2024" | Number normalization patterns |

**Recommended Strategy (3-Tier Approach):**
1. **Domain-specific pretrained tokenizer** — Use models pretrained on domain corpora (PubMedBERT, LegalBERT, SciBERT) that already have domain-optimized vocabularies.
2. **Vocabulary augmentation** — Add domain terms to an existing tokenizer using `add_tokens()`, then continue pretraining the model embeddings.
3. **Custom pre-tokenization** — Define regex rules to protect domain patterns (citations, formulas, codes) before subword splitting.

```python
# Pipeline: Domain-specific tokenization for medical text
from transformers import AutoTokenizer

medical_text = "Patient presents with acute myocardial infarction (STEMI). "\
               "Started on aspirin 325mg PO and heparin 5000U IV bolus. "\
               "ECG shows ST-elevation in leads V1-V4."

# Compare general vs domain tokenizers
general_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
domain_tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

gen_tokens = general_tok.tokenize(medical_text)
dom_tokens = domain_tok.tokenize(medical_text)

print(f"General BERT ({len(gen_tokens)} tokens):")
print(f"  {gen_tokens}")
print(f"\nBiomedBERT ({len(dom_tokens)} tokens):")
print(f"  {dom_tokens}")
print(f"\nTokens saved: {len(gen_tokens) - len(dom_tokens)} "
      f"({(len(gen_tokens)-len(dom_tokens))/len(gen_tokens)*100:.1f}% reduction)")

# Custom pre-tokenization for domain patterns
import re

def medical_pretokenize(text):
    """Protect medical patterns from splitting."""
    # Protect dosage patterns
    text = re.sub(r'(\d+)(mg|mcg|mL|U|IU)', r'\1 \2', text)
    # Protect abbreviations
    text = re.sub(r'\b(STEMI|NSTEMI|ECG|EKG|IV|PO|PRN|BID)\b',
                  lambda m: f'<{m.group()}>',  text)
    # Protect lead ranges
    text = re.sub(r'(V\d)-(V\d)', r'\1_to_\2', text)
    return text

protected = medical_pretokenize(medical_text)
print(f"\nProtected text: {protected}")

# Vocabulary augmentation
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
new_terms = ["myocardial", "infarction", "stemi", "heparin"]
num_added = tok.add_tokens(new_terms)
print(f"\nAdded {num_added} domain tokens. New vocab size: {len(tok)}")
print(f"After augmentation: {tok.tokenize('myocardial infarction')}")
```

**Interview Tips:**
- Always start with domain-pretrained models: PubMedBERT (medical), LegalBERT (legal), SciBERT (science)
- Vocabulary augmentation + continued pretraining is the standard recipe for domain adaptation
- Mention **fertility reduction** as the metric: domain tokenizers produce fewer tokens for domain text
- Domain tokenization directly impacts max sequence length — fewer tokens = more content in 512-token window
- Pre-tokenization rules for citations, formulas, and codes prevent information-destroying splits

---

## Question 9
- [ ] Done
**How do you handle tokenization for languages without clear word boundaries?**

**Answer:**

Languages like Chinese, Japanese, Thai, Khmer, and Lao do not use spaces between words, making word boundary detection itself a significant NLP task. **Word segmentation** for these languages requires statistical models, dictionary-based matching, or neural sequence labeling approaches rather than simple whitespace splitting.

**Core Concepts — Approaches by Language:**

| Language | Challenge | Tools | Approach |
|---|---|---|---|
| Chinese | No spaces, huge char set | jieba, pkuseg, LTP | Dict + HMM / neural |
| Japanese | 3 scripts (hiragana, katakana, kanji) | MeCab, SudachiPy, Janome | Lattice-based (Viterbi) |
| Thai | No spaces, complex consonant clusters | PyThaiNLP, deepcut | CRF / CNN / BiLSTM |
| Korean | Syllable blocks, agglutinative | KoNLPy (Mecab-ko) | Morphological analysis |
| Vietnamese | Spaces between syllables (not words) | VnCoreNLP, underthesea | Multi-syllable word detection |

**Key Approaches:**
1. **Dictionary-based (Maximum Matching)** — Forward/backward maximum matching using a word dictionary. Fast but fails on ambiguous segmentations and OOV.
2. **Statistical (HMM/CRF)** — Treat segmentation as sequence labeling (B-I-E-S tags). jieba uses HMM for unknown words.
3. **Neural (BiLSTM-CRF / BERT)** — Fine-tune BERT for word segmentation as token classification. State-of-the-art accuracy.
4. **Character-level / SentencePiece** — Skip word segmentation entirely; let the subword model learn useful units. Used by multilingual transformers.

```python
# Pipeline: Word segmentation for Chinese and Japanese

# Chinese segmentation with jieba
import jieba

chinese_text = "我们在学习自然语言处理技术"  # "We are learning NLP technology"

# Default mode (accurate)
seg_default = list(jieba.cut(chinese_text, cut_all=False))
print(f"jieba (accurate): {' / '.join(seg_default)}")

# Full mode (all possible words)
seg_full = list(jieba.cut(chinese_text, cut_all=True))
print(f"jieba (full):     {' / '.join(seg_full)}")

# Search engine mode (for IR)
seg_search = list(jieba.cut_for_search(chinese_text))
print(f"jieba (search):   {' / '.join(seg_search)}")

# Add custom domain words
jieba.add_word("自然语言处理")
seg_custom = list(jieba.cut(chinese_text, cut_all=False))
print(f"jieba (custom):   {' / '.join(seg_custom)}")

# Subword approach (SentencePiece) — no segmentation needed
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-chinese")
cn_tokens = tok.tokenize(chinese_text)
print(f"\nBERT-Chinese: {cn_tokens}")

# XLM-R handles Chinese with SentencePiece
tok_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlm_tokens = tok_xlm.tokenize(chinese_text)
print(f"XLM-R:        {xlm_tokens}")

# Maximum Matching algorithm (educational)
def forward_max_match(text, dictionary, max_len=5):
    """Forward maximum matching segmenter."""
    words, i = [], 0
    while i < len(text):
        matched = False
        for length in range(min(max_len, len(text) - i), 0, -1):
            candidate = text[i:i+length]
            if candidate in dictionary:
                words.append(candidate)
                i += length
                matched = True
                break
        if not matched:
            words.append(text[i])
            i += 1
    return words

word_dict = {"我们", "在", "学习", "自然语言", "处理", "技术", "自然", "语言"}
result = forward_max_match(chinese_text, word_dict)
print(f"\nMax-match:    {' / '.join(result)}")
```

**Interview Tips:**
- Chinese segmentation is the classic example — know jieba (dictionary + HMM hybrid)
- Japanese MeCab uses a lattice + Viterbi algorithm for optimal path segmentation
- Modern approach: skip explicit segmentation and use character/SentencePiece tokenization (BERT-Chinese, XLM-R)
- Mention **segmentation ambiguity**: "下雨天留客天留我不留" has multiple valid segmentations with different meanings
- Word segmentation errors propagate to all downstream tasks — neural approaches reduce this cascade

---

## Question 11
- [ ] Done
**How do you implement lemmatization for morphologically rich languages?**

**Answer:**

**Morphologically rich languages** (MRLs) like Turkish, Finnish, Hungarian, Arabic, and Russian have extensive inflection systems where a single lemma can produce hundreds or thousands of surface forms. For example, a Turkish verb can have over 100 inflected forms. Simple dictionary lookup fails because the combinatorial explosion makes full form listing impractical. Effective lemmatization for MRLs requires **morphological decomposition** — analyzing the word's internal structure (root + affixes) to recover the base form.

**Core Concepts — MRL Challenges:**

| Language | Morphology Type | Example | Challenge |
|---|---|---|---|
| Turkish | Agglutinative | "çalıştırılamıyabileceklerimizdenmiydi" | 10+ suffixes per word |
| Finnish | Agglutinative | "talossanikin" → talo+ssa+ni+kin | Cases, possessives, clitics |
| Arabic | Root-pattern (templatic) | "كِتَاب" / "كُتُب" → root k-t-b | Non-concatenative morphology |
| Russian | Fusional | "книгами" → "книга" | Fused case/number/gender |
| Hungarian | Agglutinative | "megérthetetlenül" | Vowel harmony affects suffixes |

**Approaches for MRL Lemmatization:**
1. **Morphological analyzers (FST)** — Finite-state transducers (HFST, Xerox/foma) encode morphological rules. High precision but require linguist-crafted rule sets.
2. **Neural seq2seq** — Treat lemmatization as character-level transduction (inflected form → lemma). Used by Stanza, UDPipe.
3. **Morphological taggers (Stanza/UDPipe)** — Joint POS tagging + morphological feature prediction + lemmatization via neural pipeline.
4. **Unsupervised (Morfessor)** — Learn morpheme segmentation from raw corpus statistics without labeled data.

```python
# Pipeline: Lemmatization for morphologically rich languages
import stanza

# Turkish lemmatization with Stanza (neural pipeline)
stanza.download('tr', processors='tokenize,mwt,pos,lemma', verbose=False)
nlp_tr = stanza.Pipeline('tr', processors='tokenize,mwt,pos,lemma', verbose=False)

turkish_text = "Kitaplarımı çocuğuma verdim"
# "I gave my books to my child"
doc = nlp_tr(turkish_text)

print("Turkish lemmatization:")
printed = []
for sent in doc.sentences:
    for word in sent.words:
        print(f"  {word.text:20s} -> {word.lemma:15s} [POS: {word.upos}, "
              f"Feats: {word.feats or 'None'}]")

# Russian lemmatization
stanza.download('ru', processors='tokenize,pos,lemma', verbose=False)
nlp_ru = stanza.Pipeline('ru', processors='tokenize,pos,lemma', verbose=False)

russian_text = "Она читала интересные книги"
# "She read interesting books"
doc_ru = nlp_ru(russian_text)

print("\nRussian lemmatization:")
for sent in doc_ru.sentences:
    for word in sent.words:
        print(f"  {word.text:20s} -> {word.lemma:15s} [POS: {word.upos}]")

# Compare: simple dictionary lookup vs neural
# Dictionary approach fails for unseen inflections
simple_dict = {"kitaplarımı": "kitap", "verdim": "vermek"}
word = "çocuğuma"  # not in dict
result = simple_dict.get(word, "<UNKNOWN>")
print(f"\nDict lookup for '{word}': {result}")
print("Neural pipeline correctly handles it via morphological analysis.")
```

**Interview Tips:**
- Stanza/UDPipe are go-to tools for MRL lemmatization — they support 70+ languages out of the box
- Arabic is especially tricky due to **non-concatenative morphology** (root pattern interleaving), requiring specialized tools like CAMeL Tools
- Neural seq2seq lemmatization outperforms rule-based FSTs on unseen word forms
- For Turkish/Finnish, subword tokenization (SentencePiece) can partially replace lemmatization by splitting morphemes
- Mention the Universal Dependencies (UD) treebanks as the standard training data for multilingual lemmatization

---

## Question 12
- [ ] Done
**What strategies work best for handling tokenization of noisy or corrupted text data?**

**Answer:**

**Noisy text** contains OCR errors, typos, encoding artifacts ("Ã©" instead of "é"), mixed formatting, broken HTML, and random corruption. Standard tokenizers assume clean input and produce garbage tokens on noisy text, degrading all downstream tasks. Robust tokenization requires **noise-aware preprocessing**, **error-tolerant tokenization**, and **post-correction** strategies.

**Core Concepts — Noise Types and Solutions:**

| Noise Type | Example | Solution |
|---|---|---|
| OCR errors | "rnachine learning" (m→rn) | Spell correction, visual similarity models |
| Encoding corruption | "Ã©" for "é", "\x00" | Encoding detection (chardet) + ftfy |
| HTML/XML artifacts | "&amp;amp;" | Strip/decode markup before tokenization |
| Random insertions | "mac#hine lea$rning" | Regex noise removal, character filtering |
| Missing spaces | "machinelearningis" | Word segmentation (WordNinja, ekphrasis) |
| Extra whitespace | "machine   learning" | Whitespace normalization |
| Mixed case/format | "MACHINE LeArNiNg" | Case normalization |

**Recommended Pipeline:**
1. **Encoding normalization** — Use `ftfy` to fix encoding errors; detect encoding with `chardet`.
2. **Text cleaning** — Remove/decode HTML entities, strip control characters, normalize Unicode (NFKC).
3. **Noise-tolerant tokenization** — Byte-level BPE (GPT-2 style) handles corrupted text gracefully since it operates on raw bytes.
4. **Post-tokenization correction** — Spell correction on tokens, merge broken tokens.

```python
# Pipeline: Robust tokenization of noisy text
import re
import unicodedata

def clean_noisy_text(text):
    """Multi-stage noise removal pipeline."""
    # Step 1: Fix encoding issues
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    
    # Step 2: Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # Step 3: Remove control characters (keep newlines, tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Step 4: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 5: Remove HTML entities
    import html
    text = html.unescape(text)
    
    # Step 6: Fix common OCR errors
    ocr_fixes = {'rn': 'm', 'vv': 'w', 'cl': 'd', '1': 'l'}  # context-dependent
    
    return text

# Noisy text examples
noisy_samples = [
    "Ã©tude sur le machine\x00 learning &amp; deep learning",  # encoding + null + HTML
    "The   rnodel   achieved   95%   accuracy",                  # OCR + extra spaces
    "machinelearningisgreat",                                     # missing spaces
]

for noisy in noisy_samples:
    cleaned = clean_noisy_text(noisy)
    print(f"Noisy:   {repr(noisy)}")
    print(f"Cleaned: {cleaned}\n")

# Byte-level BPE handles noise gracefully
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

corrupted = "Th3 m0del w@s tr@ined on n01sy d@ta"
tokens = tok.tokenize(corrupted)
print(f"Byte-BPE on noisy text: {tokens}")
print(f"Token count: {len(tokens)} (no UNK tokens!)")

# Word segmentation for missing spaces
try:
    import wordninja
    joined = "machinelearningisgreat"
    segmented = wordninja.split(joined)
    print(f"\nWordNinja: '{joined}' -> {segmented}")
except ImportError:
    print("Install wordninja: pip install wordninja")
```

**Interview Tips:**
- `ftfy` library is the gold standard for fixing encoding errors in Python
- Byte-level BPE is inherently noise-robust because it can tokenize *any* byte sequence
- For OCR text, mention visual similarity models that learn character confusion patterns
- Always normalize Unicode (NFKC) before tokenization — prevents duplicate tokens for visually identical characters
- Noise handling order matters: encoding → unicode → HTML → whitespace → tokenize

---

## Question 14
- [ ] Done
**What approaches help with tokenization of code-mixed or transliterated text?**

**Answer:**

**Code-mixed text** interleaves words from two or more languages within a sentence ("Maine aaj meeting attend ki" — Hindi-English), while **transliterated text** writes a language in a non-native script (Hindi in Latin: "Kaise ho?"). Both break standard tokenizers because: (1) vocabulary crosses language boundaries, (2) transliteration creates non-standard spellings with high variation, and (3) language boundaries are unpredictable.

**Core Concepts — Challenges:**

| Challenge | Example | Impact |
|---|---|---|
| Language switching | "Yeh bahut amazing hai" | Tokenizer optimized for one language fails on the other |
| Transliteration variants | "achha" / "acha" / "accha" | Same word, multiple spellings → vocabulary explosion |
| Script mixing | "मैं office जा रहा हूं" | Mixed Devanagari + Latin confuses tokenizers |
| No standard orthography | Romanized Hindi has no spelling rules | OOV rate is very high |
| Shared cognates | "computer" used in Hindi context | Should it be Hindi or English? |

**Key Approaches:**
1. **Multilingual subword tokenizer** — SentencePiece trained on code-mixed corpora (used by IndicBERT, MuRIL).
2. **Language identification at token level** — Tag each token's language, then route to appropriate processing.
3. **Transliteration normalization** — Convert transliterated text to native script before tokenization using tools like `indic-transliteration`, AI4Bharat.
4. **Phonetic encoding** — Map transliteration variants to a canonical phonetic form (Soundex, Metaphone for Indian languages).
5. **Code-mixed pretrained models** — Models like GLUECoS-BERT, HingBERT pretrained on code-mixed data.

```python
# Pipeline: Code-mixed and transliterated text tokenization
from transformers import AutoTokenizer
import re

# Code-mixed text (Hindi-English, Romanized)
codemixed_text = "Maine aaj office mein ek amazing presentation di, boss ne bola well done"

# Problem: Standard English tokenizer
en_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
en_tokens = en_tok.tokenize(codemixed_text)
print(f"English BERT: {en_tokens}")
print(f"  Token count: {len(en_tokens)}")

# Solution 1: Multilingual tokenizer (XLM-R)
ml_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
ml_tokens = ml_tok.tokenize(codemixed_text)
print(f"\nXLM-R: {ml_tokens}")
print(f"  Token count: {len(ml_tokens)}")

# Solution 2: Language-level token tagging
def tag_language(text):
    """Simple heuristic language tagger for Hindi-English."""
    hindi_words = {"maine", "aaj", "mein", "ek", "di", "ne", "bola"}
    tagged = []
    for word in text.lower().split():
        if word in hindi_words:
            tagged.append((word, "HI"))
        else:
            tagged.append((word, "EN"))
    return tagged

tagged = tag_language(codemixed_text)
print("\nLanguage tags:")
for word, lang in tagged:
    print(f"  {word:15s} [{lang}]")

# Solution 3: Transliteration normalization concept
translit_variants = {
    "accha": "अच्छा", "achha": "अच्छा", "acha": "अच्छा",
    "bahut": "बहुत", "bohot": "बहुत", "bhot": "बहुत",
}
def normalize_transliteration(word):
    return translit_variants.get(word.lower(), word)

for variant in ["accha", "achha", "acha"]:
    print(f"  {variant} -> {normalize_transliteration(variant)}")

# Solution 4: Script-mixed text handling
mixed_script = "मैं office जा रहा हूं"
ml_tokens_mixed = ml_tok.tokenize(mixed_script)
print(f"\nScript-mixed XLM-R: {ml_tokens_mixed}")
```

**Interview Tips:**
- MuRIL (Google) and IndicBERT (AI4Bharat) are specifically designed for Indian code-mixed text
- Key challenge is **transliteration normalization** — mapping spelling variants to canonical form before tokenization
- Token-level language ID is a prerequisite for many code-mixed NLP pipelines
- XLM-R handles code-mixing reasonably well due to multilingual SentencePiece vocabulary
- Mention that code-mixing is extremely common in multilingual societies (India, Philippines, North Africa)

---

## Question 15
- [ ] Done
**How do you handle tokenization for real-time processing with computational constraints?**

**Answer:**

Real-time NLP systems (chatbots, autocomplete, search-as-you-type, streaming analytics) require tokenization latency in the **sub-millisecond** range per input, processing thousands of requests per second. This rules out heavy neural tokenizers and requires optimized implementations, caching, pre-computation, and efficient data structures.

**Core Concepts — Optimization Strategies:**

| Strategy | Speedup | Trade-off | Use Case |
|---|---|---|---|
| Compiled tokenizers (Rust) | 10–100× | None (same output) | HuggingFace `tokenizers` library |
| Vocabulary trie/hash | 5–20× | Memory for data structure | Custom BPE/WordPiece |
| Pre-tokenization cache | 50–100× for cache hits | Memory for cache | Repeated queries, common words |
| Batch tokenization | 2–5× | Latency for throughput | Offline/batch processing |
| Simpler algorithm | 2–10× | Slight quality loss | Whitespace + regex fallback |
| GPU tokenization | 10–50× | GPU memory required | cuDF / RAPIDS |
| Reduced vocabulary | 1.5–3× | OOV increases | Edge/mobile deployment |

**Key Optimizations:**
1. **Use HuggingFace `tokenizers`** — Rust-based implementation is 10–100× faster than Python-based tokenizers with identical output.
2. **LRU caching** — Cache tokenization results for frequent inputs (search queries, common phrases).
3. **Batch processing** — Tokenize multiple inputs in a single call to amortize overhead.
4. **Lazy tokenization** — Tokenize only what's needed (e.g., first 512 tokens for BERT, skip the rest).
5. **Pre-compiled vocabulary lookup** — Use trie or hash map for O(1) token lookup instead of iterative merging.

```python
# Pipeline: Optimized tokenization for real-time processing
import time
from functools import lru_cache

# Benchmark: Python vs Rust tokenizer speed
from transformers import AutoTokenizer, BertTokenizer

# Slow: Python-based tokenizer
slow_tok = BertTokenizer.from_pretrained("bert-base-uncased")

# Fast: Rust-based tokenizer (HuggingFace tokenizers)
fast_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

text = "Natural language processing with transformers is revolutionizing AI" * 10
n_iterations = 1000

# Benchmark slow tokenizer
start = time.time()
for _ in range(n_iterations):
    slow_tok.tokenize(text)
slow_time = time.time() - start

# Benchmark fast tokenizer
start = time.time()
for _ in range(n_iterations):
    fast_tok.tokenize(text)
fast_time = time.time() - start

print(f"Python tokenizer: {slow_time:.3f}s for {n_iterations} iterations")
print(f"Rust tokenizer:   {fast_time:.3f}s for {n_iterations} iterations")
print(f"Speedup:          {slow_time/fast_time:.1f}x")

# Strategy 1: LRU cache for repeated inputs
@lru_cache(maxsize=10000)
def cached_tokenize(text):
    return tuple(fast_tok.tokenize(text))

# First call: cache miss
start = time.time()
cached_tokenize("Hello world")
miss_time = time.time() - start

# Second call: cache hit
start = time.time()
cached_tokenize("Hello world")
hit_time = time.time() - start
print(f"\nCache miss: {miss_time*1e6:.1f}µs, Cache hit: {hit_time*1e6:.1f}µs")

# Strategy 2: Batch tokenization
texts = [f"Sample text number {i} for batch processing" for i in range(100)]

start = time.time()
batch_result = fast_tok(texts, padding=False, truncation=False)
batch_time = time.time() - start

start = time.time()
for t in texts:
    fast_tok(t, padding=False, truncation=False)
seq_time = time.time() - start

print(f"\nBatch ({len(texts)} texts): {batch_time*1000:.2f}ms")
print(f"Sequential:         {seq_time*1000:.2f}ms")
print(f"Batch speedup:      {seq_time/batch_time:.1f}x")

# Strategy 3: Truncation for bounded latency
long_text = "word " * 10000
start = time.time()
result = fast_tok(long_text, max_length=512, truncation=True)
trunc_time = time.time() - start
print(f"\nTruncated tokenization (10K words -> 512 tokens): {trunc_time*1000:.2f}ms")
```

**Interview Tips:**
- Always recommend HuggingFace `tokenizers` (Rust backend) — it's the industry standard for fast tokenization
- LRU caching is a simple but highly effective optimization for search/autocomplete systems
- Batch tokenization amortizes Python overhead and enables SIMD/parallelism
- For edge/mobile: mention ONNX Runtime tokenizer or TFLite with pre-tokenized inputs
- Mention that tokenization is rarely the bottleneck — model inference dominates latency, but for high-QPS systems it matters

---

## Question 17
- [ ] Done
**How do you implement adaptive tokenization that adjusts to different text domains?**

**Answer:**

**Definition:**
Adaptive tokenization dynamically selects or adjusts tokenization strategies based on the detected text domain (e.g., medical, legal, social media, code). Instead of applying a single static tokenizer, the system identifies domain-specific characteristics and routes text through the most appropriate tokenization pipeline, preserving domain-critical tokens that a generic tokenizer would split or merge incorrectly.

**Core Concepts:**

| Component | Purpose | Example |
|---|---|---|
| Domain Detector | Classifies input text domain | fastText classifier on text features |
| Domain-Specific Vocab | Specialized token vocabularies per domain | Medical: "myocardial infarction" as single token |
| Router/Dispatcher | Selects tokenizer based on detected domain | If medical → SciSpaCy tokenizer |
| Fallback Tokenizer | Default for unrecognized domains | BPE/WordPiece general-purpose |
| Vocabulary Augmentation | Adds domain terms to base vocabulary | Fine-tune BPE with domain corpus |
| Composite Tokenizer | Merges outputs from multiple tokenizers | Union of general + domain tokens |

**Key Strategies:**
- **Fine-tune BPE/Unigram** on domain-specific corpora to learn domain subword distributions
- **Domain-specific regex rules** for patterns like chemical formulas, legal citations, or code identifiers
- **Vocabulary extension** — add domain terms to a pre-trained tokenizer's vocab without full retraining
- **Confidence-based routing** — use classifier confidence to blend general and domain tokenizers
- **Online adaptation** — incrementally update tokenizer vocabulary as new domain data arrives

**Python Code Example:**
```python
# Pipeline: Input text -> Domain detection -> Route to domain tokenizer -> Tokens

from transformers import AutoTokenizer
from collections import defaultdict
import re

class AdaptiveTokenizer:
    def __init__(self):
        # Domain-specific tokenizers
        self.tokenizers = {
            'general': AutoTokenizer.from_pretrained('bert-base-uncased'),
            'biomedical': AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1'),
            'code': AutoTokenizer.from_pretrained('microsoft/codebert-base'),
        }
        # Domain detection patterns
        self.domain_patterns = {
            'biomedical': re.compile(r'\b(patient|diagnosis|mg/dL|symptom|clinical)\b', re.I),
            'code': re.compile(r'(def |class |import |function |=>|\{\})'),
        }
    
    def detect_domain(self, text: str) -> str:
        scores = {}
        for domain, pattern in self.domain_patterns.items():
            scores[domain] = len(pattern.findall(text))
        if max(scores.values(), default=0) >= 2:
            return max(scores, key=scores.get)
        return 'general'
    
    def tokenize(self, text: str) -> dict:
        domain = self.detect_domain(text)
        tokenizer = self.tokenizers[domain]
        tokens = tokenizer.tokenize(text)
        return {'domain': domain, 'tokens': tokens, 'ids': tokenizer.encode(text)}

# Usage
at = AdaptiveTokenizer()
result = at.tokenize("The patient presents with elevated mg/dL levels and symptoms of hyperglycemia.")
print(f"Domain: {result['domain']}, Tokens: {result['tokens'][:10]}")
```

**Interview Tips:**
- Emphasize that domain detection should be lightweight (regex or small classifier) to avoid adding latency
- Mention that pre-trained model tokenizers (BERT, GPT) are already quite general, so domain adaptation matters most for specialized vocabularies
- Discuss the trade-off: domain-specific tokenizers improve quality but increase system complexity and maintenance
- Note that vocabulary extension (adding special tokens) is simpler than retraining a full tokenizer

---

## Question 18
- [ ] Done
**What strategies help with tokenization of historical or archaic text varieties?**

**Answer:**

**Definition:**
Tokenizing historical or archaic text is challenging because such texts use non-standard spelling, obsolete vocabulary, different scripts or character sets, missing or inconsistent punctuation, and grammatical structures absent from modern training corpora. Effective strategies involve normalization, specialized vocabularies, and transfer learning from modern text adapted to historical forms.

**Core Concepts:**

| Challenge | Example | Solution |
|---|---|---|
| Spelling variation | "colour" vs "color", "ye" vs "the" | Spelling normalization maps |
| Obsolete vocabulary | "forsooth", "hitherto", "prithee" | Historical dictionary integration |
| Non-standard punctuation | Long-s (ſ), missing spaces, run-on text | Character normalization + regex |
| Script variations | Blackletter, runic, old orthography | Unicode normalization (NFKD) |
| OCR artifacts | Scanned historical documents with errors | Post-OCR correction pipeline |
| Merged/split words | "to day" → "today", "can not" → "cannot" | Compound word handling rules |

**Key Strategies:**
- **Spelling normalization** — map archaic forms to modern equivalents using dictionaries (e.g., VARD2 for Early Modern English)
- **Character normalization** — convert long-s (ſ) to 's', normalize Unicode variants via NFKD decomposition
- **Subword tokenization with historical fine-tuning** — train BPE on historical corpora to learn archaic subword patterns
- **OCR post-correction** — fix common OCR errors before tokenization (e.g., 'rn' misread as 'm')
- **Multi-stage pipeline** — normalize → correct → tokenize → optionally modernize tokens

**Python Code Example:**
```python
# Pipeline: Historical text -> Normalize characters -> Spelling normalization -> Tokenize

import unicodedata
import re

class HistoricalTokenizer:
    def __init__(self):
        # Archaic-to-modern spelling map
        self.spelling_map = {
            'ye': 'the', 'thou': 'you', 'thee': 'you',
            'hath': 'has', 'doth': 'does', 'art': 'are',
            'thy': 'your', 'thine': 'yours', 'whilst': 'while',
            'amongst': 'among', 'betwixt': 'between',
            'forsooth': 'indeed', 'prithee': 'please',
        }
        # Character normalization: long-s, ligatures
        self.char_map = {'ſ': 's', 'æ': 'ae', 'œ': 'oe', 'ꝑ': 'per'}
    
    def normalize_chars(self, text: str) -> str:
        # Unicode NFKD normalization
        text = unicodedata.normalize('NFKD', text)
        for old, new in self.char_map.items():
            text = text.replace(old, new)
        return text
    
    def normalize_spelling(self, tokens: list) -> list:
        return [self.spelling_map.get(t.lower(), t) for t in tokens]
    
    def tokenize(self, text: str) -> dict:
        # Step 1: Character normalization
        normalized = self.normalize_chars(text)
        # Step 2: Basic tokenization
        raw_tokens = re.findall(r"\b[\w']+\b|[^\w\s]", normalized)
        # Step 3: Spelling normalization
        modern_tokens = self.normalize_spelling(raw_tokens)
        return {'original': raw_tokens, 'modernized': modern_tokens}

# Usage
ht = HistoricalTokenizer()
result = ht.tokenize("Ye ſhall know that thou hath ſinned betwixt theſe walls.")
print(f"Original:  {result['original']}")
print(f"Modernized: {result['modernized']}")
```

**Interview Tips:**
- Mention real tools: VARD2 (spelling normalization), OCRopus/Tesseract for historical OCR
- Emphasize that normalization should be reversible — keep original text for provenance
- Note that BPE trained on modern text handles archaic words poorly because rare subwords get over-segmented
- Discuss the trade-off between full modernization (better for downstream NLP) vs preserving original forms (better for scholarly research)

---

## Question 19
- [ ] Done
**How do you handle tokenization quality control and error detection?**

**Answer:**

**Definition:**
Tokenization quality control (QC) involves systematically detecting, measuring, and correcting errors in the tokenization process. Errors include incorrect token boundaries, lost information, excessive fragmentation, vocabulary misses, and encoding issues. A robust QC pipeline monitors tokenization outputs with quantitative metrics and automated alerts to prevent bad tokens from degrading downstream model performance.

**Core Concepts:**

| QC Dimension | Metric | Red Flag |
|---|---|---|
| Token count ratio | tokens / words ratio | Ratio > 2.0 means excessive fragmentation |
| UNK token rate | % of [UNK] tokens | > 1-2% indicates vocab gaps |
| Roundtrip fidelity | decode(encode(text)) == text | Any mismatch = information loss |
| Token length distribution | Mean/std of token lengths | Spike in 1-char tokens = over-splitting |
| Coverage | % of text characters covered | < 99% means dropped characters |
| Encoding errors | Non-UTF8 or replacement chars | Any U+FFFD replacement character |

**Key Strategies:**
- **Roundtrip testing** — encode then decode; compare to original text for lossless fidelity
- **Statistical monitoring** — track token/word ratio, UNK rate, and token length distributions over time
- **Anomaly detection** — flag inputs where tokenization metrics deviate from historical baselines
- **Human-in-the-loop sampling** — periodically review random tokenization outputs for correctness
- **Domain-specific validation** — ensure critical terms (medical codes, legal terms) tokenize as expected

**Python Code Example:**
```python
# Pipeline: Tokenize -> Compute QC metrics -> Flag anomalies -> Report

from transformers import AutoTokenizer
from dataclasses import dataclass
import numpy as np

@dataclass
class TokenizationQCReport:
    text_length: int
    token_count: int
    token_word_ratio: float
    unk_rate: float
    roundtrip_match: bool
    avg_token_length: float
    flags: list

class TokenizationQC:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.unk_id = self.tokenizer.unk_token_id
    
    def check(self, text: str) -> TokenizationQCReport:
        flags = []
        words = text.split()
        encoding = self.tokenizer(text, add_special_tokens=False)
        token_ids = encoding['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Metrics
        token_word_ratio = len(token_ids) / max(len(words), 1)
        unk_count = token_ids.count(self.unk_id) if self.unk_id else 0
        unk_rate = unk_count / max(len(token_ids), 1)
        decoded = self.tokenizer.decode(token_ids)
        roundtrip_ok = decoded.strip() == text.lower().strip()  # BERT lowercases
        avg_len = np.mean([len(t.replace('##', '')) for t in tokens]) if tokens else 0
        
        # Flag anomalies
        if token_word_ratio > 2.0:
            flags.append(f"HIGH_FRAGMENTATION: ratio={token_word_ratio:.2f}")
        if unk_rate > 0.02:
            flags.append(f"HIGH_UNK_RATE: {unk_rate:.2%}")
        if not roundtrip_ok:
            flags.append("ROUNDTRIP_MISMATCH")
        if avg_len < 2.0:
            flags.append(f"SHORT_TOKENS: avg_len={avg_len:.1f}")
        
        return TokenizationQCReport(
            text_length=len(text), token_count=len(token_ids),
            token_word_ratio=token_word_ratio, unk_rate=unk_rate,
            roundtrip_match=roundtrip_ok, avg_token_length=avg_len, flags=flags
        )

# Usage
qc = TokenizationQC()
report = qc.check("Patient diagnosed with Pneumonoultramicroscopicsilicovolcanoconiosis yesterday.")
print(f"Tokens: {report.token_count}, Ratio: {report.token_word_ratio:.2f}, Flags: {report.flags}")
```

**Interview Tips:**
- Roundtrip fidelity testing (`decode(encode(x)) == x`) is the single most important QC check
- In production, log tokenization metrics per-request and set up alerting on UNK rate spikes
- Mention that subword tokenizers (BPE/WordPiece) almost never produce UNK, but can over-fragment rare words
- Note that QC should run both offline (batch validation) and online (per-request monitoring)

---

## Question 20
- [ ] Done
**What approaches work best for tokenization in federated learning scenarios?**

**Answer:**

**Definition:**
In federated learning (FL), model training occurs across decentralized clients (devices/organizations) without sharing raw data. Tokenization in FL must address a unique challenge: building or applying a consistent vocabulary across all clients without centralizing sensitive text. The tokenizer must be shared or synchronized while respecting data privacy, vocabulary heterogeneity across clients, and communication constraints.

**Core Concepts:**

| Approach | Description | Trade-off |
|---|---|---|
| Shared Pre-trained Tokenizer | All clients use the same frozen tokenizer (e.g., BERT) | Simple but may miss client-specific terms |
| Federated Vocabulary Construction | Clients share token frequency histograms, server merges | Better coverage but leaks frequency info |
| Differential Privacy (DP) Vocab | Add noise to frequency counts before sharing | Privacy-preserving but noisier vocabulary |
| Local Tokenizer + Adapter | Each client has local tokenizer; adapter maps to shared space | Flexible but adds complexity |
| Hash-based Tokenization | Feature hashing avoids explicit vocabulary sharing | No vocab sync needed but hash collisions |

**Key Strategies:**
- **Use pre-trained tokenizers** — simplest approach; distribute a fixed BPE/WordPiece tokenizer to all clients
- **Federated BPE** — clients compute local pair frequencies, send DP-noised frequencies to server, server runs BPE merge operations
- **Secure aggregation** — use cryptographic protocols to aggregate vocabulary statistics without revealing individual contributions
- **Vocabulary intersection/union** — build local vocabularies, then compute a global vocab via federated set operations
- **Hash embeddings** — replace vocabulary lookup with hashing to eliminate vocab synchronization entirely

**Python Code Example:**
```python
# Pipeline: Local vocab stats -> DP noise -> Aggregate -> Build global tokenizer

import numpy as np
from collections import Counter

class FederatedTokenizerBuilder:
    """Simulates federated vocabulary construction with differential privacy."""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # DP privacy budget
        self.client_stats = []
    
    def compute_local_stats(self, client_texts: list[str]) -> Counter:
        """Each client computes local token frequencies."""
        freq = Counter()
        for text in client_texts:
            freq.update(text.lower().split())
        return freq
    
    def add_dp_noise(self, freq: Counter) -> dict:
        """Add Laplace noise for differential privacy before sharing."""
        sensitivity = 1.0  # Each user contributes at most 1 to each count
        noised = {}
        for token, count in freq.items():
            noise = np.random.laplace(0, sensitivity / self.epsilon)
            noised_count = max(0, count + noise)
            noised[token] = noised_count
        return noised
    
    def aggregate(self, noised_stats: list[dict], min_freq: int = 5) -> list[str]:
        """Server aggregates noised stats to build global vocabulary."""
        global_freq = Counter()
        for stats in noised_stats:
            for token, count in stats.items():
                global_freq[token] += count
        # Filter by minimum frequency
        vocab = [t for t, c in global_freq.most_common() if c >= min_freq]
        return vocab

# Simulate 3 clients
builder = FederatedTokenizerBuilder(epsilon=1.0)
client_data = [
    ["the patient has fever and cough"] * 100,
    ["the model predicts label accurately"] * 100,
    ["the stock market rose sharply today"] * 100,
]

noised_stats = []
for data in client_data:
    local = builder.compute_local_stats(data)
    noised = builder.add_dp_noise(local)
    noised_stats.append(noised)

global_vocab = builder.aggregate(noised_stats, min_freq=10)
print(f"Global vocab size: {len(global_vocab)}, Sample: {global_vocab[:10]}")
```

**Interview Tips:**
- The simplest and most common FL approach is to just distribute a pre-trained tokenizer (BERT, GPT) — no federation needed for tokenization
- Federated vocab construction is only needed when domains are highly specialized (medical, legal) and a generic vocab is insufficient
- Mention differential privacy (DP) as the standard mechanism for protecting frequency statistics
- Highlight that hash-based approaches (feature hashing) eliminate the vocab sync problem entirely but sacrifice interpretability

---

## Question 21
- [ ] Done
**How do you implement efficient tokenization pipelines for large-scale text processing?**

**Answer:**

**Definition:**
Efficient large-scale tokenization pipelines process millions to billions of documents by combining parallelism, batching, memory-mapped I/O, compiled tokenizers, and streaming architectures. The goal is to maximize throughput (documents/second) while minimizing memory usage and latency. This is critical for pretraining data preparation, search engine indexing, and high-throughput inference systems.

**Core Concepts:**

| Technique | Speedup Factor | Use Case |
|---|---|---|
| HuggingFace `tokenizers` (Rust) | 10-100x vs pure Python | Any production tokenization |
| Batch tokenization | 5-20x vs single-doc | All batch workloads |
| Multiprocessing | ~Nx for N cores | CPU-bound tokenization |
| Memory-mapped files | Avoids loading full dataset in RAM | Datasets larger than RAM |
| Streaming/generator | Constant memory | Unbounded data streams |
| Pre-tokenization caching | Avoids recomputation | Repeated text (templates, common phrases) |
| Arrow/Parquet columnar | Fast serialization, zero-copy | Dataset storage and loading |

**Python Code Example:**
```python
# Pipeline: Large corpus -> Parallel batch tokenize -> Stream to disk (Arrow format)

from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial
import time

def tokenize_batch(texts: list[str], model_name: str) -> list[list[int]]:
    """Tokenize a batch of texts (runs in worker process)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = tokenizer(texts, truncation=True, max_length=512,
                        padding=False, return_attention_mask=False)
    return results['input_ids']

def parallel_tokenize(corpus: list[str], model_name: str = 'bert-base-uncased',
                      batch_size: int = 1000, num_workers: int = 4) -> list:
    """Tokenize large corpus using multiprocessing."""
    # Split corpus into batches
    batches = [corpus[i:i+batch_size] for i in range(0, len(corpus), batch_size)]
    
    worker_fn = partial(tokenize_batch, model_name=model_name)
    
    all_ids = []
    with Pool(num_workers) as pool:
        for batch_result in pool.imap(worker_fn, batches):
            all_ids.extend(batch_result)
    return all_ids

# For HuggingFace Datasets (best practice for large-scale)
def tokenize_with_datasets():
    from datasets import load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset('text', data_files='large_corpus.txt', streaming=True)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    # .map() handles batching, multiprocessing, and Arrow caching
    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    return tokenized

# Benchmark
corpus = ["This is a sample sentence for tokenization benchmarking."] * 100_000
start = time.time()
result = parallel_tokenize(corpus, num_workers=4)
print(f"Tokenized {len(corpus)} docs in {time.time()-start:.2f}s")
```

**Interview Tips:**
- Always mention HuggingFace `datasets.map(batched=True, num_proc=N)` — it's the industry standard for large-scale tokenization with automatic caching
- The Rust-based `tokenizers` library is 10-100x faster than pure Python tokenizers — it's what HuggingFace uses under the hood
- For truly massive scale (TB+), mention Apache Spark/Beam with UDF tokenizers or Ray for distributed tokenization
- Memory-mapped Arrow files let you tokenize once and reuse without recomputation — a huge efficiency win for iterative experimentation

---

## Question 22
- [ ] Done
**What techniques help with explaining tokenization decisions to end users?**

**Answer:**

**Definition:**
Tokenization explainability involves making the tokenization process transparent and understandable to end users, developers, and stakeholders. Since tokenization determines how text is broken down for model consumption, explaining why a word was split, merged, or mapped to [UNK] helps users debug model behavior, understand output quality, and build trust in NLP systems.

**Core Concepts:**

| Technique | Description | Audience |
|---|---|---|
| Visual token highlighting | Color-code tokens in original text | End users |
| Token-to-word alignment | Show which words map to which tokens | Developers |
| Merge tree visualization | Show BPE merge hierarchy | ML engineers |
| Vocabulary lookup display | Show if token is in-vocab vs subword-split | All audiences |
| Side-by-side comparison | Show tokenization from different tokenizers | Researchers |
| Interactive playground | Let users type text and see tokens in real time | All audiences |

**Key Strategies:**
- **Color-coded visualization** — highlight each token with a distinct color in the original text
- **Subword annotation** — show subword boundaries with markers (e.g., "un##believ##able")
- **Vocabulary membership indicators** — mark tokens as full-word vs subword vs unknown
- **Token statistics dashboard** — display token count, compression ratio, and fragmentation metrics
- **Counterfactual explanations** — "This word was split because it's not in the vocabulary of 30,522 tokens"

**Python Code Example:**
```python
# Pipeline: Text -> Tokenize -> Generate visual explanation -> Display

from transformers import AutoTokenizer

class TokenizationExplainer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def explain(self, text: str) -> dict:
        encoding = self.tokenizer(text, return_offsets_mapping=True,
                                   add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']
        
        explanations = []
        for token, (start, end) in zip(tokens, offsets):
            original_span = text[start:end]
            is_subword = token.startswith('##')
            is_unk = token == self.tokenizer.unk_token
            
            explanation = {
                'token': token,
                'original_text': original_span,
                'token_id': self.tokenizer.convert_tokens_to_ids(token),
                'type': 'UNKNOWN' if is_unk else ('SUBWORD' if is_subword else 'FULL_WORD'),
                'reason': self._get_reason(token, is_subword, is_unk)
            }
            explanations.append(explanation)
        
        return {
            'text': text,
            'num_tokens': len(tokens),
            'compression_ratio': len(text.split()) / len(tokens),
            'details': explanations
        }
    
    def _get_reason(self, token, is_subword, is_unk):
        if is_unk:
            return "Not in vocabulary — replaced with [UNK]"
        if is_subword:
            return f"Subword piece — parent word not in vocabulary, split into subwords"
        return "Full word found in vocabulary"
    
    def display(self, text: str):
        result = self.explain(text)
        print(f"Input: '{text}'")
        print(f"Tokens: {result['num_tokens']} | Compression: {result['compression_ratio']:.2f}")
        for d in result['details']:
            print(f"  [{d['type']:10s}] '{d['token']}' ← '{d['original_text']}' | {d['reason']}")

# Usage
explainer = TokenizationExplainer()
explainer.display("Transformers handle antidisestablishmentarianism gracefully.")
```

**Interview Tips:**
- `return_offsets_mapping=True` is the key HuggingFace feature for aligning tokens back to original text spans
- For user-facing products, visual highlighting (HTML/CSS) is far more effective than text-based explanations
- Mention tools like BertViz, LIT (Language Interpretability Tool) for interactive tokenization exploration
- Explain that tokenization transparency helps debug issues like: "why did my model fail on this input?" → often it's because a key term was split into meaningless subwords

---

## Question 23
- [ ] Done
**How do you handle tokenization for texts with special formatting or markup?**

**Answer:**

**Definition:**
Texts with special formatting or markup (HTML, XML, Markdown, LaTeX, JSON, code with comments) require tokenization strategies that distinguish between content text and structural elements. Naive tokenization treats markup tags as text, fragmenting them into meaningless tokens, polluting the vocabulary, and consuming sequence length. Proper handling involves stripping, preserving, or separately encoding structural elements.

**Core Concepts:**

| Format | Challenge | Strategy |
|---|---|---|
| HTML/XML | Tags become tokens (`<`, `div`, `>`) | Strip tags or use tag-aware tokenizer |
| Markdown | `**bold**`, `[links](url)` mixed with text | Parse markdown AST, tokenize text nodes |
| LaTeX | Math expressions: `$\frac{a}{b}$` | Separate math tokenizer or special tokens |
| JSON/YAML | Keys, values, structural chars mixed | Parse structure, tokenize values only |
| Rich text (RTF) | Embedded formatting codes | Strip formatting, keep plain text |
| Code + comments | Mixed natural language and code | Language-aware splitting |

**Key Strategies:**
- **Strip-then-tokenize** — remove all markup, tokenize clean text (simplest, lossy)
- **Parse-then-tokenize** — parse markup into AST, tokenize text nodes only, preserve structure separately
- **Special token replacement** — replace markup elements with placeholders (`[HTML_TAG]`, `[LATEX_MATH]`) before tokenization
- **Multi-stream tokenization** — separate content and markup into parallel streams, tokenize independently
- **Format-aware pre-tokenization** — use regex to identify markup boundaries before subword tokenization

**Python Code Example:**
```python
# Pipeline: Markup text -> Parse/strip formatting -> Tokenize content -> Reconstruct

import re
from html.parser import HTMLParser
from transformers import AutoTokenizer

class MarkupAwareTokenizer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Patterns for common markup elements
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'markdown_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'markdown_formatting': re.compile(r'[*_]{1,3}([^*_]+)[*_]{1,3}'),
            'latex_math': re.compile(r'\$[^$]+\$'),
        }
    
    def strip_and_track(self, text: str) -> dict:
        """Strip markup while tracking positions for reconstruction."""
        markup_elements = []
        clean_text = text
        
        # Extract and replace HTML tags
        for tag in self.patterns['html_tags'].finditer(text):
            markup_elements.append({'type': 'html', 'content': tag.group(),
                                     'start': tag.start(), 'end': tag.end()})
        clean_text = self.patterns['html_tags'].sub(' ', clean_text)
        
        # Extract markdown links (keep link text)
        clean_text = self.patterns['markdown_links'].sub(r'\1', clean_text)
        # Remove markdown formatting chars
        clean_text = self.patterns['markdown_formatting'].sub(r'\1', clean_text)
        # Replace LaTeX with placeholder
        clean_text = self.patterns['latex_math'].sub('[MATH]', clean_text)
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return {'clean_text': clean_text, 'markup': markup_elements}
    
    def tokenize(self, text: str) -> dict:
        parsed = self.strip_and_track(text)
        encoding = self.tokenizer(parsed['clean_text'], return_tensors='pt')
        tokens = self.tokenizer.tokenize(parsed['clean_text'])
        return {
            'original': text,
            'clean_text': parsed['clean_text'],
            'tokens': tokens,
            'num_markup_elements': len(parsed['markup']),
            'encoding': encoding
        }

# Usage
mat = MarkupAwareTokenizer()
html_text = "<p>The <b>quick</b> brown fox <a href='url'>jumped</a> over $E=mc^2$.</p>"
result = mat.tokenize(html_text)
print(f"Clean: {result['clean_text']}")
print(f"Tokens: {result['tokens']}")
```

**Interview Tips:**
- The right approach depends on whether markup carries semantic meaning — HTML tags usually don't, but LaTeX math does
- For code-based models (CodeBERT, StarCoder), markup is part of the content and should NOT be stripped
- Mention BeautifulSoup for robust HTML parsing, markdown-it for Markdown AST parsing
- Stripping markup reduces sequence length significantly — important when working with 512-token BERT limits
- Always preserve the original text alongside cleaned text for traceability

---

## Question 24
- [ ] Done
**What strategies work best for tokenization of multilingual documents?**

**Answer:**

**Definition:**
Multilingual tokenization handles documents containing multiple languages (code-switching), diverse scripts (Latin, CJK, Arabic, Devanagari), and varying morphological complexities within a single unified pipeline. The challenge is building or selecting a tokenizer that maintains consistent quality across all languages without over-fragmenting low-resource languages or under-segmenting morphologically rich ones.

**Core Concepts:**

| Challenge | Description | Solution |
|---|---|---|
| Script diversity | Latin, CJK, Arabic, Cyrillic in one doc | Unicode-aware pre-tokenization |
| Vocabulary imbalance | English dominates shared BPE vocab | Temperature-sampled vocab balancing |
| Code-switching | Languages mixed within sentences | Language-agnostic subword models |
| No whitespace | Chinese, Japanese, Thai lack word boundaries | Character/SentencePiece tokenization |
| Morphological variation | Turkish/Finnish have very long words | Aggressive subword segmentation |
| Script direction | RTL (Arabic/Hebrew) mixed with LTR | Bidirectional text handling |

**Key Strategies:**
- **SentencePiece with Unigram model** — language-agnostic, works directly on raw text without pre-tokenization assumptions
- **Balanced vocabulary sampling** — use temperature scaling when building vocab to prevent high-resource languages from dominating
- **Shared multilingual tokenizer** — XLM-RoBERTa's SentencePiece (250K vocab) covers 100+ languages
- **Language-specific pre-tokenization** — detect script/language first, apply appropriate word segmentation (Jieba for Chinese, MeCab for Japanese)
- **Vocab overlap monitoring** — ensure each language gets sufficient dedicated vocabulary tokens

**Python Code Example:**
```python
# Pipeline: Detect scripts -> Language-specific pre-tokenize -> Unified subword tokenize

import re
import unicodedata
from transformers import AutoTokenizer

class MultilingualTokenizer:
    def __init__(self):
        # XLM-R handles 100+ languages natively
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    def detect_scripts(self, text: str) -> dict:
        """Detect Unicode script blocks present in text."""
        scripts = {}
        for char in text:
            if char.isalpha():
                script = unicodedata.name(char, '').split()[0]
                scripts[script] = scripts.get(script, 0) + 1
        return scripts
    
    def analyze_tokenization(self, text: str) -> dict:
        tokens = self.tokenizer.tokenize(text)
        scripts = self.detect_scripts(text)
        
        # Compute per-language fragmentation
        words = text.split()
        fragmentation = len(tokens) / max(len(words), 1)
        
        return {
            'text': text,
            'scripts_detected': scripts,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'fragmentation_ratio': fragmentation,
        }
    
    def tokenize_with_language_tags(self, text: str, language: str = 'auto') -> dict:
        """Tokenize with language metadata preservation."""
        encoding = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=512
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'language': language,
            'num_tokens': encoding['input_ids'].shape[1]
        }

# Usage: multilingual document with code-switching
mt = MultilingualTokenizer()
mixed_text = "The meeting is at 3pm. 会议在三点开始。 La réunion est à 15h."
result = mt.analyze_tokenization(mixed_text)
print(f"Scripts: {result['scripts_detected']}")
print(f"Tokens ({result['num_tokens']}): {result['tokens'][:15]}")
```

**Interview Tips:**
- XLM-RoBERTa (SentencePiece, 250K vocab) is the go-to for multilingual tokenization — mention it first
- The key insight is that SentencePiece treats text as raw bytes/characters, making it truly language-agnostic
- Vocabulary imbalance is the #1 problem: English gets full words while low-resource languages get character-level fragmentation
- For CJK languages, each character is often a token — this is acceptable because characters carry meaning independently
- Mention the "fertility" metric (tokens per word) as a way to measure tokenizer fairness across languages

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving tokenization for sensitive text data?**

**Answer:**

**Definition:**
Privacy-preserving tokenization processes sensitive text (medical records, financial data, personal messages) while protecting personally identifiable information (PII) and confidential content. It combines NLP tokenization with privacy techniques such as redaction, anonymization, differential privacy, and secure computation to ensure tokens don't leak sensitive information while remaining useful for downstream models.

**Core Concepts:**

| Technique | Protection Level | Use Case |
|---|---|---|
| PII Redaction | Removes identifiers pre-tokenization | Medical/legal text processing |
| Named Entity Anonymization | Replaces entities with typed placeholders | "John Smith" → "[PERSON_1]" |
| Token-level DP | Adds noise to token embeddings | Federated/distributed training |
| Secure tokenization (MPC) | Cryptographic multi-party computation | Cross-org data collaboration |
| Hash-based tokens | One-way hash replaces raw tokens | Search/matching without raw text |
| K-anonymity tokenization | Generalize rare tokens to prevent re-identification | Publishing anonymized datasets |

**Key Strategies:**
- **Pre-tokenization PII detection and redaction** — use NER to detect names, emails, SSNs, then replace with placeholders before tokenization
- **Consistent anonymization** — map same entity to same placeholder across document for coreference preservation
- **Vocabulary filtering** — remove tokens that appear fewer than k times to prevent rare-token re-identification
- **Token-level differential privacy** — add calibrated noise to token embeddings during training
- **On-device tokenization** — tokenize on user's device, send only token IDs (not raw text) to server

**Python Code Example:**
```python
# Pipeline: Raw text -> PII detection -> Anonymize -> Tokenize -> Secure output

import re
import hashlib
from transformers import AutoTokenizer
from typing import Optional

class PrivacyPreservingTokenizer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # PII patterns
        self.pii_patterns = {
            'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'PHONE': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'IP_ADDR': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }
        self._entity_map = {}  # Consistent anonymization mapping
    
    def detect_and_redact(self, text: str) -> dict:
        """Detect PII and replace with typed placeholders."""
        redacted = text
        found_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            for match in pattern.finditer(redacted):
                original = match.group()
                # Consistent mapping: same PII -> same placeholder
                if original not in self._entity_map:
                    idx = sum(1 for k, v in self._entity_map.items()
                             if v.startswith(f'[{pii_type}'))
                    self._entity_map[original] = f'[{pii_type}_{idx}]'
                placeholder = self._entity_map[original]
                found_pii.append({'type': pii_type, 'placeholder': placeholder})
                redacted = redacted.replace(original, placeholder)
        
        return {'redacted_text': redacted, 'pii_found': found_pii}
    
    def tokenize_private(self, text: str) -> dict:
        """Full privacy-preserving tokenization pipeline."""
        # Step 1: Detect and redact PII
        redaction = self.detect_and_redact(text)
        # Step 2: Tokenize redacted text
        tokens = self.tokenizer.tokenize(redaction['redacted_text'])
        encoding = self.tokenizer(redaction['redacted_text'], return_tensors='pt')
        
        return {
            'original_pii_count': len(redaction['pii_found']),
            'redacted_text': redaction['redacted_text'],
            'tokens': tokens,
            'encoding': encoding
        }

# Usage
ppt = PrivacyPreservingTokenizer()
text = "Contact John at john@example.com or 555-123-4567. SSN: 123-45-6789"
result = ppt.tokenize_private(text)
print(f"Redacted: {result['redacted_text']}")
print(f"PII found: {result['original_pii_count']}")
print(f"Tokens: {result['tokens']}")
```

**Interview Tips:**
- PII redaction BEFORE tokenization is critical — once PII is tokenized, subword pieces may still leak information
- Consistent anonymization (same person → same placeholder) preserves coreference relationships needed for downstream NLP
- Mention real tools: Presidio (Microsoft), AWS Comprehend PII detection, spaCy NER for entity-based redaction
- For production systems, combine regex patterns (structured PII) with NER models (unstructured PII like names)
- On-device tokenization is the strongest privacy guarantee — the server never sees raw text

---

## Question 26
- [ ] Done
**What approaches help with tokenization of streaming text in real-time applications?**

**Answer:**

**Definition:**
Streaming tokenization processes text incrementally as it arrives — character by character, chunk by chunk, or line by line — without waiting for the complete document. This is essential for real-time applications like live transcription, chatbots, social media monitoring, and log analysis where latency matters and input is unbounded. The key challenge is handling token boundaries that may span across chunk boundaries.

**Core Concepts:**

| Challenge | Description | Solution |
|---|---|---|
| Incomplete tokens | Chunk ends mid-word/mid-token | Buffer incomplete tokens across chunks |
| Boundary ambiguity | Can't determine token boundary without future context | Lookahead buffer or delayed emission |
| Stateful processing | Must maintain state between chunks | Incremental tokenizer with state |
| Backpressure | Tokenizer slower than input stream | Async processing with queue |
| Ordering guarantees | Parallel processing may reorder | Sequence numbering |
| Memory management | Unbounded stream can't be stored | Sliding window / circular buffer |

**Key Strategies:**
- **Buffered chunking** — accumulate text until a clear boundary (sentence/paragraph), then tokenize
- **Incremental tokenization** — process each new character/word, maintain state for incomplete tokens
- **Sliding window** — tokenize overlapping windows to handle boundary tokens
- **Async pipeline** — use asyncio/queues to decouple text arrival from tokenization processing
- **Sentence boundary detection** — split stream at sentence boundaries for natural tokenization units

**Python Code Example:**
```python
# Pipeline: Text stream -> Buffer -> Sentence split -> Batch tokenize -> Emit tokens

import asyncio
from collections import deque
from transformers import AutoTokenizer

class StreamingTokenizer:
    def __init__(self, model_name: str = 'bert-base-uncased', buffer_size: int = 1000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.buffer = ""
        self.buffer_size = buffer_size
        self.sentence_endings = {'.', '!', '?', '\n'}
    
    def _find_split_point(self, text: str) -> int:
        """Find the last sentence boundary for safe splitting."""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in self.sentence_endings:
                return i + 1
        return -1  # No safe split point found
    
    def feed(self, chunk: str) -> list[dict]:
        """Feed a text chunk, return any complete tokenized segments."""
        self.buffer += chunk
        results = []
        
        while len(self.buffer) >= self.buffer_size:
            split_idx = self._find_split_point(self.buffer[:self.buffer_size])
            if split_idx == -1:
                # No sentence boundary; force split at word boundary
                split_idx = self.buffer[:self.buffer_size].rfind(' ')
                if split_idx == -1:
                    break  # Buffer has one huge word; wait for more
            
            segment = self.buffer[:split_idx]
            self.buffer = self.buffer[split_idx:]
            
            tokens = self.tokenizer.tokenize(segment)
            ids = self.tokenizer.encode(segment, add_special_tokens=False)
            results.append({'text': segment.strip(), 'tokens': tokens, 'ids': ids})
        
        return results
    
    def flush(self) -> list[dict]:
        """Process remaining buffer content."""
        if self.buffer.strip():
            tokens = self.tokenizer.tokenize(self.buffer)
            ids = self.tokenizer.encode(self.buffer, add_special_tokens=False)
            result = [{'text': self.buffer.strip(), 'tokens': tokens, 'ids': ids}]
            self.buffer = ""
            return result
        return []

# Usage: simulate streaming input
streamer = StreamingTokenizer(buffer_size=50)
chunks = [
    "The quick brown fox ",
    "jumped over the lazy dog. ",
    "Machine learning models ",
    "require tokenized input. ",
    "This is the final chunk."
]

for chunk in chunks:
    results = streamer.feed(chunk)
    for r in results:
        print(f"Emitted {len(r['tokens'])} tokens: {r['tokens']}")

# Flush remaining
for r in streamer.flush():
    print(f"Flushed {len(r['tokens'])} tokens: {r['tokens']}")
```

**Interview Tips:**
- The critical insight is that you must buffer incomplete tokens — splitting mid-word breaks subword tokenization
- Sentence boundaries are the natural safe split points for streaming tokenization
- For chat/LLM applications, tokenization is incremental by design (token-by-token generation)
- Mention Kafka/Flink for production streaming NLP pipelines at scale
- Latency vs completeness trade-off: smaller buffers = lower latency but more boundary issues

---

## Question 27
- [ ] Done
**How do you handle tokenization adaptation to emerging text formats and platforms?**

**Answer:**

**Definition:**
Emerging text formats from new platforms (TikTok, Discord, Threads, AI assistants), communication styles (memes, reactions, voice-to-text), and data types (multimodal captions, structured prompts) continuously challenge existing tokenizers. Adaptation requires a systematic approach to detect new patterns, extend vocabularies, add preprocessing rules, and update pipelines without breaking backward compatibility.

**Core Concepts:**

| Emerging Format | Example | Tokenization Challenge |
|---|---|---|
| Short-form social (TikTok) | "its giving main character energy" | Slang, no punctuation, platform jargon |
| Discord/chat | "LMAO :skull: ratio + L" | Emojis as words, custom emoji codes |
| AI prompts | "Act as a... Think step by step" | Structured instructions, role markers |
| Voice-to-text | "um like you know what I mean" | Disfluencies, no punctuation |
| Multimodal captions | "[IMAGE] A cat sitting on..." | Mixed modality markers |
| Thread/long-form | Numbered lists, nested quotes | Structural formatting mixed with text |

**Key Strategies:**
- **Continuous vocabulary monitoring** — track OOV/fragmentation rates on new data; spike = vocabulary drift
- **Modular pre-processing rules** — add platform-specific normalizers as pluggable components
- **Vocabulary extension without retraining** — add new special tokens to existing tokenizers via `add_tokens()`
- **Pattern registry** — maintain a regex registry for new text patterns; update when new formats emerge
- **A/B testing tokenizers** — compare existing vs updated tokenizer on downstream task metrics before deployment

**Python Code Example:**
```python
# Pipeline: Detect platform -> Apply platform normalizer -> Extend vocab -> Tokenize

import re
from transformers import AutoTokenizer

class AdaptiveFormatTokenizer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._add_platform_tokens()
        
        # Platform-specific normalizers
        self.normalizers = {
            'discord': self._normalize_discord,
            'tiktok': self._normalize_tiktok,
            'voice': self._normalize_voice,
        }
    
    def _add_platform_tokens(self):
        """Extend vocabulary with platform-specific tokens."""
        new_tokens = [
            '[EMOJI]', '[IMAGE]', '[VIDEO]', '[LINK]', '[MENTION]',
            '[HASHTAG]', '[REACTION]', '[VOICE_NOTE]', '[STICKER]',
        ]
        self.tokenizer.add_tokens(new_tokens, special_tokens=True)
    
    def _normalize_discord(self, text: str) -> str:
        # Replace custom emojis: <:name:id> -> [EMOJI]
        text = re.sub(r'<:[a-zA-Z0-9_]+:\d+>', '[EMOJI]', text)
        # Replace mentions: <@id> -> [MENTION]
        text = re.sub(r'<@!?\d+>', '[MENTION]', text)
        return text
    
    def _normalize_tiktok(self, text: str) -> str:
        # Replace hashtags: #word -> [HASHTAG] word
        text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
        # Common abbreviations
        slang = {'ngl': 'not gonna lie', 'fr': 'for real', 'imo': 'in my opinion'}
        for abbr, full in slang.items():
            text = re.sub(rf'\b{abbr}\b', full, text, flags=re.I)
        return text
    
    def _normalize_voice(self, text: str) -> str:
        # Remove disfluencies
        text = re.sub(r'\b(um|uh|er|like|you know)\b', '', text, flags=re.I)
        return re.sub(r'\s+', ' ', text).strip()
    
    def tokenize(self, text: str, platform: str = 'auto') -> dict:
        if platform in self.normalizers:
            text = self.normalizers[platform](text)
        tokens = self.tokenizer.tokenize(text)
        return {'normalized': text, 'tokens': tokens, 'platform': platform}

# Usage
aft = AdaptiveFormatTokenizer()

result = aft.tokenize("ngl this is giving main character energy #fyp #viral", platform='tiktok')
print(f"TikTok: {result['normalized']} -> {result['tokens'][:10]}")

result = aft.tokenize("LMAO <:pepe:12345> <@987654> ratio", platform='discord')
print(f"Discord: {result['normalized']} -> {result['tokens']}")
```

**Interview Tips:**
- `tokenizer.add_tokens()` is the simplest way to extend a pre-trained tokenizer for new formats — no retraining needed
- Emphasize monitoring: track UNK rates and fragmentation on fresh data to detect when adaptation is needed
- Platform normalization should be a pluggable module so new platforms can be added without modifying core tokenization
- Mention that LLM tokenizers (GPT-4, Llama) already handle diverse text well because they're trained on internet-scale data including social media

---

## Question 28
- [ ] Done
**What techniques work best for tokenization with minimal computational resources?**

**Answer:**

**Definition:**
Resource-constrained tokenization targets environments with limited CPU, memory, or power — such as mobile devices, IoT sensors, edge servers, or embedded systems. The goal is to achieve acceptable tokenization quality while minimizing compute time, memory footprint, and energy consumption. This involves using lightweight algorithms, smaller vocabularies, compiled tokenizers, and caching strategies.

**Core Concepts:**

| Technique | Memory | Speed | Quality |
|---|---|---|
| Whitespace + regex splitting | < 1 MB | Very fast | Low (no subword) |
| Pre-compiled vocabulary lookup | 5-50 MB | Fast | Medium |
| Minimal BPE (small vocab) | 10-50 MB | Fast | Good |
| ONNX-exported tokenizer | 5-20 MB | Very fast | Full quality |
| Character-level tokenization | < 1 MB | Fast | Low (long sequences) |
| Cached/memoized tokenizer | +cache MB | Fast (cache hits) | Same as base |
| Hash-based tokenization | < 5 MB | Very fast | Medium (collisions) |

**Key Strategies:**
- **Smaller vocabulary** — reduce vocab from 30K+ to 8-16K tokens; trades quality for speed/memory
- **Trie-based lookup** — compile vocabulary into a trie for O(n) tokenization without regex overhead
- **Pre-tokenized lookup tables** — cache tokenization of the most common words/phrases
- **Byte-level fallback** — use byte-level BPE (like GPT-2) which needs no pre-tokenization rules
- **Quantized/compiled models** — export tokenizer to ONNX or use Rust `tokenizers` for native speed

**Python Code Example:**
```python
# Pipeline: Text -> Lightweight trie-based tokenizer -> Token IDs (minimal memory)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_id = None

class LightweightTokenizer:
    """Memory-efficient trie-based tokenizer for resource-constrained environments."""
    
    def __init__(self, vocab: dict[str, int]):
        self.root = TrieNode()
        self.unk_id = 0
        self._build_trie(vocab)
    
    def _build_trie(self, vocab: dict[str, int]):
        for token, token_id in vocab.items():
            node = self.root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.token_id = token_id
    
    def tokenize(self, text: str) -> list[int]:
        """Greedy longest-match tokenization using trie."""
        ids = []
        i = 0
        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue
            
            # Find longest matching token
            node = self.root
            last_match = None
            last_match_end = i
            
            for j in range(i, len(text)):
                char = text[j].lower()
                if char not in node.children:
                    break
                node = node.children[char]
                if node.token_id is not None:
                    last_match = node.token_id
                    last_match_end = j + 1
            
            if last_match is not None:
                ids.append(last_match)
                i = last_match_end
            else:
                ids.append(self.unk_id)  # Unknown character
                i += 1
        
        return ids

# Build a small vocabulary
vocab = {'[UNK]': 0, 'the': 1, 'quick': 2, 'brown': 3, 'fox': 4,
         'jump': 5, 'ed': 6, 'over': 7, 'lazy': 8, 'dog': 9,
         'machine': 10, 'learn': 11, 'ing': 12}

tokenizer = LightweightTokenizer(vocab)
ids = tokenizer.tokenize("the quick brown fox jumped over the lazy dog")
print(f"Token IDs: {ids}")

import sys
print(f"Tokenizer memory: ~{sys.getsizeof(vocab)} bytes (vocab dict only)")
```

**Interview Tips:**
- For mobile/edge, recommend HuggingFace `tokenizers` (Rust) compiled as a shared library — it's already optimized for speed and memory
- Vocabulary size is the primary knob: 8K vocab uses ~4x less memory than 32K but increases sequence lengths
- LRU caching of frequent tokenizations is the easiest optimization with the biggest impact
- Mention ONNX Runtime for exporting tokenizers that run natively on mobile (iOS/Android)
- Character-level tokenization is the ultimate minimal option but produces very long sequences (bad for transformers)

---

## Question 29
- [ ] Done
**How do you implement robust error handling for tokenization in production systems?**

**Answer:**

**Definition:**
Production tokenization must handle every possible input gracefully — including malformed text, encoding errors, extremely long inputs, empty strings, adversarial inputs, and unexpected formats — without crashing or producing silently incorrect results. Robust error handling combines input validation, graceful degradation, fallback tokenizers, comprehensive logging, and circuit breaker patterns to ensure 24/7 reliability.

**Core Concepts:**

| Error Type | Example | Handling Strategy |
|---|---|---|
| Encoding errors | Invalid UTF-8 bytes, mixed encodings | Detect and re-encode with `errors='replace'` |
| Empty/null input | None, "", whitespace-only | Return empty token list, don't crash |
| Extremely long text | 1M+ character input | Truncate with warning or chunk |
| Adversarial input | Zalgo text, Unicode exploits | Unicode normalization + length limits |
| Memory exhaustion | Vocab + input exceeds RAM | Streaming tokenization, memory limits |
| Tokenizer model missing | Model files deleted/corrupted | Fallback tokenizer + alert |
| Timeout | Tokenization takes too long | Deadline enforcement + circuit breaker |

**Key Strategies:**
- **Input validation layer** — check encoding, length, and content before tokenization
- **Fallback chain** — primary tokenizer → backup tokenizer → whitespace split (never crash)
- **Timeout enforcement** — kill tokenization if it exceeds SLA (e.g., 100ms)
- **Circuit breaker pattern** — if error rate spikes, temporarily switch to fallback tokenizer
- **Structured logging** — log every error with input hash (not raw PII), error type, and stack trace

**Python Code Example:**
```python
# Pipeline: Input validation -> Try primary tokenizer -> Fallback -> Log errors

import logging
import time
import unicodedata
from transformers import AutoTokenizer
from dataclasses import dataclass, field

logger = logging.getLogger('tokenizer')

@dataclass
class TokenizationResult:
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    success: bool = True
    fallback_used: bool = False
    error: str = None

class RobustTokenizer:
    def __init__(self, model_name: str = 'bert-base-uncased',
                 max_length: int = 100_000, timeout_ms: int = 500):
        self.max_length = max_length
        self.timeout_ms = timeout_ms
        try:
            self.primary = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load primary tokenizer: {e}")
            self.primary = None
    
    def _validate_input(self, text) -> str:
        """Validate and sanitize input text."""
        if text is None:
            raise ValueError("Input text is None")
        if not isinstance(text, str):
            text = str(text)
        # Fix encoding issues
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        # Unicode normalization (handles Zalgo text)
        text = unicodedata.normalize('NFC', text)
        # Length check
        if len(text) > self.max_length:
            logger.warning(f"Input truncated from {len(text)} to {self.max_length} chars")
            text = text[:self.max_length]
        return text
    
    def _fallback_tokenize(self, text: str) -> TokenizationResult:
        """Simple whitespace tokenizer as last resort."""
        tokens = text.split()[:512]
        return TokenizationResult(tokens=tokens, token_ids=list(range(len(tokens))),
                                   success=True, fallback_used=True)
    
    def tokenize(self, text) -> TokenizationResult:
        # Step 1: Validate input
        try:
            text = self._validate_input(text)
        except ValueError as e:
            return TokenizationResult(success=False, error=str(e))
        
        if not text.strip():
            return TokenizationResult()  # Empty input -> empty result
        
        # Step 2: Try primary tokenizer
        if self.primary:
            try:
                start = time.monotonic()
                tokens = self.primary.tokenize(text)
                ids = self.primary.encode(text, add_special_tokens=False)
                elapsed_ms = (time.monotonic() - start) * 1000
                
                if elapsed_ms > self.timeout_ms:
                    logger.warning(f"Tokenization slow: {elapsed_ms:.0f}ms")
                
                return TokenizationResult(tokens=tokens, token_ids=ids)
            except Exception as e:
                logger.error(f"Primary tokenizer failed: {e}")
        
        # Step 3: Fallback
        logger.warning("Using fallback tokenizer")
        return self._fallback_tokenize(text)

# Usage
rt = RobustTokenizer()

# Normal input
print(rt.tokenize("Hello world"))
# Null input
print(rt.tokenize(None))
# Empty input
print(rt.tokenize("   "))
# Adversarial input (Zalgo text)
print(rt.tokenize("H\u0336e\u0336l\u0336l\u0336o\u0336"))
```

**Interview Tips:**
- The fallback chain (primary → backup → whitespace) is the most critical pattern — production systems must never crash on tokenization
- Always validate encoding first — `encode('utf-8', errors='replace')` handles the majority of encoding issues
- Unicode normalization (NFC/NFKC) prevents Zalgo text and homoglyph attacks from breaking tokenization
- Log input hashes (not raw text) for debugging to avoid PII exposure in logs
- Set hard limits on input length to prevent memory exhaustion — this is a common DoS vector

---

## Question 30
- [ ] Done
**What strategies help with combining tokenization with other text preprocessing tasks?**

**Answer:**

**Definition:**
Combining tokenization with other preprocessing tasks (normalization, stopword removal, stemming/lemmatization, POS tagging, NER, spell correction) into a unified pipeline improves efficiency, consistency, and maintainability. The order of operations matters critically — some steps must precede tokenization (encoding normalization) while others depend on tokenization output (POS tagging, NER). A well-designed combined pipeline avoids redundant text passes and ensures each step's output is compatible with the next.

**Core Concepts:**

| Pipeline Stage | Order | Purpose | Dependency |
|---|---|---|---|
| Encoding normalization | 1 (pre-tokenize) | Fix encoding, normalize Unicode | None |
| Text cleaning | 2 (pre-tokenize) | Remove noise, fix formatting | Encoding |
| Tokenization | 3 (core) | Split text into tokens | Clean text |
| Normalization | 4 (post-tokenize) | Lowercase, accent removal | Tokens |
| Stopword removal | 5 (post-tokenize) | Remove function words | Tokens |
| Stemming/Lemmatization | 6 (post-tokenize) | Reduce to base forms | Tokens (+ POS for lemma) |
| POS Tagging | 6 (post-tokenize) | Grammatical tags | Tokens |
| NER | 7 (post-tokenize) | Named entity detection | Tokens + POS |

**Key Strategies:**
- **Pipeline pattern** — chain processing steps as composable, reusable functions
- **Lazy evaluation** — only execute pipeline steps that downstream tasks actually need
- **Single-pass processing** — combine compatible operations (e.g., tokenize + lowercase in one pass) to avoid multiple text scans
- **spaCy-style pipeline** — use spaCy's `nlp.pipe()` which runs tokenization + all NLP in one optimized pass
- **Configuration-driven** — enable/disable steps via config rather than code changes

**Python Code Example:**
```python
# Pipeline: Raw text -> Clean -> Tokenize -> Normalize -> Lemmatize -> Filter -> Output

import re
import unicodedata
from dataclasses import dataclass, field

@dataclass
class ProcessedText:
    original: str = ""
    cleaned: str = ""
    tokens: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)

class TextPreprocessingPipeline:
    def __init__(self, config: dict = None):
        self.config = config or {
            'lowercase': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'min_token_length': 2,
        }
        self.stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in',
                          'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but',
                          'it', 'this', 'that', 'with', 'as', 'by', 'from'}
    
    def clean(self, text: str) -> str:
        """Stage 1: Encoding fix + text cleaning."""
        text = unicodedata.normalize('NFC', text)
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    def tokenize(self, text: str) -> list[str]:
        """Stage 2: Tokenization with optional lowercasing."""
        tokens = re.findall(r"\b\w+\b", text)
        if self.config.get('lowercase'):
            tokens = [t.lower() for t in tokens]
        return tokens
    
    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Stage 3: Simple suffix-based lemmatization."""
        lemma_rules = [
            (r'(\w+)ies$', r'\1y'), (r'(\w+)es$', r'\1e'),
            (r'(\w+)s$', r'\1'), (r'(\w+)ing$', r'\1'),
            (r'(\w+)ed$', r'\1'),
        ]
        lemmas = []
        for token in tokens:
            lemma = token
            for pattern, replacement in lemma_rules:
                result = re.sub(pattern, replacement, token)
                if result != token:
                    lemma = result
                    break
            lemmas.append(lemma)
        return lemmas
    
    def filter_tokens(self, tokens: list[str]) -> list[str]:
        """Stage 4: Remove stopwords and short tokens."""
        min_len = self.config.get('min_token_length', 2)
        result = tokens
        if self.config.get('remove_stopwords'):
            result = [t for t in result if t not in self.stopwords]
        result = [t for t in result if len(t) >= min_len]
        return result
    
    def process(self, text: str) -> ProcessedText:
        """Run full pipeline."""
        result = ProcessedText(original=text)
        result.cleaned = self.clean(text)
        result.tokens = self.tokenize(result.cleaned)
        result.lemmas = self.lemmatize(result.tokens) if self.config.get('lemmatize') else result.tokens
        result.filtered = self.filter_tokens(result.lemmas)
        return result
    
    def process_batch(self, texts: list[str]) -> list[ProcessedText]:
        return [self.process(t) for t in texts]

# Usage
pipeline = TextPreprocessingPipeline()
result = pipeline.process(
    "The <b>researchers</b> are studying advanced tokenization techniques "
    "for NLP applications. Visit https://example.com for details."
)
print(f"Cleaned:  {result.cleaned}")
print(f"Tokens:   {result.tokens}")
print(f"Lemmas:   {result.lemmas}")
print(f"Filtered: {result.filtered}")
```

**Interview Tips:**
- Pipeline ordering matters critically: encoding fix → cleaning → tokenization → normalization → linguistic processing
- spaCy's `nlp.pipe()` is the production standard — it runs tokenization + POS + NER + lemmatization in a single optimized pass
- Mention that for transformer models, you typically skip traditional preprocessing (stopwords, stemming) since the model handles this internally
- Configuration-driven pipelines let you easily A/B test different preprocessing combinations
- For production, make each pipeline step independently testable and add metrics (token count, processing time) at each stage

---

## Question 31
- [ ] Done
**How do you handle tokenization for texts requiring high accuracy for downstream tasks?**

**Answer:**

**Definition:**
High-accuracy tokenization for downstream tasks involves selecting and fine-tuning tokenization strategies that minimize information loss and maximize alignment between token boundaries and the semantic units the model needs to learn. The goal is to ensure tokenization does not introduce errors that propagate through classification, NER, QA, or generation pipelines.

**Core Concepts — Accuracy-Critical Tokenization Strategies:**

| Strategy | How It Helps Accuracy | Best For |
|---|---|---|
| Task-aligned tokenizer | Match tokenizer to model's pretraining | Any transformer-based task |
| Domain-adapted BPE/Unigram | Reduce fertility on domain text | Medical, legal, scientific NLP |
| Pre-tokenization rules | Protect entities, numbers, codes from splitting | NER, information extraction |
| Subword regularization | Expose model to multiple segmentations during training | Robust translation, generation |
| Post-tokenization alignment | Map token spans back to original chars | Span-level tasks (NER, QA) |

**Key Techniques:**
1. **Tokenizer-model alignment** — Always use the exact tokenizer the pretrained model was trained with. Mismatched tokenizers cause vocabulary misalignment and degrade accuracy.
2. **Subword regularization (BPE-dropout / Unigram sampling)** — During training, randomly sample different segmentations to make the model robust to tokenization variance.
3. **Pre-tokenization protection** — Use regex rules to prevent splitting of entities, URLs, numbers, and domain terms before subword tokenization.
4. **Offset mapping** — Maintain character-to-token alignment for span prediction tasks (NER, extractive QA) using `return_offsets_mapping=True`.

**Python Code Example:**
```python
# Pipeline: High-accuracy tokenization with alignment and subword regularization
from transformers import AutoTokenizer
import re

text = "Patient ID: P-12345 diagnosed with COVID-19 on 2024-01-15."

# 1. Use model-aligned tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Pre-tokenization: protect critical patterns
def protect_patterns(text):
    """Shield entities from subword splitting."""
    patterns = {
        r'P-\d+': lambda m: m.group().replace('-', '_'),      # Patient IDs
        r'\d{4}-\d{2}-\d{2}': lambda m: m.group().replace('-', '_'),  # Dates
        r'COVID-19': lambda m: 'COVID19',                      # Known entities
    }
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    return text

protected = protect_patterns(text)
print(f"Protected: {protected}")

# 3. Tokenize with offset mapping for span alignment
encoding = tokenizer(protected, return_offsets_mapping=True, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
offsets = encoding["offset_mapping"][0].tolist()

print(f"\nTokens with offsets:")
for tok, (start, end) in zip(tokens, offsets):
    if start != end:  # Skip special tokens
        print(f"  {tok:15s} -> chars [{start}:{end}] = '{protected[start:end]}'")

# 4. Fertility check (accuracy proxy)
base_tokens = tokenizer.tokenize(text)
print(f"\nFertility: {len(base_tokens)} tokens for {len(text.split())} words")
print(f"Tokens/word ratio: {len(base_tokens)/len(text.split()):.2f}")
```

**Interview Tips:**
- Emphasize that tokenizer-model mismatch is the #1 silent accuracy killer in production NLP
- Mention subword regularization (BPE-dropout) as a training-time technique to improve robustness
- Offset mapping is critical for span-level tasks — without it, you can't map predictions back to original text
- Fertility (tokens per word) is a quick proxy for tokenization quality on domain text
- Pre-tokenization rules should be deterministic and reversible to avoid data corruption

---

## Question 32
- [ ] Done
**What approaches work best for tokenization of conversational or dialogue text?**

**Answer:**

**Definition:**
Conversational text tokenization addresses the unique challenges of dialogue data: informal language, speaker turns, emoticons/emoji, contractions, slang, disfluencies ("um", "uh"), and non-standard punctuation. Standard tokenizers trained on formal text often fail on these patterns, requiring specialized preprocessing and tokenization strategies.

**Core Concepts — Dialogue Tokenization Challenges:**

| Challenge | Example | Standard Tokenizer Failure | Solution |
|---|---|---|---|
| Contractions/slang | "gonna", "wanna", "y'all" | Splits incorrectly or marks as UNK | Custom normalization rules |
| Emoticons/emoji | ":)", "😂", "<3" | Dropped or split | Protected token patterns |
| Speaker turns | "User: hi\nBot: hello" | No turn awareness | Turn-aware pre-tokenization |
| Disfluencies | "I um want to uh go" | Treated as normal words | Disfluency detection + filtering |
| Code-switching | "That's so kawaii desu" | Mixed language confusion | Multilingual tokenizer |
| Repeated chars | "sooooo gooood" | OOV or over-segmented | Normalization before tokenization |

**Key Approaches:**
1. **Normalization pipeline** — Standardize contractions, reduce repeated characters, normalize Unicode emoji before tokenization.
2. **Turn-aware tokenization** — Parse speaker labels and turn boundaries as structural metadata, not regular text.
3. **Emoticon/emoji preservation** — Add emoji and emoticon patterns to the tokenizer vocabulary or protect them via regex.
4. **Dialogue-pretrained models** — Use models pretrained on conversational data (DialoGPT, BlenderBot) whose tokenizers already handle dialogue patterns.

**Python Code Example:**
```python
# Pipeline: Dialogue text normalization -> tokenization -> turn parsing
import re
from transformers import AutoTokenizer

dialogue = """User: heyyy!! how r u doing?? 😊😊
Bot: I'm doing great, thanks! How can I help you today?
User: i wanna know about ur return policy... its soooo confusing :(
Bot: I'd be happy to help with that! 🎉"""

# Step 1: Dialogue-specific normalization
def normalize_dialogue(text):
    """Clean conversational text while preserving meaning."""
    # Reduce repeated characters (3+ -> 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Normalize common contractions/slang
    slang_map = {
        r"\br\b": "are", r"\bu\b": "you", r"\bur\b": "your",
        r"\bwanna\b": "want to", r"\bgonna\b": "going to",
        r"\bits\b": "it's", r"\bthx\b": "thanks",
    }
    for pattern, replacement in slang_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Normalize repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text

# Step 2: Parse turns
def parse_turns(dialogue_text):
    """Extract speaker turns with metadata."""
    turns = []
    for line in dialogue_text.strip().split('\n'):
        match = re.match(r'^(\w+):\s*(.+)$', line)
        if match:
            turns.append({'speaker': match.group(1), 'text': match.group(2)})
    return turns

turns = parse_turns(dialogue)
normalized_turns = [{'speaker': t['speaker'], 'text': normalize_dialogue(t['text'])}
                    for t in turns]

# Step 3: Tokenize with dialogue-aware model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

for orig, norm in zip(turns, normalized_turns):
    orig_tokens = tokenizer.tokenize(orig['text'])
    norm_tokens = tokenizer.tokenize(norm['text'])
    print(f"[{orig['speaker']}] Original ({len(orig_tokens)} tokens): {orig['text']}")
    print(f"  Normalized ({len(norm_tokens)} tokens): {norm['text']}")
    print(f"  Tokens: {norm_tokens[:10]}{'...' if len(norm_tokens)>10 else ''}")
    print()
```

**Interview Tips:**
- Highlight that conversational NLP is one of the fastest-growing domains (chatbots, virtual assistants)
- Normalization should be **reversible** for display but **consistent** for model input
- Speaker turn structure is critical metadata — don't tokenize speaker labels as regular words
- DialoGPT and BlenderBot tokenizers handle contractions and informal text much better than standard BERT
- Emoji carry sentiment signal — always preserve them rather than stripping

---

## Question 33
- [ ] Done
**How do you implement customizable tokenization for different user requirements?**

**Answer:**

**Definition:**
Customizable tokenization provides a flexible framework that adapts tokenization behavior to different user needs — varying languages, domains, granularity levels, and output formats — through configuration rather than code changes. This typically involves building a tokenization pipeline with pluggable components (normalizer, pre-tokenizer, model, post-processor) that users can configure via parameters or config files.

**Core Concepts — Customization Dimensions:**

| Dimension | Options | Use Case |
|---|---|---|
| Granularity | Word / Subword / Character / Sentence | Task-dependent token units |
| Normalization | Lowercase, accent removal, Unicode NFC/NFKD | Language and case sensitivity |
| Special patterns | Regex rules for URLs, emails, codes | Domain-specific protection |
| Vocabulary | Custom vocab, vocab augmentation, size control | Domain adaptation |
| Output format | Token strings, IDs, offsets, attention masks | Model input requirements |
| Language | Monolingual vs multilingual, script detection | Multilingual deployments |

**Architecture — Plugin Pipeline:**
1. **Normalizer** → Unicode normalization, lowercasing, accent stripping
2. **Pre-tokenizer** → Whitespace/punctuation splitting, regex pattern protection
3. **Model** → BPE / WordPiece / Unigram / Word-level tokenization
4. **Post-processor** → Add special tokens ([CLS], [SEP]), truncation, padding

**Python Code Example:**
```python
# Pipeline: Configurable tokenization framework with pluggable components
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import re
import unicodedata

@dataclass
class TokenizerConfig:
    """User-configurable tokenization settings."""
    lowercase: bool = True
    remove_accents: bool = False
    protected_patterns: List[str] = field(default_factory=list)  # Regex patterns to protect
    split_pattern: str = r'\s+'          # How to split tokens
    max_token_length: int = 50           # Max chars per token
    include_offsets: bool = False        # Return char offsets
    language: str = "en"                 # Target language

class CustomizableTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._protected_re = [re.compile(p) for p in config.protected_patterns]

    def normalize(self, text: str) -> str:
        """Step 1: Apply configured normalization."""
        text = unicodedata.normalize('NFC', text)
        if self.config.remove_accents:
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                         if unicodedata.category(c) != 'Mn')
        if self.config.lowercase:
            text = text.lower()
        return text

    def pre_tokenize(self, text: str):
        """Step 2: Protect patterns, then split."""
        protected = {}
        for i, pattern in enumerate(self._protected_re):
            for match in pattern.finditer(text):
                placeholder = f"__PROTECTED_{i}_{match.start()}__"
                protected[placeholder] = match.group()
                text = text[:match.start()] + placeholder + text[match.end():]
        tokens = re.split(self.config.split_pattern, text)
        # Restore protected tokens
        result = []
        for tok in tokens:
            if tok in protected:
                result.append(protected[tok])
            elif tok.strip():
                result.append(tok[:self.config.max_token_length])
        return result

    def tokenize(self, text: str) -> dict:
        """Full configurable tokenization pipeline."""
        normalized = self.normalize(text)
        tokens = self.pre_tokenize(normalized)
        result = {'tokens': tokens, 'count': len(tokens)}
        if self.config.include_offsets:
            offsets = []
            pos = 0
            for tok in tokens:
                idx = normalized.find(tok, pos)
                offsets.append((idx, idx + len(tok)))
                pos = idx + len(tok)
            result['offsets'] = offsets
        return result

# Usage with different configs
text = "Email support@example.com about order #ORD-12345 ASAP!"

# Config 1: General purpose
general = CustomizableTokenizer(TokenizerConfig())
print("General:", general.tokenize(text))

# Config 2: Domain-specific with protected patterns
domain = CustomizableTokenizer(TokenizerConfig(
    protected_patterns=[r'\S+@\S+', r'#ORD-\d+'],
    include_offsets=True,
    lowercase=False
))
print("Domain:", domain.tokenize(text))
```

**Interview Tips:**
- Frame customizable tokenization as **configuration over code** — users change a config file, not source code
- The HuggingFace `tokenizers` library uses exactly this pluggable pipeline architecture (Normalizer → PreTokenizer → Model → PostProcessor)
- Mention that protected patterns are the most common user customization request in production
- Config-driven design enables A/B testing different tokenization strategies without code changes
- Always validate configs at initialization rather than failing silently during tokenization

---

## Question 34
- [ ] Done
**What techniques help with tokenization consistency across different text sources?**

**Answer:**

**Definition:**
Tokenization consistency ensures that semantically equivalent text from different sources (web scraping, PDFs, OCR, APIs, user input) produces identical or equivalent token sequences. Inconsistencies arise from encoding differences, whitespace variations, Unicode normalization forms, HTML artifacts, and source-specific formatting, which can degrade model performance when training and inference data come from different sources.

**Core Concepts — Sources of Inconsistency:**

| Source Issue | Example | Impact on Tokenization |
|---|---|---|
| Unicode variants | "café" (NFC) vs "café" (NFD) | Different byte sequences → different tokens |
| Whitespace types | Regular space vs non-breaking space (\u00A0) | Split vs not-split at boundary |
| Quote styles | "smart quotes" vs "straight quotes" | Different token IDs |
| HTML artifacts | `&amp;`, `&nbsp;`, `<br>` | Extra garbage tokens |
| PDF extraction | Ligatures (fi→fi), hyphenation | Broken words, merged tokens |
| OCR errors | "rn" misread as "m", "l" as "1" | Semantic corruption |

**Key Techniques:**
1. **Canonical normalization layer** — Apply Unicode NFC/NFKC normalization, whitespace standardization, and quote normalization as the first pipeline step.
2. **Source-specific preprocessors** — Build adapters for each source type (HTML stripper, PDF cleaner, OCR post-correction) that output standardized plain text.
3. **Consistency validation** — Hash-based checks that identical content from different sources produces identical token sequences.
4. **Normalization regression tests** — Maintain a test suite of known equivalent inputs from different sources and verify tokenization parity.

**Python Code Example:**
```python
# Pipeline: Multi-source text normalization -> consistent tokenization
import unicodedata
import re
import hashlib
from typing import Dict, List

class ConsistentTokenizer:
    """Ensures tokenization consistency across text sources."""

    def __init__(self):
        # Canonical replacements for consistency
        self.char_map = {
            '\u2018': "'", '\u2019': "'",   # Smart single quotes
            '\u201C': '"', '\u201D': '"',   # Smart double quotes
            '\u2013': '-', '\u2014': '-',   # En/em dashes
            '\u00A0': ' ',                   # Non-breaking space
            '\u200B': '',                     # Zero-width space
            '\uFEFF': '',                     # BOM
        }

    def normalize_unicode(self, text: str) -> str:
        """Apply canonical Unicode normalization."""
        text = unicodedata.normalize('NFKC', text)
        for old, new in self.char_map.items():
            text = text.replace(old, new)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Standardize all whitespace variants."""
        text = re.sub(r'[\t\r\f\v]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clean_source(self, text: str, source: str) -> str:
        """Source-specific cleaning."""
        if source == 'html':
            text = re.sub(r'<[^>]+>', ' ', text)
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&nbsp;', ' ').replace('&quot;', '"')
        elif source == 'pdf':
            text = text.replace('\xad', '')       # Soft hyphens
            text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Dehyphenate
            text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')  # Ligatures
        elif source == 'ocr':
            text = re.sub(r'(?<=[a-z])0(?=[a-z])', 'o', text)  # 0 -> o
            text = re.sub(r'(?<=[a-z])1(?=[a-z])', 'l', text)  # 1 -> l
        return text

    def tokenize(self, text: str, source: str = 'plain') -> Dict:
        cleaned = self.clean_source(text, source)
        normalized = self.normalize_unicode(cleaned)
        normalized = self.normalize_whitespace(normalized)
        tokens = normalized.split()
        token_hash = hashlib.md5(' '.join(tokens).encode()).hexdigest()[:8]
        return {'tokens': tokens, 'hash': token_hash, 'source': source}

# Verify consistency across sources
tokenizer = ConsistentTokenizer()
sources = {
    'plain': 'The \u201csmart\u201d quote\u2014test café',
    'html':  'The &quot;smart&quot; quote&mdash;test caf\u00e9',
    'pdf':   'The \u201csmart\u201d quote\u2014test cafe\u0301',
}

for src_type, text in sources.items():
    result = tokenizer.tokenize(text, src_type)
    print(f"[{src_type:5s}] hash={result['hash']} tokens={result['tokens']}")
```

**Interview Tips:**
- Unicode normalization (NFKC) is the single most impactful consistency technique — always apply it first
- Smart quotes vs straight quotes is the most common real-world inconsistency bug
- Hashing tokenized output is a simple but effective consistency check for CI/CD pipelines
- PDF ligatures (fi, fl) silently corrupt text if not handled — always deligate them
- Source-specific adapters should output a single canonical format that the downstream tokenizer consumes

---

## Question 35
- [ ] Done
**How do you handle tokenization optimization for specific NLP model architectures?**

**Answer:**

**Definition:**
Tokenization optimization for specific model architectures involves tailoring tokenization parameters — vocabulary size, subword algorithm, special tokens, sequence length, and encoding format — to match the architectural constraints and training objectives of the target model. Different architectures (encoder-only, decoder-only, encoder-decoder) have distinct tokenization requirements for optimal performance.

**Core Concepts — Architecture-Specific Tokenization:**

| Architecture | Models | Tokenizer | Key Optimization |
|---|---|---|---|
| Encoder-only | BERT, RoBERTa, ELECTRA | WordPiece / BPE | [CLS]/[SEP] tokens, segment IDs, 512 max length |
| Decoder-only | GPT-2, LLaMA, Mistral | BPE (byte-level) | No [CLS]/[SEP], left-padding for batched generation |
| Encoder-Decoder | T5, BART, mBART | SentencePiece / BPE | Prefix tokens, decoder start token, cross-attention alignment |
| Vision-Language | CLIP, LLaVA | BPE + image patches | Image token placeholders, multimodal alignment |
| Sparse/MoE | Mixtral, Switch Transformer | BPE | Token routing awareness, load balancing |

**Key Optimizations:**
1. **Vocabulary size tuning** — Smaller vocab (16K-32K) for compute efficiency, larger vocab (50K-100K) for multilingual coverage. Must match model embedding dimensions.
2. **Padding strategy** — Right-padding for encoders (BERT), left-padding for autoregressive decoders (GPT) to keep generation position at the end.
3. **Special token alignment** — Architecture-specific tokens: `[CLS]`/`[SEP]` for BERT, `<s>`/`</s>` for RoBERTa, `<pad>`/`</s>` for T5.
4. **Sequence length optimization** — Dynamic padding, sequence packing, and chunking strategies tailored to the model's positional encoding (absolute vs rotary vs ALiBi).

**Python Code Example:**
```python
# Pipeline: Architecture-specific tokenization optimization
from transformers import AutoTokenizer

text = "How does tokenization differ across model architectures?"

# BERT (Encoder-only): WordPiece, [CLS]/[SEP], right-padding
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_enc = bert_tok(text, padding="max_length", max_length=32,
                    truncation=True, return_tensors="pt")
print("BERT (Encoder):")
print(f"  Tokens: {bert_tok.convert_ids_to_tokens(bert_enc['input_ids'][0][:15])}")
print(f"  Pad side: {bert_tok.padding_side}")
print(f"  Special: [CLS]={bert_tok.cls_token_id}, [SEP]={bert_tok.sep_token_id}")

# GPT-2 (Decoder-only): Byte-level BPE, left-padding for generation
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt_tok.pad_token = gpt_tok.eos_token
gpt_tok.padding_side = "left"  # Critical for batched generation
gpt_enc = gpt_tok(text, padding="max_length", max_length=32,
                  truncation=True, return_tensors="pt")
print("\nGPT-2 (Decoder):")
print(f"  Tokens: {gpt_tok.convert_ids_to_tokens(gpt_enc['input_ids'][0][:15])}")
print(f"  Pad side: {gpt_tok.padding_side}")
print(f"  No CLS/SEP — autoregressive")

# T5 (Encoder-Decoder): SentencePiece, task prefix
t5_tok = AutoTokenizer.from_pretrained("t5-small")
t5_input = "summarize: " + text  # T5 requires task prefix
t5_enc = t5_tok(t5_input, padding="max_length", max_length=32,
                truncation=True, return_tensors="pt")
print("\nT5 (Encoder-Decoder):")
print(f"  Tokens: {t5_tok.convert_ids_to_tokens(t5_enc['input_ids'][0][:15])}")
print(f"  EOS token: {t5_tok.eos_token} (id={t5_tok.eos_token_id})")
print(f"  Task prefix required: 'summarize:', 'translate:', etc.")

# Sequence packing for training efficiency
def pack_sequences(token_ids_list, max_length, pad_id, sep_id):
    """Pack multiple short sequences into one to reduce padding waste."""
    packed, current = [], []
    for ids in token_ids_list:
        if len(current) + len(ids) + 1 <= max_length:
            current.extend(ids + [sep_id])
        else:
            current.extend([pad_id] * (max_length - len(current)))
            packed.append(current)
            current = ids + [sep_id]
    if current:
        current.extend([pad_id] * (max_length - len(current)))
        packed.append(current)
    return packed

print(f"\nSequence packing reduces padding waste by ~30-50% in training")
```

**Interview Tips:**
- Left-padding for decoder-only models is the most commonly missed optimization — it causes bugs in batched generation if wrong
- T5/BART require task prefixes ("translate:", "summarize:") as part of input tokenization
- Sequence packing can reduce training time by 30-50% by eliminating padding waste
- RoPE/ALiBi models support dynamic sequence lengths better than absolute positional encodings
- Vocabulary size directly impacts embedding table memory — 100K vocab × 768 dim = 300MB just for embeddings

---

## Question 36
- [ ] Done
**What strategies work best for tokenization of technical or scientific literature?**

**Answer:**

**Definition:**
Technical and scientific literature contains domain-specific vocabulary (chemical formulas, mathematical notation, gene names, citations), structured elements (tables, figures, equations), and specialized conventions (abbreviations, units, nomenclature) that general-purpose tokenizers handle poorly. Effective tokenization requires domain-aware pre-processing, specialized vocabularies, and pattern protection to preserve the semantic integrity of scientific content.

**Core Concepts — Scientific Text Challenges:**

| Domain | Tokens to Protect | Example | Standard Tokenizer Failure |
|---|---|---|---|
| Chemistry | Molecular formulas | C₂H₅OH, NaCl | Splits into meaningless fragments |
| Biology | Gene/protein names | BRCA1, TP53, IL-6 | Lowercased or split on hyphens |
| Mathematics | Equations | E=mc², ∇·F | Symbols dropped or misencoded |
| Physics | Units + values | 9.8 m/s², 3×10⁸ | Number-unit separation lost |
| CS/Engineering | Code identifiers | O(n log n), HTTP/2.0 | Split on parentheses/slashes |
| Citations | References | (Smith et al., 2024) | Parentheses split from content |

**Key Strategies:**
1. **Domain-pretrained tokenizers** — Use SciBERT, PubMedBERT, MatSciBERT, ChemBERTa whose vocabularies include domain terms as single tokens.
2. **Pattern-preserving pre-tokenization** — Regex rules to protect formulas, citations, units, and identifiers before subword tokenization.
3. **LaTeX/equation handling** — Either tokenize LaTeX source as a sequence or replace equations with typed placeholders (`[EQ_1]`).
4. **Custom vocabulary augmentation** — Add domain terms to an existing tokenizer and continue pretraining embeddings.

**Python Code Example:**
```python
# Pipeline: Scientific text protection -> domain tokenization -> comparison
import re
from transformers import AutoTokenizer

scientific_text = (
    "The reaction of C₂H₅OH with O₂ produces CO₂ and H₂O. "
    "Gene BRCA1 (MIM: 113705) shows p53-dependent regulation. "
    "The time complexity is O(n log n) per (Smith et al., 2024)."
)

# Step 1: Protect scientific patterns
def protect_scientific(text):
    """Shield domain patterns from subword splitting."""
    protections = []
    patterns = [
        (r'[A-Z][a-z]?\d*(?:[\u2080-\u2089]\d*)*(?:[A-Z][a-z]?\d*(?:[\u2080-\u2089]\d*)*)*',
         'FORMULA'),                                    # Chemical formulas
        (r'\b[A-Z][A-Z0-9]{1,5}(?:-\d+)?\b', 'GENE'),  # Gene names
        (r'O\([^)]+\)', 'COMPLEXITY'),                   # Big-O notation
        (r'\([^)]*et al\.[^)]*\)', 'CITATION'),          # Citations
        (r'MIM:\s*\d+', 'IDENTIFIER'),                   # Database IDs
    ]
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            protections.append((match.start(), match.end(), match.group(), label))
    return protections

protections = protect_scientific(scientific_text)
print("Protected patterns:")
for start, end, text_match, label in protections:
    print(f"  [{label}] '{text_match}' at [{start}:{end}]")

# Step 2: Compare general vs domain tokenizer
general_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
sci_tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

gen_tokens = general_tok.tokenize(scientific_text)
sci_tokens = sci_tok.tokenize(scientific_text)

print(f"\nGeneral BERT: {len(gen_tokens)} tokens")
print(f"SciBERT:      {len(sci_tokens)} tokens")
print(f"Reduction:    {len(gen_tokens)-len(sci_tokens)} tokens "
      f"({(len(gen_tokens)-len(sci_tokens))/len(gen_tokens)*100:.1f}%)")

# Step 3: Show fragmentation difference
for term in ["BRCA1", "C₂H₅OH"]:
    gen = general_tok.tokenize(term)
    sci = sci_tok.tokenize(term)
    print(f"\n'{term}':")
    print(f"  General: {gen}")
    print(f"  SciBERT: {sci}")
```

**Interview Tips:**
- SciBERT reduces token count by 15-25% on scientific text compared to general BERT — this means more content fits in the 512-token window
- Chemical formulas and gene names are the hardest patterns — they look like random character sequences to general tokenizers
- LaTeX equations should be handled as atomic units or replaced with typed placeholders
- Citation patterns `(Author et al., Year)` should be protected or normalized consistently
- Domain tokenization directly impacts retrieval quality in scientific search engines

---

## Question 37
- [ ] Done
**How do you implement batch processing pipelines for large-scale tokenization?**

**Answer:**

**Definition:**
Batch processing pipelines for large-scale tokenization process millions to billions of text documents efficiently by leveraging parallelism, memory-mapped I/O, streaming, and hardware acceleration. The goal is to maximize throughput (documents/second) while controlling memory usage and ensuring fault tolerance for production-scale NLP data preprocessing.

**Core Concepts — Scalability Techniques:**

| Technique | Throughput Gain | Memory Impact | Complexity |
|---|---|---|---|
| Batched tokenization | 3-10× | Low (batch size control) | Low |
| Multi-process parallelism | Linear with cores | Per-process overhead | Medium |
| Rust-based tokenizers (HF) | 10-20× vs Python | Minimal | Low (drop-in) |
| Memory-mapped datasets | N/A | Constant regardless of data size | Medium |
| GPU-accelerated tokenization | 50-100× | GPU memory bound | High |
| Streaming/chunked processing | N/A | O(batch_size) vs O(dataset) | Medium |

**Pipeline Architecture:**
1. **Data loading** → Stream from disk/S3 using memory-mapped files or iterators
2. **Batching** → Group documents into optimal batch sizes (1000-10000)
3. **Parallel tokenization** → Distribute batches across CPU cores or GPU
4. **Dynamic padding** → Pad to longest-in-batch, not global max, to reduce waste
5. **Serialization** → Write tokenized output to Arrow/Parquet format for fast loading

**Python Code Example:**
```python
# Pipeline: Large-scale batch tokenization with HuggingFace datasets + multiprocessing
from transformers import AutoTokenizer
from datasets import Dataset
import time

# Simulate large dataset
texts = [f"Document {i}: Natural language processing enables machines to "
         f"understand human language with example content number {i}."
         for i in range(10000)]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Method 1: Naive loop (baseline)
start = time.time()
naive_results = [tokenizer(t, truncation=True, max_length=128) for t in texts[:1000]]
naive_time = time.time() - start
print(f"Naive loop (1K docs):  {naive_time:.2f}s")

# Method 2: Batched tokenization (single call)
start = time.time()
batched_results = tokenizer(texts[:1000], truncation=True, max_length=128,
                            padding=True, return_tensors="pt")
batched_time = time.time() - start
print(f"Batched (1K docs):     {batched_time:.2f}s ({naive_time/batched_time:.1f}x faster)")

# Method 3: HuggingFace Datasets with multiprocessing (production-scale)
dataset = Dataset.from_dict({"text": texts})

def tokenize_batch(examples):
    """Batch tokenization function for dataset.map()."""
    return tokenizer(examples["text"], truncation=True,
                     max_length=128, padding="max_length")

start = time.time()
tokenized_ds = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,           # Process 1000 docs per batch
    num_proc=4,                # Parallelize across 4 CPU cores
    remove_columns=["text"],   # Drop raw text to save memory
)
ds_time = time.time() - start
print(f"Dataset.map (10K docs, 4 proc): {ds_time:.2f}s")
print(f"Throughput: {len(texts)/ds_time:.0f} docs/sec")

# Method 4: Streaming for datasets that don't fit in memory
from datasets import load_dataset

def streaming_tokenize(dataset_stream, batch_size=1000):
    """Process streaming dataset in chunks."""
    batch = []
    total = 0
    for example in dataset_stream:
        batch.append(example["text"])
        if len(batch) >= batch_size:
            tokenized = tokenizer(batch, truncation=True, max_length=128)
            total += len(batch)
            batch = []
            yield tokenized
    if batch:  # Process remaining
        yield tokenizer(batch, truncation=True, max_length=128)
        total += len(batch)

print(f"\nStreaming pipeline ready for arbitrarily large datasets")
print(f"Memory usage: O(batch_size) = O({1000}) regardless of dataset size")
```

**Interview Tips:**
- HuggingFace `tokenizers` library is written in Rust and is 10-20× faster than pure Python tokenization
- `dataset.map(batched=True, num_proc=N)` is the standard production recipe for large-scale tokenization
- Dynamic padding (`padding='longest'` per batch) saves 30-50% compute vs fixed max-length padding
- Arrow format (used by HF Datasets) enables memory-mapped access — tokenized datasets don't need to fit in RAM
- For truly massive scale (billions of docs), use Spark/Ray with the tokenizer as a UDF

---

## Question 38
- [ ] Done
**What approaches help with tokenization quality assessment without ground truth?**

**Answer:**

**Definition:**
Tokenization quality assessment without ground truth uses intrinsic metrics and proxy measures to evaluate how well a tokenizer segments text when no human-annotated gold-standard tokenization exists. Since most real-world text lacks gold tokenization, these unsupervised evaluation methods are critical for comparing tokenizers, detecting regressions, and guiding vocabulary design decisions.

**Core Concepts — Intrinsic Quality Metrics:**

| Metric | What It Measures | Ideal Value | Interpretation |
|---|---|---|---|
| Fertility | Tokens per word | 1.0-1.5 | Lower = words kept intact |
| Token coverage | % of words tokenized as single tokens | > 70% | Higher = better vocabulary fit |
| UNK rate | % of tokens mapped to unknown | 0% | Any UNK = vocabulary gap |
| Vocabulary utilization | % of vocab used on a corpus | > 50% | Low = wasted vocabulary slots |
| Token frequency entropy | Shannon entropy of token distribution | High | More uniform = balanced vocab |
| Sequence length ratio | Output tokens / input chars | Low | Lower = more efficient encoding |
| Morphological alignment | Token boundaries vs morpheme boundaries | High overlap | Linguistically meaningful splits |

**Key Approaches:**
1. **Fertility analysis** — Compute tokens-per-word ratio across domains. High fertility on domain text signals poor vocabulary coverage.
2. **Vocabulary utilization audit** — Check what fraction of the vocabulary is actually used. Unused tokens waste embedding capacity.
3. **Cross-domain consistency** — Compare fertility and coverage metrics across different text domains to detect domain bias.
4. **Roundtrip fidelity** — Encode and decode text, then verify the output matches the input exactly (no information loss).
5. **Downstream proxy tasks** — Evaluate tokenization quality indirectly via simple tasks (language modeling perplexity, classification accuracy) without task-specific labels.

**Python Code Example:**
```python
# Pipeline: Tokenization quality assessment without ground truth
from transformers import AutoTokenizer
from collections import Counter
import math

def assess_tokenizer_quality(tokenizer, texts, name="Tokenizer"):
    """Compute intrinsic quality metrics for a tokenizer."""
    all_tokens = []
    total_words = 0
    total_chars = 0
    unk_count = 0
    roundtrip_errors = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        total_chars += len(text)

        tokens = tokenizer.tokenize(text)
        all_tokens.extend(tokens)

        # Count UNK tokens
        unk_id = tokenizer.unk_token_id
        if unk_id is not None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            unk_count += ids.count(unk_id)

        # Roundtrip fidelity check
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        if decoded.strip().lower() != text.strip().lower():
            roundtrip_errors += 1

    # Compute metrics
    fertility = len(all_tokens) / total_words if total_words > 0 else 0
    seq_ratio = len(all_tokens) / total_chars if total_chars > 0 else 0
    unk_rate = unk_count / len(all_tokens) * 100 if all_tokens else 0
    roundtrip_accuracy = (len(texts) - roundtrip_errors) / len(texts) * 100

    # Token frequency entropy
    freq = Counter(all_tokens)
    total = sum(freq.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())

    # Vocabulary utilization
    unique_tokens = len(freq)
    vocab_size = tokenizer.vocab_size
    vocab_utilization = unique_tokens / vocab_size * 100

    # Single-token word coverage
    single_token_words = sum(1 for w in set(' '.join(texts).split())
                            if len(tokenizer.tokenize(w)) == 1)
    total_unique_words = len(set(' '.join(texts).split()))
    coverage = single_token_words / total_unique_words * 100

    print(f"\n=== {name} Quality Report ===")
    print(f"  Fertility (tokens/word):   {fertility:.2f}")
    print(f"  Sequence ratio (tok/char): {seq_ratio:.3f}")
    print(f"  UNK rate:                  {unk_rate:.2f}%")
    print(f"  Single-token coverage:     {coverage:.1f}%")
    print(f"  Vocab utilization:         {vocab_utilization:.1f}% ({unique_tokens}/{vocab_size})")
    print(f"  Token entropy:             {entropy:.2f} bits")
    print(f"  Roundtrip accuracy:        {roundtrip_accuracy:.1f}%")
    return {'fertility': fertility, 'unk_rate': unk_rate, 'coverage': coverage}

# Compare tokenizers on sample texts
texts = [
    "The patient was diagnosed with acute myocardial infarction.",
    "Machine learning models require careful hyperparameter tuning.",
    "The cryptocurrency market experienced significant volatility.",
    "Quantum computing leverages superposition and entanglement.",
]

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

assess_tokenizer_quality(bert_tok, texts, "BERT")
assess_tokenizer_quality(gpt2_tok, texts, "GPT-2")
```

**Interview Tips:**
- Fertility is the most practical single metric — it correlates with downstream performance and is easy to compute
- UNK rate should be exactly 0% for modern subword tokenizers — any UNKs indicate a serious vocabulary problem
- Roundtrip fidelity (encode → decode → compare) catches silent data corruption bugs
- Vocabulary utilization below 30% suggests the vocab is poorly fitted to the domain
- These metrics enable tokenizer comparison without any labeled data — critical for domain adaptation decisions

---

## Question 39
- [ ] Done
**How do you handle tokenization for texts with cultural or linguistic variations?**

**Answer:**

**Definition:**
Cultural and linguistic variations in text include dialectal differences (e.g., American vs British English), code-switching (mixing languages mid-sentence), culturally-specific expressions (idioms, honorifics, naming conventions), script variations (simplified vs traditional Chinese), and regional spelling/vocabulary differences. Effective tokenization for such text requires cultural awareness, flexible normalization, and multilingual tokenization strategies.

**Core Concepts — Variation Types and Challenges:**

| Variation Type | Example | Tokenization Challenge |
|---|---|---|
| Regional spelling | colour/color, analyse/analyze | Same meaning, different tokens |
| Dialects | "y'all", "gonna", "innit" | OOV in standard tokenizers |
| Code-switching | "Let's go, vamonos!" | Mixed-language vocabulary needs |
| Honorifics/titles | お客様 (Japanese), Sri (Indonesian) | Cultural prefixes split incorrectly |
| Script variants | 简体/繁體 (Simplified/Traditional Chinese) | Same word, different characters |
| Transliteration | Москва/Moskva/Moscow | Multiple representations of one entity |
| Right-to-left text | Arabic/Hebrew mixed with Latin | Bidirectional text handling |

**Key Approaches:**
1. **Multilingual tokenizers** — Use XLM-R, mBERT, or NLLB tokenizers trained on 100+ languages that handle code-switching natively.
2. **Variant normalization** — Map regional variants to canonical forms (colour→color) when consistency matters more than preservation.
3. **Script detection + routing** — Detect script per segment and apply language-specific tokenization rules.
4. **Cultural pattern preservation** — Protect honorifics, naming conventions, and culturally significant expressions from splitting.

**Python Code Example:**
```python
# Pipeline: Culturally-aware tokenization with variant handling
import re
from transformers import AutoTokenizer
import unicodedata

class CulturalTokenizer:
    """Handles cultural + linguistic text variations."""

    # Regional spelling normalization (British -> American)
    SPELLING_MAP = {
        'colour': 'color', 'favourite': 'favorite', 'analyse': 'analyze',
        'organisation': 'organization', 'centre': 'center',
        'behaviour': 'behavior', 'defence': 'defense',
    }

    def __init__(self, model_name="xlm-roberta-base", normalize_variants=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalize_variants = normalize_variants

    def detect_script(self, text):
        """Detect dominant script in text."""
        scripts = {}
        for char in text:
            if char.isalpha():
                script = unicodedata.name(char, '').split()[0]
                scripts[script] = scripts.get(script, 0) + 1
        return max(scripts, key=scripts.get) if scripts else 'UNKNOWN'

    def normalize_spelling(self, text):
        """Normalize regional spelling variants."""
        if not self.normalize_variants:
            return text
        words = text.split()
        return ' '.join(self.SPELLING_MAP.get(w.lower(), w) for w in words)

    def handle_code_switching(self, text):
        """Segment code-switched text by script."""
        segments = []
        current_script, current_text = None, []
        for char in text:
            if char.isalpha():
                script = unicodedata.name(char, '').split()[0]
                if script != current_script and current_text:
                    segments.append((''.join(current_text), current_script))
                    current_text = []
                current_script = script
            current_text.append(char)
        if current_text:
            segments.append((''.join(current_text), current_script))
        return segments

    def tokenize(self, text):
        """Full culturally-aware tokenization."""
        normalized = self.normalize_spelling(text)
        tokens = self.tokenizer.tokenize(normalized)
        return {
            'original': text,
            'normalized': normalized,
            'tokens': tokens,
            'dominant_script': self.detect_script(text),
            'token_count': len(tokens)
        }

# Example usage
tok = CulturalTokenizer()

texts = [
    "My favourite colour is blue.",                # British English
    "Let's go, ¡vamonos al parque!",               # Code-switching EN/ES
    "The meeting is at 3pm, お願いします.",          # EN/JA code-switching
    "He went to the centre for defence analysis.",  # British spelling
]

for text in texts:
    result = tok.tokenize(text)
    print(f"Input:      {result['original']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Script:     {result['dominant_script']}")
    print(f"Tokens({result['token_count']}): {result['tokens'][:10]}...")
    print()
```

**Interview Tips:**
- XLM-RoBERTa is the go-to model for culturally diverse text — trained on 100 languages with SentencePiece
- Spelling normalization is a **policy decision** — some tasks require preservation (machine translation), others benefit from canonicalization (search/retrieval)
- Code-switching is increasingly common in social media — multilingual tokenizers handle it better than language-specific ones
- Script detection helps route text to the appropriate tokenization rules in multilingual pipelines
- Cultural honorifics and naming conventions (e.g., Chinese family name first) need domain knowledge, not just NLP

---

## Question 40
- [ ] Done
**What techniques work best for tokenization in resource-constrained environments?**

**Answer:**

**Definition:**
Resource-constrained tokenization refers to running tokenization efficiently on devices or systems with limited CPU, memory, storage, or power — such as mobile phones, IoT devices, edge servers, or low-cost cloud instances. The challenge is maintaining tokenization quality while minimizing vocabulary size, memory footprint, inference latency, and binary size.

**Core Concepts — Resource Optimization Techniques:**

| Technique | Memory Savings | Speed Impact | Quality Impact |
|---|---|---|---|
| Smaller vocabulary (8K-16K) | 50-75% embedding reduction | Slightly longer sequences | Minor quality drop |
| Vocabulary pruning | 30-50% vocab reduction | Faster lookup | Minimal on domain text |
| Hash-based tokenization | No vocabulary storage needed | Fast O(1) lookup | Hash collisions reduce quality |
| BPE with byte fallback | Zero UNK, minimal vocab | Good throughput | Handles all inputs |
| Quantized embeddings (INT8) | 75% memory reduction | Faster on ARM/mobile | < 1% quality loss |
| Cached tokenization | N/A | Huge for repeated inputs | No quality change |
| CPP/Rust tokenizer | 80%+ less than Python | 10-20× faster | Identical quality |

**Key Strategies:**
1. **Compact vocabularies** — Train BPE/Unigram with 8K-16K vocab instead of 32K-50K. Smaller embedding tables, at the cost of slightly longer sequences.
2. **Byte-level BPE** — Models like GPT-2 and LLaMA use byte-level BPE that requires no UNK token and handles arbitrary UTF-8 input with a base vocabulary of only 256 bytes.
3. **Native-code tokenizers** — Use HuggingFace `tokenizers` (Rust) or SentencePiece (C++) instead of Python implementations for 10-20× speed improvement.
4. **Vocabulary pruning** — Remove unused or rare tokens from the vocabulary based on domain corpus frequency, then retrain embeddings.

**Python Code Example:**
```python
# Pipeline: Resource-efficient tokenization strategies
import sys
import time
from transformers import AutoTokenizer

text = "Natural language processing on edge devices requires efficient tokenization."
texts_batch = [text] * 1000  # Simulate batch

# Strategy 1: Compare vocabulary sizes and memory
models = {
    "BERT (30K vocab)": "bert-base-uncased",
    "DistilBERT (30K)": "distilbert-base-uncased",
    "Albert (30K)": "albert-base-v2",  # Shared embeddings = smaller
}

for name, model in models.items():
    tok = AutoTokenizer.from_pretrained(model)
    tokens = tok.tokenize(text)
    vocab_mem = tok.vocab_size * 4  # 4 bytes per int32 ID
    print(f"{name}: {tok.vocab_size} vocab, ~{vocab_mem/1024:.0f}KB vocab memory, "
          f"{len(tokens)} tokens")

# Strategy 2: Tokenization speed comparison (Python vs Rust backend)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Fast tokenizer (Rust backend)
start = time.time()
for t in texts_batch:
    _ = tokenizer(t, truncation=True, max_length=128)
fast_time = time.time() - start

print(f"\nRust tokenizer (1K docs): {fast_time*1000:.1f}ms")
print(f"Throughput: {len(texts_batch)/fast_time:.0f} docs/sec")

# Strategy 3: LRU Cache for repeated tokenizations
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_tokenize(text):
    """Cache tokenization results for repeated inputs."""
    return tuple(tokenizer.encode(text, truncation=True, max_length=128))

# First call: computes
start = time.time()
_ = cached_tokenize(text)
cold_time = time.time() - start

# Second call: cache hit
start = time.time()
_ = cached_tokenize(text)
hot_time = time.time() - start

print(f"\nCached tokenization:")
print(f"  Cold call: {cold_time*1e6:.0f}µs")
print(f"  Cache hit: {hot_time*1e6:.0f}µs")
print(f"  Speedup:   {cold_time/hot_time:.0f}x")

# Strategy 4: Memory estimation for deployment
vocab_sizes = [8000, 16000, 30522, 50257]
embed_dim = 768
for vs in vocab_sizes:
    fp32_mb = vs * embed_dim * 4 / 1024 / 1024
    int8_mb = vs * embed_dim * 1 / 1024 / 1024
    print(f"\nVocab {vs:>6d}: FP32={fp32_mb:.1f}MB, INT8={int8_mb:.1f}MB")
```

**Interview Tips:**
- Byte-level BPE is the most resource-friendly approach — 256 base tokens, no UNK, handles arbitrary input
- HuggingFace Rust tokenizers are the standard for production — always use `use_fast=True` (default)
- ALBERT uses shared embeddings across layers, reducing total model size but not vocabulary size
- LRU caching tokenization results is highly effective for chatbots with repetitive inputs
- For mobile deployment, SentencePiece C++ library can be compiled to < 1MB binary

---

## Question 41
- [ ] Done
**How do you implement fairness-aware tokenization to avoid bias across languages?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies help with tokenization of emerging text types and genres?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle tokenization integration with modern transformer-based models?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What approaches work best for tokenization with specific encoding requirements?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement monitoring and quality control for tokenization systems?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What techniques help with tokenization of texts requiring semantic preservation?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle tokenization adaptation to user-specific vocabulary and terminology?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What strategies work best for tokenization in multilingual neural machine translation?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you implement efficient storage and retrieval of tokenization vocabularies?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What approaches help with balancing tokenization granularity and computational efficiency?**
**Answer:** _To be filled_

---


---

# --- NER Questions (from 08_nlp/02_ner) ---

# Named Entity Recognition (NER) - Theory Questions

## Question 1
- [ ] Done
**How do you handle NER for entities that span multiple tokens or have complex internal structure?**
**Answer:**

Multi-token entities like "New York City" or "Bank of America" require models that capture token dependencies across spans rather than classifying tokens independently. The standard approach uses **BIO/BILOU tagging** schemes where B marks the beginning, I marks inside tokens, and O marks outside. More advanced approaches use **span-based models** that directly classify candidate spans.

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| BIO tagging | B-PER, I-PER, O labels | Simple, well-supported | Cannot handle nested entities |
| BILOU tagging | Adds L (last) and U (unit) | Better boundary detection | More labels to predict |
| Span-based | Enumerate & classify spans | Handles nested entities | O(n²) candidate spans |
| Pointer networks | Predict start/end positions | Flexible boundaries | Complex training |
| CRF layer | Model label transitions | Enforces valid sequences | Slower inference |

```python
# Pipeline: Text -> Tokenize -> BiLSTM-CRF -> BIO tags -> Entity spans

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "Barack Obama visited the United Nations headquarters in New York City."
entities = ner_pipeline(text)
for ent in entities:
    print(f"{ent['word']:30s} {ent['entity_group']:6s} ({ent['score']:.3f})")
# Barack Obama                   PER    (0.998)
# United Nations                 ORG    (0.995)
# New York City                  LOC    (0.997)
```

**Interview Tips:**
- BIO tagging is default; BILOU improves boundary detection by 1-2% F1
- `aggregation_strategy="simple"` in HuggingFace merges sub-tokens automatically
- CRF layer ensures valid transitions (no I-PER after B-LOC)
- Span-based models (SpanBERT) are preferred for nested NER

---

## Question 2
- [ ] Done
**What techniques work best for NER in low-resource languages with limited training data?**
**Answer:**

Low-resource NER faces the challenge of insufficient labeled data. Key strategies include **cross-lingual transfer** (train on high-resource language, apply to low-resource), **few-shot learning**, **data augmentation**, and **active learning** to maximize annotation efficiency.

| Technique | Description | Data Needed |
|-----------|-------------|-------------|
| Cross-lingual transfer | Fine-tune multilingual model (XLM-R) on English NER, apply to target | 0 target labels |
| Few-shot prompting | Use LLM with examples in prompt | 5-20 examples |
| Data augmentation | Synonym replacement, entity swapping | Existing small set |
| Active learning | Select most informative samples to label | Human annotator |
| Gazetteers/dictionaries | Inject entity lists as features | Domain knowledge |
| Self-training | Train on labeled, pseudo-label unlabeled | Unlabeled corpus |

```python
# Pipeline: Multilingual model -> Zero-shot cross-lingual NER

from transformers import pipeline

# XLM-R trained on English NER, applied to other languages
ner = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="simple")

# Works across languages without target-language training data
texts = [
    "Barack Obama visited Berlin.",      # English
    "Angela Merkel besuchte Paris.",      # German
    "東京でオリンピックが開催された。",         # Japanese
]
for text in texts:
    entities = ner(text)
    print(f"\n{text}")
    for e in entities:
        print(f"  {e['word']:20s} -> {e['entity_group']}")
```

**Interview Tips:**
- XLM-RoBERTa is the go-to model for cross-lingual NER transfer
- Zero-shot transfer works because multilingual models learn shared representations
- Data augmentation: replace entities with same-type entities from gazetteers
- Active learning reduces annotation cost by 50-70% compared to random sampling

---

## Question 3
- [ ] Done
**How do you implement domain adaptation for NER models across different text domains?**
**Answer:**

Domain adaptation for NER addresses the performance drop when models trained on one domain (news) are applied to another (biomedical, legal). Strategies include **continued pre-training** on target domain text, **fine-tuning** on small target domain labeled data, and **feature augmentation** with domain-specific gazetteers.

| Strategy | Approach | Labeled Data Needed |
|----------|----------|--------------------|
| Continued pre-training | MLM on target domain corpus | None (unsupervised) |
| Fine-tuning | Supervised on target labels | Small labeled set |
| Multi-task learning | Train on source + target jointly | Some target labels |
| Gazetteer features | Add domain entity dictionaries | Domain knowledge |
| Domain-adversarial | Learn domain-invariant features | Both domains |

```python
# Pipeline: General BERT -> Domain pre-train -> Fine-tune NER

from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer

# Step 1: Start with domain-specific pretrained model
# Biomedical: "dmis-lab/biobert-v1.1"
# Clinical: "emilyalsentzer/Bio_ClinicalBERT"
# Legal: "nlpaueb/legal-bert-base-uncased"

model_name = "dmis-lab/biobert-v1.1"  # Domain-adapted BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Fine-tune on domain NER data
# Entity types differ by domain:
# News: PER, ORG, LOC, MISC
# Biomedical: Gene, Disease, Chemical, Species
# Legal: Court, Judge, Statute, Date

label_list = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical", "B-Gene", "I-Gene"]
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
print(f"Model: {model_name}, Labels: {label_list}")
```

**Interview Tips:**
- BioBERT, SciBERT, LegalBERT, FinBERT are domain-adapted BERT variants
- Continued pre-training (MLM) on domain text is the most impactful single step
- Entity types change across domains — design label schema for your domain
- Gazetteer features provide strong signal for domain entities (drug names, legal terms)

---

## Question 4
- [ ] Done
**What strategies help with handling nested or overlapping entities in NER tasks?**
**Answer:**

Nested NER handles entities contained within other entities, e.g., "[Bank of [America]_LOC]_ORG". Standard BIO tagging cannot represent this. Solutions include **span-based models**, **layered/stacked sequence labeling**, and **hypergraph-based decoding**.

| Approach | Description | Complexity |
|----------|-------------|------------|
| Span enumeration | Score all possible (start, end, type) spans | O(n² × T) |
| Layered CRF | Stack multiple CRF layers, each finding one nesting level | O(n × K × L) |
| Biaffine model | Biaffine attention scores for start-end pairs | O(n²) |
| Seq2seq | Generate linearized entity mentions | Depends on output |
| Set prediction | Predict entity set without ordering | Flexible |

```python
# Pipeline: Text -> Span enumeration -> Score spans -> Resolve nesting

import torch
import torch.nn as nn

def enumerate_spans(tokens, max_span_length=8):
    """Generate all candidate spans up to max length."""
    spans = []
    for start in range(len(tokens)):
        for end in range(start, min(start + max_span_length, len(tokens))):
            spans.append((start, end, " ".join(tokens[start:end+1])))
    return spans

tokens = ["Bank", "of", "America", "headquarters"]
spans = enumerate_spans(tokens, max_span_length=4)
print("Candidate spans:")
for s, e, text in spans:
    print(f"  [{s}:{e}] '{text}'")

# In a real model, each span gets scored for each entity type:
# score(start, end, type) = biaffine(h_start, h_end) + type_embedding
# Nested entities are naturally handled since spans can overlap
```

**Interview Tips:**
- Standard BIO cannot represent nested entities — this is a key limitation
- Span-based approach is most popular: enumerate spans, classify each
- Biaffine model (Yu et al., 2020) is state-of-the-art for nested NER
- ACE2005 and GENIA are standard nested NER benchmarks
- In practice, nesting is rare outside biomedical domain

---

## Question 5
- [ ] Done
**How do you design NER models that can identify new entity types with minimal examples?**
**Answer:**

Few-shot NER identifies new entity types from just a handful of examples. Approaches include **prototypical networks** (learn type representations from examples), **prompt-based methods** (frame NER as text generation), and **in-context learning** with LLMs.

| Approach | Examples Needed | Method |
|----------|----------------|--------|
| Prototypical networks | 5-10 per type | Compare token embeddings to type prototypes |
| Prompt-based (GPT) | 1-5 per type | In-context learning with formatted examples |
| Template filling | 5-20 per type | "[X] is a [TYPE]" templates |
| Retrieval-augmented | 5-10 per type | Retrieve similar labeled examples at inference |
| Meta-learning (MAML) | 5-10 per type | Learn to adapt quickly from few examples |

```python
# Pipeline: Define new entity type -> Few examples -> LLM-based NER

def few_shot_ner(text, entity_type, examples):
    """Few-shot NER using prompt engineering."""
    prompt = f"""Extract all {entity_type} entities from the text.

Examples:
"""
    for ex_text, ex_entities in examples:
        prompt += f"Text: {ex_text}\nEntities: {', '.join(ex_entities)}\n\n"
    prompt += f"Text: {text}\nEntities:"
    return prompt

# Define new entity type with just 3 examples
examples = [
    ("Tesla released Model Y in Austin", ["Model Y"]),
    ("Apple launched iPhone 15 Pro Max", ["iPhone 15 Pro Max"]),
    ("Samsung Galaxy S24 Ultra is available", ["Galaxy S24 Ultra"]),
]

prompt = few_shot_ner(
    "Google announced Pixel 8 and Pixel Watch 2 yesterday",
    "product name", examples
)
print(prompt)
# Send to LLM -> extracts: "Pixel 8", "Pixel Watch 2"
```

**Interview Tips:**
- Prototypical networks are the classic few-shot NER approach
- GPT-4/Claude can do few-shot NER via prompting with 90%+ accuracy
- Key challenge: distinguishing similar entity types with few examples
- Combine few-shot with gazetteers for better recall

---

## Question 6
- [ ] Done
**What approaches work best for NER in noisy or informal text like social media posts?**
**Answer:**

Social media NER faces unique challenges: non-standard spelling, abbreviations, hashtags, @mentions, emojis, and lack of capitalization. Best approaches use **social-media-pretrained models** (BERTweet), **multi-task learning** with normalization, and **robust training** with noise augmentation.

| Challenge | Example | Solution |
|-----------|---------|----------|
| No capitalization | "went to paris with john" | Context-only models, not case features |
| Hashtags | "#VisitNewYork" | Hashtag segmentation ("Visit New York") |
| @mentions | "@elikiowa talked about..." | User entity linking |
| Abbreviations | "nyc", "sf" | Normalization dictionary |
| Emojis as context | "🇺🇸 in LA" | Emoji-aware tokenization |

```python
# Pipeline: Social text -> Normalize -> NER with social-media model

import re

def preprocess_social_text(text):
    """Normalize social media text for NER."""
    # Segment hashtags: #NewYorkCity -> New York City
    text = re.sub(r'#(\S+)', lambda m: re.sub(r'([a-z])([A-Z])', r'\1 \2', m.group(1)), text)
    # Normalize @mentions
    text = re.sub(r'@(\w+)', r'USER_\1', text)
    # Remove excessive punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text

texts = [
    "@john went to #NewYorkCity!!! best trip ever",
    "omg #BarackObama spoke at #UnitedNations today",
]

for t in texts:
    print(f"Original: {t}")
    print(f"Cleaned:  {preprocess_social_text(t)}\n")

# Use BERTweet for social media NER
# from transformers import pipeline
# ner = pipeline("ner", model="tner/bertweet-large-tweetner7-all")
```

**Interview Tips:**
- BERTweet and TweetNER are purpose-built for social media NER
- Hashtag segmentation is a preprocessing must — "#NewYorkCity" → "New York City"
- WNUT shared task is the standard benchmark for noisy text NER
- Noise augmentation during training (random case changes, typo injection) improves robustness

---

## Question 7
- [ ] Done
**How do you handle NER for entities with ambiguous boundaries or unclear definitions?**
**Answer:**

Boundary ambiguity arises when entity extent is unclear: is it "President Biden" or just "Biden"? Is "New York" the city or the state? Solutions include **annotation guidelines** with clear rules, **span-level confidence scores**, and **multiple annotation** with adjudication.

| Ambiguity Type | Example | Resolution |
|---------------|---------|------------|
| Extent ambiguity | "President Barack Obama" vs "Barack Obama" | Guideline: minimal vs maximal span |
| Type ambiguity | "Apple" (company vs fruit) | Context-dependent classification |
| Metonymy | "Washington" (city/government/person) | Entity linking to knowledge base |
| Coordination | "North and South Korea" | Split into two entities |

```python
# Pipeline: Text -> NER with confidence -> Filter by threshold -> Resolve

from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

ambiguous_texts = [
    "Apple released a new product in Washington.",
    "President Obama met with the Jordan delegation.",
]

for text in ambiguous_texts:
    entities = ner(text)
    print(f"\nText: {text}")
    for e in entities:
        confidence = "HIGH" if e['score'] > 0.95 else "AMBIGUOUS"
        print(f"  {e['word']:25s} {e['entity_group']:6s} score={e['score']:.3f} [{confidence}]")
```

**Interview Tips:**
- Clear annotation guidelines are the #1 factor for boundary consistency
- Inter-annotator agreement (Cohen's kappa) measures boundary ambiguity
- Entity linking resolves type ambiguity by grounding to knowledge base
- Confidence thresholds help flag ambiguous predictions for human review

---

## Question 8
- [ ] Done
**What techniques help with NER in multilingual texts with code-switching?**
**Answer:**

Code-switching (mixing languages within a sentence, e.g., "I went to the mercado to buy some frutas") challenges NER because entity boundaries cross language boundaries and models need multilingual understanding. Best approaches use **multilingual transformers** (XLM-R) and **language-aware features**.

| Technique | Description |
|-----------|-------------|
| XLM-RoBERTa | Pretrained on 100+ languages, handles mixed input |
| Language ID features | Add language tag per token as auxiliary feature |
| Code-switch augmentation | Artificially create code-switched training data |
| Joint language ID + NER | Multi-task learning predicts language and entities |
| Transliteration normalization | Normalize romanized text to native script |

```python
# Pipeline: Code-switched text -> Language detect -> Multilingual NER

from transformers import pipeline

# XLM-R NER handles code-switched text naturally
ner = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="simple")

code_switched = [
    "Mi amigo John fue a New York ayer.",       # Spanish-English
    "I ate Ramen at the neue Restaurant in Berlin.",  # English-German
]

for text in code_switched:
    entities = ner(text)
    print(f"\nText: {text}")
    for e in entities:
        print(f"  {e['word']:20s} -> {e['entity_group']} ({e['score']:.3f})")
```

**Interview Tips:**
- XLM-RoBERTa is the default choice for multilingual/code-switched NER
- LinCE benchmark is the standard for code-switching NLP tasks
- Code-switch augmentation: randomly replace entities/phrases with translations
- Real-world code-switching is common in multilingual communities (Spanglish, Hinglish)

---

## Question 10
- [ ] Done
**What strategies work best for NER in specialized domains like biomedical or legal texts?**
**Answer:**

Domain-specific NER requires models trained on domain corpora with domain entity types. Use **domain-pretrained models** (BioBERT, LegalBERT), **domain gazetteers** (UMLS for biomedical, case law databases), and **custom annotation schemes** matching domain ontologies.

| Domain | Model | Entity Types | Gazetteer |
|--------|-------|-------------|----------|
| Biomedical | BioBERT, PubMedBERT | Gene, Disease, Chemical, Species | UMLS, DrugBank |
| Clinical | ClinicalBERT | Symptom, Treatment, Dosage | SNOMED-CT |
| Legal | LegalBERT | Court, Statute, Judge, Party | Case law DBs |
| Financial | FinBERT | Company, Ticker, Amount | SEC filings |

```python
# Pipeline: Domain text -> Domain model -> Domain entity types

from transformers import pipeline

# Biomedical NER with domain-specific model
bio_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

medical_text = "Aspirin 325mg was prescribed for acute myocardial infarction. BRCA1 gene mutation was detected."
entities = bio_ner(medical_text)
for e in entities:
    print(f"{e['word']:35s} -> {e['entity_group']} ({e['score']:.3f})")
```

**Interview Tips:**
- Always start with domain-pretrained model, not general BERT
- BioBERT + continued fine-tuning on your data is the standard recipe
- Domain gazetteers significantly boost recall for known entities
- Annotation guidelines must be written by domain experts

---

## Question 12
- [ ] Done
**What approaches help with explaining NER decisions and predicted entity boundaries?**
**Answer:**

Explainable NER helps users understand why a model tagged certain spans as entities. Techniques include **attention visualization**, **token-level feature importance** (SHAP/LIME), and **rule extraction** from model predictions.

| Method | What It Shows | Complexity |
|--------|---------------|------------|
| Attention heatmaps | Which tokens the model attends to | Low |
| SHAP values | Per-token contribution to prediction | Medium |
| Integrated gradients | Gradient-based attribution | Medium |
| Probing classifiers | What features the model uses | High |
| Rule extraction | Human-readable decision rules | High |

```python
# Pipeline: NER prediction -> Extract attention -> Visualize explanations

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def explain_ner_prediction(text, model_name="dslim/bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    labels = [model.config.id2label[p.item()] for p in predictions]
    
    # Show predictions with confidence
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    for tok, label, prob in zip(tokens[1:-1], labels[1:-1], probs[1:-1]):
        conf = prob.max().item()
        if label != 'O':
            print(f"{tok:15s} -> {label:10s} (confidence: {conf:.3f})")

explain_ner_prediction("Barack Obama visited Google in Mountain View.")
```

**Interview Tips:**
- Attention weights are easy to visualize but may not reflect true reasoning
- SHAP provides more faithful explanations than attention alone
- Integrated gradients are the gold standard for attribution
- Explainability is critical for NER in healthcare and legal domains

---

## Question 13
- [ ] Done
**How do you implement knowledge distillation for compressing large NER models?**
**Answer:**

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" NER model, retaining most accuracy with significantly fewer parameters. The student learns from the teacher's **soft label distributions** rather than hard ground truth, capturing uncertainty and inter-class relationships.

| Aspect | Teacher | Student |
|--------|---------|--------|
| Model | BERT-large | DistilBERT, TinyBERT |
| Parameters | 340M | 66M (5x reduction) |
| Inference | ~50ms | ~15ms (3x faster) |
| F1 (CoNLL) | ~92.8 | ~91.5 (98.6% retained) |

```python
# Pipeline: Teacher NER -> Soft labels -> Train student -> Deploy

import torch
import torch.nn as nn

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """Combined distillation + task loss for NER."""
    # Soft label loss (KL divergence)
    soft_student = torch.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
    
    # Hard label loss (cross-entropy)
    ce_loss = nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
    return alpha * kd_loss + (1 - alpha) * ce_loss

# Temperature controls softness: higher = softer distributions
# Alpha balances teacher imitation vs ground truth
print("Distillation: teacher soft labels guide student learning")
print("Typical: temperature=2-5, alpha=0.5-0.7")
```

**Interview Tips:**
- Temperature is key: higher values produce softer distributions with more information
- DistilBERT retains ~97% of BERT's NER performance at 40% parameters
- Layer-wise distillation (TinyBERT) copies intermediate representations too
- Combine with pruning and quantization for maximum compression

---

## Question 14
- [ ] Done
**What techniques work best for NER with limited computational resources?**
**Answer:**

Resource-efficient NER uses **smaller models** (DistilBERT, TinyBERT), **quantization**, **pruning**, and **efficient architectures** to run NER on CPUs, mobile devices, or edge hardware without significant accuracy loss.

| Technique | Speedup | Accuracy Loss | Method |
|-----------|---------|---------------|--------|
| DistilBERT | 2-3x | ~1% F1 | Knowledge distillation |
| Quantization (INT8) | 2-4x | <0.5% F1 | Post-training quantization |
| Pruning | 2-3x | <1% F1 | Remove low-magnitude weights |
| ONNX Runtime | 2-5x | 0% | Optimized inference runtime |
| BiLSTM-CRF | 10-50x | 2-3% F1 | Non-transformer architecture |

```python
# Pipeline: Full model -> Quantize -> ONNX export -> Fast inference

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Option 1: Use smaller model
model_name = "distilbert-base-uncased"  # 66M params vs BERT's 110M

# Option 2: Dynamic quantization (fastest to implement)
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Compare sizes
import os
torch.save(model.state_dict(), "original.pt")
torch.save(quantized_model.state_dict(), "quantized.pt")
print(f"Original: {os.path.getsize('original.pt') / 1e6:.1f} MB")
print(f"Quantized: {os.path.getsize('quantized.pt') / 1e6:.1f} MB")
```

**Interview Tips:**
- Quantization is the easiest win: 2-4x speedup with minimal accuracy loss
- ONNX Runtime + quantization gives the best inference performance
- BiLSTM-CRF is still viable for edge/mobile with 50x fewer parameters
- For extreme constraints, use spaCy's small models (en_core_web_sm: ~12MB)

---

## Question 15
- [ ] Done
**How do you handle NER for entities that change over time or have temporal significance?**
**Answer:**

Temporal entity dynamics include **new entities** (emerging companies, people), **changing categories** (Twitter → X), and **time-dependent context**. Solutions involve **continual learning**, **temporal gazetteers**, and **periodic model retraining**.

| Challenge | Example | Solution |
|-----------|---------|----------|
| New entities | "OpenAI", "ChatGPT" | Continual learning, gazetteer updates |
| Name changes | "Twitter" → "X" | Alias mapping, entity linking |
| Historical context | "Czechoslovakia" | Time-stamped knowledge base |
| Emerging categories | "cryptocurrency" entities | New label addition + fine-tuning |

```python
# Pipeline: Detect temporal drift -> Update gazetteers -> Retrain incrementally

class TemporalNERManager:
    def __init__(self):
        self.entity_registry = {}  # entity -> {type, first_seen, aliases}
    
    def register_entity(self, name, entity_type, timestamp, aliases=None):
        self.entity_registry[name] = {
            'type': entity_type,
            'first_seen': timestamp,
            'aliases': aliases or [],
            'active': True
        }
    
    def update_entity(self, old_name, new_name, timestamp):
        if old_name in self.entity_registry:
            self.entity_registry[old_name]['aliases'].append(new_name)
            self.entity_registry[new_name] = self.entity_registry[old_name].copy()
    
    def get_gazetteer(self, timestamp=None):
        """Get active entities as of timestamp."""
        return {name: info['type'] for name, info in self.entity_registry.items() if info['active']}

manager = TemporalNERManager()
manager.register_entity("Twitter", "ORG", "2006-03-21")
manager.update_entity("Twitter", "X", "2023-07-24")
print(manager.get_gazetteer())
```

**Interview Tips:**
- Entity drift is a real production challenge — models degrade over time
- Solution: periodic retraining + continuously updated gazetteers
- Continual learning avoids catastrophic forgetting of old entities
- Time-aware entity linking grounds entities to the correct time period

---

## Question 16
- [ ] Done
**What strategies help with NER consistency across different text formats and sources?**
**Answer:**

NER consistency ensures the same entities are tagged identically regardless of source format (HTML, PDF, plain text, JSON). Key strategies: **standardized preprocessing**, **format-agnostic feature extraction**, and **post-processing normalization**.

| Format Issue | Example | Solution |
|-------------|---------|----------|
| HTML markup | `<b>Obama</b>` | Strip tags, preserve text |
| PDF extraction | Broken words across lines | Text reconstruction |
| OCR noise | "0bama" for "Obama" | Spell correction + fuzzy matching |
| Encoding issues | "caf\xe9" | Unicode normalization (NFC) |
| Case variation | "OBAMA", "Obama", "obama" | Case-insensitive matching |

```python
# Pipeline: Multi-format input -> Normalize -> Unified NER -> Consistent output

import re
import unicodedata

def normalize_for_ner(text, source_format='plain'):
    """Standardize text from different sources for consistent NER."""
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    # Strip HTML if present
    if source_format == 'html':
        text = re.sub(r'<[^>]+>', '', text)
    # Fix common OCR errors
    ocr_fixes = {'0bama': 'Obama', 'G00gle': 'Google', 'l': 'I'}
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sources = [
    ("<p>Barack <b>Obama</b> visited Google.</p>", 'html'),
    ("Barack 0bama  visited  G00gle.", 'ocr'),
]
for text, fmt in sources:
    print(f"{fmt:5s}: {normalize_for_ner(text, fmt)}")
```

**Interview Tips:**
- Preprocessing standardization is more impactful than model changes
- Unicode normalization (NFC) is a must for multilingual consistency
- Post-processing entity linking canonicalizes different surface forms
- Integration tests should verify NER consistency across input formats

---

## Question 17
- [ ] Done
**How do you implement online learning for NER models adapting to new entity types?**
**Answer:**

Online learning for NER enables models to learn new entity types incrementally without retraining from scratch, avoiding **catastrophic forgetting** of previously learned types. Key techniques: **elastic weight consolidation (EWC)**, **adapter modules**, and **expandable output layers**.

| Approach | Description | Forgetting Risk |
|----------|-------------|----------------|
| Full retraining | Retrain on all data (old + new) | None (but expensive) |
| Fine-tuning on new | Only train on new type data | High |
| EWC | Penalize changes to important weights | Low |
| Adapter layers | Add small adapter per entity type | None |
| Replay buffer | Mix old examples with new during training | Low |

```python
# Pipeline: Base NER model -> Add adapter for new type -> Incremental training

import torch
import torch.nn as nn

class IncrementalNER(nn.Module):
    def __init__(self, base_model, base_num_labels):
        super().__init__()
        self.base_model = base_model
        self.classifiers = nn.ModuleDict({
            'base': nn.Linear(768, base_num_labels)
        })
    
    def add_entity_type(self, type_name, num_new_labels):
        """Add classifier head for new entity type without modifying base."""
        self.classifiers[type_name] = nn.Linear(768, num_new_labels)
        # Freeze base model to prevent forgetting
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"Added new entity type: {type_name} ({num_new_labels} labels)")
    
    def forward(self, x, task='base'):
        features = self.base_model(x)  # shared encoder
        return self.classifiers[task](features)

print("Expandable NER: freeze base, add adapter per new entity type")
```

**Interview Tips:**
- EWC adds a penalty term that prevents changing weights important for old tasks
- Adapter modules (small bottleneck layers) are the most practical approach
- Replay buffer: keep ~5% of old data to prevent forgetting
- Production systems use periodic full retraining + incremental updates between cycles

---

## Question 18
- [ ] Done
**What approaches work best for NER in conversational or dialogue systems?**
**Answer:**

Dialogue NER must handle informal language, coreferences ("he", "that place"), multi-turn context, and disfluencies ("um", "like"). Best approaches use **context-aware models** that consider conversation history and **joint intent+entity extraction**.

| Challenge | Example | Solution |
|-----------|---------|----------|
| Coreference | "Book it" (it = restaurant) | Dialog state tracking |
| Ellipsis | "And for Tuesday?" | Previous turn context |
| Disfluency | "I want um Chicago" | Disfluency removal |
| Implicit entities | "The same place" | Dialog memory |

```python
# Pipeline: Dialog history -> Context window -> NER with history features

def dialogue_ner(turns, ner_model):
    """NER with conversation context."""
    context_entities = {}  # Track entities across turns
    for i, turn in enumerate(turns):
        # Prepend previous turn for context
        context = turns[i-1] + " " + turn if i > 0 else turn
        entities = ner_model(context)
        # Update entity memory
        for e in entities:
            context_entities[e['entity_group']] = e['word']
        print(f"Turn {i}: {turn}")
        print(f"  Entities: {entities}")
        print(f"  Context: {context_entities}")

# Dialogue NER benefits from joint training with intent classification
# Frameworks: Rasa NLU, JointBERT (intent + slot filling)
```

**Interview Tips:**
- JointBERT handles intent classification + slot filling (entity extraction) jointly
- Context window of 1-2 previous turns significantly improves dialogue NER
- Rasa NLU is the standard framework for dialogue entity extraction
- Disfluency removal should happen before NER in spoken dialogue

---

## Question 19
- [ ] Done
**How do you handle NER optimization for specific downstream applications?**
**Answer:**

NER optimization depends on the downstream task. **Information extraction** needs high recall (find all entities). **Search/indexing** needs high precision (avoid false positives). **Relation extraction** needs accurate boundaries. Tune the model's threshold, training objective, and post-processing accordingly.

| Application | Priority | Optimization Strategy |
|-------------|----------|----------------------|
| Search/indexing | Precision | Higher confidence threshold |
| Knowledge graph | Recall | Lower threshold + entity linking |
| Redaction/privacy | Recall | Maximize sensitivity (catch all PII) |
| Question answering | Boundary accuracy | Span-level training |
| Summarization | Type accuracy | Focus on key entity types |

```python
# Pipeline: NER output -> Application-specific post-processing

def optimize_ner_for_task(entities, task='search'):
    """Post-process NER results based on downstream application."""
    if task == 'search':
        # High precision: only keep confident entities
        return [e for e in entities if e['score'] > 0.95]
    elif task == 'redaction':
        # High recall: keep even low-confidence PII
        return [e for e in entities if e['score'] > 0.5 and e['entity_group'] in ['PER', 'LOC', 'ORG']]
    elif task == 'knowledge_graph':
        # Balanced: moderate threshold + deduplication
        seen = set()
        unique = []
        for e in entities:
            if e['score'] > 0.7 and e['word'] not in seen:
                seen.add(e['word'])
                unique.append(e)
        return unique
    return entities

print("Application-specific NER optimization adjusts thresholds and filters")
```

**Interview Tips:**
- PII redaction demands near-100% recall (missing one SSN is a compliance failure)
- Search needs precision (false positive entities pollute search index)
- Adjust classification threshold rather than retraining for different precision/recall
- Mention F-beta score: F2 weights recall more, F0.5 weights precision more

---

## Question 20
- [ ] Done
**What techniques help with NER for entities requiring cultural or contextual knowledge?**
**Answer:**

Culturally-dependent entities require **external knowledge** beyond text patterns. Names, titles, organizations, and locations vary by culture. "Dr." may be a title or abbreviation; "Bangalore" and "Bengaluru" refer to the same city. Solutions: **knowledge-grounded models**, **cultural gazetteers**, and **entity linking** to knowledge bases.

| Cultural Challenge | Example | Solution |
|--------------------|---------|----------|
| Name formats | "Yao Ming" (family name first) | Culture-aware name parsing |
| Titles/honorifics | "Sheikh", "Sri", "Prof" | Title gazetteers by culture |
| Transliteration | "Путин" / "Putin" | Cross-lingual entity linking |
| Location aliases | "Mumbai" / "Bombay" | Alias mapping |
| Cultural entities | "Diwali", "Eid" | Cultural knowledge base |

```python
# Pipeline: Text -> NER -> Entity linking to knowledge base -> Cultural enrichment

cultural_aliases = {
    'Bombay': 'Mumbai', 'Peking': 'Beijing', 'Calcutta': 'Kolkata',
    'Ceylon': 'Sri Lanka', 'Siam': 'Thailand'
}

def culturally_aware_ner(entities):
    """Normalize entities with cultural context."""
    for e in entities:
        canonical = cultural_aliases.get(e['word'], e['word'])
        if canonical != e['word']:
            e['canonical'] = canonical
            e['original'] = e['word']
        else:
            e['canonical'] = e['word']
    return entities

print("Cultural NER requires external knowledge + entity linking")
print("Wikidata is an excellent multilingual knowledge base for entity grounding")
```

**Interview Tips:**
- Wikidata and Wikipedia are the primary knowledge bases for entity linking
- Cross-lingual entity linking maps entities across languages/scripts
- Cultural bias in NER: models trained on English news perform poorly on non-Western names
- Multilingual training data is essential for culturally-aware NER

---

## Question 21
- [ ] Done
**How do you implement fairness-aware NER to avoid bias across different entity types?**
**Answer:**

NER models can exhibit bias: better performance on common names ("John Smith") than uncommon ones ("Amarjeet Singh"), or on Western locations versus non-Western ones. Fairness-aware NER uses **balanced training data**, **disaggregated evaluation**, and **debiasing techniques**.

| Bias Type | Example | Mitigation |
|-----------|---------|------------|
| Name bias | Better on English names | Diverse name training data |
| Geographic bias | Better on US/EU locations | Multilingual gazetteers |
| Gender bias | "chairman" vs "chairperson" | Gender-balanced training |
| Frequency bias | Common entities scored higher | Class-balanced loss |

```python
# Pipeline: NER model -> Disaggregated evaluation -> Bias detection -> Mitigation

def evaluate_ner_fairness(model, test_sets):
    """Evaluate NER performance across demographic groups."""
    results = {}
    for group_name, data in test_sets.items():
        predictions = model(data['texts'])
        f1 = compute_f1(predictions, data['labels'])
        results[group_name] = f1
    
    # Compute fairness gap
    max_f1 = max(results.values())
    min_f1 = min(results.values())
    fairness_gap = max_f1 - min_f1
    
    print(f"Fairness gap: {fairness_gap:.3f}")
    print(f"Per-group F1: {results}")
    return results

def compute_f1(preds, labels):
    # Simplified F1 computation
    return 0.90  # placeholder

print("Key: evaluate F1 separately per name origin, gender, region")
print("Acceptable fairness gap: <5% F1 difference across groups")
```

**Interview Tips:**
- Disaggregated evaluation (F1 per demographic group) is the #1 fairness tool
- Name bias is the most common NER fairness issue
- Mitigation: oversample underrepresented name origins in training
- Fairness constraints can be added to the training loss function

---

## Question 22
- [ ] Done
**What strategies work best for NER in real-time processing scenarios?**
**Answer:**

Real-time NER requires sub-100ms latency for streaming text (chat, social media, voice assistants). Strategies: **model optimization** (distillation, quantization), **batch inference**, **caching**, and **lightweight models**.

| Optimization | Latency Reduction | Implementation |
|-------------|-------------------|----------------|
| Model distillation | 2-3x | DistilBERT, TinyBERT |
| INT8 quantization | 2-4x | PyTorch/ONNX quantize |
| ONNX Runtime | 2-5x | Export + optimize |
| GPU batching | 5-10x throughput | Batch incoming requests |
| Entity caching | 100x for known text | LRU cache on input hash |
| spaCy small | ~1ms/doc | en_core_web_sm |

```python
# Pipeline: Text stream -> Cache check -> Fast NER -> Stream results

from functools import lru_cache
import hashlib

class RealTimeNER:
    def __init__(self, model_name="en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model_name)
        self._cache = {}
    
    def extract(self, text):
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]
        # Run NER
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        self._cache[text_hash] = entities
        return entities

ner = RealTimeNER()
import time
start = time.time()
for _ in range(100):
    ner.extract("Barack Obama visited Google in Mountain View.")
print(f"100 calls: {(time.time()-start)*1000:.1f}ms (with caching)")
```

**Interview Tips:**
- spaCy is fastest for production NER (C-optimized, ~1ms per doc)
- ONNX export + TensorRT can achieve <10ms for BERT NER
- Cache frequently seen text (chat messages, product names)
- Batch requests to maximize GPU utilization

---

## Question 23
- [ ] Done
**How do you handle NER quality assessment when ground truth annotations vary?**
**Answer:**

Annotation disagreement is common in NER due to ambiguous boundaries, subjective entity types, and annotator skill differences. Quality assessment uses **inter-annotator agreement (IAA)**, **adjudication processes**, and **soft evaluation metrics** that account for partial matches.

| Metric | Measures | Range |
|--------|----------|-------|
| Cohen's Kappa | Agreement between 2 annotators | -1 to 1 (>0.8 = excellent) |
| Fleiss' Kappa | Agreement among 3+ annotators | -1 to 1 |
| Span-level F1 | Exact match on entity spans | 0 to 1 |
| Partial match F1 | Overlap-based matching | 0 to 1 |
| Token-level accuracy | Per-token agreement | 0 to 1 |

```python
# Pipeline: Multiple annotators -> IAA calculation -> Adjudication -> Gold standard

from sklearn.metrics import cohen_kappa_score

def compute_ner_iaa(annotator1_labels, annotator2_labels):
    """Compute inter-annotator agreement for NER."""
    kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
    agreement = sum(a == b for a, b in zip(annotator1_labels, annotator2_labels)) / len(annotator1_labels)
    return {'kappa': kappa, 'agreement': agreement}

# Example: token-level annotations
ann1 = ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']
ann2 = ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC']  # Disagree on LOC span

result = compute_ner_iaa(ann1, ann2)
print(f"Kappa: {result['kappa']:.3f}, Agreement: {result['agreement']:.3f}")
```

**Interview Tips:**
- Cohen's Kappa >0.8 = reliable annotations; <0.6 = guidelines need revision
- SemEval uses partial match scoring: credit for overlapping spans
- Adjudication: expert resolves disagreements (not majority vote for NER)
- Double annotation of 10-20% of data is standard for quality monitoring

---

## Question 24
- [ ] Done
**What approaches help with NER for entities in multiple languages within the same text?**
**Answer:**

Multilingual text within a single document requires models that understand multiple languages simultaneously. A medical report might mix Latin terms with English, or a social post might use English and Hindi. Use **multilingual models**, **language ID per token**, and **unified entity schemas**.

| Approach | Handles |
|----------|--------|
| XLM-RoBERTa | 100+ languages, code-switching |
| mBERT | 104 languages, shared representation |
| Per-language NER + merge | Best per-language but complex pipeline |
| Language-agnostic features | Capitalization, position, shape (no language dependency) |

```python
# Pipeline: Mixed-language text -> Multilingual NER -> Unified entities

from transformers import pipeline

# XLM-R handles multilingual text naturally
ner = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="simple")

multilingual_texts = [
    "Dr. Müller visited Tokyo for the ICML conference.",  # German name, Japanese city
    "El presidente Macron viajó a Washington D.C.",    # Spanish with French name
]

for text in multilingual_texts:
    entities = ner(text)
    print(f"\n{text}")
    for e in entities:
        print(f"  {e['word']:25s} {e['entity_group']}")
```

**Interview Tips:**
- XLM-RoBERTa is the standard for multilingual NER (zero-shot cross-lingual)
- Shared subword vocabulary across languages enables cross-lingual transfer
- Evaluate per-language to detect bias toward high-resource languages
- WikiANN dataset provides NER annotations in 282 languages

---

## Question 27
- [ ] Done
**How do you handle NER adaptation to emerging entity categories and definitions?**
**Answer:**

New entity types emerge as domains evolve (e.g., "cryptocurrency token", "mRNA vaccine brand"). Adaptation requires **extensible architectures**, **few-shot learning**, and **dynamic label schemas** that add new types without retraining the full model.

| Strategy | When to Use |
|----------|------------|
| Add classification head | Well-defined new type with training data |
| Few-shot prompting | Urgent need, minimal data |
| Prototype network | New type with 5-20 examples |
| Active learning | Budget for annotation |
| Zero-shot with descriptions | No examples, only type description |

```python
# Pipeline: Define new type -> Few examples -> Extend NER

def extend_ner_schema(existing_labels, new_type, examples):
    """Extend NER with a new entity type."""
    updated_labels = existing_labels + [f"B-{new_type}", f"I-{new_type}"]
    print(f"Labels: {len(existing_labels)} -> {len(updated_labels)}")
    print(f"New type '{new_type}' with {len(examples)} examples")
    return updated_labels

labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
new_labels = extend_ner_schema(labels, 'CRYPTO', [
    ('Bitcoin surged past $60K', ['Bitcoin']),
    ('Ethereum merge completed', ['Ethereum']),
])
```

**Interview Tips:**
- Modular architecture: shared encoder + task-specific classification heads
- Few-shot NER with LLMs can bootstrap new entity types immediately
- Maintain backward compatibility when adding new types (old types still work)
- Version your NER schemas for reproducibility

---

## Question 28
- [ ] Done
**What strategies help with NER for entities requiring external knowledge or context?**
**Answer:**

Some entities can only be identified with external knowledge: "Jordan" (country vs person vs brand) requires world knowledge. Solutions: **knowledge-enhanced models**, **entity linking**, and **retrieval-augmented NER**.

| Knowledge Source | Integration Method |
|-----------------|-------------------|
| Knowledge graphs | Entity embeddings (TransE, ComplEx) |
| Wikipedia | Entity descriptions as context |
| Gazetteers | Soft dictionary matching features |
| Type constraints | "CEO" implies next entity is likely PER |

```python
# Pipeline: Text -> NER -> Ambiguous entities -> Knowledge base lookup -> Disambiguate

def knowledge_enhanced_ner(text, entities, knowledge_base):
    """Disambiguate entities using external knowledge."""
    resolved = []
    for entity in entities:
        candidates = knowledge_base.get(entity['word'], [])
        if len(candidates) > 1:
            # Use context to disambiguate
            best = max(candidates, key=lambda c: context_similarity(text, c['description']))
            entity['resolved_type'] = best['type']
        resolved.append(entity)
    return resolved

def context_similarity(text, description):
    return 0.5  # placeholder

knowledge_base = {
    'Jordan': [
        {'type': 'PER', 'description': 'Michael Jordan, basketball player'},
        {'type': 'LOC', 'description': 'Country in Middle East'},
        {'type': 'ORG', 'description': 'Jordan brand by Nike'},
    ]
}
print("Knowledge-enhanced NER disambiguates using external context")
```

**Interview Tips:**
- Entity linking (EL) = NER + disambiguation to knowledge base
- BLINK (Facebook) and REL are popular entity linking systems
- Knowledge graphs provide type constraints that improve NER accuracy
- Retrieval-augmented approaches fetch relevant KB entries at inference time

---

## Question 29
- [ ] Done
**How do you implement robust error handling for NER in production systems?**
**Answer:**

Production NER must handle edge cases gracefully: empty inputs, encoding errors, extremely long texts, model timeouts, and OOM errors. Implement **input validation**, **fallback pipelines**, **timeout handling**, and **graceful degradation**.

| Error Type | Cause | Handling |
|-----------|-------|----------|
| Empty input | User sends blank text | Return empty entity list |
| Encoding error | Non-UTF8 characters | Unicode normalization fallback |
| Text too long | Exceeds model max length | Chunking with overlap |
| Model timeout | GPU overloaded | Return cached/partial results |
| OOM error | Batch too large | Reduce batch size dynamically |

```python
# Pipeline: Input validation -> Safe NER -> Error recovery -> Output

import logging

class ProductionNER:
    def __init__(self, model, max_length=512, timeout=5.0):
        self.model = model
        self.max_length = max_length
        self.timeout = timeout
        self.logger = logging.getLogger('ProductionNER')
    
    def extract(self, text):
        try:
            # Input validation
            if not text or not text.strip():
                return []
            # Handle encoding issues
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            # Chunk if too long
            if len(text) > self.max_length * 4:
                return self._chunked_ner(text)
            return self.model(text)
        except MemoryError:
            self.logger.error("OOM: falling back to CPU")
            return self._fallback_ner(text)
        except Exception as e:
            self.logger.error(f"NER error: {e}")
            return []
    
    def _chunked_ner(self, text, chunk_size=1000, overlap=100):
        entities = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            entities.extend(self.model(chunk))
        return entities
    
    def _fallback_ner(self, text):
        return []  # Graceful degradation

print("Production NER: validate input, chunk long text, handle errors gracefully")
```

**Interview Tips:**
- Always chunk long documents (BERT max = 512 tokens)
- Overlap chunks by 50-100 tokens to avoid splitting entities
- Circuit breaker pattern: after N failures, stop calling model temporarily
- Health checks and monitoring are essential for NER services

---

## Question 30
- [ ] Done
**What approaches work best for combining NER with other information extraction tasks?**
**Answer:**

NER is often the first step in a pipeline: **NER → Entity Linking → Relation Extraction → Knowledge Graph**. Joint models that extract entities and relations simultaneously outperform pipeline approaches by sharing representations and avoiding error propagation.

| Task Combination | Architecture | Benefit |
|-----------------|-------------|--------|
| NER + Relation Extraction | Joint span-based model | Shared entity representations |
| NER + Entity Linking | End-to-end EL model | Disambiguation informs NER |
| NER + Coreference | Joint model | Resolves pronoun entities |
| NER + Event Extraction | Multi-task learning | Events need entity arguments |

```python
# Pipeline: Text -> Joint NER + RE -> Knowledge triplets

def joint_ner_re(text, ner_model, re_model):
    """Combined NER and Relation Extraction."""
    # Step 1: Extract entities
    entities = ner_model(text)
    # Step 2: For each entity pair, predict relation
    triplets = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i+1:]:
            relation = re_model(text, e1, e2)
            if relation != 'no_relation':
                triplets.append((e1['word'], relation, e2['word']))
    return triplets

# Example output:
# "Elon Musk is the CEO of Tesla" -> 
# [("Elon Musk", "CEO_of", "Tesla")]

print("Joint NER+RE avoids error propagation from sequential pipeline")
print("Multi-task loss = L_ner + L_re + L_shared")
```

**Interview Tips:**
- Pipeline approach: NER errors propagate to downstream tasks
- Joint models share encoder but have task-specific heads
- Multi-task learning improves both NER and RE by 2-5% F1
- OpenIE extracts (subject, relation, object) triplets in an open-domain way

---

## Question 31
- [ ] Done
**How do you handle NER for entities with varying granularity levels?**
**Answer:**

Granularity varies from coarse (PER, LOC, ORG) to fine-grained (politician, scientist, city, country). Fine-grained NER provides more useful information but is harder to learn. Solutions: **hierarchical label schemas**, **coarse-to-fine models**, and **ontology-aware training**.

| Granularity | Labels | Use Case |
|------------|--------|----------|
| Coarse (3-4 types) | PER, ORG, LOC, MISC | General NER |
| Medium (10-20 types) | Politician, Company, City | Domain-specific |
| Fine-grained (100+) | Football_Player, Tech_Company | Knowledge graphs |

```python
# Pipeline: Text -> Coarse NER -> Fine-grained classification

fine_grained_hierarchy = {
    'PER': ['Politician', 'Athlete', 'Scientist', 'Artist', 'CEO'],
    'ORG': ['Company', 'University', 'Government', 'NGO', 'Sports_Team'],
    'LOC': ['City', 'Country', 'State', 'Landmark', 'Continent'],
}

def hierarchical_ner(text, coarse_ner, fine_classifier):
    """Two-stage NER: coarse then fine-grained."""
    entities = coarse_ner(text)
    for entity in entities:
        coarse_type = entity['entity_group']
        if coarse_type in fine_grained_hierarchy:
            fine_type = fine_classifier(entity['word'], fine_grained_hierarchy[coarse_type])
            entity['fine_type'] = fine_type
    return entities

# Example: "Elon Musk" -> PER -> CEO
print("Hierarchical NER: coarse type first, then fine-grained subtype")
```

**Interview Tips:**
- FIGER and TypeNet provide fine-grained entity type ontologies
- Hierarchical loss functions enforce type consistency (CEO must be PER)
- Coarse-to-fine is more practical than flat fine-grained classification
- Few-shot works well for adding new fine-grained types

---

## Question 32
- [ ] Done
**What techniques help with NER consistency in federated learning scenarios?**
**Answer:**

Federated NER trains models across distributed data sources (hospitals, banks) without sharing raw data. Challenge: annotation inconsistencies across sites, non-IID data distributions, and communication overhead.

| Challenge | Solution |
|-----------|----------|
| Label inconsistency | Shared annotation guidelines, unified schema |
| Non-IID data | FedProx, personalized FL |
| Communication cost | Gradient compression, fewer rounds |
| Privacy | Differential privacy, secure aggregation |

```python
# Pipeline: Local NER training -> Aggregate gradients -> Global model

def federated_ner_round(global_model, local_datasets, learning_rate=0.01):
    """One round of federated NER training."""
    local_updates = []
    for dataset in local_datasets:
        # Each site trains locally
        local_model = global_model.copy()
        local_model.train(dataset)
        # Compute update (gradient)
        update = local_model.parameters() - global_model.parameters()
        local_updates.append(update)
    
    # Aggregate updates (FedAvg)
    avg_update = sum(local_updates) / len(local_updates)
    global_model.parameters() += learning_rate * avg_update
    return global_model

print("Federated NER: train locally, aggregate globally, no data sharing")
print("FedAvg is the standard aggregation algorithm")
```

**Interview Tips:**
- FedAvg (McMahan et al., 2017) is the standard federated learning algorithm
- Healthcare NER is a prime use case (HIPAA prevents data sharing)
- Differential privacy adds noise to gradients for formal privacy guarantees
- Non-IID data (different entity distributions per site) is the main challenge

---

## Question 33
- [ ] Done
**How do you implement efficient batch processing for large-scale NER applications?**
**Answer:**

Large-scale NER processes millions of documents. Efficiency requires **GPU batching**, **dynamic padding**, **multiprocessing**, and **distributed processing** frameworks like Spark NLP or Ray.

| Optimization | Speedup | Implementation |
|-------------|---------|----------------|
| GPU batching | 10-50x | Pad sequences, batch inference |
| Dynamic padding | 1.5-2x | Pad to longest in batch, not max |
| Multi-GPU | Linear with GPUs | DataParallel, DistributedDataParallel |
| CPU multiprocessing | ~Nx for N cores | Process queue or Ray |
| Spark NLP | Cluster-scale | Distributed Spark + NER annotator |

```python
# Pipeline: Document queue -> Batch -> GPU NER -> Collect results

from transformers import pipeline
import time

def batch_ner(texts, model_name="dslim/bert-base-NER", batch_size=32):
    """Efficient batch NER processing."""
    ner = pipeline("ner", model=model_name, device=0 if __import__('torch').cuda.is_available() else -1,
                   batch_size=batch_size, aggregation_strategy="simple")
    
    start = time.time()
    results = ner(texts)
    elapsed = time.time() - start
    
    print(f"Processed {len(texts)} docs in {elapsed:.2f}s ({len(texts)/elapsed:.0f} docs/sec)")
    return results

# batch_ner(["text"] * 1000, batch_size=64)
print("Key: batch_size parameter in HuggingFace pipeline enables GPU batching")
```

**Interview Tips:**
- HuggingFace pipeline's `batch_size` parameter enables automatic batching
- Dynamic padding (pad to longest in batch) saves 30-50% compute vs max-length padding
- For millions of documents, use Spark NLP or Ray for distributed processing
- Sort documents by length before batching to minimize padding waste

---

## Question 34
- [ ] Done
**What strategies work best for NER with specific accuracy requirements?**
**Answer:**

Different applications need different accuracy profiles. Medical NER requires near-perfect recall; search indexing needs high precision. Strategies: **threshold tuning**, **ensemble models**, **human-in-the-loop**, and **targeted fine-tuning**.

| Accuracy Need | Strategy | Trade-off |
|--------------|----------|----------|
| High precision | Raise confidence threshold | Lower recall |
| High recall | Lower threshold + post-filter | More false positives |
| High F1 | Ensemble multiple models | Higher compute cost |
| Near-perfect | Human review of low-confidence | Higher latency + cost |

```python
# Pipeline: NER prediction -> Confidence thresholding -> Ensemble -> Human review

def accuracy_optimized_ner(text, models, strategy='high_precision'):
    """NER with configurable accuracy strategy."""
    all_entities = []
    for model in models:
        entities = model(text)
        all_entities.extend(entities)
    
    if strategy == 'high_precision':
        # Only keep entities found by majority of models
        from collections import Counter
        entity_counts = Counter((e['word'], e['entity_group']) for e in all_entities)
        threshold = len(models) // 2 + 1
        return [{'word': w, 'entity_group': t} for (w, t), c in entity_counts.items() if c >= threshold]
    elif strategy == 'high_recall':
        # Keep entities found by any model (union)
        seen = set()
        unique = []
        for e in all_entities:
            key = (e['word'], e['entity_group'])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

print("Ensemble voting: majority vote for precision, union for recall")
```

**Interview Tips:**
- Precision-recall trade-off is the core concept — all NER optimization flows from this
- Model ensemble (3-5 models) can boost F1 by 2-3%
- Human-in-the-loop for low-confidence predictions is common in high-stakes NER
- Confidence calibration ensures the model's confidence scores are meaningful

---

## Question 35
- [ ] Done
**How do you handle NER for entities that require disambiguation or linking?**
**Answer:**

Entity disambiguation (Entity Linking) maps NER mentions to canonical entries in a knowledge base (Wikidata, UMLS). "Apple" → Q312 (company) vs Q89 (fruit). This is essential for building knowledge graphs and answering questions accurately.

| Stage | Task | Method |
|-------|------|--------|
| Mention detection | Find entity spans | NER model |
| Candidate generation | Find KB candidates | String matching, alias table |
| Candidate ranking | Select best candidate | Context similarity scoring |
| NIL detection | Entity not in KB | Threshold on best score |

```python
# Pipeline: Text -> NER -> Candidate generation -> Context-based ranking -> KB link

def entity_linking(mention, context, knowledge_base):
    """Link entity mention to knowledge base."""
    # Step 1: Candidate generation
    candidates = knowledge_base.get(mention.lower(), [])
    if not candidates:
        return {'mention': mention, 'linked_to': 'NIL', 'confidence': 0}
    
    # Step 2: Rank by context similarity
    best_candidate = max(candidates, key=lambda c: context_overlap(context, c['description']))
    return {'mention': mention, 'linked_to': best_candidate['id'], 'confidence': 0.95}

def context_overlap(text, description):
    words1 = set(text.lower().split())
    words2 = set(description.lower().split())
    return len(words1 & words2) / max(len(words1 | words2), 1)

kb = {
    'apple': [
        {'id': 'Q312', 'type': 'Company', 'description': 'Apple Inc technology company iPhone'},
        {'id': 'Q89', 'type': 'Fruit', 'description': 'apple fruit tree edible'},
    ]
}

print(entity_linking('Apple', 'Apple released new iPhone 15 today', kb))
print(entity_linking('Apple', 'I bought a fresh apple at the market', kb))
```

**Interview Tips:**
- BLINK (Facebook) uses a bi-encoder for fast candidate retrieval + cross-encoder for ranking
- NIL detection handles entities not in the knowledge base (new/emerging entities)
- Entity linking is essential for knowledge graph construction
- Cross-lingual EL links mentions in any language to a unified KB

---

## Question 36
- [ ] Done
**What approaches help with NER adaptation to user-specific entity definitions?**
**Answer:**

User-specific NER allows custom entity types (e.g., a law firm defining "Case Number" or a retailer defining "Product SKU"). Approaches: **few-shot learning**, **user-defined gazetteers**, **configurable entity schemas**, and **active learning** with user feedback.

| Approach | User Effort | Accuracy |
|----------|------------|----------|
| Gazetteer upload | List entities | High for known entities |
| Few-shot examples | 5-20 labeled examples | Moderate-high |
| Regex patterns | Define patterns | High for structured entities |
| Active learning | Label model suggestions | High with iteration |

```python
# Pipeline: User defines entity type -> Provide examples -> Adapt NER model

class CustomNER:
    def __init__(self, base_model):
        self.base_model = base_model
        self.custom_gazetteers = {}
        self.custom_patterns = {}
    
    def add_entity_type(self, type_name, gazetteers=None, pattern=None):
        if gazetteers:
            self.custom_gazetteers[type_name] = set(g.lower() for g in gazetteers)
        if pattern:
            self.custom_patterns[type_name] = pattern
    
    def extract(self, text):
        entities = self.base_model(text)  # Base NER
        # Add gazetteer matches
        for type_name, gazetteer in self.custom_gazetteers.items():
            for term in gazetteer:
                if term in text.lower():
                    entities.append({'word': term, 'entity_group': type_name})
        # Add pattern matches
        import re
        for type_name, pattern in self.custom_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({'word': match.group(), 'entity_group': type_name})
        return entities

print("Custom NER: base model + user gazetteers + regex patterns")
```

**Interview Tips:**
- Layered approach: base NER + user gazetteers + regex rules
- Prodigy (by spaCy) is designed for user-in-the-loop NER annotation
- Few-shot with LLMs can bootstrap custom entity types quickly
- Allow users to correct predictions → feed back into training

---

## Question 37
- [ ] Done
**How do you implement monitoring and quality control for NER systems?**
**Answer:**

NER monitoring tracks model performance in production, detecting data drift, accuracy degradation, and edge cases. Key: **automated evaluation**, **drift detection**, **confidence monitoring**, and **annotation sampling**.

| Metric | What to Monitor | Alert Threshold |
|--------|----------------|----------------|
| Entity count per doc | Distribution shift | >2σ from baseline |
| Confidence distribution | Model uncertainty | Mean confidence <0.8 |
| Entity type distribution | Label drift | >10% change |
| Throughput | Performance degradation | >2x latency increase |
| Error rate | Manual review findings | >5% error rate |

```python
# Pipeline: NER predictions -> Log metrics -> Detect drift -> Alert

import logging
from collections import Counter

class NERMonitor:
    def __init__(self):
        self.entity_counts = []
        self.confidence_scores = []
        self.type_distribution = Counter()
        self.logger = logging.getLogger('NERMonitor')
    
    def log_prediction(self, entities):
        self.entity_counts.append(len(entities))
        for e in entities:
            self.confidence_scores.append(e.get('score', 0))
            self.type_distribution[e['entity_group']] += 1
    
    def check_health(self):
        import numpy as np
        avg_entities = np.mean(self.entity_counts[-100:]) if self.entity_counts else 0
        avg_confidence = np.mean(self.confidence_scores[-100:]) if self.confidence_scores else 0
        
        alerts = []
        if avg_confidence < 0.8:
            alerts.append(f"Low confidence: {avg_confidence:.3f}")
        if avg_entities < 0.5:
            alerts.append(f"Low entity rate: {avg_entities:.1f} per doc")
        return alerts

monitor = NERMonitor()
print("Monitor: confidence, entity rates, type distribution, latency")
```

**Interview Tips:**
- Sample 1-5% of predictions for human review (spot-checking)
- Data drift detection: compare entity distributions over rolling windows
- Model retraining trigger: accuracy drops below threshold on sampled data
- Log everything: input text, predictions, confidence, latency

---

## Question 38
- [ ] Done
**What techniques work best for NER in texts with complex formatting or structure?**
**Answer:**

Structured documents (tables, forms, PDFs, HTML) require understanding layout and formatting, not just text. Entities may span cells, be defined by position, or use formatting as a signal (bold = header). Solutions: **layout-aware models** (LayoutLM), **structure-preserving extraction**, and **format-specific preprocessing**.

| Document Type | Challenge | Solution |
|--------------|-----------|----------|
| Tables | Entities in cells | Cell-aware NER (LayoutLM) |
| PDFs | Text extraction order | OCR + layout analysis |
| Forms | Key-value pairs | Form understanding models |
| HTML | Nested tags | DOM-aware extraction |

```python
# Pipeline: Structured document -> Extract text + layout -> Layout-aware NER

def extract_from_html_table(html):
    """Extract entities from HTML tables preserving structure."""
    from html.parser import HTMLParser
    import re
    
    # Strip tags but preserve cell boundaries
    clean = re.sub(r'</?tr[^>]*>', '\n', html)
    clean = re.sub(r'</?td[^>]*>', ' | ', clean)
    clean = re.sub(r'<[^>]+>', '', clean)
    return clean.strip()

html = "<table><tr><td>John Smith</td><td>CEO</td><td>Apple Inc</td></tr></table>"
print(extract_from_html_table(html))
# John Smith | CEO | Apple Inc

# For advanced document NER, use LayoutLM:
# from transformers import AutoModelForTokenClassification
# model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
print("LayoutLM uses text + 2D position info for document NER")
```

**Interview Tips:**
- LayoutLM (Microsoft) combines text + bounding box + image for document NER
- LayoutLMv3 is multimodal (text + layout + vision) and state-of-the-art
- For PDFs, use OCR (Tesseract) then LayoutLM for NER
- Table understanding requires cell-level entity extraction, not row-level

---

## Question 39
- [ ] Done
**How do you handle NER optimization when balancing precision and recall?**
**Answer:**

Precision-recall trade-off is fundamental in NER. Precision = correct entities / predicted entities. Recall = correct entities / actual entities. The operating point depends on the application: redaction needs recall, search needs precision.

| Metric | Formula | Optimize When |
|--------|---------|---------------|
| Precision | TP / (TP + FP) | False positives are costly |
| Recall | TP / (TP + FN) | Missing entities is costly |
| F1 | 2 × P × R / (P + R) | Balanced need |
| Fβ | (1+β²) × P × R / (β²P + R) | Weighted preference |

```python
# Pipeline: NER model -> Threshold tuning -> Precision-recall curve

import numpy as np

def precision_recall_at_threshold(confidences, labels, threshold):
    """Compute precision and recall at a confidence threshold."""
    predictions = [1 if c >= threshold else 0 for c in confidences]
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

confidences = [0.95, 0.87, 0.72, 0.91, 0.45, 0.33, 0.88, 0.60]
labels = [1, 1, 1, 1, 0, 0, 1, 0]

for t in [0.3, 0.5, 0.7, 0.9]:
    p, r = precision_recall_at_threshold(confidences, labels, t)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"Threshold={t:.1f}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
```

**Interview Tips:**
- Threshold tuning is the primary lever for precision-recall balance
- F2 score weights recall 2x more than precision (use for redaction)
- F0.5 score weights precision 2x more (use for search indexing)
- CoNLL uses strict span-level F1: both boundaries and type must match exactly

---

## Question 40
- [ ] Done
**What strategies help with NER for entities in emerging text types and platforms?**
**Answer:**

New platforms (TikTok, Threads, Discord) generate novel text formats with platform-specific conventions. Adaptation requires **platform-specific preprocessing**, **domain adaptation**, and **continuous data collection** from new sources.

| Platform | Unique Features | NER Challenges |
|----------|----------------|----------------|
| TikTok | Short captions, hashtags | Extreme abbreviation |
| Discord | Emotes, markdown, channels | Custom formatting |
| Reddit | Subreddit jargon, flairs | Community-specific terms |
| Voice transcripts | Disfluencies, no punctuation | No capitalization cues |

```python
# Pipeline: Platform-specific preprocessing -> Adapted NER -> Platform entities

def preprocess_for_platform(text, platform):
    """Platform-specific text normalization for NER."""
    import re
    if platform == 'discord':
        text = re.sub(r'<:\w+:\d+>', '', text)  # Remove custom emotes
        text = re.sub(r'<@!?\d+>', 'USER', text)  # Normalize mentions
    elif platform == 'tiktok':
        text = re.sub(r'#(\S+)', lambda m: ' '.join(re.findall('[A-Z][^A-Z]*', m.group(1))), text)
    elif platform == 'reddit':
        text = re.sub(r'r/(\w+)', 'subreddit_\\1', text)  # Subreddit references
        text = re.sub(r'u/(\w+)', 'USER', text)  # User references
    return text

print(preprocess_for_platform('#ElonMuskVisitsBerlin', 'tiktok'))
print(preprocess_for_platform('<@123456> check r/MachineLearning', 'reddit'))
```

**Interview Tips:**
- Each platform has unique text conventions requiring specific preprocessing
- Continuously collect and annotate data from new platforms
- Pre-trained models on social media (BERTweet) transfer reasonably well
- Platform-specific entity types may be needed (hashtag entities, mention entities)

---

## Question 41
- [ ] Done
**How do you implement cross-lingual transfer learning for multilingual NER?**
**Answer:**

Cross-lingual NER transfers knowledge from high-resource languages (English) to low-resource languages (Swahili, Yoruba) using **multilingual models** that share representations across languages. Key: **multilingual pre-training** + **zero-shot or few-shot transfer**.

| Transfer Strategy | Target Data Needed | Performance |
|-------------------|-------------------|-------------|
| Zero-shot (English-only) | None | 60-80% of supervised |
| Few-shot (10-50 examples) | Minimal | 80-90% of supervised |
| Translate-train | MT system | 85-95% of supervised |
| Translate-test | MT system | 75-90% of supervised |

```python
# Pipeline: English NER data -> Multilingual model -> Apply to target language

from transformers import pipeline

# Model: XLM-R trained on English NER
ner = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="simple")

# Zero-shot transfer: works on languages never seen during NER training
test_texts = {
    'English': 'Barack Obama visited Berlin.',
    'German': 'Angela Merkel besuchte Washington.',
    'French': 'Emmanuel Macron est à Tokyo.',
    'Spanish': 'El presidente viajó a Buenos Aires.',
}

for lang, text in test_texts.items():
    entities = ner(text)
    ent_str = ', '.join(f"{e['word']}({e['entity_group']})" for e in entities)
    print(f"{lang:10s}: {ent_str}")
```

**Interview Tips:**
- XLM-RoBERTa's shared subword vocabulary enables cross-lingual transfer
- Translate-train: translate English NER data to target language, train on translations
- WikiANN provides silver-standard NER data in 282 languages
- Performance gap between zero-shot and supervised is 10-20% F1 on average

---

## Question 42
- [ ] Done
**What approaches work best for NER with minimal false positive rates?**
**Answer:**

Minimizing false positives is critical when NER feeds into high-stakes downstream systems (legal documents, financial reports). Techniques: **high confidence thresholds**, **ensemble consensus**, **post-processing filters**, and **negative mining** during training.

| Strategy | False Positive Reduction | Trade-off |
|----------|------------------------|-----------|
| High threshold (>0.95) | High | Lower recall |
| Ensemble voting | High | Higher compute cost |
| Type-specific rules | Medium | Rule maintenance |
| Negative mining | High | Longer training |
| Post-processing filters | Medium | Pipeline complexity |

```python
# Pipeline: NER -> High-confidence filter -> Rule-based validation -> Output

def minimize_false_positives(entities, min_confidence=0.95, min_length=2):
    """Filter NER predictions to minimize false positives."""
    filtered = []
    for e in entities:
        # High confidence threshold
        if e.get('score', 0) < min_confidence:
            continue
        # Length filter (single-char entities are usually FP)
        if len(e['word'].strip()) < min_length:
            continue
        # Type-specific validation
        if e['entity_group'] == 'PER' and not any(c.isalpha() for c in e['word']):
            continue  # Person names must contain letters
        filtered.append(e)
    return filtered

entities = [
    {'word': 'Obama', 'entity_group': 'PER', 'score': 0.99},
    {'word': 'the', 'entity_group': 'PER', 'score': 0.52},  # FP
    {'word': '.', 'entity_group': 'ORG', 'score': 0.61},    # FP
    {'word': 'Google', 'entity_group': 'ORG', 'score': 0.97},
]
print(minimize_false_positives(entities))
```

**Interview Tips:**
- Negative mining: train on hard negatives (common FP examples) to reduce errors
- Post-processing rules are underrated — simple filters catch many common FPs
- Calibrate model confidence scores before using thresholds
- Entity type-specific validation (e.g., locations must be in gazetteer) reduces FPs

---

## Question 43
- [ ] Done
**How do you handle NER integration with knowledge graphs and databases?**
**Answer:**

Integrating NER with knowledge graphs (KGs) creates structured knowledge from unstructured text. The pipeline: **NER → Entity Linking → Relation Extraction → KG population**. NER identifies mentions, entity linking grounds them to KG nodes, and relation extraction connects them.

| Component | Input | Output |
|-----------|-------|--------|
| NER | Raw text | Entity mentions |
| Entity Linking | Mentions + KG | KB-grounded entities |
| Relation Extraction | Entity pairs + text | Relations |
| KG Population | Entities + relations | New KG triples |

```python
# Pipeline: Text -> NER -> Entity Linking -> Relation Extraction -> KG triple

def text_to_knowledge_graph(text, ner_model, linker, re_model):
    """End-to-end text to knowledge graph pipeline."""
    # Step 1: NER
    entities = ner_model(text)
    # Step 2: Entity Linking
    for e in entities:
        e['kb_id'] = linker.link(e['word'], text)
    # Step 3: Relation Extraction for entity pairs
    triples = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i+1:]:
            relation = re_model.predict(text, e1, e2)
            if relation != 'none':
                triples.append((e1['kb_id'], relation, e2['kb_id']))
    return triples

# Example output:
# Text: "Elon Musk is the CEO of Tesla, headquartered in Austin."
# Triples: [(Q317521, 'CEO_of', Q478214), (Q478214, 'headquartered_in', Q16559)]
print("NER + EL + RE = Knowledge Graph population pipeline")
```

**Interview Tips:**
- Wikidata QIDs are the standard for entity linking (e.g., Q317521 = Elon Musk)
- Neo4j and Amazon Neptune are popular graph databases for storing KGs
- REBEL (relation extraction) models extract triples end-to-end
- Knowledge graph completion uses NER-extracted entities to enrich existing KGs

---

## Question 44
- [ ] Done
**What techniques help with NER for entities requiring temporal or spatial context?**
**Answer:**

Temporal entities (dates, durations, events) and spatial entities (coordinates, relative locations) require specialized parsing beyond standard NER. Solutions: **temporal taggers** (SUTime, HeidelTime), **spatial reasoning models**, and **normalized representations**.

| Entity Type | Example | Normalization |
|------------|---------|---------------|
| Date | "next Friday" | ISO 8601: 2024-01-19 |
| Duration | "for 3 weeks" | P3W (ISO duration) |
| Relative location | "5 miles north of Austin" | Geocoordinates |
| Event time | "during WWII" | 1939-1945 |

```python
# Pipeline: Text -> NER -> Temporal/spatial normalization

import re
from datetime import datetime, timedelta

def normalize_temporal(text, reference_date=None):
    """Extract and normalize temporal expressions."""
    if reference_date is None:
        reference_date = datetime.now()
    
    patterns = {
        r'next (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)': 'relative_day',
        r'(\d{1,2})/(\d{1,2})/(\d{2,4})': 'date',
        r'(yesterday|today|tomorrow)': 'relative',
        r'(\d+) (days?|weeks?|months?) ago': 'duration_past',
    }
    
    for pattern, ptype in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {'text': match.group(), 'type': ptype, 'normalized': str(reference_date.date())}
    return None

print(normalize_temporal("The meeting is next Friday."))
print(normalize_temporal("I visited 3 weeks ago."))
```

**Interview Tips:**
- SUTime (Stanford) and HeidelTime are standard temporal taggers
- ISO 8601 is the standard for temporal normalization
- Spatial NER often requires geocoding (address → coordinates)
- Temporal expressions are highly language-dependent ("next Friday" changes meaning daily)

---

## Question 45
- [ ] Done
**How do you implement customizable NER systems for different user needs?**
**Answer:**

Customizable NER allows end users (non-ML engineers) to define entity types, provide examples, and adjust model behavior. Key: **configuration-driven design**, **annotation interfaces**, and **modular architecture** separating preprocessing, model, and post-processing.

| Customization Level | User Interface | Technical Implementation |
|-------------------|---------------|------------------------|
| Entity types | Dropdown/config file | Dynamic output layer |
| Gazetteers | Upload CSV/list | Dictionary matching layer |
| Confidence threshold | Slider | Post-processing filter |
| Domain model | Model selection | Pre-trained model registry |

```python
# Pipeline: User config -> Build NER pipeline -> Custom extraction

import json

class ConfigurableNER:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def build_pipeline(self):
        """Build NER pipeline from config."""
        from transformers import pipeline
        model = self.config.get('model', 'dslim/bert-base-NER')
        threshold = self.config.get('confidence_threshold', 0.7)
        entity_types = self.config.get('entity_types', None)  # Filter types
        
        ner = pipeline('ner', model=model, aggregation_strategy='simple')
        return lambda text: [
            e for e in ner(text)
            if e['score'] >= threshold
            and (entity_types is None or e['entity_group'] in entity_types)
        ]

# Example config:
config = {
    'model': 'dslim/bert-base-NER',
    'confidence_threshold': 0.9,
    'entity_types': ['PER', 'ORG'],
    'gazetteers': {'PRODUCT': ['iPhone', 'Galaxy', 'Pixel']}
}
print(json.dumps(config, indent=2))
```

**Interview Tips:**
- Config-driven NER is essential for SaaS NER products
- Prodigy (spaCy) provides annotation UI for custom NER
- A/B testing different NER configurations measures impact
- API design: accept entity type filter, confidence threshold, model version

---

## Question 46
- [ ] Done
**What strategies work best for NER in streaming text processing applications?**
**Answer:**

Streaming NER processes text in real-time as it arrives (chat messages, news feeds, social media). Challenges: **low latency**, **stateful processing** across message boundaries, and **dynamic entity context**.

| Requirement | Solution |
|------------|----------|
| Low latency (<100ms) | Lightweight model (spaCy, ONNX) |
| Cross-message context | Entity memory/state |
| Throughput | Micro-batching |
| Fault tolerance | Checkpoint + replay |

```python
# Pipeline: Message stream -> Buffer -> Micro-batch NER -> Emit entities

import time
from collections import deque

class StreamingNER:
    def __init__(self, model, batch_size=16, flush_interval=0.1):
        self.model = model
        self.buffer = deque()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.entity_memory = {}  # Track entities across stream
    
    def process_message(self, text, message_id):
        self.buffer.append((message_id, text))
        if len(self.buffer) >= self.batch_size:
            return self._flush()
        return []
    
    def _flush(self):
        if not self.buffer:
            return []
        batch = [text for _, text in self.buffer]
        ids = [mid for mid, _ in self.buffer]
        results = self.model(batch)  # Batch inference
        self.buffer.clear()
        return list(zip(ids, results))

print("Streaming NER: buffer messages, micro-batch for efficiency")
print("Kafka + Flink/Spark Streaming are common infrastructure choices")
```

**Interview Tips:**
- Micro-batching (buffer N messages, batch inference) balances latency and throughput
- spaCy on CPU achieves <5ms per message for real-time NER
- Kafka + Flink is the standard streaming infrastructure for NER at scale
- Entity state tracking across messages enables coreference resolution

---

## Question 47
- [ ] Done
**How do you handle NER quality benchmarking across different model architectures?**
**Answer:**

Fair NER benchmarking compares models on the same data, metrics, and evaluation protocol. Key: **standardized datasets**, **consistent evaluation** (span-level F1), **statistical significance testing**, and **controlled experiments**.

| Benchmark | Domain | Entity Types | Size |
|-----------|--------|-------------|------|
| CoNLL-2003 | News | PER, LOC, ORG, MISC | 23K sentences |
| OntoNotes 5.0 | Mixed | 18 types | 76K sentences |
| WNUT-2017 | Social | 6 types | 5.7K tweets |
| ACE-2005 | News/Web | 7 types + nested | 33K sentences |

```python
# Pipeline: Models -> Same test data -> Standardized metrics -> Comparison

from seqeval.metrics import classification_report, f1_score

def benchmark_ner(models, test_data):
    """Compare NER models on the same benchmark."""
    results = {}
    for model_name, model in models.items():
        predictions = [model.predict(tokens) for tokens, _ in test_data]
        references = [labels for _, labels in test_data]
        f1 = f1_score(references, predictions)
        results[model_name] = f1
        print(f"{model_name:30s}: F1 = {f1:.4f}")
    
    # Statistical significance (bootstrap)
    print("\nUse bootstrap test for significance (p < 0.05)")
    return results

print("Standard: CoNLL-2003 with seqeval span-level F1")
print("Always report: entity-level precision, recall, F1 per type")
```

**Interview Tips:**
- seqeval is the standard library for NER evaluation (span-level F1)
- CoNLL-2003 is the most common benchmark; state-of-the-art >93% F1
- Always report per-entity-type F1, not just macro F1
- Use bootstrap resampling for statistical significance (paired test)

---

## Question 48
- [ ] Done
**What approaches help with NER for entities with evolving definitions or categories?**
**Answer:**

Entity categories evolve: "social media company" didn't exist 20 years ago, "cryptocurrency exchange" is recent. Handling evolving definitions requires **continuous monitoring**, **schema versioning**, and **adaptive models**.

| Challenge | Example | Solution |
|-----------|---------|----------|
| New categories | "AI company" becomes distinct from "tech company" | Add new label, retrain |
| Category merger | "tablet" and "laptop" merging into "2-in-1" | Update guidelines |
| Boundary shift | "startup" → "scaleup" → "enterprise" | Temporal label versioning |
| Definition drift | "cloud" (weather → computing) | Context-dependent classification |

```python
# Pipeline: Track entity definitions over time -> Detect drift -> Update schema

class EvolvingNERSchema:
    def __init__(self):
        self.schema_versions = []
        self.current_labels = set()
    
    def add_version(self, version_id, labels, changes):
        self.schema_versions.append({
            'version': version_id,
            'labels': labels,
            'changes': changes
        })
        self.current_labels = set(labels)
    
    def migrate_annotations(self, old_version, new_version, data):
        """Migrate annotations between schema versions."""
        old_schema = self.schema_versions[old_version]
        new_schema = self.schema_versions[new_version]
        mappings = new_schema.get('changes', {}).get('renamed', {})
        return [{**d, 'label': mappings.get(d['label'], d['label'])} for d in data]

schema = EvolvingNERSchema()
schema.add_version('v1', ['PER', 'ORG', 'LOC'], {})
schema.add_version('v2', ['PER', 'ORG', 'LOC', 'PRODUCT', 'EVENT'], {'added': ['PRODUCT', 'EVENT']})
print(f"Current labels: {schema.current_labels}")
```

**Interview Tips:**
- Schema versioning is essential for annotation consistency over time
- Annotation migration: re-label old data when categories change
- Periodic human review detects category drift before model accuracy degrades
- Few-shot learning enables quick addition of new entity categories

---

## Question 49
- [ ] Done
**How do you implement efficient storage and retrieval of NER results?**
**Answer:**

Efficient NER storage enables fast retrieval for search, analytics, and knowledge graph population. Key decisions: **storage format** (standoff annotations, inline, database), **indexing strategy**, and **query patterns**.

| Storage Format | Pros | Cons |
|---------------|------|------|
| Standoff (offset-based) | Original text preserved | Offset maintenance |
| Inline (tagged text) | Self-contained | Hard to query |
| Database (relational) | Fast queries, relationships | Schema design needed |
| JSON/document store | Flexible schema | Less structured queries |

```python
# Pipeline: NER results -> Structured storage -> Indexed retrieval

import json
from collections import defaultdict

class NERStorage:
    def __init__(self):
        self.documents = {}  # doc_id -> text
        self.entities = []   # list of entity records
        self.index = defaultdict(list)  # entity_type -> [entity_records]
    
    def store(self, doc_id, text, entities):
        self.documents[doc_id] = text
        for e in entities:
            record = {'doc_id': doc_id, 'word': e['word'], 'type': e['entity_group'],
                      'start': e.get('start', 0), 'end': e.get('end', 0), 'score': e.get('score', 0)}
            self.entities.append(record)
            self.index[e['entity_group']].append(record)
    
    def query_by_type(self, entity_type):
        return self.index.get(entity_type, [])
    
    def query_by_text(self, entity_text):
        return [e for e in self.entities if entity_text.lower() in e['word'].lower()]

store = NERStorage()
store.store('doc1', 'Obama visited Google', [{'word': 'Obama', 'entity_group': 'PER', 'score': 0.99}])
print(store.query_by_type('PER'))
```

**Interview Tips:**
- Standoff annotations (character offsets) are the standard for NER storage
- Elasticsearch is ideal for entity search and aggregation
- PostgreSQL JSONB enables flexible entity storage with SQL queries
- Index by entity type, entity text, and document for fast retrieval

---

## Question 50
- [ ] Done
**What techniques work best for balancing NER accuracy with computational efficiency?**
**Answer:**

Balancing NER accuracy vs. efficiency requires choosing the right model size, optimization technique, and inference strategy for your latency and accuracy requirements.

| Accuracy Tier | Model | Latency | F1 (CoNLL) |
|--------------|-------|---------|------------|
| Highest | BERT-large + CRF | ~100ms | ~93% |
| High | BERT-base | ~30ms | ~91% |
| Good | DistilBERT | ~15ms | ~90% |
| Fast | spaCy (sm) | ~1ms | ~85% |
| Fastest | Rule-based + gazetteer | <1ms | ~70% |

```python
# Pipeline: Select model based on requirements -> Optimize -> Deploy

def select_ner_model(max_latency_ms, min_f1, has_gpu=False):
    """Select optimal NER model for requirements."""
    models = [
        {'name': 'bert-large-NER', 'latency_gpu': 30, 'latency_cpu': 150, 'f1': 0.93},
        {'name': 'bert-base-NER', 'latency_gpu': 15, 'latency_cpu': 60, 'f1': 0.91},
        {'name': 'distilbert-NER', 'latency_gpu': 8, 'latency_cpu': 30, 'f1': 0.90},
        {'name': 'spacy-lg', 'latency_gpu': 2, 'latency_cpu': 5, 'f1': 0.87},
        {'name': 'spacy-sm', 'latency_gpu': 1, 'latency_cpu': 2, 'f1': 0.85},
    ]
    
    latency_key = 'latency_gpu' if has_gpu else 'latency_cpu'
    valid = [m for m in models if m[latency_key] <= max_latency_ms and m['f1'] >= min_f1]
    if valid:
        best = max(valid, key=lambda m: m['f1'])
        print(f"Recommended: {best['name']} (F1={best['f1']}, latency={best[latency_key]}ms)")
        return best
    print("No model meets both requirements. Relax constraints.")
    return None

select_ner_model(max_latency_ms=50, min_f1=0.88, has_gpu=True)
select_ner_model(max_latency_ms=10, min_f1=0.85, has_gpu=False)
```

**Interview Tips:**
- spaCy is the go-to for production NER with latency constraints
- DistilBERT gives ~97% of BERT accuracy at 40% of parameters
- ONNX + quantization can 4x speed up BERT-based NER
- Profile your actual latency — don't assume based on parameter count

---


---

# --- POS Tagging Questions (from 08_nlp/03_pos_tagging) ---

# Part-of-Speech (POS) Tagging - Theory Questions

## Question 1
- [ ] Done
**How do you handle POS tagging for morphologically rich languages with complex inflectional systems?**
**Answer:**

Morphologically rich languages (Turkish, Finnish, Hungarian, Arabic) have extensive inflectional systems where a single root can generate hundreds of surface forms, making POS tagging significantly harder than in English.

**Core Concepts:**

| Concept | Description | Example |
|---|---|---|
| Morphological Decomposition | Breaking words into morphemes before tagging | Turkish "evlerden" → ev+ler+den (house+plural+from) |
| Subword Tokenization | Using BPE/WordPiece to handle inflected forms | Captures shared morphological patterns |
| Character-Level Models | CNNs/LSTMs over characters for morphological features | Learns inflectional suffixes automatically |
| Joint Morphological Analysis | Combining POS tagging with morphological parsing | Predicts POS + morphological features together |
| Paradigm Tables | Leveraging known inflection paradigms as constraints | Noun declension tables guide predictions |

**Python Code Example:**
```python
# Pipeline: Subword-aware POS tagger for morphologically rich languages
import torch
import torch.nn as nn

class MorphAwarePOSTagger(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.char_embed = nn.Embedding(char_vocab_size, 30)
        self.char_lstm = nn.LSTM(30, 50, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(embed_dim + 100, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, words, chars):
        word_emb = self.word_embed(words)
        # Character-level features capture morphological patterns
        char_emb = self.char_embed(chars)
        batch, seq_len, char_len, _ = char_emb.shape
        char_emb = char_emb.view(-1, char_len, 30)
        _, (h, _) = self.char_lstm(char_emb)
        char_feats = torch.cat([h[0], h[1]], dim=-1).view(batch, seq_len, 100)
        combined = torch.cat([word_emb, char_feats], dim=-1)
        out, _ = self.lstm(combined)
        return self.fc(out)
```

**Interview Tips:**
- Emphasize that character-level models are essential for morphologically rich languages
- Mention that joint morphological analysis + POS tagging outperforms pipeline approaches
- Discuss how subword tokenization (BPE) implicitly captures morphological structure
- Note that morphological analyzers (e.g., Apertium) can provide features to neural taggers

---

## Question 2
- [ ] Done
**What techniques work best for POS tagging in low-resource languages with limited annotated data?**
**Answer:**

Low-resource POS tagging addresses the challenge of building accurate taggers when only a few hundred or thousand annotated sentences are available, which is the reality for most of the world's 7,000+ languages.

**Core Concepts:**

| Technique | Description | Typical Gain |
|---|---|---|
| Cross-Lingual Transfer | Train on high-resource language, apply to target | +15-25% accuracy over zero-shot |
| Multilingual Pretraining | Fine-tune mBERT/XLM-R on small target data | Best overall performance |
| Annotation Projection | Project tags through parallel corpora alignments | Works without any target annotations |
| Active Learning | Strategically select sentences for annotation | 50% annotation reduction |
| Few-Shot Learning | Prototypical networks with handful of examples | Effective with <100 examples |
| Semi-Supervised Methods | Self-training on unlabeled target language text | +3-8% accuracy boost |

**Python Code Example:**
```python
# Pipeline: Cross-lingual POS tagging with multilingual BERT
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=17  # Universal POS tagset
)

# Fine-tune on small target language data (e.g., 500 sentences)
training_args = TrainingArguments(
    output_dir="./pos_low_resource",
    num_train_epochs=20,          # More epochs for small data
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

# Strategy: first fine-tune on high-resource language, then on target
# This two-stage transfer significantly improves low-resource accuracy
trainer = Trainer(model=model, args=training_args, train_dataset=target_dataset)
trainer.train()
```

**Interview Tips:**
- Highlight the Universal Dependencies (UD) project enabling cross-lingual transfer via shared tagset
- Mention that XLM-R outperforms mBERT for low-resource transfer learning
- Discuss annotation projection through word-aligned parallel texts (Bible corpus)
- Note that even 100-200 annotated sentences with transfer learning can achieve 85%+ accuracy

---

## Question 3
- [ ] Done
**How do you implement domain adaptation for POS tagging across different text genres?**
**Answer:**

Domain adaptation for POS tagging addresses accuracy drops when taggers trained on one genre (e.g., newswire) are applied to another (e.g., biomedical, legal, social media), often seeing 5-15% degradation.

**Core Concepts:**

| Strategy | Description | When to Use |
|---|---|---|
| Feature Augmentation | Add domain-specific features alongside general ones | When domain vocabulary differs significantly |
| Fine-Tuning | Continue training on small in-domain data | When some labeled target data exists |
| Domain-Adversarial Training | Learn domain-invariant representations | When only unlabeled target data available |
| Self-Training | Use high-confidence predictions as pseudo-labels | Unsupervised domain adaptation |
| Curriculum Learning | Gradually shift from source to target domain | Smooth transition between domains |

**Python Code Example:**
```python
# Pipeline: Domain-adversarial POS tagger for genre adaptation
import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainAdaptivePOSTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tags, num_domains=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pos_classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.domain_classifier = nn.Linear(hidden_dim * 2, num_domains)

    def forward(self, x, alpha=1.0):
        encoded, _ = self.encoder(x)
        pos_logits = self.pos_classifier(encoded)
        reversed_feats = GradientReversalLayer.apply(encoded, alpha)
        domain_logits = self.domain_classifier(reversed_feats)
        return pos_logits, domain_logits
```

**Interview Tips:**
- Explain that domain shift in POS tagging comes from vocabulary, syntax, and style differences
- Mention that self-training with confidence thresholds is simple and effective
- Discuss how pre-trained language models (BERT) already provide some domain robustness
- Note that the biggest gains come from even small amounts (100-500 sentences) of in-domain data

---

## Question 4
- [ ] Done
**What strategies help with handling ambiguous words that can have multiple POS tags?**
**Answer:**

POS ambiguity is the core challenge in tagging — words like "run" (verb/noun), "light" (adjective/noun/verb), or "that" (determiner/pronoun/conjunction) can have multiple valid tags depending on context. ~40% of English word types are ambiguous.

**Core Concepts:**

| Strategy | Description | Example |
|---|---|---|
| Contextual Embeddings | Use surrounding words to disambiguate | BERT embeddings differ for "bank" in financial vs. river context |
| Sequence Labeling | CRF/HMM models tag constraints (e.g., DET→NOUN likely) | "The run" → DET NOUN (not DET VERB) |
| N-gram Context | Local context windows capture syntactic patterns | Previous tag + current word features |
| Morphological Cues | Word shape/suffix helps disambiguate | "-ing" suffix strongly suggests verb |
| Selectional Preferences | Verbs select for specific argument POS patterns | "quickly run" → ADV VERB |
| Ensemble Methods | Combine multiple taggers for difficult cases | Vote on ambiguous tokens |

**Python Code Example:**
```python
# Pipeline: Analyzing and resolving POS ambiguity with spaCy and context
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def analyze_pos_ambiguity(word, sentences):
    """Analyze how a word's POS varies across contexts."""
    tag_contexts = {}
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            if token.text.lower() == word.lower():
                tag = token.pos_
                if tag not in tag_contexts:
                    tag_contexts[tag] = []
                tag_contexts[tag].append(sent)
    return tag_contexts

# Example: "light" is ambiguous
sentences = [
    "The light was bright.",          # NOUN
    "Please light the candle.",       # VERB
    "She wore a light jacket.",       # ADJ
]
results = analyze_pos_ambiguity("light", sentences)
for tag, sents in results.items():
    print(f"POS={tag}: {sents}")
```

**Interview Tips:**
- State that contextual language models (BERT) have largely solved POS ambiguity for well-resourced languages
- Mention that CRF layers enforce tag transition constraints (e.g., two determiners rarely appear consecutively)
- Discuss the difference between type ambiguity (dictionary) and token ambiguity (in context)
- Note that most tokens (95%+) in context have only one plausible tag — ambiguity is concentrated in common words

---

## Question 5
- [ ] Done
**How do you design POS taggers that can handle out-of-vocabulary words effectively?**
**Answer:**

Out-of-vocabulary (OOV) words — words not seen during training — are a major source of POS tagging errors. OOV rates range from 2-5% in similar-domain text to 15-30% in cross-domain settings.

**Core Concepts:**

| Strategy | Description | Effectiveness |
|---|---|---|
| Character-Level Models | CNN/LSTM over characters to capture morphological patterns | Best overall OOV handling |
| Subword Tokenization | BPE/WordPiece breaks OOV into known subwords | Eliminates true OOV entirely |
| Word Shape Features | Capitalization, digit patterns, hyphenation | "McDonalds" → Xx+ pattern suggests PROPN |
| Suffix/Prefix Features | Last 3-4 characters strongly predict POS | "-tion" → NOUN, "-ly" → ADV |
| Pre-trained Embeddings | FastText subword embeddings for unseen words | Provides approximate representations |
| Morphological Analyzers | Rule-based decomposition for known affixes | Language-specific but reliable |

**Python Code Example:**
```python
# Pipeline: OOV-robust POS tagger using character features and word shape
import torch
import torch.nn as nn

class OOVRobustTagger(nn.Module):
    def __init__(self, char_vocab, word_vocab, num_tags):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab, 30)
        self.char_cnn = nn.Conv1d(30, 50, kernel_size=3, padding=1)
        self.word_embed = nn.Embedding(word_vocab, 100)
        # Word shape features: 8 binary features
        self.shape_proj = nn.Linear(8, 20)
        self.lstm = nn.LSTM(100 + 50 + 20, 128, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, num_tags)

    def get_word_shape(self, word):
        return [
            word[0].isupper(), word.isupper(), word.islower(),
            word.isdigit(), any(c.isdigit() for c in word),
            '-' in word, '_' in word, len(word) > 10
        ]

    def forward(self, words, chars, shapes):
        w_emb = self.word_embed(words)  # Falls back to UNK for OOV
        c_emb = self.char_embed(chars).permute(0, 2, 1)
        c_feats = self.char_cnn(c_emb).max(dim=-1)[0].unsqueeze(1)
        s_feats = self.shape_proj(shapes.float())
        combined = torch.cat([w_emb, c_feats.expand_as(w_emb[:,:,:50]), s_feats], dim=-1)
        out, _ = self.lstm(combined)
        return self.classifier(out)
```

**Interview Tips:**
- Stress that character-level representations are the most important OOV strategy
- Mention that FastText's subword embeddings provide vectors for any word
- Discuss how pre-trained transformers (BERT) with WordPiece/BPE essentially eliminate OOV
- Note that word shape features (capitalization, digits, hyphens) are cheap but highly informative

---

## Question 6
- [ ] Done
**What approaches work best for POS tagging in noisy or informal text environments?**
**Answer:**

Noisy text (tweets, SMS, chat messages, OCR output) violates the assumptions of standard POS taggers trained on edited text — containing misspellings, abbreviations, slang, non-standard punctuation, and creative syntax.

**Core Concepts:**

| Approach | Description | Example |
|---|---|---|
| Text Normalization | Map noisy forms to standard before tagging | "u r gr8" → "you are great" |
| Noise-Aware Training | Train/fine-tune on noisy annotated data | Twitter POS corpora (Ritter et al.) |
| Extended Tagsets | Add tags for social media elements | USR (@ mentions), HT (hashtags), URL, EMO (emoticons) |
| Data Augmentation | Inject synthetic noise into clean training data | Random char swaps, abbreviations |
| Robust Embeddings | Embeddings trained on noisy text corpora | Twitter GloVe, BERTweet |
| Spelling Correction | Pre-process with edit-distance spell checker | Correct before tagging |

**Python Code Example:**
```python
# Pipeline: POS tagging for social media text with normalization
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

# Simple normalization rules for social media
def normalize_text(text):
    replacements = {
        r'\bu\b': 'you', r'\br\b': 'are', r'\bur\b': 'your',
        r'\bw/': 'with', r'\bb4\b': 'before', r'\bthx\b': 'thanks',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # sooooo → soo
    return text

# Use BERTweet - pre-trained on Twitter data
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForTokenClassification.from_pretrained("vinai/bertweet-base")

tweet = "@john u r sooo funny lol #comedy"
normalized = normalize_text(tweet)
print(f"Original: {tweet}")
print(f"Normalized: {normalized}")
```

**Interview Tips:**
- Mention that dedicated social media tagsets (ARK, Ritter) add tags for URLs, hashtags, emoticons
- Discuss normalize-then-tag vs. train-on-noisy-data approaches (latter is generally better)
- Highlight BERTweet as a strong pre-trained model specifically for Twitter text
- Note that noise-robust POS tagging is critical for real-world NLP applications

---

## Question 7
- [ ] Done
**How do you handle POS tagging for languages with non-standard orthography or spelling?**
**Answer:**

Non-standard orthography includes languages without fixed spelling conventions (many African/Asian languages), historical texts with variable spelling, and dialects without standard written forms. This creates massive vocabulary expansion where the same word has many surface forms.

**Core Concepts:**

| Challenge | Strategy | Example |
|---|---|---|
| Spelling Variation | Phonetic normalization / canonicalization | "colour"/"color" → single form |
| No Standard Script | Transliteration to common script | Romanization of unscripted languages |
| Historical Texts | Spelling normalization rules | Middle English "ye" → "the" |
| Dialectal Writing | Dialect-to-standard mapping | Swiss German → Standard German |
| Character-Level Robustness | Char-CNN/LSTM encoding | Learns across spelling variants |
| Phonological Features | Sound-based rather than spelling-based features | IPA representations |

**Python Code Example:**
```python
# Pipeline: Spelling-robust POS tagging with phonetic normalization
import re
from collections import defaultdict

class SpellingNormalizer:
    def __init__(self):
        self.canonical_forms = {}  # Maps variants to canonical
        self.rules = [
            (r'ou', 'o'),   # colour → color
            (r'ise', 'ize'),  # recognise → recognize
            (r'(.)\1{2,}', r'\1'),  # goood → god
        ]

    def add_variant_mapping(self, variants, canonical):
        for v in variants:
            self.canonical_forms[v.lower()] = canonical.lower()

    def normalize(self, word):
        lower = word.lower()
        if lower in self.canonical_forms:
            return self.canonical_forms[lower]
        # Apply rule-based normalization
        result = lower
        for pattern, replacement in self.rules:
            result = re.sub(pattern, replacement, result)
        return result

    def normalize_sentence(self, tokens):
        return [self.normalize(t) for t in tokens]

# Usage: normalize before feeding to POS tagger
normalizer = SpellingNormalizer()
normalizer.add_variant_mapping(['colour', 'colur', 'coulor'], 'color')
tokens = ['The', 'colour', 'was', 'beautifull']
normalized = normalizer.normalize_sentence(tokens)
print(normalized)  # ['the', 'color', 'was', 'beautifull']
```

**Interview Tips:**
- Emphasize that character-level models are naturally robust to spelling variation
- Mention the VARD tool for historical English spelling normalization
- Discuss how multilingual models (mBERT) handle orthographic variation implicitly
- Note that phonetic encodings (Soundex, Metaphone) can group spelling variants

---

## Question 8
- [ ] Done
**What techniques help with POS tagging in multilingual or code-switched texts?**
**Answer:**

Code-switching occurs when speakers alternate between languages within a conversation or sentence (e.g., "I went to the tienda to buy some leche"). POS tagging code-switched text requires handling multiple languages' grammars simultaneously.

**Core Concepts:**

| Technique | Description | Application |
|---|---|---|
| Language Identification | Tag each token's language first, then POS | Pipeline: LID → POS per language |
| Multilingual Models | Single model trained on multiple languages | mBERT/XLM-R handle code-switching natively |
| Joint LID+POS | Predict language and POS simultaneously | Captures cross-lingual syntactic patterns |
| Shared Tagset | Universal POS tags across languages | Universal Dependencies (UPOS) |
| Code-Switch Corpora | Train on annotated code-switched data | LinCE benchmark, CALCS shared tasks |
| Linked Embeddings | Align embedding spaces across languages | Cross-lingual word embedding alignment |

**Python Code Example:**
```python
# Pipeline: Code-switched POS tagging with multilingual BERT
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# XLM-R handles code-switched text well due to multilingual pretraining
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=17  # Universal POS tagset
)

def tag_code_switched(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'}).logits
    predictions = torch.argmax(logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return list(zip(tokens, predictions[0].tolist()))

# Code-switched example (English-Spanish)
result = tag_code_switched("I went to the tienda to buy some leche")
```

**Interview Tips:**
- Highlight that XLM-R is the strongest baseline for code-switched NLP tasks
- Mention the importance of the universal POS tagset for cross-lingual consistency
- Discuss matrix language frame theory for understanding code-switching patterns
- Note that code-switching POS tagging is an active research area with shared tasks (CALCS)

---

## Question 9
- [ ] Done
**How do you implement active learning strategies for efficient POS tagging annotation?**
**Answer:**

Active learning for POS tagging selects the most informative sentences for human annotation, reducing labeling cost by 50-70% while maintaining accuracy. This is critical for bootstrapping taggers in new domains or languages.

**Core Concepts:**

| Strategy | Selection Criterion | Best For |
|---|---|---|
| Uncertainty Sampling | Sentences where model is least confident | Simple, effective baseline |
| Query-by-Committee | Sentences where multiple models disagree | Exploits model diversity |
| Token Entropy | Highest average per-token entropy | Targets ambiguous words |
| Information Density | Uncertain AND representative sentences | Avoids outlier selection |
| Batch-Mode AL | Select diverse batch, not just top-k uncertain | Reduces redundancy in batches |
| Expected Model Change | Sentences that would most change model parameters | Theoretically optimal |

**Python Code Example:**
```python
# Pipeline: Active learning loop for POS tagging annotation
import numpy as np
from scipy.stats import entropy

class ActivePOSLearner:
    def __init__(self, model, unlabeled_pool):
        self.model = model
        self.unlabeled = unlabeled_pool
        self.labeled = []

    def score_sentence(self, sentence):
        """Score sentence by average token uncertainty."""
        probs = self.model.predict_proba(sentence)  # (seq_len, num_tags)
        token_entropy = [entropy(p) for p in probs]
        return np.mean(token_entropy)

    def select_batch(self, batch_size=20):
        """Select most informative sentences for annotation."""
        scores = [(i, self.score_sentence(s)) for i, s in enumerate(self.unlabeled)]
        scores.sort(key=lambda x: -x[1])  # Highest uncertainty first
        selected_indices = [idx for idx, _ in scores[:batch_size]]
        return [self.unlabeled[i] for i in selected_indices]

    def update(self, newly_labeled):
        """Add newly annotated data and retrain."""
        self.labeled.extend(newly_labeled)
        self.unlabeled = [s for s in self.unlabeled if s not in newly_labeled]
        self.model.fit(self.labeled)
```

**Interview Tips:**
- Emphasize that uncertainty sampling on token entropy is the simplest effective strategy
- Mention that sentence-level selection (not token-level) is practical for annotation workflows
- Discuss the cold-start problem: initial model is random, so first batch should be diverse
- Note that active learning is especially valuable in low-resource and domain adaptation scenarios

---

## Question 10
- [ ] Done
**What strategies work best for POS tagging in specialized domains with domain-specific vocabulary?**
**Answer:**

Specialized domains (biomedical, legal, financial, scientific) introduce vocabulary unseen in general training corpora. Terms like "methylation" (bio), "estoppel" (legal), or "hedging" (finance) may not appear in standard tagsets or training data.

**Core Concepts:**

| Strategy | Description | Domain Example |
|---|---|---|
| Domain-Specific Pretraining | Pretrain LM on domain text | BioBERT, SciBERT, LegalBERT |
| Fine-Tuning | Adapt general tagger on small domain data | 500 annotated domain sentences |
| Domain Lexicons | Inject domain vocabulary with POS labels | UMLS for biomedical terms |
| Feature Engineering | Add domain-specific features | Chemical suffixes ("-ase", "-ine") |
| Transfer + Adaptation | Two-stage: general → domain fine-tuning | Most practical approach |
| Custom Tagsets | Add domain-specific POS categories | GENE, CHEMICAL, CITATION tags |

**Python Code Example:**
```python
# Pipeline: Domain-adapted POS tagging for biomedical text
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Use domain-specific pre-trained model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.2", num_labels=17
)

# Domain lexicon for POS hints
bio_lexicon = {
    'methylation': 'NOUN', 'phosphorylate': 'VERB', 'transcriptomic': 'ADJ',
    'upregulate': 'VERB', 'pathogenesis': 'NOUN', 'immunogenic': 'ADJ',
}

def domain_aware_tag(text, lexicon):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    # Override with lexicon for known domain terms
    results = []
    for token, pred in zip(tokens, pred_ids):
        if token.lower() in lexicon:
            results.append((token, lexicon[token.lower()]))
        else:
            results.append((token, model.config.id2label.get(pred.item(), "UNK")))
    return results
```

**Interview Tips:**
- Recommend domain-specific pre-trained models (BioBERT, SciBERT) as first choice
- Mention that domain lexicons provide a simple, interpretable fallback for known terms
- Discuss how suffix patterns in domains (biomedical: "-ase", "-itis", "-emia") help OOV tagging
- Note that even 200-500 annotated domain sentences can significantly boost accuracy

---

## Question 11
- [ ] Done
**How do you handle POS tagging quality control and confidence assessment?**
**Answer:**

POS tagging quality control ensures predictions meet required accuracy thresholds, identifies unreliable predictions, and provides confidence scores that downstream systems can use for decision-making.

**Core Concepts:**

| Aspect | Method | Purpose |
|---|---|---|
| Confidence Scoring | Softmax probabilities / CRF marginals | Per-token reliability estimate |
| Uncertainty Estimation | MC Dropout / ensemble disagreement | Detect predictions likely to be wrong |
| Confusion Analysis | Per-tag precision/recall matrix | Identify systematic errors |
| Human-in-the-Loop | Flag low-confidence tokens for review | Hybrid automation |
| Cross-Validation | K-fold evaluation on held-out data | Estimate generalization accuracy |
| Calibration | Temperature scaling / Platt scaling | Align confidence with actual accuracy |

**Python Code Example:**
```python
# Pipeline: POS tagging with confidence scoring and quality flags
import numpy as np
import torch
import torch.nn.functional as F

class ConfidentPOSTagger:
    def __init__(self, model, confidence_threshold=0.85):
        self.model = model
        self.threshold = confidence_threshold

    def predict_with_confidence(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)

        results = []
        for i, (pred, conf) in enumerate(zip(predictions[0], confidences[0])):
            results.append({
                'token': tokens[i],
                'tag': pred.item(),
                'confidence': conf.item(),
                'needs_review': conf.item() < self.threshold
            })
        return results

    def quality_report(self, results):
        confs = [r['confidence'] for r in results]
        flagged = [r for r in results if r['needs_review']]
        return {
            'mean_confidence': np.mean(confs),
            'min_confidence': np.min(confs),
            'flagged_count': len(flagged),
            'flagged_tokens': flagged,
        }
```

**Interview Tips:**
- Explain that softmax probabilities are often overconfident and need calibration
- Mention MC Dropout: run inference N times with dropout on, measure variance
- Discuss that CRF marginal probabilities are better calibrated than softmax for sequence labeling
- Note that inter-annotator agreement (Cohen’s kappa ~97% for POS) sets the ceiling for quality

---

## Question 12
- [ ] Done
**What approaches help with explaining POS tagging decisions for linguistic analysis?**
**Answer:**

Explainability in POS tagging helps linguists understand model decisions, debug errors, and gain insights into language patterns. This is essential for linguistic research and educational applications.

**Core Concepts:**

| Approach | Description | Output |
|---|---|---|
| Attention Visualization | Show which context tokens influence tag decisions | Heatmap over input tokens |
| Feature Importance | LIME/SHAP for local feature contributions | Per-feature contribution scores |
| Rule Extraction | Distill neural models into interpretable rules | IF suffix="-ly" AND prev_tag=VERB THEN ADV |
| Probing Classifiers | Test what linguistic info is encoded in representations | Layer-wise probing accuracy |
| Decision Boundary Analysis | Visualize embedding space clusters by POS | t-SNE/UMAP plots by tag |
| Counterfactual Explanations | Show minimal changes that flip predictions | "Running" (VERB) → "Runner" (NOUN) |

**Python Code Example:**
```python
# Pipeline: Explainable POS tagging with attention and feature analysis
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def explain_pos_prediction(text, model, tokenizer, target_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract attention weights for target token
    attentions = outputs.attentions[-1]  # Last layer
    avg_attention = attentions.mean(dim=1)[0]  # Average over heads
    token_attention = avg_attention[target_idx]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    explanation = sorted(
        zip(tokens, token_attention.tolist()),
        key=lambda x: -x[1]
    )
    return explanation[:5]  # Top 5 influential tokens

def extract_simple_rules(tagged_corpus, min_confidence=0.95):
    """Extract interpretable suffix-based POS rules from data."""
    suffix_tag_counts = {}
    for sent in tagged_corpus:
        for word, tag in sent:
            for suffix_len in [2, 3, 4]:
                if len(word) > suffix_len:
                    suffix = word[-suffix_len:].lower()
                    key = (suffix, tag)
                    suffix_tag_counts[key] = suffix_tag_counts.get(key, 0) + 1
    return suffix_tag_counts
```

**Interview Tips:**
- Mention that attention weights provide intuitive but imperfect explanations
- Discuss probing classifiers as the standard method for analyzing what models learn
- Highlight that rule extraction can create interpretable approximations of neural taggers
- Note the tension between model accuracy (complex models) and interpretability (simple rules)

---

## Question 13
- [ ] Done
**How do you implement knowledge distillation for compressing POS tagging models?**
**Answer:**

Knowledge distillation compresses a large teacher POS tagger (e.g., BERT-large) into a smaller student model (e.g., BiLSTM) that runs faster with minimal accuracy loss, enabling deployment on edge devices and real-time applications.

**Core Concepts:**

| Concept | Description | Typical Result |
|---|---|---|
| Soft Labels | Student learns from teacher's probability distribution | Captures inter-tag similarities |
| Temperature Scaling | Soften teacher's outputs to reveal dark knowledge | T=2-5 works well for POS |
| Layer-wise Distillation | Match intermediate representations | Better than output-only distillation |
| Data Augmentation | Generate more training data with teacher | Compensates for capacity gap |
| Architecture Search | Find optimal small architecture | Task-specific compression |
| Quantization + Distillation | Combine with weight quantization | 10-50x speedup possible |

**Python Code Example:**
```python
# Pipeline: Knowledge distillation from BERT to BiLSTM POS tagger
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, teacher, student, temperature=3.0, alpha=0.7):
        self.teacher = teacher
        self.student = student
        self.T = temperature
        self.alpha = alpha  # Balance between hard and soft labels

    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        # Soft loss: KL divergence with temperature
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        # Hard loss: cross-entropy with true labels
        hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                                     true_labels.view(-1))
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_step(self, batch):
        tokens, labels = batch
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(tokens)
        student_logits = self.student(tokens)
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        return loss
```

**Interview Tips:**
- Explain that temperature T=3-5 softens the distribution to reveal tag similarity structure
- Mention that distilled BiLSTM can achieve 95-98% of BERT accuracy at 10x speed
- Discuss DistilBERT as a general-purpose distilled model that works well for POS
- Note that alpha balancing (0.5-0.7 for soft loss) is a key hyperparameter

---

## Question 14
- [ ] Done
**What techniques work best for POS tagging with limited computational resources?**
**Answer:**

Resource-constrained POS tagging serves mobile, IoT, and high-throughput applications where large transformer models are impractical. The goal is maximizing accuracy within strict latency and memory budgets.

**Core Concepts:**

| Technique | Speed | Accuracy | Memory |
|---|---|---|
| Rule-Based / Regex | Fastest | 85-90% | Minimal |
| HMM / Viterbi | Very Fast | 93-95% | Low |
| Averaged Perceptron | Fast | 96-97% | Low |
| BiLSTM-CRF (small) | Moderate | 97%+ | Medium |
| Distilled BERT (TinyBERT) | Moderate | 97-98% | Medium |
| Quantized Models | Fast | Near full-precision | 4x smaller |

**Python Code Example:**
```python
# Pipeline: Lightweight POS taggers for resource-constrained environments
import numpy as np
from collections import defaultdict

class AveragedPerceptronTagger:
    """Fast, accurate POS tagger using averaged perceptron - NLTK's default."""
    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(lambda: defaultdict(float))
        self.timestamps = defaultdict(lambda: defaultdict(int))
        self.i = 0

    def predict(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat in self.weights:
                for tag, weight in self.weights[feat].items():
                    scores[tag] += weight * value
        return max(scores, key=scores.get) if scores else 'NOUN'

    def update(self, truth, guess, features):
        self.i += 1
        if truth == guess:
            return
        for feat in features:
            self.weights[feat][truth] += 1
            self.weights[feat][guess] -= 1

    def extract_features(self, tokens, i, prev_tag):
        word = tokens[i]
        return {
            'word': word, 'suffix3': word[-3:], 'suffix2': word[-2:],
            'prefix1': word[0], 'prev_tag': prev_tag,
            'is_upper': word[0].isupper(), 'is_digit': word.isdigit(),
            'prev_word': tokens[i-1] if i > 0 else '<START>',
        }
```

**Interview Tips:**
- Recommend averaged perceptron as the best speed/accuracy tradeoff for CPU-only environments
- Mention that ONNX Runtime + quantization can make BERT-based taggers production-viable
- Discuss that for most applications, a BiLSTM-CRF at 97% accuracy is sufficient and fast
- Note that model size matters more than architecture for mobile deployment

---

## Question 15
- [ ] Done
**How do you handle POS tagging for historical or archaic text varieties?**
**Answer:**

Historical POS tagging deals with texts from earlier periods where vocabulary, spelling, syntax, and even grammatical categories differ significantly from modern language. Accuracy of modern taggers on historical text drops 10-30%.

**Core Concepts:**

| Challenge | Description | Solution |
|---|---|---|
| Spelling Variation | "knight"/"knyght"/"kniht" | VARD spelling normalization |
| Archaic Vocabulary | "thee", "hath", "whence" | Period-specific lexicons |
| Different Syntax | SOV word order, case marking | Period-appropriate grammar models |
| Evolving Tagsets | Grammatical categories changed over time | Custom historical tagsets |
| OCR Errors | Digitized historical texts have noise | OCR post-correction pipeline |
| Limited Training Data | Few annotated historical corpora | Transfer from modern + adaptation |

**Python Code Example:**
```python
# Pipeline: Historical text POS tagging with normalization
import re

class HistoricalTextNormalizer:
    def __init__(self):
        # Common historical English spelling mappings
        self.mappings = {
            'ye': 'the', 'hath': 'has', 'doth': 'does',
            'thee': 'you', 'thou': 'you', 'thy': 'your',
            'hither': 'here', 'thither': 'there',
            'wherefore': 'why', 'whence': 'where',
        }
        self.suffix_rules = [
            (r'eth$', 's'),       # "runneth" → "runs"
            (r'est$', ''),        # "runnest" → "run"
            (r'ck$', 'k'),        # "musick" → "musik"
            (r'our$', 'or'),      # historical spelling
        ]

    def normalize_token(self, token):
        lower = token.lower()
        if lower in self.mappings:
            return self.mappings[lower]
        for pattern, replacement in self.suffix_rules:
            if re.search(pattern, lower):
                return re.sub(pattern, replacement, lower)
        return token

    def normalize_text(self, tokens):
        return [self.normalize_token(t) for t in tokens]

# Example: Middle English text
tokens = ['Ye', 'knyght', 'hath', 'ridden', 'hither']
normalizer = HistoricalTextNormalizer()
normalized = normalizer.normalize_text(tokens)
print(normalized)  # ['the', 'knyght', 'has', 'ridden', 'here']
```

**Interview Tips:**
- Mention VARD (Variant Detector) as the standard tool for historical English normalization
- Discuss the Penn-Helsinki Parsed Corpus of Middle English for training data
- Highlight that normalization-then-tagging pipelines work well for historical text
- Note that character-level models handle spelling variation better than word-level models

---

## Question 16
- [ ] Done
**What strategies help with POS tagging consistency across different annotation schemes?**
**Answer:**

Different corpora use different tagsets (PTB: 45 tags, CLAWS: 137 tags, Universal: 17 tags), creating interoperability challenges when combining datasets or comparing systems. Tagset mapping and universal standards address this.

**Core Concepts:**

| Strategy | Description | Benefit |
|---|---|---|
| Universal POS Tagset | 17 coarse-grained tags (UPOS) | Cross-corpora/cross-lingual compatibility |
| Tagset Mapping Tables | Manual mappings between tagsets | PTB-to-Universal, CLAWS-to-Universal |
| Hierarchical Tagsets | Coarse → fine-grained nesting | ADJ → ADJ.comparative, ADJ.superlative |
| Multi-Tagset Training | Train single model on multiple conventions | Learn shared representations |
| Annotation Guidelines | Detailed decision trees for ambiguous cases | Improve inter-annotator agreement |
| Conversion Tools | Automated tagset converters | Universal Dependencies toolkit |

**Python Code Example:**
```python
# Pipeline: Tagset mapping and conversion for POS consistency
class TagsetMapper:
    def __init__(self):
        # Penn Treebank to Universal POS mapping
        self.ptb_to_upos = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
            'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
            'DT': 'DET', 'IN': 'ADP', 'CC': 'CCONJ',
            'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON',
            'TO': 'PART', 'MD': 'AUX', 'CD': 'NUM',
        }

    def convert_ptb_to_upos(self, ptb_tag):
        return self.ptb_to_upos.get(ptb_tag, 'X')  # X for unknown

    def convert_sentence(self, tagged_sentence, source='ptb'):
        if source == 'ptb':
            return [(w, self.convert_ptb_to_upos(t)) for w, t in tagged_sentence]
        raise ValueError(f"Unknown source tagset: {source}")

    def evaluate_consistency(self, corpus1_tags, corpus2_tags):
        """Measure agreement between two tagsets after mapping."""
        matches = sum(1 for t1, t2 in zip(corpus1_tags, corpus2_tags) if t1 == t2)
        return matches / len(corpus1_tags)
```

**Interview Tips:**
- Mention Universal Dependencies (UD) as the standard for cross-lingual tagset consistency
- Discuss that fine-grained tagsets capture more info but are harder to annotate consistently
- Note that tagset mapping is lossy — fine-to-coarse loses distinctions
- Highlight that UPOS has 17 tags: NOUN, VERB, ADJ, ADV, PROPN, DET, ADP, PRON, etc.

---

## Question 17
- [ ] Done
**How do you implement online learning for POS taggers adapting to new text domains?**
**Answer:**

Online learning for POS taggers enables continuous adaptation to new domains without full retraining, processing one example at a time and updating model parameters incrementally. This is crucial for production systems encountering evolving text.

**Core Concepts:**

| Method | Description | Pros/Cons |
|---|---|---|
| Online Perceptron | Update weights on each misclassified token | Fast, simple, no learning rate |
| Online SGD | Stochastic gradient descent one example at a time | Standard for neural models |
| Elastic Weight Consolidation | Prevent catastrophic forgetting of old knowledge | Retains source domain accuracy |
| Continual Learning | Learn new domains without forgetting old ones | Maintains multi-domain capability |
| Replay Buffer | Mix old and new domain examples | Prevents drift |
| Learning Rate Scheduling | Reduce LR over time to stabilize | Prevents oscillation |

**Python Code Example:**
```python
# Pipeline: Online POS tagger with domain adaptation and replay buffer
import numpy as np
from collections import deque

class OnlinePOSTagger:
    def __init__(self, model, replay_size=1000, lr=0.001):
        self.model = model
        self.replay_buffer = deque(maxlen=replay_size)
        self.lr = lr
        self.examples_seen = 0

    def online_update(self, sentence, gold_tags):
        """Update model with a single new example."""
        self.examples_seen += 1
        # Store in replay buffer
        self.replay_buffer.append((sentence, gold_tags))
        # Compute loss and update on new example
        loss = self.model.compute_loss(sentence, gold_tags)
        loss.backward()
        self.model.update_params(self.lr)
        # Replay: also train on one old example to prevent forgetting
        if len(self.replay_buffer) > 1:
            old_sent, old_tags = self.replay_buffer[
                np.random.randint(len(self.replay_buffer) - 1)
            ]
            old_loss = self.model.compute_loss(old_sent, old_tags)
            old_loss.backward()
            self.model.update_params(self.lr * 0.5)  # Lower LR for replay

    def adapt_to_domain(self, domain_stream):
        """Continuously adapt to incoming domain data."""
        for sentence, tags in domain_stream:
            self.online_update(sentence, tags)
            if self.examples_seen % 100 == 0:
                print(f"Processed {self.examples_seen} examples")
```

**Interview Tips:**
- Highlight that the averaged perceptron is the classic online POS tagging algorithm
- Mention Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
- Discuss the replay buffer strategy as simple and effective against domain drift
- Note that learning rate decay is critical for online stability

---

## Question 18
- [ ] Done
**What approaches work best for POS tagging in conversational or dialogue systems?**
**Answer:**

Conversational text differs from standard written text in fragmentation, disfluencies, informal grammar, and turn-taking structures. Dialogue POS taggers must handle incomplete sentences, fillers, and backchannel responses.

**Core Concepts:**

| Challenge | Description | Example |
|---|---|---|
| Disfluencies | Hesitations, repairs, false starts | "I um want to the uh store" |
| Fragments | Incomplete sentences/utterances | "Yeah sure" (no verb) |
| Backchannels | Short acknowledgment tokens | "uh-huh", "right", "okay" |
| Informal Grammar | Non-standard constructions | "gonna", "wanna", "ain't" |
| Turn Context | Previous speaker's utterance affects tagging | Ellipsis resolved by prior turn |
| Special Tokens | Discourse markers, interjections | "well", "so", "like" as fillers |

**Python Code Example:**
```python
# Pipeline: Dialogue-aware POS tagging with disfluency handling
import re

class DialoguePOSPreprocessor:
    # Common disfluency patterns
    FILLERS = {'um', 'uh', 'er', 'ah', 'hmm', 'mm'}
    BACKCHANNELS = {'uh-huh', 'mm-hm', 'yeah', 'right', 'okay', 'ok'}
    CONTRACTIONS = {
        "gonna": "going to", "wanna": "want to",
        "gotta": "got to", "ain't": "is not",
        "'cause": "because", "y'all": "you all",
    }

    def preprocess_utterance(self, utterance):
        tokens = utterance.split()
        processed = []
        for token in tokens:
            lower = token.lower()
            if lower in self.FILLERS:
                processed.append((token, 'INTJ'))  # Tag fillers directly
            elif lower in self.CONTRACTIONS:
                expanded = self.CONTRACTIONS[lower].split()
                processed.extend([(w, None) for w in expanded])  # Expand, tag later
            else:
                processed.append((token, None))  # Tag with model
        return processed

    def handle_dialogue_context(self, current_turn, previous_turn=None):
        """Add dialogue context features for POS tagging."""
        features = {'is_response': previous_turn is not None}
        if previous_turn:
            features['prev_ends_question'] = previous_turn.strip().endswith('?')
            features['prev_speaker_tag'] = 'Q' if features['prev_ends_question'] else 'S'
        return features
```

**Interview Tips:**
- Mention the Switchboard corpus as the standard annotated dialogue POS dataset
- Discuss that disfluency detection can be a preprocessing step before POS tagging
- Highlight that dialogue acts (question, statement, backchannel) provide useful context
- Note that conversational AI systems often skip formal POS tagging in favor of end-to-end models

---

## Question 19
- [ ] Done
**How do you handle POS tagging optimization for specific downstream NLP tasks?**
**Answer:**

Different downstream tasks benefit from different POS tagging strategies. Information extraction may need fine-grained tags while text classification may need only coarse categories. Task-aware optimization tailors POS tagging to its consumer.

**Core Concepts:**

| Downstream Task | POS Optimization | Rationale |
|---|---|---|
| Parsing | Fine-grained tags (PTB 45-tag) | Syntax depends on tag distinctions |
| Information Extraction | Focus on NOUN, VERB, ADJ accuracy | Key content word categories |
| Sentiment Analysis | ADJ, ADV accuracy critical | Sentiment-bearing POS categories |
| Machine Translation | Full tagset with morphological features | Morphology aids generation |
| Text Classification | Coarse UPOS sufficient | Bag-of-POS features |
| NER | PROPN vs NOUN distinction critical | Named entity POS patterns |

**Python Code Example:**
```python
# Pipeline: Task-optimized POS tagging with weighted loss
import torch
import torch.nn as nn

class TaskOptimizedPOSTagger(nn.Module):
    def __init__(self, base_model, num_tags, task='ner'):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(base_model.hidden_size, num_tags)
        self.task = task
        # Weight important tags higher for specific downstream tasks
        self.task_weights = self._get_task_weights(num_tags, task)

    def _get_task_weights(self, num_tags, task):
        weights = torch.ones(num_tags)
        # Tag index mapping example: 0=NOUN, 1=VERB, 2=ADJ, 3=PROPN...
        if task == 'ner':
            weights[0] = 2.0  # NOUN
            weights[3] = 3.0  # PROPN - critical for NER
        elif task == 'sentiment':
            weights[2] = 2.5  # ADJ
            weights[4] = 2.0  # ADV
        return weights

    def compute_loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss(weight=self.task_weights.to(logits.device))
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)
```

**Interview Tips:**
- Explain that joint training (POS + downstream task) often outperforms pipeline approaches
- Mention that for modern LLMs, explicit POS tags are less needed as features
- Discuss task-specific evaluation: measure downstream task accuracy, not just POS accuracy
- Note that POS tag errors on content words (NOUN, VERB) hurt more than function word errors

---

## Question 20
- [ ] Done
**What techniques help with POS tagging for languages with flexible word order?**
**Answer:**

Languages with free or flexible word order (Russian, Latin, Japanese, Korean, Turkish, Hindi) challenge POS taggers that rely on positional patterns. In English, word position strongly predicts POS, but in free-order languages, morphology and agreement are more important cues.

**Core Concepts:**

| Technique | Description | Applicable Languages |
|---|---|---|
| Morphological Features | Use case, gender, number markers | Russian, German, Latin |
| Case Marking | Grammatical case reveals syntactic role | Turkish, Finnish, Japanese |
| Agreement Features | Subject-verb agreement, gender agreement | Hindi, Arabic, Spanish |
| Graph Neural Networks | Model word dependencies, not just sequence | Any free-order language |
| Self-Attention Models | Attend to relevant context regardless of distance | Universal (transformers) |
| Dependency-Aware Tagging | Use dependency parse trees as features | Syntactically rich languages |

**Python Code Example:**
```python
# Pipeline: Morphology-aware POS tagger for free word order languages
import torch
import torch.nn as nn

class MorphAwareFreeOrderTagger(nn.Module):
    def __init__(self, vocab_size, morph_feat_size, embed_dim, num_tags):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.morph_proj = nn.Linear(morph_feat_size, 50)  # Morphological features
        # Self-attention captures long-range dependencies
        self.self_attn = nn.MultiheadAttention(embed_dim + 50, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(embed_dim + 50, num_tags)

    def forward(self, words, morph_features):
        w_emb = self.word_embed(words)
        m_emb = self.morph_proj(morph_features.float())
        combined = torch.cat([w_emb, m_emb], dim=-1)
        # Self-attention: each token attends to all others (order-invariant)
        attended, _ = self.self_attn(combined, combined, combined)
        return self.classifier(attended)

# Morphological features: case, number, gender, person, tense, aspect
# Encoded as binary/categorical vectors per token
```

**Interview Tips:**
- Emphasize that self-attention models (transformers) naturally handle free word order
- Mention that morphological features are more important than positional features for free-order languages
- Discuss that BiLSTMs still capture local context but miss non-adjacent dependencies
- Note that Universal Dependencies provides consistent annotation across free/fixed order languages

---

## Question 21
- [ ] Done
**How do you implement fairness-aware POS tagging to avoid bias across language varieties?**
**Answer:**

POS taggers trained on standard language often perform worse on dialects, sociolects, and minority language varieties (e.g., African American Vernacular English, Indian English, Singaporean English), creating fairness disparities in downstream NLP applications.

**Core Concepts:**

| Aspect | Description | Example |
|---|---|---|
| Dialect Bias | Lower accuracy on non-standard varieties | AAVE: "she be working" tagged incorrectly |
| Training Data Bias | Over-representation of standard language | WSJ-trained taggers fail on tweets |
| Equalized Accuracy | Same POS accuracy across all varieties | Performance parity metric |
| Multi-Dialect Training | Include diverse variety data in training | Balance standard + dialectal data |
| Bias Auditing | Measure accuracy stratified by variety | Detect disparate performance |
| Debiased Embeddings | Remove sociolinguistic bias from representations | Fairer feature representations |

**Python Code Example:**
```python
# Pipeline: Fairness auditing for POS taggers across language varieties
import numpy as np
from collections import defaultdict

class POS_FairnessAuditor:
    def __init__(self, tagger):
        self.tagger = tagger

    def audit_by_variety(self, test_sets):
        """Evaluate POS accuracy per language variety."""
        results = {}
        for variety_name, data in test_sets.items():
            correct, total = 0, 0
            for sentence, gold_tags in data:
                predicted = self.tagger.predict(sentence)
                for pred, gold in zip(predicted, gold_tags):
                    total += 1
                    if pred == gold:
                        correct += 1
            results[variety_name] = correct / total if total > 0 else 0
        return results

    def compute_fairness_gap(self, results):
        """Compute max accuracy gap between any two varieties."""
        accuracies = list(results.values())
        gap = max(accuracies) - min(accuracies)
        return {
            'max_gap': gap,
            'best': max(results, key=results.get),
            'worst': min(results, key=results.get),
            'fair': gap < 0.05  # <5% gap is "fair"
        }

    def suggest_mitigation(self, results):
        worst_variety = min(results, key=results.get)
        return f"Collect more training data for '{worst_variety}' variety"
```

**Interview Tips:**
- Mention that POS accuracy can vary 5-15% between Standard American English and AAVE
- Discuss the importance of including diverse language varieties in training data
- Highlight that fairness in NLP means equitable performance across demographic groups
- Note that bias in POS tagging propagates to downstream tasks (NER, parsing, sentiment)

---

## Question 22
- [ ] Done
**What strategies work best for POS tagging in real-time text processing applications?**
**Answer:**

Real-time POS tagging requires sub-millisecond per-token latency for applications like autocomplete, live captioning, typing assistance, and streaming text analysis. The key tradeoff is speed vs. accuracy.

**Core Concepts:**

| Strategy | Latency | Accuracy | Use Case |
|---|---|---|---|
| Lookup Table | <0.01ms/token | 90% | Autocomplete, IME |
| Averaged Perceptron | ~0.05ms/token | 96% | General real-time |
| BiLSTM (compiled) | ~0.1ms/token | 97% | Moderate real-time |
| ONNX-Optimized BERT | ~0.5ms/token | 98% | High-accuracy real-time |
| Batched GPU Inference | ~0.01ms/token (batched) | 98% | High-throughput servers |
| Cached Results | O(1) for repeated text | 100% (on cache hits) | Repetitive text streams |

**Python Code Example:**
```python
# Pipeline: Real-time POS tagging with caching and fallback
import time
from functools import lru_cache
from collections import OrderedDict

class RealTimePOSTagger:
    def __init__(self, fast_model, accurate_model, latency_budget_ms=1.0):
        self.fast = fast_model       # Perceptron or lookup
        self.accurate = accurate_model  # BERT
        self.budget = latency_budget_ms
        self.cache = OrderedDict()  # LRU cache
        self.cache_limit = 10000

    def tag(self, sentence):
        cache_key = tuple(sentence)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        start = time.time()
        result = self.fast.predict(sentence)
        elapsed = (time.time() - start) * 1000

        # If fast model is within budget and sentence is short, try accurate model
        if elapsed < self.budget * 0.3 and len(sentence) < 50:
            result = self.accurate.predict(sentence)

        # Cache the result
        if len(self.cache) >= self.cache_limit:
            self.cache.popitem(last=False)
        self.cache[cache_key] = result
        return result

    def tag_stream(self, token_stream):
        buffer = []
        for token in token_stream:
            buffer.append(token)
            if len(buffer) >= 10:  # Tag in chunks
                yield self.tag(buffer)
                buffer = []
```

**Interview Tips:**
- Recommend averaged perceptron or lookup tables for sub-millisecond requirements
- Mention ONNX Runtime and TorchScript for optimizing neural model inference
- Discuss batching strategies for throughput vs. latency optimization
- Note that caching is highly effective since POS tagging is deterministic for the same input

---

## Question 23
- [ ] Done
**How do you handle POS tagging quality assessment with inter-annotator disagreement?**
**Answer:**

Inter-annotator agreement (IAA) measures how consistently multiple annotators assign POS tags to the same text. POS tagging typically has 95-97% IAA, with disagreements concentrated on ambiguous categories (e.g., NOUN vs. VERB for gerunds).

**Core Concepts:**

| Metric | Description | POS Tagging Typical Value |
|---|---|---|
| Cohen's Kappa | Agreement corrected for chance | 0.90-0.97 |
| Fleiss' Kappa | Multi-annotator agreement | 0.88-0.95 |
| Percent Agreement | Simple agreement rate | 95-97% |
| Confusion Pairs | Most commonly disagreed tag pairs | NOUN/VERB, ADJ/NOUN, ADV/PART |
| Adjudication | Expert resolves disagreements | Gold standard creation |
| Soft Labels | Use disagreement distribution as training signal | Uncertainty-aware training |

**Python Code Example:**
```python
# Pipeline: Inter-annotator agreement analysis for POS tagging
import numpy as np
from collections import Counter

def cohens_kappa(annotator1, annotator2):
    """Calculate Cohen's Kappa between two annotators."""
    assert len(annotator1) == len(annotator2)
    n = len(annotator1)
    # Observed agreement
    agree = sum(1 for a, b in zip(annotator1, annotator2) if a == b)
    po = agree / n
    # Expected agreement by chance
    tags1 = Counter(annotator1)
    tags2 = Counter(annotator2)
    all_tags = set(tags1) | set(tags2)
    pe = sum((tags1[t] / n) * (tags2[t] / n) for t in all_tags)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def find_disagreement_patterns(annotator1, annotator2, tokens):
    """Identify systematic disagreement patterns."""
    disagreements = Counter()
    for tok, t1, t2 in zip(tokens, annotator1, annotator2):
        if t1 != t2:
            pair = tuple(sorted([t1, t2]))
            disagreements[pair] += 1
    return disagreements.most_common(10)

def create_soft_labels(annotations_list):
    """Convert multiple annotations into soft probability labels."""
    n_annotators = len(annotations_list)
    soft = []
    for token_idx in range(len(annotations_list[0])):
        tag_counts = Counter(ann[token_idx] for ann in annotations_list)
        soft.append({tag: count / n_annotators for tag, count in tag_counts.items()})
    return soft
```

**Interview Tips:**
- State that Cohen's Kappa >0.9 is considered excellent for POS tagging
- Mention common disagreement pairs: participles (VBN vs. JJ), gerunds (VBG vs. NN)
- Discuss adjudication procedures: expert tiebreaker, majority vote, or soft labels
- Note that training on soft labels from multiple annotators can improve model robustness

---

## Question 24
- [ ] Done
**What approaches help with POS tagging for texts requiring high linguistic accuracy?**
**Answer:**

High-accuracy POS tagging (99%+) is required for linguistic research, treebank construction, and applications where downstream systems are highly sensitive to tag errors. Achieving near-human accuracy requires ensemble methods and careful error analysis.

**Core Concepts:**

| Approach | Accuracy Boost | Description |
|---|---|---|
| Ensemble Methods | +0.3-0.5% | Combine multiple diverse taggers |
| BERT-Large + CRF | 97.8-98.2% | State-of-the-art single model |
| Gold Morphological Features | +0.5-1% | External analyzer features |
| Constrained Decoding | +0.2% | Enforce linguistic constraints |
| Human-in-the-Loop | 99%+ | Flag uncertain tokens for expert review |
| Error Pattern Correction | +0.3% | Post-hoc rules for known error patterns |

**Python Code Example:**
```python
# Pipeline: High-accuracy POS tagging with ensemble and post-correction
class HighAccuracyPOSEnsemble:
    def __init__(self, taggers, correction_rules=None):
        self.taggers = taggers
        self.rules = correction_rules or []

    def majority_vote(self, predictions):
        """Weighted majority vote across taggers."""
        from collections import Counter
        results = []
        for token_preds in zip(*predictions):
            vote = Counter(token_preds).most_common(1)[0][0]
            results.append(vote)
        return results

    def apply_corrections(self, tokens, tags):
        """Apply post-hoc linguistic rules."""
        corrected = list(tags)
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            # Rule: "the" is always DET
            if token.lower() == 'the' and tag != 'DET':
                corrected[i] = 'DET'
            # Rule: Token after "to" + verb form is likely VERB
            if i > 0 and tokens[i-1].lower() == 'to' and tag == 'NOUN':
                corrected[i] = 'VERB'
        return corrected

    def predict(self, tokens):
        all_predictions = [tagger.predict(tokens) for tagger in self.taggers]
        ensemble_tags = self.majority_vote(all_predictions)
        corrected_tags = self.apply_corrections(tokens, ensemble_tags)
        return corrected_tags
```

**Interview Tips:**
- Note that state-of-the-art POS accuracy plateaus around 97.5-98% due to annotation inconsistencies
- Mention that remaining errors are often genuinely ambiguous cases where annotators also disagree
- Discuss that post-hoc correction rules for deterministic patterns ("the" = DET) boost accuracy cheaply
- Highlight that human-in-the-loop for uncertain tokens is the only way to approach 99%+

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving POS tagging for sensitive text data?**
**Answer:**

Privacy-preserving POS tagging enables linguistic analysis of sensitive documents (medical records, legal communications, personal messages) without exposing the actual text content to the tagging system.

**Core Concepts:**

| Technique | Privacy Level | Performance Impact |
|---|---|---|
| Federated Learning | Data stays on device | Minimal if enough participants |
| Differential Privacy | Mathematical privacy guarantee | 1-3% accuracy drop |
| On-Device Inference | No data leaves device | Requires lightweight model |
| Encrypted Computation | Homomorphic encryption | 100-1000x slower |
| Data Anonymization | Replace PII before tagging | No model change needed |
| Secure Enclaves | Hardware-protected processing | Near-zero performance drop |

**Python Code Example:**
```python
# Pipeline: Privacy-preserving POS tagging with text anonymization
import re
import hashlib

class PrivatePOSTagger:
    def __init__(self, base_tagger):
        self.tagger = base_tagger
        self.pii_patterns = {
            'email': r'[\w.-]+@[\w.-]+\.\w+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'name_prefix': r'\b(Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b',
        }

    def anonymize_text(self, text):
        """Replace PII with typed placeholders before tagging."""
        anonymized = text
        mapping = {}  # Store reversible mapping if needed
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, anonymized)
            for match in matches:
                placeholder = f"[{pii_type.upper()}]"
                mapping[placeholder] = match.group()
                anonymized = anonymized.replace(match.group(), placeholder, 1)
        return anonymized, mapping

    def tag_privately(self, text):
        anonymized, mapping = self.anonymize_text(text)
        tags = self.tagger.predict(anonymized)
        # PII placeholders get appropriate default tags
        for i, (token, tag) in enumerate(tags):
            if token.startswith('[') and token.endswith(']'):
                tags[i] = (token, 'PROPN')  # PII is typically proper nouns
        return tags
```

**Interview Tips:**
- Recommend on-device inference as the simplest privacy-preserving approach
- Mention that federated learning enables collaborative POS tagger training without sharing data
- Discuss that differential privacy adds calibrated noise to gradients during training
- Note that PII anonymization before tagging is pragmatic and widely used in healthcare NLP

---

## Question 26
- [ ] Done
**What techniques work best for POS tagging with fine-grained tagset distinctions?**
**Answer:**

Fine-grained POS tagsets (PTB 45 tags, CLAWS7 137 tags, morphological tagsets 500+ tags) capture subtle distinctions like verb tense, noun case, and adjective degree. Accuracy naturally decreases as tagset size increases.

**Core Concepts:**

| Tagset | Size | Distinctions | Accuracy |
|---|---|---|---|
| Universal POS (UPOS) | 17 tags | Coarse categories only | 97-98% |
| Penn Treebank (PTB) | 45 tags | Verb tense, noun number | 97-97.5% |
| CLAWS7 | 137 tags | Detailed morphosyntactic | 95-96% |
| Morphological Tags | 500+ | Full morphological features | 90-95% |
| Two-Level Prediction | Coarse→Fine | Hierarchical refinement | Better than flat |

**Python Code Example:**
```python
# Pipeline: Hierarchical fine-grained POS tagging
import torch
import torch.nn as nn

class HierarchicalPOSTagger(nn.Module):
    """Two-level: coarse POS first, then fine-grained within category."""
    def __init__(self, input_dim, hidden_dim, coarse_tags, fine_tags_per_coarse):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.coarse_classifier = nn.Linear(hidden_dim * 2, coarse_tags)
        # Separate fine-grained classifier per coarse tag
        self.fine_classifiers = nn.ModuleDict({
            str(i): nn.Linear(hidden_dim * 2, n_fine)
            for i, n_fine in enumerate(fine_tags_per_coarse)
        })

    def forward(self, x):
        encoded, _ = self.encoder(x)
        coarse_logits = self.coarse_classifier(encoded)
        coarse_preds = coarse_logits.argmax(dim=-1)
        # Fine-grained prediction conditioned on coarse
        fine_logits = []
        for t in range(encoded.size(1)):
            coarse_idx = str(coarse_preds[0, t].item())
            if coarse_idx in self.fine_classifiers:
                fine = self.fine_classifiers[coarse_idx](encoded[:, t])
            else:
                fine = torch.zeros(1)  # Fallback
            fine_logits.append(fine)
        return coarse_logits, fine_logits

# Example tagset hierarchy:
# VERB -> VB, VBD, VBG, VBN, VBP, VBZ
# NOUN -> NN, NNS, NNP, NNPS
# ADJ -> JJ, JJR, JJS
```

**Interview Tips:**
- Explain that hierarchical tagging (coarse-to-fine) improves fine-grained accuracy by 1-2%
- Mention that morphological feature prediction is often modeled as multi-label classification
- Discuss that Universal Dependencies features (case, number, gender) complement UPOS tags
- Note that fine-grained distinctions matter most for downstream tasks like machine translation

---

## Question 27
- [ ] Done
**How do you handle POS tagging adaptation to emerging language varieties and dialects?**
**Answer:**

New language varieties constantly emerge through social media, youth slang, professional jargon, and dialectal evolution. POS taggers must adapt to these varieties without losing performance on standard language.

**Core Concepts:**

| Challenge | Description | Example |
|---|---|---|
| Lexical Innovation | New words not in training vocabulary | "yeet", "simp", "bussin" |
| Grammatical Innovation | New syntactic patterns | "I can't even" (intransitive use) |
| Code-Meshing | Blending dialects with standard | Academic AAVE writing |
| Register Shift | Formal/informal mixing | Corporate tweets |
| Neologisms | Newly coined words | Tech: "tokenize", "blockchain" |
| Emoji Integration | Emoji as linguistic elements | 😊 functions as ADJ/INTJ |

**Python Code Example:**
```python
# Pipeline: Adaptive POS tagger for emerging language varieties
from collections import defaultdict
import re

class AdaptivePOSTagger:
    def __init__(self, base_tagger):
        self.base = base_tagger
        self.neologism_cache = {}  # Learned new word → tag mappings
        self.variety_rules = defaultdict(list)

    def detect_neologism(self, token):
        """Check if token is a potential new word."""
        return (token.lower() not in self.base.vocabulary
                and not token.startswith('#')
                and not token.startswith('@'))

    def infer_neologism_pos(self, token, context):
        """Infer POS of unknown word from morphology and context."""
        lower = token.lower()
        # Morphological cues
        if lower.endswith('ing'):
            return 'VERB'
        if lower.endswith('tion') or lower.endswith('ness'):
            return 'NOUN'
        if lower.endswith('ly'):
            return 'ADV'
        if lower.endswith('ify') or lower.endswith('ize'):
            return 'VERB'
        # Default to NOUN for novel words
        return 'NOUN'

    def tag_adaptive(self, tokens):
        base_tags = self.base.predict(tokens)
        adapted = []
        for i, (token, tag) in enumerate(zip(tokens, base_tags)):
            if self.detect_neologism(token):
                if token.lower() in self.neologism_cache:
                    tag = self.neologism_cache[token.lower()]
                else:
                    tag = self.infer_neologism_pos(token, tokens)
                    self.neologism_cache[token.lower()] = tag
            adapted.append((token, tag))
        return adapted
```

**Interview Tips:**
- Mention that morphological cues (suffixes) are the most reliable strategy for neologisms
- Discuss how continual learning enables taggers to evolve with language
- Highlight that Urban Dictionary and Wiktionary can serve as resources for new vocabulary
- Note that emoji POS tagging is an active research area (typically tagged INTJ or SYM)

---

## Question 28
- [ ] Done
**What strategies help with POS tagging for texts with complex syntactic constructions?**
**Answer:**

Complex syntactic constructions — garden path sentences, center-embedded clauses, long-distance dependencies, and coordination — challenge POS taggers because local context is insufficient for disambiguation.

**Core Concepts:**

| Construction | Challenge | Example |
|---|---|---|
| Garden Path | Initial parse misleads tagger | "The horse raced past the barn fell" |
| Center-Embedding | Nested clauses blur context | "The rat the cat the dog chased ate died" |
| Coordination | Shared head across conjuncts | "old men and women" (scope ambiguity) |
| Long-Distance Dep. | Key context far from target | "What did John say Mary thought..." |
| Relative Clauses | Potential PP-attachment ambiguity | "The man with the telescope saw..." |
| Ellipsis | Missing elements need inference | "John ran and Mary too" |

**Python Code Example:**
```python
# Pipeline: Syntax-aware POS tagging with extended context
import torch
from transformers import AutoTokenizer, AutoModel

class SyntaxAwarePOSTagger:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_long_range_features(self, sentence, target_idx):
        """Extract features that capture long-distance dependencies."""
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Attention from target token reveals syntactic dependencies
        attentions = outputs.attentions[-1].mean(dim=1)[0]  # Last layer, avg heads
        target_attention = attentions[target_idx + 1]  # +1 for [CLS]

        # Find most-attended tokens (likely syntactic heads/dependents)
        top_attended = torch.topk(target_attention, k=5)
        return {
            'target_hidden': outputs.last_hidden_state[0, target_idx + 1],
            'top_attended_indices': top_attended.indices.tolist(),
            'top_attended_weights': top_attended.values.tolist(),
        }

    def analyze_complexity(self, sentence):
        """Detect syntactically complex constructions."""
        tokens = sentence.split()
        complexity_signals = {
            'nested_clauses': sentence.count(','),
            'relative_pronouns': sum(1 for t in tokens if t.lower() in {'which', 'that', 'who', 'whom'}),
            'coordination': sum(1 for t in tokens if t.lower() in {'and', 'or', 'but'}),
            'sentence_length': len(tokens),
        }
        complexity_signals['is_complex'] = (
            complexity_signals['nested_clauses'] > 2 or
            complexity_signals['sentence_length'] > 30
        )
        return complexity_signals
```

**Interview Tips:**
- Explain that transformers handle complex syntax better than LSTMs due to self-attention
- Mention garden path sentences as classic examples where local context misleads taggers
- Discuss that joint POS tagging + parsing handles syntactic complexity better than POS alone
- Note that very long sentences should be chunked or processed with models supporting long context

---

## Question 29
- [ ] Done
**How do you implement robust error handling for POS tagging in production systems?**
**Answer:**

Production POS tagging systems must handle edge cases gracefully: empty inputs, extremely long texts, encoding issues, unexpected characters, model loading failures, and degraded performance without crashing.

**Core Concepts:**

| Error Category | Examples | Handling Strategy |
|---|---|---|
| Input Validation | Empty text, null values, wrong encoding | Validate and sanitize before tagging |
| Length Limits | Text exceeds model max length | Chunk and recombine |
| Character Issues | Unicode errors, control characters | Normalize/strip invalid chars |
| Model Failures | OOM, corrupted weights, timeout | Fallback to simpler model |
| Unexpected Tokens | Emoji, URLs, code snippets | Assign default tags |
| Performance Degradation | New domain, accuracy drop | Monitor and alert |

**Python Code Example:**
```python
# Pipeline: Production-ready POS tagger with comprehensive error handling
import logging
import unicodedata

logger = logging.getLogger(__name__)

class ProductionPOSTagger:
    def __init__(self, primary_model, fallback_model, max_length=512):
        self.primary = primary_model
        self.fallback = fallback_model
        self.max_length = max_length

    def validate_input(self, text):
        if not text or not isinstance(text, str):
            return []
        # Remove control characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t')
        return text.strip()

    def chunk_text(self, tokens):
        """Split long texts into processable chunks."""
        chunks = []
        for i in range(0, len(tokens), self.max_length):
            chunks.append(tokens[i:i + self.max_length])
        return chunks

    def tag(self, text):
        text = self.validate_input(text)
        if not text:
            return []

        tokens = text.split()
        chunks = self.chunk_text(tokens)
        all_tags = []

        for chunk in chunks:
            try:
                tags = self.primary.predict(chunk)
                all_tags.extend(tags)
            except MemoryError:
                logger.warning("OOM with primary model, using fallback")
                tags = self.fallback.predict(chunk)
                all_tags.extend(tags)
            except Exception as e:
                logger.error(f"Tagging failed: {e}")
                all_tags.extend([(t, 'X') for t in chunk])  # X = unknown

        return all_tags
```

**Interview Tips:**
- Emphasize input validation as the first line of defense
- Mention fallback models (simpler/faster) for when primary model fails
- Discuss chunking strategies for texts exceeding model length limits
- Note that monitoring tag distribution drift can detect accuracy degradation early

---

## Question 30
- [ ] Done
**What approaches work best for combining POS tagging with other linguistic annotation tasks?**
**Answer:**

Multi-task learning (MTL) jointly trains POS tagging with related tasks (NER, dependency parsing, morphological analysis, chunking), sharing representations to improve all tasks simultaneously through positive transfer.

**Core Concepts:**

| Combined Tasks | Benefit | Shared Signal |
|---|---|---|
| POS + NER | POS features help NER (PROPN → entity) | Lexical and syntactic patterns |
| POS + Parsing | POS constrains parse tree structure | Syntactic category knowledge |
| POS + Morphology | Joint prediction of related annotations | Same word-form analysis |
| POS + Chunking | POS determines chunk boundaries | Phrase structure patterns |
| POS + Lemmatization | POS disambiguates lemma | "saw" → NOUN(saw) vs VERB(see) |

**Python Code Example:**
```python
# Pipeline: Multi-task model for joint POS tagging + NER + chunking
import torch
import torch.nn as nn

class MultiTaskNLPTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, pos_tags, ner_tags, chunk_tags):
        super().__init__()
        # Shared encoder
        self.shared_encoder = nn.LSTM(
            input_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=2
        )
        # Task-specific heads
        self.pos_head = nn.Linear(hidden_dim * 2, pos_tags)
        self.ner_head = nn.Linear(hidden_dim * 2, ner_tags)
        self.chunk_head = nn.Linear(hidden_dim * 2, chunk_tags)

    def forward(self, x, task='all'):
        shared_repr, _ = self.shared_encoder(x)
        outputs = {}
        if task in ('all', 'pos'):
            outputs['pos'] = self.pos_head(shared_repr)
        if task in ('all', 'ner'):
            outputs['ner'] = self.ner_head(shared_repr)
        if task in ('all', 'chunk'):
            outputs['chunk'] = self.chunk_head(shared_repr)
        return outputs

    def combined_loss(self, outputs, labels, weights={'pos': 1.0, 'ner': 1.0, 'chunk': 0.5}):
        total_loss = 0
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        for task, logits in outputs.items():
            task_loss = criterion(logits.view(-1, logits.size(-1)), labels[task].view(-1))
            total_loss += weights.get(task, 1.0) * task_loss
        return total_loss
```

**Interview Tips:**
- Highlight that joint POS+parsing is the most well-established beneficial combination
- Mention that POS tagging serves as an auxiliary task for many NLP models
- Discuss task weighting as a critical hyperparameter in multi-task learning
- Note that hard parameter sharing (shared encoder) is simpler and often sufficient

---

## Question 31
- [ ] Done
**How do you handle POS tagging for texts with varying levels of formality?**
**Answer:**

Formality ranges from casual (texts, chats) to formal (academic, legal), each with distinct vocabulary, syntax, and POS distributions. A tagger trained only on formal text (WSJ) degrades on informal text and vice versa.

**Core Concepts:**

| Formality Level | Characteristics | POS Impact |
|---|---|---|
| Very Informal | Abbreviations, slang, fragments | More INTJ, fewer DET |
| Informal | Contractions, colloquial vocab | Standard but relaxed syntax |
| Neutral | Mixed register, general text | Standard distribution |
| Formal | Complex sentences, technical vocab | More NOUN, longer sentences |
| Very Formal | Legal/academic, passive voice | More AUX, complex NP structures |

**Python Code Example:**
```python
# Pipeline: Formality-aware POS tagging with register detection
from collections import Counter

class FormalityAwareTagger:
    def __init__(self, formal_tagger, informal_tagger, general_tagger):
        self.taggers = {
            'formal': formal_tagger,
            'informal': informal_tagger,
            'neutral': general_tagger,
        }

    def detect_formality(self, text):
        tokens = text.lower().split()
        informal_signals = sum(1 for t in tokens if t in {
            'gonna', 'wanna', 'lol', 'tbh', 'ngl', 'idk', 'bruh'
        })
        formal_signals = sum(1 for t in tokens if t in {
            'furthermore', 'notwithstanding', 'hereby', 'whereas',
            'pursuant', 'aforementioned', 'henceforth'
        })
        has_contractions = any("'" in t for t in tokens)
        avg_word_len = sum(len(t) for t in tokens) / max(len(tokens), 1)

        if informal_signals > 0 or (has_contractions and avg_word_len < 4.5):
            return 'informal'
        elif formal_signals > 0 or avg_word_len > 6:
            return 'formal'
        return 'neutral'

    def tag(self, text):
        formality = self.detect_formality(text)
        return self.taggers[formality].predict(text)
```

**Interview Tips:**
- Mention that register/formality detection is a simple preprocessing step that improves tagging
- Discuss that domain-adapted BERT models inherently learn register differences
- Highlight that informal text has different POS distributions (more interjections, fewer articles)
- Note that multi-domain training on diverse registers is the most robust approach

---

## Question 32
- [ ] Done
**What techniques help with POS tagging consistency in federated learning scenarios?**
**Answer:**

Federated POS tagging trains models across multiple organizations/devices without sharing raw text data. Consistency challenges arise because each participant may have different text distributions, leading to model divergence.

**Core Concepts:**

| Technique | Description | Benefit |
|---|---|---|
| FedAvg | Average model weights from all clients | Simple baseline |
| FedProx | Add proximal term to prevent drift | Better convergence |
| Tagset Standardization | Enforce same tagset across clients | Consistent labels |
| Gradient Compression | Reduce communication overhead | Scalability |
| Secure Aggregation | Aggregate without revealing individual updates | Privacy guarantee |
| Client Weighting | Weight updates by data quality/quantity | Handles heterogeneity |

**Python Code Example:**
```python
# Pipeline: Federated POS tagging with FedAvg and consistency checks
import copy
import torch

class FederatedPOSTrainer:
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.num_clients = num_clients

    def client_train(self, model, local_data, epochs=3, lr=0.01):
        """Train on local client data."""
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for tokens, tags in local_data:
                optimizer.zero_grad()
                logits = model(tokens)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tags.view(-1)
                )
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def federated_round(self, client_data_list):
        """One round of federated averaging."""
        client_states = []
        for client_data in client_data_list:
            local_model = copy.deepcopy(self.global_model)
            state = self.client_train(local_model, client_data)
            client_states.append(state)
        # Average all client model weights
        avg_state = {}
        for key in client_states[0]:
            avg_state[key] = torch.stack(
                [s[key].float() for s in client_states]
            ).mean(dim=0)
        self.global_model.load_state_dict(avg_state)
        return self.global_model
```

**Interview Tips:**
- Explain that federated learning preserves data privacy while training collaborative POS models
- Mention non-IID data distribution as the main challenge (each client has different text types)
- Discuss FedProx as improvement over FedAvg for heterogeneous client data
- Note that communication efficiency (gradient compression) is critical for practical deployment

---

## Question 33
- [ ] Done
**How do you implement efficient batch processing for large-scale POS tagging?**
**Answer:**

Large-scale POS tagging of millions or billions of documents requires efficient batching, parallelism, and pipeline optimization to maximize throughput while controlling costs.

**Core Concepts:**

| Strategy | Description | Speedup |
|---|---|---|
| Dynamic Batching | Group similar-length sentences | 2-3x vs. fixed padding |
| Token Bucketing | Sort by length, batch within buckets | Minimizes padding waste |
| Multi-GPU Inference | Distribute across GPUs with DataParallel | Linear scaling |
| ONNX Runtime | Optimize model graph for inference | 2-4x over vanilla PyTorch |
| Async I/O Pipeline | Overlap data loading with inference | Hide I/O latency |
| Quantization | INT8 model weights | 2-4x faster, slight accuracy drop |

**Python Code Example:**
```python
# Pipeline: Efficient batch POS tagging with dynamic batching
import torch
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor

class BatchPOSTagger:
    def __init__(self, model, tokenizer, batch_size=64, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def dynamic_batch(self, sentences):
        """Group sentences by length for efficient batching."""
        indexed = [(i, s) for i, s in enumerate(sentences)]
        indexed.sort(key=lambda x: len(x[1].split()))
        batches = []
        for i in range(0, len(indexed), self.batch_size):
            batch = indexed[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def process_batch(self, batch):
        indices, texts = zip(*batch)
        encodings = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encodings).logits
        predictions = torch.argmax(logits, dim=-1)
        return list(zip(indices, predictions))

    def tag_corpus(self, sentences):
        """Tag entire corpus with dynamic batching."""
        batches = self.dynamic_batch(sentences)
        all_results = [None] * len(sentences)
        for batch in batches:
            results = self.process_batch(batch)
            for idx, preds in results:
                all_results[idx] = preds
        return all_results
```

**Interview Tips:**
- Highlight dynamic batching by sentence length as the single biggest throughput optimization
- Mention that ONNX Runtime + INT8 quantization gives best inference performance
- Discuss that multi-GPU data parallelism scales linearly for inference workloads
- Note that async I/O pipelines hide data loading latency behind model computation

---

## Question 34
- [ ] Done
**What strategies work best for POS tagging with specific linguistic theory requirements?**
**Answer:**

Different linguistic theories (generative grammar, dependency grammar, construction grammar, cognitive linguistics) define POS categories differently, requiring taggers aligned with specific theoretical frameworks.

**Core Concepts:**

| Linguistic Theory | POS View | Tagset Design |
|---|---|---|
| Generative Grammar | Functional vs. lexical categories | DET, COMP, INFL as distinct heads |
| Dependency Grammar | Based on syntactic function | Head-dependent relations define category |
| Construction Grammar | POS partially determined by construction | More context-dependent tagging |
| Distributional | POS defined by word distribution | Cluster-based tagsets |
| Traditional Grammar | 8-9 parts of speech | Noun, verb, adj, adv, prep, conj, pron, interj |
| Universal Grammar | Cross-linguistic universal categories | UPOS 17-tag set |

**Python Code Example:**
```python
# Pipeline: Theory-customizable POS tagger with pluggable tagsets
class TheorySpecificTagger:
    def __init__(self, base_tagger, theory='traditional'):
        self.base = base_tagger
        self.theory = theory
        self.mapping = self._get_theory_mapping(theory)

    def _get_theory_mapping(self, theory):
        if theory == 'traditional':
            return {  # UPOS → Traditional 8 parts of speech
                'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb',
                'AUX': 'verb', 'ADJ': 'adjective', 'ADV': 'adverb',
                'ADP': 'preposition', 'CCONJ': 'conjunction',
                'SCONJ': 'conjunction', 'PRON': 'pronoun',
                'DET': 'article', 'INTJ': 'interjection',
                'NUM': 'adjective', 'PART': 'adverb',
            }
        elif theory == 'generative':
            return {  # UPOS → Generative (functional/lexical split)
                'NOUN': 'N', 'VERB': 'V', 'ADJ': 'A', 'ADV': 'Adv',
                'DET': 'D', 'ADP': 'P', 'CCONJ': 'Conj',
                'SCONJ': 'C',  # Complementizer
                'AUX': 'I',    # INFL head
                'PRON': 'D',   # Pronouns as Det in some theories
            }
        return {}  # Identity mapping

    def tag(self, text):
        base_tags = self.base.predict(text)
        return [(token, self.mapping.get(tag, tag)) for token, tag in base_tags]
```

**Interview Tips:**
- Mention that Universal Dependencies (UD) aims to be theory-neutral but still makes theoretical choices
- Discuss the distinction between functional (DET, AUX, COMP) and lexical (NOUN, VERB, ADJ) categories
- Note that distributional approaches (Brown clusters) can induce POS-like categories automatically
- Highlight that tagset choice should align with the downstream application's theoretical needs

---

## Question 35
- [ ] Done
**How do you handle POS tagging for texts requiring syntactic parsing downstream?**
**Answer:**

When POS tagging feeds into syntactic parsing, tag accuracy is critical because parse quality heavily depends on correct POS assignments. Error propagation from tagger to parser is a well-known problem.

**Core Concepts:**

| Strategy | Description | Impact on Parsing |
|---|---|---|
| Joint POS+Parsing | Predict POS and parse tree simultaneously | Eliminates error propagation |
| N-Best POS Tags | Pass top-N tag candidates to parser | Parser resolves remaining ambiguity |
| Fine-Grained Tags | Use PTB-level tags, not coarse UPOS | More syntactic information for parser |
| Lattice Parsing | Parser considers multiple tag hypotheses | Handles POS ambiguity gracefully |
| Confidence Weighting | Weight parse candidates by tagger confidence | Prioritize reliable tags |
| Reranking | Use parser output to rerank POS sequences | Mutual disambiguation |

**Python Code Example:**
```python
# Pipeline: POS tagging optimized for downstream parsing
import spacy
import torch

class ParsingAwarePOSTagger:
    def __init__(self, tagger_model, n_best=3):
        self.model = tagger_model
        self.n_best = n_best

    def get_n_best_tags(self, sentence):
        """Return top-N POS tag candidates per token."""
        with torch.no_grad():
            logits = self.model(sentence)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_tags = torch.topk(probs, self.n_best, dim=-1)
        return top_tags, top_probs

    def create_tag_lattice(self, sentence):
        """Create a tag lattice for lattice parsing."""
        top_tags, top_probs = self.get_n_best_tags(sentence)
        lattice = []
        for i in range(len(sentence)):
            token_options = []
            for j in range(self.n_best):
                token_options.append({
                    'tag': top_tags[0, i, j].item(),
                    'prob': top_probs[0, i, j].item()
                })
            lattice.append(token_options)
        return lattice

# Joint POS + parsing with spaCy
def joint_pos_parse(text):
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)  # POS and parsing done jointly in transformer pipeline
    return [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]
```

**Interview Tips:**
- Strongly recommend joint POS+parsing models (modern standard) over pipeline approaches
- Mention that spaCy's transformer pipeline does joint POS+parsing by default
- Discuss lattice parsing as an elegant solution to POS ambiguity propagation
- Note that POS errors on function words (DET, ADP) cause fewer parsing errors than content word errors

---

## Question 36
- [ ] Done
**What approaches help with POS tagging adaptation to user-specific annotation needs?**
**Answer:**

Different users and projects may need custom POS annotation schemes, different granularity levels, or domain-specific tag definitions. Customization enables taggers to match specific project requirements.

**Core Concepts:**

| Need | Solution | Example |
|---|---|---|
| Custom Tagset | Train on user-defined categories | Merge NOUN+PROPN for some projects |
| Annotation Preferences | Fine-tune on user-corrected examples | User prefers "VERB" for gerunds |
| Granularity Control | Configurable coarse/fine output | Slider from 5 to 50 tags |
| Active Corrections | Learn from user feedback | Retrain on corrections in real-time |
| Template Annotations | Pre-configure for common use cases | NER-focused, parsing-focused, etc. |
| Export Formats | Output in user-preferred format | CoNLL, JSON, XML |

**Python Code Example:**
```python
# Pipeline: Customizable POS tagger adapting to user preferences
class CustomizablePOSTagger:
    def __init__(self, base_model):
        self.base = base_model
        self.custom_rules = {}     # User override rules
        self.corrections_log = []   # User correction history
        self.tag_mapping = {}       # Custom tagset mapping

    def set_custom_tagset(self, mapping):
        """User defines their own tagset via mapping from base tags."""
        self.tag_mapping = mapping  # e.g., {'NOUN': 'N', 'PROPN': 'N', 'VERB': 'V'}

    def add_correction(self, word, context, correct_tag):
        """User corrects a specific tagging decision."""
        self.corrections_log.append({
            'word': word, 'context': context, 'tag': correct_tag
        })
        # Simple rule: always tag this word as corrected in similar context
        self.custom_rules[(word.lower(), context)] = correct_tag

    def tag(self, tokens):
        base_tags = self.base.predict(tokens)
        result = []
        for i, (token, tag) in enumerate(zip(tokens, base_tags)):
            # Check user corrections first
            context = tuple(tokens[max(0,i-1):i+2])
            key = (token.lower(), context)
            if key in self.custom_rules:
                tag = self.custom_rules[key]
            # Apply custom tagset mapping
            if tag in self.tag_mapping:
                tag = self.tag_mapping[tag]
            result.append((token, tag))
        return result

    def retrain_on_corrections(self):
        """Fine-tune base model on accumulated corrections."""
        if len(self.corrections_log) >= 50:
            self.base.fine_tune(self.corrections_log)
            self.corrections_log = []
```

**Interview Tips:**
- Emphasize that user-in-the-loop correction is the most effective adaptation method
- Mention that tagset mapping provides instant customization without retraining
- Discuss active learning to prioritize which examples to show users for correction
- Note that Prodigy (from SpaCy) provides an excellent annotation UI for custom POS tagsets

---

## Question 37
- [ ] Done
**How do you implement monitoring and quality control for POS tagging systems?**
**Answer:**

Production POS tagging systems require continuous monitoring to detect accuracy degradation, distribution drift, and system failures. Monitoring ensures consistent quality as input data evolves over time.

**Core Concepts:**

| Monitor | What It Tracks | Alert Trigger |
|---|---|---|
| Tag Distribution | POS frequency distribution over time | >5% shift from baseline |
| Confidence Scores | Average model confidence | Drop below threshold |
| OOV Rate | Percentage of unknown words | Spike above normal |
| Latency | Processing time per sentence | Exceeds SLA |
| Error Sampling | Random audit of predictions | Expert accuracy check |
| Drift Detection | Statistical test on feature distributions | p-value < 0.05 |

**Python Code Example:**
```python
# Pipeline: POS tagging monitoring and quality control system
import time
from collections import Counter
import numpy as np

class POSTaggingMonitor:
    def __init__(self, baseline_distribution, alert_threshold=0.05):
        self.baseline = baseline_distribution  # Expected POS frequencies
        self.threshold = alert_threshold
        self.recent_tags = []
        self.alerts = []

    def log_prediction(self, tags, confidence, latency_ms):
        self.recent_tags.extend(tags)
        if confidence < 0.7:
            self.alerts.append(f"Low confidence: {confidence:.2f}")
        if latency_ms > 100:
            self.alerts.append(f"High latency: {latency_ms:.0f}ms")

    def check_distribution_drift(self, window_size=10000):
        if len(self.recent_tags) < window_size:
            return None
        recent = Counter(self.recent_tags[-window_size:])
        total = sum(recent.values())
        current_dist = {tag: count/total for tag, count in recent.items()}
        # Jensen-Shannon divergence from baseline
        drift = sum(
            abs(current_dist.get(tag, 0) - self.baseline.get(tag, 0))
            for tag in set(current_dist) | set(self.baseline)
        ) / 2
        if drift > self.threshold:
            self.alerts.append(f"Distribution drift detected: {drift:.3f}")
        return drift

    def quality_report(self):
        return {
            'total_predictions': len(self.recent_tags),
            'alerts': self.alerts[-10:],
            'drift': self.check_distribution_drift(),
            'tag_distribution': dict(Counter(self.recent_tags[-1000:]).most_common())
        }
```

**Interview Tips:**
- Highlight tag distribution drift as the most informative signal of accuracy degradation
- Mention that periodic expert auditing (random sample) provides ground truth quality measures
- Discuss that A/B testing between model versions helps validate updates safely
- Note that monitoring should cover both model quality (accuracy) and system health (latency, errors)

---

## Question 38
- [ ] Done
**What techniques work best for POS tagging in texts with special formatting or markup?**
**Answer:**

Texts with HTML, XML, LaTeX, Markdown, or code-mixed formatting contain structural elements that aren't natural language but appear in the token stream. POS taggers must handle or filter these appropriately.

**Core Concepts:**

| Format Type | Challenge | Strategy |
|---|---|---|
| HTML/XML | Tags, attributes interleaved | Strip markup, preserve text |
| LaTeX | Math, commands mixed with text | Parse LaTeX, extract text spans |
| Markdown | Headers, lists, links | Remove formatting marks |
| Code Blocks | Programming constructs in docs | Detect and skip code sections |
| Tables | Tabular data with text | Extract cell text individually |
| Inline Code | `variable_name` in text | Tag as NOUN or X |

**Python Code Example:**
```python
# Pipeline: POS tagging with markup-aware preprocessing
import re
from html import unescape

class MarkupAwarePOSTagger:
    def __init__(self, base_tagger):
        self.tagger = base_tagger

    def strip_html(self, text):
        """Remove HTML tags while preserving text and positions."""
        text = unescape(text)
        clean = re.sub(r'<[^>]+>', ' ', text)
        return re.sub(r'\s+', ' ', clean).strip()

    def strip_markdown(self, text):
        text = re.sub(r'#{1,6}\s', '', text)           # Headers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)  # Bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Inline code
        return text

    def strip_latex(self, text):
        text = re.sub(r'\$[^$]+\$', 'MATH', text)       # Inline math
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # Commands
        return text

    def tag_formatted_text(self, text, format_type='html'):
        cleaners = {'html': self.strip_html, 'markdown': self.strip_markdown,
                     'latex': self.strip_latex}
        cleaner = cleaners.get(format_type, lambda x: x)
        clean_text = cleaner(text)
        return self.tagger.predict(clean_text)
```

**Interview Tips:**
- Emphasize that markup stripping should preserve sentence boundaries and whitespace
- Mention that some tools (spaCy) handle basic HTML natively
- Discuss that code blocks should be detected and excluded, not tagged
- Note that inline code (`variable_name`) can be tagged as NOUN or given a special X tag

---

## Question 39
- [ ] Done
**How do you handle POS tagging optimization when balancing speed and accuracy?**
**Answer:**

The speed-accuracy tradeoff in POS tagging ranges from simple lookup tables (fastest, lowest accuracy) to large transformers (slowest, highest accuracy). Optimizing this tradeoff depends on application requirements.

**Core Concepts:**

| Model | Speed (tokens/sec) | Accuracy | When to Use |
|---|---|---|---|
| Lookup Table | >1M | ~90% | Autocomplete, real-time |
| HMM/Viterbi | ~500K | ~95% | Embedded devices |
| Perceptron | ~200K | ~97% | General-purpose fast |
| BiLSTM-CRF | ~50K | ~97.5% | Good balance |
| BERT-base | ~5K | ~97.8% | Accuracy-critical |
| BERT-large | ~2K | ~98% | Research, offline |

**Python Code Example:**
```python
# Pipeline: Adaptive speed-accuracy POS tagger
import time

class AdaptivePOSTagger:
    def __init__(self, models, latency_budgets):
        """
        models: dict of {name: model} ordered by speed (fastest first)
        latency_budgets: dict of {name: max_ms_per_token}
        """
        self.models = models
        self.budgets = latency_budgets

    def select_model(self, text_length, max_latency_ms):
        """Select fastest model that fits within latency budget."""
        for name, model in self.models.items():
            estimated_ms = text_length * self.budgets[name]
            if estimated_ms <= max_latency_ms:
                return name, model
        # Fallback to fastest model
        name = list(self.models.keys())[0]
        return name, self.models[name]

    def tag(self, tokens, max_latency_ms=50):
        model_name, model = self.select_model(len(tokens), max_latency_ms)
        start = time.time()
        result = model.predict(tokens)
        elapsed_ms = (time.time() - start) * 1000
        return {
            'tags': result,
            'model_used': model_name,
            'latency_ms': elapsed_ms,
            'within_budget': elapsed_ms <= max_latency_ms
        }

# Usage: configure models by speed tier
# tagger = AdaptivePOSTagger(
#     models={'lookup': lookup_model, 'perceptron': perc_model, 'bert': bert_model},
#     latency_budgets={'lookup': 0.001, 'perceptron': 0.005, 'bert': 0.2}
# )
```

**Interview Tips:**
- Note that most production systems use the fastest model that meets accuracy requirements
- Mention ONNX Runtime + quantization as the key optimization for deploying transformers
- Discuss that the perceptron tagger offers the best speed/accuracy Pareto optimum
- Highlight that batching GPU inference amortizes overhead for high-throughput scenarios

---

## Question 40
- [ ] Done
**What strategies help with POS tagging for emerging text types and social media platforms?**
**Answer:**

New social media platforms (TikTok, Discord, Mastodon) and text types (memes, threads, stories) introduce novel linguist features: platform-specific jargon, formatting conventions, emoji usage, and interaction patterns that challenge standard taggers.

**Core Concepts:**

| Platform/Type | Unique Features | POS Challenge |
|---|---|---|
| Twitter/X | Hashtags, @mentions, 280 chars | Compressed syntax, hashtag POS |
| Reddit | Markdown, subreddit jargon | Mixed formality, long threads |
| Discord | Emoji reactions, code blocks | Informal, fragmented |
| TikTok | Short captions, trend vocab | Rapidly evolving slang |
| Memes | Image-text, cultural references | Context-dependent meaning |
| Chat/IM | Abbreviations, stickers | Extreme informality |

**Python Code Example:**
```python
# Pipeline: Social media-aware POS tagging with platform-specific handling
import re

class SocialMediaPOSTagger:
    SPECIAL_TOKENS = {
        'hashtag': (r'#\w+', 'PROPN'),     # Hashtags as proper nouns
        'mention': (r'@\w+', 'PROPN'),     # Mentions as proper nouns
        'url': (r'https?://\S+', 'NOUN'),  # URLs as nouns
        'emoji': (r'[\U0001F600-\U0001F64F]', 'INTJ'),  # Emoji as interjections
        'emoticon': (r'[:;][-]?[)(DPp]', 'INTJ'),       # Text emoticons
    }

    def __init__(self, base_tagger):
        self.tagger = base_tagger

    def preprocess(self, text):
        """Identify social media tokens before POS tagging."""
        special_spans = []
        for token_type, (pattern, default_tag) in self.SPECIAL_TOKENS.items():
            for match in re.finditer(pattern, text):
                special_spans.append({
                    'start': match.start(), 'end': match.end(),
                    'text': match.group(), 'type': token_type,
                    'tag': default_tag
                })
        return special_spans

    def tag(self, text):
        special = self.preprocess(text)
        special_ranges = {(s['start'], s['end']) for s in special}
        # Tag non-special tokens with base model
        clean_text = text
        for s in reversed(sorted(special, key=lambda x: x['start'])):
            clean_text = clean_text[:s['start']] + f" {s['type'].upper()} " + clean_text[s['end']:]
        base_tags = self.tagger.predict(clean_text)
        # Merge special token tags back
        return base_tags  # Simplified; production code would align indices
```

**Interview Tips:**
- Mention BERTweet and TweetNLP as state-of-the-art tools for social media POS tagging
- Discuss the ARK Twitter POS tagset with special tags for URLs, hashtags, and emoticons
- Highlight that social media language evolves rapidly, requiring frequent model updates
- Note that platform-specific fine-tuning yields significant improvements over general models

---

## Question 41
- [ ] Done
**How do you implement cross-lingual transfer learning for multilingual POS tagging?**
**Answer:**

Cross-lingual transfer trains a POS tagger on a high-resource language (English, German) and applies it to a low-resource target language. Multilingual pre-trained models enable this through shared representation spaces.

**Core Concepts:**

| Method | Requirement | Typical Performance |
|---|---|---|
| Zero-Shot Transfer | Multilingual model + source data only | 70-85% on target language |
| Few-Shot Transfer | + small target language data | 85-95% |
| Annotation Projection | Parallel corpus with word alignments | 80-90% |
| Cross-Lingual Embeddings | Aligned embedding spaces | 75-85% |
| Universal Dependencies | Shared tagset across 100+ languages | Enables direct transfer |
| Typological Features | Language similarity features | Improves source language selection |

**Python Code Example:**
```python
# Pipeline: Cross-lingual POS tagging transfer with XLM-R
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch

def cross_lingual_pos_tagger(source_data, target_data=None):
    # Step 1: Fine-tune on high-resource source language
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base", num_labels=17  # UPOS tagset
    )

    training_args = TrainingArguments(
        output_dir="./cross_lingual_pos",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
    )

    # Train on source language (e.g., English UD treebank)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=source_data,
    )
    trainer.train()

    # Step 2: Optionally fine-tune on small target language data
    if target_data:
        target_args = TrainingArguments(
            output_dir="./cross_lingual_pos_adapted",
            num_train_epochs=10,  # More epochs for small data
            per_device_train_batch_size=8,
            learning_rate=1e-5,   # Lower LR for fine-tuning
        )
        trainer = Trainer(model=model, args=target_args, train_dataset=target_data)
        trainer.train()

    return model
```

**Interview Tips:**
- State that XLM-R is the strongest baseline for zero-shot cross-lingual POS transfer
- Mention that Universal Dependencies enables direct transfer via shared UPOS tagset
- Discuss that typologically similar languages transfer better (Spanish→Portuguese > English→Japanese)
- Note that even 100 annotated target sentences significantly improve over zero-shot transfer

---

## Question 42
- [ ] Done
**What approaches work best for POS tagging with minimal error propagation?**
**Answer:**

Error propagation occurs when POS tagging mistakes cascade into downstream tasks (parsing, NER, MT). Minimizing propagation requires confidence-aware communication between pipeline stages and joint modeling.

**Core Concepts:**

| Approach | Description | Error Reduction |
|---|---|---|
| Joint Modeling | Train POS+downstream jointly | Eliminates propagation |
| N-Best Lists | Pass top N tag sequences, not just best | Downstream selects best |
| Soft Labels | Pass probability distributions, not hard tags | Preserves uncertainty |
| CRF Layer | Sequence-level constraints reduce local errors | 0.3-0.5% accuracy gain |
| Error-Tolerant Features | Downstream uses POS softly | Reduce sensitivity |
| Pipeline Reranking | Downstream scores rerank POS hypotheses | Mutual improvement |

**Python Code Example:**
```python
# Pipeline: POS tagging with soft outputs to minimize error propagation
import torch
import torch.nn.functional as F

class SoftPOSTagger:
    """Returns probability distributions instead of hard tags."""
    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature

    def predict_soft(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens)
            probs = F.softmax(logits / self.temperature, dim=-1)
        return probs  # (batch, seq_len, num_tags)

    def predict_hard(self, tokens):
        probs = self.predict_soft(tokens)
        return probs.argmax(dim=-1)

    def predict_n_best(self, tokens, n=3):
        probs = self.predict_soft(tokens)
        top_probs, top_tags = torch.topk(probs, n, dim=-1)
        return top_tags, top_probs


class PipelineWithSoftPOS:
    """Downstream task using soft POS features."""
    def __init__(self, pos_tagger, downstream_model):
        self.tagger = pos_tagger
        self.downstream = downstream_model

    def process(self, tokens):
        # Use soft POS probabilities as features (not hard labels)
        pos_probs = self.tagger.predict_soft(tokens)
        # Concatenate soft POS features with token embeddings
        result = self.downstream(tokens, pos_features=pos_probs)
        return result
```

**Interview Tips:**
- Strongly recommend joint modeling as the best solution for error propagation
- Mention that soft labels preserve uncertainty and let downstream models make informed decisions
- Discuss that modern transformer pipelines (spaCy v3) implicitly do joint modeling
- Note that POS tagging error propagation is most critical for syntactic parsing

---

## Question 43
- [ ] Done
**How do you handle POS tagging integration with modern neural language models?**
**Answer:**

Modern neural LMs (BERT, GPT, T5) capture POS information implicitly in their representations. Integrating POS tagging with these models involves fine-tuning, probing, and architectural choices.

**Core Concepts:**

| Integration Approach | Description | When to Use |
|---|---|---|
| Fine-Tuning | Add classification head on top of LM | Standard approach, best accuracy |
| Feature Extraction | Use frozen LM embeddings + shallow classifier | When LM can't be modified |
| Probing | Test which layers encode POS information | Research and analysis |
| Multi-Task Head | POS as auxiliary task during LM training | Improves LM representations |
| Prompt-Based | Ask LM to tag in natural language | Zero-shot, no training needed |
| Adapter Layers | Add small trainable modules to frozen LM | Parameter-efficient fine-tuning |

**Python Code Example:**
```python
# Pipeline: POS tagging with modern LMs - multiple integration approaches
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
import torch
import torch.nn as nn

# Approach 1: Fine-tuning (standard)
def finetune_pos_tagger(model_name="bert-base-uncased", num_tags=17):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_tags
    )
    return model

# Approach 2: Feature extraction with frozen LM
class FrozenLMPOSTagger(nn.Module):
    def __init__(self, model_name, num_tags):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        for param in self.lm.parameters():
            param.requires_grad = False  # Freeze LM
        self.classifier = nn.Sequential(
            nn.Linear(self.lm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_tags)
        )

    def forward(self, **inputs):
        with torch.no_grad():
            lm_output = self.lm(**inputs)
        return self.classifier(lm_output.last_hidden_state)

# Approach 3: Probing different layers
def probe_pos_in_layers(model, tokenizer, text, num_tags):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Test POS separability at each layer
    for i, hidden_state in enumerate(outputs.hidden_states):
        print(f"Layer {i}: representation shape = {hidden_state.shape}")
```

**Interview Tips:**
- State that BERT middle layers (6-8 of 12) encode the most POS information
- Mention that fine-tuning the full model gives best accuracy but is expensive
- Discuss adapter layers (LoRA) as parameter-efficient alternatives to full fine-tuning
- Note that GPT-style models can do POS tagging via prompting but are less efficient

---

## Question 44
- [ ] Done
**What techniques help with POS tagging for texts requiring morphological analysis?**
**Answer:**

Morphological analysis decomposes words into morphemes and assigns morphological features (case, number, gender, tense, aspect). Joint POS+morphological tagging leverages the synergy between POS categories and morphological features.

**Core Concepts:**

| Technique | Description | Example |
|---|---|---|
| Joint POS+Morph | Predict POS tag + morphological features together | "running" → VERB + Tense=Pres|VerbForm=Part |
| Morphological Analyzers | FST-based analyzers enumerate possible analyses | Apertium, HFST |
| Character Seq2Seq | Sequence model learns morpheme boundaries | Neural morphological segmentation |
| Feature Templates | Predict each morph feature independently | Case=Nom, Number=Sg, Gender=Masc |
| Paradigm Completion | Fill in missing inflection forms | Given "run/runs", predict "running" |
| Subword Composition | BPE segments approximate morphemes | Implicitly captures morphology |

**Python Code Example:**
```python
# Pipeline: Joint POS + morphological feature prediction
import torch
import torch.nn as nn

class JointPOSMorphTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, pos_tags, morph_features):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pos_head = nn.Linear(hidden_dim * 2, pos_tags)
        # Separate head for each morphological feature
        self.morph_heads = nn.ModuleDict({
            feat_name: nn.Linear(hidden_dim * 2, num_values)
            for feat_name, num_values in morph_features.items()
        })

    def forward(self, x):
        encoded, _ = self.encoder(x)
        pos_logits = self.pos_head(encoded)
        morph_logits = {
            feat: head(encoded) for feat, head in self.morph_heads.items()
        }
        return pos_logits, morph_logits

# Feature set example:
# morph_features = {
#     'Case': 6,    # Nom, Acc, Dat, Gen, Ins, Loc
#     'Number': 3,  # Sg, Pl, Dual
#     'Gender': 4,  # Masc, Fem, Neut, Com
#     'Tense': 4,   # Past, Pres, Fut, None
#     'Person': 4,  # 1, 2, 3, None
# }
```

**Interview Tips:**
- Mention that Universal Dependencies includes morphological features alongside UPOS tags
- Discuss that joint prediction outperforms pipeline (POS→morph) for morphologically rich languages
- Highlight that character-level models implicitly learn morphological patterns
- Note UDPipe and Stanza as tools that perform joint POS+morphological analysis

---

## Question 45
- [ ] Done
**How do you implement customizable POS tagging for different linguistic frameworks?**
**Answer:**

Different linguistic frameworks define POS categories differently. A customizable tagger provides pluggable tagsets, configurable features, and framework-aligned output formats to serve diverse linguistic communities.

**Core Concepts:**

| Framework | Tagset Style | Key Differences |
|---|---|---|
| Universal Dependencies | 17 UPOS + features | Cross-lingual, morphological features |
| Penn Treebank | 45 fine-grained tags | English-specific, syntax-oriented |
| CLAWS | 62-137 tags | Corpus-linguistics oriented |
| STTS | 54 tags | German-specific |
| Custom/Domain | User-defined | Application-specific categories |

**Python Code Example:**
```python
# Pipeline: Framework-configurable POS tagger
class ConfigurablePOSTagger:
    FRAMEWORKS = {
        'upos': {
            'tags': ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN',
                     'NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X'],
            'features': True,
        },
        'ptb': {
            'tags': ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD',
                     'NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR',
                     'RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ',
                     'WDT','WP','WP$','WRB'],
            'features': False,
        },
    }

    def __init__(self, model, framework='upos'):
        self.model = model
        self.framework = framework
        self.config = self.FRAMEWORKS.get(framework, self.FRAMEWORKS['upos'])

    def tag(self, text):
        base_tags = self.model.predict(text)
        # Convert to target framework
        converted = self.convert_tags(base_tags)
        result = {'tokens': text.split(), 'tags': converted}
        if self.config['features']:
            result['features'] = self.model.predict_features(text)
        return result

    def convert_tags(self, tags):
        if self.framework == self.model.native_framework:
            return tags
        # Apply mapping between frameworks
        mapping = self.get_mapping(self.model.native_framework, self.framework)
        return [mapping.get(t, 'X') for t in tags]

    def get_mapping(self, source, target):
        # Predefined mappings between common frameworks
        mappings = {
            ('ptb', 'upos'): {'NN': 'NOUN', 'NNS': 'NOUN', 'VB': 'VERB'},
            ('upos', 'ptb'): {'NOUN': 'NN', 'VERB': 'VB'},  # Lossy
        }
        return mappings.get((source, target), {})
```

**Interview Tips:**
- Emphasize that Universal Dependencies is the de facto standard for multilingual POS tagging
- Mention that framework conversion is inherently lossy (fine→coarse loses information)
- Discuss that Stanza and spaCy support both UPOS and language-specific tagsets
- Note that custom tagsets require custom annotated training data

---

## Question 46
- [ ] Done
**What strategies work best for POS tagging in streaming text processing scenarios?**
**Answer:**

Streaming POS tagging processes text incrementally as tokens arrive, without access to the complete sentence. This is essential for real-time captioning, live translation, typing prediction, and streaming analytics.

**Core Concepts:**

| Strategy | Description | Tradeoff |
|---|---|---|
| Left-to-Right Tagging | Tag each token using only left context | Fast, lower accuracy |
| Buffered Tagging | Collect N tokens, tag buffer, slide window | Latency vs. accuracy |
| Incremental LSTM | Maintain hidden state across token arrivals | Good balance |
| Revision-Based | Tag tentative, revise when more context arrives | Higher accuracy, complex |
| Lookahead Buffer | Wait for k future tokens before committing | k-token delay |
| Streaming Transformer | Causal attention mask for left-only context | Compatible with GPT-style |

**Python Code Example:**
```python
# Pipeline: Streaming POS tagger with buffered windowed approach
import torch
from collections import deque

class StreamingPOSTagger:
    def __init__(self, model, buffer_size=10, lookahead=3):
        self.model = model
        self.buffer = deque(maxlen=buffer_size)
        self.lookahead = lookahead
        self.pending = []  # Tokens waiting for lookahead
        self.committed_tags = []  # Finalized tags

    def process_token(self, token):
        """Process one incoming token."""
        self.pending.append(token)
        self.buffer.append(token)

        # Wait for lookahead context
        if len(self.pending) <= self.lookahead:
            return None  # Not ready to commit yet

        # Tag the buffer and commit the oldest pending token
        buffer_list = list(self.buffer)
        tags = self.model.predict(buffer_list)
        # Commit tag for the token that now has enough right context
        commit_idx = len(buffer_list) - self.lookahead - 1
        if 0 <= commit_idx < len(tags):
            committed_tag = tags[commit_idx]
            committed_token = self.pending.pop(0)
            self.committed_tags.append((committed_token, committed_tag))
            return (committed_token, committed_tag)
        return None

    def flush(self):
        """Flush remaining tokens at end of stream."""
        if self.pending:
            tags = self.model.predict(self.pending)
            for token, tag in zip(self.pending, tags):
                self.committed_tags.append((token, tag))
            self.pending = []
        return self.committed_tags
```

**Interview Tips:**
- Mention that buffered approaches with 3-5 token lookahead achieve near-batch accuracy
- Discuss that revision-based tagging (tag then correct) works well for UI applications
- Highlight that GPT-style models are naturally suited for left-to-right streaming
- Note that streaming adds 2-5% accuracy cost compared to full-sentence batch tagging

---

## Question 47
- [ ] Done
**How do you handle POS tagging quality benchmarking across different languages?**
**Answer:**

Cross-lingual benchmarking evaluates POS taggers consistently across languages using shared metrics, tagsets, and evaluation protocols. Universal Dependencies (UD) provides the standard framework for this.

**Core Concepts:**

| Aspect | Standard | Details |
|---|---|---|
| Tagset | Universal POS (UPOS) | 17 tags, consistent across 100+ languages |
| Data | Universal Dependencies Treebanks | 200+ treebanks, 100+ languages |
| Metrics | Accuracy, per-tag F1 | Token-level evaluation |
| Splits | Standard train/dev/test | Fixed splits for reproducibility |
| Evaluation Script | `conll18_ud_eval.py` | Official shared task evaluator |
| Baselines | UDPipe, Stanza, Trankit | Standard comparison models |

**Python Code Example:**
```python
# Pipeline: Cross-lingual POS benchmarking with UD data
from collections import Counter, defaultdict

class POSBenchmark:
    def __init__(self, languages):
        self.languages = languages
        self.results = defaultdict(dict)

    def evaluate_language(self, tagger, test_data, language):
        """Evaluate POS accuracy on one language."""
        correct, total = 0, 0
        tag_correct = Counter()
        tag_total = Counter()

        for sentence, gold_tags in test_data:
            predicted = tagger.predict(sentence)
            for pred, gold in zip(predicted, gold_tags):
                total += 1
                tag_total[gold] += 1
                if pred == gold:
                    correct += 1
                    tag_correct[gold] += 1

        accuracy = correct / total if total > 0 else 0
        per_tag_f1 = {
            tag: tag_correct[tag] / tag_total[tag]
            for tag in tag_total
        }
        self.results[language] = {
            'accuracy': accuracy, 'per_tag': per_tag_f1, 'total_tokens': total
        }
        return accuracy

    def cross_lingual_report(self):
        print(f"{'Language':<15} {'Accuracy':<10} {'Tokens':<10}")
        print("-" * 35)
        for lang, res in sorted(self.results.items()):
            print(f"{lang:<15} {res['accuracy']:.4f}    {res['total_tokens']:<10}")
        avg = sum(r['accuracy'] for r in self.results.values()) / len(self.results)
        print(f"\nAverage: {avg:.4f}")
```

**Interview Tips:**
- Mention Universal Dependencies as the gold standard for cross-lingual POS benchmarking
- Discuss that accuracy varies significantly by language typology (90-98%)
- Highlight that per-tag F1 reveals which categories are problematic in each language
- Note that CoNLL shared tasks (2017, 2018) established standard evaluation protocols

---

## Question 48
- [ ] Done
**What approaches help with POS tagging for texts with evolving grammatical patterns?**
**Answer:**

Language evolves continuously: new grammatical constructions emerge ("because NOUN"), function words shift categories ("like" as discourse marker), and usage patterns change. POS taggers must adapt without full retraining.

**Core Concepts:**

| Pattern | Example | Tagging Challenge |
|---|---|---|
| Conversion/Zero-Derivation | "to google" (PROPN→VERB) | New POS for existing word |
| Grammaticalization | "going to" → future marker | Function shift |
| New Constructions | "because reasons" | Nonstandard syntax |
| Semantic Bleaching | "literally" as intensifier | ADV usage changes |
| Category Shift | "adult" as verb ("adulting") | Novel word class |
| Compounding | "doomscrolling" | New compound words |

**Python Code Example:**
```python
# Pipeline: Evolving grammar POS tagger with temporal adaptation
from collections import defaultdict
import datetime

class EvolvingGrammarTagger:
    def __init__(self, base_tagger):
        self.base = base_tagger
        self.temporal_overrides = defaultdict(list)  # word → [(date, tag)]
        self.construction_patterns = []  # Detected new constructions

    def add_temporal_rule(self, word, new_tag, effective_date=None):
        """Register that a word's POS has shifted."""
        if effective_date is None:
            effective_date = datetime.date.today()
        self.temporal_overrides[word.lower()].append((effective_date, new_tag))

    def add_construction(self, pattern, tag_override):
        """Register a new grammatical construction."""
        self.construction_patterns.append((pattern, tag_override))

    def tag(self, tokens, text_date=None):
        base_tags = self.base.predict(tokens)
        result = list(zip(tokens, base_tags))

        # Apply temporal overrides
        for i, (token, tag) in enumerate(result):
            overrides = self.temporal_overrides.get(token.lower(), [])
            if overrides and text_date:
                applicable = [(d, t) for d, t in overrides if d <= text_date]
                if applicable:
                    _, new_tag = max(applicable)  # Most recent override
                    result[i] = (token, new_tag)

        return result

# Example: "google" shifted from PROPN to also being a VERB
tagger = EvolvingGrammarTagger(base_tagger=None)
tagger.add_temporal_rule('google', 'VERB', datetime.date(2005, 1, 1))
tagger.add_temporal_rule('adulting', 'VERB', datetime.date(2016, 1, 1))
```

**Interview Tips:**
- Mention that language change happens at multiple levels: lexical, syntactic, and semantic
- Discuss that continual learning approaches enable models to adapt to linguistic evolution
- Highlight "verbing" (noun→verb conversion) as the most common productive POS change in English
- Note that diachronic corpora track how word categories shift over time

---

## Question 49
- [ ] Done
**How do you implement efficient storage and retrieval of POS tagging results?**
**Answer:**

Storing and retrieving POS tags for large corpora requires efficient data formats, indexing, and compression. Annotated corpora can be billions of tokens, requiring careful storage design.

**Core Concepts:**

| Format | Pros | Cons | Best For |
|---|---|---|---|
| CoNLL-U | Standard, human-readable | Large files | Linguistic research |
| Binary/Protobuf | Compact, fast loading | Not human-readable | Production systems |
| SQLite/DB | Queryable, indexed | Overhead per query | Interactive search |
| Columnar (Parquet) | Efficient compression, fast analytics | Complex setup | Large-scale analytics |
| In-Memory Cache | Fastest retrieval | RAM-limited | Frequent re-access |
| Compressed JSON | Web-compatible, flexible | Larger than binary | API responses |

**Python Code Example:**
```python
# Pipeline: Efficient POS tagging storage and retrieval
import json
import sqlite3
import struct

class POSTagStorage:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS tagged_sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT, tokens TEXT, tags TEXT,
            source TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON tagged_sentences(source)')

    def store(self, text, tokens, tags, source='default'):
        self.conn.execute(
            'INSERT INTO tagged_sentences (text, tokens, tags, source) VALUES (?, ?, ?, ?)',
            (text, json.dumps(tokens), json.dumps(tags), source)
        )
        self.conn.commit()

    def retrieve_by_tag(self, tag, limit=100):
        """Find sentences containing a specific POS tag."""
        cursor = self.conn.execute(
            'SELECT text, tokens, tags FROM tagged_sentences WHERE tags LIKE ? LIMIT ?',
            (f'%"{tag}"%', limit)
        )
        results = []
        for text, tokens_json, tags_json in cursor:
            tokens = json.loads(tokens_json)
            tags = json.loads(tags_json)
            matching = [(t, tg) for t, tg in zip(tokens, tags) if tg == tag]
            results.append({'text': text, 'matches': matching})
        return results

    def export_conllu(self, output_path, source=None):
        """Export to CoNLL-U format."""
        query = 'SELECT tokens, tags FROM tagged_sentences'
        if source:
            query += f" WHERE source = '{source}'"
        cursor = self.conn.execute(query)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent_id, (tokens_json, tags_json) in enumerate(cursor, 1):
                tokens = json.loads(tokens_json)
                tags = json.loads(tags_json)
                f.write(f"# sent_id = {sent_id}\n")
                for i, (token, tag) in enumerate(zip(tokens, tags), 1):
                    f.write(f"{i}\t{token}\t_\t{tag}\t_\t_\t_\t_\t_\t_\n")
                f.write("\n")
```

**Interview Tips:**
- Recommend CoNLL-U format for interoperability with linguistic tools
- Mention that columnar formats (Parquet, Arrow) are best for large-scale analytics
- Discuss that tag-level indexing enables fast queries like "find all PROPN tokens"
- Note that compression ratios of 5-10x are typical for POS-annotated text

---

## Question 50
- [ ] Done
**What techniques work best for balancing POS tagging accuracy with processing efficiency?**
**Answer:**

Balancing accuracy and efficiency requires understanding the accuracy-cost curve and selecting the optimal operating point for each application. The marginal accuracy gain from larger models often doesn't justify the cost.

**Core Concepts:**

| Technique | Accuracy Impact | Efficiency Gain | Method |
|---|---|---|---|
| Model Selection | Choose right model size | Avoid over-engineering | Perceptron for simple tasks |
| Quantization | -0.1-0.3% | 2-4x speedup | INT8 weights |
| Pruning | -0.2-0.5% | 1.5-3x speedup | Remove redundant neurons |
| Knowledge Distillation | -0.3-1% | 5-10x speedup | Compress BERT to BiLSTM |
| Early Exit | Varies by input | 1.5-2x average speedup | Easy inputs exit early |
| Caching | 0% (exact) | 10-100x for cache hits | Memoize results |

**Python Code Example:**
```python
# Pipeline: Efficiency-optimized POS tagging with early exit and caching
import torch
from functools import lru_cache

class EfficientPOSTagger:
    def __init__(self, layers, classifiers, confidence_threshold=0.95):
        self.layers = layers          # List of encoder layers
        self.classifiers = classifiers  # Classifier per layer for early exit
        self.threshold = confidence_threshold
        self.stats = {'early_exits': 0, 'full_passes': 0}

    def tag_with_early_exit(self, tokens):
        """Exit early when confidence exceeds threshold."""
        hidden = self.layers[0].embed(tokens)
        for i, (layer, classifier) in enumerate(zip(self.layers, self.classifiers)):
            hidden = layer(hidden)
            logits = classifier(hidden)
            confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
            if confidence.min() > self.threshold:
                self.stats['early_exits'] += 1
                return logits.argmax(dim=-1)
        self.stats['full_passes'] += 1
        return logits.argmax(dim=-1)

    @lru_cache(maxsize=50000)
    def tag_cached(self, tokens_tuple):
        """Cache results for repeated inputs."""
        return self.tag_with_early_exit(list(tokens_tuple))

    def efficiency_report(self):
        total = self.stats['early_exits'] + self.stats['full_passes']
        return {
            'early_exit_rate': self.stats['early_exits'] / max(total, 1),
            'avg_layers_used': 'varies',  # Track per-sample
            'cache_info': self.tag_cached.cache_info(),
        }
```

**Interview Tips:**
- State that early exit is one of the most effective efficiency techniques for transformers
- Mention that 60-80% of tokens can be correctly tagged by early layers
- Discuss the diminishing returns curve: last 1% accuracy costs 5-10x compute
- Note that the optimal accuracy-efficiency tradeoff depends entirely on the application SLA

---


---

# --- Text Classification Questions (from 08_nlp/04_text_classification) ---

# Text Classification - Theory Questions

## Question 1
- [ ] Done
**How do you handle text classification for extremely imbalanced datasets with rare classes?**
**Answer:**

Class imbalance is common in real-world text classification — spam detection (1% spam), medical coding (rare diseases), and fraud detection. Standard classifiers become biased toward the majority class, achieving high accuracy but poor recall on minority classes.

**Core Concepts:**

| Strategy | Category | Description |
|---|---|---|
| Oversampling (SMOTE) | Data-Level | Generate synthetic minority examples | 
| Undersampling | Data-Level | Reduce majority class size |
| Class Weights | Algorithm-Level | Weight loss inversely proportional to frequency |
| Focal Loss | Loss Function | Down-weight easy, up-weight hard examples |
| Few-Shot Learning | Model-Level | Prototypical/siamese networks for rare classes |
| Data Augmentation | Data-Level | Back-translation, synonym replacement for minority |
| Threshold Tuning | Post-Processing | Adjust decision threshold per class |

**Python Code Example:**
```python
# Pipeline: Handling imbalanced text classification with multiple strategies
import torch
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def compute_class_weights(labels):
    """Compute inverse frequency weights for class balancing."""
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)

# Usage with BERT classifier
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
class_weights = compute_class_weights(train_labels)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

**Interview Tips:**
- Recommend class-weighted loss as the simplest first approach for imbalance
- Mention that focal loss (gamma=2) outperforms class weights for extreme imbalance (>100:1)
- Discuss evaluation metrics: use macro-F1 or AUC-PR instead of accuracy
- Note that oversampling + augmentation of minority class text is very effective

---

## Question 2
- [ ] Done
**What techniques work best for multi-label text classification with label dependencies?**
**Answer:**

Multi-label classification assigns multiple labels to each text (e.g., a news article can be "politics", "economy", AND "international"). Label dependencies mean some labels frequently co-occur or are mutually exclusive.

**Core Concepts:**

| Technique | Description | Handles Dependencies |
|---|---|---|
| Binary Relevance | Independent classifier per label | No (baseline) |
| Classifier Chains | Sequential classifiers, each uses previous predictions | Yes (sequential) |
| Label Powerset | Treat each label combination as one class | Yes (but exponential) |
| Graph Neural Networks | Model label co-occurrence as graph | Yes (pairwise) |
| Seq2Seq | Generate label set as sequence | Yes (flexible) |
| Attention-Based | Cross-attention between text and labels | Yes (soft) |

**Python Code Example:**
```python
# Pipeline: Multi-label text classification with label correlation
import torch
import torch.nn as nn
from transformers import AutoModel

class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels, label_adj_matrix=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        # Label correlation matrix (from training data co-occurrence)
        if label_adj_matrix is not None:
            self.label_gcn = nn.Linear(num_labels, num_labels)
            self.adj = nn.Parameter(torch.FloatTensor(label_adj_matrix), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(cls_repr)
        # Refine with label correlation (optional GCN-style)
        if hasattr(self, 'label_gcn'):
            label_corr = torch.sigmoid(self.label_gcn(self.adj))
            logits = logits * label_corr.mean(dim=0)
        return logits

# Use BCEWithLogitsLoss for multi-label
criterion = nn.BCEWithLogitsLoss()
# threshold = 0.5 default, tune per-label on validation set
```

**Interview Tips:**
- Mention BCEWithLogitsLoss as the standard loss for multi-label classification
- Discuss that per-label threshold tuning on validation data improves F1 significantly
- Highlight classifier chains as a simple way to model label dependencies
- Note that label co-occurrence statistics from training data can be injected as priors

---

## Question 3
- [ ] Done
**How do you implement domain adaptation for text classifiers across different text sources?**
**Answer:**

Domain adaptation bridges the distribution gap between source training data and target deployment data. A sentiment classifier trained on product reviews may fail on restaurant reviews due to vocabulary and style differences.

**Core Concepts:**

| Method | Requirement | Description |
|---|---|---|
| Fine-Tuning | Small labeled target data | Continue training on target domain |
| Domain-Adversarial (DANN) | Unlabeled target data | Learn domain-invariant features |
| Self-Training | Unlabeled target data | Use confident predictions as pseudo-labels |
| Pivot Features | Shared vocabulary analysis | Identify domain-shared features |
| Pre-train + Adapt | Domain text (unlabeled) | MLM on domain text, then fine-tune |
| Data Selection | Large mixed corpus | Select source examples similar to target |

**Python Code Example:**
```python
# Pipeline: Domain adaptation for text classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Strategy: Domain-adaptive pretraining (DAPT) + fine-tuning
def domain_adapt_classifier(source_data, target_unlabeled, target_labeled=None):
    # Step 1: Continue pre-training on target domain (MLM)
    from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
    mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    mlm_args = TrainingArguments(
        output_dir="./domain_mlm", num_train_epochs=5,
        per_device_train_batch_size=16, learning_rate=5e-5,
    )
    mlm_trainer = Trainer(
        model=mlm_model, args=mlm_args, train_dataset=target_unlabeled,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15),
    )
    mlm_trainer.train()
    mlm_model.save_pretrained("./domain_adapted_bert")

    # Step 2: Fine-tune classifier on source + optional target labeled data
    classifier = AutoModelForSequenceClassification.from_pretrained(
        "./domain_adapted_bert", num_labels=3
    )
    return classifier
```

**Interview Tips:**
- Recommend DAPT (domain-adaptive pretraining) as the standard first approach
- Mention that self-training with confidence filtering is simple and surprisingly effective
- Discuss the don't-stop-pretraining paper (Gururangan et al., 2020) as key reference
- Note that even a few hundred labeled target examples with fine-tuning closes most domain gaps

---

## Question 4
- [ ] Done
**What strategies help with handling text classification for very long documents?**
**Answer:**

Standard transformers have 512-token limits, but many real texts (legal contracts, research papers, medical records) are thousands of tokens. Special strategies handle these long documents without truncation information loss.

**Core Concepts:**

| Strategy | Description | Max Length |
|---|---|---|
| Truncation | Keep first/last N tokens | 512 tokens |
| Chunking + Aggregation | Split into chunks, aggregate predictions | Unlimited |
| Hierarchical Attention | Sentence-level then document-level attention | 10K+ tokens |
| Longformer/BigBird | Sparse attention patterns | 4096-16K tokens |
| Extractive Summary | Summarize first, classify summary | Unlimited |
| Key Sentence Selection | Select most informative sentences | Unlimited |

**Python Code Example:**
```python
# Pipeline: Long document classification with chunking and aggregation
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class LongDocClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=5, max_chunk=512, overlap=50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.max_chunk = max_chunk
        self.overlap = overlap

    def chunk_document(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        stride = self.max_chunk - self.overlap - 2  # Account for [CLS], [SEP]
        for i in range(0, len(tokens), stride):
            chunk = [self.tokenizer.cls_token_id] + tokens[i:i+self.max_chunk-2] + [self.tokenizer.sep_token_id]
            chunks.append(chunk)
        return chunks

    def classify(self, text):
        chunks = self.chunk_document(text)
        chunk_reprs = []
        for chunk in chunks:
            inputs = torch.tensor([chunk])
            attention = torch.ones_like(inputs)
            with torch.no_grad():
                output = self.encoder(input_ids=inputs, attention_mask=attention)
            chunk_reprs.append(output.last_hidden_state[:, 0])  # [CLS]

        # Aggregate: mean pooling over chunks
        doc_repr = torch.stack(chunk_reprs).mean(dim=0)
        logits = self.classifier(doc_repr)
        return logits
```

**Interview Tips:**
- Recommend Longformer or BigBird for documents up to 4096 tokens
- For documents >4096 tokens, use chunking + hierarchical aggregation
- Mention that first + last 256 tokens often capture most classification signal
- Note that hierarchical attention (sentence→document) is interpretable and effective

---

## Question 5
- [ ] Done
**How do you design text classifiers that work effectively with limited labeled data?**
**Answer:**

Low-data text classification is critical for bootstrapping new categories, niche domains, and rapid prototyping. Techniques range from few-shot learning (5-20 examples) to semi-supervised approaches (100s of examples + unlabeled data).

**Core Concepts:**

| Technique | Data Required | Expected Performance |
|---|---|---|
| Zero-Shot (LLM Prompting) | 0 labeled examples | 60-80% (varies by task) |
| Few-Shot In-Context Learning | 5-20 examples | 70-85% |
| SetFit | 8-16 examples per class | 85-90% |
| Pattern-Exploiting Training (PET) | 10-50 examples | 85-92% |
| Fine-Tuning + Augmentation | 100-500 examples | 90-95% |
| Semi-Supervised (UDA, MixText) | 100 labeled + unlabeled | 90-95% |

**Python Code Example:**
```python
# Pipeline: Few-shot text classification with SetFit
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

class FewShotClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000)

    def train(self, texts, labels):
        """Train with as few as 5 examples per class."""
        embeddings = self.encoder.encode(texts)
        self.classifier.fit(embeddings, labels)

    def predict(self, texts):
        embeddings = self.encoder.encode(texts)
        return self.classifier.predict(embeddings)

    def predict_proba(self, texts):
        embeddings = self.encoder.encode(texts)
        return self.classifier.predict_proba(embeddings)

# Zero-shot with LLM
def zero_shot_classify(text, candidate_labels):
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return dict(zip(result['labels'], result['scores']))
```

**Interview Tips:**
- Recommend SetFit for 8-16 examples per class — outperforms fine-tuning with so few examples
- Mention zero-shot classification with BART-MNLI as requiring zero labeled data
- Discuss data augmentation (back-translation, EDA) to multiply limited labeled examples
- Note that sentence embeddings + logistic regression is a strong few-shot baseline

---

## Question 6
- [ ] Done
**What approaches work best for text classification in multilingual or cross-lingual settings?**
**Answer:**

Multilingual text classification requires models that work across languages, either through shared multilingual models or cross-lingual transfer from high-resource to low-resource languages.

**Core Concepts:**

| Approach | Description | Languages Supported |
|---|---|---|
| XLM-R Fine-Tuning | Fine-tune multilingual model on source language | 100+ languages |
| Translate-Train | Translate training data, train per-language | Any with MT available |
| Translate-Test | Translate test inputs to source language | Any with MT available |
| Zero-Shot Transfer | Train on English, apply to other languages | 100+ with XLM-R |
| Language-Agnostic BERT (LaBSE) | Language-agnostic sentence embeddings | 100+ languages |
| Multi-Source Transfer | Train on multiple source languages | Better than single-source |

**Python Code Example:**
```python
# Pipeline: Multilingual text classification with XLM-R
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def multilingual_classifier(train_data, num_labels=3):
    # XLM-R provides strong cross-lingual transfer
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="./multilingual_clf",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_data,
    )
    trainer.train()
    return model

# Zero-shot: train on English, evaluate on French/German/etc.
# XLM-R trained on English sentiment achieves ~85% on French sentiment
```

**Interview Tips:**
- State that XLM-R is the strongest general-purpose multilingual model
- Mention translate-train as a practical alternative when MT quality is good
- Discuss that typologically similar languages transfer better (Romance→Romance)
- Note that multi-source training (English + French + German) gives best results

---

## Question 7
- [ ] Done
**How do you handle text classification for noisy or poorly formatted text data?**
**Answer:**

Noisy text includes OCR errors, ASR transcription mistakes, user-generated content with typos, formatting artifacts, and HTML/code mixed with natural language. Noise can reduce classification accuracy by 10-20% if unaddressed.

**Core Concepts:**

| Noise Type | Cleaning Strategy | Example |
|---|---|---|
| Typos/Misspellings | Spell correction, character-level models | "teh" → "the" |
| HTML/Markup | Strip tags, extract text | Remove `<div>`, `<p>` tags |
| Encoding Issues | Detect and fix encoding | Mojibake repair |
| OCR Errors | Error correction model, confidence filtering | "rn" → "m" |
| Formatting Noise | Normalize whitespace, remove headers/footers | Clean extraction |
| Data Augmentation | Train on synthetically noised data | Noise-robust model |

**Python Code Example:**
```python
# Pipeline: Noise-robust text classification with preprocessing
import re
import unicodedata
from html import unescape

class NoisyTextPreprocessor:
    def __init__(self):
        self.noise_patterns = [
            (r'<[^>]+>', ' '),            # HTML tags
            (r'&\w+;', ' '),              # HTML entities
            (r'http\S+', '[URL]'),         # URLs
            (r'\S+@\S+', '[EMAIL]'),       # Emails
            (r'[^\w\s.,!?;:\'-]', ''),     # Special characters
            (r'\s+', ' '),                 # Multiple whitespace
        ]

    def clean(self, text):
        text = unescape(text)  # Decode HTML entities
        text = unicodedata.normalize('NFKD', text)  # Normalize unicode
        for pattern, replacement in self.noise_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()

    def noise_augment(self, text, p=0.1):
        """Add synthetic noise for training robustness."""
        import random
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p:
                action = random.choice(['delete', 'swap', 'insert'])
                if action == 'delete' and len(chars) > 1:
                    chars[i] = ''
                elif action == 'swap' and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif action == 'insert':
                    chars[i] = chars[i] + random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(chars)
```

**Interview Tips:**
- Recommend preprocessing pipeline + noise-augmented training as dual defense
- Mention that character-level models (CharCNN) are inherently robust to spelling noise
- Discuss that pre-trained models (BERT) handle moderate noise reasonably well without special treatment
- Note that noise augmentation during training is more robust than perfect cleaning

---

## Question 8
- [ ] Done
**What techniques help with explaining text classification decisions to end users?**
**Answer:**

Explainability in text classification helps users understand why a document was classified a certain way, enabling trust, debugging, and compliance with regulations like GDPR's "right to explanation."

**Core Concepts:**

| Technique | Type | Output |
|---|---|---|
| LIME | Model-Agnostic | Word-level importance scores |
| SHAP | Model-Agnostic | Shapley value attributions |
| Attention Visualization | Model-Specific | Attention weight heatmaps |
| Integrated Gradients | Gradient-Based | Input feature attributions |
| Anchor Explanations | Rule-Based | Sufficient conditions for prediction |
| Counterfactual | Contrastive | Minimal change that flips prediction |

**Python Code Example:**
```python
# Pipeline: Explaining text classification with LIME
from lime.lime_text import LimeTextExplainer
import numpy as np

class ExplainableTextClassifier:
    def __init__(self, model, tokenizer, class_names):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(self, texts):
        """Wrapper for LIME compatibility."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        import torch
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1).numpy()

    def explain(self, text, num_features=10):
        exp = self.explainer.explain_instance(
            text, self.predict_proba, num_features=num_features
        )
        return {
            'prediction': exp.predict_proba,
            'top_features': exp.as_list(),  # (word, weight) pairs
            'html': exp.as_html(),
        }
```

**Interview Tips:**
- Recommend LIME for production explainability due to model-agnostic simplicity
- Mention that attention weights are intuitive but NOT faithful explanations
- Discuss integrated gradients as the most theoretically grounded attribution method
- Note that counterfactual explanations are most useful for end users ("change X to flip")

---

## Question 9
- [ ] Done
**How do you implement active learning strategies for efficient text classification annotation?**
**Answer:**

Active learning selects the most informative examples for human annotation, reducing labeling costs by 50-80% while maintaining classification accuracy. This is essential when labeling budgets are limited.

**Core Concepts:**

| Strategy | Description | Selection Criterion |
|---|---|---|
| Uncertainty Sampling | Pick examples where model is least confident | Lowest max probability |
| Margin Sampling | Pick examples with smallest margin between top-2 classes | Smallest P(1st) - P(2nd) |
| Entropy Sampling | Pick highest prediction entropy | Most uncertain overall |
| Query-by-Committee | Pick where ensemble members disagree most | Maximum vote entropy |
| Diversity Sampling | Pick representative diverse examples | Cluster-based selection |
| Expected Error Reduction | Pick examples that would most reduce error | Computationally expensive |

**Python Code Example:**
```python
# Pipeline: Active learning for text classification
import numpy as np

class ActiveTextClassifier:
    def __init__(self, model, strategy='entropy'):
        self.model = model
        self.strategy = strategy
        self.labeled_data = []

    def score_unlabeled(self, texts):
        probs = self.model.predict_proba(texts)  # (n_samples, n_classes)
        if self.strategy == 'entropy':
            scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif self.strategy == 'margin':
            sorted_probs = np.sort(probs, axis=1)
            scores = -(sorted_probs[:, -1] - sorted_probs[:, -2])  # Negative margin
        elif self.strategy == 'uncertainty':
            scores = 1 - np.max(probs, axis=1)
        return scores

    def select_batch(self, unlabeled_texts, batch_size=50):
        scores = self.score_unlabeled(unlabeled_texts)
        top_indices = np.argsort(scores)[-batch_size:]  # Highest scores
        return [(i, unlabeled_texts[i]) for i in top_indices]

    def update(self, texts, labels):
        self.labeled_data.extend(zip(texts, labels))
        all_texts, all_labels = zip(*self.labeled_data)
        self.model.fit(list(all_texts), list(all_labels))
```

**Interview Tips:**
- Recommend entropy sampling as the default active learning strategy for text classification
- Mention that batch active learning with diversity prevents selecting redundant examples
- Discuss cold-start: first batch should be randomly selected for diversity
- Note that Prodigy combines active learning with efficient annotation UI for production use

---

## Question 10
- [ ] Done
**What strategies work best for text classification in specialized domains like legal or medical?**
**Answer:**

Specialized domains have unique vocabularies, complex sentence structures, and require domain expertise for accurate classification. Domain-specific pre-trained models and careful data curation are essential.

**Core Concepts:**

| Domain | Key Challenge | Recommended Model |
|---|---|---|
| Medical | Complex terminology, abbreviations | BioBERT, PubMedBERT, ClinicalBERT |
| Legal | Long documents, cross-references | LegalBERT, Longformer |
| Financial | Formal language, numerical reasoning | FinBERT |
| Scientific | Technical vocabulary, citations | SciBERT, ScholarBERT |
| Cybersecurity | Jargon, evolving threats | SecureBERT |

**Python Code Example:**
```python
# Pipeline: Domain-specific text classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def domain_specific_classifier(domain='medical', train_data=None, num_labels=5):
    domain_models = {
        'medical': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'legal': 'nlpaueb/legal-bert-base-uncased',
        'financial': 'yiyanghkust/finbert-tone',
        'scientific': 'allenai/scibert_scivocab_uncased',
    }
    model_name = domain_models.get(domain, 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=f"./{domain}_classifier",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()
    return model, tokenizer
```

**Interview Tips:**
- Always start with a domain-specific pre-trained model instead of general BERT
- Mention that PubMedBERT outperforms BioBERT on biomedical tasks
- Discuss that legal text classification benefits from Longformer due to document length
- Note that domain expertise is needed for annotation guidelines and error analysis

---

## Question 11
- [ ] Done
**How do you handle text classification quality control and confidence scoring?**
**Answer:**

Quality control ensures predictions meet required standards and provides calibrated confidence scores that downstream systems can use for routing, flagging, or human review.

**Core Concepts:**

| Aspect | Method | Purpose |
|---|---|---|
| Calibration | Temperature scaling, Platt scaling | Align confidence with accuracy |
| Abstention | Reject low-confidence predictions | Maintain precision at cost of coverage |
| Ensemble Agreement | Multiple models must agree | Higher confidence threshold |
| Conformal Prediction | Prediction sets with coverage guarantees | Statistical confidence guarantee |
| Stratified Evaluation | Per-class, per-domain accuracy | Detect weak spots |
| Double-Coding | Two humans independently label | Estimate achievable accuracy |

**Python Code Example:**
```python
# Pipeline: Text classification with calibrated confidence and abstention
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

class QualityControlledClassifier:
    def __init__(self, model, abstain_threshold=0.7):
        self.model = model
        self.threshold = abstain_threshold

    def predict_with_quality(self, texts):
        probs = self.model.predict_proba(texts)
        results = []
        for i, text in enumerate(texts):
            pred_class = np.argmax(probs[i])
            confidence = probs[i][pred_class]
            margin = confidence - np.sort(probs[i])[-2] if len(probs[i]) > 1 else confidence
            results.append({
                'text': text[:100],
                'prediction': pred_class,
                'confidence': float(confidence),
                'margin': float(margin),
                'status': 'accepted' if confidence >= self.threshold else 'needs_review'
            })
        return results

    def quality_report(self, results):
        accepted = [r for r in results if r['status'] == 'accepted']
        review = [r for r in results if r['status'] == 'needs_review']
        return {
            'total': len(results),
            'accepted': len(accepted),
            'needs_review': len(review),
            'avg_confidence': np.mean([r['confidence'] for r in results]),
            'abstain_rate': len(review) / len(results),
        }
```

**Interview Tips:**
- Emphasize that raw softmax probabilities are poorly calibrated and need post-hoc calibration
- Mention temperature scaling as the simplest effective calibration method
- Discuss that abstention (reject option) trades coverage for precision
- Note conformal prediction as a modern approach providing statistical coverage guarantees

---

## Question 12
- [ ] Done
**What approaches help with text classification robustness against adversarial attacks?**
**Answer:**

Adversarial attacks manipulate text to fool classifiers while preserving human-perceived meaning. Examples include synonym substitution, character perturbations, and paraphrase attacks. Robust models must resist such manipulation.

**Core Concepts:**

| Attack Type | Description | Defense |
|---|---|---|
| Character-Level | Typos, homoglyphs, unicode tricks | Spell check, char normalization |
| Word-Level | Synonym substitution (TextFooler) | Adversarial training, certified robustness |
| Sentence-Level | Paraphrase, style transfer | Semantic similarity-based input validation |
| Universal Triggers | Short text prepended to any input | Input filtering |
| Backdoor | Poisoned training data | Data validation, cleanlab |

**Python Code Example:**
```python
# Pipeline: Adversarial robustness for text classification
import re
import numpy as np

class RobustTextClassifier:
    def __init__(self, base_model, defense='ensemble'):
        self.model = base_model
        self.defense = defense

    def normalize_input(self, text):
        """Basic adversarial input normalization."""
        # Fix common homoglyph attacks
        homoglyphs = {'\u0430': 'a', '\u0435': 'e', '\u043e': 'o'}  # Cyrillic
        for fake, real in homoglyphs.items():
            text = text.replace(fake, real)
        # Normalize whitespace and invisible characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # Zero-width
        return text

    def adversarial_augmentation(self, text, n_augments=3):
        """Generate adversarial-like augmentations for training."""
        augmented = [text]
        words = text.split()
        for _ in range(n_augments):
            idx = np.random.randint(len(words))
            # Random character swap within word
            word = list(words[idx])
            if len(word) > 2:
                i = np.random.randint(1, len(word) - 1)
                word[i], word[i-1] = word[i-1], word[i]
            modified = words.copy()
            modified[idx] = ''.join(word)
            augmented.append(' '.join(modified))
        return augmented

    def predict_robust(self, text):
        text = self.normalize_input(text)
        return self.model.predict(text)
```

**Interview Tips:**
- Mention TextFooler and TextAttack as standard adversarial attack/defense benchmarks
- Discuss that adversarial training (training on adversarial examples) is the most effective defense
- Highlight input normalization (homoglyphs, zero-width chars) as essential first defense
- Note that certified robustness (randomized smoothing) provides mathematical guarantees

---

## Question 13
- [ ] Done
**How do you implement knowledge distillation for compressing text classification models?**
**Answer:**

Knowledge distillation compresses a large teacher (BERT-large, ensemble) into a smaller, faster student (DistilBERT, BiLSTM) for production deployment, maintaining 95-99% of teacher accuracy at 2-10x speed.

**Core Concepts:**

| Aspect | Teacher | Student |
|---|---|---|
| Architecture | BERT-large (340M params) | DistilBERT (66M) or BiLSTM |
| Speed | ~5 ms/sample | ~0.5-1 ms/sample |
| Accuracy | 98% (benchmark) | 96-97% |
| Training Signal | Ground truth labels | Soft teacher + hard labels |
| Temperature | N/A | T=3-5 for soft labels |
| Data | Original training data | Original + teacher-labeled unlabeled |

**Python Code Example:**
```python
# Pipeline: Knowledge distillation for text classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassDistiller:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher = teacher.eval()
        self.student = student
        self.T = temperature
        self.alpha = alpha

    def distill_loss(self, student_logits, teacher_logits, labels):
        # Soft loss: learn from teacher's soft predictions
        soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
        soft_pred = F.log_softmax(student_logits / self.T, dim=-1)
        soft_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (self.T ** 2)
        # Hard loss: learn from true labels
        hard_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_step(self, batch):
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        student_logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits
        return self.distill_loss(student_logits, teacher_logits, labels)
```

**Interview Tips:**
- Mention that DistilBERT achieves 97% of BERT performance at 60% size and 2x speed
- Discuss that teacher-labeling unlabeled data creates more training signal for the student
- Highlight temperature T=4 as a good starting point for classification distillation
- Note that task-specific distillation outperforms general distillation (DistilBERT)

---

## Question 14
- [ ] Done
**What techniques work best for text classification with computational efficiency constraints?**
**Answer:**

Efficient text classification serves mobile apps, embedded systems, and high-throughput servers where latency and compute budgets are strict. Model choice spans from classical ML to optimized neural models.

**Core Concepts:**

| Model | Speed | Accuracy | Memory |
|---|---|---|---|
| TF-IDF + LogReg | ~100K docs/sec | 85-90% | <100MB |
| FastText | ~500K docs/sec | 88-92% | <100MB |
| CNN TextClassifier | ~50K docs/sec | 90-93% | ~200MB |
| DistilBERT (quantized) | ~5K docs/sec | 95-97% | ~100MB |
| TinyBERT | ~10K docs/sec | 94-96% | ~50MB |
| ONNX-Optimized BERT | ~10K docs/sec | 96-97% | ~400MB |

**Python Code Example:**
```python
# Pipeline: Efficient text classification from lightweight to optimized
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# Ultra-fast: TF-IDF + Linear (CPU, <1ms per doc)
def fast_classifier(train_texts, train_labels):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ('clf', SGDClassifier(loss='hinge', max_iter=100)),
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline

# FastText: Very fast and accurate
def fasttext_classifier(train_file, num_labels):
    import fasttext
    model = fasttext.train_supervised(
        input=train_file,
        epoch=25, lr=1.0, wordNgrams=2, dim=100,
        loss='softmax' if num_labels < 10 else 'hs'  # Hierarchical softmax for many labels
    )
    return model

# ONNX optimization for BERT
def optimize_bert_onnx(model_path, output_path):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    model.save_pretrained(output_path)
    return model
```

**Interview Tips:**
- Recommend TF-IDF + LogReg as the baseline; it's surprisingly competitive for many tasks
- Mention FastText as the best balance of speed and accuracy for large-scale classification
- Discuss ONNX Runtime + quantization for making BERT production-viable
- Note that accuracy gains from BERT over FastText are often 2-5%—not always worth the cost

---

## Question 15
- [ ] Done
**How do you handle text classification for streaming or real-time text processing?**
**Answer:**

Streaming text classification processes incoming documents (tweets, emails, messages, news) in real-time with strict latency requirements, often as part of event processing pipelines.

**Core Concepts:**

| Component | Description | Technology |
|---|---|---|
| Stream Ingestion | Consume real-time text feed | Kafka, Kinesis, Pub/Sub |
| Model Serving | Low-latency inference endpoint | TorchServe, Triton, ONNX Runtime |
| Micro-Batching | Small batches for throughput | 10-100ms windows |
| Result Caching | Cache frequent classification results | Redis, in-memory LRU |
| Model Hot-Reload | Update model without downtime | A/B testing, canary deploy |
| Backpressure | Handle traffic spikes gracefully | Queue-based buffering |

**Python Code Example:**
```python
# Pipeline: Streaming text classification with async processing
import asyncio
from collections import deque
import time

class StreamingTextClassifier:
    def __init__(self, model, batch_size=32, max_latency_ms=100):
        self.model = model
        self.batch_size = batch_size
        self.max_latency = max_latency_ms / 1000
        self.buffer = deque()
        self.results = {}

    async def process_stream(self, text_stream):
        batch = []
        batch_start = time.time()

        async for text_id, text in text_stream:
            batch.append((text_id, text))

            # Flush when batch is full or max latency reached
            if len(batch) >= self.batch_size or (time.time() - batch_start) > self.max_latency:
                await self.classify_batch(batch)
                batch = []
                batch_start = time.time()

        if batch:  # Flush remaining
            await self.classify_batch(batch)

    async def classify_batch(self, batch):
        ids, texts = zip(*batch)
        predictions = self.model.predict(list(texts))
        for text_id, pred in zip(ids, predictions):
            self.results[text_id] = {'prediction': pred, 'timestamp': time.time()}
```

**Interview Tips:**
- Recommend micro-batching (10-100ms windows) for throughput without sacrificing latency
- Mention that model serving frameworks (Triton, TorchServe) handle batching automatically
- Discuss backpressure strategies for handling traffic spikes without dropping messages
- Note that result caching is highly effective for repetitive text streams (e.g., templated emails)

---

## Question 16
- [ ] Done
**What strategies help with text classification consistency across different text formats?**
**Answer:**

Text arrives in diverse formats (plain text, HTML, PDF, Word, email headers, JSON) and consistent classification requires format-aware preprocessing to extract clean text before classification.

**Core Concepts:**

| Format | Extraction Challenge | Tool |
|---|---|---|
| HTML | Boilerplate, navigation, ads | BeautifulSoup, readability |
| PDF | Layout, tables, images | PyPDF2, pdfplumber |
| Email | Headers, signatures, quotes | email.parser |
| Word/DOCX | Formatting, embedded objects | python-docx |
| JSON/XML | Structured data extraction | json, lxml |
| OCR Text | Recognition errors, layout | Tesseract, layout analysis |

**Python Code Example:**
```python
# Pipeline: Format-agnostic text classification
import re
from pathlib import Path

class FormatAgnosticClassifier:
    def __init__(self, classifier):
        self.clf = classifier
        self.extractors = {
            '.html': self.extract_html,
            '.txt': self.extract_text,
            '.pdf': self.extract_pdf,
            '.eml': self.extract_email,
        }

    def extract_html(self, content):
        from html import unescape
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        return unescape(text).strip()

    def extract_text(self, content):
        return content.strip()

    def extract_pdf(self, content):
        # Placeholder - would use pdfplumber in production
        return content

    def extract_email(self, content):
        import email
        msg = email.message_from_string(content)
        body = msg.get_payload(decode=True)
        return body.decode('utf-8', errors='replace') if body else ''

    def classify(self, content, format_type='.txt'):
        extractor = self.extractors.get(format_type, self.extract_text)
        clean_text = extractor(content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return self.clf.predict(clean_text)
```

**Interview Tips:**
- Emphasize that format normalization is a critical preprocessing step often overlooked
- Mention that boilerplate removal (readability-lxml) is essential for HTML classification
- Discuss that PDF extraction quality varies wildly; test multiple tools
- Note that a format-detection step (magic bytes, extensions) should route to appropriate extractors

---

## Question 17
- [ ] Done
**How do you implement online learning for text classifiers adapting to new categories?**
**Answer:**

Online learning enables text classifiers to incorporate new categories and adapt to evolving data without full retraining. This is critical for systems where categories expand over time (e.g., support ticket routing, content moderation).

**Core Concepts:**

| Approach | Description | New Classes? |
|---|---|---|
| Incremental Fine-Tuning | Continue training with new category data | Yes |
| Elastic Weight Consolidation | Protect important weights while learning new | Yes, prevents forgetting |
| Expandable Output Layer | Add new output neurons for new classes | Yes |
| Prototype Networks | New class from few examples | Yes, zero/few-shot |
| Replay Buffer | Mix old + new data during updates | Prevents forgetting |
| Class-Incremental Learning | Systematic framework for adding classes | Yes |

**Python Code Example:**
```python
# Pipeline: Online text classifier with expandable categories
import torch
import torch.nn as nn
import numpy as np

class ExpandableTextClassifier(nn.Module):
    def __init__(self, encoder, initial_classes):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.config.hidden_size
        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(initial_classes)
        ])
        self.class_names = list(range(initial_classes))
        self.replay_buffer = []  # Store examples for replay

    def add_class(self, class_name, examples):
        """Add a new class with initial examples."""
        hidden_dim = self.encoder.config.hidden_size
        new_head = nn.Linear(hidden_dim, 1)
        self.class_heads.append(new_head)
        self.class_names.append(class_name)
        # Add examples to replay buffer
        for text, label in examples:
            self.replay_buffer.append((text, len(self.class_names) - 1))

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = features.last_hidden_state[:, 0]
        logits = torch.cat([head(cls_repr) for head in self.class_heads], dim=-1)
        return logits

    def incremental_train(self, new_data, epochs=3, replay_ratio=0.3):
        # Mix new data with replay buffer
        replay_size = int(len(new_data) * replay_ratio)
        if self.replay_buffer:
            replay = [self.replay_buffer[i] for i in
                      np.random.choice(len(self.replay_buffer), min(replay_size, len(self.replay_buffer)), replace=False)]
            training_data = new_data + replay
        else:
            training_data = new_data
        return training_data  # Pass to trainer
```

**Interview Tips:**
- Mention that expandable output layers avoid catastrophic forgetting of old classes
- Discuss replay buffers as the simplest effective anti-forgetting strategy
- Highlight that prototype/metric learning enables adding new classes from few examples
- Note that EWC (Elastic Weight Consolidation) is the standard continual learning method

---

## Question 18
- [ ] Done
**What approaches work best for text classification in conversational or dialogue contexts?**
**Answer:**

Dialogue classification assigns intents, topics, dialogue acts, or sentiment to conversation turns. The key challenge is that each utterance's meaning depends on the conversation history.

**Core Concepts:**

| Task | Description | Example |
|---|---|---|
| Intent Detection | Classify user intent in chatbot | "Book flight" → BOOKING intent |
| Dialogue Act | Classify function of utterance | "Sure, that works" → ACCEPT act |
| Topic Tracking | Track conversation topic over turns | Detect topic switches |
| Sentiment Tracking | Sentiment across conversation | Detect escalation |
| Emotion Detection | Classify emotion per turn | Joy, anger, frustration |

**Python Code Example:**
```python
# Pipeline: Context-aware dialogue text classification
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DialogueClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_intents=20):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        # Context encoder for conversation history
        self.context_lstm = nn.LSTM(hidden, hidden // 2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden * 2, num_intents)  # Current + context

    def forward(self, current_input_ids, current_mask, history_reprs=None):
        # Encode current utterance
        current = self.encoder(input_ids=current_input_ids, attention_mask=current_mask)
        current_repr = current.last_hidden_state[:, 0]

        if history_reprs is not None:
            # Encode conversation history
            context_out, _ = self.context_lstm(history_reprs)
            context_repr = context_out[:, -1]  # Last hidden state
            combined = torch.cat([current_repr, context_repr], dim=-1)
        else:
            combined = torch.cat([current_repr, torch.zeros_like(current_repr)], dim=-1)

        return self.classifier(combined)
```

**Interview Tips:**
- Emphasize that conversation history (2-5 previous turns) significantly improves intent detection
- Mention the ATIS and SNIPS datasets as standard intent classification benchmarks
- Discuss that joint intent+slot filling models (e.g., JointBERT) are the standard for task-oriented dialogue
- Note that dialogue act classification benefits from both text content and position in conversation

---

## Question 19
- [ ] Done
**How do you handle text classification optimization for specific downstream applications?**
**Answer:**

Different applications have different requirements: spam filters need high recall, medical diagnosis needs high precision, content recommendation needs balanced F1. Optimizing for the right metric is critical.

**Core Concepts:**

| Application | Priority Metric | Threshold Strategy |
|---|---|---|
| Spam Detection | High recall (catch all spam) | Low threshold (aggressive) |
| Medical Diagnosis | High precision (no false alarms) | High threshold (conservative) |
| Content Moderation | High recall (safety-critical) | Low threshold + human review |
| Recommendation | Balanced F1 | Default threshold |
| Legal Document Review | High recall (completeness) | Low threshold, accept more |
| Fraud Detection | High precision (minimize false positives) | High threshold |

**Python Code Example:**
```python
# Pipeline: Application-optimized text classification
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

class ApplicationOptimizedClassifier:
    def __init__(self, model, optimization_target='f1'):
        self.model = model
        self.target = optimization_target
        self.threshold = 0.5  # Default

    def optimize_threshold(self, val_texts, val_labels):
        """Find optimal threshold for target metric."""
        probs = self.model.predict_proba(val_texts)[:, 1]  # Binary case
        precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)

        if self.target == 'f1':
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1s)
        elif self.target == 'recall_90':
            # Find threshold giving 90% recall
            best_idx = np.argmax(recalls >= 0.9)
        elif self.target == 'precision_95':
            # Find threshold giving 95% precision
            valid = precisions >= 0.95
            best_idx = np.argmax(recalls[valid]) if valid.any() else 0
        else:
            best_idx = len(thresholds) // 2

        self.threshold = thresholds[min(best_idx, len(thresholds) - 1)]
        return {
            'threshold': self.threshold,
            'precision': precisions[best_idx],
            'recall': recalls[best_idx],
        }

    def predict(self, texts):
        probs = self.model.predict_proba(texts)[:, 1]
        return (probs >= self.threshold).astype(int)
```

**Interview Tips:**
- Emphasize that threshold tuning on validation data is the cheapest way to optimize for a specific metric
- Mention that different operating points on the precision-recall curve serve different applications
- Discuss cost-sensitive learning as an alternative to threshold tuning
- Note that the default 0.5 threshold is almost never optimal for real applications

---

## Question 20
- [ ] Done
**What techniques help with text classification for texts requiring cultural context?**
**Answer:**

Cultural context affects text interpretation: sarcasm, humor, idioms, taboo topics, and sentiment polarity vary across cultures. A classifier trained on Western data may misclassify texts from other cultural contexts.

**Core Concepts:**

| Cultural Factor | Impact | Example |
|---|---|---|
| Sarcasm/Irony | Culture-specific expression | British vs. American sarcasm |
| Idioms | Non-literal meaning varies by culture | "Break a leg" (positive in Western) |
| Sentiment Polarity | Same topic, different valence | Food reviews vary by cuisine culture |
| Taboo/Sensitivity | Culture-specific offensive topics | Political topics vary by region |
| Formality Norms | Expected register varies | Japanese keigo vs. casual |
| Humor | Culture-specific comedic patterns | Puns, wordplay, references |

**Python Code Example:**
```python
# Pipeline: Culture-aware text classification
class CultureAwareClassifier:
    def __init__(self, base_model, culture_adaptors=None):
        self.base = base_model
        self.culture_models = culture_adaptors or {}
        self.culture_lexicons = {}

    def add_culture_lexicon(self, culture, sentiment_overrides):
        """Add culture-specific sentiment/meaning overrides."""
        self.culture_lexicons[culture] = sentiment_overrides

    def detect_cultural_markers(self, text):
        """Detect potential culture-specific elements."""
        markers = {
            'idioms': self.find_idioms(text),
            'sarcasm_indicators': self.detect_sarcasm_cues(text),
            'formality_level': self.assess_formality(text),
        }
        return markers

    def find_idioms(self, text):
        common_idioms = [
            'break a leg', 'piece of cake', 'cost an arm and a leg',
            'hit the nail on the head', 'under the weather'
        ]
        found = [idiom for idiom in common_idioms if idiom in text.lower()]
        return found

    def detect_sarcasm_cues(self, text):
        cues = ['/s', 'NOT', 'totally', 'obviously', 'sure...']
        return any(cue in text for cue in cues)

    def assess_formality(self, text):
        informal_count = sum(1 for w in text.split() if w.lower() in {'lol', 'bruh', 'ngl', 'tbh'})
        return 'informal' if informal_count > 0 else 'formal'

    def classify(self, text, culture='default'):
        if culture in self.culture_models:
            return self.culture_models[culture].predict(text)
        return self.base.predict(text)
```

**Interview Tips:**
- Mention that cultural adaptation requires region-specific training data, not just translation
- Discuss that sarcasm detection accuracy drops 10-20% across cultures
- Highlight that sentiment lexicons are NOT culturally universal (e.g., "spicy" = positive vs. negative)
- Note that cultural bias in training data is a major source of classification errors

---

## Question 21
- [ ] Done
**How do you implement fairness-aware text classification to reduce bias across groups?**
**Answer:**

Bias in text classification can lead to discriminatory outcomes: toxicity detectors flag African American English more, hiring tools penalize female applicants, and sentiment classifiers rate certain demographics differently.

**Core Concepts:**

| Bias Type | Description | Example |
|---|---|---|
| Label Bias | Training labels reflect societal bias | "Professional" associated with male names |
| Representation Bias | Underrepresented groups in training data | Non-English dialects underrepresented |
| Feature Bias | Protected attributes correlate with features | Names, zip codes as proxy for race |
| Association Bias | Stereotypical word associations | "Nurse" associated with female |

| Mitigation | Stage | Method |
|---|---|---|
| Data Augmentation | Pre-processing | Balance groups in training data |
| Debiased Embeddings | Pre-processing | Remove protected attribute directions |
| Adversarial Debiasing | In-processing | Adversary predicts protected attribute |
| Threshold Adjustment | Post-processing | Different thresholds per group |
| Constraint Optimization | In-processing | Equalized odds constraint |

**Python Code Example:**
```python
# Pipeline: Fairness-aware text classification with bias detection
import numpy as np
from collections import defaultdict

class FairTextClassifier:
    def __init__(self, model):
        self.model = model

    def audit_bias(self, texts, labels, group_labels):
        """Audit classification bias across demographic groups."""
        predictions = self.model.predict(texts)
        group_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})

        for pred, true, group in zip(predictions, labels, group_labels):
            m = group_metrics[group]
            if pred == 1 and true == 1: m['tp'] += 1
            elif pred == 1 and true == 0: m['fp'] += 1
            elif pred == 0 and true == 1: m['fn'] += 1
            else: m['tn'] += 1

        results = {}
        for group, m in group_metrics.items():
            precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
            recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
            fpr = m['fp'] / (m['fp'] + m['tn']) if (m['fp'] + m['tn']) > 0 else 0
            results[group] = {'precision': precision, 'recall': recall, 'fpr': fpr}

        # Check equalized odds
        fprs = [r['fpr'] for r in results.values()]
        results['bias_gap'] = max(fprs) - min(fprs)
        return results
```

**Interview Tips:**
- Mention Perspective API controversy: higher toxicity scores for African American English
- Discuss equalized odds and demographic parity as standard fairness metrics
- Highlight that counterfactual fairness (swap demographics, same prediction) is intuitive
- Note that fairness constraints often trade off slightly with accuracy (1-3%)

---

## Question 22
- [ ] Done
**What strategies work best for text classification with hierarchical category structures?**
**Answer:**

Hierarchical classification organizes categories in a tree or DAG structure (e.g., Sports > Football > NFL, or Patent classification IPC codes). This leverages category relationships for better prediction.

**Core Concepts:**

| Strategy | Description | Benefit |
|---|---|---|
| Flat Classification | Ignore hierarchy, predict leaf nodes | Simple baseline |
| Top-Down (Local) | Classify at each level sequentially | Coarse → fine refinement |
| Global (Big-Bang) | Single model, hierarchy-aware loss | No error propagation |
| Per-Node | One classifier per internal node | Specialized decisions |
| Label Embedding | Embed categories in hierarchy-aware space | Captures label similarity |
| Hierarchical Loss | Penalize more for distant misclassifications | Better error distribution |

**Python Code Example:**
```python
# Pipeline: Hierarchical text classification
import torch
import torch.nn as nn

class HierarchicalTextClassifier(nn.Module):
    def __init__(self, encoder, hierarchy):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.level_classifiers = nn.ModuleList()
        for level_size in hierarchy:  # [5, 20, 100] = 3 levels
            self.level_classifiers.append(nn.Linear(hidden, level_size))

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = features.last_hidden_state[:, 0]
        level_logits = [clf(cls_repr) for clf in self.level_classifiers]
        return level_logits

    def hierarchical_loss(self, level_logits, level_labels, weights=[1.0, 1.5, 2.0]):
        """Weight deeper levels more heavily."""
        total_loss = 0
        for logits, labels, weight in zip(level_logits, level_labels, weights):
            total_loss += weight * nn.functional.cross_entropy(logits, labels)
        return total_loss

    def predict_hierarchical(self, input_ids, attention_mask):
        level_logits = self.forward(input_ids, attention_mask)
        predictions = [logits.argmax(dim=-1) for logits in level_logits]
        return predictions  # [coarse_pred, mid_pred, fine_pred]
```

**Interview Tips:**
- Recommend top-down classification for deep hierarchies (4+ levels)
- Mention that global approaches avoid error propagation between hierarchy levels
- Discuss hierarchical loss that penalizes "Animals > Dog" mislabeled as "Sports" more than "Animals > Cat"
- Note that patent classification (IPC/CPC) and library categorization (Dewey) are classic applications

---

## Question 23
- [ ] Done
**How do you handle text classification quality assessment with subjective categories?**
**Answer:**

Subjective categories (sentiment, offensiveness, helpfulness, quality) lack clear ground truth because different annotators may legitimately disagree. Quality assessment must account for inherent subjectivity.

**Core Concepts:**

| Aspect | Method | Purpose |
|---|---|---|
| Inter-Annotator Agreement | Cohen's/Fleiss' Kappa | Measure label reliability |
| Multi-Annotator Labels | Keep all annotations, don't force consensus | Preserve disagreement information |
| Soft Labels | Use annotation distribution as training target | Model inherent uncertainty |
| Annotator Modeling | Model individual annotator tendencies | Separate signal from noise |
| Calibrated Evaluation | Compare to human agreement ceiling | Realistic accuracy expectations |
| Perplexity-Based | Compare model uncertainty to human uncertainty | Alignment metric |

**Python Code Example:**
```python
# Pipeline: Handling subjective classification with multi-annotator labels
import numpy as np
from collections import Counter

class SubjectiveClassifier:
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

    def create_soft_labels(self, annotations_per_item):
        """Convert multiple annotations into soft probability labels."""
        soft_labels = []
        for annotations in annotations_per_item:
            counts = Counter(annotations)
            total = sum(counts.values())
            label_dist = np.zeros(self.num_classes)
            for cls, count in counts.items():
                label_dist[cls] = count / total
            soft_labels.append(label_dist)
        return np.array(soft_labels)

    def compute_human_ceiling(self, annotations_per_item):
        """Compute inter-annotator agreement as accuracy ceiling."""
        agreements = []
        for annotations in annotations_per_item:
            majority = Counter(annotations).most_common(1)[0][0]
            agree_rate = sum(1 for a in annotations if a == majority) / len(annotations)
            agreements.append(agree_rate)
        return np.mean(agreements)

    def evaluate_against_human(self, predictions, annotations_per_item):
        """Compare model to human consensus and disagreement."""
        human_ceiling = self.compute_human_ceiling(annotations_per_item)
        majority_labels = [Counter(a).most_common(1)[0][0] for a in annotations_per_item]
        model_accuracy = np.mean([p == m for p, m in zip(predictions, majority_labels)])
        return {
            'model_accuracy': model_accuracy,
            'human_ceiling': human_ceiling,
            'relative_performance': model_accuracy / human_ceiling,
        }
```

**Interview Tips:**
- Emphasize that model accuracy cannot exceed inter-annotator agreement for subjective tasks
- Mention that soft labels (probability distributions) from multiple annotators improve model calibration
- Discuss that Cohen's Kappa of 0.4-0.6 is typical for subjective NLP tasks like sentiment
- Note that annotator modeling (learning individual biases) can improve gold label quality

---

## Question 24
- [ ] Done
**What approaches help with text classification for texts in multiple languages?**
**Answer:**

Multilingual text classification aims to classify text across multiple languages using shared or transferable models, enabling zero-shot cross-lingual transfer from high-resource to low-resource languages.

**Core Concepts:**

| Approach | Description | Languages |
|---|---|---|
| Multilingual BERT (mBERT) | Pretrained on 104 languages | 104 |
| XLM-RoBERTa | Cross-lingual language model | 100 |
| Translate-Train | Translate training data, train monolingual | Any with MT |
| Translate-Test | Translate test data to English, classify | Any with MT |
| LaBSE | Language-agnostic sentence embeddings | 109 |
| Adapter-Based | Language-specific adapter modules | Extensible |

**Python Code Example:**
```python
# Pipeline: Multilingual text classification with XLM-RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MultilingualClassifier:
    def __init__(self, num_labels=5):
        model_name = "xlm-roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def predict(self, texts, language=None):
        """Classify text in any supported language."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=-1).tolist()

    def zero_shot_transfer(self, en_train_data, target_texts):
        """Train on English, classify in any language."""
        self.fine_tune(en_train_data)
        return self.predict(target_texts)

    def fine_tune(self, data):
        pass  # Training loop placeholder
```

**Interview Tips:**
- XLM-RoBERTa outperforms mBERT for cross-lingual classification
- Zero-shot cross-lingual transfer works within 5% of supervised performance
- Translate-train often outperforms zero-shot for languages with good MT
- Subword tokenization (SentencePiece) is key for handling diverse scripts

---

## Question 25
- [ ] Done
**How do you implement privacy-preserving text classification for sensitive documents?**
**Answer:**

Text classification on sensitive data (medical records, legal documents, personal messages) requires privacy-preserving techniques that protect individual information while maintaining classification accuracy.

**Core Concepts:**

| Technique | Description | Privacy Guarantee |
|---|---|---|
| Differential Privacy | Add calibrated noise to gradients | Mathematical (ε-DP) |
| Federated Learning | Train without centralizing data | Data stays on-device |
| PII Redaction | Remove personally identifiable info | Best-effort |
| Secure Aggregation | Encrypt model updates | Cryptographic |
| K-Anonymity | Generalize quasi-identifiers | Anonymization |
| Synthetic Data | Generate privacy-safe training data | Approximate |

**Python Code Example:**
```python
# Pipeline: Privacy-preserving text classification
import re
import hashlib

class PrivacyAwareClassifier:
    def __init__(self, model):
        self.model = model
        self.pii_patterns = {
            'email': r'[\w.+-]+@[\w-]+\.[\w.-]+',
            'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        }

    def redact_pii(self, text):
        redacted = text
        for pii_type, pattern in self.pii_patterns.items():
            redacted = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', redacted)
        return redacted

    def pseudonymize(self, text):
        pseudonymized = text
        for pii_type, pattern in self.pii_patterns.items():
            for match in set(re.findall(pattern, pseudonymized)):
                pseudo = hashlib.sha256(match.encode()).hexdigest()[:8]
                pseudonymized = pseudonymized.replace(match, f'[{pii_type}_{pseudo}]')
        return pseudonymized

    def classify_private(self, text, method='redact'):
        safe_text = self.redact_pii(text) if method == 'redact' else self.pseudonymize(text)
        return self.model.predict(safe_text)
```

**Interview Tips:**
- PII redaction is the minimum baseline for sensitive text classification
- Differential privacy (DP-SGD) provides mathematical guarantees
- Federated learning keeps data on-device but has gradient leakage risks
- GDPR and HIPAA compliance require documented privacy protections

---

## Question 26
- [ ] Done
**What techniques work best for text classification with fine-grained category distinctions?**
**Answer:**

Fine-grained classification uses detailed category taxonomies (e.g., 3-class → 50-class sentiment). The challenge is subtle distinctions between classes and smaller per-class training samples.

**Core Concepts:**

| Method | Description | Example |
|---|---|---|
| Aspect-Level | Classify attributes separately | Food:positive, Service:negative |
| Multi-Label | Text belongs to multiple categories | Sports + Technology article |
| Ordinal Classification | Ordered categories | 1-star to 5-star |
| Hierarchical | Tree-structured labels | Sports > Football > Transfers |
| Prototype Networks | Learn from few examples per class | Class prototype matching |
| Data Augmentation | Increase minority class samples | Back-translation, paraphrase |

**Python Code Example:**
```python
# Pipeline: Fine-grained multi-aspect text classification
import torch
import torch.nn as nn
from transformers import AutoModel

class FineGrainedClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", aspects=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.aspects = aspects or {'topic': 20, 'sentiment': 5, 'stance': 3}
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(hidden // 2, nc)
            ) for name, nc in self.aspects.items()
        })

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = features.last_hidden_state[:, 0]
        return {name: head(cls_repr) for name, head in self.heads.items()}

    def predict_all_aspects(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        return {name: logits.argmax(dim=-1) for name, logits in outputs.items()}
```

**Interview Tips:**
- Multi-task learning (shared encoder + separate heads) is standard for fine-grained tasks
- Class imbalance worsens with finer granularity — use focal loss or oversampling
- Ordinal regression penalizes proportional to rank distance for ordered categories
- ABSA (Aspect-Based Sentiment Analysis) is the flagship fine-grained task

---

## Question 27
- [ ] Done
**How do you handle text classification adaptation to emerging topics or categories?**
**Answer:**

Emerging topics (new diseases like COVID-19, new products, breaking events) appear suddenly without labeled historical data. Rapid classifier development requires few-shot and zero-shot approaches.

**Core Concepts:**

| Approach | Data Needed | Speed |
|---|---|---|
| Zero-Shot Classification | Only label descriptions | Immediate |
| Few-Shot (Prompt-Based) | 5-20 examples | Minutes |
| Active Learning | Iteratively label most informative | Hours |
| Data Augmentation | Expand few examples | Minutes |
| Weak Supervision (Snorkel) | Labeling functions, no manual labels | Hours |
| Transfer from Related Topics | Related labeled data | Hours |

**Python Code Example:**
```python
# Pipeline: Emerging topic classification with zero-shot
from transformers import pipeline

class EmergingTopicClassifier:
    def __init__(self):
        self.zero_shot = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli")

    def classify_zero_shot(self, text, candidate_labels):
        result = self.zero_shot(text, candidate_labels=candidate_labels)
        return {
            'label': result['labels'][0],
            'score': result['scores'][0],
            'all_scores': dict(zip(result['labels'], result['scores']))
        }

    def create_weak_labels(self, texts, keyword_rules):
        labels = []
        for text in texts:
            text_lower = text.lower()
            matched = [label for label, kws in keyword_rules.items()
                      if any(kw in text_lower for kw in kws)]
            labels.append(matched[0] if len(matched) == 1 else 'UNKNOWN')
        return labels

# Usage
clf = EmergingTopicClassifier()
result = clf.classify_zero_shot(
    "The vaccine rollout has been accelerated in rural areas",
    candidate_labels=["healthcare", "politics", "economy", "technology"]
)
```

**Interview Tips:**
- Zero-shot classification with BART/MNLI is the fastest way to classify emerging topics
- Active learning efficiently builds labeled datasets with minimal human effort
- Weak supervision via labeling functions can bootstrap classifiers in hours
- Prompt engineering with GPT-style models enables rapid few-shot classification

---

## Question 28
- [ ] Done
**What strategies help with text classification for texts requiring temporal context?**
**Answer:**

Temporal context affects text meaning: "Tweet" meant bird sounds before 2006; "corona" referred to a beer before 2020. Language drift causes classifiers to degrade without temporal adaptation.

**Core Concepts:**

| Drift Type | Description | Example |
|---|---|---|
| Concept Drift | Label meaning changes | "Viral" = disease → social media |
| Vocabulary Drift | New words/slang emerge | "Yeet", "based", "mid" |
| Distribution Drift | Topic frequency changes | COVID spike in health articles |
| Annotation Drift | Labeling standards evolve | Toxicity norms change |

| Strategy | Description |
|---|---|
| Periodic Retraining | Retrain on recent data windows |
| Exponential Decay | Weight recent data more heavily |
| Drift Detection | Monitor performance, trigger retraining |
| Temporal Features | Add time-based features |
| Continual Pretraining | Update LM on new text |

**Python Code Example:**
```python
# Pipeline: Temporal drift detection and adaptive classification
import numpy as np
from collections import deque

class TemporalAdaptiveClassifier:
    def __init__(self, model, window_size=1000, drift_threshold=0.05):
        self.model = model
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.perf_window = deque(maxlen=window_size)
        self.baseline_acc = None

    def detect_drift(self):
        if len(self.perf_window) < self.window_size:
            return False
        current_acc = np.mean(list(self.perf_window))
        if self.baseline_acc is None:
            self.baseline_acc = current_acc
            return False
        return (self.baseline_acc - current_acc) > self.drift_threshold

    def classify_with_monitoring(self, text, true_label=None):
        prediction = self.model.predict(text)
        if true_label is not None:
            self.perf_window.append(int(prediction == true_label))
        if self.detect_drift():
            self.baseline_acc = None
            self.perf_window.clear()
        return prediction
```

**Interview Tips:**
- Temporal drift affects all deployed NLP models — it's when, not if
- Continual pretraining on new text handles vocabulary drift
- Use Page-Hinkley or ADWIN algorithms for drift detection
- Retraining windows should overlap to avoid distribution gaps

---

## Question 29
- [ ] Done
**How do you implement robust error handling for text classification in production?**
**Answer:**

Production text classification pipelines face input anomalies: encoding issues, extreme lengths, adversarial inputs, empty content, and system failures. Robust error handling ensures graceful degradation.

**Core Concepts:**

| Error Type | Cause | Strategy |
|---|---|---|
| Encoding Errors | Non-UTF8 text | Detect + normalize |
| Empty Input | Missing text field | Return default/flag |
| Extreme Length | Very long documents | Truncation + sliding window |
| OOV Heavy | Many unknown tokens | Fallback to char model |
| Adversarial Input | Intentionally confusing | Detect and flag |
| Model Error | CUDA OOM, timeout | Fallback to lighter model |
| Low Confidence | Uncertain prediction | Route to human review |

**Python Code Example:**
```python
# Pipeline: Robust text classification with error handling
import logging

logger = logging.getLogger(__name__)

class RobustClassifier:
    def __init__(self, primary_model, fallback_model=None, conf_threshold=0.3):
        self.primary = primary_model
        self.fallback = fallback_model
        self.conf_threshold = conf_threshold

    def preprocess_safe(self, text):
        if not text or not text.strip():
            return None, 'empty_input'
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        text = text.replace('\x00', '').strip()
        if len(text.split()) < 2:
            return None, 'too_short'
        if len(text) > 50000:
            text = text[:50000]
            logger.warning("Input truncated to 50K chars")
        return text, 'ok'

    def classify(self, text):
        text, status = self.preprocess_safe(text)
        if status != 'ok':
            return {'label': 'UNKNOWN', 'confidence': 0.0, 'error': status}
        try:
            result = self.primary.predict_with_confidence(text)
        except Exception as e:
            logger.error(f"Primary model error: {e}")
            if self.fallback:
                result = self.fallback.predict_with_confidence(text)
            else:
                return {'label': 'ERROR', 'confidence': 0.0, 'error': str(e)}
        if result['confidence'] < self.conf_threshold:
            result['needs_review'] = True
        return result
```

**Interview Tips:**
- Production systems need graceful degradation, not hard failures
- Fallback models (lighter, simpler) ensure reliability
- Confidence thresholds route uncertain predictions to human review
- Log every error and edge case for continuous improvement

---

## Question 30
- [ ] Done
**What approaches work best for combining text classification with other NLP tasks?**
**Answer:**

Multi-task learning (MTL) jointly trains text classification with related tasks (NER, POS tagging, summarization), sharing representations to improve all tasks via inductive transfer.

**Core Concepts:**

| Combination | Tasks | Benefit |
|---|---|---|
| Intent + Slot Filling | Classification + NER | Chatbot understanding |
| Sentiment + Aspect | Classification + Extraction | Opinion mining |
| Topic + Summary | Classification + Generation | News processing |
| Spam + Phishing Entity | Classification + NER | Email security |
| Language ID + Classification | Detection + Classification | Multilingual pipelines |

| MTL Architecture | Description |
|---|---|
| Hard Parameter Sharing | Shared encoder, task-specific heads |
| Soft Parameter Sharing | Separate models, regularized similarity |
| Cross-Stitch Networks | Learned combination of task features |
| Auxiliary Tasks | Secondary tasks aid primary classification |

**Python Code Example:**
```python
# Pipeline: Multi-task text classification + NER
import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskNLPModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_cls=10, n_ner=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.cls_head = nn.Linear(hidden, n_cls)
        self.ner_head = nn.Linear(hidden, n_ner)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_logits = self.cls_head(out.last_hidden_state[:, 0])
        ner_logits = self.ner_head(out.last_hidden_state)
        return {'classification': cls_logits, 'ner': ner_logits}

    def compute_loss(self, outputs, cls_labels, ner_labels, w_cls=1.0, w_ner=0.5):
        cls_loss = nn.functional.cross_entropy(outputs['classification'], cls_labels)
        ner_loss = nn.functional.cross_entropy(
            outputs['ner'].view(-1, outputs['ner'].size(-1)),
            ner_labels.view(-1), ignore_index=-100
        )
        return w_cls * cls_loss + w_ner * ner_loss
```

**Interview Tips:**
- Hard parameter sharing (shared encoder) is the most common MTL approach
- Joint intent detection + slot filling is the canonical MTL success story
- Task weighting (loss balancing) is the main hyperparameter in MTL
- Auxiliary tasks like language modeling or POS tagging regularize classification

---

## Question 31
- [ ] Done
**How do you handle text classification for texts with varying lengths and structures?**
**Answer:**

Real-world text varies from single words (search queries) to multi-page documents. Length-agnostic classification requires strategies that handle both extremes effectively.

**Core Concepts:**

| Length Category | Challenge | Strategy |
|---|---|---|
| Very Short (<10 tokens) | Sparse signal | Character-level features, subword models |
| Short (10-128 tokens) | Standard range | Standard BERT/transformer |
| Medium (128-512 tokens) | Near max length | Truncation with key sentence selection |
| Long (512-4096 tokens) | Exceeds BERT limit | Longformer, BigBird |
| Very Long (4096+ tokens) | Memory constraints | Hierarchical, chunk & aggregate |

| Strategy | Description |
|---|---|
| Sliding Window | Classify overlapping chunks, aggregate |
| Hierarchical Attention | Sentence-level → document-level attention |
| Key Sentence Extraction | Select most informative sentences |
| Longformer/BigBird | Efficient transformers for long documents |
| Padding/Truncation | Standard handling for fixed-length models |

**Python Code Example:**
```python
# Pipeline: Variable-length text classification
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class VariableLengthClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=5, max_chunks=8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.max_len = 512
        self.max_chunks = max_chunks

    def classify(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_len - 2:
            return self._classify_short(text)
        else:
            return self._classify_long_chunked(tokens)

    def _classify_short(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_len)
        with torch.no_grad():
            out = self.encoder(**inputs)
        return self.classifier(out.last_hidden_state[:, 0])

    def _classify_long_chunked(self, tokens):
        chunk_size = self.max_len - 2
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)][:self.max_chunks]
        chunk_reprs = []
        for chunk in chunks:
            input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
            inputs = {'input_ids': torch.tensor([input_ids]), 'attention_mask': torch.ones(1, len(input_ids))}
            with torch.no_grad():
                out = self.encoder(**inputs)
            chunk_reprs.append(out.last_hidden_state[:, 0])
        aggregated = torch.mean(torch.stack(chunk_reprs), dim=0)
        return self.classifier(aggregated)
```

**Interview Tips:**
- Longformer uses sliding window attention for O(n) complexity on long documents
- Hierarchical approaches (HAN) are still competitive for very long documents
- Discuss that truncation strategy matters: head, tail, or head+tail often beats random
- For short texts, character-level CNNs or subword models capture limited signal better

---

## Question 32
- [ ] Done
**What techniques help with text classification consistency in federated learning scenarios?**
**Answer:**

Federated learning (FL) trains text classifiers across multiple decentralized data sources (devices, organizations) without sharing raw text. Each client trains locally and shares only model updates.

**Core Concepts:**

| Challenge | Description | Solution |
|---|---|---|
| Non-IID Data | Clients have different label distributions | FedProx, SCAFFOLD |
| Communication Cost | Model updates are large | Gradient compression, quantization |
| Privacy Leakage | Gradients can reveal text | Secure aggregation, DP |
| Heterogeneous Hardware | Variable compute across clients | Asynchronous FL |
| Data Quantity Imbalance | Some clients have minimal data | Weighted aggregation |

| FL Algorithm | Description |
|---|---|
| FedAvg | Average model weights from all clients |
| FedProx | FedAvg + proximal regularization |
| FedMA | Match and average neurons across clients |
| SCAFFOLD | Variance reduction for non-IID data |

**Python Code Example:**
```python
# Pipeline: Simplified federated text classification
import copy
import torch

class FederatedTextClassification:
    def __init__(self, global_model, num_clients=10):
        self.global_model = global_model
        self.num_clients = num_clients

    def federated_round(self, client_datasets, local_epochs=3, lr=0.001):
        client_models = []
        client_sizes = []

        for dataset in client_datasets:
            local_model = copy.deepcopy(self.global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            # Local training
            local_model.train()
            for epoch in range(local_epochs):
                for batch in dataset:
                    optimizer.zero_grad()
                    loss = local_model.compute_loss(batch)
                    loss.backward()
                    optimizer.step()
            client_models.append(local_model)
            client_sizes.append(len(dataset))

        # Weighted aggregation (FedAvg)
        self._aggregate(client_models, client_sizes)

    def _aggregate(self, client_models, client_sizes):
        total = sum(client_sizes)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = sum(
                client_models[i].state_dict()[key] * (client_sizes[i] / total)
                for i in range(len(client_models))
            )
        self.global_model.load_state_dict(global_dict)
```

**Interview Tips:**
- FedAvg is the baseline but struggles with non-IID text data across clients
- Discuss that differential privacy (DP-SGD) can be added per-client for formal guarantees
- Mention that communication efficiency is critical: send only gradients, not full models
- Note that federated NLP is common in healthcare (hospitals) and mobile keyboards (GBoard)

---

## Question 33
- [ ] Done
**How do you implement efficient batch processing for large-scale text classification?**
**Answer:**

Large-scale classification (millions/billions of documents) requires optimized batch processing for throughput, latency, and resource efficiency.

**Core Concepts:**

| Optimization | Description | Speedup |
|---|---|---|
| Dynamic Batching | Group similar-length texts | 2-3x (less padding waste) |
| Model Distillation | Use smaller student model | 5-10x faster |
| ONNX Runtime | Optimized inference engine | 2-4x |
| Quantization | INT8/FP16 model weights | 2-4x (less memory) |
| Async Processing | Pipeline CPU/GPU stages | Better utilization |
| Multi-GPU | Distribute across GPUs | Linear scaling |

**Python Code Example:**
```python
# Pipeline: Efficient batch text classification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import islice

class EfficientBatchClassifier:
    def __init__(self, model_name="distilbert-base-uncased", batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.model = self.model.cuda().half()  # FP16

    def classify_batch(self, texts):
        """Classify a batch with dynamic padding."""
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).cpu().tolist()

    def classify_stream(self, text_iterator):
        """Process large dataset as stream of batches."""
        results = []
        batch = list(islice(text_iterator, self.batch_size))
        while batch:
            # Sort by length for efficient padding
            indexed = sorted(enumerate(batch), key=lambda x: len(x[1]))
            sorted_texts = [t for _, t in indexed]
            preds = self.classify_batch(sorted_texts)
            # Restore original order
            ordered = [None] * len(batch)
            for (orig_idx, _), pred in zip(indexed, preds):
                ordered[orig_idx] = pred
            results.extend(ordered)
            batch = list(islice(text_iterator, self.batch_size))
        return results
```

**Interview Tips:**
- Dynamic batching (sorting by length) reduces padding waste by 30-50%
- DistilBERT is 60% faster than BERT with 97% of performance
- ONNX export + TensorRT is the standard production optimization path
- Discuss that async I/O (reading next batch while GPU processes current) maximizes throughput

---

## Question 34
- [ ] Done
**What strategies work best for text classification with specific regulatory requirements?**
**Answer:**

Regulated industries (healthcare, finance, legal) impose constraints on how text classifiers operate: explainability requirements, audit trails, bias testing, and data handling rules.

**Core Concepts:**

| Regulation | Domain | Key Requirement |
|---|---|---|
| GDPR | General (EU) | Right to explanation, data minimization |
| HIPAA | Healthcare (US) | PHI protection, access controls |
| ECOA/Fair Lending | Finance (US) | No discriminatory decisions |
| EU AI Act | General (EU) | Risk-based AI regulation |
| SOX | Finance (US) | Audit trail, internal controls |

| Compliance Strategy | Implementation |
|---|---|
| Explainability | LIME, SHAP for per-prediction explanations |
| Audit Trail | Log all inputs, outputs, model versions |
| Bias Testing | Regular fairness audits across groups |
| Model Cards | Document model capabilities and limitations |
| Human-in-the-Loop | Mandatory review for high-stakes decisions |

**Python Code Example:**
```python
# Pipeline: Regulatory-compliant text classification
import json
import datetime
import hashlib

class RegulatoryCompliantClassifier:
    def __init__(self, model, model_version, explainer=None):
        self.model = model
        self.model_version = model_version
        self.explainer = explainer
        self.audit_log = []

    def classify_with_audit(self, text, user_id, purpose):
        prediction = self.model.predict(text)
        explanation = self.explainer.explain(text) if self.explainer else None

        record = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'input_hash': hashlib.sha256(text.encode()).hexdigest(),
            'model_version': self.model_version,
            'prediction': prediction,
            'explanation': explanation,
            'user_id': user_id,
            'purpose': purpose,
        }
        self.audit_log.append(record)
        return {'prediction': prediction, 'explanation': explanation, 'audit_id': len(self.audit_log)}

    def export_audit_log(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)

    def generate_model_card(self):
        return {
            'model_version': self.model_version,
            'intended_use': 'Document classification for regulatory filings',
            'limitations': 'Not tested on handwritten documents',
            'ethical_considerations': 'Regular bias audits required',
            'performance_metrics': {},  # Fill from evaluation
        }
```

**Interview Tips:**
- GDPR's "right to explanation" means you may need interpretable models or post-hoc explanations
- Discuss model cards (Mitchell et al., 2019) as standard documentation practice
- Mention that regulated industries often prefer simpler, explainable models over black-box accuracy
- Note that audit trails must include model version, timestamp, input hash, and output

---

## Question 35
- [ ] Done
**How do you handle text classification for texts requiring domain expertise?**
**Answer:**

Domain-specific text classification (medical, legal, scientific, financial) requires specialized knowledge embedded in vocabularies, relationships, and reasoning patterns distinct from general text.

**Core Concepts:**

| Strategy | Description | Example |
|---|---|---|
| Domain Pretraining | Continue LM pretraining on domain text | BioBERT, LegalBERT, FinBERT |
| Domain Vocabulary | Domain-specific tokenizer | Medical terms as single tokens |
| Knowledge Injection | Incorporate domain ontologies | UMLS for medical, FIBO for finance |
| Expert Features | Hand-crafted domain features | Regex for legal citations |
| Active Learning | Domain experts label informative samples | Reduce annotation cost |
| Transfer Learning | Start from domain-pretrained model | Fine-tune on specific task |

**Python Code Example:**
```python
# Pipeline: Domain-specific text classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DomainExpertClassifier:
    # Recommended domain-pretrained models
    DOMAIN_MODELS = {
        'medical': 'dmis-lab/biobert-base-cased-v1.1',
        'legal': 'nlpaueb/legal-bert-base-uncased',
        'finance': 'ProsusAI/finbert',
        'scientific': 'allenai/scibert_scivocab_uncased',
        'general': 'bert-base-uncased',
    }

    def __init__(self, domain='general', num_labels=5):
        model_name = self.DOMAIN_MODELS.get(domain, self.DOMAIN_MODELS['general'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def add_domain_terms(self, terms):
        """Add domain-specific terms to tokenizer."""
        new_tokens = [t for t in terms if t not in self.tokenizer.vocab]
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        return len(new_tokens)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).item()
```

**Interview Tips:**
- Domain pretraining (BioBERT, SciBERT) improves 3-10% over general BERT on domain tasks
- Adding domain vocabulary prevents important terms from being split into subwords
- Knowledge graph injection (ERNIE, K-BERT) incorporates structured domain knowledge
- Expert-in-the-loop active learning is cost-effective for specialized domains

---

## Question 36
- [ ] Done
**What approaches help with text classification adaptation to user-specific categories?**
**Answer:**

User-specific (personalized) classification adapts to individual users' categories, preferences, and labeling styles. Each user may define categories differently or have unique classification needs.

**Core Concepts:**

| Approach | Description | When to Use |
|---|---|---|
| Per-User Fine-Tuning | Separate model per user | Few power users |
| User Embeddings | Learnable user preference vectors | Many users, shared model |
| Meta-Learning (MAML) | Quick adaptation from few user examples | New users, few-shot |
| Adapter Layers | User-specific adapter modules | Moderate user count |
| Personalized Thresholds | User-specific decision boundaries | Simple personalization |
| Collaborative Filtering | Similar users share preferences | Cold-start problem |

**Python Code Example:**
```python
# Pipeline: User-adaptive text classification
import torch
import torch.nn as nn
from transformers import AutoModel

class PersonalizedClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_users=1000, num_classes=10, user_dim=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.user_embeddings = nn.Embedding(num_users, user_dim)
        self.classifier = nn.Linear(hidden + user_dim, num_classes)

    def forward(self, input_ids, attention_mask, user_id):
        text_repr = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = text_repr.last_hidden_state[:, 0]
        user_repr = self.user_embeddings(user_id)
        combined = torch.cat([cls_repr, user_repr], dim=-1)
        return self.classifier(combined)

    def adapt_to_new_user(self, user_examples, epochs=5, lr=0.01):
        """Quickly adapt to new user from few labeled examples."""
        new_user_id = self.user_embeddings.num_embeddings
        new_embed = nn.Embedding(1, self.user_embeddings.embedding_dim)
        optimizer = torch.optim.Adam(new_embed.parameters(), lr=lr)
        for _ in range(epochs):
            for text_features, label in user_examples:
                optimizer.zero_grad()
                user_repr = new_embed(torch.tensor([0]))
                combined = torch.cat([text_features, user_repr], dim=-1)
                logits = self.classifier(combined)
                loss = nn.functional.cross_entropy(logits, label)
                loss.backward()
                optimizer.step()
        return new_embed.weight.data
```

**Interview Tips:**
- User embeddings + shared model is the most scalable personalization approach
- Meta-learning (MAML, Prototypical Networks) enables rapid adaptation from 5-10 examples
- Discuss cold-start: new users need a good default before personalization kicks in
- Note that personalization creates fairness challenges — ensure equitable quality across users

---

## Question 37
- [ ] Done
**How do you implement monitoring and quality control for text classification systems?**
**Answer:**

Production text classifiers need continuous monitoring to detect degradation, data drift, and errors before they impact users.

**Core Concepts:**

| Monitor | What to Track | Alert Threshold |
|---|---|---|
| Accuracy/F1 | Overall performance on labeled sample | Drop > 2% from baseline |
| Prediction Distribution | Class proportion over time | Shift > 10% from training |
| Confidence Distribution | Model certainty levels | Rising low-confidence rate |
| Latency | Prediction speed | P99 > SLA threshold |
| Input Distribution | Feature statistics, text length | Statistical divergence |
| Error Rate Patterns | Systematic failure modes | New error clusters |

**Python Code Example:**
```python
# Pipeline: Text classification monitoring system
import numpy as np
from collections import Counter, deque
import datetime

class ClassificationMonitor:
    def __init__(self, num_classes, window_size=10000):
        self.num_classes = num_classes
        self.window = deque(maxlen=window_size)
        self.alerts = []
        self.baseline_dist = None

    def log_prediction(self, prediction, confidence, latency_ms):
        self.window.append({
            'pred': prediction, 'conf': confidence,
            'latency': latency_ms, 'time': datetime.datetime.utcnow()
        })

    def set_baseline(self):
        preds = [r['pred'] for r in self.window]
        counts = Counter(preds)
        total = sum(counts.values())
        self.baseline_dist = {k: v/total for k, v in counts.items()}

    def check_distribution_drift(self, threshold=0.1):
        if not self.baseline_dist:
            return None
        preds = [r['pred'] for r in self.window]
        counts = Counter(preds)
        total = sum(counts.values())
        current_dist = {k: v/total for k, v in counts.items()}
        max_drift = max(
            abs(current_dist.get(k, 0) - self.baseline_dist.get(k, 0))
            for k in set(list(current_dist.keys()) + list(self.baseline_dist.keys()))
        )
        if max_drift > threshold:
            self.alerts.append({'type': 'distribution_drift', 'drift': max_drift})
        return max_drift

    def get_health_report(self):
        records = list(self.window)
        return {
            'total_predictions': len(records),
            'avg_confidence': np.mean([r['conf'] for r in records]),
            'avg_latency_ms': np.mean([r['latency'] for r in records]),
            'low_confidence_rate': np.mean([r['conf'] < 0.3 for r in records]),
            'alerts': len(self.alerts),
        }
```

**Interview Tips:**
- Discuss the "two-sigma" approach: alert when metrics deviate >2 std from baseline
- Mention that prediction distribution drift is the earliest signal of problems
- Note that periodic human evaluation on random samples catches systematic blind spots
- Emphasize dashboards (Grafana, Evidently AI) for real-time monitoring visualization

---

## Question 38
- [ ] Done
**What techniques work best for text classification in texts with special formatting?**
**Answer:**

Special formatting includes code snippets, mathematical equations, tables, bullet lists, markdown, emojis, hashtags, and mixed-media content. Standard tokenizers may destroy formatting information that carries semantic meaning.

**Core Concepts:**

| Format Type | Challenge | Handling |
|---|---|---|
| Code Snippets | Programming syntax | CodeBERT, preserve structure |
| Math Equations | LaTeX/symbolic notation | Specialized tokenizer |
| Tables | Row/column structure | Linearize with markers |
| Markdown/HTML | Structural tags | Parse to extract structure |
| Emojis | Semantic content | Emoji embeddings, text conversion |
| Hashtags | Compound words (#NeverForget) | CamelCase splitting |
| URLs | Reference signals | Extract domain, replace or keep |

**Python Code Example:**
```python
# Pipeline: Format-aware text classification preprocessing
import re
import unicodedata

class FormatAwarePreprocessor:
    def __init__(self):
        self.emoji_map = {}  # emoji -> text description

    def process_hashtags(self, text):
        """Split CamelCase hashtags into words."""
        def split_hashtag(match):
            tag = match.group(1)
            words = re.sub(r'([A-Z])', r' \1', tag).strip()
            return words
        return re.sub(r'#(\w+)', split_hashtag, text)

    def process_emojis(self, text):
        """Convert emojis to text descriptions."""
        result = []
        for char in text:
            if unicodedata.category(char).startswith('So'):
                name = unicodedata.name(char, '').lower()
                result.append(f'[emoji:{name}]')
            else:
                result.append(char)
        return ''.join(result)

    def process_code_blocks(self, text):
        """Mark code blocks with special tokens."""
        text = re.sub(r'```(\w*)\n(.*?)```', r'[CODE_START \1] \2 [CODE_END]', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'[INLINE_CODE] \1 [/INLINE_CODE]', text)
        return text

    def process_tables(self, text):
        """Linearize markdown tables."""
        lines = text.split('\n')
        result = []
        for line in lines:
            if '|' in line and not line.strip().startswith('|-'):
                cells = [c.strip() for c in line.split('|') if c.strip()]
                result.append(' [COL] '.join(cells))
            else:
                result.append(line)
        return '\n'.join(result)

    def preprocess(self, text):
        text = self.process_hashtags(text)
        text = self.process_emojis(text)
        text = self.process_code_blocks(text)
        return text
```

**Interview Tips:**
- Emojis carry significant sentiment signal — converting them to text descriptions preserves info
- CamelCase splitting for hashtags recovers 2-3 tokens per hashtag on average
- Code classification (CodeBERT, GraphCodeBERT) uses AST structure, not just text
- Discuss that tables should be linearized with column markers rather than flattened

---

## Question 39
- [ ] Done
**How do you handle text classification optimization when balancing precision and recall?**
**Answer:**

Precision-recall tradeoffs are fundamental: increasing recall (catching more positives) typically decreases precision (more false positives). The optimal balance depends on the application's cost structure.

**Core Concepts:**

| Metric | Definition | Optimized When |
|---|---|---|
| Precision | TP / (TP + FP) | False positives are costly (spam filter, legal) |
| Recall | TP / (TP + FN) | False negatives are costly (disease, fraud) |
| F1 | Harmonic mean of P and R | Balanced importance |
| F-beta | Weighted harmonic mean | β>1 favors recall, β<1 favors precision |
| PR-AUC | Area under PR curve | Overall model quality |

| Optimization Technique | Effect |
|---|---|
| Threshold Tuning | Shift decision boundary on PR curve |
| Class Weights | Higher weight on minority class increases recall |
| Focal Loss | Down-weight easy examples, focus on hard ones |
| Cost-Sensitive Learning | Different misclassification costs |
| Ensemble + Voting | Majority = precision, Any = recall |

**Python Code Example:**
```python
# Pipeline: Precision-recall optimization for text classification
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

class PrecisionRecallOptimizer:
    def __init__(self, model, beta=1.0):
        self.model = model
        self.beta = beta
        self.optimal_threshold = 0.5

    def find_optimal_threshold(self, val_texts, val_labels):
        probs = self.model.predict_proba(val_texts)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)

        # F-beta score at each threshold
        f_scores = ((1 + self.beta**2) * precisions * recalls /
                    (self.beta**2 * precisions + recalls + 1e-10))
        best_idx = np.argmax(f_scores)
        self.optimal_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        return {
            'threshold': self.optimal_threshold,
            'precision': precisions[best_idx],
            'recall': recalls[best_idx],
            'f_beta': f_scores[best_idx],
        }

    def predict_optimized(self, texts):
        probs = self.model.predict_proba(texts)[:, 1]
        return (probs >= self.optimal_threshold).astype(int)

    def multi_threshold_analysis(self, val_texts, val_labels):
        """Show precision/recall at different operating points."""
        probs = self.model.predict_proba(val_texts)[:, 1]
        results = []
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            preds = (probs >= threshold).astype(int)
            results.append({
                'threshold': threshold,
                'precision': np.mean(preds[val_labels == 1] == 1) if preds.sum() > 0 else 0,
                'recall': np.mean(val_labels[preds == 1] == 1) if (val_labels == 1).sum() > 0 else 0,
            })
        return results
```

**Interview Tips:**
- The default 0.5 threshold is almost never optimal for real applications
- F-beta with β=2 is common in medical/safety (values recall 2x more than precision)
- Discuss that PR-AUC is more informative than ROC-AUC for imbalanced datasets
- Mention that ensemble voting strategy controls the tradeoff: unanimous = precision, any = recall

---

## Question 40
- [ ] Done
**What strategies help with text classification for emerging text types and platforms?**
**Answer:**

New platforms (TikTok captions, Discord threads, Mastodon toots) and text types (voice transcripts, AR overlays) emerge regularly. Classifiers need to adapt to new registers, conventions, and constraints.

**Core Concepts:**

| Platform | Text Characteristics | Challenge |
|---|---|---|
| TikTok/Reels | Short captions, hashtags, emojis | Very short, heavy slang |
| Discord/Slack | Thread-based, informal, reactions | Context from thread needed |
| Reddit | Long-form, subreddit-specific jargon | Community-specific language |
| Voice Transcripts | ASR errors, disfluencies | Noisy text, no punctuation |
| Code Reviews | Technical + natural language mix | Hybrid content |
| Chat/SMS | Abbreviations, typos | Non-standard spelling |

| Adaptation Strategy | Description |
|---|---|
| Platform-Specific Pretraining | Continue pretraining on platform text |
| Slang Normalization | Map slang to standard forms |
| Few-Shot Adaptation | Quick tune on 10-50 platform examples |
| Data Collection Pipeline | Continuously collect from new platforms |
| Cross-Platform Transfer | Fine-tune from similar platform |

**Python Code Example:**
```python
# Pipeline: Platform-adaptive text classification
import re

class PlatformAdaptiveClassifier:
    def __init__(self, base_model):
        self.base = base_model
        self.normalizers = {
            'tiktok': self._normalize_tiktok,
            'discord': self._normalize_discord,
            'voice': self._normalize_voice_transcript,
        }

    def _normalize_tiktok(self, text):
        text = re.sub(r'#(\w+)', lambda m: re.sub(r'([A-Z])', r' \1', m.group(1)).strip(), text)
        slang = {'fr': 'for real', 'ngl': 'not gonna lie', 'imo': 'in my opinion',
                 'smh': 'shaking my head', 'tbh': 'to be honest'}
        for abbr, full in slang.items():
            text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
        return text

    def _normalize_discord(self, text):
        text = re.sub(r'<@!?\d+>', '[USER_MENTION]', text)  # User mentions
        text = re.sub(r'<#\d+>', '[CHANNEL]', text)  # Channel mentions
        text = re.sub(r'<:\w+:\d+>', '[CUSTOM_EMOJI]', text)  # Custom emojis
        return text

    def _normalize_voice_transcript(self, text):
        fillers = ['um', 'uh', 'like', 'you know', 'i mean']
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()

    def classify(self, text, platform='general'):
        normalizer = self.normalizers.get(platform)
        if normalizer:
            text = normalizer(text)
        return self.base.predict(text)
```

**Interview Tips:**
- Platform-specific pretraining (TweetBERT, RedditBERT) captures platform conventions
- Slang normalization is essential for short-form social media text
- Discuss that ASR transcript classification needs disfluency removal and punctuation restoration
- Note that cross-platform transfer (Twitter → TikTok) often works with minimal fine-tuning

---

## Question 41
- [ ] Done
**How do you implement transfer learning for multilingual text classification?**
**Answer:**

Multilingual transfer learning uses models pretrained on many languages (XLM-R, mBERT) to transfer classification knowledge from high-resource languages (English) to low-resource ones without target-language training data.

**Core Concepts:**

| Transfer Type | Source | Target | Data Needed |
|---|---|---|
| Zero-Shot Cross-Lingual | English labels | Any language | None in target |
| Few-Shot Cross-Lingual | English + few target | Target language | 10-100 target examples |
| Translate-Train | MT of English data | Target language | MT system only |
| Multi-Source Transfer | Multiple source languages | Target language | Diverse sources |

| Model | Languages | Key Feature |
|---|---|---|
| mBERT | 104 | Original multilingual BERT |
| XLM-RoBERTa | 100 | Best cross-lingual transfer |
| mT5 | 101 | Generative multilingual |
| BLOOM | 46+ | Large multilingual LM |

**Python Code Example:**
```python
# Pipeline: Cross-lingual transfer learning for text classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class CrossLingualTransfer:
    def __init__(self, model_name="xlm-roberta-base", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def train_on_english(self, en_texts, en_labels):
        """Fine-tune on English labeled data."""
        dataset = Dataset.from_dict({'text': en_texts, 'label': en_labels})
        dataset = dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, max_length=512), batched=True)
        args = TrainingArguments(
            output_dir='./results', num_train_epochs=3,
            per_device_train_batch_size=16, learning_rate=2e-5,
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)
        trainer.train()

    def predict_any_language(self, texts):
        """Classify text in ANY language after English fine-tuning."""
        import torch
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).tolist()

# Usage: Train on English, predict on German/Chinese/Arabic
transfer = CrossLingualTransfer(num_labels=3)
transfer.train_on_english(english_texts, english_labels)
# Works on any language without target-language training!
predictions = transfer.predict_any_language(german_texts)
```

**Interview Tips:**
- XLM-RoBERTa achieves within 5% of supervised performance in zero-shot cross-lingual transfer
- Translate-train with back-translation often outperforms zero-shot transfer
- Discuss that language-specific fine-tuning adapters are more parameter-efficient than full fine-tuning
- Note that tokenizer coverage for low-resource languages is a common bottleneck

---

## Question 42
- [ ] Done
**What approaches work best for text classification with minimal false positive requirements?**
**Answer:**

High-precision requirements (minimal false positives) are critical in applications where a false positive is very costly: legal discovery, medical alerts, automated account suspension, and fraud detection.

**Core Concepts:**

| Strategy | Mechanism | Tradeoff |
|---|---|---|
| High Threshold | Only predict positive at high confidence | Fewer predictions |
| Multi-Stage Pipeline | Screen → Verify → Confirm | Lower throughput |
| Ensemble Consensus | Require all models to agree | Conservative predictions |
| Human-in-the-Loop | Human reviews positives | Higher cost |
| Negative Mining | Train specifically on hard negatives | Better discrimination |
| Calibration | Ensure predicted probabilities are accurate | Reliable thresholds |

**Python Code Example:**
```python
# Pipeline: High-precision text classification
import numpy as np

class HighPrecisionClassifier:
    def __init__(self, models, min_precision=0.95):
        self.models = models
        self.min_precision = min_precision
        self.threshold = 0.5

    def calibrate_threshold(self, val_texts, val_labels):
        """Find threshold achieving target precision."""
        all_probs = np.mean([m.predict_proba(val_texts)[:, 1] for m in self.models], axis=0)
        for threshold in np.arange(0.99, 0.01, -0.01):
            preds = (all_probs >= threshold).astype(int)
            if preds.sum() == 0:
                continue
            precision = val_labels[preds == 1].mean()
            if precision >= self.min_precision:
                self.threshold = threshold
                return {'threshold': threshold, 'precision': precision,
                        'recall': preds[val_labels == 1].mean()}
        return {'threshold': 0.99, 'precision': 1.0, 'recall': 0.0}

    def predict(self, texts):
        # Ensemble averaging
        all_probs = np.mean([m.predict_proba(texts)[:, 1] for m in self.models], axis=0)
        predictions = (all_probs >= self.threshold).astype(int)
        confidences = all_probs
        return [{'label': int(p), 'confidence': float(c), 'above_threshold': bool(p)}
                for p, c in zip(predictions, confidences)]

    def predict_with_consensus(self, texts):
        """Require ALL models to agree for positive prediction."""
        all_preds = [m.predict(texts) for m in self.models]
        consensus = np.all(all_preds, axis=0).astype(int)
        return consensus
```

**Interview Tips:**
- Ensemble consensus (all models agree) is the simplest high-precision strategy
- Discuss temperature scaling for better calibration before threshold tuning
- Multi-stage verification pipelines sacrifice throughput for precision
- Hard negative mining during training improves the model's ability to discriminate borderline cases

---

## Question 43
- [ ] Done
**How do you handle text classification integration with information retrieval systems?**
**Answer:**

Text classification enhances information retrieval (IR) by adding semantic labels, filtering results, routing queries, and reranking retrieved documents by category relevance.

**Core Concepts:**

| Integration Pattern | Description | Example |
|---|---|---|
| Query Classification | Classify query intent before search | "buy shoes" → COMMERCIAL |
| Document Tagging | Pre-classify documents for faceted search | Auto-tag news by topic |
| Result Filtering | Filter IR results by category | Only show "scientific" results |
| Reranking | Boost results matching query category | Topic-aware reranking |
| Query Expansion | Use category to add relevant terms | MEDICAL query → add synonyms |
| Routing | Route query to category-specific index | Legal vs. Medical index |

**Python Code Example:**
```python
# Pipeline: Classification-enhanced information retrieval
class ClassificationEnhancedIR:
    def __init__(self, classifier, search_engine):
        self.classifier = classifier
        self.search = search_engine

    def classify_and_search(self, query, top_k=10):
        # Step 1: Classify query intent
        query_category = self.classifier.predict(query)

        # Step 2: Retrieve documents
        results = self.search.query(query, top_k=top_k * 2)  # Over-retrieve

        # Step 3: Classify and filter/boost results
        scored_results = []
        for doc in results:
            doc_category = self.classifier.predict(doc['text'])
            category_match = 1.0 if doc_category == query_category else 0.5
            combined_score = doc['relevance_score'] * category_match
            scored_results.append({**doc, 'combined_score': combined_score,
                                   'query_category': query_category,
                                   'doc_category': doc_category})

        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return scored_results[:top_k]

    def faceted_search(self, query, category_filter=None):
        results = self.search.query(query)
        if category_filter:
            results = [r for r in results
                      if self.classifier.predict(r['text']) == category_filter]
        return results
```

**Interview Tips:**
- Query classification (informational vs. navigational vs. transactional) improves search quality significantly
- Pre-classifying documents enables faceted search without manual tagging
- Discuss that classification reranking adds semantic understanding to keyword-based IR
- Note that modern search (Elasticsearch) supports ML-based reranking pipelines natively

---

## Question 44
- [ ] Done
**What techniques help with text classification for texts requiring contextual understanding?**
**Answer:**

Contextual understanding means the classifier must consider information beyond the immediate text: surrounding documents, conversation history, author identity, temporal context, or world knowledge.

**Core Concepts:**

| Context Type | Source | Example |
|---|---|---|
| Document Context | Surrounding paragraphs | Paragraph classification in a paper |
| Conversational | Previous dialogue turns | Intent depends on history |
| Author Context | Author's prior writings/profile | Author style affects meaning |
| Temporal | Time of writing | "Current president" changes over time |
| Knowledge Base | External facts | Entity disambiguation requires world knowledge |
| Cross-Document | Related documents | Classify based on citation network |

**Python Code Example:**
```python
# Pipeline: Context-enhanced text classification
import torch
import torch.nn as nn
from transformers import AutoModel

class ContextualClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=5, context_methods=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        # Context fusion: text + surrounding context
        self.context_attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def encode_text(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, target_ids, target_mask, context_ids_list, context_mask_list):
        # Encode target text
        target_repr = self.encode_text(target_ids, target_mask)

        # Encode context documents
        context_reprs = []
        for ctx_ids, ctx_mask in zip(context_ids_list, context_mask_list):
            context_reprs.append(self.encode_text(ctx_ids, ctx_mask))
        context_stack = torch.stack(context_reprs, dim=1)  # [batch, n_ctx, hidden]

        # Attend over context
        attended, _ = self.context_attention(
            target_repr.unsqueeze(1), context_stack, context_stack
        )
        attended = attended.squeeze(1)

        combined = torch.cat([target_repr, attended], dim=-1)
        return self.classifier(combined)
```

**Interview Tips:**
- Cross-attention over context documents is the standard approach for context-aware classification
- Discuss that conversation history (2-5 turns) significantly improves dialogue intent detection
- Mention knowledge-enhanced models (ERNIE, K-BERT) that inject entity knowledge
- Note that temporal context can be added as positional encoding or explicit date features

---

## Question 45
- [ ] Done
**How do you implement customizable text classification for different user needs?**
**Answer:**

Customizable classifiers allow end users to define their own categories, provide examples, and adjust behavior without ML expertise. This is essential for platforms where users have diverse classification needs.

**Core Concepts:**

| Customization Level | User Effort | Setup |
|---|---|---|
| Category Labels Only | Define label names | Zero-shot classification |
| Few Examples | 5-20 labeled examples | Few-shot, prompt-based |
| Keyword Rules | Define keywords per category | Rule-based + ML hybrid |
| Full Training Data | Large labeled dataset | Standard fine-tuning |
| Feedback Loop | Correct predictions over time | Active learning |

**Python Code Example:**
```python
# Pipeline: User-customizable text classification
from transformers import pipeline

class CustomizableClassifier:
    def __init__(self):
        self.zero_shot = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli")
        self.user_configs = {}

    def create_classifier(self, user_id, categories, examples=None, keywords=None):
        self.user_configs[user_id] = {
            'categories': categories,
            'examples': examples or {},
            'keywords': keywords or {},
        }

    def classify(self, user_id, text):
        config = self.user_configs[user_id]

        # Try keyword matching first (fast)
        for category, kws in config.get('keywords', {}).items():
            if any(kw.lower() in text.lower() for kw in kws):
                return {'label': category, 'method': 'keyword', 'confidence': 0.8}

        # Fall back to zero-shot with user's categories
        result = self.zero_shot(text, candidate_labels=config['categories'])
        return {
            'label': result['labels'][0],
            'confidence': result['scores'][0],
            'method': 'zero-shot',
            'all_scores': dict(zip(result['labels'], result['scores'])),
        }

    def add_feedback(self, user_id, text, correct_label):
        """User corrects a prediction to improve future results."""
        config = self.user_configs[user_id]
        if 'examples' not in config:
            config['examples'] = {}
        config['examples'].setdefault(correct_label, []).append(text)

# Usage
clf = CustomizableClassifier()
clf.create_classifier('user1', ['urgent', 'routine', 'spam'],
                       keywords={'urgent': ['asap', 'critical', 'emergency']})
result = clf.classify('user1', 'We need this fix ASAP')
```

**Interview Tips:**
- Zero-shot classification enables instant setup with just category names
- Keywords + ML hybrid catches obvious cases fast and uses ML for ambiguous ones
- User feedback loops create a virtuous cycle improving classification over time
- Discuss that user-customizable systems need guardrails to prevent misuse

---

## Question 46
- [ ] Done
**What strategies work best for text classification in high-throughput processing scenarios?**
**Answer:**

High-throughput scenarios (millions of classifications per hour) require optimized inference pipelines that maximize throughput while maintaining acceptable latency and accuracy.

**Core Concepts:**

| Optimization | Technique | Speedup |
|---|---|---|
| Model Distillation | DistilBERT, TinyBERT | 2-6x faster |
| Quantization | INT8/FP16 weights | 2-4x + less memory |
| ONNX Runtime | Optimized inference graph | 2-3x |
| TensorRT | GPU-optimized kernels | 3-5x on NVIDIA |
| Batching | Dynamic batching server | Better GPU utilization |
| Caching | Hash-based result cache | Instant for repeats |
| Model Pruning | Remove redundant weights | 1.5-3x |
| Cascade Classification | Fast model → complex model only if needed | 3-10x avg |

**Python Code Example:**
```python
# Pipeline: High-throughput text classification
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

class HighThroughputClassifier:
    def __init__(self, fast_model, full_model, cache_size=100000, confidence_threshold=0.8):
        self.fast = fast_model  # DistilBERT or similar
        self.full = full_model  # Full BERT/RoBERTa
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.conf_threshold = confidence_threshold

    def _cache_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def classify_single(self, text):
        # Check cache first
        key = self._cache_key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        # Try fast model first (cascade)
        result = self.fast.predict_with_confidence(text)
        if result['confidence'] < self.conf_threshold:
            result = self.full.predict_with_confidence(text)

        # Cache result
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def classify_batch_parallel(self, texts, num_workers=4):
        """Parallel batch classification."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.classify_single, texts))
        return results
```

**Interview Tips:**
- Cascade classification (fast model → slow model only when uncertain) gives best throughput/accuracy tradeoff
- DistilBERT + ONNX + INT8 quantization can achieve 10-20x speedup over vanilla BERT
- LRU caching is highly effective for repetitive text streams (templated emails, duplicate content)
- Discuss Triton Inference Server for production GPU batching and model management

---

## Question 47
- [ ] Done
**How do you handle text classification quality benchmarking across different models?**
**Answer:**

Systematic benchmarking compares models fairly across multiple metrics, datasets, and conditions to select the best model for production deployment.

**Core Concepts:**

| Benchmark Dimension | What to Compare | Metrics |
|---|---|---|
| Accuracy | Prediction quality | F1, accuracy, PR-AUC |
| Speed | Inference throughput | Predictions/second, latency P50/P99 |
| Model Size | Resource requirements | Parameters, disk size, GPU memory |
| Robustness | Performance under noise | Accuracy on perturbed data |
| Fairness | Bias across groups | Equalized odds, demographic parity |
| Cost | Total resource cost | $/1M predictions |

**Python Code Example:**
```python
# Pipeline: Model benchmarking framework for text classification
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

class ModelBenchmark:
    def __init__(self, models, test_texts, test_labels):
        self.models = models  # dict of {name: model}
        self.texts = test_texts
        self.labels = test_labels

    def benchmark_accuracy(self):
        results = {}
        for name, model in self.models.items():
            preds = model.predict(self.texts)
            results[name] = {
                'accuracy': accuracy_score(self.labels, preds),
                'f1_macro': f1_score(self.labels, preds, average='macro'),
                'f1_weighted': f1_score(self.labels, preds, average='weighted'),
            }
        return results

    def benchmark_speed(self, n_runs=3):
        results = {}
        for name, model in self.models.items():
            times = []
            for _ in range(n_runs):
                start = time.time()
                model.predict(self.texts)
                times.append(time.time() - start)
            avg_time = np.mean(times)
            results[name] = {
                'total_time_s': avg_time,
                'predictions_per_sec': len(self.texts) / avg_time,
                'avg_latency_ms': avg_time / len(self.texts) * 1000,
            }
        return results

    def full_benchmark(self):
        return {
            'accuracy': self.benchmark_accuracy(),
            'speed': self.benchmark_speed(),
        }

    def generate_report(self):
        results = self.full_benchmark()
        print(f"{'Model':<20} {'F1-Macro':<12} {'Preds/sec':<12} {'Latency(ms)':<12}")
        print('-' * 56)
        for name in self.models:
            acc = results['accuracy'][name]
            spd = results['speed'][name]
            print(f"{name:<20} {acc['f1_macro']:<12.4f} {spd['predictions_per_sec']:<12.1f} {spd['avg_latency_ms']:<12.2f}")
```

**Interview Tips:**
- Always benchmark on the same held-out test set with statistical significance testing
- Discuss that speed/accuracy Pareto front helps choose based on deployment constraints
- Mention that robustness benchmarks (TextAttack, CheckList) reveal model weaknesses
- Note that cost-per-prediction matters in production: a 2% accuracy gain may not justify 10x cost

---

## Question 48
- [ ] Done
**What approaches help with text classification for texts with evolving language patterns?**
**Answer:**

Language evolves continuously: new slang ("skibidi", "rizz"), meaning shifts ("mid" = mediocre, "cap" = lie), and new domains emerge. Static classifiers degrade without adaptation to these changes.

**Core Concepts:**

| Evolution Type | Example | Adaptation |
|---|---|---|
| New Vocabulary | "Skibidi", "rizz" (2023+) | Continual vocabulary expansion |
| Meaning Drift | "Sick" = ill → great | Periodic re-evaluation |
| Style Evolution | Formal → casual business writing | Domain adaptation |
| Platform Language | TikTok-specific language patterns | Platform-specific fine-tuning |
| Topic Emergence | AI/LLM discussions (2023+) | Active data collection |

| Adaptation Strategy | Description |
|---|---|
| Continual Pretraining | Update LM on recent text periodically |
| Vocabulary Expansion | Add new tokens to tokenizer |
| Sliding Window Training | Train on recent N months of data |
| Curriculum Learning | Start with stable, add evolving data |
| User Feedback | Collect corrections from users |

**Python Code Example:**
```python
# Pipeline: Adaptive classifier for evolving language
from collections import Counter
import re

class EvolvingLanguageClassifier:
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
        self.new_term_detector = Counter()
        self.oov_threshold = 50  # Track terms seen this many times

    def detect_new_terms(self, texts, window_size=10000):
        """Detect frequently appearing OOV terms."""
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if len(encoded) > 2:  # Heavily subword-split = potentially new word
                    self.new_term_detector[token] += 1

        new_terms = [
            term for term, count in self.new_term_detector.items()
            if count >= self.oov_threshold
        ]
        return new_terms

    def expand_vocabulary(self, new_terms):
        """Add detected new terms to tokenizer."""
        added = self.tokenizer.add_tokens(new_terms)
        self.model.resize_token_embeddings(len(self.tokenizer))
        return added

    def retrain_incremental(self, recent_data, epochs=1):
        """Fine-tune on recent data to adapt to language changes."""
        # Implement incremental training on recent data window
        pass

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        return self.model(**inputs).logits.argmax(dim=-1).item()
```

**Interview Tips:**
- Continual pretraining (monthly/quarterly) on new domain text is the primary adaptation strategy
- Vocabulary expansion for new terms prevents excessive subword splitting
- Discuss that sliding window training (last 6-12 months) balances freshness and stability
- Note that user feedback loops are the most direct signal for language evolution

---

## Question 49
- [ ] Done
**How do you implement efficient storage and indexing of text classification results?**
**Answer:**

At scale, storing and retrieving classification results efficiently is essential for downstream analytics, auditing, and serving. The storage strategy depends on query patterns and volume.

**Core Concepts:**

| Storage Option | Best For | Query Speed |
|---|---|---|
| PostgreSQL | Structured results, complex queries | Good with indexes |
| Elasticsearch | Full-text search + classification facets | Excellent for search |
| Redis | Real-time serving, caching | Sub-millisecond |
| Parquet/S3 | Batch analytics, archival | Fast columnar reads |
| MongoDB | Flexible schema, varied metadata | Good for documents |
| ClickHouse | Time-series analytics on results | Excellent for aggregation |

| Indexing Strategy | Purpose |
|---|---|
| Category Index | Fast lookup by predicted class |
| Confidence Index | Find uncertain predictions for review |
| Temporal Index | Time-range queries for monitoring |
| Composite Index | Category + time for trend analysis |

**Python Code Example:**
```python
# Pipeline: Efficient storage and retrieval of classification results
import json
import sqlite3
from datetime import datetime

class ClassificationResultStore:
    def __init__(self, db_path='classifications.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_label ON results(predicted_label)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON results(confidence)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_time ON results(created_at)')
        self.conn.commit()

    def store(self, text_hash, label, confidence, model_version, metadata=None):
        self.conn.execute(
            'INSERT INTO results (text_hash, predicted_label, confidence, model_version, metadata) VALUES (?, ?, ?, ?, ?)',
            (text_hash, label, confidence, model_version, json.dumps(metadata or {}))
        )
        self.conn.commit()

    def query_by_label(self, label, limit=100):
        return self.conn.execute(
            'SELECT * FROM results WHERE predicted_label = ? ORDER BY created_at DESC LIMIT ?',
            (label, limit)
        ).fetchall()

    def query_low_confidence(self, threshold=0.5, limit=100):
        return self.conn.execute(
            'SELECT * FROM results WHERE confidence < ? ORDER BY confidence ASC LIMIT ?',
            (threshold, limit)
        ).fetchall()

    def get_label_distribution(self, start_date=None):
        query = 'SELECT predicted_label, COUNT(*) as cnt FROM results'
        params = []
        if start_date:
            query += ' WHERE created_at >= ?'
            params.append(start_date)
        query += ' GROUP BY predicted_label ORDER BY cnt DESC'
        return self.conn.execute(query, params).fetchall()
```

**Interview Tips:**
- Store text hashes, not raw text, to save space and protect privacy
- Discuss that Elasticsearch is ideal when you need both search and classification facets
- Mention that columnar storage (Parquet) is most efficient for batch analytics
- Note that confidence-indexed queries enable efficient human review of uncertain predictions

---

## Question 50
- [ ] Done
**What techniques work best for balancing text classification accuracy with interpretability?**
**Answer:**

The accuracy-interpretability tradeoff is central to deploying text classifiers: complex models (deep transformers) are more accurate but opaque, while simpler models (logistic regression, decision trees) are interpretable but less accurate.

**Core Concepts:**

| Model | Accuracy | Interpretability | Best For |
|---|---|---|---|
| Logistic Regression + TF-IDF | Moderate | High (feature weights) | Regulated domains |
| Decision Tree/Random Forest | Moderate | Medium (feature importance) | Explainable baselines |
| BERT/RoBERTa | High | Low (black box) | Maximum accuracy |
| BERT + LIME/SHAP | High | Medium (post-hoc) | Accuracy + explanation |
| Attention Visualization | High | Medium (attention maps) | Research, debugging |
| Distilled Interpretable | Good | High (student = simple model) | Production compromise |

| Interpretability Method | Approach | Scope |
|---|---|---|
| LIME | Local linear approximation | Per-prediction |
| SHAP | Shapley value attribution | Per-prediction |
| Integrated Gradients | Gradient-based attribution | Per-prediction |
| Attention Weights | Transformer attention visualization | Per-prediction |
| Feature Importance | Global feature ranking | Global |
| Decision Rules | Extract rules from model | Global |

**Python Code Example:**
```python
# Pipeline: Interpretable text classification with LIME
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class InterpretableClassifier:
    def __init__(self, class_names):
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.model = LogisticRegression(max_iter=1000)
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def explain_prediction(self, text, num_features=10):
        exp = self.explainer.explain_instance(
            text, self.predict_proba, num_features=num_features
        )
        return {
            'prediction': self.class_names[np.argmax(self.predict_proba([text]))],
            'top_features': exp.as_list(),
            'html': exp.as_html(),
        }

    def global_feature_importance(self, top_k=20):
        """Get most important features across all classes."""
        feature_names = self.vectorizer.get_feature_names_out()
        importances = np.abs(self.model.coef_).mean(axis=0)
        top_indices = importances.argsort()[-top_k:][::-1]
        return [(feature_names[i], importances[i]) for i in top_indices]
```

**Interview Tips:**
- LIME and SHAP are the standard post-hoc explanation methods for any black-box model
- Attention weights are NOT faithful explanations — they show correlation, not causation
- Discuss that regulated domains (finance, healthcare) may require inherently interpretable models
- Note that knowledge distillation from complex to simple model preserves some accuracy while gaining interpretability

---


---

# --- NLP (from 06_ai_nlp/01_nlp) - theory questions ---

# Nlp Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What isNatural Language Processing (NLP)and why is it important?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Explain the significance ofPart-of-Speech (POS) taggingin NLP.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Describelemmatizationandstemming. When would you use one over the other?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**What is a ‘named entity’ and how isNamed Entity Recognition (NER)useful in NLP tasks?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How does adependency parserwork, and what information does it provide?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What aren-grams, and how do they contribute tolanguage modeling?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Describe what a ‘bag of words’ model is and its limitations.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Explain how theNaive Bayes classifieris used in NLP.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**What are the advantages of usingRandom Forestsin NLP?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Explain howDecision Treesare utilized for NLP problems.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Briefly explainword embeddingsand their importance in NLP.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Describe the architecture and applications ofRecurrent Neural Networks (RNN)in NLP.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**What are the benefits of usingAttention Mechanismsin NLP models?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Explain the concept and capabilities ofTransformer modelslikeBERTandGPT.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Describe theTF-IDFstatistic and its significance in document retrieval.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**What is the idea behindLatent Semantic Analysis (LSA)in NLP?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Describe a typical workflow with theNatural Language Toolkit (NLTK)in Python.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the benefits of using libraries likeHugging Face’s Transformers?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Explain howPyTorchandTensorFlowfacilitate NLP model building.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**What arecontext-free grammars, and how do they apply toparsingin NLP?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**What is the difference betweenrule-based,statistical, andneural approachesin NLP?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Explain howmachine translation modelsare evaluated for accuracy.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Explain the importance of domain-specificcorporaand language resources in NLP.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Describe an approach to automaticallysummarizelong documents.**

**Answer:** _[To be filled]_

---

---
