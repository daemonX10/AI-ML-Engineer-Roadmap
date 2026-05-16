# LLM Applications & Engineering - Theory Questions

## Prompt Engineering Fundamentals

### Question 1
- [ ] Done
**How do you design prompts that consistently elicit desired behaviors across different LLM architectures?**

**Answer:**

**Definition:**
Cross-architecture prompt design: **clear instructions** (explicit task description), **structured format** (consistent layout), **examples** (demonstrate expected behavior), **constraints** (specify what to avoid). Different models interpret prompts differently; robust design minimizes variation.

**Design Principles:**

| Principle | Implementation | Example |
|-----------|---------------|--------|
| **Explicit role** | Define persona clearly | "You are a senior software engineer..." |
| **Task clarity** | State objective first | "Your task is to analyze..." |
| **Output format** | Specify structure | "Return JSON with fields..." |
| **Constraints** | State limitations | "Do not include personal opinions" |
| **Examples** | Demonstrate expected behavior | Provide 1-3 examples |

**Python Code Example:**
```python
class CrossArchitecturePrompt:
    """Design prompts that work across different LLMs"""
    
    def __init__(self):
        self.prompt_template = """
# Role
{role}

# Task
{task}

# Output Format
{output_format}

# Constraints
{constraints}

# Examples
{examples}

# Input
{input}

# Output"""
    
    def build_prompt(self, role: str, task: str, 
                     output_format: str, constraints: list,
                     examples: list, user_input: str) -> str:
        
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        examples_str = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        return self.prompt_template.format(
            role=role,
            task=task,
            output_format=output_format,
            constraints=constraints_str,
            examples=examples_str,
            input=user_input
        )
    
    def adapt_for_model(self, prompt: str, model: str) -> str:
        """Adapt prompt for specific model quirks"""
        
        if 'gpt' in model.lower():
            # OpenAI models like system messages
            return prompt
        
        elif 'claude' in model.lower():
            # Claude prefers clear structure
            return prompt
        
        elif 'llama' in model.lower():
            # Llama may need explicit instruction format
            return f"[INST] {prompt} [/INST]"
        
        elif 'mistral' in model.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        return prompt
```

**Interview Tips:**
- Explicit > implicit instructions
- Put most important info at beginning and end
- Test on multiple models before deploying
- Avoid model-specific tricks for portability

---

### Question 2
- [ ] Done
**Explain chain-of-thought (CoT) prompting and when it significantly improves reasoning performance.**

**Answer:**

**Definition:**
**Chain-of-Thought (CoT)**: prompt the model to show its reasoning step-by-step before giving the final answer. Significantly improves performance on: **math problems**, **multi-step reasoning**, **logical deduction**, **complex analysis**. Less helpful for: simple retrieval, classification.

**CoT Triggers:**

| Trigger | Example |
|---------|--------|
| "Let's think step by step" | Zero-shot CoT |
| "Explain your reasoning" | Request explanation |
| Providing worked examples | Few-shot CoT |
| "First... Then... Finally..." | Structured steps |

**When CoT Helps Most:**

| Task Type | CoT Benefit | Why |
|-----------|------------|-----|
| **Math word problems** | +20-40% | Intermediate calculations |
| **Multi-hop reasoning** | +15-30% | Connect multiple facts |
| **Logical puzzles** | +25-40% | Explicit deduction |
| **Code debugging** | +10-20% | Trace execution |
| **Simple QA** | ~0% | No reasoning needed |

**Python Code Example:**
```python
class ChainOfThought:
    def __init__(self, llm):
        self.llm = llm
    
    def zero_shot_cot(self, question: str) -> str:
        """Zero-shot chain of thought"""
        prompt = f"""Question: {question}

Let's think through this step by step:"""
        
        return self.llm.generate(prompt)
    
    def few_shot_cot(self, question: str, examples: list) -> str:
        """Few-shot with reasoning examples"""
        examples_text = ""
        for ex in examples:
            examples_text += f"""Question: {ex['question']}

Reasoning:
{ex['reasoning']}

Answer: {ex['answer']}

---

"""
        
        prompt = f"""{examples_text}Question: {question}

Reasoning:"""
        
        return self.llm.generate(prompt)
    
    def structured_cot(self, question: str, 
                       step_names: list = None) -> str:
        """Structured step-by-step reasoning"""
        if step_names is None:
            step_names = ["Understand the problem", 
                         "Identify key information",
                         "Apply relevant methods",
                         "Calculate/Derive",
                         "State final answer"]
        
        steps_template = "\n".join(
            f"Step {i+1} - {name}: " 
            for i, name in enumerate(step_names)
        )
        
        prompt = f"""Question: {question}

Solve this problem by following these steps:

{steps_template}"""
        
        return self.llm.generate(prompt)
    
    def extract_answer(self, cot_response: str) -> str:
        """Extract final answer from CoT response"""
        # Look for common answer patterns
        patterns = [
            "the answer is", "therefore", "final answer:",
            "in conclusion", "answer:"
        ]
        
        response_lower = cot_response.lower()
        for pattern in patterns:
            if pattern in response_lower:
                idx = response_lower.rfind(pattern)
                return cot_response[idx:].strip()
        
        # Return last sentence if no pattern found
        sentences = cot_response.split('.')
        return sentences[-1].strip() if sentences else cot_response
```

**Interview Tips:**
- "Let's think step by step" improves GSM8K math by ~40%
- CoT works better on larger models (>10B params)
- Can combine with self-consistency (multiple reasoning paths)
- Useful for debugging: see WHERE model makes errors

---

### Question 3
- [ ] Done
**How do you implement few-shot prompting with optimal example selection strategies?**

**Answer:**

**Definition:**
**Few-shot prompting**: provide examples in the prompt to guide model behavior. **Optimal selection**: choose examples that are **diverse** (cover edge cases), **similar** (to current query), **representative** (of target distribution), and **correctly labeled**.

**Selection Strategies:**

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Random** | Random sample | Baseline |
| **Semantic similarity** | Nearest to query | Query-specific tasks |
| **Diversity sampling** | Cover example space | General coverage |
| **Difficulty-based** | Include edge cases | Robust performance |
| **Stratified** | Balance by category | Classification |

**Python Code Example:**
```python
import numpy as np
from typing import List, Tuple

class FewShotSelector:
    def __init__(self, examples: List[dict], embedder):
        """examples: [{'input': ..., 'output': ..., 'category': ...}]"""
        self.examples = examples
        self.embedder = embedder
        
        # Pre-compute embeddings
        self.embeddings = np.array([
            embedder.embed(ex['input']) for ex in examples
        ])
    
    def select_random(self, k: int) -> List[dict]:
        """Random selection"""
        indices = np.random.choice(len(self.examples), k, replace=False)
        return [self.examples[i] for i in indices]
    
    def select_similar(self, query: str, k: int) -> List[dict]:
        """Select k most similar to query"""
        query_emb = self.embedder.embed(query)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * 
            np.linalg.norm(query_emb) + 1e-9
        )
        
        # Get top k
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k]
    
    def select_diverse(self, k: int) -> List[dict]:
        """Select k diverse examples using max-marginal relevance"""
        selected = []
        remaining = list(range(len(self.examples)))
        
        # Start with random
        first = np.random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        while len(selected) < k and remaining:
            # Find most different from selected
            min_max_sim = float('inf')
            best_idx = None
            
            for idx in remaining:
                max_sim = max(
                    np.dot(self.embeddings[idx], self.embeddings[s]) / (
                        np.linalg.norm(self.embeddings[idx]) *
                        np.linalg.norm(self.embeddings[s]) + 1e-9
                    )
                    for s in selected
                )
                if max_sim < min_max_sim:
                    min_max_sim = max_sim
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [self.examples[i] for i in selected]
    
    def select_stratified(self, k: int) -> List[dict]:
        """Select balanced by category"""
        by_category = {}
        for i, ex in enumerate(self.examples):
            cat = ex.get('category', 'default')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(i)
        
        selected = []
        categories = list(by_category.keys())
        per_category = max(1, k // len(categories))
        
        for cat in categories:
            indices = by_category[cat]
            n = min(per_category, len(indices))
            selected.extend(np.random.choice(indices, n, replace=False))
        
        return [self.examples[i] for i in selected[:k]]
    
    def select_hybrid(self, query: str, k: int,
                      similarity_weight: float = 0.5) -> List[dict]:
        """Balance similarity and diversity"""
        similar = self.select_similar(query, k * 2)
        
        # Then select diverse subset
        if len(similar) <= k:
            return similar
        
        # MMR within similar set
        embeddings_subset = np.array([
            self.embedder.embed(ex['input']) for ex in similar
        ])
        query_emb = self.embedder.embed(query)
        
        selected = [0]  # Start with most similar
        remaining = list(range(1, len(similar)))
        
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                sim_to_query = np.dot(embeddings_subset[idx], query_emb)
                max_sim_to_selected = max(
                    np.dot(embeddings_subset[idx], embeddings_subset[s])
                    for s in selected
                )
                
                score = (similarity_weight * sim_to_query - 
                        (1 - similarity_weight) * max_sim_to_selected)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [similar[i] for i in selected]

def build_few_shot_prompt(examples: List[dict], query: str) -> str:
    """Build prompt from selected examples"""
    prompt = ""
    for ex in examples:
        prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt
```

**Interview Tips:**
- Semantic similarity is most effective for query-dependent tasks
- 3-5 examples usually sufficient; more can cause context overflow
- Order matters: put most relevant examples closer to query
- Include edge cases for robustness

---

### Question 4
- [ ] Done
**What is zero-shot vs few-shot vs many-shot prompting and when to use each approach?**

**Answer:**

**Definition:**
- **Zero-shot**: no examples, just instructions
- **Few-shot**: 1-10 examples in prompt
- **Many-shot**: 10-100+ examples (requires large context)

Trade-off: more examples = better task understanding but uses more tokens and context.

**Comparison:**

| Approach | Examples | Token Cost | Best For |
|----------|----------|------------|----------|
| **Zero-shot** | 0 | Lowest | Simple, well-understood tasks |
| **One-shot** | 1 | Low | Format demonstration |
| **Few-shot** | 3-5 | Medium | Most tasks |
| **Many-shot** | 10-100 | High | Complex, nuanced tasks |

**When to Use Each:**

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Simple classification | Zero-shot | Model knows task |
| Custom format | One-shot | Show format once |
| Domain-specific | Few-shot | Demonstrate domain |
| Subtle distinctions | Many-shot | Show edge cases |
| Low latency required | Zero/One-shot | Fewer tokens |

**Python Code Example:**
```python
class PromptingStrategies:
    def __init__(self, llm):
        self.llm = llm
    
    def zero_shot(self, task: str, input_text: str) -> str:
        """No examples - rely on instructions only"""
        prompt = f"""Task: {task}

Input: {input_text}

Output:"""
        return self.llm.generate(prompt)
    
    def one_shot(self, task: str, example: dict, 
                 input_text: str) -> str:
        """Single example to demonstrate format"""
        prompt = f"""Task: {task}

Example:
Input: {example['input']}
Output: {example['output']}

Now process:
Input: {input_text}
Output:"""
        return self.llm.generate(prompt)
    
    def few_shot(self, task: str, examples: list,
                 input_text: str) -> str:
        """3-5 examples for task understanding"""
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        prompt = f"""Task: {task}

Examples:
{examples_text}

Now process:
Input: {input_text}
Output:"""
        return self.llm.generate(prompt)
    
    def many_shot(self, task: str, examples: list,
                  input_text: str, max_examples: int = 50) -> str:
        """Many examples for complex tasks (requires large context)"""
        # Truncate to fit context window
        examples = examples[:max_examples]
        
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        prompt = f"""Task: {task}

Here are {len(examples)} examples:

{examples_text}

Now process:
Input: {input_text}
Output:"""
        return self.llm.generate(prompt)
    
    def adaptive_shot(self, task: str, examples: list,
                      input_text: str, max_tokens: int = 4000) -> str:
        """Adapt number of examples to fit context"""
        
        base_prompt = f"Task: {task}\n\nInput: {input_text}\nOutput:"
        base_tokens = len(base_prompt.split())  # Rough estimate
        
        selected_examples = []
        tokens_used = base_tokens
        
        for ex in examples:
            ex_text = f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
            ex_tokens = len(ex_text.split())
            
            if tokens_used + ex_tokens > max_tokens * 0.8:  # 80% budget
                break
            
            selected_examples.append(ex)
            tokens_used += ex_tokens
        
        if selected_examples:
            return self.few_shot(task, selected_examples, input_text)
        return self.zero_shot(task, input_text)
```

**Performance Scaling:**
```
Accuracy
   ^
   |              ______ many-shot plateau
   |         ____/
   |    ____/  few-shot
   | __/
   |/  zero-shot
   +-------------------> # Examples
```

**Interview Tips:**
- Zero-shot good for GPT-4 class models on common tasks
- Few-shot critical for custom/domain tasks
- Many-shot shows diminishing returns after ~30 examples
- Consider cost: 100 examples = 100x base tokens

---

### Question 5
- [ ] Done
**What approaches work best for reducing prompt sensitivity and improving output robustness?**

**Answer:**

**Definition:**
**Prompt sensitivity**: small prompt changes cause large output changes. Reduce via: **explicit instructions** (less ambiguity), **structured formats** (constrain outputs), **multiple prompt versions** (ensemble), **temperature tuning**, **output validation**, **guardrails**.

**Robustness Techniques:**

| Technique | How It Helps | Implementation |
|-----------|--------------|----------------|
| **Clear structure** | Reduces ambiguity | Use delimiters, sections |
| **Output schema** | Constrains format | JSON schema, templates |
| **Prompt ensembling** | Reduces variance | Average multiple prompts |
| **Low temperature** | More deterministic | temp=0 or 0.1 |
| **Validation** | Catch errors | Parse and verify |
| **Fallback prompts** | Handle failures | Retry with different prompt |

**Python Code Example:**
```python
import json
import re
from typing import Optional, List

class RobustPrompting:
    def __init__(self, llm):
        self.llm = llm
    
    def structured_prompt(self, task: str, input_text: str,
                          output_schema: dict) -> str:
        """Use explicit structure to reduce sensitivity"""
        schema_desc = json.dumps(output_schema, indent=2)
        
        prompt = f"""### Task
{task}

### Required Output Format
Return a JSON object matching this schema:
```json
{schema_desc}
```

### Input
{input_text}

### Output (JSON only, no other text)
```json"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        # Extract JSON from response
        return self._extract_json(response)
    
    def _extract_json(self, response: str) -> Optional[dict]:
        """Robustly extract JSON from response"""
        # Try direct parse
        try:
            return json.loads(response)
        except:
            pass
        
        # Try extracting from code blocks
        json_match = re.search(r'```json?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try finding JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        return None
    
    def ensemble_prompts(self, prompts: List[str], 
                         aggregation: str = 'majority') -> str:
        """Use multiple prompt versions, aggregate results"""
        responses = []
        
        for prompt in prompts:
            response = self.llm.generate(prompt, temperature=0.3)
            responses.append(response)
        
        if aggregation == 'majority':
            # Simple majority vote
            from collections import Counter
            return Counter(responses).most_common(1)[0][0]
        
        elif aggregation == 'longest':
            # Return most detailed response
            return max(responses, key=len)
        
        return responses[0]
    
    def prompt_with_validation(self, prompt: str, 
                                validator, 
                                max_retries: int = 3) -> Optional[str]:
        """Retry with validation until valid output"""
        
        for attempt in range(max_retries):
            temperature = 0.1 * attempt  # Increase temp on retry
            response = self.llm.generate(prompt, temperature=temperature)
            
            if validator(response):
                return response
            
            # Add correction hint on retry
            if attempt < max_retries - 1:
                prompt = f"{prompt}\n\n(Previous attempt was invalid. Please ensure correct format.)"
        
        return None
    
    def self_consistency(self, prompt: str, n_samples: int = 5) -> str:
        """Generate multiple samples, return most consistent"""
        responses = [
            self.llm.generate(prompt, temperature=0.7)
            for _ in range(n_samples)
        ]
        
        # Find answer that appears most
        from collections import Counter
        
        # Extract final answers (simplified)
        answers = [r.split('.')[-1].strip() for r in responses]
        most_common = Counter(answers).most_common(1)[0][0]
        
        # Return full response with most common answer
        for r in responses:
            if most_common in r:
                return r
        
        return responses[0]

def create_robust_prompt(task: str) -> str:
    """Create prompt with robustness features"""
    return f"""<task>
{task}
</task>

<instructions>
1. Read the task carefully
2. Think through your approach
3. Provide a clear, structured response
4. Use the exact format specified
</instructions>

<output_format>
Respond with ONLY the requested output, no explanations.
</output_format>"""
```

**Interview Tips:**
- Temperature 0 for deterministic tasks
- Self-consistency improves reasoning by ~10-15%
- Always validate outputs programmatically
- Use delimiters (<task>, ###) to separate sections

---

### Question 6
- [ ] Done
**How do you design prompts that minimize harmful, biased, or hallucinated outputs?**

**Answer:**

**Definition:**
Reduce harmful outputs via: **explicit safety constraints**, **factuality grounding** (cite sources), **uncertainty acknowledgment** ("I don't know"), **input validation**, **output filtering**, and **Constitutional AI** principles. Layer multiple defenses.

**Defense Layers:**

| Layer | Technique | What It Catches |
|-------|-----------|----------------|
| **Input** | Content filter | Harmful queries |
| **Prompt** | Safety instructions | Unsafe responses |
| **Grounding** | Source citation | Hallucination |
| **Output** | Post-processing filter | Slipped content |
| **Human** | Review pipeline | Edge cases |

**Python Code Example:**
```python
import re
from typing import Optional

class SafePrompting:
    def __init__(self, llm, content_filter=None):
        self.llm = llm
        self.content_filter = content_filter
        
        self.safety_prompt = """IMPORTANT GUIDELINES:
- Provide factual, evidence-based information only
- If uncertain, explicitly say "I'm not certain" or "I don't have enough information"
- Do not make up facts, statistics, or sources
- Avoid harmful, biased, or discriminatory content
- If a request seems harmful, politely decline
- Base answers on provided context when available"""
    
    def build_safe_prompt(self, task: str, context: str = None,
                          require_citations: bool = False) -> str:
        """Build prompt with safety guardrails"""
        
        prompt = f"""{self.safety_prompt}

### Task
{task}"""
        
        if context:
            prompt += f"""\n\n### Reference Information
Use ONLY the following information to answer. If the answer is not in the reference, say so.

{context}"""
        
        if require_citations:
            prompt += """\n\n### Citation Requirement
For each claim, cite the source using [Source: ...].
If you cannot cite a source, indicate the claim is your general knowledge and may need verification."""
        
        prompt += "\n\n### Response"
        return prompt
    
    def detect_hallucination_risk(self, response: str,
                                   context: str = None) -> dict:
        """Assess hallucination risk in response"""
        risk_factors = []
        
        # Check for specific numbers/dates without context
        numbers = re.findall(r'\b\d{4,}\b|\$[\d,]+', response)
        if numbers and (not context or 
                       not any(n in context for n in numbers)):
            risk_factors.append('ungrounded_numbers')
        
        # Check for strong claims without hedging
        strong_claims = ['definitely', 'certainly', 'always', 'never',
                        'proven fact', 'studies show']
        for claim in strong_claims:
            if claim.lower() in response.lower():
                risk_factors.append(f'strong_claim: {claim}')
        
        # Check for uncertainty acknowledgment (good sign)
        uncertainty_terms = ["not certain", "may", "might", 
                            "I don't know", "approximately"]
        has_uncertainty = any(t in response.lower() 
                             for t in uncertainty_terms)
        
        return {
            'risk_level': 'high' if len(risk_factors) > 2 else 
                         'medium' if risk_factors else 'low',
            'risk_factors': risk_factors,
            'acknowledges_uncertainty': has_uncertainty
        }
    
    def safe_generate(self, task: str, context: str = None) -> dict:
        """Generate with full safety pipeline"""
        
        # Input validation
        if self.content_filter:
            if not self.content_filter.is_safe(task):
                return {
                    'response': None,
                    'blocked': True,
                    'reason': 'Input flagged as unsafe'
                }
        
        # Build safe prompt
        prompt = self.build_safe_prompt(task, context)
        
        # Generate
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Assess hallucination risk
        risk = self.detect_hallucination_risk(response, context)
        
        # Output validation
        if self.content_filter:
            if not self.content_filter.is_safe(response):
                return {
                    'response': None,
                    'blocked': True,
                    'reason': 'Output flagged as unsafe'
                }
        
        return {
            'response': response,
            'blocked': False,
            'hallucination_risk': risk
        }

def constitutional_ai_prompt(task: str) -> str:
    """Apply Constitutional AI principles"""
    return f"""You are a helpful, harmless, and honest assistant.

Principles:
1. Helpfulness: Provide useful, accurate information
2. Harmlessness: Avoid harmful, illegal, or unethical content
3. Honesty: Be truthful, acknowledge limitations

If there's tension between these principles, prioritize: Harmlessness > Honesty > Helpfulness

Task: {task}

Response:"""
```

**Interview Tips:**
- Ground responses in provided context (RAG)
- Encourage "I don't know" responses
- Multiple defense layers (input + prompt + output)
- Constitutional AI: explicitly state principles

---

### Question 7
- [ ] Done
**What techniques help with measuring and evaluating prompt effectiveness quantitatively?**

**Answer:**

**Definition:**
Evaluate prompts via: **task-specific metrics** (accuracy, F1), **output quality** (coherence, relevance), **efficiency** (tokens, latency), **robustness** (variance across runs), and **user satisfaction**. Use both automated metrics and human evaluation.

**Evaluation Dimensions:**

| Dimension | Metrics | Measurement |
|-----------|---------|-------------|
| **Correctness** | Accuracy, F1, Exact Match | Automated |
| **Quality** | Coherence, fluency, relevance | Human/LLM |
| **Efficiency** | Tokens used, latency | Automated |
| **Robustness** | Variance, consistency | Automated |
| **Safety** | Toxicity, bias scores | Automated |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import time

@dataclass
class PromptEvaluation:
    prompt_name: str
    accuracy: float
    avg_tokens: float
    avg_latency_ms: float
    consistency: float  # % same answer across runs
    quality_score: float  # 1-5 scale

class PromptEvaluator:
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_prompt(self, prompt_template: str,
                        test_cases: List[dict],
                        n_runs: int = 3) -> PromptEvaluation:
        """Comprehensive prompt evaluation"""
        
        results = []
        latencies = []
        tokens_used = []
        
        for case in test_cases:
            prompt = prompt_template.format(**case['inputs'])
            case_responses = []
            
            for _ in range(n_runs):
                start = time.perf_counter()
                response = self.llm.generate(prompt)
                latency = (time.perf_counter() - start) * 1000
                
                case_responses.append(response)
                latencies.append(latency)
                tokens_used.append(len(response.split()))  # Approximate
            
            # Check correctness (first run)
            is_correct = self._check_correctness(
                case_responses[0], case['expected']
            )
            results.append({
                'correct': is_correct,
                'responses': case_responses,
                'consistency': self._measure_consistency(case_responses)
            })
        
        return PromptEvaluation(
            prompt_name=prompt_template[:50],
            accuracy=np.mean([r['correct'] for r in results]),
            avg_tokens=np.mean(tokens_used),
            avg_latency_ms=np.mean(latencies),
            consistency=np.mean([r['consistency'] for r in results]),
            quality_score=0  # Set by human evaluation
        )
    
    def _check_correctness(self, response: str, expected: str) -> bool:
        """Check if response matches expected (flexible)"""
        response_clean = response.lower().strip()
        expected_clean = expected.lower().strip()
        
        # Exact match
        if expected_clean in response_clean:
            return True
        
        # Contains key answer
        return expected_clean in response_clean
    
    def _measure_consistency(self, responses: List[str]) -> float:
        """Measure consistency across multiple runs"""
        if len(responses) <= 1:
            return 1.0
        
        # Simple: check if all responses match
        normalized = [r.lower().strip() for r in responses]
        unique = len(set(normalized))
        
        return 1.0 / unique
    
    def compare_prompts(self, prompts: Dict[str, str],
                        test_cases: List[dict]) -> Dict[str, PromptEvaluation]:
        """Compare multiple prompt versions"""
        results = {}
        
        for name, template in prompts.items():
            results[name] = self.evaluate_prompt(template, test_cases)
        
        # Rank by accuracy then efficiency
        ranked = sorted(
            results.items(),
            key=lambda x: (x[1].accuracy, -x[1].avg_tokens),
            reverse=True
        )
        
        return {name: eval for name, eval in ranked}
    
    def llm_as_judge(self, response: str, criteria: List[str]) -> dict:
        """Use LLM to evaluate response quality"""
        
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        
        eval_prompt = f"""Evaluate the following response on these criteria:
{criteria_text}

Response to evaluate:
{response}

For each criterion, score 1-5 (5=excellent) and explain briefly.
Return as JSON: {{"criterion": {{"score": N, "explanation": "..."}}, ...}}"""
        
        eval_response = self.llm.generate(eval_prompt)
        
        try:
            import json
            return json.loads(eval_response)
        except:
            return {'error': 'Failed to parse evaluation'}

# Usage example
def create_evaluation_suite():
    test_cases = [
        {
            'inputs': {'question': 'What is 2+2?'},
            'expected': '4'
        },
        {
            'inputs': {'question': 'Capital of France?'},
            'expected': 'Paris'
        }
    ]
    
    prompts = {
        'simple': 'Answer: {question}',
        'detailed': 'Question: {question}\nProvide a clear answer:',
        'cot': 'Question: {question}\nLet\'s think step by step:'
    }
    
    return test_cases, prompts
```

**Interview Tips:**
- Always measure on held-out test set
- Consistency (low variance) often as important as accuracy
- LLM-as-judge correlates well with human evaluation
- Track token usage for cost optimization

---

## Advanced Prompting Techniques

### Question 8
- [ ] Done
**Explain self-consistency prompting and how it improves reasoning reliability.**

**Answer:**

**Definition:**
**Self-consistency**: generate multiple reasoning paths (with temperature > 0), then select the most common final answer via majority voting. Improves reasoning by ~10-20% on math/logic tasks by reducing sensitivity to any single reasoning path.

**How It Works:**
```
Query → [CoT Path 1] → Answer A
      → [CoT Path 2] → Answer B
      → [CoT Path 3] → Answer A
      → [CoT Path 4] → Answer A
      → [CoT Path 5] → Answer C
                           ↓
                  Majority Vote → Answer A (3/5)
```

**Python Code Example:**
```python
import numpy as np
from collections import Counter
from typing import List, Tuple
import re

class SelfConsistency:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_with_consistency(self, prompt: str,
                                   n_samples: int = 5,
                                   temperature: float = 0.7) -> dict:
        """Generate multiple samples and vote"""
        
        # Generate multiple reasoning paths
        responses = []
        for _ in range(n_samples):
            response = self.llm.generate(prompt, temperature=temperature)
            responses.append(response)
        
        # Extract final answers
        answers = [self._extract_answer(r) for r in responses]
        
        # Majority vote
        answer_counts = Counter(answers)
        best_answer, count = answer_counts.most_common(1)[0]
        
        # Find a response with the best answer (for explanation)
        best_response = None
        for r, a in zip(responses, answers):
            if a == best_answer:
                best_response = r
                break
        
        return {
            'answer': best_answer,
            'confidence': count / n_samples,
            'full_response': best_response,
            'all_answers': dict(answer_counts)
        }
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        
        # Look for explicit answer markers
        patterns = [
            r'(?:the answer is|answer:|therefore)[:\s]*([^.\n]+)',
            r'(?:final answer)[:\s]*([^.\n]+)',
            r'(?:=\s*)([\d.]+)',  # For math
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1).strip()
        
        # Fall back to last sentence
        sentences = response.split('.')
        if sentences:
            return sentences[-1].strip()
        
        return response.strip()
    
    def weighted_consistency(self, prompt: str,
                             n_samples: int = 5) -> dict:
        """Weight answers by confidence signals"""
        
        responses = []
        for _ in range(n_samples):
            response = self.llm.generate(prompt, temperature=0.7)
            responses.append(response)
        
        # Score each response
        scored_answers = []
        for response in responses:
            answer = self._extract_answer(response)
            confidence = self._estimate_confidence(response)
            scored_answers.append((answer, confidence))
        
        # Weighted voting
        answer_scores = {}
        for answer, conf in scored_answers:
            answer_scores[answer] = answer_scores.get(answer, 0) + conf
        
        best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'answer': best_answer,
            'answer_scores': answer_scores
        }
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence from response text"""
        
        # High confidence signals
        high_conf = ['definitely', 'certainly', 'clearly', 
                     'the answer is', 'therefore']
        
        # Low confidence signals
        low_conf = ['might be', 'possibly', 'not sure',
                    'approximately', 'around']
        
        response_lower = response.lower()
        
        high_count = sum(1 for t in high_conf if t in response_lower)
        low_count = sum(1 for t in low_conf if t in response_lower)
        
        # Base confidence + adjustments
        confidence = 1.0 + 0.1 * high_count - 0.1 * low_count
        return max(0.5, min(1.5, confidence))
    
    def progressive_consistency(self, prompt: str,
                                 max_samples: int = 10,
                                 threshold: float = 0.8) -> dict:
        """Stop early if confident enough"""
        
        answers = []
        
        for i in range(max_samples):
            response = self.llm.generate(prompt, temperature=0.7)
            answer = self._extract_answer(response)
            answers.append(answer)
            
            # Check if we have enough consistency
            counts = Counter(answers)
            top_answer, top_count = counts.most_common(1)[0]
            
            # Need at least 3 samples and high agreement
            if len(answers) >= 3:
                agreement = top_count / len(answers)
                if agreement >= threshold:
                    return {
                        'answer': top_answer,
                        'confidence': agreement,
                        'samples_used': len(answers),
                        'early_stop': True
                    }
        
        counts = Counter(answers)
        top_answer, top_count = counts.most_common(1)[0]
        
        return {
            'answer': top_answer,
            'confidence': top_count / len(answers),
            'samples_used': len(answers),
            'early_stop': False
        }
```

**When Self-Consistency Helps:**

| Task | Improvement | Why |
|------|-------------|-----|
| Math reasoning | +15-20% | Multiple calculation paths |
| Logic puzzles | +10-15% | Different reasoning chains |
| Commonsense | +5-10% | Diverse perspectives |
| Simple QA | ~0% | Already deterministic |

**Interview Tips:**
- Requires temperature > 0 for diverse samples
- 5-10 samples usually sufficient
- Can combine with CoT for best results
- Trade-off: n samples = n × cost and latency

---

### Question 9
- [ ] Done
**What is Tree-of-Thoughts (ToT) prompting and when does it outperform linear CoT?**

**Answer:**

**Definition:**
**Tree-of-Thoughts (ToT)**: explore multiple reasoning branches, evaluate each, backtrack from dead ends, and explore alternatives. Unlike linear CoT, ToT allows deliberate **search and backtracking**. Best for: puzzles, planning, creative tasks requiring exploration.

**ToT vs CoT:**

| Aspect | Chain-of-Thought | Tree-of-Thoughts |
|--------|-----------------|------------------|
| **Structure** | Linear sequence | Branching tree |
| **Backtracking** | No | Yes |
| **Exploration** | Single path | Multiple paths |
| **Cost** | 1x | 10-100x |
| **Best for** | Step-by-step problems | Search/planning |

**ToT Process:**
```
           [Start]
          /       \
     [Step 1a]  [Step 1b]
      /    \        \
  [2a]    [2b]     [2c] ← evaluate, prune
    |       ×        |
  [3a]            [3c]
    ↓
 [Solution]
```

**Python Code Example:**
```python
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ThoughtNode:
    thought: str
    parent: Optional['ThoughtNode']
    score: float
    depth: int

class TreeOfThoughts:
    def __init__(self, llm):
        self.llm = llm
    
    def solve(self, problem: str, 
              max_depth: int = 5,
              branching_factor: int = 3,
              beam_width: int = 2) -> str:
        """Solve using tree-of-thoughts search"""
        
        # Initialize with root
        root = ThoughtNode(
            thought=f"Problem: {problem}",
            parent=None,
            score=0,
            depth=0
        )
        
        # Beam search through thought tree
        current_beam = [root]
        
        for depth in range(max_depth):
            all_children = []
            
            for node in current_beam:
                # Generate child thoughts
                children = self._generate_thoughts(
                    node, branching_factor
                )
                
                # Evaluate each child
                for child in children:
                    child.score = self._evaluate_thought(child, problem)
                    all_children.append(child)
            
            if not all_children:
                break
            
            # Check for solution
            for child in all_children:
                if self._is_solution(child):
                    return self._extract_solution(child)
            
            # Keep top beam_width thoughts
            all_children.sort(key=lambda x: x.score, reverse=True)
            current_beam = all_children[:beam_width]
        
        # Return best path found
        if current_beam:
            return self._extract_solution(current_beam[0])
        return "No solution found"
    
    def _generate_thoughts(self, node: ThoughtNode,
                           n: int) -> List[ThoughtNode]:
        """Generate n possible next thoughts"""
        
        # Build context from path
        path = self._get_path(node)
        context = "\n".join([n.thought for n in path])
        
        prompt = f"""{context}

Generate {n} different possible next steps to continue solving this problem.
For each step, explain the reasoning briefly.

Format:
Step 1: [thought]
Step 2: [thought]
Step 3: [thought]"""
        
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Parse thoughts
        children = []
        for line in response.split('\n'):
            if line.strip().startswith('Step'):
                thought = line.split(':', 1)[-1].strip()
                children.append(ThoughtNode(
                    thought=thought,
                    parent=node,
                    score=0,
                    depth=node.depth + 1
                ))
        
        return children[:n]
    
    def _evaluate_thought(self, node: ThoughtNode, 
                          problem: str) -> float:
        """Evaluate how promising a thought path is"""
        
        path = self._get_path(node)
        path_text = "\n".join([n.thought for n in path])
        
        prompt = f"""Problem: {problem}

Current reasoning path:
{path_text}

Evaluate this reasoning path on a scale of 1-10:
- Is the reasoning logical and correct?
- Is it making progress toward a solution?
- Are there any errors or dead ends?

Score (just the number):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        try:
            return float(response.strip().split()[0])
        except:
            return 5.0  # Default middle score
    
    def _is_solution(self, node: ThoughtNode) -> bool:
        """Check if node represents a complete solution"""
        keywords = ['therefore', 'answer is', 'solution:', 'final answer']
        return any(kw in node.thought.lower() for kw in keywords)
    
    def _get_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Get path from root to node"""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]
    
    def _extract_solution(self, node: ThoughtNode) -> str:
        """Extract solution from path"""
        path = self._get_path(node)
        return "\n".join([f"Step {i+1}: {n.thought}" 
                         for i, n in enumerate(path)])
```

**When ToT Outperforms CoT:**

| Task | ToT Advantage | Example |
|------|--------------|--------|
| **Game of 24** | +50% | Explore number combinations |
| **Creative writing** | +30% | Explore narrative branches |
| **Planning** | +40% | Backtrack from dead ends |
| **Puzzles** | +35% | Try multiple approaches |

**Interview Tips:**
- ToT is much more expensive (10-100x CoT)
- Use for search/planning problems, not factual QA
- Beam search balances exploration vs cost
- Evaluation function is critical for pruning

---

### Question 10
- [ ] Done
**How do you implement meta-prompting strategies for complex multi-step tasks?**

**Answer:**

**Definition:**
**Meta-prompting**: use prompts to generate or refine prompts. Applications: **decompose complex tasks** into sub-tasks, **self-refine** outputs, **orchestrate** multi-step workflows, **adapt** prompts dynamically. Key pattern: LLM as controller/planner.

**Meta-Prompting Patterns:**

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Decomposer** | Break task into steps | Complex problems |
| **Refiner** | Iteratively improve output | Quality improvement |
| **Orchestrator** | Route to specialized prompts | Multi-domain tasks |
| **Critic** | Evaluate and provide feedback | Self-improvement |

**Python Code Example:**
```python
from typing import List, Dict
import json

class MetaPrompting:
    def __init__(self, llm):
        self.llm = llm
    
    def decompose_task(self, complex_task: str) -> List[dict]:
        """Use LLM to break down complex task into steps"""
        
        prompt = f"""You are a task planner. Break down this complex task into smaller, manageable steps.

Complex Task: {complex_task}

Return a JSON array of steps:
[
  {{"step": 1, "task": "...", "dependencies": []}},
  {{"step": 2, "task": "...", "dependencies": [1]}},
  ...
]

Each step should be atomic and clearly defined.
Specify dependencies (which steps must complete first).

Steps (JSON only):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        try:
            return json.loads(response)
        except:
            # Fallback: single step
            return [{"step": 1, "task": complex_task, "dependencies": []}]
    
    def execute_decomposed(self, complex_task: str) -> str:
        """Decompose and execute step by step"""
        
        steps = self.decompose_task(complex_task)
        results = {}
        
        for step in steps:
            # Build context from dependencies
            context = ""
            for dep in step.get('dependencies', []):
                if dep in results:
                    context += f"\nResult of step {dep}: {results[dep]}"
            
            prompt = f"""Task: {step['task']}
{context}

Complete this step:"""
            
            result = self.llm.generate(prompt)
            results[step['step']] = result
        
        # Combine results
        return "\n\n".join([
            f"Step {s}: {r}" for s, r in sorted(results.items())
        ])
    
    def self_refine(self, task: str, max_iterations: int = 3) -> str:
        """Iteratively refine output using self-critique"""
        
        # Initial attempt
        response = self.llm.generate(f"Task: {task}\n\nResponse:")
        
        for i in range(max_iterations):
            # Critique
            critique_prompt = f"""Task: {task}

Current Response:
{response}

Critique this response:
1. What is good about it?
2. What could be improved?
3. Are there any errors or omissions?

Provide specific, actionable feedback:"""
            
            critique = self.llm.generate(critique_prompt)
            
            # Check if good enough
            if "no improvements needed" in critique.lower() or \
               "looks good" in critique.lower():
                break
            
            # Refine based on critique
            refine_prompt = f"""Task: {task}

Previous Response:
{response}

Feedback:
{critique}

Improved Response (address all feedback):"""
            
            response = self.llm.generate(refine_prompt)
        
        return response
    
    def orchestrate(self, task: str, 
                    specialists: Dict[str, str]) -> str:
        """Route to specialized prompts based on task type"""
        
        # Determine task type
        router_prompt = f"""Analyze this task and determine its type:

Task: {task}

Available specialists: {list(specialists.keys())}

Return the most appropriate specialist name (one word):"""
        
        specialist_name = self.llm.generate(
            router_prompt, temperature=0
        ).strip().lower()
        
        # Get specialist prompt or use default
        specialist_prompt = specialists.get(
            specialist_name,
            specialists.get('default', 'Answer the following: {task}')
        )
        
        # Execute with specialist
        full_prompt = specialist_prompt.format(task=task)
        return self.llm.generate(full_prompt)
    
    def generate_prompt(self, task_description: str,
                        examples: List[dict] = None) -> str:
        """Meta: use LLM to generate an optimal prompt"""
        
        examples_text = ""
        if examples:
            examples_text = "\n\nExample input/output pairs:\n"
            for ex in examples:
                examples_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
        
        meta_prompt = f"""You are a prompt engineering expert. Create an optimal prompt for this task:

Task Description: {task_description}
{examples_text}

Create a prompt that:
1. Clearly defines the role and context
2. Specifies the exact output format
3. Includes constraints and guidelines
4. Uses best practices (delimiters, structure)

Generated Prompt:"""
        
        return self.llm.generate(meta_prompt)

# Example usage
def create_specialists():
    return {
        'code': """You are an expert programmer. Write clean, efficient code.

Task: {task}

Code:""",
        'analysis': """You are a data analyst. Provide thorough analysis.

Task: {task}

Analysis:""",
        'creative': """You are a creative writer. Be imaginative and engaging.

Task: {task}

Response:""",
        'default': """Complete the following task:

{task}

Response:"""
    }
```

**Interview Tips:**
- Meta-prompting enables complex agentic workflows
- Self-refinement typically improves quality 10-20%
- Decomposition helps with context length limits
- Cost scales with iterations (plan accordingly)

---

### Question 11
- [ ] Done
**How do you design prompts for specific output formats (JSON, code, structured data)?**

**Answer:**

**Definition:**
Structured output design: **explicit schema** (show exact format), **delimiters** (clear boundaries), **examples** (demonstrate format), **validation instructions** (check before output), **parsing hints** ("only JSON, no explanation"). Different formats need different strategies.

**Format Strategies:**

| Format | Strategy | Key Instruction |
|--------|----------|----------------|
| **JSON** | Schema + example | "Return ONLY valid JSON" |
| **Code** | Language + style | "No explanations in code" |
| **Table** | Header + alignment | "Use markdown table" |
| **List** | Numbered vs bullets | "Return as numbered list" |
| **XML** | Tags + nesting | "Use proper XML syntax" |

**Python Code Example:**
```python
import json
import re
from typing import Optional, Any

class StructuredOutputPrompts:
    def __init__(self, llm):
        self.llm = llm
    
    def json_output(self, task: str, schema: dict,
                    example: dict = None) -> Optional[dict]:
        """Get JSON output matching schema"""
        
        schema_str = json.dumps(schema, indent=2)
        example_str = ""
        if example:
            example_str = f"\n\nExample output:\n```json\n{json.dumps(example, indent=2)}\n```"
        
        prompt = f"""Task: {task}

Required JSON schema:
```json
{schema_str}
```
{example_str}

Return ONLY a valid JSON object matching the schema.
Do not include any explanation, markdown, or extra text.
Output:"""
        
        response = self.llm.generate(prompt, temperature=0)
        return self._parse_json(response)
    
    def _parse_json(self, response: str) -> Optional[dict]:
        """Robustly parse JSON from response"""
        # Try direct parse
        try:
            return json.loads(response.strip())
        except:
            pass
        
        # Extract from code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
        
        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        return None
    
    def code_output(self, task: str, language: str,
                    style_guide: str = None) -> str:
        """Get code in specific language"""
        
        style = ""
        if style_guide:
            style = f"\n\nStyle requirements:\n{style_guide}"
        
        prompt = f"""Write {language} code for the following task:

{task}
{style}

Requirements:
- Only output code, no explanations
- Include helpful comments within the code
- Code should be complete and runnable
- Follow {language} best practices

```{language.lower()}"""
        
        response = self.llm.generate(prompt, temperature=0)
        return self._extract_code(response, language)
    
    def _extract_code(self, response: str, language: str) -> str:
        """Extract code from response"""
        # Look for code block
        pattern = rf'```{language.lower()}?\s*([\s\S]*?)```'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If no code block, assume entire response is code
        return response.strip()
    
    def table_output(self, task: str, columns: list) -> str:
        """Get markdown table output"""
        
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        
        prompt = f"""Task: {task}

Return the result as a markdown table with these columns:
{header}
{separator}

Add data rows below the header. Only output the table, no other text."""
        
        return self.llm.generate(prompt, temperature=0)
    
    def list_output(self, task: str, style: str = 'numbered') -> list:
        """Get list output"""
        
        if style == 'numbered':
            format_instruction = "Return as numbered list (1. 2. 3.)"
        else:
            format_instruction = "Return as bullet points (- item)"
        
        prompt = f"""Task: {task}

{format_instruction}
Only output the list items, no introduction or conclusion."""
        
        response = self.llm.generate(prompt, temperature=0)
        return self._parse_list(response)
    
    def _parse_list(self, response: str) -> list:
        """Parse list from response"""
        items = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove number or bullet
            line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
            if line:
                items.append(line)
        return items
    
    def function_call_output(self, task: str,
                              functions: list) -> dict:
        """Get function call format (like OpenAI function calling)"""
        
        func_descriptions = json.dumps(functions, indent=2)
        
        prompt = f"""Task: {task}

Available functions:
{func_descriptions}

Determine which function to call and with what arguments.
Return as JSON:
{{
  "function": "function_name",
  "arguments": {{"arg1": value1, ...}}
}}

Only return the JSON, no explanation."""
        
        return self._parse_json(self.llm.generate(prompt, temperature=0))
```

**Interview Tips:**
- "ONLY output X, no explanation" reduces chattiness
- Always validate/parse output programmatically
- Provide schema + example for complex formats
- Use temperature=0 for deterministic structured output

---

### Question 12
- [ ] Done
**What strategies help with prompt consistency across different model versions and providers?**

**Answer:**

**Definition:**
Cross-model consistency: **abstract prompt templates** (separate content from format), **version testing** (test on each model), **model-agnostic instructions** (avoid provider-specific syntax), **fallback handling** (graceful degradation), **continuous evaluation** (detect regressions).

**Consistency Strategies:**

| Strategy | Implementation | Benefit |
|----------|---------------|--------|
| **Template abstraction** | Separate prompt structure from content | Reusable across models |
| **Model adapters** | Transform prompt per model | Handle syntax differences |
| **Regression testing** | Test suite on each model | Catch breaking changes |
| **Output normalization** | Parse to standard format | Consistent downstream |
| **Feature detection** | Check model capabilities | Graceful fallback |

**Python Code Example:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
import json

class PromptTemplate:
    """Model-agnostic prompt template"""
    
    def __init__(self, 
                 role: str,
                 task: str,
                 output_format: str,
                 examples: list = None):
        self.role = role
        self.task = task
        self.output_format = output_format
        self.examples = examples or []
    
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'task': self.task,
            'output_format': self.output_format,
            'examples': self.examples
        }

class ModelAdapter(ABC):
    """Adapt prompts for specific model"""
    
    @abstractmethod
    def format_prompt(self, template: PromptTemplate, 
                      user_input: str) -> str:
        pass

class OpenAIAdapter(ModelAdapter):
    def format_prompt(self, template: PromptTemplate,
                      user_input: str) -> str:
        system = f"{template.role}\n\n{template.output_format}"
        
        examples_text = ""
        for ex in template.examples:
            examples_text += f"User: {ex['input']}\nAssistant: {ex['output']}\n\n"
        
        return {
            'system': system,
            'messages': [
                {'role': 'user', 'content': f"{template.task}\n\n{examples_text}Input: {user_input}"}
            ]
        }

class ClaudeAdapter(ModelAdapter):
    def format_prompt(self, template: PromptTemplate,
                      user_input: str) -> str:
        examples_text = ""
        for ex in template.examples:
            examples_text += f"\n\nH: {ex['input']}\nA: {ex['output']}"
        
        return f"""Human: {template.role}

{template.task}

{template.output_format}
{examples_text}

H: {user_input}

Assistant:"""

class LlamaAdapter(ModelAdapter):
    def format_prompt(self, template: PromptTemplate,
                      user_input: str) -> str:
        examples_text = ""
        for ex in template.examples:
            examples_text += f"\n### Input\n{ex['input']}\n### Output\n{ex['output']}"
        
        return f"""[INST] <<SYS>>
{template.role}
{template.output_format}
<</SYS>>

{template.task}
{examples_text}

### Input
{user_input}
[/INST]"""

class ConsistentPromptManager:
    """Manage prompts across multiple models"""
    
    def __init__(self):
        self.adapters = {
            'openai': OpenAIAdapter(),
            'gpt': OpenAIAdapter(),
            'claude': ClaudeAdapter(),
            'llama': LlamaAdapter(),
            'mistral': LlamaAdapter(),  # Similar format
        }
        self.test_results = {}  # model -> test results
    
    def get_adapter(self, model_name: str) -> ModelAdapter:
        model_lower = model_name.lower()
        for key, adapter in self.adapters.items():
            if key in model_lower:
                return adapter
        return OpenAIAdapter()  # Default
    
    def format_for_model(self, template: PromptTemplate,
                         user_input: str,
                         model_name: str) -> str:
        adapter = self.get_adapter(model_name)
        return adapter.format_prompt(template, user_input)
    
    def run_consistency_test(self, template: PromptTemplate,
                              test_cases: list,
                              models: dict) -> dict:
        """Test template across multiple models"""
        results = {}
        
        for model_name, model_client in models.items():
            model_results = []
            adapter = self.get_adapter(model_name)
            
            for case in test_cases:
                prompt = adapter.format_prompt(template, case['input'])
                response = model_client.generate(prompt)
                
                # Check correctness
                is_correct = self._check_output(
                    response, case.get('expected')
                )
                
                model_results.append({
                    'input': case['input'],
                    'output': response,
                    'correct': is_correct
                })
            
            accuracy = sum(r['correct'] for r in model_results) / len(model_results)
            results[model_name] = {
                'accuracy': accuracy,
                'details': model_results
            }
        
        return results
    
    def _check_output(self, response: str, expected: Optional[str]) -> bool:
        if expected is None:
            return True
        return expected.lower() in response.lower()

# Example usage
def create_classification_template():
    return PromptTemplate(
        role="You are a sentiment classifier.",
        task="Classify the sentiment of the text as positive, negative, or neutral.",
        output_format="Return only one word: positive, negative, or neutral",
        examples=[
            {'input': 'I love this!', 'output': 'positive'},
            {'input': 'This is terrible', 'output': 'negative'}
        ]
    )
```

**Interview Tips:**
- Test prompts on target models before deployment
- Abstract templates enable easier model switching
- Document model-specific quirks
- Monitor for degradation after model updates

---

### Question 13
- [ ] Done
**How do you implement dynamic prompting that adapts to context and user needs?**

**Answer:**

**Definition:**
**Dynamic prompting**: adjust prompt content at runtime based on **user context** (history, preferences), **query analysis** (complexity, topic), **resource constraints** (token budget), and **feedback** (previous errors). Enables personalization and optimization.

**Adaptation Dimensions:**

| Dimension | What Adapts | Example |
|-----------|-------------|--------|
| **User context** | Examples, tone | Return customer vs new user |
| **Query complexity** | Detail level | Simple vs complex question |
| **Topic** | Domain examples | Technical vs casual |
| **Token budget** | Content amount | Mobile vs desktop |
| **Performance** | Strategy | Add CoT if accuracy low |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str
    expertise_level: str  # beginner, intermediate, expert
    preferences: Dict
    history: List[Dict]  # Previous interactions
    domain: str

@dataclass
class QueryAnalysis:
    complexity: str  # simple, moderate, complex
    topic: str
    requires_reasoning: bool
    requires_examples: bool

class DynamicPromptBuilder:
    def __init__(self, example_bank: Dict[str, List[dict]]):
        self.example_bank = example_bank  # topic -> examples
        self.prompt_templates = {
            'beginner': """Explain in simple terms suitable for someone new to {topic}.
Avoid jargon and use analogies where helpful.

""",
            'intermediate': """Provide a clear explanation with some technical detail.
Assume familiarity with basic concepts.

""",
            'expert': """Provide a concise, technical response.
Assume deep expertise in the field.

"""
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine prompt strategy"""
        
        # Complexity heuristics
        complexity = 'simple'
        if len(query.split()) > 30:
            complexity = 'moderate'
        if any(word in query.lower() for word in 
               ['compare', 'analyze', 'evaluate', 'why', 'how']):
            complexity = 'moderate'
        if any(word in query.lower() for word in 
               ['trade-off', 'optimize', 'design', 'implement']):
            complexity = 'complex'
        
        # Topic detection (simplified)
        topics = ['coding', 'math', 'science', 'business', 'creative']
        topic = 'general'
        for t in topics:
            if t in query.lower():
                topic = t
                break
        
        # Reasoning detection
        requires_reasoning = any(word in query.lower() for word in 
                                  ['why', 'how', 'explain', 'reason'])
        
        return QueryAnalysis(
            complexity=complexity,
            topic=topic,
            requires_reasoning=requires_reasoning,
            requires_examples='example' in query.lower()
        )
    
    def build_prompt(self, query: str,
                     user: Optional[UserContext] = None,
                     max_tokens: int = 4000) -> str:
        """Build dynamic prompt based on context"""
        
        analysis = self.analyze_query(query)
        
        # Start with base structure
        prompt_parts = []
        
        # 1. Add expertise-appropriate intro
        if user:
            level = user.expertise_level
        else:
            level = 'intermediate'  # Default
        
        if level in self.prompt_templates:
            prompt_parts.append(
                self.prompt_templates[level].format(topic=analysis.topic)
            )
        
        # 2. Add reasoning instructions if needed
        if analysis.requires_reasoning or analysis.complexity == 'complex':
            prompt_parts.append(
                "Think through this step by step, explaining your reasoning.\n"
            )
        
        # 3. Add relevant examples if available and within budget
        examples = self.select_examples(analysis.topic, query, max_tokens // 4)
        if examples:
            examples_text = "Here are some relevant examples:\n\n"
            for ex in examples:
                examples_text += f"Q: {ex['input']}\nA: {ex['output']}\n\n"
            prompt_parts.append(examples_text)
        
        # 4. Add user history context if relevant
        if user and user.history:
            recent = user.history[-3:]  # Last 3 interactions
            context = "Previous conversation:\n"
            for h in recent:
                context += f"User: {h['query'][:100]}...\n"
            prompt_parts.append(context + "\n")
        
        # 5. Add the query
        prompt_parts.append(f"Current question: {query}\n\nResponse:")
        
        return '\n'.join(prompt_parts)
    
    def select_examples(self, topic: str, query: str,
                        token_budget: int) -> List[dict]:
        """Select relevant examples within budget"""
        
        examples = self.example_bank.get(topic, [])
        if not examples:
            examples = self.example_bank.get('general', [])
        
        # Simple selection: take examples that fit in budget
        selected = []
        tokens_used = 0
        
        for ex in examples:
            ex_tokens = len(ex['input'].split()) + len(ex['output'].split())
            if tokens_used + ex_tokens <= token_budget:
                selected.append(ex)
                tokens_used += ex_tokens
        
        return selected
    
    def adaptive_retry(self, query: str, 
                       previous_response: str,
                       error: str) -> str:
        """Adapt prompt based on previous failure"""
        
        return f"""The previous attempt had an issue: {error}

Previous response:
{previous_response[:500]}...

Please try again, addressing the issue:
{query}

Improved response:"""
```

**Interview Tips:**
- Dynamic prompts improve engagement and accuracy
- Track what adaptations work via A/B testing
- Balance personalization with consistency
- Cache common prompt variations for speed

---

### Question 14
- [ ] Done
**What approaches work best for prompt engineering in specialized domains (legal, medical, technical)?**

**Answer:**

**Definition:**
Domain-specific prompting: **terminology precision** (use exact domain terms), **regulatory compliance** (include disclaimers), **expert validation** (domain expert review), **source grounding** (cite authoritative sources), **uncertainty handling** (explicit about limitations).

**Domain-Specific Considerations:**

| Domain | Key Requirements | Critical Elements |
|--------|-----------------|------------------|
| **Medical** | Disclaimers, accuracy | "Not medical advice", cite guidelines |
| **Legal** | Jurisdiction, precision | Specific statutes, "consult attorney" |
| **Financial** | Compliance, timeliness | Disclaimers, data freshness |
| **Technical** | Accuracy, versioning | Specific versions, tested code |

**Python Code Example:**
```python
from typing import Optional

class DomainSpecificPrompts:
    def __init__(self, llm):
        self.llm = llm
        
        self.domain_configs = {
            'medical': {
                'disclaimer': "DISCLAIMER: This is general health information only, not medical advice. Always consult a qualified healthcare provider.",
                'grounding': "Base your response on established medical guidelines and peer-reviewed research.",
                'uncertainty': "If uncertain, clearly state 'This requires professional medical evaluation.'",
                'prohibited': "Do not diagnose conditions or prescribe treatments."
            },
            'legal': {
                'disclaimer': "DISCLAIMER: This is general legal information, not legal advice. Consult a licensed attorney for specific situations.",
                'grounding': "Reference specific statutes, regulations, or case law where applicable.",
                'uncertainty': "If jurisdiction-specific, note that laws vary by location.",
                'prohibited': "Do not provide specific legal recommendations."
            },
            'financial': {
                'disclaimer': "DISCLAIMER: This is general financial information, not personalized advice. Consult a licensed financial advisor.",
                'grounding': "Reference current regulations and standard financial principles.",
                'uncertainty': "Note that market conditions change and past performance doesn't guarantee future results.",
                'prohibited': "Do not recommend specific investments or trading strategies."
            },
            'technical': {
                'disclaimer': "Note: Always test code in a safe environment before production use.",
                'grounding': "Reference official documentation and best practices.",
                'uncertainty': "Specify version compatibility and potential limitations.",
                'prohibited': "Avoid deprecated methods and security anti-patterns."
            }
        }
    
    def build_domain_prompt(self, query: str, domain: str,
                            context: str = None,
                            expertise_level: str = 'general') -> str:
        """Build domain-appropriate prompt"""
        
        config = self.domain_configs.get(domain, {})
        
        prompt = f"""# Domain: {domain.title()}
# Expertise Level: {expertise_level}

## Guidelines
{config.get('grounding', '')}
{config.get('uncertainty', '')}
{config.get('prohibited', '')}

"""
        
        if context:
            prompt += f"""## Reference Information
{context}

Use the above information to inform your response.

"""
        
        prompt += f"""## Query
{query}

## Response
"""
        
        return prompt
    
    def generate_with_compliance(self, query: str, domain: str,
                                  context: str = None) -> dict:
        """Generate response with compliance handling"""
        
        config = self.domain_configs.get(domain, {})
        prompt = self.build_domain_prompt(query, domain, context)
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Add mandatory disclaimer
        disclaimer = config.get('disclaimer', '')
        
        # Check for concerning patterns
        concerns = self._check_compliance(response, domain)
        
        return {
            'response': response,
            'disclaimer': disclaimer,
            'compliance_concerns': concerns,
            'domain': domain
        }
    
    def _check_compliance(self, response: str, domain: str) -> list:
        """Check for potential compliance issues"""
        concerns = []
        
        response_lower = response.lower()
        
        if domain == 'medical':
            if any(word in response_lower for word in 
                   ['you should take', 'i recommend taking', 'dosage']):
                concerns.append('potential_treatment_recommendation')
            if any(word in response_lower for word in 
                   ['you have', 'diagnosis is', 'suffering from']):
                concerns.append('potential_diagnosis')
        
        elif domain == 'legal':
            if any(word in response_lower for word in 
                   ['you should sue', 'file a lawsuit', 'legal action']):
                concerns.append('potential_legal_recommendation')
        
        elif domain == 'financial':
            if any(word in response_lower for word in 
                   ['guaranteed returns', 'risk-free', 'buy this stock']):
                concerns.append('potential_investment_advice')
        
        return concerns

class MedicalPromptEngineer:
    """Specialized medical domain prompting"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def patient_education(self, topic: str, 
                          reading_level: str = 'general') -> str:
        """Generate patient education content"""
        
        level_instructions = {
            'general': "Use simple language at an 8th grade reading level.",
            'advanced': "Assume health literacy and some medical knowledge."
        }
        
        prompt = f"""You are a health educator creating patient information.

Topic: {topic}

{level_instructions.get(reading_level, level_instructions['general'])}

Guidelines:
- Use established medical guidelines (CDC, WHO, professional societies)
- Avoid medical jargon or define terms when used
- Include when to seek professional care
- Do not diagnose or recommend specific treatments
- Encourage consultation with healthcare providers

Patient Education Content:"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Always add disclaimer
        return f"""⚕️ Health Information

{response}

---
⚠️ IMPORTANT: This is general health information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."""
```

**Interview Tips:**
- Disclaimers are legally important in regulated domains
- Ground responses in authoritative sources (RAG)
- Have domain experts review prompts and outputs
- Monitor for compliance violations continuously

---

## Production Prompt Management

### Question 15
- [ ] Done
**How do you handle prompt versioning and management in production systems?**

**Answer:**

**Definition:**
Prompt versioning: treat prompts as **code artifacts** with version control, changelogs, rollback capability, and deployment pipelines. Key practices: **git versioning**, **semantic versioning**, **environment promotion** (dev → staging → prod), **A/B testing**, **audit trails**.

**Versioning Strategy:**

| Component | Approach | Purpose |
|-----------|----------|--------|
| **Storage** | Git repository | Version history |
| **Naming** | Semantic versioning | Change significance |
| **Deployment** | CI/CD pipeline | Controlled rollout |
| **Rollback** | Instant switch | Incident recovery |
| **Audit** | Change logs | Compliance |

**Python Code Example:**
```python
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from enum import Enum

class PromptStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    description: str
    created_at: float
    created_by: str
    status: PromptStatus
    metrics: Dict = None  # Performance metrics
    parent_version: Optional[str] = None
    
    def get_hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

class PromptRegistry:
    """Central prompt management system"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.prompts: Dict[str, Dict[str, PromptVersion]] = {}  # name -> version -> prompt
        self.active_versions: Dict[str, str] = {}  # name -> active version
        self._load()
    
    def _load(self):
        """Load prompts from storage"""
        # In production: load from database or file
        pass
    
    def register(self, prompt: PromptVersion) -> str:
        """Register a new prompt version"""
        if prompt.name not in self.prompts:
            self.prompts[prompt.name] = {}
        
        self.prompts[prompt.name][prompt.version] = prompt
        
        # Auto-activate if first version
        if prompt.name not in self.active_versions:
            self.active_versions[prompt.name] = prompt.version
        
        return f"{prompt.name}:{prompt.version}"
    
    def get_active(self, name: str) -> Optional[PromptVersion]:
        """Get active production version"""
        if name not in self.active_versions:
            return None
        
        version = self.active_versions[name]
        return self.prompts.get(name, {}).get(version)
    
    def get_version(self, name: str, version: str) -> Optional[PromptVersion]:
        """Get specific version"""
        return self.prompts.get(name, {}).get(version)
    
    def promote(self, name: str, version: str,
                user: str) -> bool:
        """Promote version to production"""
        prompt = self.get_version(name, version)
        if not prompt:
            return False
        
        # Update status
        old_active = self.active_versions.get(name)
        if old_active:
            self.prompts[name][old_active].status = PromptStatus.DEPRECATED
        
        prompt.status = PromptStatus.PRODUCTION
        self.active_versions[name] = version
        
        # Audit log
        self._audit_log({
            'action': 'promote',
            'name': name,
            'version': version,
            'previous': old_active,
            'user': user,
            'timestamp': time.time()
        })
        
        return True
    
    def rollback(self, name: str, user: str) -> bool:
        """Rollback to previous version"""
        current = self.get_active(name)
        if not current or not current.parent_version:
            return False
        
        return self.promote(name, current.parent_version, user)
    
    def _audit_log(self, entry: dict):
        """Log change for audit"""
        # In production: write to audit log
        print(f"AUDIT: {json.dumps(entry)}")

class PromptManager:
    """Production prompt manager with hot reload"""
    
    def __init__(self, registry: PromptRegistry):
        self.registry = registry
        self.cache: Dict[str, PromptVersion] = {}
        self.cache_ttl = 60  # seconds
        self.cache_times: Dict[str, float] = {}
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """Get rendered prompt with caching"""
        
        # Check cache
        now = time.time()
        if name in self.cache:
            if now - self.cache_times[name] < self.cache_ttl:
                template = self.cache[name].template
                return template.format(**kwargs)
        
        # Fetch from registry
        prompt = self.registry.get_active(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        
        # Update cache
        self.cache[name] = prompt
        self.cache_times[name] = now
        
        return prompt.template.format(**kwargs)
    
    def create_version(self, name: str, template: str,
                       description: str, user: str) -> PromptVersion:
        """Create new version from current"""
        
        current = self.registry.get_active(name)
        
        # Determine new version number
        if current:
            # Increment patch version (simplified)
            parts = current.version.split('.')
            parts[-1] = str(int(parts[-1]) + 1)
            new_version = '.'.join(parts)
            parent = current.version
        else:
            new_version = '1.0.0'
            parent = None
        
        prompt = PromptVersion(
            name=name,
            version=new_version,
            template=template,
            description=description,
            created_at=time.time(),
            created_by=user,
            status=PromptStatus.DRAFT,
            parent_version=parent
        )
        
        self.registry.register(prompt)
        return prompt
```

**Interview Tips:**
- Version prompts like code (git, semantic versioning)
- Test before promoting to production
- Enable instant rollback for incidents
- Track metrics per version for comparison

---

### Question 16
- [ ] Done
**What strategies help with A/B testing different prompts in production?**

**Answer:**

**Definition:**
Prompt A/B testing: **randomly assign users to prompt variants**, **measure outcome metrics** (accuracy, satisfaction, cost), **statistical significance** testing, **gradual rollout** (canary → full). Key: isolate prompt changes from other variables.

**A/B Testing Framework:**

| Component | Implementation | Purpose |
|-----------|---------------|--------|
| **Randomization** | Consistent user hashing | Stable assignment |
| **Metrics** | Quality, latency, cost | Compare variants |
| **Sample size** | Statistical power calculation | Valid conclusions |
| **Duration** | Sufficient data collection | Reliable results |
| **Segmentation** | By user cohort | Detect heterogeneity |

**Python Code Example:**
```python
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import math

@dataclass
class PromptVariant:
    name: str
    template: str
    traffic_pct: float  # 0-1

@dataclass
class ExperimentResult:
    variant: str
    user_id: str
    query: str
    response: str
    latency_ms: float
    quality_score: Optional[float]
    timestamp: float

class PromptABTest:
    def __init__(self, experiment_name: str,
                 variants: List[PromptVariant]):
        self.experiment_name = experiment_name
        self.variants = variants
        self.results: List[ExperimentResult] = []
        
        # Validate traffic allocation
        total = sum(v.traffic_pct for v in variants)
        assert abs(total - 1.0) < 0.01, "Traffic must sum to 100%"
    
    def get_variant(self, user_id: str) -> PromptVariant:
        """Consistently assign user to variant"""
        
        # Hash user_id for consistent assignment
        hash_input = f"{self.experiment_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000  # 0-1
        
        # Assign to variant based on traffic split
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.traffic_pct
            if bucket < cumulative:
                return variant
        
        return self.variants[-1]  # Fallback
    
    def record_result(self, user_id: str, query: str,
                      response: str, latency_ms: float,
                      quality_score: float = None):
        """Record experiment result"""
        
        variant = self.get_variant(user_id)
        
        result = ExperimentResult(
            variant=variant.name,
            user_id=user_id,
            query=query,
            response=response,
            latency_ms=latency_ms,
            quality_score=quality_score,
            timestamp=time.time()
        )
        
        self.results.append(result)
    
    def analyze(self) -> Dict:
        """Analyze experiment results"""
        
        by_variant = {}
        for result in self.results:
            if result.variant not in by_variant:
                by_variant[result.variant] = []
            by_variant[result.variant].append(result)
        
        analysis = {}
        for variant_name, results in by_variant.items():
            quality_scores = [r.quality_score for r in results 
                             if r.quality_score is not None]
            latencies = [r.latency_ms for r in results]
            
            analysis[variant_name] = {
                'n': len(results),
                'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else None,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))] if latencies else None
            }
        
        # Statistical significance test
        if len(by_variant) == 2:
            variants = list(by_variant.keys())
            analysis['significance'] = self._ttest(
                [r.quality_score for r in by_variant[variants[0]] if r.quality_score],
                [r.quality_score for r in by_variant[variants[1]] if r.quality_score]
            )
        
        return analysis
    
    def _ttest(self, a: List[float], b: List[float]) -> Dict:
        """Simple t-test for significance"""
        if len(a) < 2 or len(b) < 2:
            return {'significant': False, 'reason': 'insufficient data'}
        
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        
        var_a = sum((x - mean_a)**2 for x in a) / (len(a) - 1)
        var_b = sum((x - mean_b)**2 for x in b) / (len(b) - 1)
        
        se = math.sqrt(var_a/len(a) + var_b/len(b))
        
        if se == 0:
            return {'significant': False, 'reason': 'zero variance'}
        
        t_stat = (mean_a - mean_b) / se
        
        # Rough p-value approximation
        significant = abs(t_stat) > 1.96  # ~95% confidence
        
        return {
            'significant': significant,
            't_statistic': t_stat,
            'mean_diff': mean_a - mean_b
        }
    
    def get_winner(self, metric: str = 'avg_quality') -> Optional[str]:
        """Determine winning variant"""
        
        analysis = self.analyze()
        
        # Check significance first
        if 'significance' in analysis:
            if not analysis['significance'].get('significant', False):
                return None  # No significant winner
        
        # Find best variant
        best = None
        best_value = float('-inf')
        
        for variant, metrics in analysis.items():
            if variant == 'significance':
                continue
            
            value = metrics.get(metric)
            if value and value > best_value:
                best_value = value
                best = variant
        
        return best

# Usage
def run_prompt_experiment():
    variants = [
        PromptVariant('control', 'Answer: {query}', 0.5),
        PromptVariant('treatment', 'Think step by step: {query}', 0.5)
    ]
    
    experiment = PromptABTest('cot_test', variants)
    return experiment
```

**Interview Tips:**
- Use consistent hashing for stable user assignment
- Run until statistically significant (typically 1000+ samples)
- Measure multiple metrics (quality, latency, cost)
- Segment by user cohorts to find heterogeneous effects

---

### Question 17
- [ ] Done
**How do you implement robust error handling when prompts produce unexpected outputs?**

**Answer:**

**Definition:**
Handle unexpected outputs via: **output validation** (schema checking), **retry with modification** (rephrase prompt), **fallback responses** (graceful degradation), **error classification** (categorize failures), **monitoring** (detect systematic issues).

**Error Handling Layers:**

| Layer | Handles | Action |
|-------|---------|--------|
| **Validation** | Format errors | Retry with format hint |
| **Content check** | Harmful/off-topic | Block and fallback |
| **Timeout** | Slow responses | Return cached/default |
| **Rate limit** | API limits | Queue and retry |
| **Fallback** | All failures | Graceful degradation |

**Python Code Example:**
```python
import json
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class ErrorType(Enum):
    FORMAT_ERROR = "format_error"
    CONTENT_ERROR = "content_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"

@dataclass
class LLMResponse:
    success: bool
    content: Any
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    attempts: int = 1

class RobustLLMClient:
    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.fallback_responses = {}
    
    def generate_with_validation(self, 
                                  prompt: str,
                                  validator: Callable[[str], bool],
                                  parser: Callable[[str], Any] = None,
                                  fallback: Any = None) -> LLMResponse:
        """Generate with validation and retry"""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Generate response
                response = self._call_with_timeout(prompt)
                
                # Validate
                if not validator(response):
                    last_error = ErrorType.VALIDATION_ERROR
                    prompt = self._add_retry_hint(prompt, response, attempt)
                    continue
                
                # Parse if parser provided
                if parser:
                    try:
                        parsed = parser(response)
                        return LLMResponse(
                            success=True,
                            content=parsed,
                            attempts=attempt + 1
                        )
                    except Exception as e:
                        last_error = ErrorType.FORMAT_ERROR
                        prompt = self._add_format_hint(prompt, str(e))
                        continue
                
                return LLMResponse(
                    success=True,
                    content=response,
                    attempts=attempt + 1
                )
                
            except TimeoutError:
                last_error = ErrorType.TIMEOUT
            except RateLimitError:
                last_error = ErrorType.RATE_LIMIT
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                last_error = ErrorType.API_ERROR
        
        # All retries failed - use fallback
        return LLMResponse(
            success=False,
            content=fallback,
            error_type=last_error,
            error_message=f"Failed after {self.max_retries} attempts",
            attempts=self.max_retries
        )
    
    def _call_with_timeout(self, prompt: str, 
                           timeout: float = 30) -> str:
        """Call LLM with timeout"""
        # In production: use asyncio or threading
        return self.llm.generate(prompt)
    
    def _add_retry_hint(self, prompt: str, 
                        failed_response: str,
                        attempt: int) -> str:
        """Add hint for retry"""
        hints = [
            "\n\nIMPORTANT: Please follow the format exactly.",
            f"\n\nYour previous response was invalid. Ensure correct format.",
            f"\n\nPrevious attempt failed validation. Response must be properly formatted."
        ]
        return prompt + hints[min(attempt, len(hints)-1)]
    
    def _add_format_hint(self, prompt: str, error: str) -> str:
        """Add format correction hint"""
        return prompt + f"\n\nFormat error: {error}. Please fix."

class OutputValidator:
    """Validate LLM outputs"""
    
    @staticmethod
    def is_valid_json(response: str) -> bool:
        """Check if response is valid JSON"""
        try:
            json.loads(response.strip())
            return True
        except:
            # Try extracting from code block
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
            if match:
                try:
                    json.loads(match.group(1).strip())
                    return True
                except:
                    pass
            return False
    
    @staticmethod
    def matches_schema(response: str, schema: dict) -> bool:
        """Check if JSON matches schema"""
        try:
            data = json.loads(response.strip())
            return OutputValidator._check_schema(data, schema)
        except:
            return False
    
    @staticmethod
    def _check_schema(data: Any, schema: dict) -> bool:
        """Simple schema validation"""
        if 'type' in schema:
            type_map = {
                'string': str, 'number': (int, float),
                'boolean': bool, 'array': list, 'object': dict
            }
            expected_type = type_map.get(schema['type'])
            if expected_type and not isinstance(data, expected_type):
                return False
        
        if 'required' in schema and isinstance(data, dict):
            for field in schema['required']:
                if field not in data:
                    return False
        
        return True
    
    @staticmethod
    def is_not_harmful(response: str) -> bool:
        """Basic content safety check"""
        harmful_patterns = ['violence', 'hate', 'explicit']
        response_lower = response.lower()
        return not any(p in response_lower for p in harmful_patterns)

class RateLimitError(Exception):
    pass
```

**Interview Tips:**
- Always validate structured outputs
- Exponential backoff for rate limits
- Log failures for debugging and improvement
- Have fallback for every critical path

---

### Question 18
- [ ] Done
**What approaches work best for combining prompt engineering with fine-tuning strategies?**

**Answer:**

**Definition:**
Combine prompting and fine-tuning: **start with prompting** (quick iteration), **fine-tune for consistency** (stable behavior), **use prompts for steering** (fine-tuned base), **specialize prompts for fine-tuned model**. Trade-off: prompting is flexible, fine-tuning is consistent.

**When to Use Each:**

| Approach | Best For | Trade-off |
|----------|----------|----------|
| **Prompting only** | Prototyping, general tasks | Flexible but variable |
| **Fine-tuning only** | Consistent format, speed | Fixed behavior |
| **Prompt + Fine-tune** | Quality + consistency | Best of both |
| **RAG + Fine-tune** | Domain knowledge + capability | Comprehensive |

**Combination Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Fine-tune base, prompt for task** | Train on domain, prompt for specific tasks | Multi-task domain |
| **Fine-tune format, prompt content** | Train output format, prompt guides content | API consistency |
| **Prompt-tuning** | Train soft prompts | Efficient adaptation |
| **Instruction fine-tuning + prompt** | Train instruction following, detailed prompts | Complex instructions |

**Python Code Example:**
```python
from typing import List, Dict
import json

class HybridLLMSystem:
    """Combine fine-tuning with prompting"""
    
    def __init__(self, base_model, fine_tuned_model=None):
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model
        self.use_fine_tuned = fine_tuned_model is not None
    
    def generate(self, task: str, context: str = None,
                 use_fine_tuned: bool = None) -> str:
        """Generate using appropriate model + prompt"""
        
        model = self.fine_tuned_model if (use_fine_tuned or self.use_fine_tuned) else self.base_model
        
        if self.use_fine_tuned:
            # Fine-tuned model: simpler prompts work
            prompt = self._build_simple_prompt(task, context)
        else:
            # Base model: needs detailed prompting
            prompt = self._build_detailed_prompt(task, context)
        
        return model.generate(prompt)
    
    def _build_simple_prompt(self, task: str, context: str = None) -> str:
        """Simple prompt for fine-tuned model"""
        if context:
            return f"Context: {context}\n\nTask: {task}"
        return f"Task: {task}"
    
    def _build_detailed_prompt(self, task: str, context: str = None) -> str:
        """Detailed prompt for base model"""
        prompt = """You are a helpful assistant. Follow these guidelines:
1. Provide accurate, well-structured responses
2. Be concise but complete
3. If uncertain, acknowledge it

"""
        if context:
            prompt += f"Reference information:\n{context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt

class FineTuningDataGenerator:
    """Generate training data from successful prompts"""
    
    def __init__(self, llm):
        self.llm = llm
        self.training_examples = []
    
    def generate_training_pair(self, prompt: str, 
                                ideal_response: str = None) -> dict:
        """Create training example"""
        
        if ideal_response is None:
            # Generate with detailed prompt, use as training target
            ideal_response = self.llm.generate(prompt)
        
        # Simplify prompt for training
        simplified_prompt = self._simplify_prompt(prompt)
        
        return {
            'prompt': simplified_prompt,
            'completion': ideal_response
        }
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Remove redundant instructions for fine-tuning"""
        # In production: use more sophisticated simplification
        # Remove common boilerplate
        lines = prompt.split('\n')
        simplified = []
        skip_patterns = ['You are', 'Follow these', 'Guidelines:']
        
        for line in lines:
            if not any(p in line for p in skip_patterns):
                simplified.append(line)
        
        return '\n'.join(simplified).strip()
    
    def collect_examples(self, prompts: List[str],
                         quality_threshold: float = 0.8) -> List[dict]:
        """Collect high-quality examples for fine-tuning"""
        
        examples = []
        for prompt in prompts:
            response = self.llm.generate(prompt)
            quality = self._assess_quality(prompt, response)
            
            if quality >= quality_threshold:
                examples.append(self.generate_training_pair(prompt, response))
        
        return examples
    
    def _assess_quality(self, prompt: str, response: str) -> float:
        """Assess response quality"""
        # In production: use LLM-as-judge or metrics
        # Simple heuristics
        score = 0.5
        
        if len(response) > 50:
            score += 0.2
        if '\n' in response:  # Structured
            score += 0.1
        if response.strip().endswith(('.', '!', '?', '```')):
            score += 0.1
        
        return min(score, 1.0)

def prepare_fine_tuning_data(examples: List[dict], format: str = 'openai') -> str:
    """Format data for fine-tuning"""
    
    if format == 'openai':
        # OpenAI chat format
        formatted = []
        for ex in examples:
            formatted.append({
                'messages': [
                    {'role': 'user', 'content': ex['prompt']},
                    {'role': 'assistant', 'content': ex['completion']}
                ]
            })
        return '\n'.join(json.dumps(f) for f in formatted)
    
    elif format == 'alpaca':
        # Alpaca format
        formatted = []
        for ex in examples:
            formatted.append({
                'instruction': ex['prompt'],
                'input': '',
                'output': ex['completion']
            })
        return json.dumps(formatted, indent=2)
    
    return json.dumps(examples)
```

**Decision Framework:**
```
Need consistent output format? → Fine-tune
Need flexibility/customization? → Prompt
Need both? → Fine-tune base + task-specific prompts
Need domain knowledge? → RAG + either
```

**Interview Tips:**
- Start with prompting, fine-tune when stable
- Fine-tuning reduces prompt tokens (cost savings)
- Collect high-quality examples from successful prompts
- Fine-tuned models still benefit from good prompts

---

### Question 19
- [ ] Done
**How do you implement efficient prompt testing and validation pipelines?**

**Answer:**

**Definition:**
Prompt testing pipeline: **unit tests** (format, basic behavior), **integration tests** (end-to-end), **regression tests** (compare to baseline), **quality benchmarks** (accuracy metrics), **cost tracking** (token usage). Automate in CI/CD for every prompt change.

**Testing Layers:**

| Layer | Tests | Automation |
|-------|-------|------------|
| **Unit** | Prompt renders correctly, format valid | CI - fast |
| **Integration** | LLM produces valid output | CI - medium |
| **Regression** | Output quality vs baseline | Nightly |
| **Performance** | Latency, cost | Weekly |
| **A/B** | User metrics | Production |

**Python Code Example:**
```python
import json
import time
from typing import List, Dict, Callable
from dataclasses import dataclass
import hashlib

@dataclass
class TestCase:
    name: str
    input_vars: Dict
    expected_contains: List[str] = None
    expected_format: str = None  # 'json', 'list', 'code'
    max_tokens: int = None
    timeout_ms: int = 5000

@dataclass
class TestResult:
    test_name: str
    passed: bool
    duration_ms: float
    error: str = None
    details: Dict = None

class PromptTestSuite:
    """Automated prompt testing"""
    
    def __init__(self, llm):
        self.llm = llm
        self.baselines = {}  # prompt_hash -> baseline responses
    
    def run_tests(self, prompt_template: str,
                  test_cases: List[TestCase]) -> Dict:
        """Run all tests for a prompt"""
        
        results = []
        start_time = time.time()
        
        for test in test_cases:
            result = self._run_single_test(prompt_template, test)
            results.append(result)
        
        total_time = time.time() - start_time
        passed = sum(1 for r in results if r.passed)
        
        return {
            'total': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'pass_rate': passed / len(results),
            'total_time_s': total_time,
            'results': results
        }
    
    def _run_single_test(self, prompt_template: str,
                         test: TestCase) -> TestResult:
        """Run single test case"""
        
        start = time.perf_counter()
        
        try:
            # Render prompt
            prompt = prompt_template.format(**test.input_vars)
            
            # Generate response
            response = self.llm.generate(prompt)
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            # Run assertions
            errors = []
            
            # Check content
            if test.expected_contains:
                for expected in test.expected_contains:
                    if expected.lower() not in response.lower():
                        errors.append(f"Missing expected: {expected}")
            
            # Check format
            if test.expected_format:
                if not self._check_format(response, test.expected_format):
                    errors.append(f"Invalid format: expected {test.expected_format}")
            
            # Check token limit
            if test.max_tokens:
                tokens = len(response.split())  # Rough estimate
                if tokens > test.max_tokens:
                    errors.append(f"Too many tokens: {tokens} > {test.max_tokens}")
            
            # Check timeout
            if duration_ms > test.timeout_ms:
                errors.append(f"Timeout: {duration_ms:.0f}ms > {test.timeout_ms}ms")
            
            return TestResult(
                test_name=test.name,
                passed=len(errors) == 0,
                duration_ms=duration_ms,
                error='; '.join(errors) if errors else None,
                details={'response_length': len(response)}
            )
            
        except Exception as e:
            return TestResult(
                test_name=test.name,
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            )
    
    def _check_format(self, response: str, expected: str) -> bool:
        """Check response format"""
        if expected == 'json':
            try:
                json.loads(response.strip())
                return True
            except:
                return False
        elif expected == 'list':
            return any(c in response for c in ['1.', '-', '*'])
        elif expected == 'code':
            return '```' in response or response.strip().startswith('def ')
        return True
    
    def regression_test(self, prompt_template: str,
                        test_cases: List[TestCase],
                        threshold: float = 0.9) -> Dict:
        """Compare against baseline"""
        
        prompt_hash = hashlib.md5(prompt_template.encode()).hexdigest()[:12]
        
        current_results = self.run_tests(prompt_template, test_cases)
        
        if prompt_hash in self.baselines:
            baseline = self.baselines[prompt_hash]
            
            # Compare metrics
            regression = {
                'pass_rate_change': current_results['pass_rate'] - baseline['pass_rate'],
                'is_regression': current_results['pass_rate'] < baseline['pass_rate'] * threshold
            }
            current_results['regression'] = regression
        else:
            # First run - set as baseline
            self.baselines[prompt_hash] = {
                'pass_rate': current_results['pass_rate'],
                'timestamp': time.time()
            }
        
        return current_results

class PromptTestPipeline:
    """CI/CD pipeline for prompt testing"""
    
    def __init__(self, test_suite: PromptTestSuite):
        self.test_suite = test_suite
    
    def run_pipeline(self, prompts: Dict[str, str],
                     test_cases: Dict[str, List[TestCase]]) -> Dict:
        """Run full pipeline"""
        
        results = {
            'prompts': {},
            'overall_pass': True
        }
        
        for prompt_name, template in prompts.items():
            cases = test_cases.get(prompt_name, [])
            if not cases:
                continue
            
            result = self.test_suite.run_tests(template, cases)
            results['prompts'][prompt_name] = result
            
            if result['pass_rate'] < 1.0:
                results['overall_pass'] = False
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate test report"""
        
        report = "# Prompt Test Report\n\n"
        
        for prompt_name, result in results['prompts'].items():
            status = "✅" if result['pass_rate'] == 1.0 else "❌"
            report += f"## {status} {prompt_name}\n"
            report += f"- Pass rate: {result['pass_rate']:.1%}\n"
            report += f"- Total time: {result['total_time_s']:.2f}s\n"
            
            for test_result in result['results']:
                if not test_result.passed:
                    report += f"  - ❌ {test_result.test_name}: {test_result.error}\n"
        
        return report
```

**Interview Tips:**
- Automate tests in CI/CD pipeline
- Track metrics over time for regression detection
- Use representative test cases (edge cases + common cases)
- Cost tracking prevents budget surprises

---

### Question 20
- [ ] Done
**What techniques work best for prompt engineering with regulatory compliance needs?**

**Answer:**

**Definition:**
Compliance-focused prompting: **mandatory disclaimers**, **audit trails** (log all interactions), **access controls** (who uses what), **content restrictions** (prohibited topics), **data handling** (PII protection), **documentation** (explainability).

**Compliance Requirements by Domain:**

| Domain | Requirements | Prompt Considerations |
|--------|-------------|----------------------|
| **Financial** | FINRA, SEC | No investment advice, disclosures |
| **Healthcare** | HIPAA | PHI handling, medical disclaimers |
| **Legal** | Bar rules | No legal advice, jurisdiction notes |
| **Education** | FERPA | Student data protection |
| **General** | GDPR, CCPA | Data minimization, consent |

**Python Code Example:**
```python
import time
import json
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ComplianceLevel(Enum):
    STANDARD = "standard"
    REGULATED = "regulated"
    STRICT = "strict"

@dataclass
class ComplianceConfig:
    level: ComplianceLevel
    required_disclaimers: List[str]
    prohibited_topics: List[str]
    required_fields: List[str]  # Must be in output
    pii_handling: str  # 'mask', 'reject', 'allow'
    audit_required: bool

@dataclass
class AuditRecord:
    timestamp: float
    user_id: str
    prompt_hash: str
    response_hash: str
    compliance_level: str
    disclaimers_included: bool
    pii_detected: bool
    flags: List[str]

class CompliancePromptManager:
    """Manage prompts with regulatory compliance"""
    
    def __init__(self, llm, audit_logger):
        self.llm = llm
        self.audit = audit_logger
        
        self.configs = {
            'financial': ComplianceConfig(
                level=ComplianceLevel.STRICT,
                required_disclaimers=[
                    "This is not financial advice.",
                    "Consult a licensed financial advisor."
                ],
                prohibited_topics=['specific investment recommendations',
                                   'guaranteed returns', 'insider information'],
                required_fields=['disclaimer'],
                pii_handling='mask',
                audit_required=True
            ),
            'healthcare': ComplianceConfig(
                level=ComplianceLevel.STRICT,
                required_disclaimers=[
                    "This is not medical advice.",
                    "Consult a healthcare provider."
                ],
                prohibited_topics=['diagnosis', 'prescriptions', 'treatment plans'],
                required_fields=['disclaimer'],
                pii_handling='reject',
                audit_required=True
            ),
            'general': ComplianceConfig(
                level=ComplianceLevel.STANDARD,
                required_disclaimers=[],
                prohibited_topics=['illegal activities', 'harm'],
                required_fields=[],
                pii_handling='mask',
                audit_required=False
            )
        }
    
    def build_compliant_prompt(self, task: str, domain: str,
                                user_id: str) -> str:
        """Build prompt with compliance requirements"""
        
        config = self.configs.get(domain, self.configs['general'])
        
        prompt = f"""# Compliance Guidelines ({domain.upper()})

## Restrictions
- Do NOT provide: {', '.join(config.prohibited_topics)}
- Always include required disclaimers
- If asked about prohibited topics, politely decline

## Required Elements
Every response must include:
{chr(10).join('- ' + d for d in config.required_disclaimers)}

## Task
{task}

## Response
"""
        
        return prompt
    
    def generate_compliant(self, task: str, domain: str,
                           user_id: str) -> Dict:
        """Generate with full compliance handling"""
        
        config = self.configs.get(domain, self.configs['general'])
        
        # Check for PII in input
        pii_detected = self._detect_pii(task)
        if pii_detected and config.pii_handling == 'reject':
            return {
                'success': False,
                'error': 'PII detected in input',
                'response': None
            }
        
        if pii_detected and config.pii_handling == 'mask':
            task = self._mask_pii(task)
        
        # Build and execute prompt
        prompt = self.build_compliant_prompt(task, domain, user_id)
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Verify compliance
        compliance_check = self._verify_compliance(response, config)
        
        # Add disclaimers if missing
        if not compliance_check['disclaimers_present']:
            response = self._add_disclaimers(response, config.required_disclaimers)
        
        # Check for prohibited content
        if compliance_check['prohibited_content']:
            return {
                'success': False,
                'error': 'Response contained prohibited content',
                'response': None,
                'flags': compliance_check['flags']
            }
        
        # Audit logging
        if config.audit_required:
            self._log_audit(user_id, prompt, response, config, 
                           pii_detected, compliance_check['flags'])
        
        return {
            'success': True,
            'response': response,
            'disclaimers': config.required_disclaimers,
            'compliance_level': config.level.value
        }
    
    def _detect_pii(self, text: str) -> bool:
        """Detect PII in text"""
        import re
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        return any(re.search(p, text) for p in patterns)
    
    def _mask_pii(self, text: str) -> str:
        """Mask PII in text"""
        import re
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{16}\b', '[CARD]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                      '[EMAIL]', text)
        return text
    
    def _verify_compliance(self, response: str, 
                           config: ComplianceConfig) -> Dict:
        """Verify response compliance"""
        
        response_lower = response.lower()
        
        # Check disclaimers
        disclaimers_present = all(
            d.lower() in response_lower 
            for d in config.required_disclaimers
        )
        
        # Check prohibited content
        flags = []
        for topic in config.prohibited_topics:
            if topic.lower() in response_lower:
                flags.append(f"prohibited: {topic}")
        
        return {
            'disclaimers_present': disclaimers_present,
            'prohibited_content': len(flags) > 0,
            'flags': flags
        }
    
    def _add_disclaimers(self, response: str, 
                         disclaimers: List[str]) -> str:
        """Add required disclaimers to response"""
        disclaimer_text = "\n\n---\n" + "\n".join(disclaimers)
        return response + disclaimer_text
    
    def _log_audit(self, user_id: str, prompt: str, response: str,
                   config: ComplianceConfig, pii_detected: bool,
                   flags: List[str]):
        """Log for audit trail"""
        record = AuditRecord(
            timestamp=time.time(),
            user_id=user_id,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            response_hash=hashlib.sha256(response.encode()).hexdigest()[:16],
            compliance_level=config.level.value,
            disclaimers_included=True,
            pii_detected=pii_detected,
            flags=flags
        )
        self.audit.log(asdict(record))
```

**Interview Tips:**
- Audit trails are legally required in regulated industries
- Always add disclaimers programmatically, not in prompt
- PII detection prevents accidental data exposure
- Document compliance controls for regulators

---

## RAG Architecture Fundamentals

### Question 21
- [ ] Done
**How do you design RAG systems that effectively balance retrieval relevance and generation quality?**

**Answer:**

**Definition:**
RAG balance: **retrieval precision** (right documents), **retrieval recall** (enough coverage), **context quality** (informative chunks), **generation faithfulness** (use retrieved info), **coherence** (natural response). Trade-offs exist at each stage.

**Key Design Decisions:**

| Component | Options | Trade-off |
|-----------|---------|----------|
| **Chunk size** | Small (256) vs Large (1024) | Precision vs context |
| **Top-k** | Few (3) vs Many (10) | Focus vs coverage |
| **Reranking** | None vs Cross-encoder | Quality vs latency |
| **Fusion** | Stuffing vs MapReduce | Simple vs thorough |

**Python Code Example:**
```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str
    metadata: dict

class BalancedRAG:
    """RAG system balancing retrieval and generation"""
    
    def __init__(self, retriever, reranker, llm,
                 max_context_tokens: int = 4000):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.max_context_tokens = max_context_tokens
    
    def query(self, question: str,
              initial_k: int = 20,
              final_k: int = 5,
              min_relevance: float = 0.5) -> dict:
        """Balanced retrieval and generation"""
        
        # Stage 1: Broad retrieval (recall-focused)
        candidates = self.retriever.search(question, k=initial_k)
        
        # Stage 2: Reranking (precision-focused)
        if self.reranker:
            reranked = self.reranker.rerank(question, candidates)
        else:
            reranked = candidates
        
        # Stage 3: Filter by relevance threshold
        relevant = [c for c in reranked if c.score >= min_relevance]
        
        # Stage 4: Fit within context window
        context_chunks = self._fit_context(relevant[:final_k * 2], 
                                           self.max_context_tokens)
        
        # Stage 5: Generate with retrieved context
        response = self._generate(question, context_chunks)
        
        # Stage 6: Verify faithfulness
        faithfulness = self._check_faithfulness(response, context_chunks)
        
        return {
            'answer': response,
            'sources': [c.source for c in context_chunks],
            'faithfulness_score': faithfulness,
            'chunks_used': len(context_chunks)
        }
    
    def _fit_context(self, chunks: List[RetrievedChunk],
                     max_tokens: int) -> List[RetrievedChunk]:
        """Select chunks that fit in context window"""
        selected = []
        tokens_used = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk.text.split())  # Rough estimate
            if tokens_used + chunk_tokens <= max_tokens:
                selected.append(chunk)
                tokens_used += chunk_tokens
        
        return selected
    
    def _generate(self, question: str,
                  chunks: List[RetrievedChunk]) -> str:
        """Generate answer from context"""
        
        context = "\n\n---\n\n".join([
            f"Source: {c.source}\n{c.text}" for c in chunks
        ])
        
        prompt = f"""Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _check_faithfulness(self, response: str,
                            chunks: List[RetrievedChunk]) -> float:
        """Check if response is grounded in context"""
        
        # Simple: check for claims not in context
        context_text = ' '.join([c.text.lower() for c in chunks])
        response_sentences = response.split('.')
        
        grounded = 0
        for sentence in response_sentences:
            # Check if key words appear in context
            words = sentence.lower().split()
            key_words = [w for w in words if len(w) > 5]
            if not key_words:
                grounded += 1
                continue
            
            found = sum(1 for w in key_words if w in context_text)
            if found / len(key_words) > 0.3:
                grounded += 1
        
        return grounded / len(response_sentences) if response_sentences else 0

class AdaptiveRAG:
    """Adapt retrieval strategy based on query"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
    
    def query(self, question: str) -> dict:
        """Adapt parameters based on query complexity"""
        
        complexity = self._assess_complexity(question)
        
        if complexity == 'simple':
            return self.rag.query(question, initial_k=10, final_k=3)
        elif complexity == 'moderate':
            return self.rag.query(question, initial_k=20, final_k=5)
        else:  # complex
            return self.rag.query(question, initial_k=30, final_k=8)
    
    def _assess_complexity(self, question: str) -> str:
        """Assess query complexity"""
        indicators = {
            'complex': ['compare', 'analyze', 'relationship', 'multiple'],
            'simple': ['what is', 'define', 'who is']
        }
        
        q_lower = question.lower()
        
        if any(i in q_lower for i in indicators['complex']):
            return 'complex'
        if any(i in q_lower for i in indicators['simple']):
            return 'simple'
        return 'moderate'
```

**Interview Tips:**
- Reranking typically adds 50-100ms but improves quality significantly
- Retrieve more than you need, then filter
- Monitor faithfulness to detect hallucination
- Adapt strategy to query complexity

---

### Question 22
- [ ] Done
**What techniques work best for chunking and indexing documents in RAG knowledge bases?**

**Answer:**

**Definition:**
Chunking strategies: **fixed-size** (simple, consistent), **semantic** (by topic/section), **recursive** (hierarchical), **sentence-based** (natural boundaries). Indexing: **vector index** for semantic search, **keyword index** for exact match, **metadata index** for filtering.

**Chunking Strategies:**

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Fixed-size** | Split by token/char count | General purpose |
| **Sentence** | Split at sentence boundaries | Short-answer QA |
| **Paragraph** | Split at paragraph breaks | Structured documents |
| **Semantic** | Split by topic/heading | Long documents |
| **Recursive** | Hierarchical splitting | Complex documents |

**Python Code Example:**
```python
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: dict
    start_idx: int
    end_idx: int

class DocumentChunker:
    """Multiple chunking strategies"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_fixed(self, text: str, 
                    metadata: dict = None) -> List[Chunk]:
        """Fixed-size chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:  # Skip tiny final chunks
                continue
            
            chunk_text = ' '.join(chunk_words)
            chunks.append(Chunk(
                text=chunk_text,
                metadata=metadata or {},
                start_idx=i,
                end_idx=i + len(chunk_words)
            ))
        
        return chunks
    
    def chunk_sentence(self, text: str,
                       metadata: dict = None) -> List[Chunk]:
        """Sentence-based chunking"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence.split())
            
            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunks.append(Chunk(
                    text=' '.join(current_chunk),
                    metadata=metadata or {},
                    start_idx=0, end_idx=0  # Simplified
                ))
                # Overlap: keep last sentence
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_len = len(current_chunk[0].split()) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_len += sentence_len
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(Chunk(
                text=' '.join(current_chunk),
                metadata=metadata or {},
                start_idx=0, end_idx=0
            ))
        
        return chunks
    
    def chunk_semantic(self, text: str, 
                       metadata: dict = None) -> List[Chunk]:
        """Semantic chunking by headings/sections"""
        # Split by headings (markdown style)
        sections = re.split(r'\n(?=#+\s)', text)
        
        chunks = []
        for section in sections:
            if not section.strip():
                continue
            
            # Extract heading if present
            heading_match = re.match(r'^(#+)\s*(.+?)\n', section)
            section_metadata = dict(metadata or {})
            
            if heading_match:
                section_metadata['heading'] = heading_match.group(2)
                section_metadata['heading_level'] = len(heading_match.group(1))
            
            # If section is too long, sub-chunk it
            if len(section.split()) > self.chunk_size:
                sub_chunks = self.chunk_fixed(section, section_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=section,
                    metadata=section_metadata,
                    start_idx=0, end_idx=0
                ))
        
        return chunks
    
    def chunk_recursive(self, text: str,
                        metadata: dict = None,
                        separators: List[str] = None) -> List[Chunk]:
        """Recursive chunking with fallback separators"""
        
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ']
        
        if len(text.split()) <= self.chunk_size:
            return [Chunk(text=text, metadata=metadata or {},
                         start_idx=0, end_idx=0)]
        
        chunks = []
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                
                for part in parts:
                    if len(part.split()) <= self.chunk_size:
                        if part.strip():
                            chunks.append(Chunk(
                                text=part,
                                metadata=metadata or {},
                                start_idx=0, end_idx=0
                            ))
                    else:
                        # Recurse with next separator
                        remaining_seps = separators[separators.index(sep)+1:]
                        if remaining_seps:
                            sub_chunks = self.chunk_recursive(
                                part, metadata, remaining_seps
                            )
                            chunks.extend(sub_chunks)
                        else:
                            # Force split
                            chunks.extend(self.chunk_fixed(part, metadata))
                
                if chunks:
                    return chunks
        
        # Fallback to fixed-size
        return self.chunk_fixed(text, metadata)

class DocumentIndexer:
    """Index chunks for retrieval"""
    
    def __init__(self, embedder, vector_db):
        self.embedder = embedder
        self.vector_db = vector_db
        self.chunker = DocumentChunker()
    
    def index_document(self, doc_id: str, text: str,
                       metadata: dict, strategy: str = 'semantic'):
        """Chunk and index a document"""
        
        # Choose chunking strategy
        if strategy == 'fixed':
            chunks = self.chunker.chunk_fixed(text, metadata)
        elif strategy == 'sentence':
            chunks = self.chunker.chunk_sentence(text, metadata)
        elif strategy == 'semantic':
            chunks = self.chunker.chunk_semantic(text, metadata)
        else:
            chunks = self.chunker.chunk_recursive(text, metadata)
        
        # Index each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedder.embed(chunk.text)
            
            self.vector_db.add(
                id=chunk_id,
                embedding=embedding,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    'doc_id': doc_id,
                    'chunk_idx': i
                }
            )
        
        return len(chunks)
```

**Interview Tips:**
- Overlap prevents losing context at boundaries
- Semantic chunking preserves document structure
- Different document types need different strategies
- Index metadata for efficient filtering

---

### Question 23
- [ ] Done
**How do you determine optimal chunk size and overlap for different document types?**

**Answer:**

**Definition:**
Optimal chunk size depends on: **query type** (specific vs broad), **document structure** (dense vs sparse info), **embedding model** (context window), **answer length** (single fact vs explanation). General guidance: 256-512 tokens for QA, 512-1024 for analysis.

**Chunk Size Guidelines:**

| Document Type | Recommended Size | Overlap | Reason |
|---------------|-----------------|---------|--------|
| **FAQ** | 128-256 | 0-20 | Self-contained answers |
| **Technical docs** | 512-768 | 50-100 | Need context |
| **Legal/Contracts** | 256-512 | 50 | Precise clauses |
| **News articles** | 512-1024 | 100 | Narrative flow |
| **Code** | By function | 0 | Logical units |

**Python Code Example:**
```python
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int
    overlap: int
    strategy: str

class ChunkOptimizer:
    """Find optimal chunk configuration"""
    
    def __init__(self, embedder, retriever, evaluator):
        self.embedder = embedder
        self.retriever = retriever
        self.evaluator = evaluator
    
    def optimize(self, documents: List[str],
                 eval_queries: List[dict],
                 chunk_sizes: List[int] = None,
                 overlaps: List[int] = None) -> ChunkConfig:
        """Grid search for optimal configuration"""
        
        if chunk_sizes is None:
            chunk_sizes = [128, 256, 512, 768, 1024]
        if overlaps is None:
            overlaps = [0, 25, 50, 100]
        
        best_config = None
        best_score = 0
        
        results = []
        
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue
                
                # Index with this configuration
                config = ChunkConfig(size, overlap, 'fixed')
                self._reindex(documents, config)
                
                # Evaluate
                score = self._evaluate(eval_queries)
                results.append({
                    'chunk_size': size,
                    'overlap': overlap,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config
        
        return best_config, results
    
    def _reindex(self, documents: List[str], config: ChunkConfig):
        """Reindex with new configuration"""
        self.retriever.clear()
        
        for i, doc in enumerate(documents):
            chunks = self._chunk(doc, config)
            for j, chunk in enumerate(chunks):
                embedding = self.embedder.embed(chunk)
                self.retriever.add(f"doc{i}_chunk{j}", embedding, chunk)
    
    def _chunk(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk with configuration"""
        words = text.split()
        chunks = []
        
        step = config.chunk_size - config.overlap
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + config.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _evaluate(self, eval_queries: List[dict]) -> float:
        """Evaluate retrieval quality"""
        scores = []
        
        for query in eval_queries:
            results = self.retriever.search(query['question'], k=5)
            
            # Check if relevant info is retrieved
            retrieved_text = ' '.join([r['text'] for r in results])
            
            if 'expected_content' in query:
                # Check for expected content
                found = query['expected_content'].lower() in retrieved_text.lower()
                scores.append(1.0 if found else 0.0)
        
        return np.mean(scores) if scores else 0
    
    def analyze_document(self, document: str) -> ChunkConfig:
        """Recommend config based on document analysis"""
        
        # Analyze document characteristics
        words = document.split()
        sentences = document.split('.')
        paragraphs = document.split('\n\n')
        
        avg_sentence_len = len(words) / len(sentences) if sentences else 0
        avg_paragraph_len = len(words) / len(paragraphs) if paragraphs else 0
        
        # Heuristics for recommendation
        if avg_paragraph_len < 200:
            # Short paragraphs - use paragraph-based
            return ChunkConfig(
                chunk_size=int(avg_paragraph_len * 2),
                overlap=int(avg_paragraph_len * 0.2),
                strategy='paragraph'
            )
        elif avg_sentence_len > 30:
            # Long sentences (legal/technical) - smaller chunks
            return ChunkConfig(
                chunk_size=256,
                overlap=50,
                strategy='sentence'
            )
        else:
            # Default
            return ChunkConfig(
                chunk_size=512,
                overlap=50,
                strategy='fixed'
            )
    
    def validate_config(self, documents: List[str],
                        config: ChunkConfig) -> Dict:
        """Validate configuration produces good chunks"""
        
        all_chunks = []
        for doc in documents:
            chunks = self._chunk(doc, config)
            all_chunks.extend(chunks)
        
        chunk_sizes = [len(c.split()) for c in all_chunks]
        
        return {
            'total_chunks': len(all_chunks),
            'avg_size': np.mean(chunk_sizes),
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'std_size': np.std(chunk_sizes),
            'too_small': sum(1 for s in chunk_sizes if s < 50),
            'too_large': sum(1 for s in chunk_sizes if s > config.chunk_size * 1.5)
        }
```

**Optimization Process:**
```
1. Start with 512 tokens, 50 overlap
2. Evaluate on test queries
3. If missing context → increase size or overlap
4. If too noisy → decrease size
5. Iterate until recall@k is acceptable
```

**Interview Tips:**
- Smaller chunks = higher precision, lower recall
- Overlap prevents context loss at boundaries
- Test with representative queries, not just accuracy
- Different document types may need different configs

---

### Question 24
- [ ] Done
**Explain dense retrieval vs sparse retrieval (BM25) and when to use hybrid approaches.**

**Answer:**

**Definition:**
- **Sparse (BM25)**: keyword matching, term frequency-based, exact matching
- **Dense**: semantic embeddings, meaning-based, handles synonyms
- **Hybrid**: combine both for best of both worlds

**Comparison:**

| Aspect | Sparse (BM25) | Dense (Embeddings) |
|--------|---------------|--------------------|
| **Matching** | Exact keywords | Semantic meaning |
| **Synonyms** | ❌ | ✅ |
| **Speed** | Very fast | Fast (with ANN) |
| **Index size** | Small | Large |
| **Rare terms** | ✅ Strong | ❌ May miss |
| **Cross-lingual** | ❌ | ✅ |

**When to Use Each:**

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Entity search (names, codes) | BM25 | Exact match critical |
| Conceptual search | Dense | Semantic understanding |
| Mixed queries | Hybrid | Best of both |
| Limited compute | BM25 | No GPU needed |
| Multilingual | Dense | Cross-lingual embeddings |

**Python Code Example:**
```python
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict

class HybridRetriever:
    """Combine sparse and dense retrieval"""
    
    def __init__(self, embedder, vector_db):
        self.embedder = embedder
        self.vector_db = vector_db
        self.documents = []
        self.bm25 = None
    
    def index(self, documents: List[Dict]):
        """Index documents for both retrieval methods"""
        self.documents = documents
        
        # BM25 index
        tokenized = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Dense index
        for doc in documents:
            embedding = self.embedder.embed(doc['text'])
            self.vector_db.add(
                id=doc['id'],
                embedding=embedding,
                metadata=doc
            )
    
    def search_bm25(self, query: str, k: int) -> List[Dict]:
        """Sparse retrieval with BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [
            {
                'doc': self.documents[i],
                'score': scores[i],
                'method': 'bm25'
            }
            for i in top_indices
        ]
    
    def search_dense(self, query: str, k: int) -> List[Dict]:
        """Dense retrieval with embeddings"""
        query_embedding = self.embedder.embed(query)
        results = self.vector_db.search(query_embedding, k=k)
        
        return [
            {
                'doc': r['metadata'],
                'score': r['score'],
                'method': 'dense'
            }
            for r in results
        ]
    
    def search_hybrid(self, query: str, k: int,
                      alpha: float = 0.5,
                      fusion: str = 'rrf') -> List[Dict]:
        """
        Hybrid search combining BM25 and dense
        alpha: weight for dense (1-alpha for BM25)
        fusion: 'rrf' (Reciprocal Rank Fusion) or 'linear'
        """
        
        bm25_results = self.search_bm25(query, k * 2)
        dense_results = self.search_dense(query, k * 2)
        
        if fusion == 'rrf':
            return self._rrf_fusion(bm25_results, dense_results, k)
        else:
            return self._linear_fusion(bm25_results, dense_results, k, alpha)
    
    def _rrf_fusion(self, bm25_results: List, dense_results: List,
                    k: int, rrf_k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        scores = {}
        doc_map = {}
        
        for rank, result in enumerate(bm25_results):
            doc_id = result['doc']['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[doc_id] = result['doc']
        
        for rank, result in enumerate(dense_results):
            doc_id = result['doc']['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[doc_id] = result['doc']
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            {
                'doc': doc_map[doc_id],
                'score': scores[doc_id],
                'method': 'hybrid_rrf'
            }
            for doc_id in sorted_ids[:k]
        ]
    
    def _linear_fusion(self, bm25_results: List, dense_results: List,
                       k: int, alpha: float) -> List[Dict]:
        """Linear combination of normalized scores"""
        
        # Normalize scores
        def normalize(results):
            if not results:
                return {}
            scores = [r['score'] for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r['doc']['id']: 0.5 for r in results}
            return {
                r['doc']['id']: (r['score'] - min_s) / (max_s - min_s)
                for r in results
            }
        
        bm25_norm = normalize(bm25_results)
        dense_norm = normalize(dense_results)
        
        # Combine
        doc_map = {}
        for r in bm25_results + dense_results:
            doc_map[r['doc']['id']] = r['doc']
        
        all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        combined = {
            doc_id: (1 - alpha) * bm25_norm.get(doc_id, 0) + 
                    alpha * dense_norm.get(doc_id, 0)
            for doc_id in all_ids
        }
        
        sorted_ids = sorted(combined.keys(), 
                           key=lambda x: combined[x], reverse=True)
        
        return [
            {
                'doc': doc_map[doc_id],
                'score': combined[doc_id],
                'method': 'hybrid_linear'
            }
            for doc_id in sorted_ids[:k]
        ]
```

**Interview Tips:**
- RRF is simple and effective for fusion
- BM25 excels at exact entity/keyword matching
- Dense handles paraphrases and semantic similarity
- Hybrid typically outperforms either alone by 5-15%

---

### Question 25
- [ ] Done
**How do you design RAG architectures that maintain factual accuracy and reduce hallucination?**

**Answer:**

**Definition:**
Reduce hallucination via: **grounding instructions** (use only provided context), **source attribution** (cite sources), **confidence signals** (acknowledge uncertainty), **verification** (fact-check against context), **retrieval quality** (better context = less need to hallucinate).

**Anti-Hallucination Strategies:**

| Strategy | Implementation | Impact |
|----------|---------------|--------|
| **Explicit grounding** | "Only use provided context" | High |
| **Citation requirement** | "Cite source for each claim" | Medium |
| **Uncertainty acknowledgment** | "Say 'not in context' if unsure" | High |
| **Retrieval quality** | Reranking, more chunks | High |
| **Verification step** | Cross-check with context | Medium |

**Python Code Example:**
```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class FactualResponse:
    answer: str
    sources: List[str]
    confidence: float
    grounded: bool
    verification_notes: List[str]

class FactualRAG:
    """RAG system designed to minimize hallucination"""
    
    def __init__(self, retriever, llm, verifier_llm=None):
        self.retriever = retriever
        self.llm = llm
        self.verifier_llm = verifier_llm or llm
    
    def query(self, question: str, k: int = 5) -> FactualResponse:
        """Generate grounded, verified response"""
        
        # Step 1: Retrieve context
        chunks = self.retriever.search(question, k=k)
        context = self._format_context(chunks)
        
        # Step 2: Generate with grounding instructions
        response = self._generate_grounded(question, context)
        
        # Step 3: Verify claims against context
        verification = self._verify_claims(response, context)
        
        # Step 4: Extract sources cited
        sources = self._extract_sources(response, chunks)
        
        return FactualResponse(
            answer=response,
            sources=sources,
            confidence=verification['confidence'],
            grounded=verification['all_grounded'],
            verification_notes=verification['notes']
        )
    
    def _format_context(self, chunks: list) -> str:
        """Format context with source labels"""
        formatted = []
        for i, chunk in enumerate(chunks):
            source = chunk.get('source', f'Source {i+1}')
            formatted.append(f"[{source}]:\n{chunk['text']}")
        return "\n\n---\n\n".join(formatted)
    
    def _generate_grounded(self, question: str, context: str) -> str:
        """Generate with strong grounding instructions"""
        
        prompt = f"""You are a helpful assistant that ONLY provides information from the given context.

CRITICAL RULES:
1. ONLY use information explicitly stated in the context below
2. If the context doesn't contain the answer, say "The provided documents do not contain this information"
3. NEVER make up facts, dates, numbers, or details not in the context
4. For each claim, cite the source in brackets like [Source name]
5. If you're uncertain, say "Based on the context, it appears that..." rather than stating definitively

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER (remember to cite sources and only use information from the context):"""
        
        return self.llm.generate(prompt, temperature=0.2)
    
    def _verify_claims(self, response: str, context: str) -> dict:
        """Verify each claim is grounded in context"""
        
        verification_prompt = f"""Analyze the following response for factual accuracy based on the context.

Context:
{context}

Response to verify:
{response}

For each factual claim in the response:
1. Is it supported by the context? (yes/no/partial)
2. Quote the supporting text if yes

Return as a list:
- Claim: [claim]
  Supported: [yes/no/partial]
  Evidence: [quote or "not found"]

Verification:"""
        
        verification_response = self.verifier_llm.generate(
            verification_prompt, temperature=0
        )
        
        # Parse verification
        notes = []
        supported_count = 0
        total_claims = 0
        
        for line in verification_response.split('\n'):
            if 'Supported:' in line:
                total_claims += 1
                if 'yes' in line.lower():
                    supported_count += 1
                elif 'no' in line.lower():
                    notes.append(f"Unsupported claim found")
        
        confidence = supported_count / total_claims if total_claims > 0 else 0
        
        return {
            'all_grounded': confidence >= 0.9,
            'confidence': confidence,
            'notes': notes
        }
    
    def _extract_sources(self, response: str, chunks: list) -> List[str]:
        """Extract cited sources from response"""
        # Find citations like [Source name]
        citations = re.findall(r'\[([^\]]+)\]', response)
        
        # Map to actual sources
        sources = []
        for citation in citations:
            for chunk in chunks:
                if citation in str(chunk.get('source', '')):
                    sources.append(chunk.get('source'))
                    break
        
        return list(set(sources))

class HallucinationDetector:
    """Detect potential hallucinations in responses"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect(self, response: str, context: str) -> dict:
        """Detect potential hallucinations"""
        
        # Heuristic checks
        issues = []
        
        # Check for specific numbers not in context
        response_numbers = re.findall(r'\b\d{4,}\b|\$[\d,]+|\d+\.\d+%', response)
        context_lower = context.lower()
        
        for num in response_numbers:
            if num not in context:
                issues.append(f"Number not in context: {num}")
        
        # Check for strong claims
        strong_claims = ['definitely', 'certainly', 'proven', 'always', 'never']
        for claim in strong_claims:
            if claim in response.lower():
                issues.append(f"Strong claim word: {claim}")
        
        # Check for uncertainty acknowledgment (good sign)
        uncertainty = ['may', 'might', 'appears', 'suggests', 'based on']
        has_uncertainty = any(u in response.lower() for u in uncertainty)
        
        return {
            'potential_issues': issues,
            'issue_count': len(issues),
            'has_hedging': has_uncertainty,
            'risk_level': 'high' if len(issues) > 2 else 
                         'medium' if issues else 'low'
        }
```

**Interview Tips:**
- Explicit grounding instructions are most effective
- Require citations for accountability
- Two-stage verify: generate then check
- Monitor hallucination rate as key metric

---

### Question 26
- [ ] Done
**What approaches work best for real-time RAG with sub-second latency requirements?**

**Answer:**

**Definition:**
Sub-second RAG: optimize every stage - **fast embedding** (cached/quantized), **efficient retrieval** (ANN indexes), **minimal reranking** (or skip), **streaming generation**, **result caching**, **connection pooling**. Target: <100ms retrieval + <500ms generation.

**Latency Breakdown:**

| Stage | Typical | Optimized | Technique |
|-------|---------|-----------|----------|
| **Embedding** | 50-100ms | 10-20ms | Caching, quantization |
| **Retrieval** | 20-50ms | 5-10ms | HNSW, less k |
| **Reranking** | 100-200ms | 0-50ms | Skip or lightweight |
| **Generation** | 500-2000ms | 300-500ms | Streaming, smaller model |
| **Total** | 700-2400ms | 320-600ms | |

**Python Code Example:**
```python
import asyncio
import time
from typing import Optional
from functools import lru_cache
import hashlib

class LowLatencyRAG:
    """Optimized for sub-second response"""
    
    def __init__(self, embedder, vector_db, llm,
                 cache_size: int = 10000):
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
        self.embedding_cache = {}
        self.result_cache = {}
    
    async def query_async(self, question: str,
                          k: int = 3,
                          skip_rerank: bool = True) -> dict:
        """Async query for maximum speed"""
        
        start = time.perf_counter()
        timings = {}
        
        # Check result cache first
        cache_key = self._cache_key(question, k)
        if cache_key in self.result_cache:
            return {
                **self.result_cache[cache_key],
                'cached': True,
                'total_ms': 0
            }
        
        # Fast embedding (cached)
        t0 = time.perf_counter()
        query_embedding = self._get_embedding_cached(question)
        timings['embedding_ms'] = (time.perf_counter() - t0) * 1000
        
        # Fast retrieval
        t0 = time.perf_counter()
        chunks = await self._retrieve_async(query_embedding, k)
        timings['retrieval_ms'] = (time.perf_counter() - t0) * 1000
        
        # Skip reranking for speed (or use lightweight reranker)
        if not skip_rerank:
            t0 = time.perf_counter()
            chunks = self._lightweight_rerank(question, chunks)
            timings['rerank_ms'] = (time.perf_counter() - t0) * 1000
        
        # Generate (streaming for perceived speed)
        t0 = time.perf_counter()
        response = await self._generate_async(question, chunks)
        timings['generation_ms'] = (time.perf_counter() - t0) * 1000
        
        total_ms = (time.perf_counter() - start) * 1000
        
        result = {
            'answer': response,
            'chunks_used': len(chunks),
            'timings': timings,
            'total_ms': total_ms,
            'cached': False
        }
        
        # Cache result
        self.result_cache[cache_key] = result
        
        return result
    
    def _get_embedding_cached(self, text: str):
        """Get embedding with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = self.embedder.embed(text)
        
        return self.embedding_cache[text_hash]
    
    async def _retrieve_async(self, embedding, k: int):
        """Async retrieval"""
        # In production: use async client
        return self.vector_db.search(embedding, k=k)
    
    def _lightweight_rerank(self, query: str, chunks: list) -> list:
        """Fast reranking using simple heuristics"""
        # Score by keyword overlap (much faster than cross-encoder)
        query_words = set(query.lower().split())
        
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((chunk, chunk['score'] + overlap * 0.1))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored]
    
    async def _generate_async(self, question: str, chunks: list) -> str:
        """Async generation with minimal prompt"""
        
        # Minimal context for speed
        context = "\n\n".join([c['text'][:500] for c in chunks[:3]])
        
        prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"""
        
        # In production: use streaming for perceived speed
        return await self.llm.generate_async(prompt)
    
    def _cache_key(self, question: str, k: int) -> str:
        return hashlib.md5(f"{question}:{k}".encode()).hexdigest()

class StreamingRAG:
    """RAG with streaming response for perceived speed"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    async def stream_query(self, question: str):
        """Stream response for perceived low latency"""
        
        # Retrieve in parallel with first token
        retrieval_task = asyncio.create_task(
            self.retriever.search_async(question, k=3)
        )
        
        # Start generating while retrieving
        # (using question-only generation initially)
        yield "Searching knowledge base...\n\n"
        
        chunks = await retrieval_task
        
        # Now stream the real answer
        context = "\n".join([c['text'] for c in chunks])
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        async for token in self.llm.stream_generate(prompt):
            yield token

# Optimization techniques
class QueryOptimizations:
    @staticmethod
    def truncate_query(query: str, max_tokens: int = 50) -> str:
        """Truncate long queries for faster embedding"""
        words = query.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return query
    
    @staticmethod
    def precompute_common_queries(queries: list, rag_system):
        """Warm cache with common queries"""
        for query in queries:
            rag_system._get_embedding_cached(query)
```

**Interview Tips:**
- Caching has highest ROI for common queries
- Skip reranking if latency is critical
- Streaming improves perceived latency significantly
- Monitor P99 latency, not just average

---

## RAG Retrieval Strategies

### Question 27
- [ ] Done
**How do you implement re-ranking strategies to improve retrieval quality?**

**Answer:**

**Definition:**
Reranking: take initial retrieval results and **reorder by more sophisticated scoring**. Methods: **cross-encoder** (most accurate), **LLM-based** (flexible), **rule-based** (fast), **learned** (custom). Typically improves precision by 10-20%.

**Reranking Methods:**

| Method | How It Works | Latency | Quality |
|--------|--------------|---------|--------|
| **Cross-encoder** | Joint query-doc encoding | 50-200ms | Best |
| **LLM reranker** | Prompt LLM to score | 200-500ms | Very good |
| **ColBERT** | Token-level interaction | 20-50ms | Good |
| **Keyword boost** | Term overlap scoring | <5ms | Moderate |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RankedResult:
    id: str
    text: str
    original_score: float
    rerank_score: float
    final_rank: int

class Reranker:
    """Multiple reranking strategies"""
    
    def __init__(self, cross_encoder=None, llm=None):
        self.cross_encoder = cross_encoder
        self.llm = llm
    
    def rerank_cross_encoder(self, query: str,
                              candidates: List[dict],
                              top_k: int = None) -> List[RankedResult]:
        """Rerank using cross-encoder model"""
        
        if not self.cross_encoder:
            raise ValueError("Cross-encoder not configured")
        
        # Score each query-document pair
        pairs = [(query, c['text']) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Create ranked results
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            results.append(RankedResult(
                id=candidate['id'],
                text=candidate['text'],
                original_score=candidate.get('score', 0),
                rerank_score=float(score),
                final_rank=0  # Will be set after sorting
            ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Assign final ranks
        for i, r in enumerate(results):
            r.final_rank = i + 1
        
        return results[:top_k] if top_k else results
    
    def rerank_llm(self, query: str,
                   candidates: List[dict],
                   top_k: int = None) -> List[RankedResult]:
        """Rerank using LLM scoring"""
        
        if not self.llm:
            raise ValueError("LLM not configured")
        
        # Format candidates for LLM
        docs_text = ""
        for i, c in enumerate(candidates):
            docs_text += f"\n[{i+1}] {c['text'][:300]}...\n"
        
        prompt = f"""Given the query, rank these documents by relevance.

Query: {query}

Documents:
{docs_text}

Return rankings as comma-separated document numbers (most relevant first).
Example: 3, 1, 5, 2, 4

Ranking:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        # Parse ranking
        try:
            ranking = [int(x.strip()) - 1 for x in response.split(',')]
        except:
            ranking = list(range(len(candidates)))  # Fallback to original
        
        # Create results in ranked order
        results = []
        for rank, idx in enumerate(ranking):
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                results.append(RankedResult(
                    id=c['id'],
                    text=c['text'],
                    original_score=c.get('score', 0),
                    rerank_score=len(candidates) - rank,
                    final_rank=rank + 1
                ))
        
        return results[:top_k] if top_k else results
    
    def rerank_keyword(self, query: str,
                       candidates: List[dict],
                       boost_weight: float = 0.3) -> List[RankedResult]:
        """Fast keyword-based reranking"""
        
        query_terms = set(query.lower().split())
        
        results = []
        for c in candidates:
            doc_terms = set(c['text'].lower().split())
            
            # Calculate overlap
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            
            # Combine with original score
            original = c.get('score', 0)
            rerank_score = original + boost_weight * overlap
            
            results.append(RankedResult(
                id=c['id'],
                text=c['text'],
                original_score=original,
                rerank_score=rerank_score,
                final_rank=0
            ))
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(results):
            r.final_rank = i + 1
        
        return results
    
    def rerank_ensemble(self, query: str,
                        candidates: List[dict],
                        methods: List[str] = None) -> List[RankedResult]:
        """Combine multiple reranking methods"""
        
        if methods is None:
            methods = ['keyword']  # Default to fast method
            if self.cross_encoder:
                methods.append('cross_encoder')
        
        # Get rankings from each method
        all_rankings = {}
        
        for method in methods:
            if method == 'keyword':
                ranked = self.rerank_keyword(query, candidates)
            elif method == 'cross_encoder' and self.cross_encoder:
                ranked = self.rerank_cross_encoder(query, candidates)
            elif method == 'llm' and self.llm:
                ranked = self.rerank_llm(query, candidates)
            else:
                continue
            
            # Store rank positions
            for r in ranked:
                if r.id not in all_rankings:
                    all_rankings[r.id] = {'doc': r, 'ranks': []}
                all_rankings[r.id]['ranks'].append(r.final_rank)
        
        # Average ranks (lower is better)
        final_scores = [
            (data['doc'], np.mean(data['ranks']))
            for data in all_rankings.values()
        ]
        final_scores.sort(key=lambda x: x[1])
        
        # Create final results
        results = []
        for i, (doc, avg_rank) in enumerate(final_scores):
            doc.rerank_score = -avg_rank  # Negative because lower rank is better
            doc.final_rank = i + 1
            results.append(doc)
        
        return results
```

**Interview Tips:**
- Cross-encoder is gold standard but slow
- Rerank more candidates than you need (retrieve 20, rerank to top 5)
- Ensemble methods can outperform single rerankers
- Monitor reranking latency vs quality trade-off

---

### Question 28
- [ ] Done
**What is HyDE (Hypothetical Document Embeddings) and when does it improve retrieval?**

**Answer:**

**Definition:**
**HyDE**: generate a hypothetical answer to the query, then embed THAT instead of the query. Bridges the gap between question semantics and document semantics. Helps when queries are short/vague but documents contain the answer phrased differently.

**How HyDE Works:**
```
Query: "climate change effects"
    ↓ LLM generates
Hypothetical doc: "Climate change causes rising sea levels,
                   extreme weather events, and biodiversity loss..."
    ↓ Embed this
Search with hypothetical doc embedding (closer to actual docs)
```

**When HyDE Helps:**

| Scenario | HyDE Benefit | Why |
|----------|-------------|-----|
| **Short queries** | High | Expands query semantics |
| **Conceptual questions** | High | Generates relevant terms |
| **FAQ matching** | Medium | Query ≠ answer format |
| **Specific entity search** | Low | Original query is precise |
| **Real-time requirements** | Low | Adds LLM latency |

**Python Code Example:**
```python
import numpy as np
from typing import List

class HyDERetriever:
    """Hypothetical Document Embeddings retrieval"""
    
    def __init__(self, llm, embedder, vector_db):
        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db
    
    def generate_hypothetical(self, query: str,
                              n_hypotheticals: int = 1) -> List[str]:
        """Generate hypothetical documents for query"""
        
        prompt = f"""Write a detailed paragraph that would answer this question.
Write as if you are an expert providing information in a document.
Do not say "this question asks" - write the actual content.

Question: {query}

Document paragraph:"""
        
        hypotheticals = []
        for _ in range(n_hypotheticals):
            response = self.llm.generate(prompt, temperature=0.7)
            hypotheticals.append(response)
        
        return hypotheticals
    
    def search(self, query: str, k: int = 5,
               use_hyde: bool = True,
               n_hypotheticals: int = 1) -> List[dict]:
        """Search using HyDE or standard embedding"""
        
        if use_hyde:
            # Generate hypothetical documents
            hypotheticals = self.generate_hypothetical(
                query, n_hypotheticals
            )
            
            # Embed hypotheticals and average
            embeddings = [self.embedder.embed(h) for h in hypotheticals]
            query_embedding = np.mean(embeddings, axis=0)
        else:
            # Standard query embedding
            query_embedding = self.embedder.embed(query)
        
        # Search
        return self.vector_db.search(query_embedding, k=k)
    
    def search_hybrid(self, query: str, k: int = 5,
                      hyde_weight: float = 0.5) -> List[dict]:
        """Combine HyDE and standard search"""
        
        # Standard embedding
        standard_emb = self.embedder.embed(query)
        
        # HyDE embedding
        hypothetical = self.generate_hypothetical(query, 1)[0]
        hyde_emb = self.embedder.embed(hypothetical)
        
        # Weighted combination
        combined_emb = (
            (1 - hyde_weight) * standard_emb +
            hyde_weight * hyde_emb
        )
        
        return self.vector_db.search(combined_emb, k=k)

class AdaptiveHyDE:
    """Use HyDE only when beneficial"""
    
    def __init__(self, hyde_retriever: HyDERetriever):
        self.hyde = hyde_retriever
    
    def should_use_hyde(self, query: str) -> bool:
        """Decide whether HyDE will help for this query"""
        
        # Short queries benefit more from HyDE
        query_words = query.split()
        if len(query_words) < 5:
            return True
        
        # Conceptual questions benefit
        conceptual_terms = ['how', 'why', 'explain', 'what is']
        if any(term in query.lower() for term in conceptual_terms):
            return True
        
        # Entity-specific queries don't need HyDE
        # (Entities are usually proper nouns - capitalized)
        capitalized = sum(1 for w in query_words if w[0].isupper())
        if capitalized > len(query_words) * 0.3:
            return False
        
        return False
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """Adaptively use HyDE based on query"""
        
        use_hyde = self.should_use_hyde(query)
        return self.hyde.search(query, k=k, use_hyde=use_hyde)

class MultiQueryHyDE:
    """Generate multiple query variations"""
    
    def __init__(self, llm, embedder, vector_db):
        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db
    
    def generate_variations(self, query: str, n: int = 3) -> List[str]:
        """Generate query variations and hypothetical docs"""
        
        prompt = f"""Generate {n} different ways to express this query, plus a hypothetical answer.

Original query: {query}

Variations (one per line):
1. [paraphrase 1]
2. [paraphrase 2]
3. [paraphrase 3]

Hypothetical answer paragraph:
[detailed answer]"""
        
        response = self.llm.generate(prompt)
        
        # Parse variations
        lines = response.split('\n')
        variations = [query]  # Include original
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.')):
                variation = line.split('.', 1)[-1].strip()
                if variation:
                    variations.append(variation)
        
        return variations
    
    def search_multi(self, query: str, k: int = 5) -> List[dict]:
        """Search with multiple query variations"""
        
        variations = self.generate_variations(query)
        
        # Embed all variations
        embeddings = [self.embedder.embed(v) for v in variations]
        avg_embedding = np.mean(embeddings, axis=0)
        
        return self.vector_db.search(avg_embedding, k=k)
```

**Interview Tips:**
- HyDE adds LLM latency (~200-500ms)
- Most effective for short, conceptual queries
- Can combine with standard search for robustness
- Not helpful when query already contains answer terms

---

### Question 29
- [ ] Done
**How do you implement dynamic retrieval strategies that adapt to query complexity?**

**Answer:**

**Definition:**
**Adaptive retrieval**: adjust retrieval parameters based on query characteristics. Simple queries → fewer chunks, fast path. Complex queries → more chunks, reranking, multi-step. Key: classify query first, then route to appropriate strategy.

**Adaptation Dimensions:**

| Query Type | k | Rerank | Strategy |
|------------|---|--------|----------|
| **Factual (simple)** | 3 | No | Direct search |
| **Analytical** | 5-10 | Yes | Multi-source |
| **Multi-hop** | 10+ | Yes | Iterative |
| **Comparative** | 8+ | Yes | Topic grouping |

**Python Code Example:**
```python
from typing import List, Dict
from enum import Enum
from dataclasses import dataclass

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"

@dataclass
class RetrievalConfig:
    k: int
    use_rerank: bool
    use_hyde: bool
    iterative: bool
    max_iterations: int = 1

class AdaptiveRetriever:
    """Adapt retrieval strategy to query complexity"""
    
    def __init__(self, base_retriever, reranker, hyde_retriever, llm):
        self.retriever = base_retriever
        self.reranker = reranker
        self.hyde = hyde_retriever
        self.llm = llm
        
        # Strategy configurations
        self.configs = {
            QueryComplexity.SIMPLE: RetrievalConfig(
                k=3, use_rerank=False, use_hyde=False, iterative=False
            ),
            QueryComplexity.MODERATE: RetrievalConfig(
                k=5, use_rerank=True, use_hyde=False, iterative=False
            ),
            QueryComplexity.COMPLEX: RetrievalConfig(
                k=10, use_rerank=True, use_hyde=True, iterative=False
            ),
            QueryComplexity.MULTI_HOP: RetrievalConfig(
                k=5, use_rerank=True, use_hyde=False, 
                iterative=True, max_iterations=3
            )
        }
    
    def classify_query(self, query: str) -> QueryComplexity:
        """Classify query complexity"""
        
        query_lower = query.lower()
        
        # Multi-hop indicators
        multi_hop = ['and then', 'after that', 'relationship between',
                     'compare', 'contrast', 'how does X affect Y']
        if any(indicator in query_lower for indicator in multi_hop):
            return QueryComplexity.MULTI_HOP
        
        # Complex indicators
        complex_terms = ['analyze', 'explain in detail', 'comprehensive',
                        'multiple factors', 'trade-offs']
        if any(term in query_lower for term in complex_terms):
            return QueryComplexity.COMPLEX
        
        # Moderate indicators
        moderate = ['how', 'why', 'what are the', 'explain']
        if any(term in query_lower for term in moderate):
            return QueryComplexity.MODERATE
        
        # Default to simple
        return QueryComplexity.SIMPLE
    
    def search(self, query: str) -> Dict:
        """Adaptive search based on query complexity"""
        
        complexity = self.classify_query(query)
        config = self.configs[complexity]
        
        if config.iterative:
            return self._iterative_search(query, config)
        else:
            return self._single_search(query, config)
    
    def _single_search(self, query: str, config: RetrievalConfig) -> Dict:
        """Single-pass retrieval"""
        
        # Choose embedding strategy
        if config.use_hyde:
            results = self.hyde.search(query, k=config.k * 2)
        else:
            results = self.retriever.search(query, k=config.k * 2)
        
        # Rerank if configured
        if config.use_rerank and self.reranker:
            results = self.reranker.rerank(query, results, top_k=config.k)
        else:
            results = results[:config.k]
        
        return {
            'results': results,
            'config_used': config,
            'iterations': 1
        }
    
    def _iterative_search(self, query: str, 
                          config: RetrievalConfig) -> Dict:
        """Multi-step iterative retrieval for multi-hop queries"""
        
        all_results = []
        current_query = query
        seen_ids = set()
        
        for iteration in range(config.max_iterations):
            # Search with current query
            results = self.retriever.search(current_query, k=config.k)
            
            # Add new results
            for r in results:
                if r['id'] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r['id'])
            
            # Check if we have enough information
            if self._has_enough_info(query, all_results):
                break
            
            # Generate follow-up query
            current_query = self._generate_followup(
                query, all_results, iteration
            )
        
        # Final reranking
        if config.use_rerank and self.reranker:
            all_results = self.reranker.rerank(
                query, all_results, top_k=config.k
            )
        
        return {
            'results': all_results[:config.k * 2],
            'config_used': config,
            'iterations': iteration + 1
        }
    
    def _has_enough_info(self, query: str, results: List) -> bool:
        """Check if retrieved info answers the query"""
        if len(results) < 3:
            return False
        
        context = ' '.join([r['text'] for r in results])
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        
        coverage = len(query_terms & context_terms) / len(query_terms)
        return coverage > 0.7
    
    def _generate_followup(self, original_query: str,
                           current_results: List,
                           iteration: int) -> str:
        """Generate follow-up query to fill gaps"""
        
        context = ' '.join([r['text'][:200] for r in current_results[-3:]])
        
        prompt = f"""Original question: {original_query}

Information found so far:
{context}

What additional information is needed to fully answer the question?
Generate a follow-up search query to find the missing information.

Follow-up query:"""
        
        return self.llm.generate(prompt, temperature=0.3)
```

**Interview Tips:**
- Query classification is key to good adaptation
- Over-fetching is better than under-fetching
- Iterative search helps multi-hop reasoning
- Monitor performance per complexity level

---

### Question 30
- [ ] Done
**How do you handle RAG for queries requiring information from multiple documents?**

**Answer:**

**Definition:**
Multi-document RAG: aggregate information across documents for **comparison**, **synthesis**, **comprehensive coverage**. Challenges: deduplication, conflict resolution, source tracking, context window limits. Strategies: clustering, summarization, iterative retrieval.

**Multi-Document Strategies:**

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Top-k fusion** | Combine top chunks from each doc | Coverage |
| **Clustering** | Group similar chunks, pick representatives | Diversity |
| **Iterative** | Retrieve → identify gaps → retrieve more | Multi-hop |
| **Map-reduce** | Summarize each doc, then synthesize | Large corpus |

**Python Code Example:**
```python
import numpy as np
from collections import defaultdict
from typing import List, Dict

class MultiDocumentRAG:
    """Handle queries requiring multiple documents"""
    
    def __init__(self, retriever, llm, embedder):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
    
    def query(self, question: str,
              min_sources: int = 3,
              max_sources: int = 10,
              strategy: str = 'diverse') -> Dict:
        """Multi-document query with source diversity"""
        
        if strategy == 'diverse':
            results = self._diverse_retrieval(question, min_sources, max_sources)
        elif strategy == 'comprehensive':
            results = self._comprehensive_retrieval(question, max_sources)
        elif strategy == 'iterative':
            results = self._iterative_retrieval(question, min_sources)
        else:
            results = self.retriever.search(question, k=max_sources)
        
        # Synthesize answer from multiple sources
        answer = self._synthesize(question, results)
        
        return {
            'answer': answer,
            'sources': self._extract_unique_sources(results),
            'num_chunks': len(results),
            'strategy': strategy
        }
    
    def _diverse_retrieval(self, question: str,
                           min_sources: int,
                           max_chunks: int) -> List[Dict]:
        """Retrieve ensuring diversity of sources"""
        
        # Get more candidates than needed
        candidates = self.retriever.search(question, k=max_chunks * 2)
        
        # Group by source document
        by_source = defaultdict(list)
        for chunk in candidates:
            source = chunk.get('source', chunk.get('doc_id', 'unknown'))
            by_source[source].append(chunk)
        
        # Select top chunk from each source, round-robin
        selected = []
        source_indices = {s: 0 for s in by_source}
        
        while len(selected) < max_chunks:
            added = False
            for source, chunks in by_source.items():
                idx = source_indices[source]
                if idx < len(chunks) and len(selected) < max_chunks:
                    selected.append(chunks[idx])
                    source_indices[source] += 1
                    added = True
            
            if not added:
                break
        
        # Ensure minimum sources
        unique_sources = self._extract_unique_sources(selected)
        if len(unique_sources) < min_sources:
            # Try to add from missing sources
            for source, chunks in by_source.items():
                if source not in unique_sources and chunks:
                    selected.append(chunks[0])
                    if len(self._extract_unique_sources(selected)) >= min_sources:
                        break
        
        return selected
    
    def _comprehensive_retrieval(self, question: str,
                                  max_chunks: int) -> List[Dict]:
        """Retrieve for comprehensive coverage"""
        
        # Decompose question into aspects
        aspects = self._decompose_question(question)
        
        all_results = []
        seen_ids = set()
        
        for aspect in aspects:
            results = self.retriever.search(aspect, k=max_chunks // len(aspects))
            for r in results:
                if r['id'] not in seen_ids:
                    r['aspect'] = aspect
                    all_results.append(r)
                    seen_ids.add(r['id'])
        
        return all_results[:max_chunks]
    
    def _decompose_question(self, question: str) -> List[str]:
        """Break down complex question into aspects"""
        
        prompt = f"""Break down this question into 2-4 specific sub-questions or aspects to search for.

Question: {question}

Aspects (one per line):"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        aspects = [line.strip() for line in response.split('\n') 
                  if line.strip() and not line.strip().startswith('#')]
        
        # Always include original question
        return [question] + aspects[:3]
    
    def _iterative_retrieval(self, question: str,
                              min_sources: int,
                              max_iterations: int = 3) -> List[Dict]:
        """Iteratively retrieve until coverage is sufficient"""
        
        all_results = []
        seen_ids = set()
        current_query = question
        
        for i in range(max_iterations):
            results = self.retriever.search(current_query, k=5)
            
            for r in results:
                if r['id'] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r['id'])
            
            # Check if we have enough diverse sources
            sources = self._extract_unique_sources(all_results)
            if len(sources) >= min_sources:
                break
            
            # Generate follow-up query
            current_query = self._generate_gap_query(question, all_results)
        
        return all_results
    
    def _generate_gap_query(self, question: str, 
                            current_results: List[Dict]) -> str:
        """Generate query to find missing information"""
        
        context = '\n'.join([r['text'][:150] for r in current_results[-3:]])
        
        prompt = f"""Question: {question}

Information already found:
{context}

What additional search query would find missing information?

Search query:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _synthesize(self, question: str, chunks: List[Dict]) -> str:
        """Synthesize answer from multiple sources"""
        
        # Format chunks with source attribution
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get('source', f'Source {i+1}')
            context_parts.append(f"[{source}]: {chunk['text']}")
        
        context = '\n\n'.join(context_parts)
        
        prompt = f"""Based on the following sources, provide a comprehensive answer.
Synthesize information from multiple sources where relevant.
Cite sources using [Source name] when making claims.
Note any conflicting information between sources.

Sources:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _extract_unique_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract unique source identifiers"""
        sources = set()
        for chunk in chunks:
            source = chunk.get('source', chunk.get('doc_id', 'unknown'))
            sources.add(source)
        return list(sources)
```

**Interview Tips:**
- Source diversity often more important than just top-k
- Decompose complex questions into aspects
- Track and cite sources for transparency
- Handle conflicting information explicitly

---

### Question 31
- [ ] Done
**What techniques help with explaining RAG decisions and providing source attribution?**

**Answer:**

**Definition:**
**Source attribution**: trace generated claims back to source documents. Techniques: **inline citations**, **highlight spans**, **confidence scores**, **chunk metadata**, **post-generation verification**. Essential for trustworthiness and compliance.

**Attribution Methods:**

| Method | How It Works | Reliability |
|--------|--------------|------------|
| **Prompt-based** | Ask LLM to cite | Medium |
| **Span highlighting** | Map output to chunks | High |
| **Verification** | LLM checks citations | High |
| **Dual generation** | Answer + sources separately | Medium |

**Python Code Example:**
```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import difflib

@dataclass
class CitedClaim:
    claim: str
    source_id: str
    source_text: str
    confidence: float
    span_match: bool

class AttributedRAG:
    """RAG with source attribution"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def query_with_citations(self, question: str,
                              k: int = 5) -> Dict:
        """Generate answer with inline citations"""
        
        # Retrieve with source tracking
        chunks = self.retriever.search(question, k=k)
        
        # Number sources for citation
        sources = {}
        for i, chunk in enumerate(chunks):
            source_id = f"[{i+1}]"
            sources[source_id] = {
                'id': chunk['id'],
                'text': chunk['text'],
                'metadata': chunk.get('metadata', {})
            }
        
        # Format context with source markers
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[{i+1}] {chunk['text']}")
        context = '\n\n'.join(context_parts)
        
        # Generate with citation instructions
        prompt = f"""Answer the question using the provided sources.
Cite sources inline using [1], [2], etc. after each claim.
Only make claims that can be supported by the sources.

Sources:
{context}

Question: {question}

Answer with citations:"""
        
        answer = self.llm.generate(prompt, temperature=0.3)
        
        # Verify citations
        verified = self._verify_citations(answer, sources)
        
        return {
            'answer': answer,
            'sources': sources,
            'verified_claims': verified['claims'],
            'unverified_claims': verified['unverified']
        }
    
    def _verify_citations(self, answer: str,
                          sources: Dict) -> Dict:
        """Verify that citations are accurate"""
        
        # Extract claims with citations
        pattern = r'([^.]+?)\s*\[(\d+)\]'
        matches = re.findall(pattern, answer)
        
        verified_claims = []
        unverified_claims = []
        
        for claim, source_num in matches:
            source_id = f"[{source_num}]"
            
            if source_id in sources:
                source_text = sources[source_id]['text']
                
                # Check if claim is supported by source
                support_score = self._check_support(
                    claim.strip(), source_text
                )
                
                claim_obj = CitedClaim(
                    claim=claim.strip(),
                    source_id=source_id,
                    source_text=source_text[:200],
                    confidence=support_score,
                    span_match=support_score > 0.3
                )
                
                if support_score > 0.3:
                    verified_claims.append(claim_obj)
                else:
                    unverified_claims.append(claim_obj)
            else:
                unverified_claims.append(CitedClaim(
                    claim=claim.strip(),
                    source_id=source_id,
                    source_text="Source not found",
                    confidence=0.0,
                    span_match=False
                ))
        
        return {
            'claims': verified_claims,
            'unverified': unverified_claims
        }
    
    def _check_support(self, claim: str, source: str) -> float:
        """Check if source supports claim"""
        
        # Word overlap approach
        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words & source_words)
        return overlap / len(claim_words)

class HighlightingRAG:
    """Highlight source spans in output"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def query_with_highlights(self, question: str) -> Dict:
        """Generate answer with highlighted source spans"""
        
        chunks = self.retriever.search(question, k=5)
        context = '\n\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Answer the question based on the context.
Wrap text from sources in <source>...</source> tags.

Context: {context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, temperature=0.3)
        
        # Find source spans and match to chunks
        source_spans = re.findall(r'<source>(.*?)</source>', answer)
        
        attributions = []
        for span in source_spans:
            match = self._find_best_match(span, chunks)
            attributions.append({
                'span': span,
                'source': match['source'],
                'similarity': match['similarity']
            })
        
        return {
            'answer': answer,
            'attributions': attributions
        }
    
    def _find_best_match(self, span: str, chunks: List) -> Dict:
        """Find which chunk the span came from"""
        
        best_match = {'source': 'unknown', 'similarity': 0.0}
        
        for chunk in chunks:
            matcher = difflib.SequenceMatcher(
                None, span.lower(), chunk['text'].lower()
            )
            ratio = matcher.ratio()
            
            if ratio > best_match['similarity']:
                best_match = {
                    'source': chunk.get('source', chunk['id']),
                    'similarity': ratio
                }
        
        return best_match
```

**Interview Tips:**
- Prompt-based citations are easy but can hallucinate sources
- Post-verification catches incorrect citations
- Track unverified claims for quality monitoring
- Consider dual-pass: generate then verify

---

### Question 32
- [ ] Done
**How do you implement parent-child document retrieval and hierarchical chunking?**

**Answer:**

**Definition:**
**Hierarchical chunking**: create chunks at multiple granularities (document → section → paragraph → sentence). Benefits: **coarse search** (fast, topic-level), **fine retrieval** (precise), **parent context** (expand for LLM). Pattern: retrieve small, expand to parent.

**Hierarchy Levels:**

| Level | Size | Use Case |
|-------|------|----------|
| **Document** | Full | Topic filtering |
| **Section** | 1000-2000 tokens | Context window |
| **Paragraph** | 200-500 tokens | Retrieval |
| **Sentence** | 20-50 tokens | Precise matching |

**Python Code Example:**
```python
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class HierarchicalChunk:
    id: str
    level: str  # 'document', 'section', 'paragraph', 'sentence'
    text: str
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict

class HierarchicalChunker:
    """Create hierarchical chunk structure"""
    
    def __init__(self, 
                 section_pattern: str = r'^#+\s+',
                 paragraph_min_chars: int = 100):
        self.section_pattern = section_pattern
        self.paragraph_min = paragraph_min_chars
    
    def chunk_document(self, text: str, 
                       doc_id: str) -> List[HierarchicalChunk]:
        """Create hierarchical chunks from document"""
        
        all_chunks = []
        
        # Level 1: Document
        doc_chunk = HierarchicalChunk(
            id=doc_id,
            level='document',
            text=text[:1000] + '...' if len(text) > 1000 else text,
            parent_id=None,
            children_ids=[],
            metadata={'char_count': len(text)}
        )
        all_chunks.append(doc_chunk)
        
        # Level 2: Sections
        sections = self._split_into_sections(text)
        section_chunks = []
        
        for i, section in enumerate(sections):
            section_id = f"{doc_id}_section_{i}"
            section_chunk = HierarchicalChunk(
                id=section_id,
                level='section',
                text=section['text'],
                parent_id=doc_id,
                children_ids=[],
                metadata={'title': section.get('title', f'Section {i}')}
            )
            section_chunks.append(section_chunk)
            doc_chunk.children_ids.append(section_id)
            
            # Level 3: Paragraphs
            paragraphs = self._split_into_paragraphs(section['text'])
            
            for j, para in enumerate(paragraphs):
                para_id = f"{section_id}_para_{j}"
                para_chunk = HierarchicalChunk(
                    id=para_id,
                    level='paragraph',
                    text=para,
                    parent_id=section_id,
                    children_ids=[],
                    metadata={}
                )
                section_chunk.children_ids.append(para_id)
                
                # Level 4: Sentences
                sentences = self._split_into_sentences(para)
                
                for k, sent in enumerate(sentences):
                    sent_id = f"{para_id}_sent_{k}"
                    sent_chunk = HierarchicalChunk(
                        id=sent_id,
                        level='sentence',
                        text=sent,
                        parent_id=para_id,
                        children_ids=[],
                        metadata={}
                    )
                    para_chunk.children_ids.append(sent_id)
                    all_chunks.append(sent_chunk)
                
                all_chunks.append(para_chunk)
        
        all_chunks.extend(section_chunks)
        return all_chunks
    
    def _split_into_sections(self, text: str) -> List[Dict]:
        """Split text into sections by headers"""
        
        lines = text.split('\n')
        sections = []
        current = {'title': 'Introduction', 'lines': []}
        
        for line in lines:
            if re.match(self.section_pattern, line):
                if current['lines']:
                    current['text'] = '\n'.join(current['lines'])
                    sections.append(current)
                current = {
                    'title': re.sub(self.section_pattern, '', line).strip(),
                    'lines': []
                }
            else:
                current['lines'].append(line)
        
        if current['lines']:
            current['text'] = '\n'.join(current['lines'])
            sections.append(current)
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs 
                if len(p.strip()) >= self.paragraph_min]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class HierarchicalRetriever:
    """Retrieve using hierarchical chunks"""
    
    def __init__(self, vector_db, chunk_store: Dict):
        self.vector_db = vector_db
        self.chunk_store = chunk_store
    
    def search(self, query: str, k: int = 5,
               retrieval_level: str = 'paragraph',
               return_level: str = 'section') -> List[Dict]:
        """Search at retrieval_level, return at return_level"""
        
        # Search at fine-grained level
        results = self.vector_db.search(
            query, k=k * 2,
            filter={'level': retrieval_level}
        )
        
        # Expand to return_level
        expanded = []
        seen_ids = set()
        
        for result in results:
            chunk = self.chunk_store[result['id']]
            parent = self._get_ancestor(chunk, return_level)
            
            if parent and parent.id not in seen_ids:
                seen_ids.add(parent.id)
                expanded.append({
                    'id': parent.id,
                    'text': parent.text,
                    'level': parent.level,
                    'score': result['score']
                })
        
        return expanded[:k]
    
    def _get_ancestor(self, chunk: HierarchicalChunk,
                      target_level: str) -> Optional[HierarchicalChunk]:
        """Walk up to target level"""
        
        level_order = ['sentence', 'paragraph', 'section', 'document']
        current = chunk
        
        while current.parent_id and current.level != target_level:
            current = self.chunk_store[current.parent_id]
        
        return current if current.level == target_level else None
```

**Interview Tips:**
- "Retrieve small, return big" is common pattern
- Hierarchical structure helps with context windowing
- Store parent-child relationships for traversal
- Consider overlapping chunks at each level

---

## RAG Evaluation & Quality

### Question 33
- [ ] Done
**How do you evaluate RAG systems when ground truth answers aren't available?**

**Answer:**

**Definition:**
RAG evaluation without ground truth: use **proxy metrics**, **LLM-as-judge**, **component metrics**, **human sampling**, **reference-free quality**. Key: evaluate retrieval and generation separately, combine for system score.

**Evaluation Approaches:**

| Approach | What It Measures | Ground Truth Needed |
|----------|-----------------|---------------------|
| **Retrieval metrics** | Chunk relevance | No (use query) |
| **LLM-as-judge** | Answer quality | No |
| **Faithfulness** | Grounded in sources | No |
| **Coherence** | Answer structure | No |
| **Human evaluation** | Overall quality | No (just judges) |

**Python Code Example:**
```python
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class RAGEvaluation:
    retrieval_relevance: float
    answer_faithfulness: float
    answer_completeness: float
    answer_coherence: float
    overall_score: float

class RAGEvaluator:
    """Evaluate RAG without ground truth answers"""
    
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
    
    def evaluate(self, query: str, 
                 retrieved_chunks: List[str],
                 answer: str) -> RAGEvaluation:
        """Comprehensive RAG evaluation"""
        
        # 1. Retrieval relevance
        retrieval_score = self._eval_retrieval_relevance(
            query, retrieved_chunks
        )
        
        # 2. Faithfulness
        faithfulness_score = self._eval_faithfulness(
            answer, retrieved_chunks
        )
        
        # 3. Completeness
        completeness_score = self._eval_completeness(query, answer)
        
        # 4. Coherence
        coherence_score = self._eval_coherence(answer)
        
        # Overall weighted score
        overall = np.average(
            [retrieval_score, faithfulness_score, 
             completeness_score, coherence_score],
            weights=[0.25, 0.30, 0.30, 0.15]
        )
        
        return RAGEvaluation(
            retrieval_relevance=retrieval_score,
            answer_faithfulness=faithfulness_score,
            answer_completeness=completeness_score,
            answer_coherence=coherence_score,
            overall_score=overall
        )
    
    def _eval_retrieval_relevance(self, query: str,
                                   chunks: List[str]) -> float:
        """Score chunk relevance to query"""
        
        if not chunks:
            return 0.0
        
        chunk_scores = []
        for chunk in chunks:
            prompt = f"""Rate relevance of passage to query.
Scale: 1 (not relevant) to 5 (highly relevant).

Query: {query}
Passage: {chunk[:500]}

Score (number only):"""
            
            response = self.llm.generate(prompt, temperature=0)
            try:
                score = int(response.strip())
                chunk_scores.append(min(5, max(1, score)))
            except:
                chunk_scores.append(3)
        
        return np.mean(chunk_scores) / 5.0
    
    def _eval_faithfulness(self, answer: str,
                           chunks: List[str]) -> float:
        """Check if answer is grounded in chunks"""
        
        context = '\n\n'.join(chunks)
        
        prompt = f"""Is the answer faithfully based on the context?
Rate 1 (many unsupported claims) to 5 (fully grounded).

Context: {context[:2000]}
Answer: {answer}

Score (1-5):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        import re
        numbers = re.findall(r'\b([1-5])\b', response)
        if numbers:
            return int(numbers[-1]) / 5.0
        return 0.6
    
    def _eval_completeness(self, query: str, 
                           answer: str) -> float:
        """Check if answer addresses query"""
        
        prompt = f"""Does this answer fully address the question?
Rate 1 (barely addresses) to 5 (comprehensive).

Question: {query}
Answer: {answer}

Score:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        import re
        numbers = re.findall(r'\b([1-5])\b', response)
        if numbers:
            return int(numbers[0]) / 5.0
        return 0.6
    
    def _eval_coherence(self, answer: str) -> float:
        """Evaluate answer clarity"""
        
        prompt = f"""Rate answer's coherence and clarity.
1 (confusing) to 5 (very clear).

Answer: {answer}

Score:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        import re
        numbers = re.findall(r'\b([1-5])\b', response)
        if numbers:
            return int(numbers[0]) / 5.0
        return 0.6
```

**Interview Tips:**
- Separate retrieval eval from generation eval
- LLM-as-judge is practical but calibrate on known cases
- Faithfulness is critical - prevents hallucinations
- Sample human evaluation for calibration

---

### Question 34
- [ ] Done
**What metrics (faithfulness, relevance, context precision) best measure RAG quality?**

**Answer:**

**Definition:**
RAG needs **retrieval metrics** (did we find right info?) + **generation metrics** (did we use it well?). Key: **Recall@K** (coverage), **Precision@K** (accuracy), **Faithfulness** (grounding), **Answer Relevance** (query addressing), **Context Utilization**.

**RAG Metrics Overview:**

| Category | Metric | What It Measures |
|----------|--------|------------------|
| **Retrieval** | Recall@K | Found relevant docs |
| **Retrieval** | MRR | Relevant doc rank |
| **Retrieval** | Context Precision | Relevant chunks / retrieved |
| **Generation** | Faithfulness | Grounded in context |
| **Generation** | Relevance | Answers query |
| **Combined** | Context utilization | Uses retrieved info |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class RAGMetrics:
    # Retrieval metrics
    recall_at_k: float
    precision_at_k: float
    mrr: float
    context_precision: float
    
    # Generation metrics
    faithfulness: float
    relevance: float
    context_utilization: float
    
    # Combined
    rag_score: float

class RAGMetricsCalculator:
    """Calculate comprehensive RAG metrics"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def calculate_all(self,
                      query: str,
                      retrieved_ids: List[str],
                      relevant_ids: Set[str],
                      retrieved_texts: List[str],
                      answer: str) -> RAGMetrics:
        """Calculate all RAG metrics"""
        
        # Retrieval metrics
        recall = self.recall_at_k(retrieved_ids, relevant_ids)
        precision = self.precision_at_k(retrieved_ids, relevant_ids)
        mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ctx_precision = self._context_precision(query, retrieved_texts)
        
        # Generation metrics (use LLM if available)
        if self.llm:
            faithfulness = self._llm_faithfulness(answer, retrieved_texts)
            relevance = self._llm_relevance(query, answer)
            utilization = self._context_utilization(answer, retrieved_texts)
        else:
            faithfulness = relevance = utilization = 0.0
        
        # Combined RAG score
        rag_score = self._combined_score(
            retrieval=np.mean([recall, precision, mrr]),
            generation=np.mean([faithfulness, relevance, utilization])
        )
        
        return RAGMetrics(
            recall_at_k=recall,
            precision_at_k=precision,
            mrr=mrr,
            context_precision=ctx_precision,
            faithfulness=faithfulness,
            relevance=relevance,
            context_utilization=utilization,
            rag_score=rag_score
        )
    
    def recall_at_k(self, retrieved: List[str], 
                    relevant: Set[str]) -> float:
        """Fraction of relevant docs retrieved"""
        if not relevant:
            return 0.0
        return len(set(retrieved) & relevant) / len(relevant)
    
    def precision_at_k(self, retrieved: List[str],
                       relevant: Set[str]) -> float:
        """Fraction of retrieved docs that are relevant"""
        if not retrieved:
            return 0.0
        return len(set(retrieved) & relevant) / len(retrieved)
    
    def mean_reciprocal_rank(self, retrieved: List[str],
                              relevant: Set[str]) -> float:
        """1 / rank of first relevant doc"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _context_precision(self, query: str,
                           chunks: List[str]) -> float:
        """How many retrieved chunks are relevant"""
        if not self.llm or not chunks:
            return 0.0
        
        relevant_count = 0
        for chunk in chunks:
            prompt = f"""Is this passage relevant to the query? Answer YES or NO.

Query: {query}
Passage: {chunk[:300]}

Relevant?"""
            response = self.llm.generate(prompt, temperature=0)
            if 'YES' in response.upper():
                relevant_count += 1
        
        return relevant_count / len(chunks)
    
    def _llm_faithfulness(self, answer: str, 
                          context: List[str]) -> float:
        """Score answer faithfulness"""
        context_text = '\n'.join(context[:3])
        
        prompt = f"""Score how well answer is grounded in context.
1 = hallucination, 5 = fully supported

Context: {context_text[:1500]}
Answer: {answer}

Score (1-5):"""
        
        response = self.llm.generate(prompt, temperature=0)
        try:
            return int(response.strip()) / 5.0
        except:
            return 0.5
    
    def _llm_relevance(self, query: str, answer: str) -> float:
        """Score answer relevance"""
        prompt = f"""Score how well answer addresses question.
1 = doesn't answer, 5 = fully answers

Question: {query}
Answer: {answer}

Score (1-5):"""
        
        response = self.llm.generate(prompt, temperature=0)
        try:
            return int(response.strip()) / 5.0
        except:
            return 0.5
    
    def _context_utilization(self, answer: str,
                              context: List[str]) -> float:
        """Measure context usage in answer"""
        answer_words = set(answer.lower().split())
        context_words = set(' '.join(context).lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        return min(1.0, overlap / len(context_words) * 10)
    
    def _combined_score(self, retrieval: float, 
                        generation: float) -> float:
        """Combined RAG score"""
        return 0.4 * retrieval + 0.6 * generation
```

**Interview Tips:**
- Retrieval and generation metrics are complementary
- High retrieval + low generation = context not used
- High generation + low retrieval = possible hallucination
- Track metrics over time for regression detection

---

### Question 35
- [ ] Done
**How do you handle RAG quality control and confidence scoring for generated answers?**

**Answer:**

**Definition:**
**RAG quality control**: real-time checks before returning response. Checks: **faithfulness** (grounded?), **safety** (appropriate?), **confidence** (retrieval quality), **completeness** (answered?). Block or flag low-quality responses.

**QC Pipeline:**
```
Retrieval → Generation → Faithfulness Check → Safety Check → Confidence Check → Return/Flag
```

**Python Code Example:**
```python
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class QCResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass
class QualityReport:
    overall: QCResult
    faithfulness: QCResult
    safety: QCResult
    confidence: QCResult
    completeness: QCResult
    should_return: bool
    fallback_response: Optional[str]

class RAGQualityController:
    """Automated quality control for RAG"""
    
    def __init__(self, llm):
        self.llm = llm
        self.min_retrieval_score = 0.5
        self.min_faithfulness_score = 0.6
    
    def check_response(self, 
                       query: str,
                       answer: str,
                       retrieved_chunks: List[Dict]) -> QualityReport:
        """Run all quality checks"""
        
        checks = {}
        
        # 1. Faithfulness check
        checks['faithfulness'] = self._check_faithfulness(
            answer, retrieved_chunks
        )
        
        # 2. Safety check
        checks['safety'] = self._check_safety(answer)
        
        # 3. Retrieval confidence
        checks['confidence'] = self._check_confidence(retrieved_chunks)
        
        # 4. Completeness
        checks['completeness'] = self._check_completeness(query, answer)
        
        # Determine overall result
        overall, should_return, fallback = self._aggregate(checks)
        
        return QualityReport(
            overall=overall,
            faithfulness=checks['faithfulness'],
            safety=checks['safety'],
            confidence=checks['confidence'],
            completeness=checks['completeness'],
            should_return=should_return,
            fallback_response=fallback
        )
    
    def _check_faithfulness(self, answer: str,
                            chunks: List[Dict]) -> QCResult:
        """Check if answer is grounded"""
        
        if not chunks:
            return QCResult.FAIL
        
        context = '\n'.join([c['text'][:500] for c in chunks])
        
        prompt = f"""Check if answer is supported by context.
Output "SUPPORTED" or "UNSUPPORTED".

Context: {context}
Answer: {answer}

Verdict:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        if 'SUPPORTED' in response.upper():
            return QCResult.PASS
        elif 'UNSUPPORTED' in response.upper():
            return QCResult.FAIL
        return QCResult.WARN
    
    def _check_safety(self, answer: str) -> QCResult:
        """Check for unsafe content"""
        
        unsafe_patterns = [
            r'\b(kill|harm|illegal|hack)\b',
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, answer.lower()):
                return QCResult.FAIL
        
        return QCResult.PASS
    
    def _check_confidence(self, chunks: List[Dict]) -> QCResult:
        """Check retrieval confidence"""
        
        if not chunks:
            return QCResult.FAIL
        
        scores = [c.get('score', 0) for c in chunks]
        max_score = max(scores)
        
        if max_score < self.min_retrieval_score:
            return QCResult.WARN
        
        return QCResult.PASS
    
    def _check_completeness(self, query: str, 
                            answer: str) -> QCResult:
        """Check if answer addresses query"""
        
        refusal_patterns = [
            "i don't know",
            "i cannot",
            "no information"
        ]
        
        answer_lower = answer.lower()
        for pattern in refusal_patterns:
            if pattern in answer_lower:
                return QCResult.WARN
        
        if len(answer.split()) < 10:
            return QCResult.WARN
        
        return QCResult.PASS
    
    def _aggregate(self, checks: Dict) -> tuple:
        """Aggregate results"""
        
        # Safety is blocking
        if checks['safety'] == QCResult.FAIL:
            return QCResult.FAIL, False, "I cannot provide that information."
        
        # Faithfulness failure is serious
        if checks['faithfulness'] == QCResult.FAIL:
            return QCResult.FAIL, False, (
                "I couldn't find reliable information. "
                "Please try rephrasing."
            )
        
        warnings = sum(1 for r in checks.values() if r == QCResult.WARN)
        
        if warnings >= 2:
            return QCResult.WARN, True, None
        
        if any(r == QCResult.FAIL for r in checks.values()):
            return QCResult.FAIL, False, "Unable to provide reliable answer."
        
        return QCResult.PASS, True, None

class ConfidenceScorer:
    """Calculate confidence scores for RAG responses"""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def calculate_confidence(self, query: str,
                              chunks: List[Dict],
                              answer: str) -> float:
        """Calculate overall confidence score"""
        
        # 1. Retrieval confidence (max similarity)
        retrieval_conf = max([c.get('score', 0) for c in chunks]) if chunks else 0
        
        # 2. Coverage (how many chunks contribute)
        coverage = min(1.0, len(chunks) / 3)
        
        # 3. Answer grounding (semantic similarity)
        answer_emb = self.embedder.embed(answer)
        context = ' '.join([c['text'] for c in chunks])
        context_emb = self.embedder.embed(context[:1000])
        
        import numpy as np
        grounding = np.dot(answer_emb, context_emb) / (
            np.linalg.norm(answer_emb) * np.linalg.norm(context_emb)
        )
        
        # Weighted combination
        confidence = (
            0.4 * retrieval_conf +
            0.2 * coverage +
            0.4 * grounding
        )
        
        return float(confidence)
```

**Interview Tips:**
- Safety checks are non-negotiable
- Faithfulness prevents hallucinations from reaching users
- Use fallbacks for blocked responses
- Log all QC failures for analysis

---

### Question 36
- [ ] Done
**What approaches help with RAG robustness against noisy or outdated knowledge sources?**

**Answer:**

**Definition:**
**RAG robustness**: handle noisy, outdated, or low-quality sources gracefully. Techniques: **source freshness scoring**, **quality filtering**, **redundancy checking**, **temporal weighting**, **conflict detection**, **fallback strategies**.

**Robustness Strategies:**

| Issue | Solution | Implementation |
|-------|----------|---------------|
| **Outdated** | Temporal scoring | Decay by age |
| **Noisy** | Quality filtering | Min similarity threshold |
| **Redundant** | Deduplication | Hash/similarity clustering |
| **Contradictory** | Conflict detection | Cross-chunk verification |

**Python Code Example:**
```python
import time
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class RobustChunk:
    id: str
    text: str
    score: float
    freshness: float
    quality: float
    final_score: float

class RobustRetriever:
    """Retriever with robustness features"""
    
    def __init__(self, base_retriever, embedder,
                 freshness_decay: float = 0.1,
                 min_quality: float = 0.3):
        self.retriever = base_retriever
        self.embedder = embedder
        self.freshness_decay = freshness_decay
        self.min_quality = min_quality
    
    def search(self, query: str, k: int = 5,
               current_time: float = None) -> List[RobustChunk]:
        """Search with robustness scoring"""
        
        if current_time is None:
            current_time = time.time()
        
        # Get more candidates for filtering
        candidates = self.retriever.search(query, k=k * 3)
        
        # Score each candidate
        scored = []
        for c in candidates:
            freshness = self._freshness_score(
                c.get('timestamp', current_time - 86400 * 30),
                current_time
            )
            quality = c.get('score', 0.5)
            
            # Combined score
            final = 0.5 * quality + 0.3 * freshness + 0.2 * c.get('source_trust', 0.5)
            
            if quality >= self.min_quality:
                scored.append(RobustChunk(
                    id=c['id'],
                    text=c['text'],
                    score=c.get('score', 0),
                    freshness=freshness,
                    quality=quality,
                    final_score=final
                ))
        
        # Deduplicate
        deduped = self._deduplicate(scored)
        
        # Sort by final score
        deduped.sort(key=lambda x: x.final_score, reverse=True)
        
        return deduped[:k]
    
    def _freshness_score(self, timestamp: float, 
                         current: float) -> float:
        """Score based on recency"""
        age_days = (current - timestamp) / 86400
        return np.exp(-self.freshness_decay * age_days / 30)
    
    def _deduplicate(self, chunks: List[RobustChunk],
                     threshold: float = 0.9) -> List[RobustChunk]:
        """Remove near-duplicate chunks"""
        
        if not chunks:
            return []
        
        # Get embeddings
        embeddings = [self.embedder.embed(c.text) for c in chunks]
        
        # Greedy dedup
        kept = [chunks[0]]
        kept_embs = [embeddings[0]]
        
        for i, chunk in enumerate(chunks[1:], 1):
            is_dup = False
            for kept_emb in kept_embs:
                sim = np.dot(embeddings[i], kept_emb) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(kept_emb)
                )
                if sim > threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                kept.append(chunk)
                kept_embs.append(embeddings[i])
        
        return kept

class SourceQualityManager:
    """Track and manage source quality"""
    
    def __init__(self):
        self.source_scores = defaultdict(lambda: {
            'positive': 0, 'negative': 0, 'trust': 0.5
        })
    
    def update_source(self, source_id: str, 
                      was_helpful: bool):
        """Update source trust based on feedback"""
        
        if was_helpful:
            self.source_scores[source_id]['positive'] += 1
        else:
            self.source_scores[source_id]['negative'] += 1
        
        # Recalculate trust (simple Bayesian)
        pos = self.source_scores[source_id]['positive']
        neg = self.source_scores[source_id]['negative']
        self.source_scores[source_id]['trust'] = (pos + 1) / (pos + neg + 2)
    
    def get_trust(self, source_id: str) -> float:
        return self.source_scores[source_id]['trust']

class ConflictDetector:
    """Detect conflicts between chunks"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_conflicts(self, chunks: List[Dict]) -> List[Dict]:
        """Find conflicting information"""
        
        conflicts = []
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if self._are_conflicting(chunks[i], chunks[j]):
                    conflicts.append({
                        'chunk1': chunks[i]['id'],
                        'chunk2': chunks[j]['id'],
                        'text1': chunks[i]['text'][:200],
                        'text2': chunks[j]['text'][:200]
                    })
        
        return conflicts
    
    def _are_conflicting(self, chunk1: Dict, chunk2: Dict) -> bool:
        """Check if two chunks conflict"""
        
        prompt = f"""Do these passages contain conflicting information?
Answer YES or NO.

Passage 1: {chunk1['text'][:300]}

Passage 2: {chunk2['text'][:300]}

Conflicting?"""
        
        response = self.llm.generate(prompt, temperature=0)
        return 'YES' in response.upper()
```

**Interview Tips:**
- Freshness scoring is critical for time-sensitive domains
- Quality filtering prevents noise from affecting generation
- Deduplication saves context window space
- Track source quality over time with user feedback

---

### Question 37
- [ ] Done
**How do you handle RAG quality assessment when sources have conflicting information?**

**Answer:**

**Definition:**
**Conflict handling in RAG**: detect, resolve, or transparently report when sources disagree. Strategies: **recency preference**, **source authority**, **majority voting**, **explicit acknowledgment**, **user-facing transparency**.

**Conflict Resolution Strategies:**

| Strategy | When to Use | Tradeoff |
|----------|-------------|----------|
| **Recency** | Fast-changing info | May miss stable truths |
| **Authority** | Clear source hierarchy | Needs source metadata |
| **Majority** | Multiple sources | Ignores minority truths |
| **Transparent** | Critical domains | Longer responses |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ConflictResolutionStrategy(Enum):
    RECENCY = "recency"
    AUTHORITY = "authority"
    MAJORITY = "majority"
    TRANSPARENT = "transparent"

@dataclass
class ConflictReport:
    has_conflict: bool
    conflicting_chunks: List[tuple]
    resolution: Optional[str]
    confidence: float

class ConflictAwareRAG:
    """RAG system that handles source conflicts"""
    
    def __init__(self, retriever, llm,
                 strategy: ConflictResolutionStrategy = 
                 ConflictResolutionStrategy.TRANSPARENT):
        self.retriever = retriever
        self.llm = llm
        self.strategy = strategy
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query with conflict handling"""
        
        chunks = self.retriever.search(question, k=k)
        
        # Detect conflicts
        conflict_report = self._detect_conflicts(question, chunks)
        
        # Generate answer based on strategy
        if conflict_report.has_conflict:
            answer = self._generate_with_conflicts(
                question, chunks, conflict_report
            )
        else:
            answer = self._generate_normal(question, chunks)
        
        return {
            'answer': answer,
            'conflict_detected': conflict_report.has_conflict,
            'conflict_details': conflict_report
        }
    
    def _detect_conflicts(self, question: str,
                          chunks: List[Dict]) -> ConflictReport:
        """Detect if chunks contain conflicting info"""
        
        if len(chunks) < 2:
            return ConflictReport(
                has_conflict=False,
                conflicting_chunks=[],
                resolution=None,
                confidence=1.0
            )
        
        # Use LLM to detect conflicts
        chunks_text = '\n\n'.join([
            f"[Source {i+1}]: {c['text'][:400]}"
            for i, c in enumerate(chunks)
        ])
        
        prompt = f"""Analyze these sources for conflicting information about: {question}

{chunks_text}

Are there any contradictions? If yes, list which sources conflict.
Format: CONFLICT: Source X vs Source Y - [topic]
Or: NO CONFLICT"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        if 'NO CONFLICT' in response.upper():
            return ConflictReport(
                has_conflict=False,
                conflicting_chunks=[],
                resolution=None,
                confidence=0.9
            )
        
        # Parse conflicts
        conflicts = []
        import re
        matches = re.findall(r'Source (\d+) vs Source (\d+)', response)
        for m in matches:
            i, j = int(m[0]) - 1, int(m[1]) - 1
            if 0 <= i < len(chunks) and 0 <= j < len(chunks):
                conflicts.append((chunks[i], chunks[j]))
        
        return ConflictReport(
            has_conflict=len(conflicts) > 0,
            conflicting_chunks=conflicts,
            resolution=None,
            confidence=0.7
        )
    
    def _generate_with_conflicts(self, question: str,
                                  chunks: List[Dict],
                                  conflict: ConflictReport) -> str:
        """Generate answer when conflicts exist"""
        
        if self.strategy == ConflictResolutionStrategy.RECENCY:
            return self._resolve_by_recency(question, chunks)
        elif self.strategy == ConflictResolutionStrategy.AUTHORITY:
            return self._resolve_by_authority(question, chunks)
        elif self.strategy == ConflictResolutionStrategy.MAJORITY:
            return self._resolve_by_majority(question, chunks)
        else:  # TRANSPARENT
            return self._resolve_transparently(question, chunks, conflict)
    
    def _resolve_by_recency(self, question: str,
                            chunks: List[Dict]) -> str:
        """Use most recent source"""
        
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('timestamp', 0),
            reverse=True
        )
        
        context = sorted_chunks[0]['text']
        
        prompt = f"""Answer based on the most recent information:

Context: {context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _resolve_by_authority(self, question: str,
                               chunks: List[Dict]) -> str:
        """Use most authoritative source"""
        
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('authority_score', 0.5),
            reverse=True
        )
        
        context = sorted_chunks[0]['text']
        
        prompt = f"""Answer based on the most authoritative source:

Context: {context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _resolve_by_majority(self, question: str,
                              chunks: List[Dict]) -> str:
        """Use consensus view"""
        
        context = '\n\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Multiple sources discuss this topic. Answer based on the consensus view.
Ignore outlier opinions that contradict the majority.

Sources: {context}

Question: {question}

Consensus answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _resolve_transparently(self, question: str,
                                chunks: List[Dict],
                                conflict: ConflictReport) -> str:
        """Acknowledge conflict in answer"""
        
        context = '\n\n'.join([
            f"[Source {i+1}]: {c['text']}"
            for i, c in enumerate(chunks)
        ])
        
        prompt = f"""Answer the question, but note that sources contain conflicting information.
Present the different perspectives with their sources.
Do not hide the disagreement.

{context}

Question: {question}

Answer (acknowledging any conflicts):"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _generate_normal(self, question: str,
                         chunks: List[Dict]) -> str:
        """Standard generation when no conflicts"""
        
        context = '\n\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
```

**Interview Tips:**
- Transparent acknowledgment is best for critical domains
- Recency works well for news/current events
- Track conflict frequency for knowledge base health
- Consider user preferences for conflict handling

---

## RAG Knowledge Management

### Question 38
- [ ] Done
**How do you handle RAG knowledge base updates while maintaining retrieval consistency?**

**Answer:**

**Definition:**
**Knowledge base updates**: add/modify/delete documents without disrupting service. Challenges: **index consistency**, **stale cache**, **embedding drift**, **versioning**. Solutions: **blue-green deployment**, **incremental updates**, **cache invalidation**, **version tracking**.

**Update Strategies:**

| Strategy | Approach | Downtime |
|----------|----------|----------|
| **Blue-green** | Build new index, swap | Zero |
| **Incremental** | Add/remove individual docs | Zero |
| **Batch window** | Scheduled rebuilds | Minimal |
| **Real-time** | Stream updates | Zero |

**Python Code Example:**
```python
import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import copy

class UpdateStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class KBVersion:
    version_id: str
    timestamp: float
    doc_count: int
    is_active: bool

class VersionedKnowledgeBase:
    """Knowledge base with versioning and consistency"""
    
    def __init__(self, vector_db, embedder):
        self.vector_db = vector_db
        self.embedder = embedder
        self.active_version = "v1"
        self.versions = {}
        self.pending_updates = []
        self.lock = threading.RLock()
    
    def add_documents(self, documents: List[Dict],
                      immediate: bool = False) -> str:
        """Add documents to knowledge base"""
        
        if immediate:
            return self._add_immediate(documents)
        else:
            return self._queue_update('add', documents)
    
    def _add_immediate(self, documents: List[Dict]) -> str:
        """Add documents immediately (incremental)"""
        
        with self.lock:
            for doc in documents:
                # Generate embedding
                embedding = self.embedder.embed(doc['text'])
                
                # Add to vector DB with version tag
                self.vector_db.insert(
                    id=doc['id'],
                    embedding=embedding,
                    metadata={
                        **doc.get('metadata', {}),
                        'version': self.active_version,
                        'added_at': time.time()
                    }
                )
            
            return f"Added {len(documents)} documents"
    
    def _queue_update(self, operation: str, 
                      data: List[Dict]) -> str:
        """Queue update for batch processing"""
        
        update_id = f"update_{int(time.time())}"
        self.pending_updates.append({
            'id': update_id,
            'operation': operation,
            'data': data,
            'status': UpdateStatus.PENDING
        })
        return update_id
    
    def blue_green_update(self, documents: List[Dict]) -> str:
        """Full rebuild with blue-green deployment"""
        
        new_version = f"v{int(time.time())}"
        
        # Build new index in background
        new_collection = f"kb_{new_version}"
        
        for doc in documents:
            embedding = self.embedder.embed(doc['text'])
            self.vector_db.insert(
                collection=new_collection,
                id=doc['id'],
                embedding=embedding,
                metadata=doc.get('metadata', {})
            )
        
        # Atomic swap
        with self.lock:
            old_version = self.active_version
            self.active_version = new_version
            self.versions[new_version] = KBVersion(
                version_id=new_version,
                timestamp=time.time(),
                doc_count=len(documents),
                is_active=True
            )
        
        # Clean up old version (optional, keep for rollback)
        return new_version
    
    def search(self, query: str, k: int = 5,
               version: Optional[str] = None) -> List[Dict]:
        """Search with version awareness"""
        
        target_version = version or self.active_version
        
        embedding = self.embedder.embed(query)
        
        results = self.vector_db.search(
            embedding, 
            k=k,
            filter={'version': target_version}
        )
        
        return results
    
    def rollback(self, target_version: str) -> bool:
        """Rollback to previous version"""
        
        if target_version not in self.versions:
            return False
        
        with self.lock:
            self.active_version = target_version
            self.versions[target_version].is_active = True
        
        return True

class CacheInvalidator:
    """Manage cache invalidation on KB updates"""
    
    def __init__(self, cache):
        self.cache = cache
        self.doc_to_queries = {}  # Track which docs affect which cached queries
    
    def on_document_update(self, doc_id: str):
        """Invalidate caches affected by document update"""
        
        if doc_id in self.doc_to_queries:
            affected_queries = self.doc_to_queries[doc_id]
            for query_hash in affected_queries:
                self.cache.delete(query_hash)
    
    def on_query_result(self, query_hash: str, doc_ids: List[str]):
        """Track which docs contributed to a query result"""
        
        for doc_id in doc_ids:
            if doc_id not in self.doc_to_queries:
                self.doc_to_queries[doc_id] = set()
            self.doc_to_queries[doc_id].add(query_hash)

class IncrementalUpdater:
    """Handle incremental updates efficiently"""
    
    def __init__(self, kb: VersionedKnowledgeBase):
        self.kb = kb
        self.update_buffer = []
        self.buffer_size = 100
    
    def add_document(self, document: Dict):
        """Buffer document for batch update"""
        
        self.update_buffer.append(document)
        
        if len(self.update_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Flush buffered updates"""
        
        if self.update_buffer:
            self.kb.add_documents(self.update_buffer, immediate=True)
            self.update_buffer = []
```

**Interview Tips:**
- Blue-green is safest for major updates
- Incremental is best for continuous additions
- Always track versions for rollback capability
- Cache invalidation is often overlooked but critical

---

### Question 39
- [ ] Done
**What strategies help with RAG for questions requiring temporal or current information?**

**Answer:**

**Definition:**
**Temporal RAG**: handle time-sensitive information appropriately. Strategies: **timestamp indexing**, **temporal filters**, **recency boosting**, **time-aware prompting**, **explicit dating**, **freshness decay**.

**Temporal Strategies:**

| Strategy | Implementation | Use Case |
|----------|---------------|----------|
| **Recency filter** | Only recent docs | News, pricing |
| **Temporal boost** | Weight by freshness | General preference |
| **Time context** | Add date to prompt | Historical queries |
| **Versioning** | Track doc versions | Policies, docs |

**Python Code Example:**
```python
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import re

class TemporalRAG:
    """RAG with temporal awareness"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def query(self, question: str,
              current_time: Optional[datetime] = None) -> Dict:
        """Query with temporal awareness"""
        
        if current_time is None:
            current_time = datetime.now()
        
        # Detect temporal intent
        temporal_intent = self._detect_temporal_intent(question)
        
        # Build temporal filter
        time_filter = self._build_time_filter(temporal_intent, current_time)
        
        # Retrieve with temporal scoring
        chunks = self._temporal_search(question, time_filter, current_time)
        
        # Generate with temporal context
        answer = self._generate_temporal(
            question, chunks, current_time, temporal_intent
        )
        
        return {
            'answer': answer,
            'temporal_intent': temporal_intent,
            'chunks': chunks
        }
    
    def _detect_temporal_intent(self, question: str) -> Dict:
        """Detect temporal requirements in query"""
        
        intent = {
            'requires_recent': False,
            'specific_time': None,
            'time_range': None
        }
        
        question_lower = question.lower()
        
        # Check for recency requirement
        recent_words = ['latest', 'recent', 'current', 'now', 'today']
        if any(w in question_lower for w in recent_words):
            intent['requires_recent'] = True
        
        # Check for specific year
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            intent['specific_time'] = int(year_match.group(1))
        
        # Check for time range
        range_match = re.search(
            r'last (\d+) (days?|weeks?|months?|years?)',
            question_lower
        )
        if range_match:
            amount = int(range_match.group(1))
            unit = range_match.group(2).rstrip('s')
            intent['time_range'] = (amount, unit)
        
        return intent
    
    def _build_time_filter(self, intent: Dict,
                           current: datetime) -> Optional[Tuple]:
        """Build time range filter"""
        
        if intent['requires_recent']:
            start = current - timedelta(days=30)
            return (start, current)
        
        if intent['specific_time']:
            year = intent['specific_time']
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            return (start, end)
        
        if intent['time_range']:
            amount, unit = intent['time_range']
            if unit == 'day':
                delta = timedelta(days=amount)
            elif unit == 'week':
                delta = timedelta(weeks=amount)
            elif unit == 'month':
                delta = timedelta(days=amount * 30)
            else:
                delta = timedelta(days=amount * 365)
            return (current - delta, current)
        
        return None
    
    def _temporal_search(self, query: str,
                         time_filter: Optional[Tuple],
                         current: datetime) -> List[Dict]:
        """Search with temporal scoring"""
        
        candidates = self.retriever.search(query, k=20)
        
        # Apply time filter
        if time_filter:
            start, end = time_filter
            candidates = [
                c for c in candidates
                if start <= self._parse_timestamp(c) <= end
            ]
        
        # Apply freshness boost
        for c in candidates:
            doc_time = self._parse_timestamp(c)
            age_days = (current - doc_time).days
            freshness = np.exp(-0.693 * age_days / 90)
            c['final_score'] = 0.7 * c['score'] + 0.3 * freshness
        
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:5]
    
    def _parse_timestamp(self, chunk: Dict) -> datetime:
        ts = chunk.get('timestamp') or chunk.get('date')
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        return datetime.now() - timedelta(days=365)
    
    def _generate_temporal(self, question: str,
                           chunks: List[Dict],
                           current: datetime,
                           intent: Dict) -> str:
        """Generate with temporal context"""
        
        context_parts = []
        for c in chunks:
            doc_time = self._parse_timestamp(c)
            date_str = doc_time.strftime("%Y-%m-%d")
            context_parts.append(f"[{date_str}] {c['text']}")
        
        context = '\n\n'.join(context_parts)
        time_note = f"Current date: {current.strftime('%Y-%m-%d')}"
        
        prompt = f"""{time_note}

Sources:\n{context}

Question: {question}

Answer (noting dates of information):"""
        
        return self.llm.generate(prompt, temperature=0.3)
```

**Interview Tips:**
- Always index timestamps with documents
- Recency decay prevents stale info from dominating
- Explicit date handling improves transparency
- Different domains need different decay rates

---

### Question 40
- [ ] Done
**How do you implement RAG with structured knowledge graphs and databases?**

**Answer:**

**Definition:**
**RAG + Knowledge Graph**: combine vector similarity with structured relationships. Benefits: **multi-hop reasoning**, **entity relationships**, **constraint enforcement**, **fact verification**. Architecture: retrieve context + traverse graph for connected entities.

**Integration Patterns:**

| Pattern | How It Works | Best For |
|---------|--------------|----------|
| **KG-enhanced retrieval** | Use KG to expand query | Entity-rich queries |
| **Hybrid retrieval** | Vector + KG in parallel | Comprehensive |
| **KG verification** | Verify facts against KG | Accuracy-critical |
| **Graph-guided generation** | Inject KG facts into prompt | Reasoning tasks |

**Python Code Example:**
```python
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: Dict

@dataclass
class Relation:
    source: str
    relation: str
    target: str

class KnowledgeGraph:
    """Simple knowledge graph"""
    
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.entity_relations = {}
    
    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.entity_relations[entity.id] = []
    
    def add_relation(self, relation: Relation):
        self.relations.append(relation)
        if relation.source in self.entity_relations:
            self.entity_relations[relation.source].append(relation)
    
    def get_neighbors(self, entity_id: str, 
                      max_hops: int = 1) -> List[Tuple[Entity, str]]:
        """Get connected entities"""
        
        result = []
        visited = {entity_id}
        current = [entity_id]
        
        for hop in range(max_hops):
            next_level = []
            for eid in current:
                for rel in self.entity_relations.get(eid, []):
                    if rel.target not in visited:
                        visited.add(rel.target)
                        next_level.append(rel.target)
                        target = self.entities.get(rel.target)
                        if target:
                            result.append((target, rel.relation))
            current = next_level
        
        return result

class KGEnhancedRAG:
    """RAG with knowledge graph"""
    
    def __init__(self, retriever, kg: KnowledgeGraph, llm, ner_model):
        self.retriever = retriever
        self.kg = kg
        self.llm = llm
        self.ner = ner_model
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query with KG enhancement"""
        
        # Extract entities
        entities = self._extract_entities(question)
        
        # Get KG context
        kg_context = self._get_kg_context(entities)
        
        # Vector retrieval
        vector_chunks = self.retriever.search(question, k=k)
        
        # Combine and generate
        combined = self._combine_contexts(vector_chunks, kg_context)
        answer = self._generate(question, combined)
        
        return {
            'answer': answer,
            'entities_found': [e.name for e in entities],
            'kg_facts': kg_context
        }
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using NER"""
        
        ner_results = self.ner.extract(text)
        entities = []
        
        for result in ner_results:
            entity = self._find_in_kg(result['text'])
            if entity:
                entities.append(entity)
        
        return entities
    
    def _find_in_kg(self, name: str) -> Entity:
        for entity in self.kg.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def _get_kg_context(self, entities: List[Entity]) -> List[str]:
        """Get relevant KG facts"""
        
        facts = []
        
        for entity in entities:
            # Entity properties
            for prop, value in entity.properties.items():
                facts.append(f"{entity.name} has {prop}: {value}")
            
            # Relationships
            neighbors = self.kg.get_neighbors(entity.id, max_hops=2)
            for neighbor, relation in neighbors:
                facts.append(f"{entity.name} {relation} {neighbor.name}")
        
        return facts
    
    def _combine_contexts(self, chunks: List[Dict],
                          kg_facts: List[str]) -> str:
        """Combine vector and KG contexts"""
        
        parts = []
        if kg_facts:
            parts.append("Known facts:")
            parts.extend([f"- {f}" for f in kg_facts[:10]])
        
        parts.append("\nRelevant passages:")
        for chunk in chunks:
            parts.append(chunk['text'])
        
        return '\n'.join(parts)
    
    def _generate(self, question: str, context: str) -> str:
        prompt = f"""Use facts and passages to answer.

{context}

Question: {question}

Answer:"""
        return self.llm.generate(prompt, temperature=0.3)
```

**Interview Tips:**
- KG provides structured facts, vectors provide context
- Entity extraction quality is critical
- Multi-hop reasoning is KG strength
- Use KG for verification to reduce hallucination

---

### Question 41
- [ ] Done
**What approaches help with RAG for low-resource domains with limited knowledge bases?**

**Answer:**

**Definition:**
**Low-resource RAG**: build effective RAG when domain knowledge is limited. Strategies: **data augmentation**, **transfer learning**, **few-shot prompting**, **synthetic data generation**, **domain adaptation**, **cross-lingual transfer**.

**Strategies:**

| Challenge | Solution | Technique |
|-----------|----------|----------|
| **Few documents** | Data augmentation | Paraphrase, back-translate |
| **No labeled data** | Zero/few-shot | LLM-generated labels |
| **Domain gap** | Domain adaptation | Fine-tune embeddings |
| **Sparse coverage** | Synthetic expansion | LLM generates examples |

**Python Code Example:**
```python
from typing import List, Dict
import random

class LowResourceRAG:
    """RAG strategies for limited knowledge bases"""
    
    def __init__(self, retriever, llm, embedder):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
    
    def augment_knowledge_base(self, documents: List[Dict]) -> List[Dict]:
        """Augment limited documents"""
        
        augmented = []
        
        for doc in documents:
            # Original
            augmented.append(doc)
            
            # Paraphrase augmentation
            paraphrased = self._paraphrase(doc['text'])
            augmented.append({
                **doc,
                'id': f"{doc['id']}_para",
                'text': paraphrased,
                'is_augmented': True
            })
            
            # Question generation (for FAQ-style retrieval)
            questions = self._generate_questions(doc['text'])
            for i, q in enumerate(questions):
                augmented.append({
                    'id': f"{doc['id']}_q{i}",
                    'text': q,
                    'source_doc': doc['id'],
                    'is_augmented': True
                })
        
        return augmented
    
    def _paraphrase(self, text: str) -> str:
        """Paraphrase text for augmentation"""
        
        prompt = f"""Paraphrase this text, keeping all information:

{text[:500]}

Paraphrase:"""
        
        return self.llm.generate(prompt, temperature=0.7)
    
    def _generate_questions(self, text: str, n: int = 3) -> List[str]:
        """Generate questions that text answers"""
        
        prompt = f"""Generate {n} questions that this text answers:

{text[:500]}

Questions (one per line):"""
        
        response = self.llm.generate(prompt, temperature=0.5)
        return [q.strip() for q in response.split('\n') if q.strip()]
    
    def generate_synthetic_docs(self, topics: List[str],
                                 docs_per_topic: int = 5) -> List[Dict]:
        """Generate synthetic documents for sparse topics"""
        
        synthetic = []
        
        for topic in topics:
            prompt = f"""Write {docs_per_topic} detailed paragraphs about: {topic}
Each paragraph should cover different aspects.

Paragraphs:"""
            
            response = self.llm.generate(prompt, temperature=0.8)
            paragraphs = response.split('\n\n')
            
            for i, para in enumerate(paragraphs):
                if len(para) > 50:
                    synthetic.append({
                        'id': f"synth_{topic}_{i}",
                        'text': para,
                        'topic': topic,
                        'is_synthetic': True
                    })
        
        return synthetic
    
    def query_with_fallback(self, question: str,
                            confidence_threshold: float = 0.5) -> Dict:
        """Query with fallback for low-confidence retrieval"""
        
        results = self.retriever.search(question, k=5)
        
        # Check confidence
        max_score = max([r['score'] for r in results]) if results else 0
        
        if max_score < confidence_threshold:
            # Low confidence - use LLM knowledge
            answer = self._fallback_generation(question, results)
            return {
                'answer': answer,
                'mode': 'fallback',
                'confidence': max_score
            }
        else:
            # Normal RAG
            context = '\n'.join([r['text'] for r in results])
            answer = self._generate_from_context(question, context)
            return {
                'answer': answer,
                'mode': 'rag',
                'confidence': max_score
            }
    
    def _fallback_generation(self, question: str,
                              weak_results: List[Dict]) -> str:
        """Generate when KB confidence is low"""
        
        # Use whatever context we have but allow LLM knowledge
        context = '\n'.join([r['text'][:200] for r in weak_results[:2]])
        
        prompt = f"""Answer based on the context if helpful, otherwise use general knowledge.
Note if information is from your training vs provided context.

Context (may be incomplete): {context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _generate_from_context(self, question: str, context: str) -> str:
        prompt = f"""Answer based on context:

Context: {context}

Question: {question}

Answer:"""
        return self.llm.generate(prompt, temperature=0.3)

class DomainAdaptation:
    """Adapt embeddings to new domain"""
    
    def __init__(self, base_embedder, llm):
        self.base_embedder = base_embedder
        self.llm = llm
    
    def generate_domain_pairs(self, domain_docs: List[str],
                               n_pairs: int = 1000) -> List[tuple]:
        """Generate similar/dissimilar pairs for fine-tuning"""
        
        pairs = []
        
        for doc in domain_docs[:100]:
            # Generate similar question
            prompt = f"""Generate a question this text answers:

{doc[:300]}

Question:"""
            question = self.llm.generate(prompt)
            pairs.append((question, doc, 1))  # Positive pair
            
            # Negative: random other doc
            neg_doc = random.choice(domain_docs)
            if neg_doc != doc:
                pairs.append((question, neg_doc, 0))
        
        return pairs[:n_pairs]
```

**Interview Tips:**
- Augmentation can 2-3x effective corpus size
- Synthetic data generation useful for coverage gaps
- Fallback to LLM knowledge when retrieval fails
- Monitor and flag low-confidence answers

---

### Question 42
- [ ] Done
**How do you implement multi-modal RAG combining text, images, and structured data?**

**Answer:**

**Definition:**
**Multimodal RAG**: retrieve and reason over multiple modalities - text, images, tables, code. Requires: **modality-specific encoders**, **cross-modal alignment**, **unified retrieval**, **multimodal LLM generation**.

**Multimodal Components:**

| Modality | Encoder | Index Type |
|----------|---------|------------|
| **Text** | Text embeddings | Dense vector |
| **Images** | CLIP/Vision embeddings | Dense vector |
| **Tables** | Table2Vec or text | Dense + structured |
| **Code** | Code embeddings | Dense vector |

**Python Code Example:**
```python
from typing import List, Dict, Union
from dataclasses import dataclass
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"

@dataclass
class MultimodalDoc:
    id: str
    modality: Modality
    content: Union[str, bytes]  # Text or image bytes
    metadata: Dict
    embedding: List[float] = None

class MultimodalEmbedder:
    """Embed different modalities into shared space"""
    
    def __init__(self, text_encoder, image_encoder, clip_model):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.clip = clip_model  # For aligned embeddings
    
    def embed(self, doc: MultimodalDoc) -> List[float]:
        """Embed document based on modality"""
        
        if doc.modality == Modality.TEXT:
            return self.text_encoder.embed(doc.content)
        
        elif doc.modality == Modality.IMAGE:
            # Use CLIP for image embedding (aligned with text)
            return self.clip.encode_image(doc.content)
        
        elif doc.modality == Modality.TABLE:
            # Linearize table to text
            text_rep = self._table_to_text(doc.content)
            return self.text_encoder.embed(text_rep)
        
        elif doc.modality == Modality.CODE:
            return self.text_encoder.embed(doc.content)
        
        return []
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query - can match any modality"""
        # Use CLIP text encoder for cross-modal matching
        return self.clip.encode_text(query)
    
    def _table_to_text(self, table: str) -> str:
        """Convert table to natural language"""
        # Simple linearization
        return f"Table data: {table}"

class MultimodalRAG:
    """RAG system for multiple modalities"""
    
    def __init__(self, embedder: MultimodalEmbedder, 
                 vector_db, multimodal_llm):
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = multimodal_llm  # Supports image input
    
    def index_documents(self, docs: List[MultimodalDoc]):
        """Index multimodal documents"""
        
        for doc in docs:
            embedding = self.embedder.embed(doc)
            doc.embedding = embedding
            
            self.vector_db.insert(
                id=doc.id,
                embedding=embedding,
                metadata={
                    'modality': doc.modality.value,
                    **doc.metadata
                }
            )
    
    def query(self, question: str,
              modalities: List[Modality] = None,
              k: int = 5) -> Dict:
        """Query across modalities"""
        
        # Embed query
        query_embedding = self.embedder.embed_query(question)
        
        # Build filter
        filter_dict = {}
        if modalities:
            filter_dict['modality'] = {
                '$in': [m.value for m in modalities]
            }
        
        # Search
        results = self.vector_db.search(
            query_embedding,
            k=k,
            filter=filter_dict
        )
        
        # Group by modality
        by_modality = {m: [] for m in Modality}
        for r in results:
            modality = Modality(r['metadata']['modality'])
            by_modality[modality].append(r)
        
        # Generate multimodal response
        answer = self._generate_multimodal(question, by_modality)
        
        return {
            'answer': answer,
            'results': results,
            'modalities_used': [m.value for m in by_modality if by_modality[m]]
        }
    
    def _generate_multimodal(self, question: str,
                              by_modality: Dict) -> str:
        """Generate answer from multimodal context"""
        
        context_parts = []
        images = []
        
        # Text context
        for r in by_modality.get(Modality.TEXT, []):
            context_parts.append(f"Text: {r['content']}")
        
        # Table context
        for r in by_modality.get(Modality.TABLE, []):
            context_parts.append(f"Table: {r['content']}")
        
        # Code context
        for r in by_modality.get(Modality.CODE, []):
            context_parts.append(f"Code:\n```\n{r['content']}\n```")
        
        # Images (pass directly to multimodal LLM)
        for r in by_modality.get(Modality.IMAGE, []):
            images.append(r['content'])
        
        text_context = '\n\n'.join(context_parts)
        
        if images:
            # Use multimodal LLM
            return self.llm.generate_with_images(
                prompt=f"""Based on the context and images, answer:

{text_context}

Question: {question}

Answer:""",
                images=images
            )
        else:
            # Text-only
            return self.llm.generate(
                f"""{text_context}

Question: {question}

Answer:"""
            )

class ImageRAG:
    """Specialized RAG for image retrieval"""
    
    def __init__(self, clip_model, vector_db, llm):
        self.clip = clip_model
        self.vector_db = vector_db
        self.llm = llm
    
    def text_to_image_search(self, query: str, k: int = 5) -> List[Dict]:
        """Find images matching text query"""
        
        query_embedding = self.clip.encode_text(query)
        return self.vector_db.search(query_embedding, k=k)
    
    def image_to_text_search(self, image: bytes, k: int = 5) -> List[Dict]:
        """Find text matching image"""
        
        image_embedding = self.clip.encode_image(image)
        return self.vector_db.search(
            image_embedding, k=k,
            filter={'modality': 'text'}
        )
```

**Interview Tips:**
- CLIP enables cross-modal alignment
- Tables need linearization for embedding
- Use multimodal LLMs (GPT-4V, Claude) for generation
- Consider separate indices per modality for efficiency

---

## RAG Production & Scaling

### Question 43
- [ ] Done
**How do you implement privacy-preserving RAG for sensitive organizational knowledge?**

**Answer:**

**Definition:**
**Privacy-preserving RAG**: protect sensitive data while enabling retrieval. Techniques: **access control**, **data masking**, **differential privacy**, **on-premise deployment**, **audit logging**, **PII detection/redaction**.

**Privacy Layers:**

| Layer | Protection | Implementation |
|-------|------------|---------------|
| **Access control** | Who can query what | RBAC, document ACLs |
| **Data masking** | Hide sensitive fields | PII redaction |
| **Query filtering** | Prevent data exposure | Content guards |
| **Audit logging** | Track access | Immutable logs |

**Python Code Example:**
```python
import re
import hashlib
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import time

class AccessLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class User:
    id: str
    roles: Set[str]
    access_level: AccessLevel
    department: str

class PrivacyPreservingRAG:
    """RAG with privacy controls"""
    
    def __init__(self, retriever, llm, pii_detector):
        self.retriever = retriever
        self.llm = llm
        self.pii_detector = pii_detector
        self.audit_log = []
    
    def query(self, question: str, user: User) -> Dict:
        """Query with access control and PII protection"""
        
        # Check query for forbidden patterns
        if self._is_forbidden_query(question):
            self._log_access(user, question, 'blocked')
            return {'answer': 'This query is not allowed.', 'blocked': True}
        
        # Retrieve with access filtering
        chunks = self._secure_retrieve(question, user)
        
        if not chunks:
            self._log_access(user, question, 'no_access')
            return {'answer': 'No accessible information found.', 'accessible': False}
        
        # Redact PII from chunks
        redacted_chunks = [self._redact_pii(c) for c in chunks]
        
        # Generate answer
        answer = self._generate(question, redacted_chunks)
        
        # Final PII check on answer
        safe_answer = self._redact_pii({'text': answer})['text']
        
        self._log_access(user, question, 'success', len(chunks))
        
        return {
            'answer': safe_answer,
            'sources_used': len(chunks),
            'redactions_applied': True
        }
    
    def _is_forbidden_query(self, query: str) -> bool:
        """Check for forbidden query patterns"""
        
        forbidden = [
            r'show me all (passwords|ssn|credit cards)',
            r'list (employee|customer) (salaries|personal)',
            r'give me (everyone|all users)'
        ]
        
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in forbidden)
    
    def _secure_retrieve(self, query: str, user: User) -> List[Dict]:
        """Retrieve with access control"""
        
        # Get candidates
        candidates = self.retriever.search(query, k=20)
        
        # Filter by user access
        accessible = []
        for chunk in candidates:
            if self._user_can_access(user, chunk):
                accessible.append(chunk)
        
        return accessible[:5]
    
    def _user_can_access(self, user: User, chunk: Dict) -> bool:
        """Check if user can access chunk"""
        
        chunk_level = AccessLevel(chunk.get('access_level', 'public'))
        chunk_dept = chunk.get('department')
        chunk_roles = set(chunk.get('required_roles', []))
        
        # Check access level
        level_order = [AccessLevel.PUBLIC, AccessLevel.INTERNAL,
                       AccessLevel.CONFIDENTIAL, AccessLevel.RESTRICTED]
        
        if level_order.index(chunk_level) > level_order.index(user.access_level):
            return False
        
        # Check department restriction
        if chunk_dept and chunk_dept != user.department:
            if chunk_level in [AccessLevel.CONFIDENTIAL, AccessLevel.RESTRICTED]:
                return False
        
        # Check role requirements
        if chunk_roles and not (user.roles & chunk_roles):
            return False
        
        return True
    
    def _redact_pii(self, chunk: Dict) -> Dict:
        """Redact PII from chunk"""
        
        text = chunk['text']
        
        # Common PII patterns
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]'),  # SSN
            (r'\b\d{16}\b', '[CARD REDACTED]'),  # Credit card
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        # Use ML-based PII detection for names, etc.
        if self.pii_detector:
            detections = self.pii_detector.detect(text)
            for entity in detections:
                text = text.replace(entity['text'], f'[{entity["type"]} REDACTED]')
        
        return {**chunk, 'text': text}
    
    def _generate(self, question: str, chunks: List[Dict]) -> str:
        """Generate with privacy-safe context"""
        
        context = '\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Answer based on the context. Do not reveal any personal information.
If asked about specific individuals, decline politely.

Context: {context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _log_access(self, user: User, query: str,
                    result: str, docs_accessed: int = 0):
        """Audit log for compliance"""
        
        self.audit_log.append({
            'timestamp': time.time(),
            'user_id': user.id,
            'query_hash': hashlib.sha256(query.encode()).hexdigest(),
            'result': result,
            'docs_accessed': docs_accessed
        })

class DataMasker:
    """Mask sensitive data for indexing"""
    
    def __init__(self, encryption_key: str):
        self.key = encryption_key
    
    def mask_document(self, doc: Dict,
                      sensitive_fields: List[str]) -> Dict:
        """Mask sensitive fields before indexing"""
        
        masked = doc.copy()
        
        for field in sensitive_fields:
            if field in masked:
                # Hash instead of store
                masked[f"{field}_hash"] = hashlib.sha256(
                    str(masked[field]).encode()
                ).hexdigest()
                del masked[field]
        
        return masked
```

**Interview Tips:**
- Access control at retrieval time, not just generation
- PII redaction must happen before LLM sees data
- Audit logs are essential for compliance
- Consider on-premise LLMs for sensitive data

---

### Question 44
- [ ] Done
**What strategies work best for RAG with regulatory compliance and audit requirements?**

**Answer:**

**Definition:**
**Compliance-ready RAG**: meet regulatory requirements (GDPR, HIPAA, SOC2). Key capabilities: **audit trails**, **data lineage**, **retention policies**, **access controls**, **explainability**, **data subject rights**.

**Compliance Requirements:**

| Regulation | Key Requirements | RAG Implementation |
|------------|-----------------|-------------------|
| **GDPR** | Right to erasure, consent | Document tracking, deletion |
| **HIPAA** | PHI protection | Encryption, access logs |
| **SOC2** | Security controls | Audit logs, access control |
| **CCPA** | Data disclosure | Query logging, lineage |

**Python Code Example:**
```python
import time
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class ComplianceEvent(Enum):
    DATA_ACCESS = "data_access"
    DATA_RETRIEVAL = "data_retrieval"
    ANSWER_GENERATED = "answer_generated"
    DATA_DELETION = "data_deletion"
    CONSENT_UPDATE = "consent_update"

@dataclass
class AuditRecord:
    id: str
    timestamp: float
    event_type: ComplianceEvent
    user_id: str
    query_hash: str
    documents_accessed: List[str]
    answer_hash: str
    metadata: Dict

class ComplianceRAG:
    """RAG with regulatory compliance features"""
    
    def __init__(self, retriever, llm, audit_store):
        self.retriever = retriever
        self.llm = llm
        self.audit_store = audit_store
        self.consent_registry = {}
        self.retention_days = 365
    
    def query(self, question: str, user_id: str,
              consent_scope: List[str] = None) -> Dict:
        """Query with full compliance tracking"""
        
        request_id = str(uuid.uuid4())
        
        # Check consent
        if not self._verify_consent(user_id, consent_scope):
            return {'error': 'Consent required', 'request_id': request_id}
        
        # Retrieve with lineage tracking
        chunks, lineage = self._retrieve_with_lineage(question)
        
        # Generate answer
        answer = self._generate(question, chunks)
        
        # Create audit record
        audit = AuditRecord(
            id=request_id,
            timestamp=time.time(),
            event_type=ComplianceEvent.ANSWER_GENERATED,
            user_id=user_id,
            query_hash=hashlib.sha256(question.encode()).hexdigest(),
            documents_accessed=[c['id'] for c in chunks],
            answer_hash=hashlib.sha256(answer.encode()).hexdigest(),
            metadata={'consent_scope': consent_scope}
        )
        
        self._store_audit(audit)
        
        return {
            'answer': answer,
            'request_id': request_id,
            'lineage': lineage,
            'audit_id': audit.id
        }
    
    def _verify_consent(self, user_id: str, 
                        scope: List[str]) -> bool:
        """Verify user consent for data usage"""
        
        if user_id not in self.consent_registry:
            return False
        
        user_consent = self.consent_registry[user_id]
        
        if not scope:
            return user_consent.get('general', False)
        
        return all(user_consent.get(s, False) for s in scope)
    
    def _retrieve_with_lineage(self, query: str) -> tuple:
        """Retrieve with data lineage tracking"""
        
        chunks = self.retriever.search(query, k=5)
        
        lineage = {
            'query_time': time.time(),
            'documents': [
                {
                    'id': c['id'],
                    'source': c.get('source'),
                    'ingestion_date': c.get('ingestion_date'),
                    'last_modified': c.get('last_modified'),
                    'data_owner': c.get('data_owner')
                }
                for c in chunks
            ]
        }
        
        return chunks, lineage
    
    def _generate(self, question: str, chunks: List[Dict]) -> str:
        context = '\n'.join([c['text'] for c in chunks])
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.generate(prompt, temperature=0.3)
    
    def _store_audit(self, audit: AuditRecord):
        """Store audit record immutably"""
        self.audit_store.append(asdict(audit))
    
    def handle_deletion_request(self, user_id: str,
                                 document_ids: List[str] = None) -> Dict:
        """Handle GDPR right to erasure"""
        
        deleted = []
        
        if document_ids:
            # Delete specific documents
            for doc_id in document_ids:
                if self.retriever.delete(doc_id):
                    deleted.append(doc_id)
        else:
            # Delete all user's documents
            user_docs = self.retriever.find_by_owner(user_id)
            for doc in user_docs:
                if self.retriever.delete(doc['id']):
                    deleted.append(doc['id'])
        
        # Audit the deletion
        audit = AuditRecord(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=ComplianceEvent.DATA_DELETION,
            user_id=user_id,
            query_hash='',
            documents_accessed=deleted,
            answer_hash='',
            metadata={'deletion_request': True}
        )
        self._store_audit(audit)
        
        return {'deleted': deleted, 'audit_id': audit.id}
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (GDPR data portability)"""
        
        # Get all documents owned by user
        documents = self.retriever.find_by_owner(user_id)
        
        # Get all audit records for user
        audits = [a for a in self.audit_store 
                  if a['user_id'] == user_id]
        
        return {
            'user_id': user_id,
            'documents': documents,
            'access_history': audits,
            'export_timestamp': time.time()
        }
    
    def apply_retention_policy(self):
        """Delete data past retention period"""
        
        cutoff = time.time() - (self.retention_days * 86400)
        
        # Find old documents
        old_docs = self.retriever.find_older_than(cutoff)
        
        for doc in old_docs:
            self.retriever.delete(doc['id'])
            
            audit = AuditRecord(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=ComplianceEvent.DATA_DELETION,
                user_id='system',
                query_hash='',
                documents_accessed=[doc['id']],
                answer_hash='',
                metadata={'reason': 'retention_policy'}
            )
            self._store_audit(audit)
```

**Interview Tips:**
- Audit trails must be immutable and complete
- Data lineage critical for explainability
- Implement deletion that actually removes embeddings
- Consider data residency requirements for cloud

---

### Question 45
- [ ] Done
**How do you implement monitoring and performance tracking for RAG systems?**

**Answer:**

**Definition:**
**RAG monitoring**: track system health, quality, and performance. Key metrics: **latency** (retrieval + generation), **quality** (relevance, faithfulness), **usage** (queries, tokens), **errors** (failures, timeouts). Dashboards for real-time visibility.

**Monitoring Dimensions:**

| Category | Metrics | Alerts |
|----------|---------|--------|
| **Latency** | P50, P95, P99 | >3s response |
| **Quality** | Faithfulness, relevance | <0.6 score |
| **Reliability** | Success rate, errors | >1% error rate |
| **Cost** | Tokens, API calls | Budget threshold |

**Python Code Example:**
```python
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics

@dataclass
class RAGMetrics:
    timestamp: float
    query_id: str
    
    # Latency breakdown
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    
    # Quality scores
    retrieval_score: float
    faithfulness_score: Optional[float]
    
    # Resource usage
    chunks_retrieved: int
    tokens_used: int
    
    # Status
    success: bool
    error: Optional[str] = None

class RAGMonitor:
    """Comprehensive RAG monitoring"""
    
    def __init__(self, window_size: int = 1000):
        self.metrics_buffer = deque(maxlen=window_size)
        self.alerts = []
        
        # Alert thresholds
        self.thresholds = {
            'latency_p95_ms': 3000,
            'error_rate': 0.01,
            'faithfulness_min': 0.6,
            'retrieval_score_min': 0.4
        }
    
    def record(self, metrics: RAGMetrics):
        """Record metrics and check alerts"""
        
        self.metrics_buffer.append(metrics)
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: RAGMetrics):
        """Check for alert conditions"""
        
        # Latency alert
        if metrics.total_latency_ms > self.thresholds['latency_p95_ms']:
            self.alerts.append({
                'type': 'high_latency',
                'value': metrics.total_latency_ms,
                'query_id': metrics.query_id,
                'timestamp': metrics.timestamp
            })
        
        # Quality alerts
        if metrics.faithfulness_score and \
           metrics.faithfulness_score < self.thresholds['faithfulness_min']:
            self.alerts.append({
                'type': 'low_faithfulness',
                'value': metrics.faithfulness_score,
                'query_id': metrics.query_id,
                'timestamp': metrics.timestamp
            })
    
    def get_stats(self, window_minutes: int = 60) -> Dict:
        """Get aggregated stats"""
        
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_buffer if m.timestamp > cutoff]
        
        if not recent:
            return {'error': 'No data in window'}
        
        latencies = [m.total_latency_ms for m in recent]
        
        return {
            'window_minutes': window_minutes,
            'query_count': len(recent),
            'latency': {
                'p50': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99': sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else None,
                'avg': statistics.mean(latencies)
            },
            'success_rate': sum(1 for m in recent if m.success) / len(recent),
            'avg_chunks': statistics.mean([m.chunks_retrieved for m in recent]),
            'avg_tokens': statistics.mean([m.tokens_used for m in recent]),
            'avg_faithfulness': statistics.mean(
                [m.faithfulness_score for m in recent if m.faithfulness_score]
            ) if any(m.faithfulness_score for m in recent) else None
        }

class InstrumentedRAG:
    """RAG system with built-in monitoring"""
    
    def __init__(self, embedder, retriever, llm, monitor: RAGMonitor):
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.monitor = monitor
    
    def query(self, question: str) -> Dict:
        """Query with full instrumentation"""
        
        import uuid
        query_id = str(uuid.uuid4())
        start_time = time.time()
        timings = {}
        
        try:
            # Embedding
            t0 = time.perf_counter()
            embedding = self.embedder.embed(question)
            timings['embedding'] = (time.perf_counter() - t0) * 1000
            
            # Retrieval
            t0 = time.perf_counter()
            chunks = self.retriever.search(embedding, k=5)
            timings['retrieval'] = (time.perf_counter() - t0) * 1000
            
            retrieval_score = max([c['score'] for c in chunks]) if chunks else 0
            
            # Generation
            t0 = time.perf_counter()
            context = '\n'.join([c['text'] for c in chunks])
            answer, token_count = self.llm.generate_with_count(
                f"Context: {context}\n\nQ: {question}\n\nA:"
            )
            timings['generation'] = (time.perf_counter() - t0) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Record metrics
            metrics = RAGMetrics(
                timestamp=start_time,
                query_id=query_id,
                embedding_latency_ms=timings['embedding'],
                retrieval_latency_ms=timings['retrieval'],
                generation_latency_ms=timings['generation'],
                total_latency_ms=total_time,
                retrieval_score=retrieval_score,
                faithfulness_score=None,  # Optional: add quality check
                chunks_retrieved=len(chunks),
                tokens_used=token_count,
                success=True
            )
            self.monitor.record(metrics)
            
            return {
                'answer': answer,
                'query_id': query_id,
                'timings': timings
            }
            
        except Exception as e:
            # Record failure
            metrics = RAGMetrics(
                timestamp=start_time,
                query_id=query_id,
                embedding_latency_ms=timings.get('embedding', 0),
                retrieval_latency_ms=timings.get('retrieval', 0),
                generation_latency_ms=timings.get('generation', 0),
                total_latency_ms=(time.time() - start_time) * 1000,
                retrieval_score=0,
                faithfulness_score=None,
                chunks_retrieved=0,
                tokens_used=0,
                success=False,
                error=str(e)
            )
            self.monitor.record(metrics)
            raise
```

**Interview Tips:**
- Monitor all three stages: embedding, retrieval, generation
- Quality metrics are as important as latency
- Set up alerts for degradation, not just failures
- Track costs per query for optimization

---

### Question 46
- [ ] Done
**What techniques work best for RAG in high-concurrency and multi-user environments?**

**Answer:**

**Definition:**
**High-concurrency RAG**: handle many simultaneous users efficiently. Techniques: **connection pooling**, **async processing**, **request batching**, **caching**, **rate limiting**, **load balancing**, **horizontal scaling**.

**Scaling Strategies:**

| Bottleneck | Solution | Impact |
|------------|----------|--------|
| **Embedding API** | Batching, caching | 10x throughput |
| **Vector DB** | Read replicas | Linear scaling |
| **LLM API** | Queue + rate limit | Prevent overload |
| **Memory** | Streaming, pagination | Constant memory |

**Python Code Example:**
```python
import asyncio
from typing import List, Dict
from dataclasses import dataclass
import time
from collections import deque
import hashlib

class ConnectionPool:
    """Pool connections for reuse"""
    
    def __init__(self, create_conn, pool_size: int = 10):
        self.create_conn = create_conn
        self.pool_size = pool_size
        self.available = asyncio.Queue()
        self.in_use = 0
    
    async def acquire(self):
        if not self.available.empty():
            return await self.available.get()
        
        if self.in_use < self.pool_size:
            self.in_use += 1
            return self.create_conn()
        
        # Wait for available connection
        return await self.available.get()
    
    async def release(self, conn):
        await self.available.put(conn)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_and_acquire(self, tokens: int = 1):
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)

class ConcurrentRAG:
    """RAG optimized for high concurrency"""
    
    def __init__(self, embedder, vector_db, llm,
                 max_concurrent: int = 100):
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
        
        # Concurrency controls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.llm_rate_limiter = RateLimiter(rate=50, burst=100)
        
        # Caching
        self.embedding_cache = {}
        self.result_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def query_async(self, question: str, user_id: str) -> Dict:
        """Async query with concurrency control"""
        
        async with self.semaphore:
            # Check result cache
            cache_key = hashlib.md5(question.encode()).hexdigest()
            cached = self._get_cached(cache_key)
            if cached:
                return {**cached, 'cached': True}
            
            # Get embedding (cached)
            embedding = await self._get_embedding(question)
            
            # Retrieve (async)
            chunks = await self._retrieve_async(embedding)
            
            # Generate with rate limiting
            await self.llm_rate_limiter.wait_and_acquire()
            answer = await self._generate_async(question, chunks)
            
            result = {
                'answer': answer,
                'chunks': len(chunks),
                'cached': False
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        embedding = await asyncio.to_thread(
            self.embedder.embed, text
        )
        
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    async def _retrieve_async(self, embedding: List[float]) -> List[Dict]:
        """Async retrieval"""
        return await asyncio.to_thread(
            self.vector_db.search, embedding, 5
        )
    
    async def _generate_async(self, question: str,
                               chunks: List[Dict]) -> str:
        """Async generation"""
        context = '\n'.join([c['text'] for c in chunks])
        prompt = f"Context: {context}\n\nQ: {question}\n\nA:"
        
        return await asyncio.to_thread(
            self.llm.generate, prompt
        )
    
    def _get_cached(self, key: str) -> Dict:
        if key in self.result_cache:
            entry = self.result_cache[key]
            if time.time() - entry['time'] < self.cache_ttl:
                return entry['result']
        return None
    
    def _cache_result(self, key: str, result: Dict):
        self.result_cache[key] = {
            'result': result,
            'time': time.time()
        }

class BatchProcessor:
    """Batch requests for efficiency"""
    
    def __init__(self, process_fn, batch_size: int = 10,
                 max_wait_ms: int = 100):
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        self.running = True
        asyncio.create_task(self._process_loop())
    
    async def add(self, item) -> asyncio.Future:
        future = asyncio.Future()
        await self.queue.put((item, future))
        return await future
    
    async def _process_loop(self):
        while self.running:
            batch = []
            futures = []
            
            # Collect batch
            try:
                item, future = await asyncio.wait_for(
                    self.queue.get(), timeout=self.max_wait
                )
                batch.append(item)
                futures.append(future)
                
                # Collect more if available
                while len(batch) < self.batch_size:
                    try:
                        item, future = self.queue.get_nowait()
                        batch.append(item)
                        futures.append(future)
                    except asyncio.QueueEmpty:
                        break
                        
            except asyncio.TimeoutError:
                continue
            
            # Process batch
            if batch:
                results = await self.process_fn(batch)
                for future, result in zip(futures, results):
                    future.set_result(result)
```

**Interview Tips:**
- Embedding caching has highest ROI
- Rate limiting prevents cascade failures
- Async processing for I/O-bound operations
- Monitor queue depths for scaling decisions

---

### Question 47
- [ ] Done
**How do you implement efficient batch processing for large-scale RAG applications?**

**Answer:**

**Definition:**
**Batch RAG processing**: process many queries efficiently. Strategies: **batch embeddings**, **parallel retrieval**, **batch LLM calls**, **async pipelines**, **chunked processing**. Key: minimize API calls, maximize throughput.

**Batch Optimizations:**

| Stage | Batch Strategy | Speedup |
|-------|----------------|--------|
| **Embedding** | Batch encode | 5-10x |
| **Retrieval** | Parallel queries | 3-5x |
| **Generation** | Batch prompts | 2-3x |
| **Full pipeline** | Async parallel | 10-20x |

**Python Code Example:**
```python
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time

class BatchRAGProcessor:
    """Efficient batch processing for RAG"""
    
    def __init__(self, embedder, vector_db, llm,
                 max_workers: int = 10):
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, queries: List[str],
                      batch_size: int = 50) -> List[Dict]:
        """Process large batch of queries"""
        
        all_results = []
        
        # Process in chunks
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            results = self._process_batch_chunk(batch)
            all_results.extend(results)
        
        return all_results
    
    def _process_batch_chunk(self, queries: List[str]) -> List[Dict]:
        """Process a batch chunk"""
        
        start = time.time()
        
        # 1. Batch embedding
        embeddings = self.embedder.embed_batch(queries)
        embed_time = time.time() - start
        
        # 2. Parallel retrieval
        start = time.time()
        all_chunks = self._parallel_retrieve(embeddings)
        retrieve_time = time.time() - start
        
        # 3. Batch generation
        start = time.time()
        answers = self._batch_generate(queries, all_chunks)
        generate_time = time.time() - start
        
        return [
            {
                'query': q,
                'answer': a,
                'chunks': len(c),
                'timings': {
                    'embed_per_query': embed_time / len(queries),
                    'retrieve_per_query': retrieve_time / len(queries),
                    'generate_per_query': generate_time / len(queries)
                }
            }
            for q, a, c in zip(queries, answers, all_chunks)
        ]
    
    def _parallel_retrieve(self, embeddings: List) -> List[List[Dict]]:
        """Retrieve in parallel using thread pool"""
        
        futures = [
            self.executor.submit(self.vector_db.search, emb, 5)
            for emb in embeddings
        ]
        
        return [f.result() for f in futures]
    
    def _batch_generate(self, queries: List[str],
                        all_chunks: List[List[Dict]]) -> List[str]:
        """Generate answers in batch"""
        
        # Prepare all prompts
        prompts = []
        for query, chunks in zip(queries, all_chunks):
            context = '\n'.join([c['text'] for c in chunks])
            prompt = f"Context: {context}\n\nQ: {query}\n\nA:"
            prompts.append(prompt)
        
        # Batch call if LLM supports it
        if hasattr(self.llm, 'generate_batch'):
            return self.llm.generate_batch(prompts)
        else:
            # Parallel individual calls
            futures = [
                self.executor.submit(self.llm.generate, p)
                for p in prompts
            ]
            return [f.result() for f in futures]

class AsyncBatchRAG:
    """Async batch processing"""
    
    def __init__(self, embedder, vector_db, llm):
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
    
    async def process_batch_async(self, queries: List[str]) -> List[Dict]:
        """Fully async batch processing"""
        
        # Batch embed
        embeddings = await asyncio.to_thread(
            self.embedder.embed_batch, queries
        )
        
        # Parallel retrieve
        retrieve_tasks = [
            asyncio.to_thread(self.vector_db.search, emb, 5)
            for emb in embeddings
        ]
        all_chunks = await asyncio.gather(*retrieve_tasks)
        
        # Parallel generate
        generate_tasks = [
            self._generate_async(q, c)
            for q, c in zip(queries, all_chunks)
        ]
        answers = await asyncio.gather(*generate_tasks)
        
        return [
            {'query': q, 'answer': a, 'chunks': len(c)}
            for q, a, c in zip(queries, answers, all_chunks)
        ]
    
    async def _generate_async(self, query: str,
                               chunks: List[Dict]) -> str:
        context = '\n'.join([c['text'] for c in chunks])
        prompt = f"Context: {context}\n\nQ: {query}\n\nA:"
        return await asyncio.to_thread(self.llm.generate, prompt)

class StreamingBatchProcessor:
    """Stream results as they complete"""
    
    def __init__(self, rag_processor):
        self.processor = rag_processor
    
    async def process_stream(self, queries: List[str]):
        """Yield results as they complete"""
        
        tasks = [
            asyncio.create_task(self._process_one(i, q))
            for i, q in enumerate(queries)
        ]
        
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result
    
    async def _process_one(self, idx: int, query: str) -> Dict:
        result = await self.processor.query_async(query)
        return {'index': idx, **result}

class ChunkedFileProcessor:
    """Process large files in chunks"""
    
    def __init__(self, batch_rag: BatchRAGProcessor):
        self.batch_rag = batch_rag
    
    def process_file(self, filepath: str,
                     chunk_size: int = 1000) -> List[Dict]:
        """Process queries from file in chunks"""
        
        results = []
        
        with open(filepath, 'r') as f:
            chunk = []
            for line in f:
                query = line.strip()
                if query:
                    chunk.append(query)
                
                if len(chunk) >= chunk_size:
                    batch_results = self.batch_rag.process_batch(chunk)
                    results.extend(batch_results)
                    chunk = []
            
            # Process remaining
            if chunk:
                batch_results = self.batch_rag.process_batch(chunk)
                results.extend(batch_results)
        
        return results
```

**Interview Tips:**
- Batch embeddings first - biggest speedup
- Use async/parallel for I/O operations
- Monitor memory with large batches
- Stream results for responsive UX

---

### Question 48
- [ ] Done
**How do you handle RAG integration with existing enterprise search and knowledge systems?**

**Answer:**

**Definition:**
**Enterprise RAG integration**: connect RAG to existing systems (SharePoint, Confluence, databases, search engines). Challenges: **connector development**, **sync strategies**, **permission mirroring**, **hybrid search**, **migration paths**.

**Integration Patterns:**

| Pattern | When to Use | Complexity |
|---------|-------------|------------|
| **Connector-based** | Pull from sources | Medium |
| **Event-driven** | Real-time sync | High |
| **Hybrid search** | Combine with existing | Medium |
| **Federated** | Query multiple systems | High |

**Python Code Example:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time
from dataclasses import dataclass

@dataclass
class EnterpriseDocument:
    id: str
    source: str  # 'sharepoint', 'confluence', 'database'
    content: str
    metadata: Dict
    permissions: List[str]
    last_modified: float

class BaseConnector(ABC):
    """Base class for enterprise connectors"""
    
    @abstractmethod
    def fetch_documents(self, since: float = None) -> List[EnterpriseDocument]:
        pass
    
    @abstractmethod
    def get_permissions(self, doc_id: str) -> List[str]:
        pass

class SharePointConnector(BaseConnector):
    """Connector for SharePoint"""
    
    def __init__(self, site_url: str, client_id: str, client_secret: str):
        self.site_url = site_url
        self.client_id = client_id
        self.client_secret = client_secret
        # Initialize SharePoint client
    
    def fetch_documents(self, since: float = None) -> List[EnterpriseDocument]:
        """Fetch documents from SharePoint"""
        
        # Query SharePoint API for documents
        # This is pseudocode - actual implementation uses SharePoint SDK
        docs = []
        
        # Example structure
        for item in self._list_items(since):
            doc = EnterpriseDocument(
                id=f"sharepoint_{item['id']}",
                source='sharepoint',
                content=self._get_content(item),
                metadata={
                    'title': item.get('title'),
                    'author': item.get('author'),
                    'url': item.get('url')
                },
                permissions=self._get_permissions(item['id']),
                last_modified=item.get('modified', time.time())
            )
            docs.append(doc)
        
        return docs
    
    def _list_items(self, since: float) -> List[Dict]:
        # SharePoint API call
        return []
    
    def _get_content(self, item: Dict) -> str:
        # Extract text from document
        return ""
    
    def get_permissions(self, doc_id: str) -> List[str]:
        # Get SharePoint permissions
        return []

class ConfluenceConnector(BaseConnector):
    """Connector for Confluence"""
    
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.api_token = api_token
    
    def fetch_documents(self, since: float = None) -> List[EnterpriseDocument]:
        # Fetch from Confluence API
        return []
    
    def get_permissions(self, doc_id: str) -> List[str]:
        return []

class EnterpriseRAGIntegration:
    """Unified RAG with enterprise integration"""
    
    def __init__(self, vector_db, embedder, llm):
        self.vector_db = vector_db
        self.embedder = embedder
        self.llm = llm
        self.connectors = {}
        self.sync_state = {}  # Track last sync per source
    
    def register_connector(self, name: str, connector: BaseConnector):
        """Register an enterprise connector"""
        self.connectors[name] = connector
        self.sync_state[name] = 0
    
    def sync_all(self, full_sync: bool = False):
        """Sync all registered sources"""
        
        for name, connector in self.connectors.items():
            since = None if full_sync else self.sync_state.get(name, 0)
            
            docs = connector.fetch_documents(since=since)
            
            for doc in docs:
                self._index_document(doc)
            
            self.sync_state[name] = time.time()
    
    def _index_document(self, doc: EnterpriseDocument):
        """Index document with permissions"""
        
        embedding = self.embedder.embed(doc.content)
        
        self.vector_db.upsert(
            id=doc.id,
            embedding=embedding,
            metadata={
                **doc.metadata,
                'source': doc.source,
                'permissions': doc.permissions,
                'last_modified': doc.last_modified
            }
        )
    
    def query(self, question: str, user_id: str,
              user_groups: List[str]) -> Dict:
        """Query with permission filtering"""
        
        embedding = self.embedder.embed(question)
        
        # Get candidates
        candidates = self.vector_db.search(embedding, k=20)
        
        # Filter by permissions
        accessible = [
            c for c in candidates
            if self._user_can_access(user_id, user_groups, c)
        ]
        
        if not accessible:
            return {'answer': 'No accessible information found.'}
        
        chunks = accessible[:5]
        context = '\n'.join([c['content'] for c in chunks])
        
        answer = self.llm.generate(
            f"Context: {context}\n\nQ: {question}\n\nA:"
        )
        
        return {
            'answer': answer,
            'sources': [{'id': c['id'], 'source': c['metadata']['source']}
                       for c in chunks]
        }
    
    def _user_can_access(self, user_id: str,
                         user_groups: List[str],
                         chunk: Dict) -> bool:
        """Check user access to chunk"""
        
        required_perms = set(chunk['metadata'].get('permissions', []))
        
        if not required_perms:
            return True
        
        user_perms = {user_id} | set(user_groups)
        return bool(required_perms & user_perms)

class HybridEnterpriseSearch:
    """Combine RAG with existing enterprise search"""
    
    def __init__(self, rag_system, enterprise_search):
        self.rag = rag_system
        self.enterprise_search = enterprise_search  # Existing system
    
    def search(self, query: str, user_id: str) -> Dict:
        """Hybrid search combining both systems"""
        
        # Get results from both
        rag_results = self.rag.query(query, user_id)
        enterprise_results = self.enterprise_search.search(query, user_id)
        
        # Merge and rerank
        combined = self._merge_results(
            rag_results.get('chunks', []),
            enterprise_results
        )
        
        return {
            'answer': rag_results.get('answer'),
            'sources': combined[:10],
            'search_mode': 'hybrid'
        }
    
    def _merge_results(self, rag_chunks: List,
                       enterprise_results: List) -> List:
        """Merge and dedupe results"""
        
        seen_ids = set()
        merged = []
        
        # Interleave results
        for r, e in zip(rag_chunks, enterprise_results):
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                merged.append({**r, 'source': 'rag'})
            if e['id'] not in seen_ids:
                seen_ids.add(e['id'])
                merged.append({**e, 'source': 'enterprise'})
        
        return merged
```

**Interview Tips:**
- Permission sync is critical for enterprise
- Incremental sync for efficiency
- Handle connector failures gracefully
- Consider hybrid approach during migration

---

## Advanced RAG Patterns

### Question 49
- [ ] Done
**What is Agentic RAG and how does it differ from standard RAG pipelines?**

**Answer:**

**Definition:**
**Agentic RAG**: LLM acts as an agent that **decides when/what to retrieve**, **iterates** based on results, **uses tools** beyond retrieval, and **reasons** about information needs. Unlike standard RAG (fixed retrieve → generate), agentic RAG is dynamic and multi-step.

**Standard vs Agentic RAG:**

| Aspect | Standard RAG | Agentic RAG |
|--------|-------------|-------------|
| **Retrieval** | Always, once | On-demand, iterative |
| **Query formation** | User query direct | Agent reformulates |
| **Tool use** | Retrieval only | Multiple tools |
| **Reasoning** | Single-step | Multi-step planning |

**Python Code Example:**
```python
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

class ToolType(Enum):
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    SEARCH_WEB = "search_web"
    QUERY_DATABASE = "query_database"
    FINAL_ANSWER = "final_answer"

@dataclass
class AgentAction:
    tool: ToolType
    input: str
    thought: str

@dataclass
class AgentStep:
    action: AgentAction
    observation: str

class AgenticRAG:
    """Agentic RAG with multi-step reasoning"""
    
    def __init__(self, llm, retriever, tools: Dict[str, Callable]):
        self.llm = llm
        self.retriever = retriever
        self.tools = {
            'retrieve': self._retrieve,
            'calculate': self._calculate,
            **tools
        }
        self.max_iterations = 5
    
    def query(self, question: str) -> Dict:
        """Answer question using agentic approach"""
        
        steps = []
        context = f"Question: {question}\n\n"
        
        for i in range(self.max_iterations):
            # Decide next action
            action = self._decide_action(context, steps)
            
            if action.tool == ToolType.FINAL_ANSWER:
                return {
                    'answer': action.input,
                    'steps': steps,
                    'iterations': i + 1
                }
            
            # Execute action
            observation = self._execute_action(action)
            
            step = AgentStep(action=action, observation=observation)
            steps.append(step)
            
            # Update context
            context += f"\nThought: {action.thought}\n"
            context += f"Action: {action.tool.value}({action.input})\n"
            context += f"Observation: {observation}\n"
        
        # Max iterations reached
        return {
            'answer': self._force_answer(context),
            'steps': steps,
            'iterations': self.max_iterations
        }
    
    def _decide_action(self, context: str,
                        previous_steps: List[AgentStep]) -> AgentAction:
        """LLM decides next action"""
        
        tools_desc = ", ".join(self.tools.keys())
        
        prompt = f"""{context}

Available tools: {tools_desc}, final_answer

Decide the next action. Think step by step.
Output format:
Thought: [your reasoning]
Action: [tool_name]
Input: [input to tool]

If you have enough information, use final_answer with the complete answer.

Your response:"""
        
        response = self.llm.generate(prompt, temperature=0.2)
        
        return self._parse_action(response)
    
    def _parse_action(self, response: str) -> AgentAction:
        """Parse LLM response into action"""
        
        lines = response.strip().split('\n')
        thought = ""
        tool = ToolType.FINAL_ANSWER
        input_val = ""
        
        for line in lines:
            if line.startswith('Thought:'):
                thought = line.replace('Thought:', '').strip()
            elif line.startswith('Action:'):
                tool_name = line.replace('Action:', '').strip().lower()
                tool = ToolType(tool_name) if tool_name in [t.value for t in ToolType] else ToolType.RETRIEVE
            elif line.startswith('Input:'):
                input_val = line.replace('Input:', '').strip()
        
        return AgentAction(tool=tool, input=input_val, thought=thought)
    
    def _execute_action(self, action: AgentAction) -> str:
        """Execute the chosen action"""
        
        tool_fn = self.tools.get(action.tool.value)
        
        if tool_fn:
            return tool_fn(action.input)
        else:
            return f"Unknown tool: {action.tool.value}"
    
    def _retrieve(self, query: str) -> str:
        """Retrieval tool"""
        chunks = self.retriever.search(query, k=3)
        return '\n'.join([c['text'][:500] for c in chunks])
    
    def _calculate(self, expression: str) -> str:
        """Simple calculation tool"""
        try:
            result = eval(expression)  # Use safe_eval in production
            return str(result)
        except:
            return "Calculation error"
    
    def _force_answer(self, context: str) -> str:
        """Force an answer after max iterations"""
        
        prompt = f"""{context}

Based on all the information gathered, provide your best answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)

class ReactAgent:
    """ReAct (Reasoning + Acting) pattern"""
    
    def __init__(self, llm, tools: Dict):
        self.llm = llm
        self.tools = tools
    
    def run(self, task: str) -> Dict:
        """Execute task with ReAct pattern"""
        
        trajectory = []
        prompt = self._build_prompt(task, trajectory)
        
        for _ in range(10):
            response = self.llm.generate(prompt, temperature=0)
            
            # Parse thought and action
            thought, action, action_input = self._parse_response(response)
            
            if action == 'finish':
                return {
                    'result': action_input,
                    'trajectory': trajectory
                }
            
            # Execute action
            observation = self.tools[action](action_input)
            
            trajectory.append({
                'thought': thought,
                'action': action,
                'action_input': action_input,
                'observation': observation
            })
            
            prompt = self._build_prompt(task, trajectory)
        
        return {'result': 'Max iterations', 'trajectory': trajectory}
    
    def _build_prompt(self, task: str, trajectory: List) -> str:
        prompt = f"Task: {task}\n\n"
        
        for step in trajectory:
            prompt += f"Thought: {step['thought']}\n"
            prompt += f"Action: {step['action']}[{step['action_input']}]\n"
            prompt += f"Observation: {step['observation']}\n\n"
        
        prompt += "Think about what to do next. Use 'finish' when done."
        return prompt
    
    def _parse_response(self, response: str) -> tuple:
        # Parse thought/action/input from response
        return "", "finish", response
```

**Interview Tips:**
- Agentic RAG is more powerful but slower and costlier
- ReAct pattern is most common implementation
- Limit iterations to prevent infinite loops
- Use for complex queries, simple RAG for basic questions

---

### Question 50
- [ ] Done
**How do you implement Corrective RAG (CRAG) for self-correcting retrieval?**

**Answer:**

**Definition:**
**CRAG (Corrective RAG)**: evaluate retrieval quality and **correct** if needed. Steps: retrieve → **evaluate relevance** → if poor, **refine query** or **web search** → generate. Key insight: don't trust retrieval blindly, verify and correct.

**CRAG Pipeline:**
```
Query → Retrieve → Evaluate → [If poor] → Correct (requery/web) → Generate
                   ↓
             [If good] → Generate
```

**Python Code Example:**
```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class RetrievalQuality(Enum):
    CORRECT = "correct"      # Retrieval is relevant
    INCORRECT = "incorrect"  # Retrieval is not relevant
    AMBIGUOUS = "ambiguous"  # Partially relevant

@dataclass
class CRAGResult:
    answer: str
    retrieval_quality: RetrievalQuality
    correction_applied: bool
    sources: List[Dict]

class CorrectiveRAG:
    """CRAG implementation with self-correction"""
    
    def __init__(self, retriever, llm, web_search=None):
        self.retriever = retriever
        self.llm = llm
        self.web_search = web_search  # Fallback web search
    
    def query(self, question: str, k: int = 5) -> CRAGResult:
        """Query with corrective retrieval"""
        
        # Step 1: Initial retrieval
        chunks = self.retriever.search(question, k=k)
        
        # Step 2: Evaluate retrieval quality
        quality = self._evaluate_retrieval(question, chunks)
        
        # Step 3: Correct if needed
        correction_applied = False
        
        if quality == RetrievalQuality.INCORRECT:
            # Try web search as fallback
            if self.web_search:
                chunks = self._web_search_fallback(question)
                correction_applied = True
            else:
                # Refine query and retry
                refined_query = self._refine_query(question, chunks)
                chunks = self.retriever.search(refined_query, k=k)
                correction_applied = True
        
        elif quality == RetrievalQuality.AMBIGUOUS:
            # Combine knowledge base + web results
            if self.web_search:
                web_chunks = self._web_search_fallback(question)
                chunks = self._combine_sources(chunks, web_chunks)
                correction_applied = True
        
        # Step 4: Generate answer
        answer = self._generate(question, chunks)
        
        return CRAGResult(
            answer=answer,
            retrieval_quality=quality,
            correction_applied=correction_applied,
            sources=chunks
        )
    
    def _evaluate_retrieval(self, question: str,
                            chunks: List[Dict]) -> RetrievalQuality:
        """Evaluate if retrieved chunks are relevant"""
        
        if not chunks:
            return RetrievalQuality.INCORRECT
        
        # Use LLM to evaluate relevance
        chunks_text = '\n\n'.join([c['text'][:300] for c in chunks])
        
        prompt = f"""Evaluate if these retrieved passages are relevant to answer the question.

Question: {question}

Retrieved passages:
{chunks_text}

Classify as:
- CORRECT: Passages contain information to answer the question
- INCORRECT: Passages are not relevant to the question
- AMBIGUOUS: Partially relevant but incomplete

Classification:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        if 'CORRECT' in response.upper():
            return RetrievalQuality.CORRECT
        elif 'INCORRECT' in response.upper():
            return RetrievalQuality.INCORRECT
        else:
            return RetrievalQuality.AMBIGUOUS
    
    def _refine_query(self, original: str,
                      poor_results: List[Dict]) -> str:
        """Refine query when initial retrieval fails"""
        
        prompt = f"""The search for "{original}" returned irrelevant results.

Generate a better search query that might find the answer.
Think about different keywords or phrasings.

Refined query:"""
        
        return self.llm.generate(prompt, temperature=0.3).strip()
    
    def _web_search_fallback(self, question: str) -> List[Dict]:
        """Fall back to web search"""
        
        if self.web_search:
            results = self.web_search.search(question, num_results=5)
            return [
                {'text': r['snippet'], 'source': r['url'], 'type': 'web'}
                for r in results
            ]
        return []
    
    def _combine_sources(self, kb_chunks: List[Dict],
                         web_chunks: List[Dict]) -> List[Dict]:
        """Combine knowledge base and web results"""
        
        combined = []
        
        # Interleave sources
        for kb, web in zip(kb_chunks[:3], web_chunks[:3]):
            kb['source_type'] = 'knowledge_base'
            web['source_type'] = 'web'
            combined.extend([kb, web])
        
        return combined
    
    def _generate(self, question: str, chunks: List[Dict]) -> str:
        """Generate answer from (corrected) chunks"""
        
        context = '\n\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Answer the question based on the context.
If information is from web search, note that it may be more current.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)

class DocumentGrader:
    """Fine-grained document relevance grading"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def grade_documents(self, question: str,
                        chunks: List[Dict]) -> List[Tuple[Dict, float]]:
        """Grade each document's relevance"""
        
        graded = []
        
        for chunk in chunks:
            score = self._grade_single(question, chunk)
            graded.append((chunk, score))
        
        # Sort by relevance
        graded.sort(key=lambda x: x[1], reverse=True)
        return graded
    
    def _grade_single(self, question: str, chunk: Dict) -> float:
        """Grade single document"""
        
        prompt = f"""Rate relevance of this document to the question.
Scale: 0 (not relevant) to 1 (highly relevant)

Question: {question}
Document: {chunk['text'][:400]}

Score (0-1):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        try:
            return float(response.strip())
        except:
            return 0.5
```

**Interview Tips:**
- CRAG adds evaluation step before generation
- Web search fallback handles knowledge gaps
- Query refinement when initial search fails
- Balance latency vs quality (evaluation adds time)

---

### Question 51
- [ ] Done
**What is Self-RAG and when does reflective retrieval improve quality?**

**Answer:**

**Definition:**
**Self-RAG**: model **decides when to retrieve** and **self-reflects** on generations. Uses **reflection tokens** to: decide if retrieval needed, evaluate if retrieved info supports generation, assess answer quality. More adaptive than always-retrieve RAG.

**Self-RAG Decisions:**

| Reflection | Question | Output |
|------------|----------|--------|
| **Retrieve?** | Need external info? | Yes/No |
| **IsRelevant?** | Chunk relevant to query? | Yes/No |
| **IsSupported?** | Generation grounded? | Full/Partial/No |
| **IsUseful?** | Answers the question? | Score 1-5 |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class RetrieveDecision(Enum):
    YES = "yes"
    NO = "no"

class SupportLevel(Enum):
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"

@dataclass
class SelfRAGResult:
    answer: str
    retrieved: bool
    support_level: Optional[SupportLevel]
    usefulness_score: float
    reflections: Dict

class SelfRAG:
    """Self-RAG with reflective retrieval"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def query(self, question: str) -> SelfRAGResult:
        """Query with self-reflection"""
        
        reflections = {}
        
        # Step 1: Decide if retrieval needed
        should_retrieve = self._decide_retrieve(question)
        reflections['retrieve_decision'] = should_retrieve.value
        
        if should_retrieve == RetrieveDecision.YES:
            # Step 2: Retrieve
            chunks = self.retriever.search(question, k=5)
            
            # Step 3: Filter relevant chunks
            relevant_chunks = self._filter_relevant(question, chunks)
            reflections['chunks_retrieved'] = len(chunks)
            reflections['chunks_relevant'] = len(relevant_chunks)
            
            # Step 4: Generate with retrieved context
            answer = self._generate_with_context(question, relevant_chunks)
            
            # Step 5: Check if supported
            support = self._check_support(answer, relevant_chunks)
            reflections['support_level'] = support.value
            
        else:
            # Generate without retrieval
            answer = self._generate_without_context(question)
            support = None
            relevant_chunks = []
        
        # Step 6: Assess usefulness
        usefulness = self._assess_usefulness(question, answer)
        reflections['usefulness'] = usefulness
        
        return SelfRAGResult(
            answer=answer,
            retrieved=should_retrieve == RetrieveDecision.YES,
            support_level=support,
            usefulness_score=usefulness,
            reflections=reflections
        )
    
    def _decide_retrieve(self, question: str) -> RetrieveDecision:
        """Decide if external retrieval is needed"""
        
        prompt = f"""Decide if external information is needed to answer this question.

Consider:
- Factual questions about specific data/events: need retrieval
- General knowledge questions: may not need retrieval
- Opinion/reasoning questions: may not need retrieval

Question: {question}

Need retrieval? (yes/no):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        if 'yes' in response.lower():
            return RetrieveDecision.YES
        return RetrieveDecision.NO
    
    def _filter_relevant(self, question: str,
                         chunks: List[Dict]) -> List[Dict]:
        """Filter chunks by relevance"""
        
        relevant = []
        
        for chunk in chunks:
            prompt = f"""Is this passage relevant to the question?

Question: {question}
Passage: {chunk['text'][:400]}

Relevant? (yes/no):"""
            
            response = self.llm.generate(prompt, temperature=0)
            
            if 'yes' in response.lower():
                relevant.append(chunk)
        
        return relevant
    
    def _generate_with_context(self, question: str,
                               chunks: List[Dict]) -> str:
        """Generate answer using context"""
        
        context = '\n\n'.join([c['text'] for c in chunks])
        
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _generate_without_context(self, question: str) -> str:
        """Generate without external context"""
        
        prompt = f"""Answer the question using your knowledge.

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _check_support(self, answer: str,
                       chunks: List[Dict]) -> SupportLevel:
        """Check if answer is supported by chunks"""
        
        context = '\n\n'.join([c['text'][:300] for c in chunks])
        
        prompt = f"""Is this answer supported by the context?

Context:
{context}

Answer: {answer}

Classify:
- FULLY_SUPPORTED: All claims are in the context
- PARTIALLY_SUPPORTED: Some claims are in context
- NOT_SUPPORTED: Claims are not in context

Support level:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        if 'FULLY' in response.upper():
            return SupportLevel.FULLY_SUPPORTED
        elif 'PARTIALLY' in response.upper():
            return SupportLevel.PARTIALLY_SUPPORTED
        return SupportLevel.NOT_SUPPORTED
    
    def _assess_usefulness(self, question: str, answer: str) -> float:
        """Assess if answer is useful"""
        
        prompt = f"""Rate how well this answer addresses the question.
Scale: 1 (not useful) to 5 (very useful)

Question: {question}
Answer: {answer}

Score (1-5):"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        try:
            return float(response.strip())
        except:
            return 3.0

class AdaptiveRAG:
    """Switch between Self-RAG and standard RAG"""
    
    def __init__(self, self_rag: SelfRAG, standard_rag):
        self.self_rag = self_rag
        self.standard_rag = standard_rag
    
    def query(self, question: str, latency_sensitive: bool = False) -> Dict:
        """Choose RAG strategy based on context"""
        
        if latency_sensitive:
            # Use standard RAG for speed
            return self.standard_rag.query(question)
        else:
            # Use Self-RAG for quality
            return self.self_rag.query(question)
```

**Interview Tips:**
- Self-RAG reduces unnecessary retrievals
- Reflection adds latency but improves quality
- NOT_SUPPORTED detection catches hallucinations
- Best for high-stakes applications

---

### Question 52
- [ ] Done
**How do you implement conversational RAG with multi-turn context tracking?**

**Answer:**

**Definition:**
**Conversational RAG**: maintain context across conversation turns. Challenges: **coreference resolution** ("it", "that"), **context accumulation**, **query reformulation**, **memory management**. Key: reformulate queries to be self-contained.

**Multi-Turn Strategies:**

| Challenge | Solution | Implementation |
|-----------|----------|---------------|
| **Pronouns** | Query rewriting | LLM reformulation |
| **Context** | Conversation memory | Sliding window |
| **Follow-ups** | Query expansion | Include history |
| **Long conversations** | Summarization | Compress old turns |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class ConversationTurn:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    retrieved_chunks: Optional[List[Dict]] = None

class ConversationalRAG:
    """RAG with multi-turn conversation support"""
    
    def __init__(self, retriever, llm, max_history: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.max_history = max_history
        self.conversations = {}  # session_id -> list of turns
    
    def query(self, question: str, session_id: str) -> Dict:
        """Query with conversation context"""
        
        # Get conversation history
        history = self.conversations.get(session_id, [])
        
        # Rewrite query to be self-contained
        reformulated = self._reformulate_query(question, history)
        
        # Retrieve using reformulated query
        chunks = self.retriever.search(reformulated, k=5)
        
        # Generate answer with conversation context
        answer = self._generate_with_history(
            question, chunks, history
        )
        
        # Update history
        self._update_history(
            session_id, question, answer, chunks
        )
        
        return {
            'answer': answer,
            'reformulated_query': reformulated,
            'chunks_used': len(chunks)
        }
    
    def _reformulate_query(self, question: str,
                           history: List[ConversationTurn]) -> str:
        """Rewrite query to be self-contained"""
        
        if not history:
            return question
        
        # Format recent history
        recent = history[-4:]  # Last 2 exchanges
        history_text = '\n'.join([
            f"{t.role}: {t.content}" for t in recent
        ])
        
        prompt = f"""Rewrite the user's question to be self-contained.
Replace pronouns and references with their actual values.
The rewritten query should make sense without the conversation history.

Conversation history:
{history_text}

User's new question: {question}

Self-contained query:"""
        
        return self.llm.generate(prompt, temperature=0).strip()
    
    def _generate_with_history(self, question: str,
                                chunks: List[Dict],
                                history: List[ConversationTurn]) -> str:
        """Generate answer considering conversation context"""
        
        # Build context from chunks
        context = '\n\n'.join([c['text'] for c in chunks])
        
        # Build conversation context
        if history:
            recent = history[-4:]
            conv_context = '\n'.join([
                f"{t.role.capitalize()}: {t.content}" for t in recent
            ])
        else:
            conv_context = ""
        
        prompt = f"""Answer the user's question based on the context and conversation.
Maintain consistency with previous answers.
Refer to previous conversation naturally.

Retrieved context:
{context}

Previous conversation:
{conv_context}

User: {question}

Assistant:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _update_history(self, session_id: str,
                        question: str, answer: str,
                        chunks: List[Dict]):
        """Update conversation history"""
        
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user turn
        self.conversations[session_id].append(
            ConversationTurn(
                role='user',
                content=question,
                timestamp=time.time()
            )
        )
        
        # Add assistant turn
        self.conversations[session_id].append(
            ConversationTurn(
                role='assistant',
                content=answer,
                timestamp=time.time(),
                retrieved_chunks=chunks
            )
        )
        
        # Trim if too long
        if len(self.conversations[session_id]) > self.max_history * 2:
            self._compress_history(session_id)
    
    def _compress_history(self, session_id: str):
        """Compress old history into summary"""
        
        history = self.conversations[session_id]
        old_turns = history[:-6]  # Keep last 3 exchanges
        
        if not old_turns:
            return
        
        # Summarize old conversation
        old_text = '\n'.join([f"{t.role}: {t.content}" for t in old_turns])
        
        prompt = f"""Summarize this conversation in 2-3 sentences:

{old_text}

Summary:"""
        
        summary = self.llm.generate(prompt, temperature=0)
        
        # Replace old turns with summary
        summary_turn = ConversationTurn(
            role='system',
            content=f"Previous conversation summary: {summary}",
            timestamp=time.time()
        )
        
        self.conversations[session_id] = [summary_turn] + history[-6:]
    
    def clear_session(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]

class QueryRewriter:
    """Dedicated query rewriting component"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite(self, query: str, history: List[Dict]) -> str:
        """Rewrite query for retrieval"""
        
        if not history:
            return query
        
        # Check if query needs rewriting
        pronouns = ['it', 'this', 'that', 'they', 'them', 'he', 'she']
        needs_rewrite = any(p in query.lower().split() for p in pronouns)
        
        if not needs_rewrite:
            # Check for implicit references
            if len(query.split()) < 5:
                needs_rewrite = True
        
        if needs_rewrite:
            return self._llm_rewrite(query, history)
        
        return query
    
    def _llm_rewrite(self, query: str, history: List[Dict]) -> str:
        """Use LLM to rewrite query"""
        
        history_text = '\n'.join([
            f"{h['role']}: {h['content']}" for h in history[-4:]
        ])
        
        prompt = f"""Rewrite this follow-up question to be standalone.

History:
{history_text}

Follow-up: {query}

Standalone question:"""
        
        return self.llm.generate(prompt, temperature=0).strip()
```

**Interview Tips:**
- Query reformulation is essential for pronoun resolution
- Keep history bounded to manage context window
- Summarize old turns to preserve important context
- Track which chunks were used for transparency

---

### Question 53
- [ ] Done
**What approaches help with RAG personalization for different user contexts?**

**Answer:**

**Definition:**
**Personalized RAG**: adapt retrieval and generation to user context. Dimensions: **preferences**, **expertise level**, **history**, **role/department**, **past interactions**. Goal: more relevant answers tailored to user.

**Personalization Dimensions:**

| Dimension | How to Use | Impact |
|-----------|------------|--------|
| **Expertise** | Adjust complexity | Answer depth |
| **Preferences** | Filter sources | Source selection |
| **History** | Learn patterns | Relevance scoring |
| **Role** | Access + context | Content focus |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time

@dataclass
class UserProfile:
    user_id: str
    expertise_level: str = "intermediate"  # beginner/intermediate/expert
    preferred_sources: List[str] = field(default_factory=list)
    role: str = "general"
    department: str = ""
    query_history: List[Dict] = field(default_factory=list)
    feedback_history: List[Dict] = field(default_factory=list)

class PersonalizedRAG:
    """RAG with user personalization"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.user_profiles = {}  # user_id -> UserProfile
    
    def query(self, question: str, user_id: str) -> Dict:
        """Personalized query"""
        
        profile = self._get_or_create_profile(user_id)
        
        # Personalized retrieval
        chunks = self._personalized_retrieve(question, profile)
        
        # Personalized generation
        answer = self._personalized_generate(
            question, chunks, profile
        )
        
        # Update user history
        self._update_history(profile, question, answer)
        
        return {
            'answer': answer,
            'personalization_applied': True,
            'expertise_level': profile.expertise_level
        }
    
    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.user_profiles[user_id]
    
    def _personalized_retrieve(self, question: str,
                                profile: UserProfile) -> List[Dict]:
        """Retrieve with personalization"""
        
        # Get candidates
        chunks = self.retriever.search(question, k=15)
        
        # Apply preference filters
        if profile.preferred_sources:
            preferred = [
                c for c in chunks
                if c.get('source') in profile.preferred_sources
            ]
            other = [c for c in chunks if c not in preferred]
            chunks = preferred + other
        
        # Boost based on user history
        chunks = self._history_boost(chunks, profile)
        
        return chunks[:5]
    
    def _history_boost(self, chunks: List[Dict],
                       profile: UserProfile) -> List[Dict]:
        """Boost chunks similar to user's past interests"""
        
        if not profile.query_history:
            return chunks
        
        # Get topics from recent queries
        recent_topics = set()
        for q in profile.query_history[-10:]:
            # Simple: extract keywords
            words = q['question'].lower().split()
            recent_topics.update([w for w in words if len(w) > 4])
        
        # Boost chunks matching recent topics
        for chunk in chunks:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(recent_topics & chunk_words)
            chunk['personalization_boost'] = overlap * 0.1
            chunk['final_score'] = chunk.get('score', 0) + chunk['personalization_boost']
        
        chunks.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return chunks
    
    def _personalized_generate(self, question: str,
                                chunks: List[Dict],
                                profile: UserProfile) -> str:
        """Generate with personalization"""
        
        context = '\n'.join([c['text'] for c in chunks])
        
        # Expertise-based prompt adjustment
        expertise_instruction = {
            'beginner': "Explain simply, avoid jargon, provide examples.",
            'intermediate': "Provide clear explanations with some technical detail.",
            'expert': "Be concise and technical, assume domain knowledge."
        }
        
        instruction = expertise_instruction.get(
            profile.expertise_level, 
            expertise_instruction['intermediate']
        )
        
        # Role-based focus
        role_context = ""
        if profile.role:
            role_context = f"\nThe user is a {profile.role}. Focus on information relevant to their role."
        
        prompt = f"""Answer the question based on the context.
{instruction}{role_context}

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _update_history(self, profile: UserProfile,
                        question: str, answer: str):
        """Update user history"""
        
        profile.query_history.append({
            'question': question,
            'timestamp': time.time()
        })
        
        # Keep bounded
        if len(profile.query_history) > 100:
            profile.query_history = profile.query_history[-50:]
    
    def record_feedback(self, user_id: str, query_id: str,
                        feedback: Dict):
        """Record user feedback for learning"""
        
        profile = self._get_or_create_profile(user_id)
        
        profile.feedback_history.append({
            'query_id': query_id,
            'feedback': feedback,
            'timestamp': time.time()
        })
        
        # Learn from feedback
        self._update_preferences(profile, feedback)
    
    def _update_preferences(self, profile: UserProfile,
                            feedback: Dict):
        """Update user preferences from feedback"""
        
        if feedback.get('too_technical'):
            # Simplify future answers
            if profile.expertise_level == 'expert':
                profile.expertise_level = 'intermediate'
            elif profile.expertise_level == 'intermediate':
                profile.expertise_level = 'beginner'
        
        if feedback.get('preferred_source'):
            source = feedback['preferred_source']
            if source not in profile.preferred_sources:
                profile.preferred_sources.append(source)

class ExpertiseDetector:
    """Detect user expertise from queries"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_expertise(self, queries: List[str]) -> str:
        """Infer expertise level from query patterns"""
        
        if len(queries) < 3:
            return 'intermediate'
        
        sample = '\n'.join(queries[-5:])
        
        prompt = f"""Analyze these queries to determine user expertise:

{sample}

Classify as: beginner, intermediate, or expert

Expertise level:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        for level in ['beginner', 'intermediate', 'expert']:
            if level in response.lower():
                return level
        
        return 'intermediate'
```

**Interview Tips:**
- Explicit profiles vs learned preferences
- Feedback loop for continuous improvement
- Expertise level affects answer complexity
- Balance personalization with diversity

---

## LLM Application Patterns

### Question 54
- [ ] Done
**How do you implement LLM-based agents with tool use and function calling?**

**Answer:**

**Definition:**
**LLM Agents**: LLM reasons about actions, calls **tools/functions**, observes results, iterates until task complete. Key components: **tool definitions**, **function calling** (structured output), **reasoning loop**, **error handling**.

**Agent Components:**

| Component | Purpose | Implementation |
|-----------|---------|---------------|
| **Tools** | External capabilities | Python functions |
| **Function calling** | Structured output | JSON schema |
| **Reasoning** | Plan actions | Prompt engineering |
| **Memory** | Track state | Context management |

**Python Code Example:**
```python
from typing import List, Dict, Callable, Any
from dataclasses import dataclass
import json

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict  # JSON schema
    function: Callable

class FunctionCallingAgent:
    """Agent with tool use via function calling"""
    
    def __init__(self, llm, tools: List[Tool], max_iterations: int = 10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
    
    def run(self, task: str) -> Dict:
        """Execute task with tool use"""
        
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task}
        ]
        
        for i in range(self.max_iterations):
            # Get LLM response with function calling
            response = self.llm.chat(
                messages=messages,
                tools=self._format_tools(),
                tool_choice="auto"
            )
            
            # Check if done (no tool call)
            if not response.tool_calls:
                return {
                    'result': response.content,
                    'iterations': i + 1,
                    'success': True
                }
            
            # Execute tool calls
            messages.append(response)  # Assistant message
            
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return {'result': 'Max iterations reached', 'success': False}
    
    def _system_prompt(self) -> str:
        return """You are a helpful assistant with access to tools.
Use tools when needed to complete tasks.
Think step by step before acting.
When you have the final answer, respond directly without calling tools."""
    
    def _format_tools(self) -> List[Dict]:
        """Format tools for API"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self.tools.values()
        ]
    
    def _execute_tool(self, tool_call) -> Any:
        """Execute a tool call"""
        
        tool_name = tool_call.function.name
        
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            args = json.loads(tool_call.function.arguments)
            result = self.tools[tool_name].function(**args)
            return result
        except Exception as e:
            return {"error": str(e)}

# Example tools
def create_tools() -> List[Tool]:
    
    def search_web(query: str) -> Dict:
        """Search the web"""
        return {"results": [f"Result for: {query}"]}
    
    def calculate(expression: str) -> Dict:
        """Evaluate math expression"""
        try:
            result = eval(expression)  # Use safe_eval in production
            return {"result": result}
        except:
            return {"error": "Invalid expression"}
    
    def get_weather(city: str) -> Dict:
        """Get weather for city"""
        return {"city": city, "temp": 72, "condition": "sunny"}
    
    return [
        Tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=search_web
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            },
            function=calculate
        ),
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            },
            function=get_weather
        )
    ]

class AgentWithMemory:
    """Agent with persistent memory"""
    
    def __init__(self, base_agent: FunctionCallingAgent):
        self.agent = base_agent
        self.memory = []  # Long-term memory
        self.scratchpad = {}  # Working memory
    
    def run(self, task: str) -> Dict:
        """Run with memory context"""
        
        # Add memory context to task
        memory_context = self._format_memory()
        enhanced_task = f"{memory_context}\n\nTask: {task}"
        
        result = self.agent.run(enhanced_task)
        
        # Update memory
        self._update_memory(task, result)
        
        return result
    
    def _format_memory(self) -> str:
        if not self.memory:
            return ""
        
        recent = self.memory[-5:]
        return "Previous context:\n" + "\n".join([
            f"- {m['summary']}" for m in recent
        ])
    
    def _update_memory(self, task: str, result: Dict):
        self.memory.append({
            'task': task,
            'result': result.get('result', ''),
            'summary': f"Task: {task[:50]}... -> Done"
        })
```

**Interview Tips:**
- Use structured function calling for reliability
- Handle tool errors gracefully
- Limit iterations to prevent infinite loops
- Consider tool selection based on task type

---

### Question 55
- [ ] Done
**What are the best practices for implementing content filtering and safety measures?**

**Answer:**

**Definition:**
**LLM Safety**: prevent harmful inputs/outputs. Layers: **input filtering** (block bad prompts), **output filtering** (block bad generations), **guardrails** (behavior constraints), **monitoring** (detect issues).

**Safety Layers:**

| Layer | What It Does | Implementation |
|-------|-------------|---------------|
| **Input filter** | Block harmful prompts | Keyword + ML classifier |
| **System prompt** | Define behavior bounds | Prompt engineering |
| **Output filter** | Block harmful outputs | Post-generation check |
| **Rate limiting** | Prevent abuse | Request throttling |

**Python Code Example:**
```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class SafetyCategory(Enum):
    SAFE = "safe"
    HARMFUL = "harmful"
    SENSITIVE = "sensitive"
    BLOCKED = "blocked"

@dataclass
class SafetyResult:
    is_safe: bool
    category: SafetyCategory
    reason: str
    modified_content: str = None

class ContentFilter:
    """Multi-layer content filtering"""
    
    def __init__(self, llm_classifier=None):
        self.llm_classifier = llm_classifier
        
        # Blocklist patterns
        self.blocked_patterns = [
            r'\b(bomb|weapon|explosive)\b.*\b(make|build|create)\b',
            r'\b(hack|crack)\b.*\b(password|system)\b',
            r'\b(kill|harm|hurt)\b.*\b(person|people)\b',
        ]
        
        # Sensitive topics (require extra care)
        self.sensitive_topics = [
            'medical advice',
            'legal advice',
            'financial advice',
            'mental health'
        ]
    
    def filter_input(self, text: str) -> SafetyResult:
        """Filter input prompt"""
        
        text_lower = text.lower()
        
        # Check blocklist
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return SafetyResult(
                    is_safe=False,
                    category=SafetyCategory.BLOCKED,
                    reason=f"Matched blocked pattern"
                )
        
        # Check ML classifier if available
        if self.llm_classifier:
            ml_result = self._ml_classify(text)
            if not ml_result['safe']:
                return SafetyResult(
                    is_safe=False,
                    category=SafetyCategory.HARMFUL,
                    reason=ml_result['reason']
                )
        
        # Check sensitive topics
        for topic in self.sensitive_topics:
            if topic in text_lower:
                return SafetyResult(
                    is_safe=True,
                    category=SafetyCategory.SENSITIVE,
                    reason=f"Contains sensitive topic: {topic}"
                )
        
        return SafetyResult(
            is_safe=True,
            category=SafetyCategory.SAFE,
            reason=""
        )
    
    def filter_output(self, text: str) -> SafetyResult:
        """Filter generated output"""
        
        # PII detection
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b\d{16}\b', 'Credit card'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
        ]
        
        modified = text
        pii_found = []
        
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, text):
                pii_found.append(pii_type)
                modified = re.sub(pattern, f'[{pii_type} REDACTED]', modified)
        
        if pii_found:
            return SafetyResult(
                is_safe=True,
                category=SafetyCategory.SENSITIVE,
                reason=f"PII redacted: {pii_found}",
                modified_content=modified
            )
        
        # Check for harmful content in output
        if self.llm_classifier:
            ml_result = self._ml_classify(text)
            if not ml_result['safe']:
                return SafetyResult(
                    is_safe=False,
                    category=SafetyCategory.HARMFUL,
                    reason=ml_result['reason']
                )
        
        return SafetyResult(
            is_safe=True,
            category=SafetyCategory.SAFE,
            reason=""
        )
    
    def _ml_classify(self, text: str) -> Dict:
        """Use ML model for classification"""
        # Placeholder - use actual classifier
        return {'safe': True, 'reason': ''}

class Guardrails:
    """LLM behavior guardrails"""
    
    def __init__(self, llm):
        self.llm = llm
        self.rules = []
    
    def add_rule(self, rule: str):
        """Add behavior rule"""
        self.rules.append(rule)
    
    def get_system_prompt(self) -> str:
        """Get system prompt with guardrails"""
        
        rules_text = '\n'.join([f"- {r}" for r in self.rules])
        
        return f"""You are a helpful assistant. Follow these rules strictly:

{rules_text}

If a user asks you to violate these rules, politely decline.
If you're unsure whether something violates a rule, err on the side of caution."""
    
    def check_response(self, response: str) -> Tuple[bool, str]:
        """Check if response violates rules"""
        
        prompt = f"""Check if this response violates any of these rules:
{chr(10).join([f'- {r}' for r in self.rules])}

Response: {response}

Does it violate any rule? (yes/no and which rule):"""
        
        check = self.llm.generate(prompt, temperature=0)
        
        if 'yes' in check.lower():
            return False, check
        return True, ""

class SafeLLM:
    """LLM wrapper with safety measures"""
    
    def __init__(self, llm, content_filter: ContentFilter,
                 guardrails: Guardrails):
        self.llm = llm
        self.filter = content_filter
        self.guardrails = guardrails
    
    def generate(self, prompt: str) -> Dict:
        """Generate with safety checks"""
        
        # Input filter
        input_check = self.filter.filter_input(prompt)
        if not input_check.is_safe:
            return {
                'response': "I can't help with that request.",
                'blocked': True,
                'reason': input_check.reason
            }
        
        # Add guardrails to system prompt
        full_prompt = f"{self.guardrails.get_system_prompt()}\n\nUser: {prompt}"
        
        # Generate
        response = self.llm.generate(full_prompt)
        
        # Output filter
        output_check = self.filter.filter_output(response)
        
        if not output_check.is_safe:
            return {
                'response': "I generated content that violated safety guidelines. Please rephrase your request.",
                'blocked': True,
                'reason': output_check.reason
            }
        
        # Use modified content if PII was redacted
        final_response = output_check.modified_content or response
        
        return {
            'response': final_response,
            'blocked': False,
            'category': output_check.category.value
        }
```

**Interview Tips:**
- Defense in depth: multiple layers
- Input filtering catches prompt injection
- Output filtering catches generation issues
- Log blocked content for analysis (but not content itself)

---

### Question 56
- [ ] Done
**How do you handle data retention and privacy concerns when using LLM APIs with user data?**

**Answer:**

**Definition:**
**LLM Privacy**: protect user data sent to LLM APIs. Concerns: **data retention** (how long stored), **training use** (is data used to train?), **transmission** (encryption), **access** (who sees data). Solutions: redaction, opt-out, self-hosted.

**Privacy Strategies:**

| Concern | Solution | Trade-off |
|---------|----------|----------|
| **API training** | Use opt-out APIs | May cost more |
| **Sensitive data** | PII redaction | Reduces context |
| **Compliance** | Self-hosted LLM | Higher cost |
| **Logging** | Minimal retention | Harder debugging |

**Python Code Example:**
```python
import re
import hashlib
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PrivacyConfig:
    redact_pii: bool = True
    log_prompts: bool = False
    retention_days: int = 30
    anonymize_logs: bool = True
    use_private_api: bool = False

class PrivacyPreservingLLM:
    """LLM wrapper with privacy protections"""
    
    def __init__(self, llm, config: PrivacyConfig):
        self.llm = llm
        self.config = config
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN', '[SSN]'),
            (r'\b\d{16}\b', 'CARD', '[CARD]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', '[EMAIL]'),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', '[PHONE]'),
            (r'\b\d{5}(-\d{4})?\b', 'ZIP', '[ZIP]'),
        ]
        self.pii_map = {}  # For restoration
    
    def generate(self, prompt: str, user_id: str = None) -> Dict:
        """Generate with privacy protections"""
        
        # Redact PII before sending to API
        if self.config.redact_pii:
            safe_prompt, pii_map = self._redact_pii(prompt)
        else:
            safe_prompt = prompt
            pii_map = {}
        
        # Call API
        response = self.llm.generate(safe_prompt)
        
        # Restore PII in response if needed
        if pii_map:
            response = self._restore_pii(response, pii_map)
        
        # Log with privacy
        if self.config.log_prompts:
            self._log_private(prompt, response, user_id)
        
        return {
            'response': response,
            'pii_redacted': bool(pii_map)
        }
    
    def _redact_pii(self, text: str) -> tuple:
        """Redact PII and return mapping for restoration"""
        
        pii_map = {}
        redacted = text
        counter = 0
        
        for pattern, pii_type, replacement in self.pii_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                placeholder = f"{replacement}{counter}"
                pii_map[placeholder] = match
                redacted = redacted.replace(match, placeholder, 1)
                counter += 1
        
        return redacted, pii_map
    
    def _restore_pii(self, text: str, pii_map: Dict) -> str:
        """Restore PII from placeholders"""
        
        restored = text
        for placeholder, original in pii_map.items():
            restored = restored.replace(placeholder, original)
        return restored
    
    def _log_private(self, prompt: str, response: str, user_id: str):
        """Log with privacy protections"""
        
        log_entry = {
            'timestamp': time.time(),
            'user_hash': hashlib.sha256(user_id.encode()).hexdigest() if user_id else None,
        }
        
        if self.config.anonymize_logs:
            # Don't log actual content
            log_entry['prompt_length'] = len(prompt)
            log_entry['response_length'] = len(response)
            log_entry['prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        else:
            # Redact before logging
            log_entry['prompt'], _ = self._redact_pii(prompt)
            log_entry['response'], _ = self._redact_pii(response)
        
        # Store log_entry
        self._store_log(log_entry)
    
    def _store_log(self, entry: Dict):
        # Store with retention policy
        pass

class DataRetentionManager:
    """Manage data retention policies"""
    
    def __init__(self, storage, retention_days: int = 30):
        self.storage = storage
        self.retention_days = retention_days
    
    def store(self, key: str, data: Dict, user_id: str):
        """Store with retention metadata"""
        
        record = {
            'data': data,
            'user_id_hash': hashlib.sha256(user_id.encode()).hexdigest(),
            'created_at': time.time(),
            'expires_at': time.time() + (self.retention_days * 86400)
        }
        
        self.storage.put(key, record)
    
    def cleanup_expired(self):
        """Delete expired records"""
        
        current_time = time.time()
        
        for key, record in self.storage.scan():
            if record.get('expires_at', 0) < current_time:
                self.storage.delete(key)
    
    def delete_user_data(self, user_id: str):
        """Delete all data for a user (GDPR right to erasure)"""
        
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()
        
        for key, record in self.storage.scan():
            if record.get('user_id_hash') == user_hash:
                self.storage.delete(key)

class PrivateAPIRouter:
    """Route to private API for sensitive requests"""
    
    def __init__(self, public_llm, private_llm, sensitivity_detector):
        self.public_llm = public_llm
        self.private_llm = private_llm
        self.detector = sensitivity_detector
    
    def generate(self, prompt: str) -> Dict:
        """Route based on sensitivity"""
        
        is_sensitive = self.detector.is_sensitive(prompt)
        
        if is_sensitive:
            # Use private/self-hosted LLM
            response = self.private_llm.generate(prompt)
            return {'response': response, 'api': 'private'}
        else:
            # OK to use public API
            response = self.public_llm.generate(prompt)
            return {'response': response, 'api': 'public'}

class SensitivityDetector:
    """Detect sensitive content"""
    
    def __init__(self):
        self.sensitive_keywords = [
            'ssn', 'social security',
            'credit card', 'bank account',
            'password', 'secret',
            'medical', 'diagnosis',
            'salary', 'compensation'
        ]
    
    def is_sensitive(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.sensitive_keywords)
```

**Interview Tips:**
- Understand API provider data policies
- PII redaction before API call is critical
- Consider self-hosted for highly sensitive data
- Implement user data deletion for compliance

---

### Question 57
- [ ] Done
**What strategies help ensure LLM-generated content meets specific brand voice and style requirements?**

**Answer:**

**Definition:**
**Brand voice control**: ensure LLM outputs match company tone, style, terminology. Techniques: **system prompts** (define voice), **few-shot examples** (show style), **fine-tuning** (train on brand content), **post-processing** (adjust output).

**Brand Control Methods:**

| Method | Effort | Consistency | Flexibility |
|--------|--------|-------------|------------|
| **System prompt** | Low | Medium | High |
| **Few-shot** | Medium | High | Medium |
| **Fine-tuning** | High | Very High | Low |
| **Post-process** | Medium | Medium | High |

**Python Code Example:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class BrandGuidelines:
    name: str
    tone: str  # e.g., "friendly and professional"
    vocabulary: Dict[str, str]  # forbidden -> preferred terms
    examples: List[Dict]  # Example Q&A pairs
    rules: List[str]  # Specific rules
    persona: str  # Brand persona description

class BrandVoiceLLM:
    """LLM with brand voice control"""
    
    def __init__(self, llm, guidelines: BrandGuidelines):
        self.llm = llm
        self.guidelines = guidelines
    
    def generate(self, prompt: str) -> Dict:
        """Generate with brand voice"""
        
        # Build branded prompt
        system_prompt = self._build_system_prompt()
        few_shot = self._build_few_shot()
        
        full_prompt = f"""{system_prompt}

{few_shot}

User: {prompt}

Assistant:"""
        
        response = self.llm.generate(full_prompt)
        
        # Post-process for vocabulary
        response = self._apply_vocabulary(response)
        
        # Validate brand compliance
        compliance = self._check_compliance(response)
        
        return {
            'response': response,
            'brand_compliant': compliance['compliant'],
            'issues': compliance.get('issues', [])
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with brand guidelines"""
        
        rules_text = '\n'.join([f"- {r}" for r in self.guidelines.rules])
        
        return f"""You are {self.guidelines.name}'s AI assistant.

Persona: {self.guidelines.persona}

Tone: {self.guidelines.tone}

Important rules:
{rules_text}

Always maintain this brand voice in all responses."""
    
    def _build_few_shot(self) -> str:
        """Build few-shot examples"""
        
        if not self.guidelines.examples:
            return ""
        
        examples = []
        for ex in self.guidelines.examples[:3]:  # Use up to 3 examples
            examples.append(f"User: {ex['user']}\nAssistant: {ex['assistant']}")
        
        return "Here are examples of how to respond:\n\n" + "\n\n".join(examples)
    
    def _apply_vocabulary(self, text: str) -> str:
        """Replace forbidden terms with preferred ones"""
        
        result = text
        for forbidden, preferred in self.guidelines.vocabulary.items():
            result = result.replace(forbidden, preferred)
            result = result.replace(forbidden.capitalize(), preferred.capitalize())
        
        return result
    
    def _check_compliance(self, text: str) -> Dict:
        """Check brand compliance"""
        
        issues = []
        
        # Check forbidden vocabulary
        for forbidden in self.guidelines.vocabulary.keys():
            if forbidden.lower() in text.lower():
                issues.append(f"Contains forbidden term: {forbidden}")
        
        # Check tone (simple heuristics)
        if self.guidelines.tone == "friendly" and text.startswith("No"):
            issues.append("Response starts with 'No' - may not be friendly")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues
        }

class StyleTransformer:
    """Transform content to match style"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def transform(self, text: str, target_style: str) -> str:
        """Transform text to target style"""
        
        prompt = f"""Rewrite this text in a {target_style} style.
Keep the same information but change the tone and word choice.

Original: {text}

Rewritten ({target_style} style):"""
        
        return self.llm.generate(prompt, temperature=0.5)
    
    def match_example(self, text: str, example: str) -> str:
        """Transform text to match example style"""
        
        prompt = f"""Rewrite the text to match the style of the example.

Example of desired style:
{example}

Text to rewrite:
{text}

Rewritten text:"""
        
        return self.llm.generate(prompt, temperature=0.5)

class BrandConsistencyChecker:
    """Check brand consistency"""
    
    def __init__(self, llm, guidelines: BrandGuidelines):
        self.llm = llm
        self.guidelines = guidelines
    
    def check(self, text: str) -> Dict:
        """Check text against brand guidelines"""
        
        prompt = f"""Evaluate if this text matches the brand guidelines.

Brand: {self.guidelines.name}
Tone: {self.guidelines.tone}
Rules: {', '.join(self.guidelines.rules[:5])}

Text: {text}

Rate compliance (1-5) and list any issues:"""
        
        response = self.llm.generate(prompt, temperature=0)
        
        # Parse response
        import re
        score_match = re.search(r'(\d)', response)
        score = int(score_match.group(1)) if score_match else 3
        
        return {
            'score': score,
            'compliant': score >= 4,
            'feedback': response
        }

# Example usage
def create_brand_guidelines():
    return BrandGuidelines(
        name="TechCorp",
        tone="friendly, professional, and helpful",
        vocabulary={
            "can't": "cannot",
            "won't": "will not",
            "cheap": "cost-effective",
            "problem": "challenge"
        },
        examples=[
            {
                "user": "I have a problem with my order.",
                "assistant": "I'd be happy to help you with that challenge! Could you share your order number so I can look into this for you?"
            },
            {
                "user": "Your product is too expensive.",
                "assistant": "I understand value is important to you! Let me share some of our cost-effective options that might be a great fit for your needs."
            }
        ],
        rules=[
            "Always start with a positive acknowledgment",
            "Never blame the customer",
            "Use 'we' instead of 'I' for company actions",
            "End with an offer to help further"
        ],
        persona="A knowledgeable and friendly tech expert who loves helping customers succeed"
    )
```

**Interview Tips:**
- Few-shot examples are most effective for style
- Vocabulary replacement is simple but effective
- Fine-tuning for very strict requirements
- Always validate outputs against guidelines

---

### Question 58
- [ ] Done
**How do you implement cost optimization strategies for large-scale LLM API usage?**

**Answer:**

**Definition:**
**LLM Cost Optimization**: reduce API costs while maintaining quality. Strategies: **caching** (reuse responses), **model routing** (smaller model when possible), **prompt optimization** (fewer tokens), **batching** (reduce overhead), **monitoring** (track spend).

**Cost Reduction Strategies:**

| Strategy | Savings | Implementation |
|----------|---------|---------------|
| **Response caching** | 50-80% | Cache common queries |
| **Model routing** | 40-60% | Use GPT-3.5 when OK |
| **Prompt compression** | 20-30% | Remove redundancy |
| **Batching** | 10-20% | Reduce API calls |

**Python Code Example:**
```python
import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class CostMetrics:
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    cached: bool

class CostOptimizedLLM:
    """LLM wrapper with cost optimization"""
    
    def __init__(self, expensive_llm, cheap_llm, 
                 cache_size: int = 10000):
        self.expensive = expensive_llm  # GPT-4
        self.cheap = cheap_llm  # GPT-3.5
        self.cache = {}  # Response cache
        self.cache_size = cache_size
        
        # Cost tracking
        self.costs = defaultdict(float)
        self.savings = 0.0
        
        # Price per 1K tokens (example)
        self.prices = {
            'expensive_input': 0.03,
            'expensive_output': 0.06,
            'cheap_input': 0.001,
            'cheap_output': 0.002
        }
    
    def generate(self, prompt: str, 
                 require_quality: bool = False) -> Dict:
        """Generate with cost optimization"""
        
        # Check cache first
        cache_key = self._cache_key(prompt)
        if cache_key in self.cache:
            self.savings += self._estimate_cost(prompt, 'expensive')
            return {
                'response': self.cache[cache_key],
                'cached': True,
                'cost': 0.0
            }
        
        # Route to appropriate model
        if require_quality or self._needs_quality(prompt):
            response, cost = self._call_expensive(prompt)
        else:
            response, cost = self._call_cheap(prompt)
        
        # Cache response
        self._cache_response(cache_key, response)
        
        return {
            'response': response,
            'cached': False,
            'cost': cost
        }
    
    def _needs_quality(self, prompt: str) -> bool:
        """Determine if prompt needs expensive model"""
        
        # Heuristics for quality routing
        quality_indicators = [
            'analyze', 'complex', 'detailed',
            'compare', 'explain in depth',
            'code review', 'debug'
        ]
        
        prompt_lower = prompt.lower()
        return any(ind in prompt_lower for ind in quality_indicators)
    
    def _call_expensive(self, prompt: str) -> tuple:
        response = self.expensive.generate(prompt)
        cost = self._calculate_cost(prompt, response, 'expensive')
        self.costs['expensive'] += cost
        return response, cost
    
    def _call_cheap(self, prompt: str) -> tuple:
        response = self.cheap.generate(prompt)
        cost = self._calculate_cost(prompt, response, 'cheap')
        self.costs['cheap'] += cost
        return response, cost
    
    def _calculate_cost(self, prompt: str, response: str,
                        model: str) -> float:
        # Estimate tokens (rough: 4 chars per token)
        prompt_tokens = len(prompt) / 4
        response_tokens = len(response) / 4
        
        input_price = self.prices[f'{model}_input']
        output_price = self.prices[f'{model}_output']
        
        return (prompt_tokens * input_price + 
                response_tokens * output_price) / 1000
    
    def _estimate_cost(self, prompt: str, model: str) -> float:
        return (len(prompt) / 4) * self.prices[f'{model}_input'] / 1000
    
    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _cache_response(self, key: str, response: str):
        if len(self.cache) >= self.cache_size:
            # Simple eviction: remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = response
    
    def get_cost_report(self) -> Dict:
        return {
            'total_cost': sum(self.costs.values()),
            'by_model': dict(self.costs),
            'cache_savings': self.savings,
            'cache_hit_rate': len(self.cache) / max(1, sum(self.costs.values()))
        }

class PromptOptimizer:
    """Optimize prompts to reduce tokens"""
    
    def __init__(self):
        pass
    
    def compress(self, prompt: str) -> str:
        """Compress prompt while preserving meaning"""
        
        # Remove excessive whitespace
        import re
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Remove redundant phrases
        redundant = [
            'please ', 'kindly ', 'I would like you to ',
            'Can you please ', 'Would you mind '
        ]
        for phrase in redundant:
            prompt = prompt.replace(phrase, '')
        
        return prompt.strip()
    
    def truncate_context(self, context: str, 
                         max_tokens: int = 2000) -> str:
        """Truncate context to fit budget"""
        
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Keep first and last parts
        half = max_chars // 2
        return context[:half] + "\n...\n" + context[-half:]

class CostMonitor:
    """Monitor and alert on LLM costs"""
    
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        self.last_reset = time.time()
        self.alerts = []
    
    def record_cost(self, cost: float):
        """Record a cost and check limits"""
        
        # Reset daily if needed
        if time.time() - self.last_reset > 86400:
            self.daily_spend = 0.0
            self.last_reset = time.time()
        
        self.daily_spend += cost
        
        # Check thresholds
        if self.daily_spend > self.daily_budget * 0.8:
            self.alerts.append({
                'type': 'budget_warning',
                'message': f'80% of daily budget used: ${self.daily_spend:.2f}',
                'timestamp': time.time()
            })
        
        if self.daily_spend > self.daily_budget:
            self.alerts.append({
                'type': 'budget_exceeded',
                'message': f'Daily budget exceeded: ${self.daily_spend:.2f}',
                'timestamp': time.time()
            })
    
    def should_block(self) -> bool:
        """Whether to block new requests"""
        return self.daily_spend > self.daily_budget * 1.2
```

**Interview Tips:**
- Caching has highest ROI for repeated queries
- Model routing can halve costs with minimal quality loss
- Track costs per feature/user for optimization
- Set alerts and budgets to prevent runaway costs

---

### Question 59
- [ ] Done
**What are the key performance indicators (KPIs) for measuring LLM application success?**

**Answer:**

**Definition:**
**LLM KPIs**: metrics to measure application success. Categories: **quality** (accuracy, relevance), **performance** (latency, throughput), **cost** (per query, per user), **business** (task completion, user satisfaction), **safety** (errors, blocked content).

**KPI Categories:**

| Category | Key Metrics | Target Range |
|----------|------------|-------------|
| **Quality** | Accuracy, faithfulness | >85% |
| **Performance** | P95 latency, throughput | <3s, >100 QPS |
| **Cost** | Cost/query, cost/user | Business-specific |
| **Business** | Task completion, CSAT | >80%, >4.0 |
| **Safety** | Error rate, safety blocks | <1%, <0.1% |

**Python Code Example:**
```python
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

@dataclass
class LLMMetric:
    timestamp: float
    value: float
    metadata: Dict = field(default_factory=dict)

class LLMKPITracker:
    """Track KPIs for LLM applications"""
    
    def __init__(self):
        self.metrics = defaultdict(list)  # metric_name -> [LLMMetric]
        self.targets = {
            'latency_p95_ms': 3000,
            'accuracy': 0.85,
            'task_completion_rate': 0.80,
            'cost_per_query': 0.05,
            'error_rate': 0.01,
            'user_satisfaction': 4.0
        }
    
    # Quality Metrics
    
    def record_accuracy(self, correct: bool, query_id: str):
        """Track answer accuracy"""
        self.metrics['accuracy'].append(LLMMetric(
            timestamp=time.time(),
            value=1.0 if correct else 0.0,
            metadata={'query_id': query_id}
        ))
    
    def record_faithfulness(self, score: float, query_id: str):
        """Track faithfulness to sources"""
        self.metrics['faithfulness'].append(LLMMetric(
            timestamp=time.time(),
            value=score,
            metadata={'query_id': query_id}
        ))
    
    # Performance Metrics
    
    def record_latency(self, latency_ms: float, query_id: str):
        """Track response latency"""
        self.metrics['latency'].append(LLMMetric(
            timestamp=time.time(),
            value=latency_ms,
            metadata={'query_id': query_id}
        ))
    
    def record_throughput(self, queries_per_second: float):
        """Track system throughput"""
        self.metrics['throughput'].append(LLMMetric(
            timestamp=time.time(),
            value=queries_per_second
        ))
    
    # Cost Metrics
    
    def record_cost(self, cost: float, query_id: str, user_id: str):
        """Track query cost"""
        self.metrics['cost'].append(LLMMetric(
            timestamp=time.time(),
            value=cost,
            metadata={'query_id': query_id, 'user_id': user_id}
        ))
    
    # Business Metrics
    
    def record_task_completion(self, completed: bool, task_id: str):
        """Track task completion"""
        self.metrics['task_completion'].append(LLMMetric(
            timestamp=time.time(),
            value=1.0 if completed else 0.0,
            metadata={'task_id': task_id}
        ))
    
    def record_user_satisfaction(self, rating: float, user_id: str):
        """Track user satisfaction (1-5 scale)"""
        self.metrics['user_satisfaction'].append(LLMMetric(
            timestamp=time.time(),
            value=rating,
            metadata={'user_id': user_id}
        ))
    
    def record_user_feedback(self, helpful: bool, query_id: str):
        """Track thumbs up/down feedback"""
        self.metrics['user_feedback'].append(LLMMetric(
            timestamp=time.time(),
            value=1.0 if helpful else 0.0,
            metadata={'query_id': query_id}
        ))
    
    # Safety Metrics
    
    def record_error(self, error_type: str, query_id: str):
        """Track errors"""
        self.metrics['errors'].append(LLMMetric(
            timestamp=time.time(),
            value=1.0,
            metadata={'error_type': error_type, 'query_id': query_id}
        ))
    
    def record_safety_block(self, reason: str, query_id: str):
        """Track safety-blocked content"""
        self.metrics['safety_blocks'].append(LLMMetric(
            timestamp=time.time(),
            value=1.0,
            metadata={'reason': reason, 'query_id': query_id}
        ))
    
    # Reporting
    
    def get_kpi_report(self, window_hours: int = 24) -> Dict:
        """Get KPI report for time window"""
        
        cutoff = time.time() - (window_hours * 3600)
        report = {'window_hours': window_hours, 'generated_at': time.time()}
        
        # Quality
        accuracy_vals = self._get_values('accuracy', cutoff)
        faithfulness_vals = self._get_values('faithfulness', cutoff)
        report['quality'] = {
            'accuracy': statistics.mean(accuracy_vals) if accuracy_vals else None,
            'faithfulness': statistics.mean(faithfulness_vals) if faithfulness_vals else None
        }
        
        # Performance
        latency_vals = self._get_values('latency', cutoff)
        report['performance'] = {
            'latency_p50': statistics.median(latency_vals) if latency_vals else None,
            'latency_p95': sorted(latency_vals)[int(len(latency_vals) * 0.95)] if len(latency_vals) > 20 else None,
            'latency_avg': statistics.mean(latency_vals) if latency_vals else None,
            'query_count': len(latency_vals)
        }
        
        # Cost
        cost_vals = self._get_values('cost', cutoff)
        report['cost'] = {
            'total': sum(cost_vals) if cost_vals else 0,
            'per_query': statistics.mean(cost_vals) if cost_vals else None,
            'query_count': len(cost_vals)
        }
        
        # Business
        completion_vals = self._get_values('task_completion', cutoff)
        satisfaction_vals = self._get_values('user_satisfaction', cutoff)
        feedback_vals = self._get_values('user_feedback', cutoff)
        report['business'] = {
            'task_completion_rate': statistics.mean(completion_vals) if completion_vals else None,
            'user_satisfaction': statistics.mean(satisfaction_vals) if satisfaction_vals else None,
            'helpful_rate': statistics.mean(feedback_vals) if feedback_vals else None
        }
        
        # Safety
        error_count = len(self._get_values('errors', cutoff))
        safety_count = len(self._get_values('safety_blocks', cutoff))
        total_queries = len(latency_vals) or 1
        report['safety'] = {
            'error_rate': error_count / total_queries,
            'safety_block_rate': safety_count / total_queries,
            'error_count': error_count,
            'safety_block_count': safety_count
        }
        
        # Health check against targets
        report['health'] = self._check_health(report)
        
        return report
    
    def _get_values(self, metric_name: str, since: float) -> List[float]:
        """Get metric values since timestamp"""
        return [
            m.value for m in self.metrics.get(metric_name, [])
            if m.timestamp >= since
        ]
    
    def _check_health(self, report: Dict) -> Dict:
        """Check KPIs against targets"""
        
        health = {'status': 'healthy', 'issues': []}
        
        # Check latency
        p95 = report['performance'].get('latency_p95')
        if p95 and p95 > self.targets['latency_p95_ms']:
            health['issues'].append(f"Latency P95 ({p95:.0f}ms) exceeds target")
        
        # Check accuracy
        accuracy = report['quality'].get('accuracy')
        if accuracy and accuracy < self.targets['accuracy']:
            health['issues'].append(f"Accuracy ({accuracy:.2%}) below target")
        
        # Check error rate
        error_rate = report['safety'].get('error_rate', 0)
        if error_rate > self.targets['error_rate']:
            health['issues'].append(f"Error rate ({error_rate:.2%}) above target")
        
        # Check cost
        cost_per_query = report['cost'].get('per_query')
        if cost_per_query and cost_per_query > self.targets['cost_per_query']:
            health['issues'].append(f"Cost/query (${cost_per_query:.3f}) exceeds target")
        
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 3 else 'critical'
        
        return health
```

**Interview Tips:**
- Balance quality, performance, cost, and safety
- Track both technical and business metrics
- Set realistic targets based on use case
- Monitor trends, not just point-in-time values

---

### Question 60
- [ ] Done
**How do you implement effective fallback mechanisms when LLM services are unavailable?**

**Answer:**

**Definition:**
**LLM Fallbacks**: maintain service when primary LLM is unavailable. Strategies: **backup providers**, **cached responses**, **degraded mode** (simpler features), **queue for later**, **graceful messaging**. Goal: never leave user with no response.

**Fallback Hierarchy:**

| Priority | Fallback | When to Use |
|----------|----------|------------|
| **1** | Secondary LLM | Primary down |
| **2** | Cached response | Seen before |
| **3** | Rule-based | Simple queries |
| **4** | Queue + notify | Can wait |
| **5** | Graceful error | Last resort |

**Python Code Example:**
```python
import time
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

class FallbackLevel(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    CACHED = "cached"
    RULE_BASED = "rule_based"
    QUEUED = "queued"
    ERROR = "error"

@dataclass
class FallbackResponse:
    content: str
    level: FallbackLevel
    latency_ms: float
    degraded: bool

class ResilientLLM:
    """LLM with multiple fallback layers"""
    
    def __init__(self, primary_llm, secondary_llm=None,
                 cache=None, rules_engine=None):
        self.primary = primary_llm
        self.secondary = secondary_llm
        self.cache = cache or {}
        self.rules = rules_engine
        self.queue = []  # Failed requests for retry
        
        # Circuit breaker state
        self.primary_failures = 0
        self.primary_circuit_open = False
        self.circuit_open_until = 0
    
    def generate(self, prompt: str, 
                 allow_fallback: bool = True) -> FallbackResponse:
        """Generate with fallback chain"""
        
        start = time.time()
        
        # Try primary (if circuit closed)
        if not self._is_circuit_open():
            try:
                response = self.primary.generate(prompt, timeout=10)
                self._record_success()
                return FallbackResponse(
                    content=response,
                    level=FallbackLevel.PRIMARY,
                    latency_ms=(time.time() - start) * 1000,
                    degraded=False
                )
            except Exception as e:
                self._record_failure()
        
        if not allow_fallback:
            return self._error_response(start)
        
        # Try secondary LLM
        if self.secondary:
            try:
                response = self.secondary.generate(prompt, timeout=10)
                return FallbackResponse(
                    content=response,
                    level=FallbackLevel.SECONDARY,
                    latency_ms=(time.time() - start) * 1000,
                    degraded=True
                )
            except:
                pass
        
        # Try cache
        cache_key = self._cache_key(prompt)
        if cache_key in self.cache:
            return FallbackResponse(
                content=self.cache[cache_key],
                level=FallbackLevel.CACHED,
                latency_ms=(time.time() - start) * 1000,
                degraded=True
            )
        
        # Try rule-based response
        if self.rules:
            rule_response = self.rules.try_match(prompt)
            if rule_response:
                return FallbackResponse(
                    content=rule_response,
                    level=FallbackLevel.RULE_BASED,
                    latency_ms=(time.time() - start) * 1000,
                    degraded=True
                )
        
        # Queue for later processing
        self._queue_request(prompt)
        
        return FallbackResponse(
            content="I'm experiencing technical difficulties. Your request has been queued and I'll get back to you shortly.",
            level=FallbackLevel.QUEUED,
            latency_ms=(time.time() - start) * 1000,
            degraded=True
        )
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.primary_circuit_open:
            if time.time() > self.circuit_open_until:
                # Try to close circuit (half-open)
                self.primary_circuit_open = False
                self.primary_failures = 0
                return False
            return True
        return False
    
    def _record_success(self):
        """Record successful call"""
        self.primary_failures = 0
    
    def _record_failure(self):
        """Record failed call and maybe open circuit"""
        self.primary_failures += 1
        
        if self.primary_failures >= 3:
            self.primary_circuit_open = True
            self.circuit_open_until = time.time() + 30  # 30s timeout
    
    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _queue_request(self, prompt: str):
        self.queue.append({
            'prompt': prompt,
            'timestamp': time.time()
        })
    
    def _error_response(self, start: float) -> FallbackResponse:
        return FallbackResponse(
            content="I'm currently unavailable. Please try again later.",
            level=FallbackLevel.ERROR,
            latency_ms=(time.time() - start) * 1000,
            degraded=True
        )
    
    async def process_queue(self):
        """Process queued requests when service recovers"""
        while self.queue:
            request = self.queue.pop(0)
            try:
                response = self.primary.generate(request['prompt'])
                # Notify user (webhook, email, etc.)
                await self._notify_user(request, response)
            except:
                # Re-queue if still failing
                self.queue.append(request)
                break
    
    async def _notify_user(self, request: Dict, response: str):
        # Implement notification logic
        pass

class RuleBasedFallback:
    """Simple rule-based responses for common queries"""
    
    def __init__(self):
        self.rules = [
            {
                'patterns': ['hello', 'hi', 'hey'],
                'response': 'Hello! How can I help you today?'
            },
            {
                'patterns': ['help', 'support'],
                'response': 'I\'m here to help. What do you need assistance with?'
            },
            {
                'patterns': ['hours', 'open'],
                'response': 'Our support is available 24/7. How can we assist you?'
            }
        ]
    
    def try_match(self, prompt: str) -> Optional[str]:
        """Try to match prompt to a rule"""
        
        prompt_lower = prompt.lower()
        
        for rule in self.rules:
            if any(p in prompt_lower for p in rule['patterns']):
                return rule['response']
        
        return None

class HealthChecker:
    """Check LLM service health"""
    
    def __init__(self, llm, check_interval: int = 30):
        self.llm = llm
        self.check_interval = check_interval
        self.is_healthy = True
        self.last_check = 0
    
    def check(self) -> bool:
        """Check if LLM is healthy"""
        
        if time.time() - self.last_check < self.check_interval:
            return self.is_healthy
        
        try:
            # Simple health check query
            response = self.llm.generate("Say 'OK'", timeout=5)
            self.is_healthy = 'ok' in response.lower()
        except:
            self.is_healthy = False
        
        self.last_check = time.time()
        return self.is_healthy

class GracefulDegradation:
    """Gracefully degrade features based on system health"""
    
    def __init__(self, full_llm, simple_llm, health_checker):
        self.full = full_llm
        self.simple = simple_llm
        self.health = health_checker
        
        self.degradation_levels = {
            'full': ['chat', 'analysis', 'generation', 'rag'],
            'degraded': ['chat', 'simple_rag'],
            'minimal': ['cached_responses'],
            'offline': []
        }
    
    def get_available_features(self) -> List[str]:
        """Get currently available features"""
        
        if self.health.check():
            return self.degradation_levels['full']
        elif self.simple and self._check_simple():
            return self.degradation_levels['degraded']
        else:
            return self.degradation_levels['minimal']
    
    def _check_simple(self) -> bool:
        try:
            self.simple.generate("test", timeout=3)
            return True
        except:
            return False
```

**Interview Tips:**
- Multiple fallback layers provide resilience
- Circuit breaker prevents cascade failures
- Cache common responses for offline mode
- Always provide graceful error messages
- Queue requests for async processing when possible

---


---

# --- ChatGPT (from 06_ai_nlp/04_chatgpt) - theory questions ---

# Chatgpt Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What isChatGPTand how does it relate to language models likeGPT-3?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Explain the concept of alanguage model.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Explain the significance offine-tuningin language models.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**How doestokenizationwork in the context oftransformer models?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**What arepositional encodings, and why are they important intransformers?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Explain the concept ofattention mechanismsinGPT models.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How does thedecoder-only architectureof GPT facilitate language generation?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Describe thetraining processof aGPT model.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**What are some common issues faced during thetraining of large language models?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What are the steps involved inpre-processing input datafor ChatGPT?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Explain how to handlepost-processingof the output generated by ChatGPT.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Describe thehyperparameter tuningprocess for improving model performance.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**What are some methods for mitigatingbiasin the responses ofChatGPT?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Explain the importance ofhardware acceleratorslikeGPUsfor trainingChatGPT.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Explain how to conducterror analysison the outputs ofChatGPT.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Explain howChatGPTcan be customized to act as apersonal assistant.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Describe howChatGPTcould be augmented formulti-lingual support.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**How doesreinforcement learning from human feedback (RLHF)shape the responses of models likeChatGPT?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**What isprompt engineering, and how does it apply to optimizingChatGPToutputs?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**What are the dataprivacy considerationswhen usingChatGPTin applications that handle sensitive information?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Explain thecompliance issuesthat might arise withChatGPTin regulated industries.**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Describe the steps you would take ifChatGPTbegan to provide incorrect or inconsistent information.**

**Answer:** _[To be filled]_

---


---

# --- ChatGPT (from 06_ai_nlp/04_chatgpt) - coding questions ---

# Chatgpt Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How would you measure the success of aChatGPT implementationin a business?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Write aPython scriptto generate creative stories using a pre-trained model likeGPT-2.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Implement achatbotthat usesChatGPTto answer domain-specific questions.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Code an example offine-tuning GPT-3forsentiment analysison movie reviews.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Demonstrate how to use thetransformers libraryto integrateChatGPTwith a web application.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Create aPython functionthat preprocesses input text fortokenizationspecific toGPT models.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Write aPython scriptthat interacts withOpenAI’s APIto useChatGPTfor summarizing text documents.**

**Answer:** _[To be filled]_

---


---

# --- ChatGPT (from 06_ai_nlp/04_chatgpt) - general questions ---

# Chatgpt Interview Questions - General Questions

## Question 1
- [ ] Done

**How dotransformers, the architecture behind ChatGPT, work?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What differentiatesChatGPTfrom rule-based chatbots?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Give a detailed description of theGPT model architecture.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**How can you fine-tune a pre-trained model likeGPT-3for a specific task?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you ensure thepromptgiven to ChatGPT is effective for the intended purpose?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Provide examples of howChatGPTcan be integrated into different industry workflows.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How canChatGPTbe used forcreative writing applications?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**How canChatGPTbe used in conjunction with otherAI services?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**What approaches can be used to reduce theinference timeof language models?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Suggest methods to improveresponse qualityandrelevancein ChatGPT.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How can you useA/B testingto improve ChatGPT interactions?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**How canexternal knowledge sourcesbe integrated intoChatGPT?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How couldintellectual property rightsaffect the deployment of ChatGPT-based solutions?**

**Answer:** _[To be filled]_

---


---

# --- ChatGPT (from 06_ai_nlp/04_chatgpt) - scenario based questions ---

# Chatgpt Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss thecostandresource implicationsof training models likeGPT-3.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Discuss the challenges of deployingChatGPTin aproduction environment.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Discuss thescalability challengesfor models likeGPT-3.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**How would you handleperformance bottleneckswhen usingChatGPT?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Discuss howquantizationandpruningmay be used withGPT models.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**How would you analyze the performance ofChatGPTon a given text task?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Discuss the role ofcontextin maintaining coherent conversations withChatGPT.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Discuss recent research advancements intransformer models.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Discusszero-shotandfew-shotlearning capabilities ofGPT models.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**How would you debug issues related to the language generation quality ofChatGPT?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**What are some commonfailure modesof ChatGPT in real-world applications and how would you address them?**

**Answer:** _[To be filled]_

---

---

# --- Hugging Face Transformers Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose the optimal Hugging Face model variant for specific NLP tasks considering computational constraints?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What are the best practices for fine-tuning Hugging Face transformers on domain-specific datasets efficiently?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you implement custom tokenization strategies with Hugging Face transformers for specialized text processing?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When should you use Hugging Face's pipeline API versus direct model/tokenizer usage for production applications?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you optimize Hugging Face model inference speed and memory usage for real-time applications?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you implement efficient batch processing with Hugging Face transformers for high-throughput scenarios?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle Hugging Face model versioning and updates in production environments?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When would you implement custom Hugging Face model architectures versus using pretrained models?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you use Hugging Face datasets library for efficient data loading and preprocessing pipelines?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you manage GPU memory when working with large Hugging Face models?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you implement Hugging Face model quantization and compression for edge deployment?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When should you use Hugging Face's Trainer class versus custom training loops for model fine-tuning?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you handle Hugging Face model integration with distributed training frameworks?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you implement effective prompt engineering with Hugging Face language models?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you optimize Hugging Face model loading and initialization times for serverless deployments?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When would you use Hugging Face's model hub versus self-hosted model repositories?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement Hugging Face model evaluation and benchmarking workflows?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for Hugging Face model deployment using containerization and orchestration?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle Hugging Face model fine-tuning with limited computational resources?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When should you use Hugging Face's automatic model selection versus manual model configuration?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement Hugging Face model integration with vector databases and retrieval systems?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What strategies help you manage Hugging Face model licensing and compliance requirements?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle Hugging Face model adaptation for multilingual and cross-lingual applications?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When would you implement custom Hugging Face data collators versus using default ones?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you optimize Hugging Face model performance for specific hardware accelerators?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What techniques help you implement Hugging Face model interpretability and explanation workflows?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle Hugging Face model integration with streaming and real-time processing systems?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When should you use Hugging Face's model parallelism versus data parallelism for large model training?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you implement Hugging Face model monitoring and performance tracking in production?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What are the considerations for Hugging Face model security and vulnerability management?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle Hugging Face model fine-tuning with noisy or limited labeled data?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When would you use Hugging Face's model optimization tools versus external optimization frameworks?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement Hugging Face model integration with MLOps and experiment tracking systems?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What strategies help you manage Hugging Face model complexity in microservices architectures?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle Hugging Face model deployment across different cloud providers and platforms?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When should you use Hugging Face's model compression techniques versus maintaining full precision?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you implement Hugging Face model A/B testing and gradual rollout strategies?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you optimize Hugging Face model training convergence and stability?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you handle Hugging Face model integration with feature stores and data versioning systems?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When would you implement custom Hugging Face model heads versus using pre-defined architectures?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you use Hugging Face transformers for multimodal applications combining text, images, and audio?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What are the best practices for Hugging Face model debugging and troubleshooting?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you handle Hugging Face model adaptation for specific industries like healthcare or finance?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When should you use Hugging Face's model distillation techniques for creating smaller, faster models?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you implement Hugging Face model integration with business intelligence and reporting systems?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What strategies help you manage Hugging Face model dependencies and environment configuration?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle Hugging Face model migration and legacy system integration?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When would you implement Hugging Face model custom loss functions versus using standard objectives?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you use Hugging Face transformers for specialized tasks like code generation or mathematical reasoning?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you integrate Hugging Face models with automated ML pipelines and continuous deployment systems?**

**Answer:** _[To be filled]_

---

