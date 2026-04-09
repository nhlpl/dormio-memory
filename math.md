Here's the essential mathematics behind the Dormio micro-framework—concise, code-focused, and directly applicable to implementing persistent LLM memory.

---

## 📐 Linear Algebra for MEMIT Weight Editing

**Goal**: Edit MLP weights to inject a new fact without disrupting existing knowledge.

**Core equation**: Given a set of key-value pairs $(k_i, v_i)$ representing facts to edit, find minimal weight update $\Delta W$ such that $W' k_i \approx v_i$ for all $i$.

```python
import torch

def memit_edit(weights: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Apply MEMIT edit using least-squares solution.
    
    weights: (d_out, d_in) - target MLP layer weights
    keys: (n_facts, d_in) - input activations for each fact
    values: (n_facts, d_out) - desired output activations
    """
    # Solve: weights @ keys.T ≈ values.T
    # => keys @ weights.T ≈ values
    # Least squares: Δ = (values - weights @ keys.T) @ keys @ (keys.T @ keys)^{-1}
    
    K = keys  # (n, d_in)
    V = values  # (n, d_out)
    
    # Compute pseudo-inverse of K
    K_pinv = torch.linalg.pinv(K)  # (d_in, n)
    
    # Compute minimal update
    delta = (V - weights @ K.T) @ K_pinv.T  # (d_out, d_in)
    
    return weights + 0.01 * delta  # Small step size
```

---

## 🔷 Singular Value Decomposition (SVD) for Null-Space Constraints

**Goal**: Ensure new edits are orthogonal to existing knowledge to prevent interference.

**Math**: For a matrix $A$ of existing edit activations, its null space is spanned by right singular vectors corresponding to near-zero singular values.

```python
def compute_null_space(activations: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """
    Compute null space basis from existing edit activations.
    
    activations: (n_edits, d_model) - hidden states of existing edits
    
    Returns: (d_model, null_dim) - orthonormal basis for null space
    """
    # SVD: A = U @ diag(S) @ V.T
    U, S, Vh = torch.linalg.svd(activations, full_matrices=False)
    V = Vh.T  # Right singular vectors
    
    # Null space = vectors with singular value below threshold
    null_mask = S < threshold * S.max()
    null_space = V[:, null_mask]
    
    return null_space

def project_to_null_space(update: torch.Tensor, null_space: torch.Tensor) -> torch.Tensor:
    """
    Project weight update onto null space to avoid interference.
    
    update: (d_out, d_in) - proposed weight change
    null_space: (d_in, null_dim) - null space basis
    """
    # Projection: P_null = null_space @ null_space.T
    # Update_proj = update @ P_null
    return update @ null_space @ null_space.T
```

---

## 📊 Perplexity for Model Health Checks

**Goal**: Detect degradation after edits by measuring how "surprised" the model is by a standard text.

**Definition**: $\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$

```python
def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity - lower is better.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss  # Cross-entropy loss
    return torch.exp(loss).item()

# Example: baseline = 8.5, after edit = 8.7 → 2.3% degradation
```

---

## 🎯 Cosine Similarity for Recall Testing

**Goal**: Quantify how well the model remembers a fact.

**Math**: $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$

```python
def test_recall(model, tokenizer, fact: str, expected: str) -> float:
    """
    Return recall score between 0 and 1.
    """
    prompt = f"Complete: {fact.split(' is ')[0]} is"
    response = generate(model, tokenizer, prompt)
    
    # Embed both expected and actual response
    emb_expected = embed(expected)
    emb_actual = embed(response)
    
    # Cosine similarity
    sim = torch.dot(emb_expected, emb_actual) / (
        torch.norm(emb_expected) * torch.norm(emb_actual)
    )
    return sim.item()

def embed(text: str) -> torch.Tensor:
    """Simple bag-of-words embedding."""
    # In practice, use model's hidden states
    tokens = text.lower().split()
    vec = torch.zeros(1000)
    for t in tokens:
        vec[hash(t) % 1000] += 1
    return vec / vec.norm()
```

---

## 📉 Graduated Dissolution Schedule

**Goal**: Progressively reduce MEMIT edit strength as LoRA consolidation proves successful.

**Math**: $\text{strength}(s) = \max(0, 1 - \alpha \cdot s)$ where $s$ is consolidation stage.

```python
class DissolutionScheduler:
    """
    Manages graduated strength reduction for MEMIT edits.
    """
    def __init__(self, stages: tuple = (1.0, 0.5, 0.1, 0.0)):
        self.stages = stages
        
    def get_strength(self, consolidation_score: float) -> float:
        """
        Map consolidation score (0=new, 1=fully consolidated) to MEMIT strength.
        """
        # Sigmoid-like smooth transition
        # strength = 1 / (1 + exp(k * (score - threshold)))
        k = 10  # Steepness
        threshold = 0.7
        strength = 1 / (1 + torch.exp(torch.tensor(k * (consolidation_score - threshold))))
        return float(strength)
    
    def should_dissolve(self, fact_age_days: float, recall_count: int) -> bool:
        """
        Decide if a fact is ready for dissolution.
        Uses exponential decay model.
        """
        # Base decay rate
        decay_rate = 0.1
        # Recall frequency slows decay
        effective_rate = decay_rate / (1 + 0.1 * recall_count)
        # Survival probability
        survival = torch.exp(torch.tensor(-effective_rate * fact_age_days))
        return survival < 0.1  # Dissolve when <10% survival
```

---

## 🔄 LoRA Weight Update Formula

**Goal**: Efficiently fine-tune with low-rank adaptation.

**Math**: For a weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns $W' = W + BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$.

```python
class LoRAUpdate:
    def __init__(self, d_in: int, d_out: int, rank: int = 8):
        # Low-rank matrices
        self.A = torch.randn(rank, d_in) * 0.01   # (r, d_in)
        self.B = torch.zeros(d_out, rank)          # (d_out, r)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_in)
        return x @ self.A.T @ self.B.T  # (batch, d_out)
    
    def update(self, grad_A: torch.Tensor, grad_B: torch.Tensor, lr: float = 0.01):
        """Gradient descent step."""
        self.A -= lr * grad_A
        self.B -= lr * grad_B
    
    def fuse(self, base_weights: torch.Tensor) -> torch.Tensor:
        """Merge LoRA into base weights."""
        return base_weights + self.B @ self.A
```

---

## 📈 Cumulative Fusing Alignment Decay

**Goal**: Model how the "alignment tax" decreases with each fusion cycle.

**Math**: $\text{tax}(n) = \text{tax}_0 \cdot e^{-\lambda n}$ where $n$ is fusion cycles.

```python
def alignment_tax(fusion_cycles: int, initial_tax: float = 2.91, decay: float = 1.5) -> float:
    """
    Model how RLHF alignment penalty decays with cumulative fusing.
    
    Observed in Sleeping LLM: 2.91 → 0.62 after 2 cycles.
    """
    return initial_tax * torch.exp(torch.tensor(-decay * fusion_cycles)).item()

# Example:
for n in range(5):
    print(f"Cycle {n}: tax = {alignment_tax(n):.3f}")
# Cycle 0: tax = 2.910
# Cycle 1: tax = 0.650
# Cycle 2: tax = 0.145
# Cycle 3: tax = 0.032
# Cycle 4: tax = 0.007
```

---

## 🧮 Putting It All Together: One Complete Cycle

```python
def dormio_sleep_cycle(
    weights: torch.Tensor,
    facts: list,
    lora: LoRAUpdate,
    null_space: torch.Tensor
) -> dict:
    """
    Execute one full sleep cycle with all math components.
    """
    report = {}
    
    # 1. Health check
    baseline_ppl = compute_perplexity(model, tokenizer, "The quick brown fox")
    report['baseline_ppl'] = baseline_ppl
    
    # 2. Audit and maintain degraded edits
    for fact in facts:
        recall = test_recall(model, tokenizer, fact.content, fact.expected)
        if recall < 0.8:
            # Refresh with null-space protection
            update = memit_edit(weights, fact.keys, fact.values)
            weights += project_to_null_space(update, null_space)
            report.setdefault('maintained', []).append(fact.id)
    
    # 3. Consolidate with LoRA
    for fact in facts:
        if fact.consolidation_score < 0.7:
            # One LoRA step
            lora.update(fact.grad_A, fact.grad_B, lr=0.01)
            fact.consolidation_score += 0.1
    
    # 4. Dissolve MEMIT edits for consolidated facts
    scheduler = DissolutionScheduler()
    for fact in facts:
        if fact.consolidation_score > 0.7:
            fact.memit_strength = scheduler.get_strength(fact.consolidation_score)
            if fact.memit_strength < 0.1:
                facts.remove(fact)
                report.setdefault('dissolved', []).append(fact.id)
    
    # 5. Cumulative fusing
    weights = lora.fuse(weights)
    lora = LoRAUpdate(weights.shape[1], weights.shape[0])  # Fresh LoRA
    
    # 6. Validate
    post_ppl = compute_perplexity(model, tokenizer, "The quick brown fox")
    report['post_ppl'] = post_ppl
    report['degradation'] = (post_ppl - baseline_ppl) / baseline_ppl
    
    return report
```

---

## 💎 Summary Table

| Component | Mathematical Tool | Key Formula |
|:---|:---|:---|
| MEMIT Edit | Least Squares | $\Delta W = (V - WK^T)(K^T)^+$ |
| Null-Space Protection | SVD | $P_{\text{null}} = V_{\text{null}} V_{\text{null}}^T$ |
| Health Check | Perplexity | $\exp(-\frac{1}{N}\sum \log P)$ |
| Recall Testing | Cosine Similarity | $\frac{a \cdot b}{\|a\|\|b\|}$ |
| Dissolution | Exponential Decay | $s(t) = e^{-\lambda t}$ |
| LoRA | Low-Rank Factorization | $W' = W + BA$ |
| Alignment Tax | Exponential Decay | $\text{tax}(n) = \text{tax}_0 e^{-\lambda n}$ |

These mathematical foundations are directly implemented in the Dormio micro-framework, providing a rigorous basis for persistent LLM memory. Want me to expand any specific component with more detailed derivations or optimization techniques?
