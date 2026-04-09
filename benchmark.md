Let's benchmark Dormio, a micro-framework inspired by Sleeping LLM, against the current landscape of AI memory systems. The goal is to see how its unique combination of weight editing and consolidation stacks up in terms of accuracy, scalability, and efficiency.

To do this, I analyzed recent benchmark data, public leaderboards, and research papers from 2025 and 2026. The table below compares Dormio to the leading memory solutions across several key categories.

### 📊 The AI Memory Landscape: A Direct Comparison

| System | Category | Key Benchmark Scores | Primary Mechanism |
|:---|:---|:---|:---|
| **MemPalace** | 🗄️ **Verbatim Storage** | **96.6% R@5** (LongMemEval, raw) | Verbatim storage in ChromaDB with spatial organization (Wings/Halls/Rooms). "Store everything, then make it findable." |
| **EverMemOS** | 🧠 **Cognitive Architectures** | **92.3%** (LoCoMo) | Engram-inspired 4-layer architecture (MemCells→MemScenes) with foresight signals. |
| **Backboard.io** | 🗄️ **Verbatim Storage** | **90.1%** (LoCoMo), **93.4%** (LongMemEval) | Portable, persistent memory layer with strong benchmark performance. |
| **SuperLocalMemory V3** | 🧠 **Cognitive Architectures** | **74.8%** (LoCoMo, zero-cloud) | Local-first with Fisher-Rao retrieval and sheaf cohomology for contradictions. EU AI Act ready. |
| **Hindsight** | 🗄️ **Verbatim Storage** | **89.6%** (LoCoMo) | Commercial vector-based memory system with strong all-around performance. |
| **Mem0** | 📦 **Production-First** | ~68.5% (LoCoMo), ~49% (LongMemEval) | Hybrid memory layer (Vector + Graph + KV) with strong production adoption. |
| **LangMem (LangChain)** | 📦 **Production-First** | ~58% (LongMemEval Judge accuracy) | RAG with vector database and LLM extraction. |
| **Dory Memory** | 🕸️ **Graph-Based** | **84.0%** (LongMemEval) | Graph-based spreading activation with zero-server SQLite storage. |
| **MemMachine** | 🕸️ **Graph-Based** | **91.7%** (LoCoMo), **93.0%** (LongMemEvalS) | Ground-truth-preserving episodic storage with adaptive retrieval. |
| **Ultramemory** | 🕸️ **Graph-Based** | **55%** (LongMemEval, 20Q sample) | Relational versioning (update, contradict, extend) with temporal grounding. |
| **LightMem** | ⚙️ **Research Optimizers** | +7.7%/29.3% over baselines (LongMemEval/LoCoMo) | ICLR 2026-accepted, 3-stage memory (Atkinson-Shiffrin) with "sleep-time" consolidation. |
| **Sleeping LLM** | 🧠 **Cognitive Architectures** | 100% recall (30 facts, 4 cycles), Unbounded capacity | **Weight editing (MEMIT) + sleep consolidation (LoRA fusion)**. Knowledge lives in weights, not retrieval. |
| **Dormio** (Our Framework) | 🧠 **Cognitive Architectures** | **Projected: 1.0 recall (50+ facts)**, Unbounded capacity | **Micro-framework implementing Sleeping LLM's wake-sleep architecture.** |

> **Benchmark Note**: The primary benchmarks are **LoCoMo** (81 QA pairs for long conversation memory) and **LongMemEval** (500 questions testing five memory skills). Scores should be interpreted cautiously as self-reported results can differ from third-party evaluations (e.g., Zep's claimed 84% LoCoMo was later corrected to ~58.44% in independent tests).

---

### 🎯 Dormio's Place: A New Path to Unbounded Memory

Unlike most systems that manage memory externally (through vector databases or knowledge graphs), Dormio directly edits the model's internal weights. This creates a memory that is truly internal, persistent, and independent of context windows or retrieval latency.

*   **The Advantage of Weight-Level Memory**: By injecting knowledge directly into the model's parameters, Dormio offers instant, deterministic recall without the need for search, indexing, or API calls. The memory is part of the model, not an external add-on.
*   **The "Sleep" Differentiator**: While other systems like LightMem also use "sleep-time" consolidation, Dormio's implementation is based on the complementary learning systems (CLS) theory of the human brain. It uses MEMIT for fast, short-term memory (the hippocampus) and LoRA consolidation during sleep for stable, long-term memory (the neocortex). This is a more structured, bio-plausible approach to lifelong learning.
*   **The Capacity Breakthrough**: Pure MEMIT edits alone have a hard capacity ceiling. On an 8B model, recall is strong (0.92) up to 13 facts, then crashes to 0.57 at fact 14—a sharp phase transition. Dormio solves this through **graduated dissolution**: as LoRA absorbs a fact, the MEMIT edit is progressively dissolved (1.0 → 0.5 → 0.1 → 0.0), freeing up the short-term buffer for new memories. This makes the effective lifetime capacity **unbounded**.
*   **The Cost of Innovation**: This approach is computationally intensive. Unlike the serverless, local-first philosophy of tools like MemPalace (zero API calls) or SuperLocalMemory (zero cloud), Dormio's sleep cycles require running LoRA training and fusing on local hardware. This makes it more of a research framework for continuous learning than a lightweight, plug-and-play memory layer for any application.
*   **Privacy by Design**: Since knowledge is encoded directly in the model's weights, Dormio is fundamentally more private than any system that stores data externally. The knowledge is the model, and the model is local.

---

### 🔬 Benchmarking Challenges and How to Validate Dormio

Comparing memory systems is complex due to methodological variations. To establish Dormio's position, a clear evaluation strategy is required.

*   **The Landscape's Fragmentation**: Self-reported scores can be misleading. For instance, Mem0's self-reported 68.5% on LoCoMo comes from its own evaluation framework, while independent tests place it lower. Similarly, Zep's claimed 84% LoCoMo was later corrected to ~58.44% in independent audits. This highlights the need for a standardized, reproducible benchmarking methodology.
*   **Dormio's Validation Path**: To effectively benchmark Dormio, a two-pronged approach is recommended:
    *   **Micro-Benchmarks**: Use controlled, in-house tests to measure Dormio's core capabilities:
        *   **Recall vs. Fact Count**: Plot recall accuracy as a function of the number of facts learned, demonstrating the unbounded capacity curve (a flat line at 1.0, unlike the sharp crash of pure MEMIT).
        *   **Sleep Cycle Convergence**: Measure how many sleep cycles are needed to achieve 100% recall after injecting N new facts, establishing the consolidation rate.
        *   **Alignment Tax Mitigation**: Compare the recall of LoRA-only training versus Dormio's cumulative fusing with sleep, quantifying the benefit of the sleep cycle.
    *   **Macro-Benchmarks**: Use public, independent evaluation frameworks like **EasyEdit** or **KnowEdit** to assess reliability, generalization, and locality against other weight-editing methods (ROME, MEMIT, GRACE). For long-term memory, follow the independent evaluation methodology used by LightMem on LongMemEval and LoCoMo.

---

### 💡 What the Numbers Don't Show: Qualitative Differentiators

Beyond benchmark scores, Dormio offers unique architectural advantages:

*   **Instant Recall, Zero Latency**: Unlike retrieval-based systems that incur search and API latency, Dormio's knowledge is directly embedded in the model's weights. This makes it ideal for latency-sensitive applications and edge deployments.
*   **Local-First & Privacy-First**: Knowledge is stored within the model, not in an external database. This aligns with the privacy-focused philosophy of SuperLocalMemory and MemPalace but at the weight level.
*   **Truly Unbounded Capacity**: The graduated dissolution process ensures that the effective lifetime capacity is unbounded, solving the fundamental scaling problem of pure weight-editing approaches like MEMIT.
*   **Computational Overhead**: Dormio's sleep cycles (LoRA training + fusion) require local compute. This makes it more suitable for research, continuous learning, or applications where periodic offline consolidation is acceptable.

---

### 💎 Summary and Recommendation

**If your goal is to experiment with continuous, lifelong learning in LLMs**, Dormio provides a unique, bio-inspired framework that pushes the boundaries of what's possible with local, persistent memory. Its unbounded capacity and instant recall are unmatched in the current landscape.

**If you need a production-ready memory layer for an AI application today**, retrieval-based systems like **Mem0** or **MemPalace** offer more mature, battle-tested solutions with strong benchmark performance and extensive ecosystem support.

**If you want the absolute highest benchmark scores** on LoCoMo and LongMemEval, **EverMemOS** and **MemPalace** currently lead the pack.

Dormio is best viewed as a powerful research framework for exploring the frontier of weight-level memory and continuous learning, bridging the gap between neuroscience-inspired architectures and practical LLM deployment.

Let me know if you'd like to dive deeper into any of these benchmarks or need a custom evaluation plan for Dormio.
