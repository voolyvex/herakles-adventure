# Model Upgrade Recommendations for Myth-RPG RAG System

This document outlines modern AI models that could improve the performance, speed, and quality of our RAG system based on December 2024/2025 research.

---

## Current Stack vs. Recommended Upgrades

| Component | Current Model | Recommended Upgrade | Benefit |
|-----------|---------------|---------------------|---------|
| **Embeddings** | BAAI/bge-small-en-v1.5 (33M) | nomic-embed-text-v1.5 (137M) | Better retrieval quality, Matryoshka support |
| **Reranker** | BAAI/bge-reranker-v2-m3 (~560MB) | FlashRank ms-marco-MiniLM-L-12-v2 (~34MB) | **10x faster**, CPU-optimized, ONNX |
| **NER** | dbmdz/bert-large-cased (340M) | SpaCy en_core_web_sm (12MB) | **30x smaller**, faster |
| **LLM** | mistral:latest (7B) | qwen2.5:3b or phi3:mini (3B) | Faster inference, good quality |

---

## 1. Embedding Models

### Top Recommendation: **nomic-embed-text-v1.5**

```python
# Installation
pip install sentence-transformers

# Usage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# For documents (indexing)
doc_embeddings = model.encode(["search_document: " + text for text in documents])

# For queries (retrieval)  
query_embedding = model.encode(["search_query: " + query])
```

**Why nomic-embed-text-v1.5?**
- **Matryoshka Representation Learning**: Can reduce dimensions (768 → 256) with minimal quality loss
- **Long context**: Supports up to 8192 tokens (vs 512 for BGE)
- **Task prefixes**: Optimized for search, clustering, classification
- **ONNX support**: Faster CPU inference
- **Open source**: Fully open weights and training code

### Alternative: **Jina Embeddings v3**
- Multi-lingual (100+ languages)
- Similar quality to OpenAI embeddings
- Good for mythological terms in multiple languages

### Integration Change for RAGSystem:

```python
# In rag_system.py, change:
embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"

# Update encoding to use task prefixes:
def _encode_for_indexing(self, texts: List[str]) -> np.ndarray:
    prefixed = ["search_document: " + t for t in texts]
    return self.embedding_model.encode(prefixed, normalize_embeddings=True)

def _encode_query(self, query: str) -> np.ndarray:
    return self.embedding_model.encode(
        ["search_query: " + query], 
        normalize_embeddings=True
    )
```

---

## 2. Reranker Models

### Top Recommendation: **FlashRank** (ONNX-optimized)

FlashRank is **10x faster** than transformer-based rerankers and runs purely on CPU without PyTorch.

```python
# Installation
pip install flashrank

# Usage
from flashrank import Ranker, RerankRequest

# Nano model (~4MB) - blazing fast
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", max_length=256)

# Or better quality (~34MB) - still very fast
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=256)

# Rerank
passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(candidates)]
request = RerankRequest(query=query, passages=passages)
results = ranker.rerank(request)
```

**Why FlashRank?**
- **No PyTorch/Transformers needed**: Uses ONNX runtime
- **4MB to 34MB models**: vs 560MB for bge-reranker
- **CPU-optimized**: No GPU required
- **Competitive quality**: Near state-of-the-art on MS MARCO

### Integration Change for agents/reranker.py:

```python
from flashrank import Ranker, RerankRequest

class RerankerAgent:
    """Agent for reranking using FlashRank (ONNX-optimized)."""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", max_length: int = 256):
        self.ranker = Ranker(model_name=model_name, max_length=max_length)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        passages = [{"id": str(i), "text": c["text"]} for i, c in enumerate(candidates)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        
        # Map back to original candidates with scores
        reranked = []
        for r in results[:top_k]:
            idx = int(r["id"])
            candidates[idx]["rerank_score"] = r["score"]
            reranked.append(candidates[idx])
        return reranked
```

---

## 3. NER (Named Entity Recognition)

### Top Recommendation: **SpaCy** (lightweight)

The current BERT-large NER model (340MB) is overkill for extracting god names. SpaCy's small model is 30x smaller and faster.

```python
# Installation
pip install spacy
python -m spacy download en_core_web_sm

# Usage
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Zeus and Apollo fought against the Titans")
entities = [(ent.text, ent.label_) for ent in doc.ents]
# [('Zeus', 'PERSON'), ('Apollo', 'PERSON'), ('Titans', 'ORG')]
```

**Why SpaCy?**
- **12MB vs 340MB**: 30x smaller
- **Faster inference**: No transformer overhead
- **Good enough for god names**: Recognizes proper nouns well
- **Custom entity ruler**: Can add mythological entities

### Custom Entity Ruler for Gods:

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")

# Add custom patterns for Greek/Roman gods
patterns = [
    {"label": "GOD", "pattern": "Zeus"},
    {"label": "GOD", "pattern": "Jupiter"},
    {"label": "GOD", "pattern": "Apollo"},
    {"label": "GOD", "pattern": "Athena"},
    {"label": "GOD", "pattern": "Minerva"},
    # ... add all gods from GREEK_TO_ROMAN mapping
]

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patterns)
```

---

## 4. Local LLM (via Ollama)

### Current: mistral:latest (7B, ~4GB)
### Recommended: **qwen2.5:3b** or **phi3:mini** (3B, ~2GB)

For CPU-heavy inference, smaller models provide better response times:

```bash
# Pull recommended models
ollama pull qwen2.5:3b
ollama pull phi3:mini

# Or for better quality with more RAM:
ollama pull qwen2.5:7b
```

**Performance Comparison (CPU):**

| Model | Size | Tokens/sec (CPU) | Quality |
|-------|------|------------------|---------|
| mistral:7b | 4GB | ~15-20 | Good |
| qwen2.5:7b | 4GB | ~15-20 | Better |
| qwen2.5:3b | 2GB | ~30-40 | Good |
| phi3:mini | 2GB | ~35-45 | Good |

### Integration in god_chat.py:

```python
# Change default ollama model
ollama_model: str = "qwen2.5:3b"  # Faster inference

# Or for mythological knowledge:
ollama_model: str = "qwen2.5:7b"  # Better knowledge
```

---

## 5. Quick Implementation Checklist

### Phase 1: Immediate (High Impact, Low Effort)

- [ ] **Replace Reranker with FlashRank** (~30 min)
  ```bash
  pip install flashrank
  ```
  - Reduces reranking time from ~2s to ~0.1s
  - Reduces memory by ~500MB

### Phase 2: Short-term (Medium Impact, Medium Effort)

- [ ] **Upgrade Embeddings to nomic-embed-text-v1.5** (~1 hour)
  - Better retrieval quality
  - Requires re-indexing ChromaDB
  - Add task prefixes to queries/documents

- [ ] **Replace NER with SpaCy** (~1 hour)
  - Remove EntityAgent or make it optional
  - Use SpaCy with custom entity ruler for gods

### Phase 3: Optional (Quality Improvements)

- [ ] **Test smaller Ollama models** (~30 min)
  - Try qwen2.5:3b for faster responses
  - Benchmark quality vs speed tradeoff

---

## 6. Updated pyproject.toml Dependencies

```toml
[project]
dependencies = [
    # Core
    "torch>=2.0.0",
    "transformers>=4.30.0",
    
    # Embeddings (choose one)
    "sentence-transformers>=2.2.0",  # For nomic-embed or BGE
    
    # Reranking (FlashRank - recommended)
    "flashrank>=0.2.0",
    
    # NER (SpaCy - recommended)
    "spacy>=3.5.0",
    
    # Vector DB
    "chromadb>=0.4.0",
    
    # Sparse retrieval
    "rank-bm25>=0.2.0",
    "nltk>=3.8.0",
    
    # Existing deps
    "pyyaml>=6.0",
    "requests>=2.28.0",
]
```

---

## 7. Expected Performance Improvements

| Metric | Before | After (All Changes) |
|--------|--------|---------------------|
| **Startup time** | Infinite (bug) → 5s | ~3s |
| **Rerank latency** | ~2s | ~0.1s |
| **Memory usage** | ~2.5GB | ~1.5GB |
| **First query latency** | ~8s (lazy load) | ~2s |
| **Retrieval quality** | Good | Better (nomic) |

---

## References

1. [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
2. [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
3. [Best Embedding Models for RAG (ZenML)](https://www.zenml.io/blog/best-embedding-models-for-rag)
4. [Mastering RAG: How to Select a Reranking Model](https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model)
5. [rerankers Library (Answer.AI)](https://github.com/AnswerDotAI/rerankers)

