import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables to suppress progress bars and warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import name mapping utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.name_mapping import (
    translate_name, 
    normalize_god_name, 
    get_name_variants,
    GREEK_TO_ROMAN,
    ROMAN_TO_GREEK
)

# Import with progress bars disabled
import torch
# Allow configuring CPU threads for PyTorch via environment variable to speed up indexing
try:
    _threads_env = int(os.getenv("MYTH_THREADS", "0"))
    if _threads_env > 0:
        torch.set_num_threads(_threads_env)
    else:
        # Sensible default: use up to 8 threads if not specified
        torch.set_num_threads(8)
except Exception:
    torch.set_num_threads(8)

# Import sentence-transformers with progress bars disabled
import chromadb
from sentence_transformers import SentenceTransformer

# Disable progress bars in sentence-transformers
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Import agents
from agents.dense_retriever import DenseRetrieverAgent
from agents.sparse_retriever import SparseRetrieverAgent
from agents.reranker import RerankerAgent
from agents.entity_agent import EntityAgent
from agents.summarizer import SummarizerAgent
from agents.orchestrator import HybridOrchestrator

class RAGSystem:
    # Embedding models that require task prefixes
    PREFIXED_MODELS = {"nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5"}
    
    def __init__(
        self,
        lore_entities_dir: str = "lore_entities",
        lore_chunks_dir: str = "lore_chunks",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",  # nomic-ai/nomic-embed-text-v1.5 is better but slower to index
        collection_name: str = "myth_lore",
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 200,
        force_reindex: bool = False,
    ):
        """
        Initialize the RAG system with lore directories and model configuration.
        
        Args:
            lore_entities_dir: Directory containing entity metadata files
            lore_chunks_dir: Directory containing lore text chunks
            embedding_model_name: Name of the sentence transformer model to use
            collection_name: Name of the ChromaDB collection to use/create
        """
        logging.info("Initializing RAG System...")
        
        # Initialize instance variables
        self.lore_entities_dir = lore_entities_dir
        self.lore_chunks_dir = lore_chunks_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size_chars = max(300, chunk_size_chars)
        self.chunk_overlap_chars = max(0, min(chunk_overlap_chars, self.chunk_size_chars // 2))
        # Increment when chunking/indexing schema changes
        self._index_version = 2
        
        # Check if model requires task prefixes
        self._uses_task_prefix = any(m in embedding_model_name for m in self.PREFIXED_MODELS)
        
        # Initialize name mapping utilities first
        self._init_name_mapping()
        
        # Initialize the embedding model
        try:
            _device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Some models like nomic require trust_remote_code
            self.embedding_model = SentenceTransformer(
                embedding_model_name, 
                device=_device,
                trust_remote_code=True
            )
            logging.info(f"Loaded embedding model: {embedding_model_name} on {_device}")
            # Embedding batch size (larger on GPU)
            try:
                self._embed_batch_size = int(os.getenv("EMBED_BATCH", "0"))
            except Exception:
                self._embed_batch_size = 0
            if self._embed_batch_size <= 0:
                self._embed_batch_size = 64 if _device == 'cuda' else 32
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(path="chroma_db")
            logging.info("Initialized ChromaDB client")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Load lore data from cache or regenerate
        cached_lore = self._load_cached_lore()
        if cached_lore is not None:
            self.lore_chunks = cached_lore
        else:
            self.lore_chunks = self._load_lore()
            self._save_lore_cache(self.lore_chunks)
        
        self.collection = self._create_collection(force_reindex=force_reindex)
        
        # Initialize the agentic RAG pipeline
        self._init_agentic_rag()
        
        logging.info("RAG System initialization complete")
    
    def _init_name_mapping(self):
        """Initialize name mapping utilities and caches."""
        # Cache for name variants to avoid recomputation
        self._name_variants_cache = {}
        
        # Set of all known god names (Greek and Roman)
        self.all_god_names = set(GREEK_TO_ROMAN.keys()) | set(ROMAN_TO_GREEK.keys()) | \
                             {v.lower() for v in GREEK_TO_ROMAN.values()} | \
                             {v.lower() for v in ROMAN_TO_GREEK.values()}

    def _encode_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Encode documents for indexing, with task prefix if required.
        
        Args:
            texts: List of document texts to encode.
            **kwargs: Additional arguments passed to encode().
            
        Returns:
            List of embedding vectors.
        """
        if self._uses_task_prefix:
            texts = ["search_document: " + t for t in texts]
        return self.embedding_model.encode(texts, **kwargs).tolist()

    def _encode_query(self, query: str, **kwargs) -> List[float]:
        """Encode a query for retrieval, with task prefix if required.
        
        Args:
            query: Query text to encode.
            **kwargs: Additional arguments passed to encode().
            
        Returns:
            Embedding vector.
        """
        if self._uses_task_prefix:
            query = "search_query: " + query
        return self.embedding_model.encode(query, **kwargs).tolist()

    def _get_lore_cache_path(self) -> Path:
        """Get the path to the lore chunks cache file."""
        return Path("chroma_db") / "lore_cache.json"

    def _is_cache_fresh(self) -> bool:
        """Check if the lore cache is fresher than all source lore files.
        
        Returns:
            True if cache exists and is newer than all lore files, False otherwise.
        """
        cache_path = self._get_lore_cache_path()
        if not cache_path.exists():
            return False
        
        cache_mtime = cache_path.stat().st_mtime
        lore_dir = Path(self.lore_chunks_dir)
        
        if not lore_dir.exists():
            return False
        
        # Check if any lore file is newer than the cache
        for lore_file in lore_dir.glob("*.md"):
            if lore_file.stat().st_mtime > cache_mtime:
                logging.info(f"Cache invalidated: {lore_file.name} is newer than cache")
                return False
        
        return True

    def _load_cached_lore(self) -> Optional[List[Dict[str, Any]]]:
        """Load lore chunks from cache if available and fresh.
        
        Returns:
            Cached lore chunks list, or None if cache is invalid/missing.
        """
        if not self._is_cache_fresh():
            return None
        
        cache_path = self._get_lore_cache_path()
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logging.info(f"Loaded {len(chunks)} lore chunks from cache")
            return chunks
        except Exception as e:
            logging.warning(f"Failed to load lore cache: {e}")
            return None

    def _save_lore_cache(self, chunks: List[Dict[str, Any]]) -> None:
        """Save lore chunks to cache file."""
        cache_path = self._get_lore_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)
            logging.info(f"Saved {len(chunks)} lore chunks to cache")
        except Exception as e:
            logging.warning(f"Failed to save lore cache: {e}")

    def _init_agentic_rag(self):
        """
        Initialize the agentic RAG pipeline with all components.
        
        This sets up the full retrieval pipeline with enhanced name variant handling.
        Heavy agents (RerankerAgent, EntityAgent) are lazy-loaded on first query.
        """
        try:
            logging.info("Initializing DenseRetrieverAgent...")
            dense_agent = DenseRetrieverAgent(
                collection=self.collection,
                embedding_model=self.embedding_model,  # Reuse pre-loaded model
                use_task_prefix=self._uses_task_prefix,  # For nomic models
            )
            logging.info("DenseRetrieverAgent initialized.")
            
            # Initialize sparse agent with error handling
            sparse_agent = None
            try:
                logging.info("Initializing SparseRetrieverAgent...")
                sparse_agent = SparseRetrieverAgent(
                    lore_chunks=self.lore_chunks
                )
                logging.info("SparseRetrieverAgent initialized.")
            except Exception as e:
                logging.warning(f"Failed to initialize SparseRetrieverAgent: {e}. Continuing without it.")
                print(f"Warning: Failed to initialize SparseRetrieverAgent. Continuing without sparse retrieval.")
            
            # Create factory functions for lazy-loaded heavy agents
            # These will only be instantiated on first query, not at startup
            def create_reranker() -> RerankerAgent:
                return RerankerAgent(model_name="BAAI/bge-reranker-v2-m3")
            
            def create_entity_agent() -> EntityAgent:
                return EntityAgent(model_name="dbmdz/bert-large-cased-finetuned-conll03-english")

            logging.info("Initializing SummarizerAgent...")
            summarizer_agent = SummarizerAgent()
            logging.info("SummarizerAgent initialized.")
            
            # Create the orchestrator with lazy-loaded agents
            # Pass factory functions instead of instances for heavy models
            logging.info("Creating HybridOrchestrator (RerankerAgent and EntityAgent will be lazy-loaded)...")
            self.agentic_rag = HybridOrchestrator(
                dense_agent=dense_agent,
                sparse_agent=sparse_agent,
                reranker_agent=create_reranker,  # Factory function for lazy loading
                entity_agent=create_entity_agent,  # Factory function for lazy loading
                summarizer_agent=summarizer_agent
            )
            
            logging.info("Agentic RAG pipeline initialized successfully with name variant support.")
            print("Agentic RAG pipeline initialized (heavy agents will load on first query).")
            
        except Exception as e:
            error_msg = f"Error initializing agentic RAG pipeline: {e}"
            logging.error(error_msg)
            print(error_msg)
            raise

    def _load_lore(self) -> List[Dict[str, Any]]:
        """Load and chunk lore markdown files into smaller passages."""
        import yaml

        def _first_nonempty_line(text: str) -> str:
            for line in text.splitlines():
                clean = line.strip()
                if clean:
                    return clean
            return ""

        def _chunk_text(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
            """Split text into overlapping chunks by paragraph boundaries.
            
            Uses a for-loop to guarantee progress and avoid infinite loops.
            Tracks character offsets without expensive text.find() calls.
            """
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks: List[Tuple[str, int]] = []
            if not paragraphs:
                return chunks
            
            # Pre-compute paragraph start offsets to avoid O(n) find() calls
            para_offsets: List[int] = []
            search_start = 0
            for p in paragraphs:
                idx = text.find(p, search_start)
                para_offsets.append(idx if idx >= 0 else search_start)
                search_start = para_offsets[-1] + len(p)
            
            current_paras: List[str] = []
            current_len = 0
            chunk_start_idx = 0
            
            for i, para in enumerate(paragraphs):
                sep_len = 2 if current_paras else 0  # "\n\n" separator
                candidate_len = current_len + sep_len + len(para)
                
                # Always add if chunk is empty (guarantees progress even for huge paragraphs)
                if candidate_len <= size or not current_paras:
                    if not current_paras:
                        chunk_start_idx = para_offsets[i]
                    current_paras.append(para)
                    current_len = candidate_len
                else:
                    # Emit current chunk
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append((chunk_text, chunk_start_idx))
                    
                    # Start new chunk: include overlap from end of previous chunk
                    if overlap > 0 and chunk_text:
                        tail = chunk_text[-overlap:]
                        current_paras = [tail, para]
                        current_len = len(tail) + 2 + len(para)
                    else:
                        current_paras = [para]
                        current_len = len(para)
                    chunk_start_idx = para_offsets[i]
            
            # Emit final chunk
            if current_paras:
                chunk_text = "\n\n".join(current_paras).strip()
                if chunk_text:
                    chunks.append((chunk_text, chunk_start_idx))
            
            return chunks

        def _detect_gods(text: str) -> Tuple[str, List[str]]:
            found: List[str] = []
            lowered = text.lower()
            all_names = set(GREEK_TO_ROMAN.keys()) | set(ROMAN_TO_GREEK.keys()) | {
                v.lower() for v in GREEK_TO_ROMAN.values()
            } | {v.lower() for v in ROMAN_TO_GREEK.values()}
            for name in all_names:
                if re.search(rf"\b{name}\b", lowered):
                    normalized, _ = translate_name(name, to_roman=False)
                    found.append(normalized.lower())
            unique = sorted(set(found))
            primary = unique[0] if len(unique) == 1 else "unknown"
            return primary, unique

        lore_chunks: List[Dict[str, Any]] = []

        if not os.path.exists(self.lore_chunks_dir):
            logging.warning(f"Lore chunks directory not found: {self.lore_chunks_dir}")
            return lore_chunks

        for filename in sorted(os.listdir(self.lore_chunks_dir)):
            if not filename.endswith('.md'):
                continue
            filepath = os.path.join(self.lore_chunks_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw = f.read()

                metadata: Dict[str, Any] = {}
                content = raw
                if raw.startswith('---'):
                    try:
                        parts = raw.split('---', 2)
                        metadata_part = parts[1]
                        content = parts[2].strip()
                        metadata = yaml.safe_load(metadata_part) or {}
                        if 'god' in metadata and isinstance(metadata['god'], str):
                            normalized_god, _ = translate_name(metadata['god'], to_roman=False)
                            metadata['god'] = normalized_god
                            metadata['god_variants'] = get_name_variants(normalized_god)
                    except Exception as e:
                        logging.warning(f"Error parsing metadata in {filename}: {e}")

                title = metadata.get('title') or _first_nonempty_line(content) or os.path.splitext(filename)[0]
                chunks = _chunk_text(content, self.chunk_size_chars, self.chunk_overlap_chars)
                # Fallback: if file is short/no chunks, index the whole content
                if not chunks:
                    chunks = [(content.strip(), 0)] if content.strip() else []
                for order, (chunk_text, start_idx) in enumerate(chunks):
                    primary_god = metadata.get('god') if 'god' in metadata else None
                    god_variants = metadata.get('god_variants') if 'god_variants' in metadata else None
                    if not primary_god:
                        detected_primary, detected_all = _detect_gods(chunk_text)
                        primary_god = detected_primary
                        god_variants = detected_all
                    chunk_id = f"{os.path.splitext(filename)[0]}_c{order}"
                    lore_chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'title': title,
                            'source_file': filename,
                            'order': order,
                            'start_char': int(start_idx),
                        },
                        'god': (primary_god or 'unknown').lower(),
                        'god_variants': god_variants or [],
                        'source_file': filename,
                        'title': title,
                        'order': order,
                    })
            except Exception as e:
                logging.error(f"Error loading lore file {filename}: {e}")

        logging.info(f"Loaded {len(lore_chunks)} chunked lore passages")
        return lore_chunks

    def _create_collection(self, force_reindex: bool = False):
        """Create or rebuild the ChromaDB collection and index lore passages."""
        try:
            needs_rebuild = force_reindex
            try:
                existing = self.chroma_client.get_collection(name=self.collection_name)
                meta = existing.metadata or {}
                stored_version = int(meta.get("index_version", 0))
                if stored_version < self._index_version:
                    needs_rebuild = True
            except Exception:
                needs_rebuild = True

            if needs_rebuild:
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                except Exception:
                    pass
                collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 300,
                        "hnsw:search_ef": 128,
                        "hnsw:M": 16,
                        "index_version": self._index_version,
                    },
                )
                if self.lore_chunks:
                    logging.info("Indexing chunked lore passages...")
                    batch_size = 50
                    for i in range(0, len(self.lore_chunks), batch_size):
                        batch = self.lore_chunks[i : i + batch_size]
                        ids = [chunk["id"] for chunk in batch]
                        texts = [chunk["text"] for chunk in batch]
                        metadatas = [chunk.get("metadata", {}) for chunk in batch]
                        embeddings = self._encode_documents(
                            texts,
                            normalize_embeddings=True,
                            batch_size=getattr(self, "_embed_batch_size", 32),
                            convert_to_numpy=True,
                            show_progress_bar=True,
                        )
                        collection.upsert(
                            ids=ids,
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas,
                        )
                        logging.info(
                            f"Indexed batch {i // batch_size + 1}/{(len(self.lore_chunks) - 1) // batch_size + 1}"
                        )
                    logging.info(f"Finished indexing {len(self.lore_chunks)} passages")
            else:
                collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 300,
                        "hnsw:search_ef": 128,
                        "hnsw:M": 16,
                        "index_version": self._index_version,
                    },
                )
                if collection.count() == 0 and self.lore_chunks:
                    logging.info("Indexing chunked lore passages (collection empty)...")
                    batch_size = 50
                    for i in range(0, len(self.lore_chunks), batch_size):
                        batch = self.lore_chunks[i : i + batch_size]
                        ids = [chunk["id"] for chunk in batch]
                        texts = [chunk["text"] for chunk in batch]
                        metadatas = [chunk.get("metadata", {}) for chunk in batch]
                        embeddings = self._encode_documents(
                            texts,
                            normalize_embeddings=True,
                            batch_size=getattr(self, "_embed_batch_size", 32),
                            convert_to_numpy=True,
                            show_progress_bar=True,
                        )
                        collection.upsert(
                            ids=ids,
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas,
                        )
                        logging.info(
                            f"Indexed batch {i // batch_size + 1}/{(len(self.lore_chunks) - 1) // batch_size + 1}"
                        )
            return collection
        except Exception as e:
            logging.error(f"Failed to create/load collection: {e}")
            raise

    def _get_god_filter(self, god: Optional[str]) -> Optional[Dict]:
        """Generate a filter for the given god, handling both Greek and Roman names."""
        if not god:
            return None
            
        # Get all name variants for the god
        name_variants = get_name_variants(god)
        
        # Create a filter that matches any of the name variants
        return {
            "$or": [
                {"god": {"$eq": variant}} for variant in name_variants
            ]
        }

    def retrieve_lore(self, query_text, k=1, god=None):
        if self.collection.count() == 0:
            logging.debug("Collection is empty.")
            return []
            
        logging.debug(f"Retrieving legacy lore for query: '{query_text[:50]}...' (god: {god}, k: {k})")
        
        # Preprocess query to handle Greek/Roman name variants
        query_for_embedding = self._preprocess_query(query_text)
        query_embedding = self._encode_query(
            query_for_embedding, normalize_embeddings=True
        )
        
        # Get filter for god name (handling variants)
        where_filter = self._get_god_filter(god)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
            where=where_filter
        )
        
        retrieved_docs = []
        if results and results.get('documents') and results.get('metadatas'):
            for i, doc_text in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "text": doc_text,
                    "metadata": results['metadatas'][0][i]
                })
        logging.debug(f"Legacy retrieved {len(retrieved_docs)} documents for query: '{query_text[:50]}...' ")
        return retrieved_docs

    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess the query to handle Greek/Roman name variants."""
        # This is a simple implementation - you might want to enhance it with more sophisticated
        # text processing or NLP techniques
        words = query_text.split()
        processed_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            
            # Check if it's a Greek or Roman god name
            if clean_word in GREEK_TO_ROMAN:
                # Add both Greek and Roman names to the query
                roman_name = GREEK_TO_ROMAN[clean_word]
                processed_words.extend([word, roman_name])
            elif clean_word in ROMAN_TO_GREEK:
                # Add both Roman and Greek names to the query
                greek_name = ROMAN_TO_GREEK[clean_word]
                processed_words.extend([word, greek_name])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)

    def retrieve_lore_with_agents(self, query_text: str, k: int = 3, god_context: Optional[str] = None) -> dict:
        """
        Retrieve lore using the agentic RAG pipeline with name variant handling.
        
        Args:
            query_text: The user's query text
            k: Number of results to return
            god_context: The name of the current god to focus the search on.
            
        Returns:
            Dictionary containing:
            - summary: A concise summary of the results
            - documents: List of relevant lore chunks with scores and metadata
        """
        if not hasattr(self, 'agentic_rag') or not self.agentic_rag:
            return {
                'summary': 'Error: Agentic RAG pipeline not initialized',
                'documents': []
            }
        
        try:
            # The orchestrator now handles query expansion. We pass the raw query.
            # The god_context is passed separately for precise filtering.
            result = self.agentic_rag.retrieve(query_text, k=k, god_context=god_context)
            
            # Post-process the result to normalize god name variants in the output
            if 'documents' in result and result['documents']:
                for doc in result['documents']:
                    if 'text' in doc:
                        doc['text'] = self._normalize_name_variants(doc['text'])
            
            # Filter out documents that start with 'http' or '##'
            result['documents'] = [doc for doc in result['documents'] if not doc['text'].startswith(('http', '##'))]
            
            return result
            
        except Exception as e:
            logging.error(f"Error in retrieve_lore_with_agents: {e}")
            return {
                'summary': f'Error retrieving results: {str(e)}',
                'documents': []
            }
    
    def _normalize_name_variants(self, text: str) -> str:
        """
        Normalize name variants in text to Greek names.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Text with name variants normalized to Greek names
        """
        if not text:
            return text
            
        # Simple word-based replacement
        # This is a fallback - the summarizer should handle most cases
        for roman, greek in ROMAN_TO_GREEK.items():
            text = re.sub(
                r'\b' + re.escape(roman) + r'\b', 
                greek,
                text,
                flags=re.IGNORECASE
            )
            
        return text

if __name__ == '__main__':
    # Example usage / test
    print("Testing RAGSystem standalone...")
    # Create dummy lore files if they don't exist for testing
    if not os.path.exists("lore_entities"):
        os.makedirs("lore_entities")
    if not os.path.exists("lore_chunks"):
        os.makedirs("lore_chunks")

    dummy_entity_path = "lore_entities/001_TEST.json"
    dummy_chunk_path = "lore_chunks/001_TEST.md"

    if not os.path.exists(dummy_entity_path):
        with open(dummy_entity_path, 'w') as f:
            json.dump({"id": "001_TEST", "title": "Test Lore", "related_chunks": ["001_TEST.md"], "summary": "This is a test summary."}, f)
    
    if not os.path.exists(dummy_chunk_path):
        with open(dummy_chunk_path, 'w') as f:
            f.write("This is the full text of the test lore about ancient gods and heroes.")

    rag = RAGSystem()
    test_query = "Tell me about ancient gods"
    lore = rag.retrieve_lore(test_query)
    if lore:
        print(f"\nQuery: {test_query}")
        for item in lore:
            print(f"  Title: {item['metadata'].get('title', 'N/A')}")
            print(f"  Text: {item['text'][:100]}...")
    else:
        print("No lore retrieved for test query.")

    def test_agentic_rag() -> None:
        """Test the agentic HybridOrchestrator RAG pipeline for structure and relevance.

        Instantiates RAGSystem, queries the orchestrator, and asserts that the output
        contains a summary and docs mentioning Apollo or Daphne.
        """
        print("\n[TEST] Agentic HybridOrchestrator RAG")
        rag = RAGSystem()
        query = "Tell me about Apollo and Daphne"
        result = rag.retrieve_lore_with_agents(query, k=3)
        assert isinstance(result, dict), "Result should be a dict"
        assert "summary" in result and isinstance(result["summary"], str), "Missing or invalid summary"
        # The key should always be 'documents' based on orchestrator.py
        assert "documents" in result and isinstance(result["documents"], list), "Missing or invalid 'documents' in result"
        assert len(result["documents"]) > 0, "No documents returned"
        print("Summary:", result["summary"])
        print("Top doc snippet:", result["documents"][0]["text"][:120])
        # Check that the summary or top doc mentions Apollo or Daphne
        assert "Apollo" in result["summary"] or "Daphne" in result["summary"] or \
               "Apollo" in result["documents"][0]["text"] or "Daphne" in result["documents"][0]["text"], \
               "Neither Apollo nor Daphne found in summary or top doc"
        print("[PASS] Agentic RAG returns relevant results.")

    test_agentic_rag()
