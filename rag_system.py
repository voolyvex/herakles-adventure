import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
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
torch.set_num_threads(1)  # Limit CPU threads to prevent excessive resource usage

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
    def __init__(self, lore_entities_dir: str = "lore_entities", 
                 lore_chunks_dir: str = "lore_chunks",
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 collection_name: str = "myth_lore"):
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
        
        # Initialize name mapping utilities first
        self._init_name_mapping()
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logging.info(f"Loaded embedding model: {embedding_model_name}")
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
        
        # Load lore data and create collection
        self.lore_chunks = self._load_lore()
        self.collection = self._create_collection()
        
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

    def _init_agentic_rag(self):
        """
        Initialize the agentic RAG pipeline with all components.
        
        This sets up the full retrieval pipeline with enhanced name variant handling.
        """
        try:
            logging.info("Initializing DenseRetrieverAgent...")
            dense_agent = DenseRetrieverAgent(
                embedding_model_name=self.embedding_model_name,
                collection=self.collection
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
            
            logging.info("Initializing RerankerAgent...")
            reranker_agent = RerankerAgent(
                model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
            )
            logging.info("RerankerAgent initialized.")
            
            logging.info("Initializing EntityAgent...")
            entity_agent = EntityAgent(
                model_name='dbmdz/bert-large-cased-finetuned-conll03-english'
            )
            logging.info("EntityAgent initialized.")

            logging.info("Initializing SummarizerAgent...")
            summarizer_agent = SummarizerAgent()
            logging.info("SummarizerAgent initialized.")
            
            # Create the orchestrator with all agents
            logging.info("Creating HybridOrchestrator...")
            self.agentic_rag = HybridOrchestrator(
                dense_agent=dense_agent,
                sparse_agent=sparse_agent,
                reranker_agent=reranker_agent,
                entity_agent=entity_agent,
                summarizer_agent=summarizer_agent
            )
            
            logging.info("Agentic RAG pipeline initialized successfully with name variant support.")
            print("Agentic RAG pipeline initialized successfully with name variant support.")
            
        except Exception as e:
            error_msg = f"Error initializing agentic RAG pipeline: {e}"
            logging.error(error_msg)
            print(error_msg)
            raise

    def _load_lore(self) -> List[Dict[str, Any]]:
        """
        Load all lore data from the lore chunks directory with metadata handling.
        
        Returns:
            List of lore chunks with metadata and normalized god names
        """
        import yaml
        
        lore_chunks = []
        
        # Ensure the directory exists
        if not os.path.exists(self.lore_chunks_dir):
            logging.warning(f"Lore chunks directory not found: {self.lore_chunks_dir}")
            return lore_chunks
        
        # Get all markdown files in the lore directory
        for filename in os.listdir(self.lore_chunks_dir):
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(self.lore_chunks_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract metadata (if any) and content
                metadata = {}
                if content.startswith('---'):
                    try:
                        metadata_part = content.split('---', 2)[1]
                        content = content.split('---', 2)[2].strip()
                        
                        # Parse YAML metadata
                        metadata = yaml.safe_load(metadata_part) or {}
                        
                        # Normalize god names in metadata
                        if 'god' in metadata:
                            normalized_god, _ = translate_name(metadata['god'], to_roman=False)
                            metadata['god'] = normalized_god
                            metadata['god_variants'] = get_name_variants(normalized_god)
                            
                    except Exception as e:
                        logging.warning(f"Error parsing metadata in {filename}: {e}")
                
                # Create a unique ID for the chunk
                chunk_id = f"{os.path.splitext(filename)[0]}_{len(lore_chunks)}"
                
                # Add to chunks
                lore_chunks.append({
                    'id': chunk_id,
                    'text': content,
                    'metadata': metadata,
                    'god': metadata.get('god', 'unknown').lower(),
                    'source_file': filename
                })
                
            except Exception as e:
                logging.error(f"Error loading lore file {filename}: {e}")
        
        logging.info(f"Loaded {len(lore_chunks)} lore chunks")
        return lore_chunks

    def _create_collection(self):
        """
        Create or get the ChromaDB collection and index the lore chunks.
        
        Returns:
            The ChromaDB collection object
        """
        try:
            # Create or get the collection with HNSW indexing
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:search_ef": 50,
                    "hnsw:M": 16
                }
            )
            
            # Check if collection is empty and needs indexing
            if collection.count() == 0 and self.lore_chunks:
                logging.info("Indexing lore chunks...")
                
                # Process in batches to avoid memory issues
                batch_size = 50
                for i in range(0, len(self.lore_chunks), batch_size):
                    batch = self.lore_chunks[i:i + batch_size]
                    
                    # Extract batch data
                    ids = [chunk['id'] for chunk in batch]
                    texts = [chunk['text'] for chunk in batch]
                    metadatas = [chunk['metadata'] for chunk in batch]
                    
                    # Generate embeddings
                    embeddings = self.embedding_model.encode(texts).tolist()
                    
                    # Add to collection
                    collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                    
                    logging.info(f"Indexed batch {i//batch_size + 1}/{(len(self.lore_chunks)-1)//batch_size + 1}")
                
                logging.info(f"Finished indexing {len(self.lore_chunks)} lore chunks")
            
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
        query_embedding = self.embedding_model.encode(query_for_embedding).tolist()
        
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
