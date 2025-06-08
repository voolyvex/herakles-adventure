import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables to suppress progress bars and warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
from agents.summarizer import SummarizerAgent
from agents.entity_agent import EntityAgent
from agents.orchestrator import HybridOrchestrator

class RAGSystem:
    def __init__(self, lore_entities_dir="lore_entities", lore_chunks_dir="lore_chunks", 
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', collection_name="myth_lore"):
        logging.info("Initializing RAG System...")
        self.lore_entities_dir = lore_entities_dir
        self.lore_chunks_dir = lore_chunks_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logging.info(f"Embedding model '{embedding_model_name}' loaded.")
        
        # Use persistent ChromaDB client with HNSW tuning
        # This will persist all embeddings in the 'chroma_db' folder
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "ip",
                "hnsw:construction_ef": 128,  # construction quality
                "hnsw:search_ef": 32  # lower for speed, raise for recall
            }
        )
        logging.info(f"ChromaDB collection '{collection_name}' ready (persistent mode).")
        
        self._load_and_index_lore()

        # === Hybrid Agent Team Setup ===
        # Load all lore chunks for agent-based retrieval
        logging.info("Preparing agent-based hybrid RAG...")
        self.lore_chunks = self._load_lore_data()
        # Instantiate agents
        self.dense_agent = DenseRetrieverAgent(embedding_model_name, self.lore_chunks)
        self.sparse_agent = SparseRetrieverAgent(self.lore_chunks)
        self.reranker_agent = RerankerAgent('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.summarizer_agent = SummarizerAgent('facebook/bart-large-cnn')
        self.entity_agent = EntityAgent('dbmdz/bert-large-cased-finetuned-conll03-english')
        self.orchestrator = HybridOrchestrator(
            dense_agent=self.dense_agent,
            sparse_agent=self.sparse_agent,
            reranker_agent=self.reranker_agent,
            summarizer_agent=self.summarizer_agent,
            entity_agent=self.entity_agent
        )

    def _chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks for finer-grained embedding."""
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            segment = words[i:i+chunk_size]
            chunks.append(" ".join(segment))
        return chunks

    def _load_lore_data(self):
        lore_data = []
        if not os.path.exists(self.lore_entities_dir):
            logging.warning(f"Lore entities directory not found: {self.lore_entities_dir}")
            return lore_data
        if not os.path.exists(self.lore_chunks_dir):
            logging.warning(f"Lore chunks directory not found: {self.lore_chunks_dir}")
            return lore_data

        for entity_fname in os.listdir(self.lore_entities_dir):
            if entity_fname.endswith(".json"):
                entity_path = os.path.join(self.lore_entities_dir, entity_fname)
                try:
                    with open(entity_path, 'r', encoding='utf-8') as f:
                        entity_meta = json.load(f)
                    
                    for chunk_fname_md in entity_meta.get("related_chunks", []):
                        chunk_path_md = os.path.join(self.lore_chunks_dir, chunk_fname_md)
                        if os.path.exists(chunk_path_md):
                            with open(chunk_path_md, 'r', encoding='utf-8') as cf:
                                chunk_text = cf.read()
                            
                            # Prepare metadata, ensuring all values are Chroma-compatible (str, int, float, bool)
                            # Attempt to infer the god from the entity file name or metadata
                            god_name = entity_meta.get("title", "").split()[0] if entity_meta.get("title") else entity_fname.split("_")[0]
                            metadata = {
                                "title": str(entity_meta.get("title", "N/A")),
                                "source_entity_id": str(entity_meta.get("id", "N/A")),
                                "chunk_file": str(chunk_fname_md),
                                "god": god_name
                                # Add other relevant metadata from entity_meta if needed, ensuring type compatibility
                            }
                            if entity_meta.get("summary"):
                                metadata["summary"] = str(entity_meta.get("summary"))

                            # Chunk the text into overlapping segments
                            for idx, segment in enumerate(self._chunk_text(chunk_text)):
                                meta_copy = metadata.copy()
                                meta_copy["chunk_part"] = idx
                                lore_data.append({
                                    "id": f"{uuid.uuid4()}_{idx}",
                                    "text": segment,
                                    "metadata": meta_copy
                                })
                        else:
                            print(f"Warning: Markdown file {chunk_fname_md} not found in {self.lore_chunks_dir}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {entity_path}")
                except Exception as e:
                    print(f"Error processing entity file {entity_path}: {e}")
        logging.info(f"Loaded {len(lore_data)} lore documents for agent processing.")
        return lore_data

    def _load_and_index_lore(self):
        logging.info("Loading and indexing lore...")
        # Check if collection is already populated (simple check, might need refinement for persistence)
        if self.collection.count() > 0:
            logging.info("Lore already indexed.")
            return

        lore_documents = self._load_lore_data()
        if not lore_documents:
            logging.info("No lore documents to index.")
            return

        batch_size = 100 # Process in batches if many documents
        for i in range(0, len(lore_documents), batch_size):
            batch = lore_documents[i:i+batch_size]
            ids_batch = [doc['id'] for doc in batch]
            texts_batch = [doc['text'] for doc in batch]
            metadata_batch = [doc['metadata'] for doc in batch]
            
            logging.info(f"Generating embeddings for batch {i//batch_size + 1}...")
            embeddings_batch = self.embedding_model.encode(texts_batch).tolist()
            
            try:
                self.collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=texts_batch,
                    metadatas=metadata_batch
                )
                logging.info(f"Indexed batch {i//batch_size + 1} ({len(batch)} documents).")
            except Exception as e:
                logging.error(f"Error indexing batch: {e}")
                # Optionally, print details of the problematic batch items
                # for k, item_meta in enumerate(metadata_batch):
                #     logging.debug(f"Item {k} metadata: {item_meta}")

        logging.info(f"Finished indexing. Total documents in collection: {self.collection.count()}")

    def retrieve_lore(self, query_text, k=1, god=None):
        if self.collection.count() == 0:
            logging.debug("Collection is empty.")
            return []
        logging.debug(f"Retrieving legacy lore for query: '{query_text[:50]}...' (god: {god}, k: {k})")
        query_embedding = self.embedding_model.encode(query_text).tolist()
        where_filter = {"god": god} if god else None
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

    def retrieve_lore_with_agents(self, query_text: str, k: int = 3) -> dict:
        """Hybrid agent-based retrieval: returns summary and ranked docs."""
        logging.debug(f"[HybridOrchestrator] Retrieving AGENTIC lore for query: '{query_text[:50]}...' (k: {k})")
        return self.orchestrator.retrieve(query_text, k=k)

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
        contains a summary and ranked docs mentioning Apollo or Daphne.
        """
        print("\n[TEST] Agentic HybridOrchestrator RAG")
        rag = RAGSystem()
        query = "Tell me about Apollo and Daphne"
        result = rag.retrieve_lore_with_agents(query, k=3)
        assert isinstance(result, dict), "Result should be a dict"
        assert "summary" in result and isinstance(result["summary"], str), "Missing or invalid summary"
        # Accept both 'documents' and 'ranked_docs' as valid keys
        docs_key = None
        if "ranked_docs" in result and isinstance(result["ranked_docs"], list):
            docs_key = "ranked_docs"
        elif "documents" in result and isinstance(result["documents"], list):
            docs_key = "documents"
        else:
            print("Result keys:", list(result.keys()))
            raise AssertionError("Missing or invalid 'documents' or 'ranked_docs' in result")
        assert len(result[docs_key]) > 0, "No ranked docs returned"
        print("Summary:", result["summary"])
        print("Top doc snippet:", result[docs_key][0]["text"][:120])
        # Check that the summary or top doc mentions Apollo or Daphne
        assert "Apollo" in result["summary"] or "Daphne" in result["summary"] or \
               "Apollo" in result[docs_key][0]["text"] or "Daphne" in result[docs_key][0]["text"], \
               "Neither Apollo nor Daphne found in summary or top doc"
        print("[PASS] Agentic RAG returns relevant results.")

    test_agentic_rag()
