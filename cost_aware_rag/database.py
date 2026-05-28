import os
import re
from typing import List, Dict, Any 
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# Path to save our local Chroma DB index files
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "icd10_guidelines"

def get_embedding_function():
    """
    Returns a free local embedding function using Sentence-Transformers.
    The 'all-MiniLM-L6-v2' model runs locally and is cached on disk.
    """

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
def get_chroma_client():
    """Initializes and returns a persistent Chroma client."""
    return chromadb.PersistentClient(path=DB_PATH)    

def get_or_create_collection():
    """Returns the ICD-10 collection, creating it if it doesn't exist."""
    client = get_chroma_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)    

def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from an ICD-10 PDF guide.
    """
    
    print(f"Extracting pages from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            pages.append({
                "page_number": idx + 1,
                "text": text,
            })
    print(f"Extracted {len(pages)} valid pages.")        
    return pages

def split_into_children(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """
    Splits parent page text into smaller overlapping child chunks.
    This creates sliding windows of text.
    """  

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start: end].strip())
        start += (chunk_size - overlap)
    return [c for c in chunks if len(c) > 10]    

def ingest_pdf(pdf_path: str):
    """
    Ingests the PDF into ChromaDB using the Parent-Child strategy:
    - Extracts pages (parents).
    - Splits each page into smaller windows (children).
    - Embeds and uploads child chunks, storing parent text in metadata.
    """

    collection = get_or_create_collection()

    #reset collection first to avoid duplicates
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_or_create_collection()

    pages = extract_pages_from_pdf(pdf_path)

    #Extract Pages (Parents)
    pages = extract_pages_from_pdf(pdf_path)

    documents = []
    metadatas = []
    ids = []

    child_count = 0
    for page in pages:
        parent_id = f"page_{page['page_number']}"
        parent_text = page['text']

        # Split parent text into child chunks
        child_chunks = split_into_children(parent_text)

        for child_idx, child_text in enumerate(child_chunks):
            child_id = f"{parent_id}_child_{child_idx}"

            documents.append(child_text)
            metadatas.append({
                "parent_id": parent_id,
                "page_num": page['page_number'],
                "parent_text": parent_text
            })
            ids.append(child_id)
            child_count += 1

            if len(documents) >= 500:
                print(f"Adding batch of {len(documents)} documents to ChromaDB...")
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents, metadatas, ids = [], [], []

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
            )   

    print(f"Ingestion complete! Embedded {child_count} child chunks mapping to {len(pages)} parent pages.") 

def query_database(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB with the user query:
    - Searches child collection.
    - Extracts parent page texts and de-duplicates them.
    - Returns a list of unique matching parent pages.
    """

    collection = get_or_create_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results or not results['metadatas'] or len(results['metadatas'][0]) == 0:
        return []

    seen_parents = set()
    unique_parents = []

    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results else [0]*len(metadatas)

    for meta, dist in zip(metadatas, distances):
        parent_id = meta['parent_id']
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            unique_parents.append({
                "page_num": meta['page_num'],
                "text": meta['parent_text'],
                "score": float(dist)
            })

    return unique_parents
            

   




