import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    PDF_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def recursive_character_splitter(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursively splits text using paragraph, newline, sentence, and space separators.
    
    Guarantees that each page's text or technical specification stays intact within
    a single chunk if under chunk_size, while cleanly chunking longer pages.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    separators = ["\n\n", "\n", ". ", " "]
    chosen_sep = " "
    for sep in separators:
        if sep in cleaned_text:
            chosen_sep = sep
            break

    splits = cleaned_text.split(chosen_sep)
    chunks = []
    current_chunk = []
    current_length = 0

    for split in splits:
        piece = split + chosen_sep
        piece_len = len(piece)
        if current_length + piece_len > chunk_size and current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Preserve trailing context for chunk_overlap
            overlap_acc = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) <= chunk_overlap:
                    overlap_acc.insert(0, item)
                    overlap_len += len(item)
                else:
                    break
            current_chunk = overlap_acc
            current_length = overlap_len

        current_chunk.append(piece)
        current_length += piece_len

    if current_chunk:
        combined = "".join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Loads PDF page-by-page preserving 1-indexed page numbers (1..50)."""
    doc = fitz.open(pdf_path)
    pages = []
    for idx in range(len(doc)):
        page_num = idx + 1
        page_text = doc[idx].get_text()
        pages.append({
            "page": page_num,
            "text": page_text.strip()
        })
    return pages


def ingest(force: bool = False) -> int:
    """Loads the manual page-by-page, chunks text, generates embeddings,
    and stores them in persistent ChromaDB.
    
    Skips re-embedding if the collection exists and is non-empty unless force=True.
    """
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF document not found at: {PDF_PATH}")

    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        count = collection.count()
        if count > 0 and not force:
            print(f"[INGEST] Collection '{COLLECTION_NAME}' already populated ({count} chunks). Skipping re-embedding.")
            return count
        elif force:
            print(f"[INGEST] '--force' active. Rebuilding collection '{COLLECTION_NAME}' (clearing {count} chunks)...")
            client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"description": "AeroVantage X9 50-page Technical Manual Evaluation Corpus"}
    )

    print(f"[INGEST] Loading pages from: {PDF_PATH}")
    pages = extract_pages(str(pdf_file))
    print(f"[INGEST] Extracted {len(pages)} pages.")

    all_ids = []
    all_documents = []
    all_metadatas = []

    for page_data in pages:
        p_num = page_data["page"]
        p_text = page_data["text"]
        chunks = recursive_character_splitter(p_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for c_idx, chunk_text in enumerate(chunks):
            chunk_id = f"page_{p_num}_chunk_{c_idx}"
            metadata = {
                "page": p_num,
                "chunk_id": chunk_id,
                "source": f"Page {p_num}"
            }
            all_ids.append(f"doc_p{p_num}_c{c_idx}")
            all_documents.append(chunk_text)
            all_metadatas.append(metadata)

    print(f"[INGEST] Generated {len(all_documents)} chunks. Generating embeddings & storing in ChromaDB...")
    collection.add(
        ids=all_ids,
        documents=all_documents,
        metadatas=all_metadatas
    )
    print(f"[INGEST] Successfully indexed {len(all_documents)} chunks into '{CHROMA_DIR}'.")
    return len(all_documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest AeroVantage X9 manual into persistent ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion and overwrite existing embeddings")
    args = parser.parse_args()
    ingest(force=args.force)
