from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import re
from pathlib import Path
import json
import time

DATA_DIR = Path("data/far_aim")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "far_aim"
PROGRESS_FILE = Path("ingest_progress.json")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

def load_pdfs():
    texts = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"Loading {pdf_path.name}...")
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                texts.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_num + 1
                })
    return texts

def chunk_text(text, chunk_size=1000, overlap=200):
    sections = re.split(r'(§\s*\d+\.\d+|\d+\.\d+\s|^\s*[A-Z][A-Za-z\s-]{5,}:|\n\s*[A-Z]{3,})', text, flags=re.MULTILINE)
    chunks = []
    current = ""
    for part in sections:
        if len(current) + len(part) > chunk_size and current:
            chunks.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        chunks.append(current.strip())
    return chunks

def main():
    print("🚀 Resumable Accurate Ingestion...")

    if DB_DIR.exists() and PROGRESS_FILE.exists():
        print("Resuming previous ingestion...")
    elif DB_DIR.exists():
        print("Clearing old DB for clean start...")
        import shutil
        shutil.rmtree(DB_DIR)

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    documents = load_pdfs()
    print(f"Loaded {len(documents)} pages")

    # Load progress if exists
    processed = 0
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            processed = json.load(f).get("processed", 0)
        print(f"Resuming from chunk {processed}")

    all_chunks = []
    metadatas = []
    ids = []
    chunk_id = processed

    for doc_idx, doc in enumerate(documents):
        if doc_idx < processed:
            continue
        print(f"Chunking {doc['source']} page {doc['page']}...")
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadatas.append({"source": doc["source"], "page": doc["page"]})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

            # Save progress and batch add
            if len(all_chunks) >= 2000:
                print(f"Adding batch of {len(all_chunks)} chunks...")
                collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
                all_chunks.clear()
                metadatas.clear()
                ids.clear()
                
                with open(PROGRESS_FILE, "w") as f:
                    json.dump({"processed": doc_idx + 1}, f)
                print(f"Progress saved at document {doc_idx + 1}")

    # Add remaining chunks
    if all_chunks:
        collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)

    # Cleanup progress file
    PROGRESS_FILE.unlink(missing_ok=True)

    print(f"\n✅ Ingestion complete! Total chunks: {chunk_id}")
    print("Aviation Brain is ready.")

if __name__ == "__main__":
    main()
