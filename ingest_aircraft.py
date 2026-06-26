import pandas as pd
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "chroma_db"
FAA_FILE_PATH = PROJECT_ROOT / "data/aircraft/ACFTREF.txt"

print("🧠 Loading embedding model for aircraft profiles...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

client = chromadb.PersistentClient(path=str(DB_DIR))
aircraft_collection = client.get_or_create_collection("aircraft_specs")

# Translation maps for FAA raw dataset column matrices
AIRCRAFT_TYPES = {
    "1": "Glider", "2": "Balloon", "3": "Blimp/Dirigible", 
    "4": "Fixed-wing single-engine land", "5": "Fixed-wing multi-engine land",
    "6": "Rotorcraft/Helicopter", "7": "Powered Lift", "8": "Weight-shift control"
}
ENGINE_TYPES = {
    "0": "None", "1": "Reciprocating/Piston", "2": "Turbo-prop", 
    "3": "Turbo-shaft", "4": "Turbo-jet", "5": "Turbo-fan", "6": "Ramjet"
}

def clean_row(val):
    return str(val).strip() if pd.notna(val) else ""

print("✈️ Processing extracted FAA ACFTREF.txt data structure...")
df = pd.read_csv(FAA_FILE_PATH, usecols=["CODE", "MFR", "MODEL", "TYPE-ACFT", "TYPE-ENG", "NO-ENG"], dtype=str)

documents = []
metadatas = []
ids = []

for _, row in df.iterrows():
    ac_code = clean_row(row["CODE"])
    mfr = clean_row(row["MFR"])
    model = clean_row(row["MODEL"])
    ac_type_code = clean_row(row["TYPE-ACFT"])
    eng_type_code = clean_row(row["TYPE-ENG"])
    num_engines = clean_row(row["NO-ENG"])
    
    if not mfr or not model or mfr == "NAN" or model == "NAN":
        continue
        
    ac_type_str = AIRCRAFT_TYPES.get(ac_type_code, "Unknown aircraft configuration")
    eng_type_str = ENGINE_TYPES.get(eng_type_code, "Unknown engine setup")
    
    # Building a natural description sentence for the vector embeddings to map cleanly
    description = f"The {mfr} {model} is a {ac_type_str} aircraft equipped with {num_engines} {eng_type_str} engine."
    
    documents.append(description)
    ids.append(ac_code)
    metadatas.append({
        "manufacturer": mfr,
        "model": model,
        "engine_count": num_engines,
        "aircraft_type": ac_type_str,
        "engine_type": eng_type_str
    })

print(f"📦 Populating {len(documents)} airframe design tokens to ChromaDB...")
BATCH_SIZE = 1000
for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]
    
    batch_embeddings = embedder.encode(batch_docs).tolist()
    
    aircraft_collection.add(
        embeddings=batch_embeddings,
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )

print("✅ Aircraft design characteristics layer successfully built!")

