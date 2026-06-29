import streamlit as st
import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="AviateGPT Brain", page_icon="🧠", layout="wide")
st.title("🧠 AviateGPT - FAA Regs RAG")
st.caption("Local FAR/AIM + General Aircraft Lookup")

OLLAMA_URL = "http://10.11.12.60:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

# Global resources
embedder = None
regs_collection = None
aircraft_collection = None

@st.cache_resource(show_spinner=True)
def get_brain():
    with st.spinner("☕ Waking up the Aviation Brain..."):
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        # Match ingestion model
        local_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        
        db_path = PROJECT_ROOT / "chroma_db"
        client = chromadb.PersistentClient(path=str(db_path))
        
        local_regs = client.get_or_create_collection("far_aim")
        local_aircraft = client.get_or_create_collection("aircraft_specs")
        
        return local_embedder, local_regs, local_aircraft

# Load resources once
embedder, regs_collection, aircraft_collection = get_brain()

def flatten_embedding(emb):
    if isinstance(emb, list):
        while isinstance(emb, list) and len(emb) == 1:
            emb = emb[0]
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            emb = [item for sublist in emb for item in sublist]
    return emb

def web_aircraft_lookup(question: str) -> str:
    try:
        query_embedding = embedder.encode([question]).tolist()[0]
        query_embedding = flatten_embedding(query_embedding)
        
        aircraft_results = aircraft_collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents"]
        )
        
        if aircraft_results and "documents" in aircraft_results and aircraft_results["documents"]:
            first_match_list = aircraft_results["documents"][0]
            if first_match_list:
                return str(first_match_list[0]).strip()
        return "No specific aircraft design features found."
    except Exception as e:
        return f"Aircraft data layer offline: {str(e)}"

def ask_reg_question(question: str) -> str:
    try:
        raw_emb = embedder.encode([question]).tolist()[0]
        query_embedding = flatten_embedding(raw_emb)
        
        aircraft_context = web_aircraft_lookup(question)
        
        results = regs_collection.query(
            query_embeddings=[query_embedding], 
            n_results=8,
            include=["documents", "metadatas", "distances"]
        )
        
        contexts = [f"[Source: {m.get('source','Unknown')} | Page: {m.get('page','N/A')} | Dist: {d:.4f}]\n{doc.strip()}" 
                    for doc, m, d in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
        context_str = "\n\n---\n\n".join(contexts)

        prompt = f"""You are an expert FAA regulations specialist with access to the full set of FAA documents.

Aircraft details:
{aircraft_context}

Relevant regulatory context:
{context_str}

Question: {question}

Answer accurately, cite sources when possible, and distinguish solo vs certificate age requirements."""

        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
        }, timeout=90)
        
        return res.json().get("response", "No response.").strip()
    
    except Exception as e:
        return f"**Error executing pipeline:** {str(e)}"

# Chat UI
if "brain_messages" not in st.session_state:
    st.session_state.brain_messages = []

for msg in st.session_state.brain_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about FAA regulations..."):
    st.session_state.brain_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🛫 Consulting full FAA regs..."):
            answer = ask_reg_question(prompt)
            st.markdown(answer)
    
    st.session_state.brain_messages.append({"role": "assistant", "content": answer})