import streamlit as st
import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="AviateGPT Brain", page_icon="🧠", layout="wide")
st.title("🧠 AviateGPT - FAA Regs RAG")
st.caption("Local FAR/AIM + General Aircraft Lookup")

OLLAMA_URL = "http://10.11.12"
MODEL_NAME = "llama3.1:8b"

@st.cache_resource(show_spinner=False)
def get_brain():
    with st.spinner("☕ Waking up the Aviation Brain..."):
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        # Enforce CPU operation for consistency with your existing scripts
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        db_path = PROJECT_ROOT / "chroma_db"
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Open both database collection layers safely
        regs_collection = client.get_or_create_collection("far_aim")
        aircraft_collection = client.get_or_create_collection("aircraft_specs")
        
        return embedder, regs_collection, aircraft_collection

# Load and unpack the multi-layer workspace database resources
embedder, regs_collection, aircraft_collection = get_brain()

def web_aircraft_lookup(question: str) -> str:
    """Queries the local FAA registry database to pull physical airframe traits."""
    try:
        query_embedding = embedder.encode([question]).tolist()
        
        # Search the aircraft_specs database for the single best row match
        aircraft_results = aircraft_collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents"]
        )
        
        # Extract the document string safely out of ChromaDB's nested array format
        if aircraft_results and "documents" in aircraft_results and aircraft_results["documents"]:
            first_match_list = aircraft_results["documents"][0]
            if first_match_list:
                return str(first_match_list[0]).strip()
                
        return "No specific aircraft design features found in local database references."
    except Exception as e:
        return f"Aircraft data layer offline: {str(e)}"

def ask_reg_question(question: str) -> str:
    try:
        query_embedding = embedder.encode([question]).tolist()
        
        # Step 1: Resolve the exact plane configuration traits from your indexed registry rows
        aircraft_context = web_aircraft_lookup(question)
        
        # Step 2: Extract relevant regulatory data fragments from your manuals collection
        results = regs_collection.query(
            query_embeddings=[query_embedding], 
            n_results=5,  # Balanced chunk count to prevent local context dilution
            include=["documents", "metadatas", "distances"]
        )
        
        # Parse text chunks cleanly out of nested layout results arrays
        contexts = [f"[Source: {m.get('source','Unknown')} p.{m.get('page','N/A')} | Distance: {d:.4f}]\n{doc.strip()}" 
                    for doc, m, d in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
        context_str = "\n\n---\n\n".join(contexts)

        # Step 3: Map the physical specifications directly against the legal text chunks
        prompt = f"""You are a strict FAA regulations expert. 
Your job is to determine the pilot requirements for the aircraft using ONLY the provided physical characteristics and the Regulatory Context block.

Aircraft Configuration (retrieved from FAA Registry):
{aircraft_context}

Regulatory Context from database:
{context_str}

Question: {question}

Answer:"""

        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
        }, timeout=90)
        
        return res.json().get("response", "No response.").strip()
    
    except Exception as e:
        return f"**Error executing pipeline:** {str(e)}"

# Chat User Interface Controller
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
        with st.spinner("🛫 Looking up aircraft + consulting regs..."):
            answer = ask_reg_question(prompt)
            st.markdown(answer)
    
    st.session_state.brain_messages.append({"role": "assistant", "content": answer})
