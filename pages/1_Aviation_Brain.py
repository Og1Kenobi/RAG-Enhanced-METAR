import streamlit as st
import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="AviateGPT Brain", page_icon="🧠", layout="wide")
st.title("🧠 AviateGPT - FAA Regs RAG")
st.caption("Local RAG over FAR/AIM • Powered by Ollama")

OLLAMA_URL = "http://10.11.12.60:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

@st.cache_resource
def get_brain():
    from sentence_transformers import SentenceTransformer
    import chromadb
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    db_path = PROJECT_ROOT / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("far_aim")
    return embedder, collection

embedder, collection = get_brain()

def ask_reg_question(question: str) -> str:
    try:
        query_embedding = embedder.encode([question])[0].tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        
        contexts = [f"[Source: {m['source']} Page {m['page']}]\n{d.strip()}" 
                    for d, m in zip(results["documents"][0], results["metadatas"][0])]
        context = "\n\n".join(contexts)

        prompt = f"""You are a strict FAA regulations expert. Answer ONLY based on regulations. No opinions.

For student pilot questions:
- Quote the relevant 14 CFR sections when possible.
- State exactly what is allowed or prohibited.

Context:
{context}

Question: {question}

Answer:"""

        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0
        }, timeout=60)
        
        return res.json().get("response", "No response.").strip()
    except Exception as e:
        return f"Error: {str(e)}"

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
        with st.spinner("Consulting the regs..."):
            answer = ask_reg_question(prompt)
            st.markdown(answer)
    
    st.session_state.brain_messages.append({"role": "assistant", "content": answer})