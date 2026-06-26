import streamlit as st
import requests
import sys
from pathlib import Path

# Dynamic paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="AviateGPT Brain", page_icon="🧠", layout="wide")
st.title("🧠 AviateGPT - FAA Regs RAG")
st.caption("Local RAG over FAR/AIM • Powered by Ollama")

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

def ask_reg_question(question):
    try:
        query_embedding = embedder.encode([question])[0].tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        
        contexts = [f"[Source: {m['source']} Page {m['page']}]\n{d.strip()}" 
                    for d, m in zip(results["documents"][0], results["metadatas"][0])]
        context = "\n\n".join(contexts)

        prompt = f"""You are a strict CFI. Answer FAA questions accurately.

For student pilot Class B questions, be balanced and quote relevant regulations when possible.

Context:
{context}

Question: {question}

Answer:"""

        # Dynamic Ollama URL from environment or default
        ollama_url = "http://10.11.12.60:11435/api/generate"
        model_name = "qwen2.5-coder:14b"

        res = requests.post(ollama_url, json={
            "model": model_name,
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
