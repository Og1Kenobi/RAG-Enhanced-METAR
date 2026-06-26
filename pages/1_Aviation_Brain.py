import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="AviateGPT Brain", page_icon="🧠", layout="wide")
st.title("🧠 AviateGPT - FAA Regs RAG")
st.caption("Local RAG over FAR/AIM • Powered by Qwen2.5-Coder:14B (Ollama)")

# Ollama settings - Updated for your server
OLLAMA_URL = "http://10.11.12.60:11435/api/generate"
MODEL_NAME = "qwen2.5-coder:14b"

@st.cache_resource(show_spinner="Loading embeddings and vector DB...")
def get_brain():
    from sentence_transformers import SentenceTransformer
    import chromadb
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    # Adjust path if needed
    db_path = Path("../chroma_db").resolve()
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_collection("far_aim")
    return embedder, collection

embedder, collection = get_brain()

def ask_reg_question(question: str) -> str:
    try:
        # Retrieve relevant chunks
        boosted = f"{question} Part 91 regulations inoperative equipment MEL"
        query_embedding = embedder.encode([boosted])[0].tolist()
        
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        
        contexts = [f"[Source: {m['source']} Page {m['page']}]\n{d.strip()}" 
                    for d, m in zip(results["documents"][0], results["metadatas"][0])]
        context = "\n\n".join(contexts)

        prompt = f"""You are an experienced CFI. Answer accurately using ONLY the FAA context below.
Be practical for real-world Part 91 flying. Cite sources when helpful.

Context:
{context}

Question: {question}

Answer:"""

        # Call Ollama (same as your METAR app)
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "top_p": 0.9
        }
        
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        return res.json().get("response", "No response from Ollama.").strip()
        
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure Ollama is running and the model is loaded."

# Chat UI
if "brain_messages" not in st.session_state:
    st.session_state.brain_messages = []

for msg in st.session_state.brain_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about FAA regulations, Part 91, procedures..."):
    st.session_state.brain_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with Qwen2.5-Coder..."):
            answer = ask_reg_question(prompt)
            st.markdown(answer)
    
    st.session_state.brain_messages.append({"role": "assistant", "content": answer})

st.sidebar.info("✅ Using Ollama • Qwen2.5-Coder:14B")
