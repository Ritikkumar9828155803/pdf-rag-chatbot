
# STREAMLIT UI FOR GENERATIVE RAG


# IMPORTS-------

import streamlit as st
import os
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama



#CONFIG----

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

LLAMA_MODEL = "tinyllama"
EMBED_MODEL = "all-MiniLM-L6-v2"





# LOAD EMBEDDING MODEL (CACHED)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()





# CHUNK FUNCTION

def chunk_text(text, chunk_size=500, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks



# BUILD OR LOAD VECTOR STORE----

@st.cache_resource
def build_or_load_index(pdf_path):

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    print("Building new FAISS index...")

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks



# RETRIEVE CONTEXT----------

def retrieve(question, index, chunks, top_k=2):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0]]




# GENERATE ANSWER (LLAMA 3)---------

def generate_answer(question, context_chunks):

    context = "\n\n".join(context_chunks)[:1200]

    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]



# ASK MAIN FUNCTION---------

def ask_pdf(question, pdf_path):

    index, chunks = build_or_load_index(pdf_path)

    context_chunks = retrieve(question, index, chunks)

    answer = generate_answer(question, context_chunks)

    return answer

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("LLaMA 3 PDF Chatbot (Generative RAG)")


# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_path = uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    question = st.text_input("Ask a question about the PDF:")

    if question:

        with st.spinner("Thinking..."):
            answer = ask_pdf(question, pdf_path)

        st.session_state.chat_history.append((question, answer))

        st.markdown("### Answer")
        st.write(answer)


    st.markdown("### Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

