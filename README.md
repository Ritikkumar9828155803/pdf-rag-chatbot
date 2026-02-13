# 📄 PDF RAG Chatbot (Local LLM + FAISS + Streamlit)

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from PDF documents using a **local LLM (TinyLlama via Ollama)**, **FAISS vector search**, and **SentenceTransformers embeddings**.

This project demonstrates an **end-to-end GenAI pipeline** with semantic search, context grounding, and low-memory optimization for local deployment.

---

## 🚀 Features

- 📂 Upload any PDF and ask questions
- 🔍 Semantic search using FAISS
- 🧠 Context-aware answers using RAG
- 🤖 Local LLM inference with Ollama (TinyLlama)
- 💬 Chat history in UI
- ⚡ Cached embeddings and index for performance
- 🧮 Low-RAM optimization (small chunks + limited context)

---

## 🏗️ Architecture

User → Streamlit UI → PDF Loader → Text Chunking →  
Embeddings (SentenceTransformers) → FAISS Vector Search →  
Context Retrieval → Ollama (TinyLlama) → Answer

---

## 🛠️ Tech Stack

- **LLM:** TinyLlama (via Ollama)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS
- **Frontend:** Streamlit
- **PDF Parsing:** pypdf
- **Language:** Python

---

## 📂 Project Structure

pdf-rag-chatbot/
│
├── streamlit_app.py # Main Streamlit UI + RAG pipeline
├── requirements.txt # Project dependencies
├── .gitignore # Ignore cache, PDFs, FAISS index
└── README.md # Project documentation




---

## ⚙️ Setup Instructions

'''md


### 1️ Clone the Repository

```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot


### 2 Create Virtual Envirenment

python -m venv venv
venv\Scripts\activate


### 3 Install Dependencies

pip install -r requirements.txt


### 4 Install and Run Ollama

ollama pull tinyllama

### 4 Run the Streamlit App

streamlit run pdf_chatbot_streamlit_app.py
