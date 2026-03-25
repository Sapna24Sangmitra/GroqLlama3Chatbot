Groq Llama 3 ChatBot

This project is an interactive chatbot that integrates with the Groq API for chat completions. It supports file uploads (CSV and TXT) and uses RAG (Retrieval-Augmented Generation) to answer questions grounded in the uploaded documents.

Features:
1. Interactive chat interface powered by Llama 3 (via Groq API).
2. File upload functionality (CSV and TXT).
3. RAG pipeline — uploaded files are chunked, embedded locally, and indexed with FAISS. Only the most relevant excerpts are sent to the LLM per query, not the entire file.
4. Local embeddings using sentence-transformers (all-MiniLM-L6-v2) — no external embedding API or cost.
5. Multi-file support — chunks from multiple uploaded files are accumulated into a single searchable index.
6. Chat history is displayed and maintained throughout the session.

How RAG works in this app:
- On file upload: the document is split into 200-word overlapping chunks, embedded into vectors, and stored in a FAISS index.
- On each query: the user's question is embedded and the top-3 most similar chunks are retrieved and injected as context into the LLM prompt.

Tech Stack:
- Streamlit — UI
- Groq API — LLM (Llama 3)
- sentence-transformers — local text embeddings
- FAISS — vector similarity search

Installation:

pip install streamlit groq pandas sentence-transformers faiss-cpu

Add your Groq API key to .streamlit/secrets.toml:

GROQ_API_KEY = "your_key_here"

Run:

streamlit run chatBot.py

My First Streamlit GenAI App 😊

![image](https://github.com/user-attachments/assets/bb6336bc-dbe3-4482-80e4-1cf5d048cb05)
