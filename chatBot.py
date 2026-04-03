import streamlit as st
import os
from groq import Groq
import pandas as pd
from rag_utils import load_model, chunk_text, build_index, retrieve

st.write("""# Groq Llama 3 ChatBot 🦙
Have a question for Llama or want to get answers from any particular text file or csv?

You are at the right place. Welcome!!! 😊
        """)

api_key = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def get_embedding_model():
    """Load the sentence-transformer model once and cache it across sessions."""
    return load_model()


embedding_model = get_embedding_model()

# --- Session state initialisation ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a file (TXT or CSV)")
if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.uploaded_files:
        file_extension = file_name.rsplit(".", 1)[-1].lower()

        if file_extension == "csv":
            dataframe = pd.read_csv(uploaded_file)
            raw_text = dataframe.to_string(index=False)
        elif file_extension == "txt":
            raw_text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload a TXT or CSV file.")
            raw_text = None

        if raw_text:
            with st.spinner(f"Indexing '{file_name}'..."):
                new_chunks = chunk_text(raw_text)
                st.session_state.rag_chunks.extend(new_chunks)
                st.session_state.rag_index = build_index(
                    st.session_state.rag_chunks, embedding_model
                )
                st.session_state.uploaded_files.append(file_name)
            st.success(
                f"'{file_name}' indexed — {len(new_chunks)} chunks "
                f"({len(st.session_state.rag_chunks)} total)"
            )

def stream_response(stream):
    """Yield text tokens from a Groq streaming response."""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


# --- Display existing chat history ---
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])

# --- Chat input ---
user_input = st.chat_input("Ask me anything!!!")

if user_input:
    client = Groq(api_key=api_key)

    # Build message list: start fresh from history each turn
    messages = list(st.session_state.chat_history)

    # If a file has been indexed, retrieve relevant chunks and inject as context
    if st.session_state.rag_index is not None and st.session_state.rag_chunks:
        relevant_chunks = retrieve(
            user_input,
            st.session_state.rag_index,
            st.session_state.rag_chunks,
            embedding_model,
            top_k=3,
        )
        context = "\n\n---\n\n".join(relevant_chunks)
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "Use the following excerpts from the uploaded document to answer "
                    "the user's question. If the answer is not in the excerpts, say so.\n\n"
                    f"{context}"
                ),
            },
        )

    messages.append({"role": "user", "content": user_input})

    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Stream assistant response token by token
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            stream=True,
        )
        assistant_reply = st.write_stream(stream_response(stream))

    # Persist both turns to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

# --- Fixed footer ---
st.markdown(
    """
    <style>
    .fixed-footer {
        position: fixed;
        bottom: 10px;
        left: 10px;
        width: auto;
        background-color: white;
        text-align: left;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        z-index: 100;
    }
    </style>
    <div class="fixed-footer">
        <a href="https://github.com/Sapna24Sangmitra" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/sapna-sangmitra" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
