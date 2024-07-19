import streamlit as st          
import os
from groq import Groq
import pandas as pd

st.write("""# Groq Llama 3 ChatBot 🦙
Have a question for Llama or want to get answers from any particular text file or csv? 
         
You are at the right place. Welcome!!! 😊
        """)

api_key = st.secrets["GROQ_API_KEY"]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_contents' not in st.session_state:
    st.session_state.file_contents = []

title = st.chat_input("Ask me anything!!!")

if title:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    # Include file contents in the prompt if any
    prompt = st.session_state.chat_history + [{"role": "system", "content": content} for content in st.session_state.file_contents]
    
    prompt.append({
                "role": "user",
                "content": title,
            })
    chat_completion = client.chat.completions.create(
        messages=prompt,
        model="llama3-8b-8192",
    )
    st.session_state.chat_history.append({
        "role": "user",
        "content": title,
    })
    st.session_state.chat_history.append({
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            })

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(file_name)
        file_extension = file_name.split('.')[-1]
    
        if file_extension == 'csv':
            # Read CSV file into a DataFrame
            dataframe = pd.read_csv(uploaded_file)
            st.session_state.file_contents.append(dataframe.to_string(index=False))    
        elif file_extension == 'txt':
            # Read text file content
            string_data = uploaded_file.read().decode("utf-8")
            st.session_state.file_contents.append(string_data)
        else:
            st.error("Unsupported file type")

# Display the chat history and file contents
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])
    elif message["role"] == "file":
        st.text(message["content"])

# Add GitHub and LinkedIn links at the bottom left
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
    unsafe_allow_html=True
)