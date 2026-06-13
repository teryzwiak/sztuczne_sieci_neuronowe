from io import StringIO
import pandas as pd

import streamlit as st
from openai import OpenAI
import os
import fitz  # PyMuPDF

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

if "file_content" not in st.session_state:
    st.session_state["file_content"] = None


def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_documents_from_folder(folder_path: str) -> list[dict]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
            documents.append({"filename": filename, "text": text})
    return documents  # poza pętlą for


# Sidebar – wczytywanie PDF
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_content = ""
        for page in doc:
            file_content += page.get_text()
        doc.close()
        st.session_state["file_content"] = file_content
        st.success(f"Wczytano: {uploaded_file.name}")
        with st.expander("Podgląd tekstu"):
            st.write(file_content)

    if st.session_state["file_content"] is not None:
        if st.button("Wyczyść plik"):
            st.session_state["file_content"] = None
            st.rerun()

# Historia czatu
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input użytkownika
if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Dołącz treść PDF do zapytania jeśli plik został wczytany
    if st.session_state["file_content"]:
        full_prompt = (
            f"Kontekst z pliku:\n{st.session_state['file_content']}\n\nPytanie: {prompt}"
        )
    else:
        full_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(prompt)  # wyświetlamy oryginalny prompt, nie z kontekstem

    # Wysyłamy do API tylko pola role i content (bez niestandardowych kluczy)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    response = client.chat.completions.create(
        model=selected_model,
        messages=api_messages,
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)