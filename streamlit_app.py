import os
import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from embedder_rag import create_index, retrieve_docs

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

# Inicjalizacja session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
if "file_content" not in st.session_state:
    st.session_state["file_content"] = None
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None


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
    return documents


# Sidebar – wczytywanie PDF i budowanie indeksu
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_content = ""
        for page in doc:
            file_content += page.get_text()
        doc.close()

        st.session_state["file_content"] = file_content

        # Buduj indeks FAISS z wczytanego PDF
        with st.spinner("Budowanie indeksu RAG..."):
            documents = [{"filename": uploaded_file.name, "text": file_content}]
            st.session_state["faiss_index"] = create_index(documents)

        st.success(f"Wczytano: {uploaded_file.name}")
        with st.expander("Podgląd tekstu"):
            st.write(file_content)

    if st.session_state["file_content"] is not None:
        if st.button("Wyczyść plik"):
            st.session_state["file_content"] = None
            st.session_state["faiss_index"] = None
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

    # RAG: pobierz trafne fragmenty jeśli indeks istnieje
    if st.session_state["faiss_index"] is not None:
        docs = retrieve_docs(prompt, st.session_state["faiss_index"], k=3)
        context = "\n\n".join([d["text"] for d in docs])
        full_prompt = f"Kontekst z dokumentu:\n{context}\n\nPytanie: {prompt}"
    else:
        full_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(prompt)  # wyświetlamy oryginalny prompt bez kontekstu

    # Wysyłamy do API tylko pola role i content
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