import os
import streamlit as st
import fitz
import faiss
import numpy as np
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Asystent UMP – Poznań", page_icon="🏛️")
st.title("🏛️ Wirtualny Asystent – Urząd Miasta Poznania")

# ──────────────────────────────────────────────
# EMBEDDINGI
# ──────────────────────────────────────────────

EMBED_MODEL_ID = "intfloat/e5-small-v2"
EMBED_MODEL_KWARGS = {"device": "cpu", "trust_remote_code": True}


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs=EMBED_MODEL_KWARGS)


def read_pdf_bytes(file_bytes):
    """Wyciąga tekst z PDF-a podanego jako bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def read_pdf_path(path):
    """Wyciąga tekst z PDF-a podanego jako ścieżka lokalna."""
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ──────────────────────────────────────────────
# SŁOWNIK – mapowanie etykiet routera
# ──────────────────────────────────────────────

CATEGORY_MAP = {
    "POJAZDY":     "pojazdy",
    "PRAWO_JAZDY": "prawo_jazdy",
    "OGOLNE":      "ogolne",
    "LOKALNY":     "lokalny",   # 3. ekspert – lokalny LLM
}

# ──────────────────────────────────────────────
# FAISS – tworzenie i przeszukiwanie
# ──────────────────────────────────────────────

class FAISSIndex:
    def __init__(self, index, metadata):
        self.index    = index
        self.metadata = metadata


def build_index(documents):
    """Buduje indeks FAISS z listy dokumentów [{'filename':..., 'text':...}]."""
    embeddings   = get_embeddings()
    chunks_meta  = [
        {"filename": doc["filename"], "text": chunk}
        for doc in documents
        for chunk in chunk_text(doc["text"])
    ]
    if not chunks_meta:
        return None
    vectors = np.array(
        [embeddings.embed_query(c["text"]) for c in chunks_meta], dtype="float32"
    )
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return FAISSIndex(index, chunks_meta)


def retrieve(query, faiss_index, k=4):
    """Zwraca skonkatenowany kontekst z k najpodobniejszych fragmentów."""
    if faiss_index is None:
        return ""
    embeddings   = get_embeddings()
    query_vector = np.array([embeddings.embed_query(query)], dtype="float32")
    _, indices   = faiss_index.index.search(query_vector, k)
    chunks = [
        faiss_index.metadata[i]["text"]
        for i in indices[0]
        if 0 <= i < len(faiss_index.metadata)
    ]
    return "\n\n".join(chunks)

# ──────────────────────────────────────────────
# POMOCNIK: ładowanie PDF-ów z folderu lokalnego
# ──────────────────────────────────────────────

def load_folder(folder_path):
    """Wczytuje wszystkie PDF-y z podanego folderu. Zwraca listę dokumentów."""
    documents = []
    if not os.path.isdir(folder_path):
        return documents
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            text = read_pdf_path(os.path.join(folder_path, fname))
            documents.append({"filename": fname, "text": text})
    return documents

# ──────────────────────────────────────────────
# ROUTER LLM
# ──────────────────────────────────────────────

def build_router_prompt(local_enabled, local_topic):
    local_line = ""
    if local_enabled:
        local_line = f"\n- LOKALNY     – {local_topic}"
    return (
        "Jesteś klasyfikatorem zapytań dla Urzędu Miasta Poznania. Odpowiedz TYLKO jednym słowem:\n"
        "- POJAZDY     – rejestracja, wyrejestrowanie, dowód rejestracyjny, tablice, sprzedaż/zakup auta, VIN, OC, złomowanie\n"
        "- PRAWO_JAZDY – prawo jazdy, PKK, PKZ, KKK, egzamin, kurs, zatrzymane PJ, tramwaj, pojazd uprzywilejowany\n"
        "- OGOLNE      – inne / powitanie / niejasne"
        + local_line
    )


def get_expert(query, gemini_client, local_enabled, local_topic):
    """Klasyfikuje zapytanie. Zwraca (klucz_wewnętrzny, ETYKIETA)."""
    router_prompt = build_router_prompt(local_enabled, local_topic)
    valid_cats    = set(CATEGORY_MAP.keys())
    if not local_enabled:
        valid_cats.discard("LOKALNY")
    try:
        resp = gemini_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user",   "content": query},
            ],
            max_tokens=25,
            temperature=0,
        )
        cat = resp.choices[0].message.content.strip().upper()
        if cat not in valid_cats:
            cat = "OGOLNE"
    except Exception:
        cat = "OGOLNE"
    return CATEGORY_MAP[cat], cat

# ──────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────

SYS_GEMINI = """Jesteś wirtualnym pracownikiem Urzędu Miasta Poznania, specjalistą ds. rejestracji pojazdów.
Specjalizacja: rejestracja/wyrejestrowanie pojazdów, dowody i tablice rejestracyjne, sprzedaż pojazdu, zmiany w pojeździe.
Zasady: mów per Pan/Pani, podawaj konkretne dokumenty i opłaty, przypominaj o rezerwacji wizyty.
Odpowiadaj na podstawie kontekstu z bazy wiedzy. Jeśli pytanie dotyczy prawa jazdy – odesłaj do eksperta ds. prawa jazdy.
Infolinia: 61 646 33 44 | bip.poznan.pl"""

SYS_GROQ = """Jesteś wirtualnym pracownikiem Urzędu Miasta Poznania, specjalistą ds. uprawnień do kierowania.
Specjalizacja: PKK, PKZ, KKK, prawo jazdy (wydanie, wymiana, wtórnik), zatrzymane PJ, międzynarodowe PJ, tramwaj, pojazd uprzywilejowany.
Zasady: mów per Pan/Pani, podawaj konkretne dokumenty i opłaty. WAŻNE: obsługa TYLKO klientów umówionych (wyjątek: wymiana zagranicznego PJ – bez rezerwacji, ul. Gronowa 22a parter).
Odpowiadaj na podstawie kontekstu z bazy wiedzy. Jeśli pytanie dotyczy rejestracji pojazdu – odesłaj do eksperta ds. pojazdów.
Infolinia: 61 646 33 44 | bip.poznan.pl"""

# ──────────────────────────────────────────────
# WYWOŁANIA EKSPERTÓW
# ──────────────────────────────────────────────

def _build_messages(system, history, query, context):
    if context:
        system += f"\n\n=== BAZA WIEDZY ===\n{context}"
    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": query})
    return msgs


def ask_gemini(query, history, context, client):
    msgs = _build_messages(SYS_GEMINI, history, query, context)
    resp = client.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.3)
    return resp.choices[0].message.content


def ask_groq(query, history, context, client):
    msgs = _build_messages(SYS_GROQ, history, query, context)
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.3)
    return resp.choices[0].message.content


def ask_local(query, history, context, client, model_name, sys_prompt):
    msgs = _build_messages(sys_prompt, history, query, context)
    resp = client.chat.completions.create(model=model_name, messages=msgs, temperature=0.3)
    return resp.choices[0].message.content

# ──────────────────────────────────────────────
# KLIENCI API (cache)
# ──────────────────────────────────────────────

@st.cache_resource
def get_gemini_client():
    return OpenAI(
        api_key=st.secrets["GEMINI_API_KEY"],
        base_url=st.secrets["GEMINI_BASE_URL"],
    )

@st.cache_resource
def get_groq_client():
    return OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

@st.cache_resource
def get_local_client(base_url, api_key):
    return OpenAI(api_key=api_key or "not-needed", base_url=base_url)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────

defaults = {
    "messages":      [],
    "faiss_pojazdy": None,
    "faiss_prawo":   None,
    "faiss_lokalny": None,
    "pojazdy_file":  None,
    "prawo_file":    None,
    "lokalny_src":   None,   # ślad źródła ostatnio zaindeksowanego lokalnego
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Bazy wiedzy")

    # ── 1. GEMINI – POJAZDY ──────────────────
    with st.expander("🚗 Gemini – Pojazdy", expanded=True):
        src_p = st.radio("Źródło PDF", ["Upload pliku", "Folder lokalny"], key="src_pojazdy")

        if src_p == "Upload pliku":
            up_pojazdy = st.file_uploader("PDF – rejestracja pojazdów", type="pdf", key="up_p")
            if up_pojazdy and st.session_state.pojazdy_file != up_pojazdy.name:
                with st.spinner("Tworzę embeddingi…"):
                    text = read_pdf_bytes(up_pojazdy.read())
                    st.session_state.faiss_pojazdy = build_index(
                        [{"filename": up_pojazdy.name, "text": text}]
                    )
                st.session_state.pojazdy_file = up_pojazdy.name
                st.success(f"Wczytano: {up_pojazdy.name}")
        else:
            folder_p = st.text_input("Ścieżka folderu z PDF-ami", key="folder_pojazdy",
                                     placeholder="np. C:/dane/pojazdy lub /home/user/pojazdy")
            if st.button("📂 Załaduj folder", key="btn_folder_p"):
                docs = load_folder(folder_p)
                if docs:
                    with st.spinner(f"Tworzę embeddingi ({len(docs)} PDF-ów)…"):
                        st.session_state.faiss_pojazdy = build_index(docs)
                    st.session_state.pojazdy_file = folder_p
                    st.success(f"Załadowano {len(docs)} pliki z: {folder_p}")
                else:
                    st.error("Folder pusty lub ścieżka nieprawidłowa.")

        if st.session_state.faiss_pojazdy:
            n = len(st.session_state.faiss_pojazdy.metadata)
            st.caption(f"✅ Indeks aktywny · {n} fragmentów")

    # ── 2. GROQ – PRAWO JAZDY ────────────────
    with st.expander("📋 Groq (Llama) – Prawo jazdy", expanded=True):
        src_c = st.radio("Źródło PDF", ["Upload pliku", "Folder lokalny"], key="src_prawo")

        if src_c == "Upload pliku":
            up_prawo = st.file_uploader("PDF – uprawnienia do kierowania", type="pdf", key="up_c")
            if up_prawo and st.session_state.prawo_file != up_prawo.name:
                with st.spinner("Tworzę embeddingi…"):
                    text = read_pdf_bytes(up_prawo.read())
                    st.session_state.faiss_prawo = build_index(
                        [{"filename": up_prawo.name, "text": text}]
                    )
                st.session_state.prawo_file = up_prawo.name
                st.success(f"Wczytano: {up_prawo.name}")
        else:
            folder_c = st.text_input("Ścieżka folderu z PDF-ami", key="folder_prawo",
                                     placeholder="np. C:/dane/prawo_jazdy")
            if st.button("📂 Załaduj folder", key="btn_folder_c"):
                docs = load_folder(folder_c)
                if docs:
                    with st.spinner(f"Tworzę embeddingi ({len(docs)} PDF-ów)…"):
                        st.session_state.faiss_prawo = build_index(docs)
                    st.session_state.prawo_file = folder_c
                    st.success(f"Załadowano {len(docs)} pliki z: {folder_c}")
                else:
                    st.error("Folder pusty lub ścieżka nieprawidłowa.")

        if st.session_state.faiss_prawo:
            n = len(st.session_state.faiss_prawo.metadata)
            st.caption(f"✅ Indeks aktywny · {n} fragmentów")

    # ── 3. LOKALNY LLM (opcjonalny) ──────────
    st.divider()
    local_enabled = st.toggle("🖥️ Włącz lokalny LLM (3. ekspert)", value=False)

    if local_enabled:
        with st.expander("⚙️ Konfiguracja lokalnego LLM", expanded=True):
            local_base_url   = st.text_input("Base URL serwera", value="http://localhost:11434/v1",
                                             help="Ollama: http://localhost:11434/v1  |  LM Studio: http://localhost:1234/v1")
            local_api_key    = st.text_input("API key (opcjonalnie)", value="", type="password",
                                             help="Zostaw puste jeśli serwer nie wymaga klucza")
            local_model_name = st.text_input("Nazwa modelu", value="llama3",
                                             help="Dokładna nazwa widoczna po 'ollama list' lub w LM Studio")
            local_topic      = st.text_input("Tematyka (dla routera)", value="",
                                             placeholder="np. historia Poznania, zabytki, kultura",
                                             help="Router użyje tego opisu, żeby kierować pasujące pytania do lokalnego LLM")
            local_sys_prompt = st.text_area("System prompt lokalnego asystenta",
                                            value="Jesteś pomocnym asystentem Urzędu Miasta Poznania. Odpowiadaj po polsku, mów per Pan/Pani.",
                                            height=100)

            # PDF dla lokalnego LLM
            st.markdown("**Baza wiedzy lokalnego LLM**")
            src_l = st.radio("Źródło PDF", ["Upload pliku", "Folder lokalny"], key="src_lokalny")

            if src_l == "Upload pliku":
                up_lokalny = st.file_uploader("PDF dla lokalnego LLM", type="pdf", key="up_l",
                                              accept_multiple_files=True)
                if up_lokalny:
                    src_id = "_".join(sorted(f.name for f in up_lokalny))
                    if st.session_state.lokalny_src != src_id:
                        with st.spinner("Tworzę embeddingi…"):
                            docs = [
                                {"filename": f.name, "text": read_pdf_bytes(f.read())}
                                for f in up_lokalny
                            ]
                            st.session_state.faiss_lokalny = build_index(docs)
                        st.session_state.lokalny_src = src_id
                        st.success(f"Wczytano {len(docs)} plik(ów)")
            else:
                folder_l = st.text_input("Ścieżka folderu z PDF-ami", key="folder_lokalny",
                                         placeholder="np. C:/dane/lokalny")
                if st.button("📂 Załaduj folder", key="btn_folder_l"):
                    docs = load_folder(folder_l)
                    if docs:
                        with st.spinner(f"Tworzę embeddingi ({len(docs)} PDF-ów)…"):
                            st.session_state.faiss_lokalny = build_index(docs)
                        st.session_state.lokalny_src = folder_l
                        st.success(f"Załadowano {len(docs)} pliki z: {folder_l}")
                    else:
                        st.error("Folder pusty lub ścieżka nieprawidłowa.")

            if st.session_state.faiss_lokalny:
                n = len(st.session_state.faiss_lokalny.metadata)
                st.caption(f"✅ Indeks aktywny · {n} fragmentów")
    else:
        local_base_url   = ""
        local_api_key    = ""
        local_model_name = ""
        local_topic      = ""
        local_sys_prompt = ""

    st.divider()
    st.caption("🚗 Gemini obsługuje sprawy pojazdowe")
    st.caption("📋 Groq / Llama obsługuje prawo jazdy")
    if local_enabled:
        st.caption("🖥️ Lokalny LLM obsługuje wybraną tematykę")
    st.caption("🔀 Router automatycznie kieruje zapytania")

    if st.button("🗑️ Wyczyść historię"):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────
# HISTORIA CZATU
# ──────────────────────────────────────────────

AVATAR = {"pojazdy": "🚗", "prawo_jazdy": "📋", "ogolne": "🏛️", "lokalny": "🖥️"}
LABEL_MAP = {
    "POJAZDY":     "🚗 Ekspert: Pojazdy (Gemini)",
    "PRAWO_JAZDY": "📋 Ekspert: Prawo jazdy (Groq / Llama)",
    "OGOLNE":      "🏛️ Asystent ogólny (Gemini)",
    "LOKALNY":     "🖥️ Ekspert lokalny",
}

for msg in st.session_state.messages:
    role   = msg["role"]
    expert = msg.get("expert", "ogolne")
    avatar = "🧑" if role == "user" else AVATAR.get(expert, "🏛️")
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and "category" in msg:
            st.caption(LABEL_MAP.get(msg["category"], ""))
        st.write(msg["content"])

# ──────────────────────────────────────────────
# OBSŁUGA WEJŚCIA
# ──────────────────────────────────────────────

if prompt := st.chat_input("Zadaj pytanie o rejestrację pojazdu lub prawo jazdy..."):
    gemini_client = get_gemini_client()
    groq_client   = get_groq_client()
    local_client  = get_local_client(local_base_url, local_api_key) if local_enabled and local_base_url else None

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    # Routing
    with st.spinner("🔀 Kieruję do właściwego eksperta…"):
        expert, category = get_expert(prompt, gemini_client, local_enabled, local_topic)

    # RAG – kontekst z właściwego indeksu
    context = ""
    if expert == "pojazdy" and st.session_state.faiss_pojazdy:
        context = retrieve(prompt, st.session_state.faiss_pojazdy)
    elif expert == "prawo_jazdy" and st.session_state.faiss_prawo:
        context = retrieve(prompt, st.session_state.faiss_prawo)
    elif expert == "lokalny" and st.session_state.faiss_lokalny:
        context = retrieve(prompt, st.session_state.faiss_lokalny)

    # Odpowiedź
    with st.chat_message("assistant", avatar=AVATAR.get(expert, "🏛️")):
        st.caption(LABEL_MAP.get(category, ""))
        with st.spinner("Generuję odpowiedź…"):
            try:
                if expert == "lokalny" and local_client:
                    answer = ask_local(
                        prompt, st.session_state.messages[:-1],
                        context, local_client, local_model_name, local_sys_prompt
                    )
                elif expert == "prawo_jazdy":
                    answer = ask_groq(prompt, st.session_state.messages[:-1], context, groq_client)
                else:
                    answer = ask_gemini(prompt, st.session_state.messages[:-1], context, gemini_client)
            except Exception as e:
                answer = f"Przepraszam, wystąpił błąd: {e}"
        st.write(answer)

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "expert":   expert,
        "category": category,
    })