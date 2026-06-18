import streamlit as st
import fitz
import faiss
import numpy as np
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Asystent UMP – Poznań", page_icon="🏛️")
st.title("🏛️ Wirtualny Asystent – Urząd Miasta Poznania")

# EMBEDDINGI

EMBED_MODEL_ID = "intfloat/e5-small-v2"
EMBED_MODEL_KWARGS = {"device": "cpu", "trust_remote_code": True}


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs=EMBED_MODEL_KWARGS)


def read_pdf(file_bytes):
    """Wyciąga tekst z PDF-a podanego jako bytes (np. uploaded_file.read())."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text, chunk_size=800, overlap=100):
    """Dzieli tekst na nakładające się fragmenty - cały PDF jako 1 wektor dawałby bardzo słabe wyszukiwanie."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# SŁOWNIK

# Mapowanie odpowiedzi routera (POJAZDY/PRAWO_JAZDY/OGOLNE) na klucze wewnętrzne.
CATEGORY_MAP = {
    "POJAZDY": "pojazdy",
    "PRAWO_JAZDY": "prawo_jazdy",
    "OGOLNE": "ogolne",
}

# FAISS – tworzenie i przeszukiwanie

class FAISSIndex:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata  # lista fragmentów [{"filename":..., "text":...}, ...]


def build_index(documents):
    """Buduje indeks FAISS z listy dokumentów [{'filename':..., 'text':...}, ...]."""
    embeddings = get_embeddings()
    chunks_meta = [
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
    """Zwraca skonkatenowany kontekst (string) z k najbardziej podobnych fragmentów."""
    if faiss_index is None:
        return ""
    embeddings = get_embeddings()
    query_vector = np.array([embeddings.embed_query(query)], dtype="float32")
    _, indices = faiss_index.index.search(query_vector, k)
    chunks = [
        faiss_index.metadata[i]["text"]
        for i in indices[0]
        if 0 <= i < len(faiss_index.metadata)
    ]
    return "\n\n".join(chunks)

# ROUTER LLM

ROUTER_PROMPT = """Jesteś klasyfikatorem zapytań dla Urzędu Miasta Poznania. Odpowiedz TYLKO jednym słowem:
- POJAZDY     – rejestracja, wyrejestrowanie, dowód rejestracyjny, tablice, sprzedaż/zakup auta, VIN, OC, złomowanie
- PRAWO_JAZDY – prawo jazdy, PKK, PKZ, KKK, egzamin, kurs, zatrzymane PJ, tramwaj, pojazd uprzywilejowany
- OGOLNE      – inne / powitanie / niejasne"""


def get_expert(query, gemini_client):
    """Klasyfikuje zapytanie przez Gemini. Zwraca ('pojazdy'|'prawo_jazdy'|'ogolne', label)."""
    try:
        resp = gemini_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": query},
            ],
            max_tokens=25,
            temperature=0,
        )
        cat = resp.choices[0].message.content.strip().upper()
        if cat not in CATEGORY_MAP:
            cat = "OGOLNE"
    except Exception:
        cat = "OGOLNE"

    return CATEGORY_MAP[cat], cat


# SYSTEM PROMPTS

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

# WYWOŁANIA EKSPERTÓW

def ask_gemini(query, history, context, client):
    system = SYS_GEMINI
    if context:
        system += f"\n\n=== BAZA WIEDZY ===\n{context}"
    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": query})
    resp = client.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.3)
    return resp.choices[0].message.content


def ask_groq(query, history, context, client):
    system = SYS_GROQ
    if context:
        system += f"\n\n=== BAZA WIEDZY ===\n{context}"
    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": query})
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.3)
    return resp.choices[0].message.content

# KLIENCI API (cache – tworzone raz)

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


# SESSION STATE

if "messages"      not in st.session_state: st.session_state.messages      = []
if "faiss_pojazdy" not in st.session_state: st.session_state.faiss_pojazdy = None
if "faiss_prawo"   not in st.session_state: st.session_state.faiss_prawo   = None

# SIDEBAR

with st.sidebar:
    st.header("Bazy wiedzy")

    st.subheader("🚗 Gemini – Pojazdy")
    up_pojazdy = st.file_uploader("PDF – rejestracja pojazdów", type="pdf", key="up_p")
    if up_pojazdy and st.session_state.get("pojazdy_file") != up_pojazdy.name:
        with st.spinner("Tworzę embeddingi..."):
            text = read_pdf(up_pojazdy.read())
            st.session_state.faiss_pojazdy = build_index([{"filename": up_pojazdy.name, "text": text}])
        st.session_state.pojazdy_file = up_pojazdy.name
        st.success(f"Wczytano: {up_pojazdy.name}")

    st.subheader("📋 Groq (Llama) – Prawo jazdy")
    up_prawo = st.file_uploader("PDF – uprawnienia do kierowania", type="pdf", key="up_c")
    if up_prawo and st.session_state.get("prawo_file") != up_prawo.name:
        with st.spinner("Tworzę embeddingi..."):
            text = read_pdf(up_prawo.read())
            st.session_state.faiss_prawo = build_index([{"filename": up_prawo.name, "text": text}])
        st.session_state.prawo_file = up_prawo.name
        st.success(f"Wczytano: {up_prawo.name}")

    st.divider()
    st.caption("🚗 Gemini obsługuje sprawy pojazdowe")
    st.caption("📋 Groq / Llama obsługuje prawo jazdy")
    st.caption("🔀 Router automatycznie kieruje zapytania")

    if st.button("🗑️ Wyczyść historię"):
        st.session_state.messages = []
        st.rerun()

# HISTORIA CZATU

AVATAR    = {"pojazdy": "🚗", "prawo_jazdy": "📋", "ogolne": "🏛️"}
LABEL_MAP = {
    "POJAZDY":     "🚗 Ekspert: Pojazdy (Gemini)",
    "PRAWO_JAZDY": "📋 Ekspert: Prawo jazdy (Groq / Llama)",
    "OGOLNE":      "🏛️ Asystent ogólny (Gemini)",
}

for msg in st.session_state.messages:
    role   = msg["role"]
    expert = msg.get("expert", "ogolne")
    avatar = "🧑" if role == "user" else AVATAR.get(expert, "🏛️")
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and "category" in msg:
            st.caption(LABEL_MAP.get(msg["category"], ""))
        st.write(msg["content"])

# OBSŁUGA WEJŚCIA

if prompt := st.chat_input("Zadaj pytanie o rejestrację pojazdu lub prawo jazdy..."):
    gemini_client = get_gemini_client()
    groq_client   = get_groq_client()

    # Zapisz i wyświetl wiadomość użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    # Routing
    with st.spinner("🔀 Kieruję do właściwego eksperta..."):
        expert, category = get_expert(prompt, gemini_client)

    # RAG – pobierz kontekst z właściwego indeksu
    context = ""
    if expert == "pojazdy" and st.session_state.faiss_pojazdy:
        context = retrieve(prompt, st.session_state.faiss_pojazdy)
    elif expert == "prawo_jazdy" and st.session_state.faiss_prawo:
        context = retrieve(prompt, st.session_state.faiss_prawo)

    # Odpowiedź eksperta
    with st.chat_message("assistant", avatar=AVATAR.get(expert, "🏛️")):
        st.caption(LABEL_MAP.get(category, ""))
        with st.spinner("Generuję odpowiedź..."):
            try:
                if expert in ("pojazdy", "ogolne"):
                    answer = ask_gemini(prompt, st.session_state.messages[:-1], context, gemini_client)
                else:
                    answer = ask_groq(prompt, st.session_state.messages[:-1], context, groq_client)
            except Exception as e:
                answer = f"Przepraszam, wystąpił błąd: {e}"
        st.write(answer)

    # Zapisz odpowiedź do historii
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "expert":   expert,
        "category": category,
    })
