import sys, os, tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from core.vector_store import get_store

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #ffffff; }
section[data-testid="stSidebar"] { background: #f9fafb !important; border-right: 1px solid #e5e7eb; }
.stButton > button {
    background: #2563eb !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }
.stTextInput > div > div > input {
    border: 1px solid #d1d5db !important; border-radius: 8px !important;
    background: #fff !important; font-size: 0.95rem !important;
}
div[data-testid="stFileUploader"] { border: 2px dashed #bfdbfe; border-radius: 10px; background: #eff6ff; }
.msg-user {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; margin: 6px 0; max-width: 75%; margin-left: auto; font-size: 0.92rem;
}
.msg-bot {
    background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 16px 16px 16px 4px;
    padding: 10px 14px; margin: 6px 0; max-width: 85%; font-size: 0.92rem; line-height: 1.6;
}
.lbl { font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: #9ca3af; margin-bottom: 4px; }
.badge { display:inline-block; background:#dbeafe; border-radius:20px; padding:2px 10px; font-size:0.68rem; color:#1d4ed8; margin:2px; }
.stat { background:#f3f4f6; border-radius:10px; padding:10px; text-align:center; }
.stat-n { font-size:1.4rem; font-weight:700; color:#2563eb; }
.stat-l { font-size:0.62rem; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i + 500])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _index_file(file_path: str, original_name: str = "") -> dict:
    store = get_store()
    path = Path(file_path)
    source_label = original_name.strip() if original_name.strip() else path.name
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".pdf":
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
    else:
        return {"error": "Only .txt and .pdf supported"}
    if not text.strip():
        return {"error": "Document is empty or unreadable"}
    chunks = _chunk_text(text)
    for i, chunk in enumerate(chunks):
        store.add(chunk, source_label, i)
    return {"status": "indexed", "file": source_label, "chunks": len(chunks)}


def _index_text_direct(text: str, source_name: str) -> dict:
    store = get_store()
    chunks = _chunk_text(text)
    for i, chunk in enumerate(chunks):
        store.add(chunk, source_name, i)
    return {"status": "indexed", "source": source_name, "chunks": len(chunks)}


def _search_direct(query: str, top_k: int = 4) -> List[dict]:
    return get_store().search(query, top_k=top_k)


def ask_groq(query: str) -> str:
    """Search docs, build context, call Groq API."""
    from groq import Groq

    results = _search_direct(query, top_k=4)

    if results:
        context_parts = [f"[Source: {r['source']}]\n{r['text']}" for r in results]
        context = "\n\n---\n\n".join(context_parts)
        system_msg = "You are a helpful RAG assistant. Answer the user's question using ONLY the provided document context. Cite the source filename. Be concise."
        user_msg = f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {query}"
    else:
        system_msg = "You are a helpful assistant."
        user_msg = f"{query}\n\n(Note: No documents are indexed yet. Tell the user to upload a document first.)"

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return "⚠️ **Groq API key missing!**\n\nGet a free key from https://console.groq.com → paste it in `.env` as `GROQ_API_KEY=...` → restart the app."

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or "No response."
    except Exception as e:
        return f"⚠️ Error: {e}"


# ── Session state ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────────
store = get_store()
with st.sidebar:
    st.markdown("## 📚 RAG Assistant")
    st.caption("Nexe-Agent Internship · Task 4")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat'><div class='stat-n'>{len(store.sources())}</div><div class='stat-l'>Docs</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat'><div class='stat-n'>{store.total_chunks()}</div><div class='stat-l'>Chunks</div></div>", unsafe_allow_html=True)

    st.divider()
    if store.sources():
        st.markdown("**Indexed Documents**")
        for s in store.sources():
            st.markdown(f"<span class='badge'>📄 {s}</span>", unsafe_allow_html=True)
        st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear All", use_container_width=True):
            store.clear()
            st.session_state.history = []
            st.rerun()
    with col2:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    api_set = bool(os.getenv("GROQ_API_KEY", "").strip())
    st.markdown(f"**API:** {'✅ Groq connected' if api_set else '❌ Set GROQ_API_KEY in .env'}")
    st.caption("Model: llama-3.3-70b-versatile\nGroq · Free tier")


# ── Main ──────────────────────────────────────────────────────────────────
st.markdown("## 📚 RAG Assistant")
st.caption("Upload documents and ask questions — answers grounded in your files.")
st.divider()

with st.expander("➕ Upload / Index Documents", expanded=len(store.sources()) == 0):
    tab1, tab2 = st.tabs(["📂 Upload File", "📋 Paste Text"])

    with tab1:
        f = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
        if f:
            st.info(f"**{f.name}** · {max(1, f.size // 1024)} KB")
            if st.button("⚡ Index Document"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                with st.spinner("Indexing…"):
                    result = _index_file(tmp_path, original_name=f.name)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success(f"✅ {result['file']} indexed! ({result['chunks']} chunks)")
                st.rerun()

    with tab2:
        name = st.text_input("Source name", placeholder="e.g. my_notes")
        text_input = st.text_area("Paste text here", height=150)
        if st.button("⚡ Index Text"):
            if name.strip() and text_input.strip():
                with st.spinner("Indexing…"):
                    result = _index_text_direct(text_input.strip(), source_name=name.strip())
                st.success(f"✅ '{name}' indexed! ({result['chunks']} chunks)")
                st.rerun()
            else:
                st.warning("Enter both a name and some text.")

st.divider()

# Chat
if not st.session_state.history:
    st.markdown("<br><p style='text-align:center;color:#9ca3af'>💬 Upload a document above, then ask anything below.</p>", unsafe_allow_html=True)
else:
    for m in st.session_state.history:
        if m["role"] == "user":
            txt = m["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(f"<div class='msg-user'><div class='lbl'>You</div>{txt}</div>", unsafe_allow_html=True)
        else:
            txt = m["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            st.markdown(f"<div class='msg-bot'><div class='lbl'>🤖 Assistant</div>{txt}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_i, col_b = st.columns([5, 1])
with col_i:
    query = st.text_input("Ask", placeholder="Ask anything about your documents…", label_visibility="collapsed", key="q")
with col_b:
    if st.button("Send →", use_container_width=True):
        if query.strip():
            with st.spinner("Thinking…"):
                reply = ask_groq(query.strip())
            st.session_state.history.append({"role": "user", "content": query.strip()})
            st.session_state.history.append({"role": "assistant", "content": reply})
            st.rerun()
