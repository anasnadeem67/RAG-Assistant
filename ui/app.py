import sys, os, asyncio, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from agents import Runner
from core.agent import build_agent
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
.stTextInput > div > div > input:focus { border-color: #2563eb !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important; }

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
.chat-wrap { max-height: 50vh; overflow-y: auto; padding: 4px 0; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "history" not in st.session_state:
    st.session_state.history = []


def run_agent(msg: str) -> str:
    st.session_state.history.append({"role": "user", "content": msg})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(Runner.run(st.session_state.agent, st.session_state.history, max_turns=50))
        reply = result.final_output
    except Exception as e:
        reply = f"⚠️ Error: {e}"
    finally:
        loop.close()
    st.session_state.history.append({"role": "assistant", "content": reply})
    return reply


# ── Sidebar ──────────────────────────────────────────────────────────────
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
        if st.button("🗑 Clear", use_container_width=True):
            run_agent("Please clear the entire vector store.")
            st.rerun()
    with col2:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption("Model: nvidia/nemotron-3-super-120b-a12b:free\nOpenRouter · OpenAI Agents SDK")


# ── Main Page ─────────────────────────────────────────────────────────────
st.markdown("## 📚 RAG Assistant")
st.caption("Upload documents and ask questions — answers are grounded in your files.")
st.divider()

# Upload section
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
                    run_agent(f"Please upload and index this document: {tmp_path}")
                try: os.unlink(tmp_path)
                except: pass
                st.success(f"✅ {f.name} indexed!")
                st.rerun()

    with tab2:
        name = st.text_input("Source name", placeholder="e.g. my_notes")
        text = st.text_area("Paste text here", height=150, placeholder="Paste any content to index…")
        if st.button("⚡ Index Text"):
            if name.strip() and text.strip():
                with st.spinner("Indexing…"):
                    run_agent(f"Add this text to the vector store with source name '{name}':\n\n{text}")
                st.success(f"✅ '{name}' indexed!")
                st.rerun()
            else:
                st.warning("Please enter both a name and some text.")

st.divider()

# Chat history
visible = [
    m for m in st.session_state.history
    if not (m["role"] == "user" and any(m["content"].startswith(p) for p in ("Please upload", "Add this text", "Please clear")))
    and not (m["role"] == "assistant" and m["content"].strip().startswith("{"))
]

if not visible:
    st.markdown("<br><p style='text-align:center;color:#9ca3af;font-size:0.95rem'>💬 Upload a document above, then ask anything below.</p>", unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    for m in visible:
        if m["role"] == "user":
            st.markdown(f"<div class='msg-user'><div class='lbl'>You</div>{m['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-bot'><div class='lbl'>🤖 Assistant</div>{m['content'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Input
st.markdown("<br>", unsafe_allow_html=True)
col_i, col_b = st.columns([5, 1])
with col_i:
    query = st.text_input("Ask a question", placeholder="Ask anything about your documents…", label_visibility="collapsed", key="q")
with col_b:
    if st.button("Send →", use_container_width=True):
        if query.strip():
            with st.spinner("Thinking…"):
                run_agent(query.strip())
            st.rerun()
