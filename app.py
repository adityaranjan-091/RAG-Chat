import streamlit as st
from ingestion import ingest_pdf
from rag_chain import get_chain

st.set_page_config(
    page_title="RAG Chat — Gemini + LangChain",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] * {
        color: #9ca3af !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f3f4f6 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1f2937 !important;
        margin: 1rem 0 !important;
    }

    /* file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 0.4rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > button {
        color: #60a5fa !important;
    }

    /* process button */
    [data-testid="stSidebar"] .stButton > button {
        background: #2563eb;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        font-size: 0.88rem;
        width: 100%;
        transition: background 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1d4ed8;
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: #1f2937 !important;
        color: #4b5563 !important;
    }

    /* sidebar alerts */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #1f2937 !important;
        border-radius: 8px !important;
        border: 1px solid #374151 !important;
        font-size: 0.84rem;
    }

    /* ── Header ─────────────────────────────────────────────────────── */
    .main-header-wrap {
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .header-divider {
        width: 40px;
        height: 2px;
        background: #2563eb;
        border-radius: 2px;
        margin: 0.8rem auto 1.2rem auto;
    }

    /* ── Status Banner ──────────────────────────────────────────────── */
    .status-banner {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.7rem 1.2rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .status-waiting {
        background: #fefce8;
        border: 1px solid #fde68a;
        color: #92400e;
    }
    .status-ready {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
    }

    /* ── Chat Messages ──────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
        border: 1px solid #e5e7eb;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f9fafb !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff !important;
    }

    /* chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1px solid #d1d5db !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    [data-testid="stChatInput"] button {
        background: #2563eb !important;
        border-radius: 8px !important;
    }

    /* ── Source Expander ─────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        margin-top: 0.5rem;
    }
    [data-testid="stExpander"] blockquote {
        border-left-color: #2563eb !important;
        font-size: 0.84rem;
        line-height: 1.55;
    }

    /* ── How-to Card ────────────────────────────────────────────────── */
    .how-to-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .how-to-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #e5e7eb !important;
    }
    .how-to-card ol {
        margin: 0;
        padding-left: 1.1rem;
        line-height: 1.75;
        font-size: 0.8rem;
        color: #9ca3af !important;
    }

    /* ── Footer ─────────────────────────────────────────────────────── */
    .powered-by {
        text-align: center;
        padding: 0.6rem;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #4b5563 !important;
        border-top: 1px solid #1f2937;
        margin-top: 0.8rem;
    }

    /* ── Empty State ────────────────────────────────────────────────── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3.5rem 2rem;
        text-align: center;
    }
    .empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.5; }
    .empty-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.3rem;
    }
    .empty-sub {
        font-size: 0.88rem;
        color: #9ca3af;
        max-width: 340px;
        line-height: 1.5;
    }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — PDF upload & processing
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 Document Upload")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to ask questions about it.",
    )

    if uploaded_file:
        st.success(f"📎 **{uploaded_file.name}** selected")

    process_btn = st.button("⚡ Process Document", disabled=not uploaded_file)

    if process_btn and uploaded_file:
        with st.spinner("Reading and processing document…"):
            try:
                vectorstore = ingest_pdf(uploaded_file, filename=uploaded_file.name)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = get_chain(vectorstore)
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.success("✅ Document processed! You can now ask questions.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

    st.markdown("---")

    st.markdown(
        """
        <div class="how-to-card">
            <h4>ℹ️ How to use</h4>
            <ol>
                <li>Upload a PDF file above</li>
                <li>Click <strong>Process Document</strong></li>
                <li>Ask questions in the chat</li>
                <li>Follow-ups are context-aware</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="powered-by">'
        "Powered by Gemini · LangChain · ChromaDB"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Main area — header and chat
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-header-wrap">'
    '  <div class="main-header">📄 RAG Chat</div>'
    '  <div class="sub-header">Upload a PDF and chat with it using LLM</div>'
    '  <div class="header-divider"></div>'
    "</div>",
    unsafe_allow_html=True,
)

if st.session_state.chain is None:
    st.markdown(
        '<div class="status-banner status-waiting">'
        "  ⏳ No document loaded — upload a PDF to get started"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-banner status-ready">'
        "  ✅ Document ready — ask anything below"
        "</div>",
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '  <div class="empty-icon">💬</div>'
        '  <div class="empty-title">No messages yet</div>'
        '  <div class="empty-sub">'
        "    Upload and process a PDF, then start asking questions. "
        "    Your conversation will appear here."
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source Snippets"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Chunk {i}** (page {src.get('page', '?')}):\n"
                        f"> {src['text'][:300]}…"
                    )

# ═══════════════════════════════════════════════════════════════════════════
# Chat input
# ═══════════════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask a question about your document…"):
    if st.session_state.chain is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.chain(
                        {
                            "question": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )

                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    sources_data = []
                    if source_docs:
                        with st.expander("📚 Source Snippets"):
                            for i, doc in enumerate(source_docs, 1):
                                page = doc.metadata.get("page", "?")
                                snippet = doc.page_content[:300]
                                st.markdown(
                                    f"**Chunk {i}** (page {page}):\n> {snippet}…"
                                )
                                sources_data.append(
                                    {"page": page, "text": doc.page_content}
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_data,
                        }
                    )
                    st.session_state.chat_history.append((prompt, answer))

                except Exception as e:
                    error_msg = f"❌ An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )