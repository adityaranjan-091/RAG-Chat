import streamlit as st
from ingestion import ingest_pdf
from rag_chain import get_chain
import uuid
from datetime import datetime

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/robot_2/default/24px.svg",
    layout="wide",
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&icon_names=robot_2" />',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style> 
    /* ── Global ─────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #334155;
        background-color: #fcfcfd;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 850px;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] h2 {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: -0.02em;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
        margin: 1rem 0 !important;
    }

    /* File Uploader styling */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: #4f46e5;
        background: #fcfdff;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > button {
        color: #334155 !important;
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.35rem 0.75rem;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > button:hover {
        background: #e2e8f0 !important;
    }

    /* File Card styling */
    .file-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        animation: slideDown 0.3s ease;
    }
    .file-icon {
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .file-info {
        flex-grow: 1;
        min-width: 0;
    }
    .file-name {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .file-size {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.1rem;
    }
    .file-status-pill {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
    }

    /* Process button styling */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
        box-shadow: 0 6px 14px rgba(79, 70, 229, 0.25);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stButton > button:active:not(:disabled) {
        transform: translateY(0);
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: #f1f5f9 !important;
        color: #94a3b8 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: none;
        cursor: not-allowed;
    }

    /* Instructions card styling */
    .instructions-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        margin: 1rem 0;
    }
    .instruction-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .instruction-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    .instruction-step:last-child {
        margin-bottom: 0;
    }
    .step-badge {
        background: #f1f5f9;
        color: #4f46e5;
        font-weight: 700;
        font-size: 0.75rem;
        width: 1.25rem;
        height: 1.25rem;
        border-radius: 9999px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 0.1rem;
        border: 1px solid #e2e8f0;
    }
    .step-content {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
    }
    .step-content strong {
        color: #0f172a;
    }

    /* ── Chat History Panel ───────────────────────────────────────────── */
    .chat-history-section {
        margin-top: 0.5rem;
    }
    .chat-history-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        padding: 0.5rem 0 0.4rem 0;
        margin-bottom: 0.25rem;
    }
    .chat-history-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 0.75rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        border: 1px solid transparent;
        margin-bottom: 0.2rem;
        text-decoration: none;
    }
    .chat-history-item:hover {
        background: #f1f5f9;
        border-color: #e2e8f0;
    }
    .chat-history-item.active {
        background: #ede9fe;
        border-color: #c4b5fd;
    }
    .chat-history-icon {
        font-size: 0.85rem;
        flex-shrink: 0;
        opacity: 0.7;
    }
    .chat-history-title {
        font-size: 0.82rem;
        font-weight: 500;
        color: #334155;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex-grow: 1;
    }
    .chat-history-item.active .chat-history-title {
        color: #4f46e5;
        font-weight: 600;
    }
    .chat-history-time {
        font-size: 0.68rem;
        color: #94a3b8;
        flex-shrink: 0;
    }
    .new-chat-btn-wrap > div > button {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.6rem 1rem !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.4rem !important;
        box-shadow: 0 2px 8px rgba(79,70,229,0.18) !important;
        transition: all 0.2s ease !important;
    }
    .new-chat-btn-wrap > div > button:hover {
        background: #4338ca !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.28) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Main Workspace Header ────────────────────────────────────────── */
    .main-header-wrap {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #1e3b8b 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* ── Status Banner ──────────────────────────────────────────────── */
    .status-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.85rem 1.25rem;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .status-waiting {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
    }
    .status-ready {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #15803d;
        border-left: 4px solid #16a34a;
    }

    /* ── Chat Messages ──────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease;
    }

    /* User Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        border: none !important;
        color: #ffffff !important;
        border-top-right-radius: 2px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Assistant Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-top-left-radius: 2px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        color: #334155;
        line-height: 1.65;
    }

    /* ── Chat Input ─────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        padding-bottom: 1.5rem;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 0.9rem 1.1rem !important;
        font-size: 0.95rem !important;
        background: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
    }
    [data-testid="stChatInput"] button {
        background: #4f46e5 !important;
        color: white !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stChatInput"] button:hover {
        background: #4338ca !important;
        transform: scale(1.05);
    }

    /* ── Source Expander ─────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #4f46e5 !important;
        background: #f8fafc !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background: #f8fafc !important;
        margin-top: 0.85rem;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.01) !important;
    }
    [data-testid="stExpander"] blockquote {
        border-left: 3px solid #cbd5e1 !important;
        color: #475569;
        font-size: 0.85rem;
        line-height: 1.6;
        margin: 0.5rem 0;
        padding-left: 0.85rem;
        background: transparent;
    }

    /* ── Empty State Suggestions Dashboard ───────────────────────────── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 2rem;
        text-align: center;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .empty-icon { 
        font-size: 3rem; 
        margin-bottom: 1rem; 
        opacity: 0.8; 
    }
    .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .empty-sub {
        font-size: 0.95rem;
        color: #64748b;
        max-width: 420px;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Preset Suggestion Grid & Buttons */
    .suggestions-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.85rem;
        text-align: left;
    }

    /* Targets suggestion buttons specifically */
    .main .stButton > button {
        background: #ffffff;
        color: #334155 !important;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        font-weight: 500;
        font-size: 0.88rem;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        line-height: 1.4;
    }
    .main .stButton > button:hover {
        border-color: #4f46e5;
        background: #f5f3ff;
        color: #4f46e5 !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.1);
        transform: translateY(-2px);
    }
    .main .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Footer ─────────────────────────────────────────────────────── */
    .powered-by {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8 !important;
    }

    /* Animations */
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════
if "conversations" not in st.session_state:
    # List of conversation dicts: {id, title, messages, chat_history, chain, vectorstore, created_at}
    st.session_state.conversations = []
if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def _new_conversation():
    """Create a brand-new empty conversation and make it active."""
    conv_id = str(uuid.uuid4())
    conv = {
        "id": conv_id,
        "title": "New Chat",
        "messages": [],
        "chat_history": [],
        "chain": None,
        "vectorstore": None,
        "created_at": datetime.now().strftime("%H:%M"),
    }
    st.session_state.conversations.insert(0, conv)
    st.session_state.active_conv_id = conv_id
    return conv


def _active_conv():
    """Return the currently active conversation dict, creating one if needed."""
    if not st.session_state.active_conv_id:
        return _new_conversation()
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_conv_id:
            return c
    return _new_conversation()


# Ensure there is always at least one conversation
if not st.session_state.conversations:
    _new_conversation()

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Document upload, history, controls
# ═══════════════════════════════════════════════════════════════════════════
def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} GB"

conv = _active_conv()

with st.sidebar:
    # ── New Chat button ──────────────────────────────────────────────────
    st.markdown('<div class="new-chat-btn-wrap">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
        _new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Chat history list ────────────────────────────────────────────────
    if len(st.session_state.conversations) > 0:
        st.markdown('<div class="chat-history-label">Recent Conversations</div>', unsafe_allow_html=True)
        for c in st.session_state.conversations:
            is_active = c["id"] == st.session_state.active_conv_id
            active_class = "active" if is_active else ""
            label = c["title"][:30] + ("…" if len(c["title"]) > 30 else "")
            # Render clickable button for each session
            btn_key = f"conv_{c['id']}"
            if st.button(
                f"{'💬' if is_active else '🗨️'}  {label}",
                key=btn_key,
                use_container_width=True,
            ):
                if not is_active:
                    st.session_state.active_conv_id = c["id"]
                    st.rerun()

    st.markdown("---")

    # ── Document Upload section ──────────────────────────────────────────
    st.markdown("## Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=f"uploader_{conv['id']}",
        help="Upload a PDF document to ask questions about it.",
    )

    if uploaded_file:
        file_size_str = format_size(uploaded_file.size)
        st.markdown(
            f"""
            <div class="file-card">
                <div class="file-icon">📄</div>
                <div class="file-info">
                    <div class="file-name">{uploaded_file.name}</div>
                    <div class="file-size">{file_size_str}</div>
                </div>
                <div class="file-status-pill">Selected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    process_btn = st.button("Process Document", disabled=not uploaded_file, key=f"proc_{conv['id']}")

    if process_btn and uploaded_file:
        with st.spinner("Reading and processing document…"):
            try:
                vectorstore = ingest_pdf(uploaded_file, filename=uploaded_file.name)
                conv["vectorstore"] = vectorstore
                conv["chain"] = get_chain(vectorstore)
                conv["chat_history"] = []
                conv["messages"] = []
                # Name the conversation after the document
                conv["title"] = uploaded_file.name.replace(".pdf", "")[:40]
                st.success("Document processed! You can now ask questions.")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    st.markdown("---")

    st.markdown(
        """
        <div class="instructions-container">
            <div class="instruction-title">Instructions</div>
            <div class="instruction-step">
                <div class="step-badge">1</div>
                <div class="step-content"><strong>Upload</strong> a PDF document using the file uploader above.</div>
            </div>
            <div class="instruction-step">
                <div class="step-badge">2</div>
                <div class="step-content">Click <strong>Process Document</strong> to index the file with vector embeddings.</div>
            </div>
            <div class="instruction-step">
                <div class="step-badge">3</div>
                <div class="step-content">Type your questions in the <strong>chat input field</strong> at the bottom.</div>
            </div>
            <div class="instruction-step">
                <div class="step-badge">4</div>
                <div class="step-content">Review <strong>Reference Sources</strong> below responses for source pages and citations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="powered-by">'
        "Powered by Gemini & LangChain"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Main area — header and chat
# ═══════════════════════════════════════════════════════════════════════════
# Always use the active conversation
conv = _active_conv()

st.markdown(
    '<div class="main-header-wrap">'
    f'  <div class="main-header">Knowledge Base Assistant</div>'
    '  <div class="sub-header">Upload a document to securely explore and extract information using AI.</div>'
    "</div>",
    unsafe_allow_html=True,
)

if conv["chain"] is None:
    st.markdown(
        '<div class="status-banner status-waiting">'
        "  <strong>Awaiting Document:</strong> Please upload and process a PDF from the sidebar to begin."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    doc_name = conv["title"]
    st.markdown(
        '<div class="status-banner status-ready">'
        f"  <strong>System Ready:</strong> <em>{doc_name}</em> is loaded. Ask questions below."
        "</div>",
        unsafe_allow_html=True,
    )

if not conv["messages"]:
    st.markdown(
        '<div class="empty-state">'
        '  <div class="empty-icon">💬</div>'
        '  <div class="empty-title">Ready to Explore</div>'
        '  <div class="empty-sub">'
        "    No conversation history yet. Once your document is loaded, ask anything using the input below, "
        "    or choose one of these suggested queries to start exploring:"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if conv["chain"] is not None:
        st.markdown('<div class="suggestions-header">Suggested Questions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Summarize main points of the document", key="s_1", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the main points of this document."
                st.rerun()
            if st.button("🔍 What are the key findings?", key="s_2", use_container_width=True):
                st.session_state.pending_prompt = "What are the key findings or recommendations?"
                st.rerun()
        with col2:
            if st.button("⚠️ Identify potential risks/limitations", key="s_3", use_container_width=True):
                st.session_state.pending_prompt = "Identify any potential risks or limitations mentioned."
                st.rerun()
            if st.button("📝 Create action items from content", key="s_4", use_container_width=True):
                st.session_state.pending_prompt = "Create a list of action items based on this text."
                st.rerun()

# Render existing chat messages for this conversation
for msg in conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Reference Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Chunk {i}** (page {src.get('page', '?')}):\n"
                        f"> {src['text'][:300]}…"
                    )

# ═══════════════════════════════════════════════════════════════════════════
# Chat input & Suggestions processing
# ═══════════════════════════════════════════════════════════════════════════
prompt = st.chat_input("Ask a question about your document…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    if conv["chain"] is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        # If this is the very first message, set a meaningful title
        if not conv["messages"]:
            conv["title"] = prompt[:45] + ("…" if len(prompt) > 45 else "")

        conv["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document…"):
                try:
                    result = conv["chain"](
                        {
                            "question": prompt,
                            "chat_history": conv["chat_history"],
                        }
                    )

                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    sources_data = []
                    if source_docs:
                        with st.expander("Reference Sources"):
                            for i, doc in enumerate(source_docs, 1):
                                page = doc.metadata.get("page", "?")
                                snippet = doc.page_content[:300]
                                st.markdown(
                                    f"**Chunk {i}** (page {page}):\n> {snippet}…"
                                )
                                sources_data.append(
                                    {"page": page, "text": doc.page_content}
                                )

                    conv["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_data,
                        }
                    )
                    conv["chat_history"].append((prompt, answer))
                    st.rerun()

                except Exception as e:
                    error_msg = f"❌ An error occurred during generation: {e}"
                    st.error(error_msg)
                    conv["messages"].append(
                        {"role": "assistant", "content": error_msg}
                    )