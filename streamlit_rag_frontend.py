import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="📚",
    layout="wide",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-start: #0b1020;
            --bg-end: #151b34;
            --panel: rgba(18, 24, 40, 0.72);
            --panel-border: rgba(255,255,255,0.12);
            --soft: rgba(255,255,255,0.08);
            --card: rgba(255,255,255,0.06);
            --text: #f4f7ff;
            --muted: #c9d1ea;
            --primary: #7c3aed;
            --secondary: #22d3ee;
            --success: #34d399;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, var(--bg-start) 0%, #111827 35%, var(--bg-end) 100%);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(12, 18, 30, 0.9));
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .stSidebar .block-container {
            padding-top: 1.2rem;
        }

        .sidebar-card {
            background: linear-gradient(135deg, rgba(124,58,237,0.22), rgba(34,211,238,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-card strong {
            color: white;
        }

        .hero-card {
            display: grid;
            grid-template-columns: 1.6fr 0.9fr;
            gap: 1rem;
            background: linear-gradient(135deg, rgba(124,58,237,0.24), rgba(34,211,238,0.18), rgba(15,23,42,0.45));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 26px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.3);
        }

        .tag {
            display: inline-block;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #dbeafe;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hero-card h1 {
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.05;
            margin: 0.7rem 0 0.6rem 0;
            background: linear-gradient(90deg, #f8fafc 0%, #c4b5fd 30%, #67e8f9 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .hero-card p {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0;
            line-height: 1.7;
        }

        .hero-image {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
        }

        .hero-image img {
            width: 100%;
            max-height: 240px;
            object-fit: cover;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 12px 32px rgba(76, 29, 149, 0.35);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .stChatMessage {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 8px 20px rgba(15,23,42,0.2);
        }

        .stChatMessage[data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.45);
        }

        .stButton > button {
            background: linear-gradient(90deg, #8b5cf6, #06b6d4);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.65rem 1rem;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
        }

        .metric-box {
            background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(124,58,237,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.8rem 1rem;
        }

        @media (max-width: 850px) {
            .hero-card {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
inject_styles()

st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <strong>LangGraph PDF Chatbot</strong><br>
        <span style="color:#dbeafe;">Thread ID: {}</span>
    </div>
    """.format(thread_key),
    unsafe_allow_html=True,
)

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.markdown(
    """
    <div class="hero-card">
        <div>
            <span class="tag">AI Research Copilot</span>
            <h1>Multi Utility Chatbot</h1>
            <p>Upload PDFs, ask rich questions, and get context-aware answers backed by LangGraph tools and memory.</p>
        </div>
        <div class="hero-image">
            <img src="https://images.unsplash.com/photo-1531482615713-2afd69097998?auto=format&fit=crop&w=900&q=80" alt="AI assistant" />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
