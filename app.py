"""
Zyro Dynamics HR Help Desk — Streamlit Chatbot
NIAT Masterclass RAG Challenge
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# ─── Load secrets from Streamlit Cloud or .env ────────────────────────────────
def load_secrets():
    try:
        # Streamlit Cloud
        os.environ["GROQ_API_KEY"]          = st.secrets["GROQ_API_KEY"]
        os.environ["LANGCHAIN_API_KEY"]     = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"]  = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_PROJECT"]     = st.secrets.get("LANGCHAIN_PROJECT", "zyro-rag-challenge")
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()

load_secrets()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(102,126,234,0.5);
}
.main-header h1 { color: #fff; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.main-header p  { color: rgba(255,255,255,0.8); margin: 0.5rem 0 0; font-size: 1rem; }

.chat-bubble {
    padding: 1rem 1.4rem;
    border-radius: 14px;
    margin: 0.6rem 0;
    line-height: 1.6;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

.user-bubble {
    background: rgba(102,126,234,0.18);
    border-left: 4px solid #667eea;
    color: #e0e0ff;
}
.bot-bubble {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #a78bfa;
    color: #f0f0f0;
}

.quick-btn button {
    background: rgba(102,126,234,0.15) !important;
    border: 1px solid rgba(102,126,234,0.4) !important;
    color: #c7d2fe !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s !important;
}
.quick-btn button:hover {
    background: rgba(102,126,234,0.35) !important;
    border-color: #667eea !important;
}

div.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
div.stTextInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.25) !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.6) !important;
}

.stSidebar { background: rgba(15,12,41,0.9) !important; }
.topic-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #c7d2fe;
    line-height: 1.8;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>Intelligent HR policy assistant · Powered by RAG + Groq</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Status")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        st.markdown('<span class="status-badge">✅ API Connected</span>', unsafe_allow_html=True)
    else:
        api_key_input = st.text_input("Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

    langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if langsmith_key:
        st.markdown('<span class="status-badge">✅ Tracing On</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Topics I Cover")
    st.markdown("""
<div class="topic-card">
• 🏖️ Leave policies (EL, SL, Maternity)<br>
• 💰 Compensation & CTC grades<br>
• 🏥 Health insurance & benefits<br>
• 📊 Performance reviews & PIP<br>
• 🏠 Work from home policy<br>
• 📋 Code of conduct<br>
• 🚀 Recruitment process<br>
• 📈 ESOP & stock options<br>
• ✈️ Travel & expense policy
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─── Pipeline (cached) ────────────────────────────────────────────────────────
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "zyro-dynamics-hr-corpus")

@st.cache_resource(show_spinner="🔍 Loading HR documents and building knowledge base...")
def init_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH, glob="**/*.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# ─── Prompts ──────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert HR assistant for Acrux Dynamics (also known as Zyro Dynamics).
Answer employee questions precisely and helpfully, based ONLY on the HR policy documents provided.
- Quote specific numbers, dates, and policy terms exactly as written.
- Structure your response clearly with bullet points when listing multiple items.
- If information is not found in the context, say: "I could not find this information in the HR policy documents."

Context from HR documents:
{context}"""),
    ("human", "{question}"),
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict classifier. Determine if a question is IN_SCOPE or OUT_OF_SCOPE for an HR chatbot.

IN_SCOPE: Leave, salary, benefits, health insurance, PIP, performance reviews, WFH, code of conduct, recruitment, ESOP, travel expenses, onboarding, separation — all specific to Acrux Dynamics / Zyro Dynamics.

OUT_OF_SCOPE: Company financials/revenue, product features (AcruxCRM etc.), competitor policies, general knowledge, anything unrelated to HR topics at this company.

Reply ONLY with: IN_SCOPE or OUT_OF_SCOPE"""),
    ("human", "Question: {question}"),
])

REFUSAL_MESSAGE = (
    "I'm sorry, I can only assist with questions about **Acrux Dynamics HR policies** — "
    "such as leave entitlements, compensation, benefits, performance management, and workplace guidelines.\n\n"
    "For questions about company financials, product details, or other organizations' policies, "
    "please reach out to the appropriate department directly. 🙏"
)

@traceable(name="ask_bot")
def ask_bot(question: str, retriever) -> str:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=768)

    # Guardrail classification
    cls_response = llm.invoke(OOS_PROMPT.invoke({"question": question}))
    classification = StrOutputParser().invoke(cls_response).strip().upper()

    if "OUT_OF_SCOPE" in classification:
        return REFUSAL_MESSAGE

    # RAG answer
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[{os.path.basename(doc.metadata.get('source','unknown'))} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )
    prompt_val = RAG_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(prompt_val)
    return StrOutputParser().invoke(response)

# ─── Chat State ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your Zyro Dynamics HR assistant. Ask me anything about our HR policies — leave, compensation, benefits, performance management, and more!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-bubble">🤖 <b>HR Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

# ─── Quick questions ───────────────────────────────────────────────────────────
st.markdown("**💡 Quick Questions:**")
quick_qs = [
    "How many earned leaves do I get per year?",
    "What is the salary credit date?",
    "Can I work from home?",
    "When is the annual performance review?",
    "What health insurance is provided?",
    "How does the PIP process work?",
]
cols = st.columns(3)
for i, q in enumerate(quick_qs):
    with st.container():
        cols[i % 3].markdown('<div class="quick-btn">', unsafe_allow_html=True)
        if cols[i % 3].button(q, key=f"qbtn_{i}"):
            st.session_state._pending = q
        cols[i % 3].markdown('</div>', unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    user_input = col1.text_input(
        "Your question", label_visibility="collapsed",
        placeholder="Ask about leave, salary, benefits, WFH policy..."
    )
    send = col2.form_submit_button("Send ➤")

# Determine question to process
question = None
if send and user_input.strip():
    question = user_input.strip()
elif hasattr(st.session_state, "_pending") and st.session_state._pending:
    question = st.session_state._pending
    st.session_state._pending = None

# Process question
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="chat-bubble user-bubble">👤 <b>You:</b> {question}</div>', unsafe_allow_html=True)

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("🔍 Searching HR policies..."):
            try:
                retriever = init_pipeline()
                answer = ask_bot(question, retriever)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(f'<div class="chat-bubble bot-bubble">🤖 <b>HR Assistant:</b><br>{answer}</div>', unsafe_allow_html=True)
            except Exception as e:
                err = f"⚠️ Error: {str(e)}"
                st.error(err)
