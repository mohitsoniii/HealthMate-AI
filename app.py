"""
HealthMate AI - Healthcare Assistant
A production-ready GenAI healthcare assistant using RAG pipeline
"""

import streamlit as st
import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
from pathlib import Path

st.set_page_config(
    page_title="HealthMate AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

from utils.pdf_processor import extract_text_from_pdf
from utils.rag_pipeline import RAGPipeline
from utils.llm_handler import LLMHandler
from utils.report_analyzer import ReportAnalyzer

def init_session_state():
    defaults = {
        "messages": [],
        "rag_pipeline": None,
        "report_loaded": False,
        "report_summary": None,
        "report_name": None,
        "extracted_text": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

@st.cache_resource(show_spinner=False)
def get_llm_handler():
    return LLMHandler()

@st.cache_resource(show_spinner=False)
def get_report_analyzer():
    return ReportAnalyzer()

llm_handler = get_llm_handler()
report_analyzer = get_report_analyzer()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">⚕️</span>
        <div>
            <div class="logo-title">HealthMate AI</div>
            <div class="logo-sub">Medical Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    available_models = llm_handler.get_available_models()

    if llm_handler.is_groq_mode():
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=0,
            help="Free models via Groq API"
        )
        st.success("✅ Connected to Groq (Free Cloud)")
    elif not available_models:
        st.error("❌ Ollama not running.")
        st.code("ollama serve", language="bash")
        selected_model = "llama3"
    else:
        selected_model = st.selectbox(
            "Select LLM Model",
            options=available_models,
            index=0,
            help="Models available via Ollama"
        )
        st.success("✅ Connected to Ollama (Local)")

    llm_handler.set_model(selected_model)

    st.markdown("---")
    st.markdown("### 📂 Upload Medical Report")

    uploaded_file = st.file_uploader(
        "Upload PDF Report",
        type=["pdf"],
        help="Upload a medical report, lab result, or clinical document"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.report_name:
            with st.spinner("🔬 Processing document..."):
                extracted_text = extract_text_from_pdf(uploaded_file)

                if extracted_text and len(extracted_text.strip()) > 50:
                    st.session_state.extracted_text = extracted_text
                    st.session_state.report_name = uploaded_file.name

                    rag = RAGPipeline()
                    rag.build_index(extracted_text)
                    st.session_state.rag_pipeline = rag
                    st.session_state.report_loaded = True

                    with st.spinner("✨ Generating summary..."):
                        summary = report_analyzer.summarize(extracted_text, llm_handler)
                        st.session_state.report_summary = summary

                    st.success("✅ Report loaded!")
                    st.rerun()
                else:
                    st.error("Could not extract text. Ensure the PDF is not a scanned image.")

    if st.session_state.report_loaded:
        st.markdown(f"""
        <div class="report-badge">
            <span>📄</span>
            <span class="report-name">{st.session_state.report_name}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Report", use_container_width=True):
            for key in ["rag_pipeline", "report_summary", "report_name", "extracted_text", "messages"]:
                st.session_state[key] = [] if key == "messages" else None
            st.session_state.report_loaded = False
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "Summarize the key findings",
        "What are the abnormal values?",
        "Explain my diagnosis in simple terms",
        "What medications are prescribed?",
        "What follow-up is recommended?",
    ]
    for q in sample_questions:
        if st.button(f"→ {q}", use_container_width=True, key=f"sample_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-footer">
        <div>⚠️ <strong>Disclaimer</strong></div>
        <div>For informational purposes only. Not a substitute for professional medical advice.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <h1>⚕️ HealthMate <span class="ai-badge">AI</span></h1>
        <p>Your intelligent healthcare companion — powered by AI & RAG</p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.report_loaded and st.session_state.report_summary:
    with st.expander("📋 **AI-Generated Report Summary**", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(st.session_state.report_summary)
        with col2:
            st.markdown("### 📊 Report Stats")
            word_count = len(st.session_state.extracted_text.split())
            chunks = len(st.session_state.rag_pipeline.chunks) if st.session_state.rag_pipeline else 0
            st.metric("Words Extracted", f"{word_count:,}")
            st.metric("Knowledge Chunks", chunks)
            st.metric("Model", selected_model)

elif not st.session_state.report_loaded:
    st.markdown("""
    <div class="welcome-grid">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3>Medical Report Analysis</h3>
            <p>Upload any PDF medical report — lab results, discharge summaries, prescriptions — and get instant AI-powered insights.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>RAG-Powered Q&A</h3>
            <p>Ask questions in plain English. Our FAISS-based semantic search finds the most relevant context before answering.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Cloud-Powered AI</h3>
            <p>Powered by Groq's ultra-fast cloud inference — no local setup required. Just upload and ask.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💊</div>
            <h3>Plain Language Explanations</h3>
            <p>Complex medical jargon translated into clear, understandable language for patients and caregivers.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 💬 Medical Q&A Chat")

if not st.session_state.messages:
    if st.session_state.report_loaded:
        st.info("✨ Report loaded! Ask me anything about your medical report.")
    else:
        st.info("📂 Upload a medical PDF report using the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚕️"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Source Chunks Used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** _{src[:300]}..._")

if prompt := st.chat_input("Ask about your medical report... (e.g., 'What does my HbA1c level mean?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚕️"):
        with st.spinner("🔍 Searching knowledge base..."):
            if st.session_state.report_loaded and st.session_state.rag_pipeline:
                relevant_chunks = st.session_state.rag_pipeline.retrieve(prompt, k=4)
                context = "\n\n".join(relevant_chunks)
                response = llm_handler.answer_with_context(prompt, context)
                sources = relevant_chunks
            else:
                response = llm_handler.answer_general(prompt)
                sources = []

        st.markdown(response)
        if sources:
            with st.expander("📎 Source Chunks Used"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:** _{src[:300]}..._")

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
