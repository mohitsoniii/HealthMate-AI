# ⚕️ HealthMate AI — Healthcare Assistant

> A production-ready GenAI healthcare assistant built with Python, Streamlit, Groq, LangChain, FAISS, and Sentence Transformers.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![Groq](https://img.shields.io/badge/Groq-Cloud_LLM-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Features

| Feature | Technology |
|---|---|
| 📄 PDF Medical Report Upload | PyMuPDF / pdfplumber / pypdf |
| 🧠 Semantic Search (RAG) | FAISS + Sentence Transformers |
| 🤖 Context-Aware Q&A | LangChain + Groq (Llama 3, Mixtral, etc.) |
| 📋 Auto Report Summary | Structured LLM Summarization |
| ⚡ Cloud-Powered AI | Groq ultra-fast inference (free tier available) |
| 💬 Multi-turn Chat | Streamlit Session State |
| 🔄 Ollama Fallback | Optional local inference via Ollama |

---

## 🏗️ Architecture

```
User Upload (PDF)
      │
      ▼
┌─────────────────┐
│  PDF Processor  │ ──► PyMuPDF / pdfplumber / pypdf
└────────┬────────┘
         │ raw text
         ▼
┌─────────────────────────┐
│  RAG Pipeline           │
│  ┌─────────────────┐   │
│  │ Text Chunker    │   │ ──► Sliding window (400 words, 80 overlap)
│  └────────┬────────┘   │
│           │ chunks      │
│  ┌────────▼────────┐   │
│  │ Sentence Trans. │   │ ──► all-MiniLM-L6-v2 embeddings
│  └────────┬────────┘   │
│           │ vectors     │
│  ┌────────▼────────┐   │
│  │ FAISS Index     │   │ ──► IndexFlatIP (cosine similarity)
│  └─────────────────┘   │
└─────────────────────────┘
         │
         │ User Query
         ▼
┌─────────────────┐
│  Retriever      │ ──► Top-k similar chunks
└────────┬────────┘
         │ context + query
         ▼
┌─────────────────┐
│  LLM Handler    │ ──► Groq API (primary) / Ollama (fallback)
└────────┬────────┘
         │ answer
         ▼
┌─────────────────┐
│  Streamlit UI   │ ──► Chat interface with source display
└─────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/healthmate-ai.git
cd healthmate-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Copy the secrets template and add your Groq API key:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and replace `your-groq-api-key-here` with your actual key.

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### (Optional) Local Ollama Fallback

If you prefer local inference or want a fallback, install [Ollama](https://ollama.ai) and pull a model:
```bash
ollama serve
ollama pull llama3
```
The app will automatically detect and offer Ollama models if available.

---

## 📁 Project Structure

```
healthmate-ai/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore
├── .streamlit/
│   ├── config.toml           # Streamlit theme config
│   └── secrets.toml.example  # API key template
├── assets/
│   └── style.css             # Custom UI styling
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF text extraction (multi-strategy)
│   ├── rag_pipeline.py       # FAISS + Sentence Transformers RAG
│   ├── llm_handler.py        # Groq / Ollama LLM integration
│   └── report_analyzer.py    # Report summarization logic
├── models/
│   └── all-MiniLM-L6-v2/    # (gitignored) local embedding model
└── data/
    └── sample_reports/       # (gitignored) sample PDFs for testing
```

---

## 🔧 Configuration

Edit `.streamlit/config.toml` to change theme colors.

Change the embedding model in `utils/rag_pipeline.py`:
```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # fast, lightweight
# or
model_name = "sentence-transformers/all-mpnet-base-v2"  # more accurate, slower
```

---

## 🛡️ Disclaimer

HealthMate AI is for **informational and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 🧑‍💻 Tech Stack

- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain
- **Cloud LLM**: Groq API (Llama 3, Mixtral, Gemma 2, etc.)
- **Local LLM (optional)**: Ollama
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (CPU)
- **PDF Parsing**: PyMuPDF, pdfplumber, pypdf (cascading fallback)
- **Language**: Python 3.10+

---

*Built with ❤️ as a portfolio project demonstrating production-grade GenAI engineering.*
