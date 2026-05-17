"""
LLM Handler - Supports both Groq (cloud, free) and Ollama (local)
Auto-detects which to use based on GROQ_API_KEY environment variable.
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

CONTEXT_PROMPT_TEMPLATE = """You are HealthMate AI, an expert medical assistant.

Use ONLY the following extracted context from the patient's medical report to answer the question.
If the answer is not in the context, say "This information is not found in the uploaded report."

--- MEDICAL REPORT CONTEXT ---
{context}
--- END OF CONTEXT ---

Patient Question: {question}

Provide a clear, accurate, and empathetic response. Use bullet points for lists. Explain any medical terms in simple language."""

GENERAL_PROMPT_TEMPLATE = """You are HealthMate AI, a helpful medical assistant.
No report has been uploaded, so answer from general medical knowledge.
Always recommend consulting a qualified healthcare professional.

Question: {question}

Answer clearly and empathetically. Explain medical terms in simple language."""

SUMMARY_PROMPT_TEMPLATE = """You are HealthMate AI. Summarize the following medical report in a structured, easy-to-understand format.

Include:
- **Patient Overview** (if available)
- **Key Findings** (bullet points)
- **Abnormal Values** (clearly highlighted)
- **Diagnoses / Conditions** (if mentioned)
- **Medications / Treatment** (if mentioned)
- **Recommendations / Follow-up**

Medical Report:
{text}

Provide a clear, structured summary using markdown formatting."""


class LLMHandler:
    def __init__(self):
        self.model = None
        self.mode = None  # "groq" or "ollama"
        self._llm = None
        self._detect_mode()

    def _detect_mode(self):
        """Auto-detect whether to use Groq or Ollama."""
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key and groq_key.startswith("gsk_"):
            self.mode = "groq"
            self.model = "llama-3.3-70b-versatile"  # default Groq model
            logger.info("🟢 LLM mode: Groq (cloud, free)")
        else:
            self.mode = "ollama"
            self.model = "llama3"
            logger.info("🟠 LLM mode: Ollama (local)")

    def set_model(self, model: str):
        """Change the active model."""
        if model != self.model:
            self.model = model
            self._llm = None  # reset cached LLM

    def get_available_models(self) -> List[str]:
        """Get available models based on current mode."""
        if self.mode == "groq":
            return [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "qwen/qwen3-32b",
            ]
        else:
            # Ollama local models
            try:
                import requests
                resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    return models if models else []
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {e}")
            return []

    def is_groq_mode(self) -> bool:
        """Check if using Groq API (True) or Ollama (False)."""
        return self.mode == "groq"

    def _get_llm(self):
        """Lazy-load the LLM (Groq or Ollama)."""
        if self._llm is not None:
            return self._llm

        if self.mode == "groq":
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1024,
                    api_key=os.environ.get("GROQ_API_KEY"),
                )
                logger.info(f"Loaded Groq model: {self.model}")
            except ImportError:
                logger.error("langchain_groq not installed. Install: pip install langchain-groq")
                raise
        else:
            try:
                from langchain_ollama import OllamaLLM
                self._llm = OllamaLLM(
                    model=self.model,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.3,
                    num_predict=1024,
                )
                logger.info(f"Loaded Ollama model: {self.model}")
            except ImportError:
                from langchain_community.llms import Ollama
                self._llm = Ollama(
                    model=self.model,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.3,
                )

        return self._llm

    def _invoke(self, prompt: str) -> str:
        """Invoke LLM and handle both chat and completion model responses."""
        try:
            llm = self._get_llm()
            result = llm.invoke(prompt)
            # ChatGroq returns AIMessage; Ollama returns str
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return f"⚠️ Error: {str(e)}"

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using retrieved RAG context."""
        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        return self._invoke(prompt)

    def answer_general(self, question: str) -> str:
        """Answer general medical question without report context."""
        prompt = GENERAL_PROMPT_TEMPLATE.format(question=question)
        return self._invoke(prompt)

    def summarize(self, text: str, max_chars: int = 6000) -> str:
        """Summarize a medical report."""
        truncated = text[:max_chars] if len(text) > max_chars else text
        prompt = SUMMARY_PROMPT_TEMPLATE.format(text=truncated)
        return self._invoke(prompt)