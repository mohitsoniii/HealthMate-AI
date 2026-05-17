"""
PDF Processor - Extract text from uploaded PDF medical reports
"""
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    """
    Extract text content from a PDF file uploaded via Streamlit.
    Tries PyMuPDF (fitz) first, falls back to pdfplumber.
    """
    pdf_bytes = uploaded_file.read()

    # Strategy 1: PyMuPDF (fastest, best quality)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            logger.info(f"PyMuPDF extracted {len(full_text)} chars")
            return full_text
    except ImportError:
        logger.warning("PyMuPDF not available, trying pdfplumber")
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # Strategy 2: pdfplumber (good table support)
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            logger.info(f"pdfplumber extracted {len(full_text)} chars")
            return full_text
    except ImportError:
        logger.warning("pdfplumber not available")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # Strategy 3: pypdf (lightweight fallback)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            logger.info(f"pypdf extracted {len(full_text)} chars")
            return full_text
    except ImportError:
        logger.warning("pypdf not available")
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")

    logger.error("All PDF extraction methods failed")
    return None
