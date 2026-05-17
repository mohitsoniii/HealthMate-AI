"""
Report Analyzer - High-level analysis wrapper
"""
import logging
from utils.llm_handler import LLMHandler

logger = logging.getLogger(__name__)


class ReportAnalyzer:
    """Orchestrates medical report analysis tasks."""

    def summarize(self, text: str, llm_handler: LLMHandler) -> str:
        """Generate a structured summary of the medical report."""
        return llm_handler.summarize(text)

    def extract_key_values(self, text: str, llm_handler: LLMHandler) -> str:
        """Extract lab values and biomarkers from the report."""
        prompt = f"""Extract all laboratory values, test results, and biomarkers from this medical report.
Format as a markdown table with columns: Test | Value | Normal Range | Status (Normal/Abnormal/Critical).
If normal ranges are not in the report, use standard medical reference ranges.

Report:
{text[:4000]}"""
        return llm_handler.answer_general(prompt)
