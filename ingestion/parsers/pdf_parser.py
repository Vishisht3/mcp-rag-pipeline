"""
ingestion/parsers/pdf_parser.py
Parser for PDF documents using pypdf with optional OCR fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("pdf",)

    def __init__(self, ocr_fallback: bool = False):
        self.ocr_fallback = ocr_fallback

    def _ocr_page(self, page) -> str:
        """Attempt OCR extraction from embedded page images when ocr_fallback is enabled."""
        ocr_texts = []
        images = getattr(page, "images", None) or []
        for img in images:
            data = getattr(img, "data", None)
            if not data:
                continue
            try:
                from rapidocr_onnxruntime import RapidOCR

                engine = RapidOCR()
                result, _ = engine(data)
                if result:
                    ocr_texts.append(" ".join(line[1] for line in result if len(line) > 1))
                continue
            except Exception:
                pass
            try:
                import io
                import pytesseract
                from PIL import Image

                pil_img = Image.open(io.BytesIO(data))
                text = pytesseract.image_to_string(pil_img)
                if text and text.strip():
                    ocr_texts.append(text.strip())
            except Exception:
                pass
        return "\n".join(ocr_texts).strip()

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package is required for PDF parsing. Install it via 'pip install pypdf'."
            )

        reader = pypdf.PdfReader(str(path))
        page_texts = []
        ocr_used = False
        for page in reader.pages:
            t = page.extract_text() or ""
            if not t.strip() and self.ocr_fallback:
                ocr_t = self._ocr_page(page)
                if ocr_t:
                    t = ocr_t
                    ocr_used = True
                else:
                    ocr_used = True
            page_texts.append(t)

        full_text = "\n\n".join(page_texts).strip()
        page_count = len(reader.pages)

        metadata = {
            "format": "pdf",
            "page_count": page_count,
        }
        if self.ocr_fallback:
            metadata["ocr_fallback_enabled"] = True
            metadata["ocr_attempted"] = ocr_used

        return ParsedDocument(
            text=full_text,
            source=path.name,
            metadata=metadata,
        )
