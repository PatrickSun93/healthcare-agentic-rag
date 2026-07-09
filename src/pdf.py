"""PDF -> transcript, for both real (text-layer) and scanned/image PDFs.

Per page: pull the text layer via PyMuPDF; if a page has almost no text (a scan/photo), render it to
an image and OCR it. Output uses '## Page N' headers so downstream chunking yields page-level citations.

This mirrors the OCR order-intake work at Fulgent, with the same discipline: a page with no extractable
text isn't silently trusted — it's routed to OCR (and low-confidence output should go to human review).

OCR backend (config.OCR_BACKEND):
  - tesseract : pytesseract + Pillow  (needs the binary: `brew install tesseract`)
  - ollama    : a local vision model via Ollama (e.g. llama3.2-vision) — stays on-box (PHI-safe)
  - none      : text layer only, skip OCR
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import fitz  # PyMuPDF (import name is `fitz`)

from . import config as C


def _ocr_image(png_bytes: bytes) -> str:
    if C.OCR_BACKEND == "none":
        return ""

    if C.OCR_BACKEND == "tesseract":
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "tesseract OCR needs: pip install pytesseract pillow  +  `brew install tesseract`"
            ) from e
        return pytesseract.image_to_string(Image.open(io.BytesIO(png_bytes))).strip()

    if C.OCR_BACKEND == "ollama":
        import httpx

        b64 = base64.b64encode(png_bytes).decode()
        with httpx.Client(timeout=600) as client:
            r = client.post(
                f"{C.OLLAMA_BASE}/api/generate",
                json={
                    "model": C.OCR_VISION_MODEL,
                    "prompt": "Transcribe ALL text in this document image verbatim. Output only the text.",
                    "images": [b64],
                    "stream": False,
                },
            )
            r.raise_for_status()
            return r.json()["response"].strip()

    raise ValueError(f"Unknown OCR_BACKEND: {C.OCR_BACKEND}")


def pdf_to_text(path: str | Path) -> str:
    """Page-delimited transcript. Uses the text layer, falling back to OCR on scanned pages."""
    doc = fitz.open(str(path))
    pages: list[str] = []
    ocr_pages: list[int] = []
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if len(text) < C.OCR_MIN_CHARS:  # little/no text layer => likely a scanned image page
            pix = page.get_pixmap(dpi=C.OCR_DPI)
            text = _ocr_image(pix.tobytes("png"))
            if text:
                ocr_pages.append(i)
        pages.append(f"## Page {i}\n{text}".rstrip())
    doc.close()
    if ocr_pages:
        print(f"  OCR'd page(s) {ocr_pages} via {C.OCR_BACKEND}")
    return "\n\n".join(pages) + "\n"


def pdf_to_md_file(path: str | Path, out_dir: str | Path) -> Path:
    """Convert one PDF to a .md transcript saved in out_dir; returns the output path."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".md")
    transcript = pdf_to_text(path)
    out_path.write_text(f"# {path.stem} (transcribed from {path.name})\n\n{transcript}")
    return out_path
