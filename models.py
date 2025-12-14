"""
Data Models for PDF-Translator

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple # Import Tuple

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
# ... (JobStatus remains unchanged)
    queued = "queued"
    analyzing = "analyzing"
    translating = "translating"
    latex_build = "latex_build"
    done = "done"
    error = "error"


class Block(BaseModel):
    """Datenstruktur für einen Textblock oder ein Strukturelement aus dem PDF."""
    page: int
    # Der Originaltext aus der PDF-Analyse
    content: Optional[str] = Field(None)
    # Der übersetzte/für LaTeX aufbereitete Text
    translated_latex: Optional[str] = Field(None)
    # Neu: Typ des Elements (z.B. "text", "figure_caption", "table_content")
    element_type: str = Field("text", description="Type of content: text, figure_caption, or table_content.")
    # Neu: Bounding Box (x0, y0, x1, y1) zur Platzierungshilfe
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="Bounding box (x0, y0, x1, y1) for layout reconstruction."
    )


class JobInfo(BaseModel):
# ... (JobInfo remains unchanged)
    """
    Statusobjekt für einen PDF-Übersetzungsjob.
    Wird über die API hin- und hergeschickt.
    """

    job_id: str = Field(..., description="Eindeutige Job-ID")
    status: JobStatus = Field(..., description="Aktueller Status des Jobs")

    # Fortschritt 0–100 %
    progress: int = Field(
        0,
        ge=0,
        le=100,
        description="Fortschritt in Prozent (0–100)",
    )

    # Freier Status-/Fehlermeldungstext
    message: Optional[str] = Field(
        None,
        description="Optionaler Status- oder Fehlermeldungstext",
    )

    # Zielsprachen/LLM-Optionen (passen zu create_job in jobs.py)
    target_language: str = Field(
        "de",
        description="Zielsprache (z.B. 'de', 'en')",
    )
    use_openai: bool = Field(
        False,
        description="Ob für diesen Job OpenAI zum Übersetzen verwendet wird",
    )
    openai_api_key_set: bool = Field(
        False,
        description="Ob für diesen Job ein API-Key hinterlegt wurde",
    )

    class Config:
        # Pydantic v2: statt orm_mode
        from_attributes = True
