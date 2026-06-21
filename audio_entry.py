from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AudioEntry:
    """Represents a single audio entry in the preprocessing pipeline."""
    path: str
    lab_text: str
    duration: float
    emotion: str = "neutral"
    stem: Optional[str] = None
    lab_path: Optional[str] = None
    origin: str = "source"  # "source", "split", or "merge"

    @property
    def absolute_path(self) -> str:
        return os.path.abspath(self.path)
