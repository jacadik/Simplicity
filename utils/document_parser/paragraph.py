"""
Paragraph dataclass for storing paragraph information extracted from documents.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Paragraph:
    """Class for storing paragraph information."""
    content: str
    doc_id: int
    paragraph_type: str  # 'normal', 'header', 'list', 'table', 'footer', etc.
    position: int
    header_content: Optional[str] = None
    column: Optional[int] = None  # Optional column position
