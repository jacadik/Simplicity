"""
Base parser module defining the abstract interface for document parsers.
"""

import os
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict

from .paragraph import Paragraph
from .paragraph_extractor import ParagraphExtractor

class BaseDocumentParser(ABC):
    """
    Abstract base class for document parsers.
    Defines the interface that all concrete parsers must implement.
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the base document parser."""
        self.logger = self._setup_logger(logging_level)
        self.paragraph_extractor = ParagraphExtractor(logging_level)
    
    @abstractmethod
    def parse_document(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """
        Parse a document and extract paragraphs.
        
        Args:
            file_path: Path to the document file
            doc_id: ID of the document in the database
            
        Returns:
            List of extracted paragraphs
        """
        pass
    
    def _is_list_item(self, text: str) -> bool:
        """
        Check if a line of text is a list item.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is a list item
        """
        text = text.strip()
        if not text:
            return False
            
        # Check for bullet points
        bullet_match = re.match(r'^\s*[•\-\*\+○◦➢➣➤►▶→➥➔❖]\s+', text)
        if bullet_match:
            return True
            
        # Check for numbered list patterns
        number_match = re.match(r'^\s*(\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+|[ivxIVX]+[\.\)]\s+)', text)
        if number_match:
            return True
            
        return False
    
    def _extract_list_items(self, text: str) -> List[str]:
        """
        Extract individual list items from text.
        
        Args:
            text: Text to extract items from
            
        Returns:
            List of extracted items
        """
        items = []
        current_item = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if self._is_list_item(line):
                if current_item:
                    items.append(current_item)
                current_item = line
            elif current_item:
                # Continuation of previous item
                current_item += ' ' + line
            else:
                # Not part of a list
                current_item = line
        
        if current_item:
            items.append(current_item)
            
        return items
    
    def _format_table(self, rows: List[List[str]]) -> str:
        """
        Format table rows as text.
        
        Args:
            rows: Table rows
            
        Returns:
            Formatted table string
        """
        if not rows:
            return ""
        
        # Clean up rows
        clean_rows = []
        for row in rows:
            clean_row = [str(cell).strip() if cell is not None else '' for cell in row]
            clean_rows.append(clean_row)
        
        # Format as string
        result = []
        for row in clean_rows:
            result.append(" | ".join(row))
        
        return "\n".join(result)
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger_name = f'{__name__}.{self.__class__.__name__}'
        logger = logging.getLogger(logger_name)
        
        if level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif level.upper() == 'INFO':
            logger.setLevel(logging.INFO)
        elif level.upper() == 'WARNING':
            logger.setLevel(logging.WARNING)
        elif level.upper() == 'ERROR':
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger