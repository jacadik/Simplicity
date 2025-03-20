"""
Base parser module defining the abstract interface for document parsers.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List

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
