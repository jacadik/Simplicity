"""
Parser Factory module for creating appropriate document parsers based on file type.
"""

import os
import logging
from typing import Dict, Type

from .base_parser import BaseDocumentParser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser

class DocumentParserFactory:
    """
    Factory class for creating appropriate document parsers based on file type.
    """
    def __init__(self):
        """Initialize the document parser factory."""
        self.parsers: Dict[str, Type[BaseDocumentParser]] = {
            '.pdf': PDFParser,
            '.docx': DOCXParser,
            '.doc': DOCXParser,  # Use the same parser for .doc files
        }
        self.logger = self._setup_logger('INFO')
    
    def get_parser(self, file_path: str, logging_level: str = 'INFO') -> BaseDocumentParser:
        """
        Get the appropriate parser for the given file path.
        
        Args:
            file_path: Path to the document file
            logging_level: Logging level for the parser
            
        Returns:
            Appropriate document parser instance
            
        Raises:
            ValueError: If the file type is not supported
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.parsers:
            parser_class = self.parsers[file_ext]
            self.logger.info(f"Creating parser for {file_ext} file: {parser_class.__name__}")
            return parser_class(logging_level)
        else:
            supported_types = ", ".join(self.parsers.keys())
            error_msg = f"Unsupported file type: {file_ext}. Supported types: {supported_types}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def register_parser(self, file_extension: str, parser_class: Type[BaseDocumentParser]) -> None:
        """
        Register a new parser for a specific file extension.
        
        Args:
            file_extension: File extension (including the dot)
            parser_class: Parser class to use for this file extension
        """
        self.parsers[file_extension.lower()] = parser_class
        self.logger.info(f"Registered parser {parser_class.__name__} for file extension {file_extension}")
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.DocumentParserFactory')
        
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
