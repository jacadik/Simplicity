"""
Document Parser Package
Main interface for document parsing functionality
"""

from .paragraph import Paragraph
from .paragraph_extractor import ParagraphExtractor
from .parser_factory import DocumentParserFactory

# Create a factory instance for convenient access
factory = DocumentParserFactory()

# For backward compatibility
class DocumentParser:
    """
    Backward compatibility class that delegates to the appropriate parser.
    This maintains the same interface as the original DocumentParser class.
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the document parser."""
        self.factory = DocumentParserFactory()
        self.paragraph_extractor = ParagraphExtractor(logging_level)
        self.logging_level = logging_level
    
    def parse_document(self, file_path: str, doc_id: int) -> list[Paragraph]:
        """
        Parse a document and extract paragraphs.
        
        Args:
            file_path: Path to the document file
            doc_id: ID of the document in the database
            
        Returns:
            List of extracted paragraphs
        """
        # Get the appropriate parser based on file extension
        parser = self.factory.get_parser(file_path, self.logging_level)
        
        # Parse the document using the appropriate parser
        return parser.parse_document(file_path, doc_id)
