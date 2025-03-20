"""
DOCX Parser module for extracting paragraphs from Word documents.
"""

import time
import logging
from typing import List, Dict

import docx

from .base_parser import BaseDocumentParser
from .paragraph import Paragraph

class DOCXParser(BaseDocumentParser):
    """DOCX document parser that extracts paragraphs from Word files."""
    
    def parse_document(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """
        Parse DOCX document and extract paragraphs.
        
        Args:
            file_path: Path to the DOCX file
            doc_id: ID of the document in the database
            
        Returns:
            List of extracted paragraphs
        """
        self.logger.info(f"Parsing DOCX document: {file_path}")
        paragraphs = []
        
        start_time = time.time()
        try:
            doc = docx.Document(file_path)
            
            # Extract paragraphs, tables and lists
            raw_paragraphs = []
            position = 0
            
            # Process document elements
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
                    # Convert element to paragraph object
                    p = docx.text.paragraph.Paragraph(element, doc)
                    text = p.text.strip()
                    
                    if text:
                        raw_paragraphs.append({
                            'content': text,
                            'type': 'unknown',  # Will classify later
                            'style': p.style.name if hasattr(p, 'style') and p.style else 'Normal',
                            'position': position
                        })
                        position += 1
                        
                elif element.tag.endswith('tbl'):  # Table
                    table = docx.table.Table(element, doc)
                    table_content = self._extract_table_from_docx(table)
                    
                    if table_content:
                        raw_paragraphs.append({
                            'content': table_content,
                            'type': 'table',
                            'position': position
                        })
                        position += 1
            
            # Process extracted content with paragraph extractor
            paragraphs = self.paragraph_extractor.process_raw_paragraphs(raw_paragraphs, doc_id)
            
            # Check if we got reasonable content
            if len(paragraphs) == 0 and len(doc.paragraphs) > 0:
                self.logger.warning(f"No paragraphs extracted from {file_path} despite having content. Trying fallback method.")
                
                # Fallback to simpler extraction
                simple_paragraphs = []
                for i, para in enumerate(doc.paragraphs):
                    text = para.text.strip()
                    if text:
                        simple_paragraphs.append({
                            'content': text,
                            'type': 'unknown',
                            'style': para.style.name if para.style else 'Normal',
                            'position': i
                        })
                
                if simple_paragraphs:
                    paragraphs = self.paragraph_extractor.process_raw_paragraphs(simple_paragraphs, doc_id)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Extracted {len(paragraphs)} paragraphs from DOCX document in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error parsing DOCX {file_path}: {str(e)}", exc_info=True)
        
        return paragraphs
    
    def _extract_table_from_docx(self, table) -> str:
        """
        Extract content from a docx table.
        
        Args:
            table: DOCX table object
            
        Returns:
            Formatted table string
        """
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cells.append(cell.text.strip())
            rows.append(" | ".join(cells))
        
        return "\n".join(rows)
