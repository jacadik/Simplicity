import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import docx
import pdfplumber
from dataclasses import dataclass

@dataclass
class Paragraph:
    """Class for storing paragraph information."""
    content: str
    doc_id: int
    paragraph_type: str  # 'normal', 'header', 'list', 'table', 'footer', etc.
    position: int
    header_content: Optional[str] = None

class DocumentParser:
    """
    Handles parsing of different document types (PDF, DOCX) and coordinates 
    extraction of paragraphs using appropriate methods.
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the document parser."""
        self.logger = self._setup_logger(logging_level)
        self.paragraph_extractor = ParagraphExtractor(logging_level)
    
    def parse_document(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """Parse a document and extract paragraphs."""
        self.logger.info(f"Starting to parse document: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._parse_pdf(file_path, doc_id)
            elif file_ext in ['.docx', '.doc']:
                return self._parse_docx(file_path, doc_id)
            else:
                self.logger.error(f"Unsupported file type: {file_ext}")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing document {file_path}: {str(e)}", exc_info=True)
            return []
    
    def _parse_pdf(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """Parse PDF document and extract paragraphs."""
        self.logger.info(f"Parsing PDF document: {file_path}")
        paragraphs = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract text from PDF with layout information
                raw_paragraphs = []
                for i, page in enumerate(pdf.pages):
                    self.logger.debug(f"Processing page {i+1}/{len(pdf.pages)}")
                    
                    # Extract text, tables, and layout info
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    # Process tables
                    if tables:
                        for table in tables:
                            table_content = self._format_table(table)
                            raw_paragraphs.append({
                                'content': table_content,
                                'type': 'table',
                                'page': i+1
                            })
                    
                    # Process text (excluding tables)
                    if text:
                        # Use text layout to separate paragraphs
                        page_paragraphs = self._extract_paragraphs_from_pdf_text(text)
                        for para in page_paragraphs:
                            raw_paragraphs.append({
                                'content': para,
                                'type': 'unknown',  # Will classify later
                                'page': i+1
                            })
                
                # Process extracted content with paragraph extractor
                paragraphs = self.paragraph_extractor.process_raw_paragraphs(raw_paragraphs, doc_id)
                
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}", exc_info=True)
        
        return paragraphs
    
    def _parse_docx(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """Parse DOCX document and extract paragraphs."""
        self.logger.info(f"Parsing DOCX document: {file_path}")
        paragraphs = []
        
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
            
        except Exception as e:
            self.logger.error(f"Error parsing DOCX {file_path}: {str(e)}", exc_info=True)
        
        return paragraphs
    
    def _extract_paragraphs_from_pdf_text(self, text: str) -> List[str]:
        """Extract paragraphs from PDF text using heuristics."""
        # Split by double newlines to separate paragraphs
        parts = re.split(r'\n\s*\n', text)
        
        # Clean up whitespace
        return [p.strip() for p in parts if p.strip()]
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format a pdfplumber table into a string representation."""
        # Filter out empty/None cells and replace with empty string
        formatted_table = [[cell if cell else '' for cell in row] for row in table]
        
        # Convert to string representation
        result = []
        for row in formatted_table:
            result.append(" | ".join(str(cell).strip() for cell in row))
        
        return "\n".join(result)
    
    def _extract_table_from_docx(self, table) -> str:
        """Extract content from a docx table."""
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cells.append(cell.text.strip())
            rows.append(" | ".join(cells))
        
        return "\n".join(rows)
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.DocumentParser')
        
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
        
        return logger


class ParagraphExtractor:
    """
    Handles the extraction and classification of paragraphs from document content.
    Implements multiple strategies for paragraph detection.
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the paragraph extractor."""
        self.logger = self._setup_logger(logging_level)
    
    def process_raw_paragraphs(self, raw_paragraphs: List[Dict], doc_id: int) -> List[Paragraph]:
        """Process raw paragraphs and return structured Paragraph objects."""
        self.logger.info(f"Processing {len(raw_paragraphs)} raw paragraphs for document {doc_id}")
        
        # 1. Classify paragraph types
        classified_paragraphs = self._classify_paragraphs(raw_paragraphs)
        
        # 2. Combine lists
        combined_lists = self._combine_lists(classified_paragraphs)
        
        # 3. Associate headers with paragraphs
        with_headers = self._associate_headers(combined_lists)
        
        # 4. Create final paragraph objects
        paragraphs = []
        for i, para in enumerate(with_headers):
            header_content = para.get('header_content')
            
            # Skip if this is an address
            if para['type'] == 'address':
                self.logger.debug(f"Skipping address paragraph: {para['content'][:30]}...")
                continue
                
            paragraph = Paragraph(
                content=para['content'],
                doc_id=doc_id,
                paragraph_type=para['type'],
                position=i,
                header_content=header_content
            )
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _classify_paragraphs(self, raw_paragraphs: List[Dict]) -> List[Dict]:
        """Classify paragraphs into different types."""
        classified = []
        
        for para in raw_paragraphs:
            para_type = para.get('type', 'unknown')
            content = para['content']
            
            # Skip empty paragraphs
            if not content.strip():
                continue
            
            # If already classified as table, keep it
            if para_type == 'table':
                classified.append(para)
                continue
            
            # Check if paragraph is a header
            if self._is_header(content, para):
                para['type'] = 'header'
                classified.append(para)
                continue
            
            # Check if paragraph is a list
            if self._is_list(content):
                para['type'] = 'list'
                classified.append(para)
                continue
            
            # Check if paragraph is an address
            if self._is_address(content):
                para['type'] = 'address'
                classified.append(para)
                continue
            
            # Check if paragraph is a footer/disclaimer
            if self._is_boilerplate(content):
                para['type'] = 'boilerplate'
                classified.append(para)
                continue
            
            # Default to normal paragraph
            para['type'] = 'normal'
            classified.append(para)
        
        return classified
    
    def _combine_lists(self, paragraphs: List[Dict]) -> List[Dict]:
        """Combine consecutive list items into a single paragraph."""
        if not paragraphs:
            return []
            
        combined = []
        current_list = None
        
        for para in paragraphs:
            if para['type'] == 'list':
                if current_list is None:
                    current_list = {
                        'content': para['content'],
                        'type': 'list',
                        'position': para.get('position', 0)
                    }
                else:
                    current_list['content'] += '\n' + para['content']
            else:
                if current_list is not None:
                    combined.append(current_list)
                    current_list = None
                combined.append(para)
        
        # Add the last list if there is one
        if current_list is not None:
            combined.append(current_list)
        
        return combined
    
    def _associate_headers(self, paragraphs: List[Dict]) -> List[Dict]:
        """Associate headers with their following paragraphs."""
        if not paragraphs:
            return []
            
        result = []
        last_header = None
        
        for para in paragraphs:
            if para['type'] == 'header':
                last_header = para['content']
                # Add the header as a standalone paragraph too
                result.append(para)
            else:
                if last_header is not None:
                    para['header_content'] = last_header
                    last_header = None  # Only associate with the immediately following paragraph
                result.append(para)
        
        return result
    
    def _is_header(self, content: str, para_info: Dict) -> bool:
        """Determine if a paragraph is a header."""
        # Check if style indicates a header (for DOCX)
        if 'style' in para_info and 'head' in para_info['style'].lower():
            return True
            
        # For other cases, use heuristics
        # Headers are typically short
        if len(content) > 100:
            return False
            
        # Headers often end without punctuation
        if not content[-1] in '.?!:;,':
            # Check for common header patterns
            if re.match(r'^[\d\.]+\s+\w+', content):  # Numbered header (e.g., "1.2 Introduction")
                return True
            if content.isupper():  # ALL CAPS headers
                return True
            if len(content.split()) <= 10:  # Short text without punctuation is likely a header
                return True
        
        return False
    
    def _is_list(self, content: str) -> bool:
        """Determine if a paragraph is a list item."""
        # Check for bullet points
        if re.match(r'^\s*[•\-\*\+○◦➢➣➤►▶→➥➔❖]\s+', content):
            return True
            
        # Check for numbered lists
        if re.match(r'^\s*(\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+)', content):
            return True
            
        return False
    
    def _is_address(self, content: str) -> bool:
        """Determine if a paragraph is an address."""
        # Check for postal code patterns
        postal_pattern = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', content)  # UK
        zip_pattern = re.search(r'\b\d{5}(-\d{4})?\b', content)  # US
        
        # Check for typical address components
        address_indicators = ['street', 'avenue', 'road', 'lane', 'drive', 'boulevard', 'st.', 'ave.', 'rd.']
        has_indicator = any(indicator in content.lower() for indicator in address_indicators)
        
        # If we find postal codes or address indicators, it's likely an address
        if (postal_pattern or zip_pattern) and has_indicator:
            return True
            
        return False
    
    def _is_boilerplate(self, content: str) -> bool:
        """Determine if a paragraph is boilerplate text (footer, disclaimer, etc.)."""
        # Check for common boilerplate indicators
        boilerplate_indicators = [
            'all rights reserved', 'copyright', '©', 'confidential', 
            'disclaimer', 'terms and conditions', 'privacy policy',
            'legal notice', 'proprietary'
        ]
        
        if any(indicator in content.lower() for indicator in boilerplate_indicators):
            return True
            
        # Check for typical footer patterns (page numbers, etc.)
        if re.match(r'^\s*Page \d+ of \d+\s*$', content):
            return True
            
        return False
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.ParagraphExtractor')
        
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
        
        return logger
