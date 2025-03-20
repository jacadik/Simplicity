"""
Paragraph Extractor module for classifying and processing raw paragraphs.
"""

import re
import logging
from typing import List, Dict, Set

from .paragraph import Paragraph

class ParagraphExtractor:
    """
    Handles the extraction and classification of paragraphs from document content.
    Implements multiple strategies for paragraph detection.
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the paragraph extractor."""
        self.logger = self._setup_logger(logging_level)
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> Set[str]:
        """Get a set of common stopwords for text processing."""
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'this', 'that', 'these', 'those', 'of', 'up', 'down'
        }
        return stopwords
    
    def process_raw_paragraphs(self, raw_paragraphs: List[Dict], doc_id: int) -> List[Paragraph]:
        """
        Process raw paragraphs and return structured Paragraph objects.
        
        Args:
            raw_paragraphs: List of raw paragraph data
            doc_id: Document ID
            
        Returns:
            List of structured Paragraph objects
        """
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
            
            # Skip if this is an address or empty paragraph
            if para['type'] == 'address' or not para['content'].strip():
                self.logger.debug(f"Skipping {para['type']} paragraph: {para['content'][:30]}...")
                continue
                
            paragraph = Paragraph(
                content=para['content'],
                doc_id=doc_id,
                paragraph_type=para['type'],
                position=i,
                header_content=header_content
            )
            paragraphs.append(paragraph)
        
        self.logger.info(f"Created {len(paragraphs)} structured paragraphs")
        return paragraphs
    
    def _classify_paragraphs(self, raw_paragraphs: List[Dict]) -> List[Dict]:
        """
        Classify paragraphs into different types.
        
        Args:
            raw_paragraphs: List of raw paragraph data
            
        Returns:
            List of classified paragraphs
        """
        classified = []
        
        for para in raw_paragraphs:
            para_type = para.get('type', 'unknown')
            content = para.get('content', '').strip()
            
            # Skip empty paragraphs
            if not content:
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
        """
        Combine consecutive list items into a single paragraph.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            List of paragraphs with combined lists
        """
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
        """
        Associate headers with their following paragraphs.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            List of paragraphs with header associations
        """
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
        """
        Determine if a paragraph is a header.
        
        Args:
            content: Paragraph content
            para_info: Additional paragraph information
            
        Returns:
            True if paragraph is a header, False otherwise
        """
        # Check if style indicates a header (for DOCX)
        if 'style' in para_info and any(term in para_info['style'].lower() for term in ['head', 'title', 'subtitle']):
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
            if content.istitle() and len(content.split()) <= 10:  # Title Case headers
                return True
            if len(content.split()) <= 6:  # Short text without punctuation is likely a header
                return True
        
        return False
    
    def _is_list(self, content: str) -> bool:
        """
        Determine if a paragraph is a list item.
        
        Args:
            content: Paragraph content
            
        Returns:
            True if paragraph is a list item, False otherwise
        """
        # Check for bullet points
        if re.match(r'^\s*[•\-\*\+○◦➢➣➤►▶→➥➔❖]\s+', content):
            return True
            
        # Check for numbered lists
        if re.match(r'^\s*(\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+)', content):
            return True
            
        return False
    
    def _is_address(self, content: str) -> bool:
        """
        Determine if a paragraph is an address.
        
        Args:
            content: Paragraph content
            
        Returns:
            True if paragraph is an address, False otherwise
        """
        # Check for postal code patterns
        postal_pattern = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', content)  # UK
        zip_pattern = re.search(r'\b\d{5}(-\d{4})?\b', content)  # US
        
        # Check for typical address components
        address_indicators = [
            'street', 'avenue', 'road', 'lane', 'drive', 'boulevard', 
            'st.', 'ave.', 'rd.', 'ln.', 'dr.', 'blvd.', 'suite', 'apt', 
            'apartment', 'floor', 'unit'
        ]
        has_indicator = any(indicator in content.lower() for indicator in address_indicators)
        
        # If we find postal codes or address indicators, it's likely an address
        if (postal_pattern or zip_pattern) and has_indicator:
            return True
            
        return False
    
    def _is_boilerplate(self, content: str) -> bool:
        """
        Determine if a paragraph is boilerplate text (footer, disclaimer, etc.).
        
        Args:
            content: Paragraph content
            
        Returns:
            True if paragraph is boilerplate text, False otherwise
        """
        # Check for common boilerplate indicators
        boilerplate_indicators = [
            'all rights reserved', 'copyright', '©', 'confidential', 
            'disclaimer', 'terms and conditions', 'privacy policy',
            'legal notice', 'proprietary', 'confidentiality notice',
            'do not copy', 'not for distribution', 'all rights reserved'
        ]
        
        if any(indicator in content.lower() for indicator in boilerplate_indicators):
            return True
            
        # Check for typical footer patterns (page numbers, etc.)
        if re.match(r'^\s*Page \d+ of \d+\s*$', content):
            return True
            
        return False
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """
        Set up a logger instance.
        
        Args:
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f'{__name__}.ParagraphExtractor')
        
        if not logger.handlers:  # Only add handlers if they don't exist
            # Set level
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
            
            # Create console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
