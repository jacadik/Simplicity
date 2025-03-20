"""
Paragraph Extractor module for classifying and processing raw paragraphs.
Enhanced with improved list handling, header association, and table detection.
"""

import re
import logging
from typing import List, Dict, Set, Optional

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
        
        # 2. Combine lists and tables
        combined = self._combine_structure_elements(classified_paragraphs)
        
        # 3. Associate headers with paragraphs
        with_headers = self._associate_headers(combined)
        
        # 4. Create final paragraph objects
        paragraphs = []
        for i, para in enumerate(with_headers):
            header_content = para.get('header_content')
            
            # Skip if this is an address or empty paragraph
            if para['type'] == 'address' or not para['content'].strip():
                self.logger.debug(f"Skipping {para['type']} paragraph: {para['content'][:30]}...")
                continue
            
            # Create paragraph object
            paragraph = Paragraph(
                content=para['content'],
                doc_id=doc_id,
                paragraph_type=para['type'],
                position=i,
                header_content=header_content,
                column=para.get('column')  # Include column position if available
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
            
            # Pass through already classified paragraphs
            if para_type != 'unknown':
                # Ensure column is preserved
                if 'column' in para:
                    classified.append(para)
                else:
                    classified.append(para)
                continue
            
            # Check if paragraph is a header
            if self._is_header(content, para):
                para['type'] = 'header'
                classified.append(para)
                continue
            
            # Check if paragraph is a table row
            if self._is_table_row(content):
                para['type'] = 'table'
                classified.append(para)
                continue
            
            # Check if paragraph is a list
            if self._is_list_item(content):
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
    
    def _paragraph_introduces_list(self, para: Dict) -> bool:
        """
        Check if a paragraph introduces a list (ends with a colon).
        
        Args:
            para: Paragraph dictionary
            
        Returns:
            True if paragraph introduces a list, False otherwise
        """
        content = para.get('content', '')
        return content.strip().endswith(':')
        
    def _combine_structure_elements(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        Combine consecutive structural elements (lists and tables) into single paragraphs
        and attach them to their introducing paragraphs.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            List of paragraphs with combined elements
        """
        if not paragraphs:
            return []
            
        combined = []
        current_element = None
        current_type = None
        
        for i, para in enumerate(paragraphs):
            para_type = para['type']
            
            # Handle structured elements (lists and tables)
            if para_type in ('list', 'table'):
                # Extract column if available
                column = para.get('column')
                
                if current_element is None:
                    # Check if the previous paragraph introduces this element
                    if combined and self._paragraph_introduces_list(combined[-1]):
                        # Same column check if applicable
                        prev_column = combined[-1].get('column')
                        if prev_column is None or column is None or prev_column == column:
                            # Attach to the previous paragraph
                            prev_para = combined[-1]
                            self.logger.debug(f"Attaching {para_type} to previous paragraph that ends with colon")
                            
                            # Create a combined paragraph + element content
                            prev_para['content'] = f"{prev_para['content']}\n{para['content']}"
                            
                            # Continue to next paragraph
                            continue
                    
                    # If not attached to previous paragraph, create a new element
                    current_element = {
                        'content': para['content'],
                        'type': para_type,
                        'position': para.get('position', 0),
                        'column': column  # Preserve column information
                    }
                    current_type = para_type
                    # Preserve metadata if available
                    if 'metadata' in para:
                        current_element['metadata'] = para['metadata']
                else:
                    # If current element is a table and this is also a table row, always combine
                    # regardless of column, ensuring table rows stay together
                    if current_type == 'table' and para_type == 'table':
                        current_element['content'] += '\n' + para['content']
                    # For list items, only combine in the same column
                    elif current_type == 'list' and para_type == 'list':
                        if column is None or current_element.get('column') is None or column == current_element['column']:
                            current_element['content'] += '\n' + para['content']
                        else:
                            # If different columns, add current list and start a new one
                            combined.append(current_element)
                            current_element = {
                                'content': para['content'],
                                'type': para_type,
                                'position': para.get('position', 0),
                                'column': column
                            }
                            if 'metadata' in para:
                                current_element['metadata'] = para['metadata']
                    # If current element is a list and this is a table row (or vice versa)
                    # finish the current element and start a new one
                    else:
                        combined.append(current_element)
                        current_element = {
                            'content': para['content'],
                            'type': para_type,
                            'position': para.get('position', 0),
                            'column': column
                        }
                        current_type = para_type
                        if 'metadata' in para:
                            current_element['metadata'] = para['metadata']
            else:
                if current_element is not None:
                    combined.append(current_element)
                    current_element = None
                    current_type = None
                combined.append(para)
        
        # Add the last element if there is one
        if current_element is not None:
            combined.append(current_element)
        
        return combined
    
    def _combine_lists(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        Combine consecutive list items into a single paragraph and
        attach lists to their introducing paragraphs.
        
        DEPRECATED: Use _combine_structure_elements instead which handles both lists and tables.
        This method is kept for backward compatibility.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            List of paragraphs with combined lists
        """
        return self._combine_structure_elements(paragraphs)
    
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
        headers_by_column = {}  # Track headers by column to maintain proper association
        
        for para in paragraphs:
            column = para.get('column')
            
            if para['type'] == 'header':
                # Store header by column
                headers_by_column[column] = para['content']
                # Add the header as a standalone paragraph too
                result.append(para)
            else:
                # Associate with appropriate header for this column
                if column in headers_by_column:
                    para['header_content'] = headers_by_column[column]
                    # Clear header association after using it (only associate with next paragraph)
                    del headers_by_column[column]
                
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
        # If already classified as header, respect that
        if para_info.get('type') == 'header':
            return True
            
        # If metadata contains header info, use that
        if 'metadata' in para_info:
            metadata = para_info['metadata']
            if metadata.get('is_bold') and metadata.get('font_size', 0) > 10:
                return True
        
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
    
    def _is_table_row(self, content: str) -> bool:
        """
        Determine if a paragraph is a table row.
        
        Args:
            content: Paragraph content
            
        Returns:
            True if paragraph is a table row, False otherwise
        """
        # Check for common table row patterns
        # Pipe-separated values (typical for markdown tables)
        if '|' in content and content.count('|') >= 2:
            return True
            
        # Check for tab-separated values
        if '\t' in content and content.count('\t') >= 1:
            return True
            
        # Check for consistent spacing that might indicate columns
        # (like 2+ groups of text separated by 3+ spaces)
        if re.search(r'\w+\s{3,}\w+', content):
            return True
            
        # Check for markdown table divider rows
        if re.match(r'^\s*\|?\s*[-:]+\s*\|(?:\s*[-:]+\s*\|)*\s*$', content):
            return True
            
        # Check for HTML table tags
        if re.search(r'<tr>|<td>|<th>', content):
            return True
            
        return False
    
    def _is_list_item(self, content: str) -> bool:
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
        if re.match(r'^\s*(\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+|[ivxIVX]+[\.\)]\s+)', content):
            return True
            
        # Check for multi-line lists with continuation lines
        lines = content.split('\n')
        if len(lines) > 1:
            if any(self._is_list_item(line) for line in lines):
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
    
    def _extract_table_rows(self, text: str) -> List[str]:
        """
        Extract individual table rows from text.
        
        Args:
            text: Text to extract rows from
            
        Returns:
            List of extracted rows
        """
        # Simple approach: split by newlines
        rows = []
        for line in text.split('\n'):
            if line.strip():
                rows.append(line.strip())
        return rows
    
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