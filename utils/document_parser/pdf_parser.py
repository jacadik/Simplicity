"""
PDF Parser module for extracting paragraphs from PDF documents.
"""

import re
import time
import logging
from typing import List, Dict, Optional

import pdfplumber

from .base_parser import BaseDocumentParser
from .paragraph import Paragraph

class PDFParser(BaseDocumentParser):
    """PDF document parser that extracts paragraphs from PDF files."""
    
    def parse_document(self, file_path: str, doc_id: int) -> List[Paragraph]:
        """
        Parse PDF document and extract paragraphs with enhanced layout analysis.
        
        Args:
            file_path: Path to the PDF file
            doc_id: ID of the document in the database
            
        Returns:
            List of extracted paragraphs
        """
        self.logger.info(f"Parsing PDF document: {file_path}")
        paragraphs = []
        
        start_time = time.time()
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
                        page_paragraphs = self._extract_paragraphs_from_pdf_text(text, page)
                        
                        # Log detailed information about paragraph extraction
                        self.logger.debug(f"Page {i+1}: Extracted {len(page_paragraphs)} paragraphs " 
                                         f"from {len(text)} characters of text")
                        
                        for para in page_paragraphs:
                            raw_paragraphs.append({
                                'content': para,
                                'type': 'unknown',  # Will classify later
                                'page': i+1
                            })
                
                # Process extracted content with paragraph extractor
                paragraphs = self.paragraph_extractor.process_raw_paragraphs(raw_paragraphs, doc_id)
                
                # Quality check - if we got very few paragraphs for a large document, try harder
                if len(paragraphs) <= 3 and len(raw_paragraphs) > 0:
                    total_text = sum(len(p['content']) for p in raw_paragraphs)
                    if total_text > 2000:  # Long document but few paragraphs detected
                        self.logger.warning(f"Few paragraphs detected ({len(paragraphs)}) for large document "
                                           f"({total_text} chars). Applying fallback extraction.")
                        
                        # Apply fallback extraction directly on raw content
                        enhanced_paragraphs = []
                        for raw_para in raw_paragraphs:
                            if len(raw_para['content']) > 1000:  # Only process long paragraphs
                                additional_paras = self._split_long_text(raw_para['content'])
                                for p in additional_paras:
                                    enhanced_paragraphs.append({
                                        'content': p,
                                        'type': 'unknown',
                                        'page': raw_para['page']
                                    })
                            else:
                                enhanced_paragraphs.append(raw_para)
                        
                        # Process enhanced paragraphs
                        if len(enhanced_paragraphs) > len(raw_paragraphs):
                            self.logger.info(f"Enhanced extraction: {len(raw_paragraphs)} → {len(enhanced_paragraphs)} paragraphs")
                            paragraphs = self.paragraph_extractor.process_raw_paragraphs(enhanced_paragraphs, doc_id)
                
                elapsed_time = time.time() - start_time
                self.logger.info(f"Extracted {len(paragraphs)} paragraphs from PDF document in {elapsed_time:.2f} seconds")
                
            # Double-check final result
            import os
            if len(paragraphs) <= 1 and os.path.getsize(file_path) > 50000:  # Reasonable size PDF
                self.logger.warning(f"Parser only extracted {len(paragraphs)} paragraphs from a {os.path.getsize(file_path)/1024:.1f}KB PDF. Possible parsing issue.")
                    
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}", exc_info=True)
        
        return paragraphs
    
    def _format_table(self, table: List[List[str]]) -> str:
        """
        Format a pdfplumber table into a string representation.
        
        Args:
            table: Table data from pdfplumber
            
        Returns:
            Formatted table string
        """
        # Filter out empty/None cells and replace with empty string
        formatted_table = [[cell if cell else '' for cell in row] for row in table]
        
        # Convert to string representation
        result = []
        for row in formatted_table:
            result.append(" | ".join(str(cell).strip() for cell in row))
        
        return "\n".join(result)
    
    def _extract_paragraphs_from_pdf_text(self, text: str, page_obj=None) -> List[str]:
        """
        Extract paragraphs from PDF text using enhanced heuristics.
        
        Args:
            text: The extracted text from a PDF page
            page_obj: The pdfplumber page object (if available) for layout analysis
            
        Returns:
            List of paragraph texts
        """
        # Method 1: Split by double newlines (existing approach but improved)
        paragraphs = []
        
        # First try splitting by standard paragraph markers (double newlines)
        initial_parts = re.split(r'\n\s*\n', text)
        
        # If we only got one part, try more aggressive splitting
        if len(initial_parts) <= 1 and len(text) > 500:  # Long text with no clear paragraphs
            # Method 2: Try splitting by single newlines with additional checks
            lines = text.split('\n')
            current_para = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Start a new paragraph if:
                # 1. Current line starts with a capital letter and previous line ends with period
                # 2. Current line is indented (starts with spaces) differently than previous
                # 3. Current line starts with a bullet point or number
                # 4. Current line has different capitalization pattern (ALL CAPS vs normal)
                
                start_new_para = False
                
                # Check for sentence boundary
                if current_para and line and line[0].isupper():
                    prev_line = current_para[-1]
                    if prev_line and prev_line[-1] in '.!?':
                        # Check if this isn't a continuation (like "Dr. Smith")
                        if not (len(prev_line) >= 2 and prev_line[-2].islower() and 
                               prev_line[-1] == '.' and len(line.split()) > 3):
                            start_new_para = True
                
                # Check for bullet points or numbered lists
                if re.match(r'^\s*[\•\-\*\+○◦➢➣➤►▶→➥➔❖]\s+', line) or \
                   re.match(r'^\s*(\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+)', line):
                    start_new_para = True
                
                # Check for header patterns (ALL CAPS or Title Case with no punctuation)
                if line.isupper() or (line.istitle() and not any(c in line for c in '.,:;!?')):
                    start_new_para = True
                    
                # If we detect a new paragraph and have content, save the current one
                if start_new_para and current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                
                current_para.append(line)
            
            # Add the last paragraph if there's content
            if current_para:
                paragraphs.append(' '.join(current_para))
        else:
            # Process the paragraphs from the initial split
            for part in initial_parts:
                clean_part = part.strip()
                if clean_part:
                    # Further split very long paragraphs that might be incorrectly joined
                    if len(clean_part) > 1000:  # Very long paragraph
                        sub_parts = self._split_long_text(clean_part)
                        paragraphs.extend(sub_parts)
                    else:
                        paragraphs.append(clean_part)
        
        # Method 3: Use layout analysis if available and we still don't have good paragraphs
        if page_obj and (len(paragraphs) <= 1) and len(text) > 500:
            try:
                # Extract character data with positions
                chars = page_obj.chars
                if chars:
                    layout_paragraphs = self._extract_paragraphs_from_layout(chars)
                    if len(layout_paragraphs) > len(paragraphs):
                        return layout_paragraphs
            except Exception as e:
                self.logger.warning(f"Layout analysis failed: {str(e)}")
        
        # If we still couldn't parse paragraphs, use a fallback method
        if len(paragraphs) <= 1 and len(text) > 500:
            # Fallback: Split by sentences as a last resort
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            # Group sentences into paragraphs (roughly 3-5 sentences per paragraph)
            if len(sentences) > 3:
                grouped_sentences = []
                for i in range(0, len(sentences), 4):  # Group by ~4 sentences
                    group = ' '.join(sentences[i:i+4])
                    if group.strip():
                        grouped_sentences.append(group)
                if grouped_sentences:
                    return grouped_sentences
        
        return paragraphs

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split very long text into paragraph-like chunks based on sentence boundaries.
        
        Args:
            text: Long text to split
            
        Returns:
            List of text chunks
        """
        # Find sentence boundaries
        sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
        
        if not sentence_boundaries:
            return [text]  # No clear sentences found
        
        # If average sentence length is very long, this might not be proper paragraphs
        avg_sentence_length = sentence_boundaries[0] if len(sentence_boundaries) == 1 else \
                            sentence_boundaries[-1] / len(sentence_boundaries)
        
        # Group sentences into paragraph-like chunks
        chunks = []
        start = 0
        
        # Aim for paragraphs of 3-5 sentences or ~500 characters
        target_chunk_size = min(3 if avg_sentence_length > 100 else 5, 
                            max(1, int(500 / avg_sentence_length)))
        
        for i in range(target_chunk_size, len(sentence_boundaries), target_chunk_size):
            end = sentence_boundaries[min(i, len(sentence_boundaries) - 1)]
            chunks.append(text[start:end].strip())
            start = end
        
        # Add the last chunk
        if start < len(text):
            chunks.append(text[start:].strip())
        
        return chunks

    def _extract_paragraphs_from_layout(self, chars: List[Dict]) -> List[str]:
        """
        Extract paragraphs based on character layout analysis.
        
        Args:
            chars: List of character data from pdfplumber
            
        Returns:
            List of extracted paragraphs
        """
        if not chars:
            return []
        
        # Sort characters by y0 (vertical position) then x0 (horizontal position)
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        # Group characters into lines based on y-position
        lines = []
        current_line = []
        current_y = None
        
        for char in sorted_chars:
            if current_y is None or abs(char['top'] - current_y) < 2:  # Same line (with small tolerance)
                current_line.append(char)
                current_y = char['top']
            else:  # New line
                if current_line:
                    lines.append(current_line)
                current_line = [char]
                current_y = char['top']
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Convert lines to text
        text_lines = []
        for line in lines:
            # Sort by x position within each line
            sorted_line = sorted(line, key=lambda c: c['x0'])
            line_text = ''.join(c['text'] for c in sorted_line if 'text' in c)
            if line_text.strip():
                text_lines.append(line_text.strip())
        
        # Group lines into paragraphs based on vertical spacing
        paragraphs = []
        current_para = []
        prev_bottom = None
        
        for i, line in enumerate(text_lines):
            if not line.strip():
                continue
                
            # If we have line position data
            if i < len(lines) and lines[i]:
                current_top = min(c['top'] for c in lines[i])
                
                # Check if this is a new paragraph based on spacing
                if prev_bottom is not None:
                    # Estimate line height
                    line_height = max(c['height'] for c in lines[i] if 'height' in c) \
                                if any('height' in c for c in lines[i]) else 10
                    
                    # If the gap is significantly larger than a typical line height, start new paragraph
                    if (current_top - prev_bottom) > (line_height * 1.5):
                        if current_para:
                            paragraphs.append(' '.join(current_para))
                            current_para = []
                
                # Update for next iteration
                prev_bottom = max(c['bottom'] for c in lines[i] if 'bottom' in c) \
                            if any('bottom' in c for c in lines[i]) else current_top + 10
            
            # Apply text-based heuristics as well
            if current_para and (line.isupper() or re.match(r'^\d+[\.\)]\s+', line)):
                # Headers or numbered items should start new paragraphs
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            
            current_para.append(line)
        
        # Add the last paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs
