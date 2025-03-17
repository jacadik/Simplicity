import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import docx
import pdfplumber
from dataclasses import dataclass

@dataclass
class Paragraph:
    """Class for storing paragraph information."""
    content: str
    doc_id: int
    paragraph_type: str  # 'normal', 'header', 'list', 'table', 'footer', 'address', etc.
    position: int
    header_content: Optional[str] = None
    column: Optional[int] = None  # Track column position

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
        """Enhanced PDF parsing with font information extraction."""
        self.logger.info(f"Parsing PDF document: {file_path}")
        paragraphs = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract text from PDF with layout information
                raw_paragraphs = []
                
                # Collect font information across the document
                all_fonts = []
                
                for i, page in enumerate(pdf.pages):
                    self.logger.debug(f"Processing page {i+1}/{len(pdf.pages)}")
                    
                    # Extract text, tables, and layout info
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    # Collect font information for this page
                    if hasattr(page, 'chars') and page.chars:
                        fonts = {}
                        for char in page.chars:
                            if 'fontname' in char and 'size' in char:
                                font_key = (char['fontname'], char['size'])
                                if font_key not in fonts:
                                    fonts[font_key] = 0
                                fonts[font_key] += 1
                        
                        all_fonts.extend([(name, size, count) for (name, size), count in fonts.items()])
                    
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
                        # Calculate average font size for this page
                        avg_font_size = 0
                        if hasattr(page, 'chars') and page.chars:
                            sizes = [char.get('size', 0) for char in page.chars if 'size' in char]
                            if sizes:
                                avg_font_size = sum(sizes) / len(sizes)
                        
                        # Try standard text-based extraction first instead of column-based
                        page_paragraphs = self._extract_paragraphs_from_pdf_text(text, page)
                        
                        # Validate paragraphs to ensure they contain meaningful text
                        valid_paragraphs = []
                        for para in page_paragraphs:
                            content = para.get('content', '') if isinstance(para, dict) else para
                            # Ensure paragraph has meaningful content
                            if content and isinstance(content, str) and len(content) > 10:
                                if isinstance(para, dict):
                                    para['page'] = i+1
                                    para['avg_font_size'] = avg_font_size
                                    valid_paragraphs.append(para)
                                else:
                                    valid_paragraphs.append({
                                        'content': content,
                                        'type': 'unknown',
                                        'page': i+1,
                                        'avg_font_size': avg_font_size
                                    })
                        
                        if valid_paragraphs:
                            raw_paragraphs.extend(valid_paragraphs)
                        else:
                            # Fall back to column-based processing only if text-based fails
                            column_paragraphs = self._process_columns(page)
                            
                            # Validate column paragraphs
                            valid_column_paragraphs = []
                            for para in column_paragraphs:
                                content = para.get('content', '')
                                # Check if paragraph contains meaningful text
                                if content and len(content.split()) > 3 and len(content) > 20:
                                    para['page'] = i+1
                                    para['avg_font_size'] = avg_font_size
                                    valid_column_paragraphs.append(para)
                            
                            if valid_column_paragraphs:
                                raw_paragraphs.extend(valid_column_paragraphs)
                            elif text.strip():  # Use the raw text as fallback if all else fails
                                raw_paragraphs.append({
                                    'content': text.strip(),
                                    'type': 'unknown',
                                    'page': i+1,
                                    'avg_font_size': avg_font_size
                                })
                
                # Calculate document-wide font statistics
                font_sizes = [size for _, size, _ in all_fonts]
                avg_doc_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12  # Default
                
                # Find the most common font size (body text)
                font_size_counts = {}
                for _, size, count in all_fonts:
                    size_rounded = round(size * 2) / 2  # Round to nearest 0.5
                    if size_rounded not in font_size_counts:
                        font_size_counts[size_rounded] = 0
                    font_size_counts[size_rounded] += count
                
                body_font_size = max(font_size_counts.items(), key=lambda x: x[1])[0] if font_size_counts else avg_doc_font_size
                
                # Update paragraphs with document-wide font info
                for para in raw_paragraphs:
                    para['avg_doc_font_size'] = avg_doc_font_size
                    para['body_font_size'] = body_font_size
                    
                    # If this paragraph has its own font size, compare to body text
                    if 'fontname' in para and 'size' in para:
                        para['is_header_font'] = para['size'] > body_font_size * 1.2
                
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
                                        'page': raw_para['page'],
                                        'avg_font_size': raw_para.get('avg_font_size', 0),
                                        'body_font_size': raw_para.get('body_font_size', 0)
                                    })
                            else:
                                enhanced_paragraphs.append(raw_para)
                        
                        # Process enhanced paragraphs
                        if len(enhanced_paragraphs) > len(raw_paragraphs):
                            self.logger.info(f"Enhanced extraction: {len(raw_paragraphs)} → {len(enhanced_paragraphs)} paragraphs")
                            paragraphs = self.paragraph_extractor.process_raw_paragraphs(enhanced_paragraphs, doc_id)
                
                self.logger.info(f"Extracted {len(paragraphs)} paragraphs from PDF document")
                
            # Double-check final result
            if len(paragraphs) <= 1 and os.path.getsize(file_path) > 50000:  # Reasonable size PDF
                self.logger.warning(f"Parser only extracted {len(paragraphs)} paragraphs from a {os.path.getsize(file_path)/1024:.1f}KB PDF. Possible parsing issue.")
                    
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
    
    def _process_columns(self, page_obj) -> List[Dict]:
        """Process text columns based on spatial layout analysis."""
        if not page_obj or not hasattr(page_obj, 'chars') or len(page_obj.chars) == 0:
            return []
            
        # Get character data with positions
        chars = page_obj.chars
        
        # Calculate x-position distribution to detect columns
        x_positions = [c['x0'] for c in chars if 'x0' in c]
        
        if not x_positions:
            return []
        
        # Find density-based clusters of x-positions to identify column starts
        x_pos_sorted = sorted(x_positions)
        x_pos_counts = {}
        for x in x_pos_sorted:
            # Round to nearest 5 pixels for clustering
            x_rounded = round(x / 5) * 5
            if x_rounded not in x_pos_counts:
                x_pos_counts[x_rounded] = 0
            x_pos_counts[x_rounded] += 1
        
        # Find x-positions with high concentration of characters (column starts)
        threshold = max(x_pos_counts.values()) * 0.3  # Increased threshold to 30% of max count
        potential_columns = [x for x, count in x_pos_counts.items() if count > threshold]
        
        # If we detect multiple columns, process them separately
        if len(potential_columns) > 1 and len(potential_columns) <= 4:  # Limit to reasonable number of columns
            self.logger.info(f"Detected {len(potential_columns)} potential columns: {potential_columns}")
            
            # Sort column positions
            potential_columns.sort()
            
            # Add page width as the end of the last column
            page_width = page_obj.width
            potential_columns.append(page_width)
            
            # Process each column
            column_paragraphs = []
            for col_idx in range(len(potential_columns) - 1):
                left_bound = potential_columns[col_idx] - 5  # Add small margin
                right_bound = potential_columns[col_idx + 1] - 5
                
                # Filter characters in this column
                column_chars = [c for c in chars 
                               if c.get('x0', 0) >= left_bound and c.get('x0', 0) < right_bound]
                
                # Process column if it has enough characters
                if len(column_chars) > 20:  # Arbitrary threshold
                    # Sort by y-position (top to bottom)
                    column_chars.sort(key=lambda c: c.get('top', 0))
                    
                    # Group characters into lines with more flexible tolerance
                    lines = []
                    current_line = []
                    current_y = None
                    line_height = 0
                    
                    for char in column_chars:
                        char_top = char.get('top', 0)
                        char_height = char.get('height', 10)  # Default height if not available
                        
                        # Adaptive line grouping based on character height
                        tolerance = max(5, char_height * 0.5)  # More flexible tolerance
                        
                        if current_y is None or abs(char_top - current_y) < tolerance:
                            current_line.append(char)
                            # Use weighted average for current_y to handle slight variations
                            if current_y is None:
                                current_y = char_top
                            else:
                                current_y = (current_y * len(current_line) + char_top) / (len(current_line) + 1)
                            line_height = max(line_height, char_height)
                        else:
                            if current_line:
                                # Sort by x-position and join
                                current_line.sort(key=lambda c: c.get('x0', 0))
                                lines.append((current_line, line_height))
                            current_line = [char]
                            current_y = char_top
                            line_height = char_height
                    
                    if current_line:
                        lines.append((sorted(current_line, key=lambda c: c.get('x0', 0)), line_height))
                    
                    # Convert lines to text
                    line_texts = []
                    for line_data in lines:
                        line, _ = line_data
                        line_text = ''.join(c.get('text', '') for c in line)
                        if line_text.strip():
                            line_texts.append(line_text)
                    
                    # Group lines into paragraphs with more sophisticated logic
                    if line_texts:
                        paragraphs = []
                        current_paragraph = line_texts[0]
                        
                        for line_idx in range(1, len(line_texts)):
                            # Check for paragraph breaks based on spacing and indentation
                            if line_idx < len(lines) and lines[line_idx][0] and lines[line_idx-1][0]:
                                prev_line, prev_height = lines[line_idx-1]
                                curr_line, curr_height = lines[line_idx]
                                
                                # Get vertical positions with safeguards
                                prev_bottoms = [c.get('bottom', c.get('top', 0) + prev_height) for c in prev_line]
                                curr_tops = [c.get('top', 0) for c in curr_line]
                                
                                if prev_bottoms and curr_tops:
                                    prev_bottom = max(prev_bottoms)
                                    curr_top = min(curr_tops)
                                    
                                    # Calculate spacing
                                    spacing = curr_top - prev_bottom
                                    avg_height = (prev_height + curr_height) / 2
                                    
                                    # Paragraph break if spacing is significantly larger than line height
                                    if spacing > avg_height * 1.5:
                                        paragraphs.append(current_paragraph)
                                        current_paragraph = line_texts[line_idx]
                                    else:
                                        current_paragraph += ' ' + line_texts[line_idx]
                                else:
                                    current_paragraph += ' ' + line_texts[line_idx]
                            else:
                                current_paragraph += ' ' + line_texts[line_idx]
                        
                        if current_paragraph:
                            paragraphs.append(current_paragraph)
                        
                        # Create paragraph dictionaries
                        for para_text in paragraphs:
                            column_paragraphs.append({
                                'content': para_text,
                                'type': 'unknown',  # Will classify later
                                'column': col_idx + 1,
                                'x_start': left_bound,
                                'x_end': right_bound
                            })
        
        return column_paragraphs
    
    def _extract_paragraphs_from_pdf_text(self, text: str, page_obj=None) -> List[Dict]:
        """
        Enhanced paragraph extraction with improved heuristics.
        
        Args:
            text: The extracted text from a PDF page
            page_obj: The pdfplumber page object (if available) for layout analysis
            
        Returns:
            List of paragraph texts or dictionaries
        """
        # Fall back to standard text-based processing
        paragraphs = []
        
        # Try splitting by double newlines
        parts = re.split(r'\n\s*\n', text)
        
        # If we only got one part, try more aggressive splitting
        if len(parts) <= 1 and len(text) > 500:  # Long text with no clear paragraphs
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
            for part in parts:
                clean_part = part.strip()
                if clean_part:
                    # Further split very long paragraphs that might be incorrectly joined
                    if len(clean_part) > 1000:  # Very long paragraph
                        sub_parts = self._split_long_text(clean_part)
                        paragraphs.extend(sub_parts)
                    else:
                        paragraphs.append(clean_part)
        
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
                    paragraphs = grouped_sentences
        
        # Convert string paragraphs to dictionary format for consistency
        result_paragraphs = []
        for para in paragraphs:
            if isinstance(para, str):
                result_paragraphs.append({'content': para, 'type': 'unknown'})
            else:
                result_paragraphs.append(para)
        
        return result_paragraphs

    def _split_long_text(self, text: str) -> List[str]:
        """Split very long text into paragraph-like chunks based on sentence boundaries."""
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
            column = para.get('column')
            
            # Skip if this is an address (we'll capture it separately, not exclude it)
            # if para['type'] == 'address':
            #     self.logger.debug(f"Skipping address paragraph: {para['content'][:30]}...")
            #     continue
                
            paragraph = Paragraph(
                content=para['content'],
                doc_id=doc_id,
                paragraph_type=para['type'],
                position=i,
                header_content=header_content,
                column=column
            )
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _classify_paragraphs(self, raw_paragraphs: List[Dict]) -> List[Dict]:
        """Classify paragraphs into different types."""
        classified = []
        
        for para in raw_paragraphs:
            para_type = para.get('type', 'unknown')
            content = para.get('content', '')
            
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
            if self._is_address(content, para):
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
        """Enhanced list combination with context awareness and indentation handling."""
        if not paragraphs:
            return []
            
        combined = []
        current_list = None
        list_indent_level = None
        list_type = None
        introduction_paragraph = None
        
        for i, para in enumerate(paragraphs):
            # Check for potential list introduction (ends with colon)
            if para['type'] != 'list' and i+1 < len(paragraphs):
                if para['content'].strip().endswith(':') and paragraphs[i+1]['type'] == 'list':
                    introduction_paragraph = para
                    continue
                
            if para['type'] == 'list':
                # Extract indentation level
                current_indent = len(re.match(r'^\s*', para['content']).group(0))
                
                # Determine list type (bullet, numbered, etc.)
                current_type = 'bullet' if re.match(r'^\s*[•\-\*\+○◦➢➣➤►▶→➥➔❖♦◆●■□]\s+', para['content']) else 'numbered'
                
                # Start new list or continue existing
                if current_list is None:
                    # If we have an introduction, include it with the list
                    if introduction_paragraph:
                        current_list = {
                            'content': introduction_paragraph['content'] + '\n' + para['content'],
                            'type': 'list',
                            'position': introduction_paragraph.get('position', 0),
                            'header_content': introduction_paragraph.get('header_content'),
                            'column': para.get('column')
                        }
                        introduction_paragraph = None
                    else:
                        current_list = {
                            'content': para['content'],
                            'type': 'list',
                            'position': para.get('position', 0),
                            'header_content': para.get('header_content'),
                            'column': para.get('column')
                        }
                    list_indent_level = current_indent
                    list_type = current_type
                else:
                    # Only combine if indentation levels are consistent or nested
                    if (abs(current_indent - list_indent_level) <= 4 or  # Same level with small variation
                        current_indent > list_indent_level + 4):  # Nested list
                        
                        # Add a newline between list items
                        current_list['content'] += '\n' + para['content']
                    else:
                        # Different indentation or list type indicates new list
                        combined.append(current_list)
                        current_list = {
                            'content': para['content'],
                            'type': 'list',
                            'position': para.get('position', 0),
                            'header_content': para.get('header_content'),
                            'column': para.get('column')
                        }
                        list_indent_level = current_indent
                        list_type = current_type
            else:
                # Non-list paragraph - add current list and reset
                if current_list is not None:
                    combined.append(current_list)
                    current_list = None
                combined.append(para)
        
        # Add the last list if there is one
        if current_list is not None:
            combined.append(current_list)
        
        return combined
    
    def _associate_headers(self, paragraphs: List[Dict]) -> List[Dict]:
        """Improved header-paragraph association logic."""
        if not paragraphs:
            return []
            
        result = []
        current_header = None
        current_column = None
        
        for i, para in enumerate(paragraphs):
            # Track column changes
            if 'column' in para and para['column'] is not None:
                if current_column != para['column']:
                    # Reset header when moving to a new column
                    current_header = None
                current_column = para['column']
                
            if para['type'] == 'header':
                # Store this as the current header
                current_header = para['content']
                
                # Add the header as a standalone paragraph
                result.append(para)
                
                # Look ahead to associate with the next paragraph if it exists
                if i + 1 < len(paragraphs) and paragraphs[i+1]['type'] != 'header':
                    next_para = paragraphs[i+1].copy()
                    
                    # Only associate if in the same column
                    if (not 'column' in para) or (not 'column' in next_para) or (para.get('column') == next_para.get('column')):
                        next_para['header_content'] = current_header
                        
                        # Skip adding the next paragraph here as we'll handle it in the next iteration
                        # Just update the original array
                        paragraphs[i+1] = next_para
            else:
                # For non-header paragraphs, check if we should associate with the current header
                if current_header is not None and i > 0 and paragraphs[i-1]['type'] == 'header':
                    # Already handled in the header case above
                    result.append(para)
                    
                    # Reset header after it's been associated
                    if para['type'] not in ['list', 'table', 'address']:
                        current_header = None
                else:
                    # Regular paragraph, not following a header
                    result.append(para)
        
        return result
    
    def _is_header(self, content: str, para_info: Dict) -> bool:
        """Enhanced header detection with improved heuristics."""
        # Check if style indicates a header (for DOCX)
        if 'style' in para_info and ('head' in para_info['style'].lower() or 'title' in para_info['style'].lower()):
            return True
        
        # Check font information for PDFs
        if 'fontname' in para_info and 'size' in para_info:
            font_size = para_info.get('size', 0)
            # If font size data exists and this is larger than average
            if 'avg_font_size' in para_info and font_size > para_info['avg_font_size'] * 1.2:
                return True
        
        # Check for header flag from font analysis in PDF parser
        if para_info.get('is_header_font', False):
            return True
        
        # Headers are typically short
        if len(content) > 150:  # Increased threshold
            return False
        
        # Check for section numbering patterns
        if re.match(r'^[\d\.]+\s+[A-Z]', content):  # e.g., "1.2 Introduction"
            return True
            
        if re.match(r'^[A-Z]\.[\s]+[A-Z]', content):  # e.g., "A. Introduction"
            return True
        
        # Check for header formatting
        if content.isupper() and len(content.split()) <= 10:
            return True
            
        if content.istitle() and not any(c in content for c in '.,:;!?') and len(content.split()) <= 10:
            return True
        
        # Check for introductory text ending with colon
        if content.strip().endswith(':') and len(content.split()) <= 15:
            return True
        
        # Check for common header keywords
        header_keywords = ['introduction', 'summary', 'overview', 'conclusion', 
                          'background', 'purpose', 'objective', 'scope', 'welcome',
                          'first', 'next', 'finally']
        
        first_word = content.strip().split()[0].lower() if content.strip() else ''
        if first_word in header_keywords and len(content.split()) <= 10:
            return True
        
        # Check for exclamation in short phrase (often a header)
        if '!' in content and len(content) < 50:
            return True
        
        # Headers often stand alone without ending punctuation
        if len(content) > 0 and not content[-1] in '.?!:;,' and len(content.split()) <= 10:
            return True
        
        return False
    
    def _is_list(self, content: str) -> bool:
        """Enhanced list item detection with support for various formats."""
        # Enhanced bullet point detection with more Unicode bullets
        bullet_pattern = r'^\s*[•\-\*\+○◦➢➣➤►▶→➥➔❖♦◆●■□]\s+'
        
        # Improved numbered list detection with various formats
        numbered_pattern = r'^\s*(?:\d+[\.\)]\s+|\([a-z\d]\)\s+|[a-z\d][\.\)]\s+)'
        
        # Alpha list detection (a., b., etc.)
        alpha_pattern = r'^\s*[a-zA-Z][\.\)]\s+'
        
        # Roman numeral detection (i., ii., iii., etc.)
        roman_pattern = r'^\s*(?:i{1,3}|iv|v|vi{1,3}|ix|x)[\.\)]\s+'
        
        is_bullet = bool(re.match(bullet_pattern, content))
        is_numbered = bool(re.match(numbered_pattern, content))
        is_alpha = bool(re.match(alpha_pattern, content))
        is_roman = bool(re.match(roman_pattern, content))
        
        return is_bullet or is_numbered or is_alpha or is_roman
    
    def _is_address(self, content: str, para_info: Dict = None) -> bool:
        """Enhanced address detection with better pattern recognition."""
        # Skip if too long or too short
        if len(content) > 300 or len(content) < 5:
            return False
            
        # Check for multi-line structure common in addresses
        lines = content.splitlines()
        if len(lines) >= 2 and len(lines) <= 8:  # Typical address has 2-8 lines
            # Check for short lines typical in addresses
            short_lines = [line for line in lines if line.strip() and len(line.strip()) < 50]
            if len(short_lines) >= 2 and len(short_lines) == len(lines):
                # Enhanced postal code patterns
                postal_patterns = [
                    r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b',  # UK
                    r'\b\d{5}(-\d{4})?\b',  # US
                    r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b',  # Canada
                    r'\b\d{4}\s?[A-Z]{2}\b'  # Netherlands
                ]
                
                # Check if any line contains a postal code
                has_postal = any(
                    any(re.search(pattern, line) for pattern in postal_patterns)
                    for line in lines
                )
                
                # Enhanced address component detection
                address_indicators = [
                    'street', 'avenue', 'road', 'lane', 'drive', 'boulevard',
                    'st.', 'ave.', 'rd.', 'ln.', 'dr.', 'blvd.', 'apt', 'suite',
                    'terrace', 'court', 'circle', 'way', 'place'
                ]
                has_indicator = any(
                    any(indicator.lower() in line.lower() for indicator in address_indicators)
                    for line in lines
                )
                
                # Check for name patterns that often start addresses
                name_pattern = any(re.match(r'^(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+', line) for line in lines)
                
                # Check for common address formats at start of lines
                address_number_pattern = any(re.match(r'^\d+\s+[A-Z]', line) for line in lines)
                
                # Spatial analysis if available
                consistent_left_margin = False
                if para_info and 'chars' in para_info:
                    # Check if lines have consistent left margin
                    x_positions = []
                    for line in lines:
                        line_chars = [c for c in para_info['chars'] if c['text'] in line[:5]]
                        if line_chars:
                            x_positions.append(line_chars[0].get('x0', 0))
                    
                    if x_positions and max(x_positions) - min(x_positions) < 10:
                        consistent_left_margin = True
                
                return (has_postal or has_indicator or name_pattern or address_number_pattern or consistent_left_margin)
        
        return False
    
    def _is_boilerplate(self, content: str) -> bool:
        """Determine if a paragraph is boilerplate text (footer, disclaimer, etc.)."""
        # Check for common boilerplate indicators
        boilerplate_indicators = [
            'all rights reserved', 'copyright', '©', 'confidential', 
            'disclaimer', 'terms and conditions', 'privacy policy',
            'legal notice', 'proprietary', 'registered in'
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