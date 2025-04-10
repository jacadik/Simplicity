"""
PDF Parser module for extracting paragraphs from PDF documents.
Enhanced with column detection, table handling, layout analysis, and list handling.
"""

import re
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

import pdfplumber
import fitz  # PyMuPDF

from .base_parser import BaseDocumentParser
from .paragraph import Paragraph

class PDFParser(BaseDocumentParser):
    """PDF document parser that extracts paragraphs from PDF files with enhanced layout analysis."""
    
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
            # First use PDFPlumber for tables
            tables_by_page = self._extract_tables_from_pdf(file_path)
            
            # Then use PyMuPDF for document structure and text
            doc = fitz.open(file_path)
            
            # Check for potential scan-only PDF
            is_scanned = self._is_scanned_pdf(doc)
            if is_scanned:
                self.logger.warning(f"Document appears to be a scanned PDF: {file_path}")
            
            # Extract document structure
            toc = doc.get_toc()
            raw_paragraphs = []
            
            if toc:
                # Process table of contents
                toc_content = self._process_toc(toc)
                if toc_content:
                    raw_paragraphs.append({
                        'content': toc_content,
                        'type': 'toc',
                        'page': 0
                    })
            
            # Process each page
            for page_num, page in enumerate(doc):
                self.logger.debug(f"Processing page {page_num+1}/{len(doc)}")
                
                # Extract and add tables first (if any)
                if page_num in tables_by_page:
                    for table_info in tables_by_page[page_num]:
                        table_element = {
                            'content': self._format_table(table_info['data']),
                            'type': 'table',
                            'page': page_num + 1
                        }
                        raw_paragraphs.append(table_element)
                
                # Get table rectangles for this page
                table_rects = []
                if page_num in tables_by_page:
                    for table_info in tables_by_page[page_num]:
                        if table_info['bbox']:
                            table_rects.append(table_info['bbox'])
                
                # Process text content
                page_elements = self._process_page_with_layout(page, table_rects, page_num)
                
                # Process each extracted element to ensure proper list formatting
                processed_elements = []
                for element in page_elements:
                    # Process any lists in the content
                    if 'content' in element:
                        element['content'] = self._process_paragraph_content(element['content'])
                    processed_elements.append(element)
                
                raw_paragraphs.extend(processed_elements)
            
            # Process extracted content with paragraph extractor
            paragraphs = self.paragraph_extractor.process_raw_paragraphs(raw_paragraphs, doc_id)
            
            # Quality check - if we got very few paragraphs for a large document, try harder
            if len(paragraphs) <= 3 and len(raw_paragraphs) > 0:
                total_text = sum(len(p['content']) for p in raw_paragraphs if isinstance(p.get('content'), str))
                if total_text > 2000:  # Long document but few paragraphs detected
                    self.logger.warning(f"Few paragraphs detected ({len(paragraphs)}) for large document "
                                      f"({total_text} chars). Applying fallback extraction.")
                    
                    # Apply fallback extraction directly on raw content
                    enhanced_paragraphs = []
                    for raw_para in raw_paragraphs:
                        if isinstance(raw_para.get('content'), str) and len(raw_para['content']) > 1000:
                            # Process the content to ensure proper list formatting
                            content = self._process_paragraph_content(raw_para['content'])
                            
                            # If the content is too long, split it into smaller chunks
                            if len(content) > 1000:
                                additional_paras = self._split_long_text(content)
                                for p in additional_paras:
                                    enhanced_paragraphs.append({
                                        'content': p,
                                        'type': 'unknown',
                                        'page': raw_para.get('page', 0)
                                    })
                            else:
                                raw_para['content'] = content
                                enhanced_paragraphs.append(raw_para)
                        else:
                            enhanced_paragraphs.append(raw_para)
                    
                    # Process enhanced paragraphs
                    if len(enhanced_paragraphs) > len(raw_paragraphs):
                        self.logger.info(f"Enhanced extraction: {len(raw_paragraphs)} → {len(enhanced_paragraphs)} paragraphs")
                        paragraphs = self.paragraph_extractor.process_raw_paragraphs(enhanced_paragraphs, doc_id)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Extracted {len(paragraphs)} paragraphs from PDF document in {elapsed_time:.2f} seconds")
            
            doc.close()
                
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}", exc_info=True)
        
        return paragraphs

    def _extract_tables_from_pdf(self, file_path: str) -> Dict[int, List[Dict]]:
        """
        Extract tables from PDF using PDFPlumber with their locations.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to lists of table information
        """
        tables_by_page = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        tables_by_page[page_num] = []
                        for table in tables:
                            if table and any(any(cell for cell in row) for row in table):
                                # Find table area to get bounding box
                                table_areas = page.find_tables(table_settings={"vertical_strategy": "text", 
                                                                             "horizontal_strategy": "text"})
                                table_bbox = None
                                if table_areas and len(table_areas) > len(tables_by_page[page_num]):
                                    table_area = table_areas[len(tables_by_page[page_num])]
                                    table_bbox = (
                                        table_area.bbox[0],
                                        table_area.bbox[1],
                                        table_area.bbox[2],
                                        table_area.bbox[3],
                                    )
                                
                                tables_by_page[page_num].append({
                                    'data': table,
                                    'bbox': table_bbox
                                })
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}", exc_info=True)
        
        return tables_by_page
    
    def _process_page_with_layout(self, page, table_rects, page_num):
        """
        Process a page with layout analysis, handling columns correctly.
        
        Args:
            page: PyMuPDF page object
            table_rects: List of table rectangles to exclude
            page_num: Page number
            
        Returns:
            List of extracted paragraph data
        """
        page_elements = []
        
        try:
            # Get page dimensions
            page_width = page.rect.width
            
            # Get raw text blocks
            blocks = page.get_text("dict")["blocks"]
            filtered_blocks = self._filter_blocks_by_tables(blocks, table_rects)
            
            # Check for multi-column layout
            is_multi_column = self._is_clearly_multi_column(filtered_blocks, page_width)
            
            if is_multi_column:
                self.logger.info(f"Page {page_num+1} appears to have multiple columns, processing accordingly")
                # Process with column detection
                column_bounds = self._detect_columns(filtered_blocks, page_width)
                if column_bounds and len(column_bounds) >= 2:
                    # Process each column separately
                    for col_idx, (x0, x1) in enumerate(column_bounds):
                        col_rect = fitz.Rect(x0, 0, x1, page.rect.height)
                        col_blocks = page.get_text("dict", clip=col_rect)["blocks"]
                        filtered_col_blocks = self._filter_blocks_by_tables(col_blocks, table_rects)
                        col_elements = self._process_blocks(filtered_col_blocks, page_num + 1)
                        
                        # Add column number to metadata
                        for element in col_elements:
                            element['column'] = col_idx + 1
                        
                        page_elements.extend(col_elements)
                else:
                    # Fall back to standard processing if column detection failed
                    standard_elements = self._process_blocks(filtered_blocks, page_num + 1)
                    page_elements.extend(standard_elements)
            else:
                # Process as single column
                standard_elements = self._process_blocks(filtered_blocks, page_num + 1)
                page_elements.extend(standard_elements)
            
            # Filter out headers and footers
            page_elements = self._filter_headers_footers(page_elements, page.rect.height)
            
        except Exception as e:
            self.logger.error(f"Error processing page {page_num+1}: {str(e)}", exc_info=True)
            # Fall back to simple text extraction
            text = page.get_text()
            if text.strip():
                paragraphs = re.split(r'\n\s*\n', text)
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    element_type = 'paragraph'
                    if len(para) < 100 and (para.isupper() or para[-1] not in '.?!:;,'):
                        element_type = 'heading'
                    
                    # Process paragraph content for lists
                    processed_content = self._process_paragraph_content(para)
                    
                    page_elements.append({
                        'content': processed_content,
                        'type': element_type,
                        'page': page_num + 1
                    })
        
        return page_elements
    
    def _is_clearly_multi_column(self, blocks, page_width):
        """
        Determine if a page clearly has multiple columns using conservative heuristics.
        
        Args:
            blocks: Text blocks from PyMuPDF
            page_width: Width of the page
            
        Returns:
            Boolean indicating if page is clearly multi-column
        """
        # Need enough blocks to make a determination
        if len(blocks) < 10:
            return False
        
        # Get text blocks only
        text_blocks = [b for b in blocks if b["type"] == 0 and "lines" in b and b["lines"]]
        
        if len(text_blocks) < 5:
            return False
        
        # Get x-midpoints of all blocks
        midpoints = []
        for block in text_blocks:
            if "bbox" in block:
                x0, y0, x1, y1 = block["bbox"]
                midpoint = (x0 + x1) / 2
                midpoints.append(midpoint)
        
        if not midpoints:
            return False
        
        # Check distribution of midpoints - use histogram
        hist, bins = np.histogram(midpoints, bins=20)
        
        # Look for clear gaps in the middle area of the page
        middle_start = int(len(hist) * 0.3)  # 30% from left
        middle_end = int(len(hist) * 0.7)    # 70% from left
        middle_region = hist[middle_start:middle_end]
        
        # We need a clear gap (near-zero values) in the middle
        if len(middle_region) > 0 and np.min(middle_region) < 0.1 * np.max(hist):
            # Check for good distribution on both sides
            left_count = np.sum(hist[:middle_start])
            right_count = np.sum(hist[middle_end:])
            
            # Both left and right sides should have substantial content
            if left_count > 3 and right_count > 3:
                # Check block widths - columns typically have blocks less than 40% of page width
                narrow_blocks = 0
                for block in text_blocks:
                    if "bbox" in block:
                        x0, _, x1, _ = block["bbox"]
                        width = x1 - x0
                        if width < 0.4 * page_width:
                            narrow_blocks += 1
                
                # At least 70% of blocks should be narrow
                if narrow_blocks >= 0.7 * len(text_blocks):
                    return True
        
        return False
    
    def _detect_columns(self, blocks, page_width):
        """
        Detect column boundaries using advanced analysis.
        
        Args:
            blocks: Text blocks from PyMuPDF
            page_width: Width of the page
            
        Returns:
            List of column bounds (x0, x1) tuples
        """
        # Get text blocks only
        text_blocks = [b for b in blocks if b["type"] == 0 and "lines" in b and b["lines"]]
        
        if len(text_blocks) < 5:
            return []
        
        # Get block boundaries
        block_bounds = []
        for block in text_blocks:
            if "bbox" in block:
                x0, y0, x1, y1 = block["bbox"]
                block_bounds.append((x0, x1))
        
        # Group blocks by horizontal position
        block_bounds.sort()  # Sort by x0
        
        # Look for gaps between blocks
        all_bounds = []
        for x0, x1 in block_bounds:
            all_bounds.append((x0, "start"))
            all_bounds.append((x1, "end"))
        
        all_bounds.sort()
        
        # Find gaps
        active_blocks = 0
        potential_gaps = []
        
        for i in range(len(all_bounds)):
            pos, action = all_bounds[i]
            if action == "start":
                active_blocks += 1
            else:  # end
                active_blocks -= 1
            
            # When active_blocks is 0, we have a gap
            if active_blocks == 0 and i < len(all_bounds) - 1:
                gap_start = pos
                # Find next start position
                if i + 1 < len(all_bounds):
                    gap_end = all_bounds[i + 1][0]
                    gap_width = gap_end - gap_start
                    
                    # Only consider significant gaps
                    if gap_width > 0.05 * page_width:
                        potential_gaps.append((gap_start, gap_end))
        
        # If we don't have any significant gaps, this is likely not a multi-column page
        if not potential_gaps or len(potential_gaps) < 1:
            return []
        
        # Group gaps that are close together
        merged_gaps = []
        if potential_gaps:
            current_gap = potential_gaps[0]
            
            for i in range(1, len(potential_gaps)):
                if potential_gaps[i][0] - current_gap[1] < 0.05 * page_width:
                    # Merge gaps
                    current_gap = (current_gap[0], potential_gaps[i][1])
                else:
                    merged_gaps.append(current_gap)
                    current_gap = potential_gaps[i]
            
            merged_gaps.append(current_gap)
        
        # Create column bounds
        if not merged_gaps:
            return []
        
        column_bounds = []
        column_start = 0
        
        for gap_start, gap_end in merged_gaps:
            column_bounds.append((column_start, gap_start))
            column_start = gap_end
        
        # Add the last column
        column_bounds.append((column_start, page_width))
        
        # Verify that we have reasonable columns (not too narrow)
        valid_bounds = []
        for x0, x1 in column_bounds:
            width = x1 - x0
            if width > 0.15 * page_width:  # Column should be at least 15% of page width
                valid_bounds.append((x0, x1))
        
        return valid_bounds if len(valid_bounds) >= 2 else []
    
    def _filter_blocks_by_tables(self, blocks, table_rects):
        """
        Filter blocks that overlap with tables.
        
        Args:
            blocks: List of text blocks
            table_rects: List of table rectangles
            
        Returns:
            Filtered list of blocks
        """
        if not table_rects:
            return blocks
            
        filtered_blocks = []
        for block in blocks:
            if "bbox" in block:
                block_bbox = block["bbox"]
                # Check if block overlaps significantly with any table
                overlaps_table = False
                for table_rect in table_rects:
                    x_overlap = max(0, min(block_bbox[2], table_rect[2]) - max(block_bbox[0], table_rect[0]))
                    y_overlap = max(0, min(block_bbox[3], table_rect[3]) - max(block_bbox[1], table_rect[1]))
                    overlap_area = x_overlap * y_overlap
                    block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
                    
                    if block_area > 0 and overlap_area / block_area > 0.5:  # 50% overlap threshold
                        overlaps_table = True
                        break
                
                if not overlaps_table:
                    filtered_blocks.append(block)
            else:
                filtered_blocks.append(block)
                
        return filtered_blocks
    
    def _process_blocks(self, blocks, page_num):
        """
        Process text blocks from PyMuPDF extraction.
        
        Args:
            blocks: List of text blocks
            page_num: Page number
            
        Returns:
            List of paragraph data
        """
        elements = []
        
        # Sort blocks by y-position (top to bottom)
        blocks = sorted(blocks, key=lambda b: b["bbox"][1] if "bbox" in b else 0)
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            if "lines" not in block:
                continue
            
            # Extract text from the block
            text = ""
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text:
                    text += line_text + " "
            
            text = text.strip()
            if not text:
                continue
            
            # Determine element type
            element_type = 'paragraph'  # Default
            metadata = {}
            
            # Store position info in metadata for header/footer detection
            metadata['bbox'] = block.get("bbox", [0, 0, 0, 0])
            
            # Get font info from first span for classification
            if block["lines"] and block["lines"][0]["spans"]:
                first_span = block["lines"][0]["spans"][0]
                font_size = first_span.get("size", 0)
                is_bold = first_span.get("font", "").lower().find("bold") >= 0
                
                # Store font information in metadata
                metadata['font_size'] = font_size
                metadata['is_bold'] = is_bold
                
                # Classify as heading
                if (is_bold and font_size > 10) or font_size > 14:
                    element_type = 'header'
                    metadata['level'] = 1 if font_size > 16 else 2
            
            # Check for list content - keep as one paragraph but mark as list
            if self._contains_list_items(text):
                element_type = 'list'
                metadata['contains_list'] = True
            
            # Check for table of contents entry
            elif self._is_toc_entry(text):
                element_type = 'toc_entry'
            
            element = {
                'content': text,
                'type': element_type,
                'page': page_num,
                'metadata': metadata
            }
            
            elements.append(element)
        
        return elements
    
    def _contains_list_items(self, text: str) -> bool:
        """
        Check if text contains list items.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains list items, False otherwise
        """
        lines = text.split('\n')
        for line in lines:
            if self._is_list_item(line.strip()):
                return True
        return False
    
    def _filter_headers_footers(self, elements, page_height):
        """
        Filter out headers and footers based on position.
        
        Args:
            elements: List of page elements
            page_height: Height of the page
            
        Returns:
            Filtered list of elements
        """
        if not elements:
            return []
        
        # Consider the top 10% of the page as header area and bottom 10% as footer area
        header_zone = page_height * 0.1
        footer_zone = page_height * 0.9
        
        # Filter elements
        filtered = []
        for element in elements:
            # Skip if no metadata or bbox
            if 'metadata' not in element or 'bbox' not in element['metadata']:
                filtered.append(element)
                continue
                
            # Get y-position
            y_pos = element['metadata']['bbox'][1]
            
            # Skip likely headers and footers
            if y_pos < header_zone and len(element['content']) < 100:
                # This is likely a header
                element['type'] = 'header'
            elif y_pos > footer_zone and len(element['content']) < 100:
                # This is likely a footer
                element['type'] = 'footer'
            
            filtered.append(element)
        
        return filtered
    
    def _format_table(self, table):
        """
        Format a table as a string.
        
        Args:
            table: Table data
            
        Returns:
            Formatted table string
        """
        if not table:
            return ""
        
        # Filter out empty/None cells and replace with empty string
        formatted_table = [[cell if cell else '' for cell in row] for row in table]
        
        # Convert to string representation
        result = []
        for row in formatted_table:
            result.append(" | ".join(str(cell).strip() for cell in row))
        
        return "\n".join(result)
    
    def _is_scanned_pdf(self, doc):
        """
        Check if PDF appears to be a scan-only document.
        
        Args:
            doc: PyMuPDF document
            
        Returns:
            Boolean indicating if document is likely scanned
        """
        # Sample a few pages
        pages_to_check = min(3, len(doc))
        text_count = 0
        
        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text()
            if len(text.strip()) > 50:  # Arbitrary threshold
                text_count += 1
        
        # If most sampled pages have meaningful text, it's probably not scan-only
        return text_count < pages_to_check / 2
    
    def _process_toc(self, toc):
        """
        Process table of contents data from PyMuPDF.
        
        Args:
            toc: Table of contents data
            
        Returns:
            Formatted table of contents string
        """
        if not toc:
            return ""
            
        result = ["Table of Contents"]
        
        for item in toc:
            level, title, page = item[:3]
            indent = "  " * (level - 1)
            result.append(f"{indent}{title} .................. {page}")
        
        return "\n".join(result)
    
    def _is_toc_entry(self, text):
        """
        Check if text is a table of contents entry.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is a TOC entry
        """
        text = text.strip()
        if not text:
            return False
            
        # Common TOC patterns
        has_dotted_line = '..' in text
        has_page_number = bool(re.search(r'\s\d+\s*$', text))
        
        return has_dotted_line and has_page_number
    
    def _split_long_text(self, text):
        """
        Split very long text into paragraph-like chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Find sentence boundaries
        sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
        
        if not sentence_boundaries:
            return [text]  # No clear sentences found
        
        # Group sentences into paragraph-like chunks
        chunks = []
        start = 0
        
        # Aim for paragraphs of ~500 characters
        target_chunk_size = 500
        current_size = 0
        
        for boundary in sentence_boundaries:
            current_size += boundary - start
            
            if current_size >= target_chunk_size:
                chunks.append(text[start:boundary].strip())
                start = boundary
                current_size = 0
        
        # Add the last chunk
        if start < len(text):
            chunks.append(text[start:].strip())
        
        return chunks