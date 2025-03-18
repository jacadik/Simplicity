import os
import re
import logging
from typing import Dict, Any, Optional, List
import pdfplumber
import docx
from datetime import datetime
import fitz  # PyMuPDF


class DocumentMetadataExtractor:
    """
    Extracts metadata from document files (PDF, DOCX)
    """
    def __init__(self, logging_level: str = 'INFO'):
        """Initialize the metadata extractor."""
        self.logger = self._setup_logger(logging_level)
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive file metadata for a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing metadata attributes
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File does not exist: {file_path}")
            return {}
            
        # Get basic file information
        metadata = {
            'file_size': os.path.getsize(file_path),
            'file_size_formatted': self._format_file_size(os.path.getsize(file_path)),
            'creation_time': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
            'modification_time': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'file_extension': os.path.splitext(file_path)[1].lower(),
            'filename': os.path.basename(file_path)
        }
        
        # Extract format-specific metadata
        try:
            if metadata['file_extension'] == '.pdf':
                pdf_metadata = self._extract_pdf_metadata(file_path)
                metadata.update(pdf_metadata)
            elif metadata['file_extension'] in ['.doc', '.docx']:
                docx_metadata = self._extract_docx_metadata(file_path)
                metadata.update(docx_metadata)
        except Exception as e:
            self.logger.error(f"Error extracting detailed metadata: {str(e)}")
        
        return metadata
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract PDF-specific metadata using both pdfplumber and PyMuPDF for comprehensive results.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        pdf_metadata = {}
        
        # First use pdfplumber for certain metadata
        try:
            with pdfplumber.open(file_path) as pdf:
                # Basic PDF info
                pdf_metadata['page_count'] = len(pdf.pages)
                
                # Extract text for paragraph counting
                all_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    all_text += page_text + "\n\n"
                
                # Count paragraphs (separated by double newlines)
                paragraphs = re.split(r'\n\s*\n', all_text)
                pdf_metadata['paragraph_count'] = len([p for p in paragraphs if p.strip()])
                
                # Document info dictionary
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    for key, value in pdf.metadata.items():
                        if key and value and isinstance(value, (str, int, float, bool)):
                            cleaned_key = key.lower().replace('/', '_').replace(':', '_')
                            pdf_metadata[cleaned_key] = value
        except Exception as e:
            self.logger.warning(f"pdfplumber metadata extraction error: {str(e)}")
        
        # Then use PyMuPDF for more detailed metadata
        try:
            doc = fitz.open(file_path)
            
            # Get document metadata
            pdf_metadata['pdf_version'] = f"PDF {doc.pdf_version}"
            pdf_metadata['is_encrypted'] = doc.is_encrypted
            pdf_metadata['has_signatures'] = bool(doc.permissions & fitz.PDF_PERM_PRINT) == False  # Simplistic check
            
            # Get document metadata
            meta = doc.metadata
            if meta:
                for key, value in meta.items():
                    if key and value and isinstance(value, (str, int, float, bool)):
                        cleaned_key = key.lower().replace('/', '_').replace(':', '_')
                        pdf_metadata[cleaned_key] = value
            
            # Extract font information
            fonts = set()
            fonts_by_page = {}
            
            for page_num, page in enumerate(doc):
                page_fonts = set()
                font_list = page.get_fonts()
                
                for font in font_list:
                    font_name = font[3]  # Font name is at index 3
                    fonts.add(font_name)
                    page_fonts.add(font_name)
                
                fonts_by_page[page_num + 1] = list(page_fonts)
            
            pdf_metadata['fonts_used'] = list(fonts)
            pdf_metadata['fonts_by_page'] = fonts_by_page
            
            # Count images
            image_count = 0
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                image_count += len(image_list)
            
            pdf_metadata['image_count'] = image_count
            
            # Check for form fields
            has_forms = False
            for page in doc:
                if page.widget_count > 0:
                    has_forms = True
                    break
            
            pdf_metadata['has_forms'] = has_forms
            
            # Check for annotations
            annotation_count = 0
            for page in doc:
                annotation_count += len(page.annots())
            
            pdf_metadata['annotation_count'] = annotation_count
            
            # Check if document has TOC/bookmarks
            toc = doc.get_toc()
            pdf_metadata['has_toc'] = len(toc) > 0
            pdf_metadata['toc_items'] = len(toc)
            
            doc.close()
        except Exception as e:
            self.logger.warning(f"PyMuPDF metadata extraction error: {str(e)}")
        
        return pdf_metadata
    
    def _extract_docx_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract DOCX-specific metadata.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with DOCX metadata
        """
        docx_metadata = {}
        
        try:
            doc = docx.Document(file_path)
            
            # Get core properties
            core_props = doc.core_properties
            if core_props:
                if core_props.author:
                    docx_metadata['author'] = core_props.author
                if core_props.created:
                    docx_metadata['created'] = core_props.created.isoformat() if hasattr(core_props.created, 'isoformat') else str(core_props.created)
                if core_props.modified:
                    docx_metadata['modified'] = core_props.modified.isoformat() if hasattr(core_props.modified, 'isoformat') else str(core_props.modified)
                if core_props.title:
                    docx_metadata['title'] = core_props.title
                if core_props.subject:
                    docx_metadata['subject'] = core_props.subject
                if core_props.keywords:
                    docx_metadata['keywords'] = core_props.keywords
                if core_props.category:
                    docx_metadata['category'] = core_props.category
                if core_props.comments:
                    docx_metadata['comments'] = core_props.comments
            
            # Count document elements
            docx_metadata['paragraph_count'] = len(doc.paragraphs)
            docx_metadata['table_count'] = len(doc.tables)
            docx_metadata['page_count'] = self._estimate_page_count(doc)
            
            # Find styles used
            styles_used = set()
            for paragraph in doc.paragraphs:
                if paragraph.style:
                    styles_used.add(paragraph.style.name)
            
            docx_metadata['styles_used'] = list(styles_used)
            
            # Count images (a bit complex in docx)
            image_count = 0
            for rel in doc.part.rels.values():
                if rel.reltype.endswith('/image'):
                    image_count += 1
            
            docx_metadata['image_count'] = image_count
            
            # Count sections
            docx_metadata['section_count'] = len(doc.sections)
            
            # Check for headers and footers
            has_headers = False
            has_footers = False
            for section in doc.sections:
                if section.header.is_linked_to_previous == False and section.header.paragraphs:
                    has_headers = True
                if section.footer.is_linked_to_previous == False and section.footer.paragraphs:
                    has_footers = True
                    
            docx_metadata['has_headers'] = has_headers
            docx_metadata['has_footers'] = has_footers
            
        except Exception as e:
            self.logger.error(f"Error extracting DOCX metadata: {str(e)}")
            
        return docx_metadata
    
    def _estimate_page_count(self, doc) -> int:
        """
        Estimate page count for a DOCX document (approximate).
        
        Args:
            doc: docx.Document object
            
        Returns:
            Estimated page count
        """
        # This is a rough estimation - accurate page count requires rendering
        total_paragraphs = len(doc.paragraphs)
        total_tables = len(doc.tables)
        total_chars = sum(len(p.text) for p in doc.paragraphs)
        
        # Rough heuristic: ~3000 chars per page, ~40 paragraphs per page, ~3 tables per page
        chars_pages = total_chars / 3000
        para_pages = total_paragraphs / 40
        table_pages = total_tables / 3
        
        # Take the max of these estimates as a conservative approach
        estimated_pages = max(chars_pages, para_pages, table_pages, 1)
        return round(estimated_pages)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable form.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted file size string (e.g., "2.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.DocumentMetadataExtractor')
        
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
