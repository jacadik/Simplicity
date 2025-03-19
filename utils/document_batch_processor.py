import os
import logging
import time
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from werkzeug.utils import secure_filename

from utils.thread_pool_manager import ThreadPoolManager
from utils.document_parser import DocumentParser
from utils.document_metadata_extractor import DocumentMetadataExtractor
from utils.database_manager import DatabaseManager

class DocumentBatchProcessor:
    """
    Handles processing batches of documents with multi-threading support.
    """
    def __init__(self, 
                 db_manager: DatabaseManager,
                 document_parser: DocumentParser,
                 metadata_extractor: DocumentMetadataExtractor,
                 upload_folder: str,
                 max_workers: int = None,
                 logging_level: str = 'INFO'):
        """
        Initialize the document batch processor.
        
        Args:
            db_manager: Database manager instance
            document_parser: Document parser instance
            metadata_extractor: Metadata extractor instance
            upload_folder: Path to the upload folder
            max_workers: Maximum number of worker threads
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.db_manager = db_manager
        self.document_parser = document_parser
        self.metadata_extractor = metadata_extractor
        self.upload_folder = upload_folder
        self.thread_pool = ThreadPoolManager(max_workers=max_workers, logging_level=logging_level)
        self.logger = self._setup_logger(logging_level)
        
    def process_uploaded_files(self, files: List, progress_callback=None) -> Dict[str, Any]:
        """
        Process uploaded files in parallel.
        
        Args:
            files: List of file objects from request.files
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        self.logger.info(f"Starting to process {len(files)} uploaded files")
        
        # Step 1: Save all files to disk first
        document_infos = []
        
        for file in files:
            if file and self._allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.upload_folder, filename)
                
                # Add timestamp to filename if it already exists
                if os.path.exists(file_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    filename = f"{name}_{timestamp}{ext}"
                    file_path = os.path.join(self.upload_folder, filename)
                
                # Save the uploaded file
                file.save(file_path)
                
                # Prepare document info for processing
                document_infos.append({
                    'filename': filename,
                    'file_path': file_path,
                    'file_type': filename.rsplit('.', 1)[1].lower()
                })
        
        # Step 2: Process all documents in parallel
        results = self.thread_pool.process_batch(
            document_infos, 
            self._process_single_document,
            progress_callback
        )
        
        # Step 3: Prepare summary
        success_count = sum(1 for result in results if result.get('success', False))
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed processing {len(files)} files in {elapsed_time:.2f} seconds")
        
        return {
            'success': success_count > 0,
            'total': len(files),
            'processed': success_count,
            'elapsed_time': elapsed_time,
            'results': results
        }
    
    def process_folder(self, folder_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process all compatible files in a folder.
        
        Args:
            folder_path: Path to the folder
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        self.logger.info(f"Starting to process files from folder: {folder_path}")
        
        # Step 1: Get all allowed files in the folder
        document_infos = []
        
        for filename in os.listdir(folder_path):
            if self._allowed_file(filename):
                source_path = os.path.join(folder_path, filename)
                dest_path = os.path.join(self.upload_folder, filename)
                
                # Add timestamp to filename if it already exists
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    new_filename = f"{name}_{timestamp}{ext}"
                    dest_path = os.path.join(self.upload_folder, new_filename)
                else:
                    new_filename = filename
                
                # Copy the file
                shutil.copy2(source_path, dest_path)
                
                # Prepare document info for processing
                document_infos.append({
                    'filename': new_filename,
                    'file_path': dest_path,
                    'file_type': filename.rsplit('.', 1)[1].lower()
                })
        
        # Step 2: Process all documents in parallel
        if document_infos:
            results = self.thread_pool.process_batch(
                document_infos, 
                self._process_single_document,
                progress_callback
            )
            
            # Step 3: Prepare summary
            success_count = sum(1 for result in results if result.get('success', False))
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed processing {len(document_infos)} files in {elapsed_time:.2f} seconds")
            
            return {
                'success': success_count > 0,
                'total': len(document_infos),
                'processed': success_count,
                'elapsed_time': elapsed_time,
                'results': results
            }
        else:
            return {
                'success': False,
                'total': 0,
                'processed': 0,
                'elapsed_time': 0,
                'results': []
            }
    
    def _process_single_document(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document in a worker thread.
        
        Args:
            doc_info: Dictionary with document information
            
        Returns:
            Dictionary with processing results
        """
        filename = doc_info['filename']
        file_path = doc_info['file_path']
        file_type = doc_info['file_type']
        
        self.logger.info(f"Processing document in thread: {filename}")
        
        try:
            # Create a new session for this thread to ensure thread safety
            session = self.db_manager.Session()
            
            try:
                # Step 1: Add document to database
                doc_id = self.db_manager.add_document(filename, file_type, file_path)
                
                if doc_id <= 0:
                    self.logger.error(f"Failed to add document to database: {filename}")
                    return {
                        'success': False,
                        'error': 'Failed to add document to database',
                        'filename': filename
                    }
                
                # Step 2: Extract and store metadata
                try:
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    if metadata:
                        metadata_saved = self.db_manager.add_document_file_metadata(doc_id, metadata)
                        if not metadata_saved:
                            self.logger.warning(f"Failed to save metadata for document {filename}")
                except Exception as e:
                    self.logger.error(f"Error extracting metadata from {filename}: {str(e)}")
                
                # Step 3: Parse document and extract paragraphs
                paragraphs = self.document_parser.parse_document(file_path, doc_id)
                
                if not paragraphs:
                    self.logger.warning(f"No paragraphs extracted from {filename}")
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0
                    }
                
                # Step 4: Add paragraphs to database
                paragraph_ids = self.db_manager.add_paragraphs(paragraphs)
                
                if not paragraph_ids:
                    self.logger.warning(f"Failed to add paragraphs from {filename}")
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0
                    }
                
                self.logger.info(f"Processed {len(paragraphs)} paragraphs from {filename}")
                
                return {
                    'success': True,
                    'doc_id': doc_id,
                    'filename': filename,
                    'paragraph_count': len(paragraphs)
                }
            finally:
                # Always close the session
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if a file has an allowed extension."""
        ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.DocumentBatchProcessor')
        
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