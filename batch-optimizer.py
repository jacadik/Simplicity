"""
Optimized DocumentBatchProcessor with improved memory management and parallelization.
This module handles processing batches of documents efficiently with optimized resource utilization.
"""

import os
import logging
import time
import shutil
import threading
import gc
import psutil
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from werkzeug.utils import secure_filename
from sqlalchemy.orm import Session

from utils.thread_pool_manager import ThreadPoolManager
from utils.document_parser import DocumentParser
from utils.document_metadata_extractor import DocumentMetadataExtractor
from utils.database.manager import DatabaseManager

# Thread-local storage for database sessions
local_sessions = threading.local()

class OptimizedDocumentBatchProcessor:
    """
    Optimized version of DocumentBatchProcessor with improved memory management
    and more efficient parallel processing.
    """
    def __init__(self, 
                 db_manager: DatabaseManager,
                 document_parser: DocumentParser,
                 metadata_extractor: DocumentMetadataExtractor,
                 upload_folder: str,
                 max_workers: int = None,
                 batch_size: int = 5,
                 logging_level: str = 'INFO'):
        """
        Initialize the optimized document batch processor.
        
        Args:
            db_manager: Database manager instance
            document_parser: Document parser instance
            metadata_extractor: Metadata extractor instance
            upload_folder: Path to the upload folder
            max_workers: Maximum number of worker threads (default: CPU count-based)
            batch_size: Number of documents to process in one batch
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.db_manager = db_manager
        self.document_parser = document_parser
        self.metadata_extractor = metadata_extractor
        self.upload_folder = upload_folder
        
        # Calculate optimal max_workers if not provided
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            # Leave at least one CPU for the main thread and other system processes
            max_workers = max(1, min(cpu_count - 1, 8))
            
        self.thread_pool = ThreadPoolManager(max_workers=max_workers, logging_level=logging_level)
        self.logger = self._setup_logger(logging_level)
        self.batch_size = batch_size
        
        # Performance metrics
        self.performance_stats = {}
        
        # Memory monitoring
        self.memory_threshold = 0.85  # 85% memory usage triggers cleanup
        
    def process_uploaded_files(self, files: List, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process uploaded files in batches with optimized memory management.
        
        Args:
            files: List of file objects from request.files
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        gc.collect()  # Run garbage collection before starting
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
        
        # Process documents in batches to manage memory better
        return self._process_in_batches(document_infos, progress_callback)
    
    def process_batch_documents(self, document_infos: List[Dict[str, Any]], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of prepared document information in optimized batches.
        
        Args:
            document_infos: List of dictionaries containing document information
                Each dict should have: filename, file_path, file_type
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        gc.collect()  # Run garbage collection before starting
        total_documents = len(document_infos)
        self.logger.info(f"Starting to process {total_documents} prepared documents")
        
        # Process documents in batches
        return self._process_in_batches(document_infos, progress_callback)
    
    def _process_in_batches(self, document_infos: List[Dict[str, Any]], progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """
        Process documents in smaller batches to manage memory better.
        
        Args:
            document_infos: List of document information dictionaries
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        total_documents = len(document_infos)
        
        # Initialize result tracking
        all_results = []
        documents_processed = 0
        success_count = 0
        
        # Split documents into batches
        batches = [document_infos[i:i+self.batch_size] for i in range(0, len(document_infos), self.batch_size)]
        self.logger.info(f"Split {total_documents} documents into {len(batches)} batches of size {self.batch_size}")
        
        # Track batch processing time
        batch_times = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            
            # Check memory before processing batch
            self._check_memory_usage()
            
            # Process the batch
            batch_results = self.thread_pool.process_batch(
                batch, 
                self._process_single_document,
                progress_callback
            )
            
            # Update result tracking
            all_results.extend(batch_results)
            batch_success_count = sum(1 for result in batch_results if result.get('success', False))
            success_count += batch_success_count
            documents_processed += len(batch)
            
            # Calculate batch processing time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            avg_batch_time = sum(batch_times) / len(batch_times)
            
            # Log batch completion
            self.logger.info(
                f"Completed batch {batch_idx+1}/{len(batches)}: "
                f"{batch_success_count}/{len(batch)} successes "
                f"in {batch_time:.2f}s (avg: {avg_batch_time:.2f}s)"
            )
            
            # Estimate remaining time
            remaining_batches = len(batches) - (batch_idx + 1)
            if remaining_batches > 0:
                est_remaining_time = remaining_batches * avg_batch_time
                self.logger.info(f"Estimated remaining time: {est_remaining_time:.2f}s")
            
            # Run garbage collection after each batch
            gc.collect()
        
        # Calculate overall processing time and statistics
        elapsed_time = time.time() - start_time
        
        # Update performance statistics
        self.performance_stats = {
            'total_time': elapsed_time,
            'documents_processed': documents_processed,
            'success_count': success_count,
            'avg_time_per_document': elapsed_time / total_documents if total_documents else 0,
            'batch_count': len(batches),
            'batch_times': batch_times,
            'avg_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0
        }
        
        self.logger.info(f"Completed processing {total_documents} documents in {elapsed_time:.2f}s")
        
        return {
            'success': success_count > 0,
            'total': total_documents,
            'processed': success_count,
            'elapsed_time': elapsed_time,
            'results': all_results,
            'performance_stats': self.performance_stats
        }
    
    def _get_session(self) -> Session:
        """
        Get a thread-local session or create a new one if it doesn't exist.
        
        Returns:
            SQLAlchemy session
        """
        if not hasattr(local_sessions, 'session'):
            local_sessions.session = self.db_manager.Session()
        return local_sessions.session
    
    def _process_single_document(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document in a worker thread using thread-local session
        with savepoints for isolation.
        
        Args:
            doc_info: Dictionary with document information
            
        Returns:
            Dictionary with processing results
        """
        filename = doc_info['filename']
        file_path = doc_info['file_path']
        file_type = doc_info['file_type']
        
        self.logger.info(f"Processing document in thread {threading.get_ident()}: {filename}")
        
        start_time = time.time()
        
        try:
            # Get thread-local session
            session = self._get_session()
            
            # Create a savepoint for this document's transaction
            if hasattr(session, 'begin_nested'):
                savepoint = session.begin_nested()
            else:
                # For databases that don't support savepoints, use plain transaction
                savepoint = session.begin()
            
            try:
                # Step 1: Add document to database
                doc_id = self._add_document(session, filename, file_type, file_path)
                
                if doc_id <= 0:
                    self.logger.error(f"Failed to add document to database: {filename}")
                    savepoint.rollback()
                    return {
                        'success': False,
                        'error': 'Failed to add document to database',
                        'filename': filename,
                        'processing_time': time.time() - start_time
                    }
                
                # Step 2: Extract and store metadata
                try:
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    if metadata:
                        metadata_saved = self._add_document_metadata(session, doc_id, metadata)
                        if not metadata_saved:
                            self.logger.warning(f"Failed to save metadata for document {filename}")
                except Exception as e:
                    self.logger.error(f"Error extracting metadata from {filename}: {str(e)}")
                
                # Step 3: Parse document and extract paragraphs
                paragraphs = self.document_parser.parse_document(file_path, doc_id)
                
                if not paragraphs:
                    self.logger.warning(f"No paragraphs extracted from {filename}")
                    # Still commit this document even if no paragraphs were found
                    savepoint.commit()
                    session.commit()
                    
                    processing_time = time.time() - start_time
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0,
                        'processing_time': processing_time
                    }
                
                # Step 4: Add paragraphs to database
                paragraph_ids = self._add_paragraphs(session, paragraphs)
                
                if not paragraph_ids:
                    self.logger.warning(f"Failed to add paragraphs from {filename}")
                    savepoint.commit()
                    session.commit()
                    
                    processing_time = time.time() - start_time
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0,
                        'processing_time': processing_time
                    }
                
                # Commit this document's changes
                savepoint.commit()
                session.commit()
                
                processing_time = time.time() - start_time
                self.logger.info(f"Processed {len(paragraphs)} paragraphs from {filename} in {processing_time:.2f}s")
                
                return {
                    'success': True,
                    'doc_id': doc_id,
                    'filename': filename,
                    'paragraph_count': len(paragraphs),
                    'processing_time': processing_time
                }
            except Exception as e:
                savepoint.rollback()
                self.logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
                
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'error': str(e),
                    'filename': filename,
                    'processing_time': processing_time
                }
        except Exception as e:
            self.logger.error(f"Session error processing document {filename}: {str(e)}", exc_info=True)
            # Try to close the session and create a new one for future operations
            self._cleanup_session()
            
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'processing_time': processing_time
            }
    
    def _check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed."""
        memory = psutil.virtual_memory()
        if memory.percent > (self.memory_threshold * 100):
            self.logger.warning(f"Memory usage high ({memory.percent}%). Running cleanup.")
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up memory by running garbage collection and clearing sessions."""
        # Close all thread-local sessions
        self._cleanup_all_sessions()
        
        # Run garbage collection forcefully
        collected = gc.collect(2)  # Full collection
        self.logger.info(f"Garbage collection freed {collected} objects")
    
    def _cleanup_session(self):
        """Close the thread-local session if it exists."""
        if hasattr(local_sessions, 'session'):
            try:
                local_sessions.session.close()
            except:
                pass
            delattr(local_sessions, 'session')
    
    def _cleanup_all_sessions(self):
        """Close all thread-local sessions."""
        # Unfortunately, there's no direct way to access all thread-local objects
        # This is a placeholder for an implementation that would depend on the specifics
        # of how thread-local sessions are managed in your application
        self._cleanup_session()  # At minimum, cleanup the current thread
        gc.collect()  # This will help free memory from unused thread-locals
    
    def _add_document(self, session: Session, filename: str, file_type: str, file_path: str) -> int:
        """Add a document to the database using the provided session."""
        from utils.database.models import Document
        
        try:
            # Create new document record
            upload_date = datetime.now().isoformat()
            
            document = Document(
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                upload_date=upload_date
            )
            
            session.add(document)
            session.flush()  # Flush to get the ID but don't commit yet
            
            return document.id
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}", exc_info=True)
            return -1
    
    def _add_document_metadata(self, session: Session, document_id: int, metadata: Dict) -> bool:
        """Add document metadata using the provided session."""
        from utils.database.models import DocumentFileMetadata
        import json
        
        try:
            # Process metadata - convert lists/dicts to JSON strings
            metadata_to_store = {}
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata_to_store[key] = json.dumps(value)
                else:
                    metadata_to_store[key] = value
            
            # Check if metadata already exists
            existing = session.query(DocumentFileMetadata).filter_by(document_id=document_id).first()
            
            if existing:
                # Update existing metadata
                for key, value in metadata_to_store.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new metadata entry
                metadata_to_store['document_id'] = document_id
                metadata_obj = DocumentFileMetadata(**metadata_to_store)
                session.add(metadata_obj)
            
            session.flush()
            return True
        except Exception as e:
            self.logger.error(f"Error adding file metadata: {str(e)}", exc_info=True)
            return False
    
    def _add_paragraphs(self, session: Session, paragraphs: List) -> List[int]:
        """Add paragraphs to the database using the provided session."""
        from utils.database.models import Paragraph
        
        try:
            paragraph_ids = []
            
            for para in paragraphs:
                # Create new paragraph record
                db_paragraph = Paragraph(
                    content=para.content,
                    document_id=para.doc_id,
                    paragraph_type=para.paragraph_type,
                    position=para.position,
                    header_content=para.header_content,
                    column=para.column if hasattr(para, 'column') else None
                )
                
                session.add(db_paragraph)
                # Flush to get the ID but don't commit yet
                session.flush()
                paragraph_ids.append(db_paragraph.id)
            
            return paragraph_ids
        except Exception as e:
            self.logger.error(f"Error adding paragraphs: {str(e)}", exc_info=True)
            return []
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if a file has an allowed extension."""
        ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.OptimizedDocumentBatchProcessor')
        
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the last run."""
        return self.performance_stats.copy()

# Example usage:
# batch_processor = OptimizedDocumentBatchProcessor(
#     db_manager=db_manager,
#     document_parser=document_parser,
#     metadata_extractor=metadata_extractor,
#     upload_folder=UPLOAD_FOLDER,
#     max_workers=4,
#     batch_size=5
# )
# results = batch_processor.process_uploaded_files(files, progress_callback=progress_callback)
