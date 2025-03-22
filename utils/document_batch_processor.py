import os
import logging
import time
import shutil
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from werkzeug.utils import secure_filename
from sqlalchemy.orm import Session
from sqlalchemy import text

from utils.thread_pool_manager import ThreadPoolManager
from utils.document_parser import DocumentParser
from utils.document_metadata_extractor import DocumentMetadataExtractor
from utils.database.manager import DatabaseManager

# Thread-local storage for database sessions
local_sessions = threading.local()

class DocumentBatchProcessor:
    """
    Handles processing batches of documents with multi-threading support
    and optimized database session management.
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
        
    def process_uploaded_files(self, files: List, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process uploaded files in parallel with optimized database session handling.
        
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
        
        # Step 2: Process all documents in parallel with thread-local sessions
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
    
    def process_batch_documents(self, document_infos: List[Dict[str, Any]], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of prepared document information in parallel.
        
        Args:
            document_infos: List of dictionaries containing document information
                Each dict should have: filename, file_path, file_type
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        total_documents = len(document_infos)
        self.logger.info(f"Starting to process {total_documents} prepared documents")
        
        # Process all documents in parallel with thread-local sessions
        results = self.thread_pool.process_batch(
            document_infos, 
            self._process_single_document,
            progress_callback
        )
        
        # Prepare summary
        success_count = sum(1 for result in results if result.get('success', False))
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed processing {total_documents} documents in {elapsed_time:.2f} seconds")
        
        return {
            'success': success_count > 0,
            'total': total_documents,
            'processed': success_count,
            'elapsed_time': elapsed_time,
            'results': results
        }
    
    def process_folder(self, folder_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process all compatible files in a folder with optimized database session handling.
        
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
        
        # Step 2: Process all documents in parallel with thread-local sessions
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
        
        self.logger.info(f"Processing document in thread: {filename}")
        
        try:
            # Get thread-local session
            session = self._get_session()
            
            # Create a savepoint for this document's transaction
            if hasattr(session, 'begin_nested'):
                savepoint = session.begin_nested()
            else:
                # For databases that don't support savepoints, use plain transaction
                # This means if this document fails, all documents in this thread will rollback
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
                        'filename': filename
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
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0
                    }
                
                # Step 4: Add paragraphs to database
                paragraph_ids = self._add_paragraphs(session, paragraphs)
                
                if not paragraph_ids:
                    self.logger.warning(f"Failed to add paragraphs from {filename}")
                    savepoint.commit()
                    session.commit()
                    return {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'paragraph_count': 0
                    }
                
                # Commit this document's changes
                savepoint.commit()
                session.commit()
                
                self.logger.info(f"Processed {len(paragraphs)} paragraphs from {filename}")
                
                return {
                    'success': True,
                    'doc_id': doc_id,
                    'filename': filename,
                    'paragraph_count': len(paragraphs)
                }
            except Exception as e:
                savepoint.rollback()
                self.logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e),
                    'filename': filename
                }
        except Exception as e:
            self.logger.error(f"Session error processing document {filename}: {str(e)}", exc_info=True)
            # Try to close the session and create a new one for future operations
            self._cleanup_session()
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
    
    def _cleanup_session(self):
        """Close the thread-local session if it exists."""
        if hasattr(local_sessions, 'session'):
            try:
                local_sessions.session.close()
            except:
                pass
            delattr(local_sessions, 'session')
    
    def _add_document(self, session: Session, filename: str, file_type: str, file_path: str) -> int:
        """Add a document to the database using the provided session."""
        from utils.database_manager import Document
        
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
        from utils.database_manager import DocumentFileMetadata
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
        from utils.database_manager import Paragraph
        
        try:
            paragraph_ids = []
            
            for para in paragraphs:
                # Create new paragraph record
                db_paragraph = Paragraph(
                    content=para.content,
                    document_id=para.doc_id,
                    paragraph_type=para.paragraph_type,
                    position=para.position,
                    header_content=para.header_content
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