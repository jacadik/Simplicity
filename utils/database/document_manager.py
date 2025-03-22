import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from sqlalchemy.orm import Session
from sqlalchemy import func

from .models import Document, DocumentFileMetadata
from .base_manager import BaseManager

class DocumentManager:
    """
    Manages document-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the document manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def add_document(self, filename: str, file_type: str, file_path: str) -> int:
        """
        Add a document to the database and return its ID.
        
        Args:
            filename: The name of the file
            file_type: The file extension or MIME type
            file_path: The path where the file is stored
            
        Returns:
            The document ID if successful, -1 otherwise
        """
        self.logger.info(f"Adding document to database: {filename}")
        
        def db_add_document(session: Session) -> int:
            # Create new document record
            upload_date = datetime.now().isoformat()
            
            document = Document(
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                upload_date=upload_date
            )
            
            session.add(document)
            session.commit()
            
            doc_id = document.id
            self.logger.info(f"Document added with ID: {doc_id}")
            return doc_id
        
        try:
            return self.base_manager._with_session(db_add_document)
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return -1
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents.
        
        Returns:
            List of document dictionaries
        """
        self.logger.info("Retrieving all documents")
        
        def db_get_documents(session: Session) -> List[Dict[str, Any]]:
            # Query all documents ordered by upload date descending
            documents = session.query(Document).order_by(Document.upload_date.desc()).all()
            
            # Convert ORM objects to dictionaries
            document_dicts = [
                {
                    'id': doc.id,
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'file_path': doc.file_path,
                    'upload_date': doc.upload_date
                }
                for doc in documents
            ]
            
            self.logger.info(f"Retrieved {len(document_dicts)} documents")
            return document_dicts
        
        try:
            return self.base_manager._with_session(db_get_documents)
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and its paragraphs.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Deleting document with ID: {document_id}")
        
        def db_delete_document(session: Session) -> Union[bool, str]:
            # Get the document
            document = session.query(Document).get(document_id)
            
            if not document:
                self.logger.warning(f"Document with ID {document_id} not found")
                return False
            
            file_path = document.file_path
            
            # Delete from database (SQLAlchemy will handle the cascade)
            session.delete(document)
            session.commit()
            
            self.logger.info(f"Deleted document from database: {document_id}")
            return file_path
        
        try:
            file_path = self.base_manager._with_session(db_delete_document)
            
            if file_path and isinstance(file_path, str):
                # Try to delete the file if it exists
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Deleted file: {file_path}")
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Could not delete file {file_path}: {str(e)}")
                
                self.logger.info(f"Deleted document with ID: {document_id}")
                return True
            return False
                
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def add_document_file_metadata(self, document_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Add file metadata for a document.
        
        Args:
            document_id: ID of the document
            metadata: Dictionary of metadata attributes
                
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Adding file metadata for document ID: {document_id}")
        
        def db_add_metadata(session: Session) -> bool:
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
            
            session.commit()
            self.logger.info(f"Successfully added file metadata for document ID: {document_id}")
            return True
        
        try:
            return self.base_manager._with_session(db_add_metadata)
        except Exception as e:
            self.logger.error(f"Error adding file metadata: {str(e)}")
            return False

    def get_document_file_metadata(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get file metadata for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Dictionary of metadata attributes or None if not found
        """
        self.logger.info(f"Retrieving file metadata for document ID: {document_id}")
        
        def db_get_metadata(session: Session) -> Optional[Dict[str, Any]]:
            metadata_obj = session.query(DocumentFileMetadata).filter_by(document_id=document_id).first()
            
            if not metadata_obj:
                self.logger.info(f"No file metadata found for document ID: {document_id}")
                return None
            
            # Convert to dictionary and process JSON strings
            metadata = {}
            for column in metadata_obj.__table__.columns:
                column_name = column.name
                if column_name != 'id':  # Skip the ID field
                    value = getattr(metadata_obj, column_name)
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            metadata[column_name] = json.loads(value)
                        except json.JSONDecodeError:
                            metadata[column_name] = value
                    else:
                        metadata[column_name] = value
            
            return metadata
        
        try:
            return self.base_manager._with_session(db_get_metadata)
        except Exception as e:
            self.logger.error(f"Error retrieving file metadata: {str(e)}")
            return None

    def delete_document_file_metadata(self, document_id: int) -> bool:
        """
        Delete file metadata for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Deleting file metadata for document ID: {document_id}")
        
        def db_delete_metadata(session: Session) -> bool:
            # Find and delete metadata
            metadata = session.query(DocumentFileMetadata).filter_by(document_id=document_id).first()
            if metadata:
                session.delete(metadata)
                session.commit()
                self.logger.info(f"Successfully deleted file metadata for document ID: {document_id}")
            else:
                self.logger.info(f"No metadata found to delete for document ID: {document_id}")
            
            return True
        
        try:
            return self.base_manager._with_session(db_delete_metadata)
        except Exception as e:
            self.logger.error(f"Error deleting file metadata: {str(e)}")
            return False