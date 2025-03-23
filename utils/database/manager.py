import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy import func, distinct
import time
from sqlalchemy.orm import Session

from .base_manager import BaseManager
from .document_manager import DocumentManager
from .paragraph_manager import ParagraphManager
from .tag_manager import TagManager
from .similarity_manager import SimilarityManager
from .cluster_manager import ClusterManager
from .export_manager import ExportManager
from .insert_manager import InsertManager
from .models import Tag

from utils.document_parser import Paragraph as ParserParagraph
from utils.similarity_analyzer import SimilarityResult as AnalyzerSimilarityResult





# Other imports...
from .models import Document, Paragraph, Tag, SimilarityResult, Cluster, cluster_paragraphs, paragraph_tags

class DatabaseManager:
    """
    Main database manager that combines all specialized managers.
    Provides backward compatibility with the old interface.
    """
    def __init__(self, db_url: str, logging_level: str = 'INFO'):
        """
        Initialize the database manager with specialized manager components.
        
        Args:
            db_url: Database connection URL
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Create base manager
        self.base_manager = BaseManager(db_url, logging_level)
        self.logger = self.base_manager.logger
        
        # Create specialized managers
        self.document_manager = DocumentManager(self.base_manager)
        self.paragraph_manager = ParagraphManager(self.base_manager)
        self.similarity_manager = SimilarityManager(self.base_manager)
        self.export_manager = ExportManager(self.base_manager)
        
        # For tag manager, pass paragraph manager to handle duplicate paragraph tagging
        self.tag_manager = TagManager(self.base_manager, self.paragraph_manager)
        
        self.cluster_manager = ClusterManager(self.base_manager)
        self.insert_manager = InsertManager(self.base_manager)
        
        # Initialize database with default tags
        self._init_default_tags()
        
        # For backward compatibility
        self.Session = self.base_manager.Session
        self.engine = self.base_manager.engine
    
    def _init_default_tags(self) -> None:
        """Initialize database with default tags if they don't exist."""
        self.logger.info("Initializing default tags")
        
        def create_default_tags(session):
            default_tags = [
                ('Header', '#007bff'),
                ('Footer', '#6c757d'),
                ('Disclaimer', '#dc3545'),
                ('Important', '#ffc107'),
                ('Common', '#28a745')
            ]
            
            for name, color in default_tags:
                if not session.query(Tag).filter_by(name=name).first():
                    session.add(Tag(name=name, color=color))
            
            session.commit()
            self.logger.info("Default tags initialized")
        
        self.base_manager._with_session(create_default_tags)
    
    # Document methods
    def add_document(self, filename: str, file_type: str, file_path: str) -> int:
        """Add a document to the database and return its ID."""
        return self.document_manager.add_document(filename, file_type, file_path)
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        return self.document_manager.get_documents()
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its paragraphs."""
        return self.document_manager.delete_document(document_id)
    
    # Document metadata methods
    def add_document_file_metadata(self, document_id: int, metadata: Dict[str, Any]) -> bool:
        """Add file metadata for a document."""
        return self.document_manager.add_document_file_metadata(document_id, metadata)
    
    def get_document_file_metadata(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get file metadata for a document."""
        return self.document_manager.get_document_file_metadata(document_id)
    
    def delete_document_file_metadata(self, document_id: int) -> bool:
        """Delete file metadata for a document."""
        return self.document_manager.delete_document_file_metadata(document_id)
    
    # Paragraph methods
    def add_paragraphs(self, paragraphs: List[ParserParagraph]) -> List[int]:
        """Add paragraphs to the database and return their IDs."""
        return self.paragraph_manager.add_paragraphs(paragraphs)
    
    def get_paragraphs(self, document_id: Optional[int] = None, collapse_duplicates: bool = True) -> List[Dict[str, Any]]:
        """Get paragraphs, optionally filtered by document."""
        return self.paragraph_manager.get_paragraphs(document_id, collapse_duplicates)
    
    # Similarity methods
    def clear_similarity_results(self) -> bool:
        """Clear all similarity results from the database."""
        return self.similarity_manager.clear_similarity_results()
    
    def add_similarity_results(self, results: List[AnalyzerSimilarityResult]) -> bool:
        """Add similarity results to the database."""
        return self.similarity_manager.add_similarity_results(results)
    
    def get_similar_paragraphs(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get similar paragraphs above the threshold."""
        return self.similarity_manager.get_similar_paragraphs(threshold)
    
    # Tag methods
    def get_tags(self) -> List[Dict[str, Any]]:
        """Get all tags with accurate usage counts."""
        return self.tag_manager.get_tags()
    
    def add_tag(self, name: str, color: str) -> int:
        """Add a new tag and return its ID."""
        return self.tag_manager.add_tag(name, color)
    
    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag and all its associations."""
        return self.tag_manager.delete_tag(tag_id)
    
    def tag_paragraph(self, paragraph_id: int, tag_id: int, tag_all_duplicates: bool = False) -> bool:
        """Associate a tag with a paragraph, and optionally tag all duplicate paragraphs."""
        return self.tag_manager.tag_paragraph(paragraph_id, tag_id, tag_all_duplicates)
    
    def untag_paragraph(self, paragraph_id: int, tag_id: int, untag_all_duplicates: bool = False) -> bool:
        """Remove a tag association from a paragraph, and optionally from all duplicate paragraphs."""
        return self.tag_manager.untag_paragraph(paragraph_id, tag_id, untag_all_duplicates)
    
    # Cluster methods
    def create_cluster(self, name: str, description: str, similarity_threshold: float, 
                       similarity_type: str = 'content') -> int:
        """Create a new cluster and return its ID."""
        return self.cluster_manager.create_cluster(name, description, similarity_threshold, similarity_type)
    
    def add_paragraphs_to_cluster(self, cluster_id: int, paragraph_ids: List[int]) -> bool:
        """Add paragraphs to a cluster."""
        return self.cluster_manager.add_paragraphs_to_cluster(cluster_id, paragraph_ids)
    
    def get_clusters(self) -> List[Dict[str, Any]]:
        """Get all clusters."""
        return self.cluster_manager.get_clusters()
    
    def get_cluster_paragraphs(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get paragraphs in a specific cluster."""
        return self.cluster_manager.get_cluster_paragraphs(cluster_id)
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster."""
        return self.cluster_manager.delete_cluster(cluster_id)
    
    def clear_all_clusters(self) -> bool:
        """Delete all clusters in the database."""
        return self.cluster_manager.clear_all_clusters()
    
    # Export methods
    def export_to_excel(self, output_path: str) -> bool:
        """Export database contents to Excel."""
        return self.export_manager.export_to_excel(output_path)
    
    # Insert methods
    def add_insert(self, name: str, filename: str, file_type: str, file_path: str) -> int:
        """Add an insert to the database and return its ID."""
        return self.insert_manager.add_insert(name, filename, file_type, file_path)
    
    def add_insert_pages(self, pages: List[Dict]) -> List[int]:
        """Add insert pages to the database."""
        return self.insert_manager.add_insert_pages(pages)
    
    def get_inserts(self) -> List[Dict[str, Any]]:
        """Get all inserts with page counts."""
        return self.insert_manager.get_inserts()
    
    def get_insert_pages(self, insert_id: int) -> List[Dict[str, Any]]:
        """Get pages for a specific insert."""
        return self.insert_manager.get_insert_pages(insert_id)
    
    def get_paginated_documents(self, page: int = 1, per_page: int = 10) -> Tuple[List[Dict[str, Any]], int]:
        self.logger.info(f"Retrieving paginated documents (page {page}, {per_page} per page)")
    
        def db_get_paginated_documents(session: Session) -> Tuple[List[Dict[str, Any]], int]:
            # Get total count first
            total_count = session.query(func.count(Document.id)).scalar()
        
            # Then get only the documents for the requested page
            offset = (page - 1) * per_page
        
            documents = session.query(Document)\
                .order_by(Document.upload_date.desc())\
                .offset(offset)\
                .limit(per_page)\
                .all()
        
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
        
            self.logger.info(f"Retrieved {len(document_dicts)} documents (page {page} of {(total_count + per_page - 1) // per_page})")
            return document_dicts, total_count
    
        try:
            return self.base_manager._with_session(db_get_paginated_documents)
        except Exception as e:
            self.logger.error(f"Error retrieving paginated documents: {str(e)}")
            return [], 0
    
    import time  # Add this at the top of your file

    def get_document_statistics(self) -> Dict[str, int]:
        self.logger.info("Retrieving document statistics with direct queries")
    
        def db_get_statistics(session: Session) -> Dict[str, int]:
            stats = {}
        
            # Get document count
            stats['total_documents'] = session.query(func.count(Document.id)).scalar() or 0
        
            # Get total paragraph count
            stats['total_paragraphs'] = session.query(func.count(Paragraph.id)).scalar() or 0
        
            # Count distinct paragraph contents (unique paragraphs)
            stats['unique_paragraphs'] = session.query(func.count(func.distinct(Paragraph.content))).scalar() or 0
        
            # Count paragraphs with duplicates
            # First find content that appears more than once
            duplicates_subquery = session.query(
                Paragraph.content, 
                func.count(Paragraph.id).label('content_count')
            ).group_by(
                Paragraph.content
            ).having(
                func.count(Paragraph.id) > 1
            ).subquery()
        
            # Count the number of duplicate content items
            stats['duplicates'] = session.query(func.count()).select_from(duplicates_subquery).scalar() or 0
        
            self.logger.info(f"Calculated statistics: {stats}")
            return stats
    
        try:
            return self.base_manager._with_session(db_get_statistics)
        except Exception as e:
            self.logger.error(f"Error getting document statistics: {str(e)}", exc_info=True)
            return {
                'total_documents': 0,
                'total_paragraphs': 0,
                'duplicates': 0,
                'unique_paragraphs': 0
            }

    def _get_cached_statistics(self) -> Optional[Dict[str, int]]:
        """Get cached statistics if available and not expired."""
        # Simple implementation using a class attribute with timestamp
        current_time = time.time()
    
        # Cache valid for 5 minutes (300 seconds)
        if hasattr(self, '_stats_cache') and hasattr(self, '_stats_cache_time'):
            if current_time - self._stats_cache_time < 300:  # 5 minutes
                return self._stats_cache
    
        return None
    
    def _cache_statistics(self, stats: Dict[str, int]) -> None:
        """Cache statistics with current timestamp."""
        self._stats_cache = stats
        self._stats_cache_time = time.time()
    
    # Database cleaning
    def clear_database(self) -> bool:
        self.logger.info("Clearing all data from database")
    
        def db_clear_database(session: Session) -> List[str]:
            # Get all file paths to delete files as well
            documents = session.query(Document).all()
            file_paths = [doc.file_path for doc in documents]
        
            # Delete all records through cascade using direct SQL with text() wrapper
            # This ensures a proper deletion order respecting foreign key constraints
        
            # Clear similarity results first
            session.execute(text("DELETE FROM similarity_results"))
        
            # Clear association tables next
            session.execute(text("DELETE FROM paragraph_tags"))
            session.execute(text("DELETE FROM cluster_paragraphs"))
        
            # Delete clusters
            session.execute(text("DELETE FROM clusters"))
        
            # Delete paragraphs
            session.execute(text("DELETE FROM paragraphs"))
        
            # Delete document metadata
            session.execute(text("DELETE FROM document_file_metadata"))
        
            # Delete documents
            session.execute(text("DELETE FROM documents"))
        
            # Delete insert pages
            session.execute(text("DELETE FROM insert_pages"))
        
            # Delete inserts
            session.execute(text("DELETE FROM inserts"))
        
            # Don't delete tags - this is intentional to preserve tag definitions
        
            session.commit()
            return file_paths
    
        try:
            file_paths = self.base_manager._with_session(db_clear_database)
        
            # Delete all files from disk
            for file_path in file_paths:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Deleted file: {file_path}")
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Could not delete file {file_path}: {str(e)}")
        
            self.logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}", exc_info=True)
            return False
