import os
import logging
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable, TypeVar, cast, Union

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table, and_, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from utils.document_parser import Paragraph as ParserParagraph
from utils.similarity_analyzer import SimilarityResult as AnalyzerSimilarityResult

# Initialize SQLAlchemy Base
Base = declarative_base()

# Type variable for generic session function
T = TypeVar('T')

# Define the many-to-many association table
paragraph_tags = Table(
    'paragraph_tags', 
    Base.metadata,
    Column('paragraph_id', Integer, ForeignKey('paragraphs.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)
)

# Define the cluster_paragraphs association table
cluster_paragraphs = Table(
    'cluster_paragraphs',
    Base.metadata,
    Column('cluster_id', Integer, ForeignKey('clusters.id', ondelete='CASCADE'), primary_key=True),
    Column('paragraph_id', Integer, ForeignKey('paragraphs.id', ondelete='CASCADE'), primary_key=True)
)

class Document(Base):
    """Document model representing a file that has been uploaded and parsed."""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(String, nullable=False)
    
    # Relationships
    paragraphs = relationship("Paragraph", back_populates="document", cascade="all, delete-orphan")
    file_metadata = relationship("DocumentFileMetadata", back_populates="document", uselist=False, cascade="all, delete-orphan")

class Paragraph(Base):
    """Paragraph model representing extracted text from documents."""
    __tablename__ = 'paragraphs'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    paragraph_type = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    header_content = Column(Text, nullable=True)
    column = Column(Integer, nullable=True)  # Column position
    
    # Relationships
    document = relationship("Document", back_populates="paragraphs")
    tags = relationship("Tag", secondary=paragraph_tags, back_populates="paragraphs")
    clusters = relationship("Cluster", secondary=cluster_paragraphs, back_populates="paragraphs")
    
    # Similarity relationships
    similarity_as_para1 = relationship(
        "SimilarityResult", 
        foreign_keys="SimilarityResult.paragraph1_id",
        cascade="all, delete-orphan", 
        back_populates="paragraph1"
    )
    
    similarity_as_para2 = relationship(
        "SimilarityResult", 
        foreign_keys="SimilarityResult.paragraph2_id",
        cascade="all, delete-orphan", 
        back_populates="paragraph2"
    )

class Tag(Base):
    """Tag model for categorizing paragraphs."""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    color = Column(String, nullable=False)
    
    # Relationships
    paragraphs = relationship("Paragraph", secondary=paragraph_tags, back_populates="tags")

class SimilarityResult(Base):
    """Model for storing similarity analysis results between paragraphs."""
    __tablename__ = 'similarity_results'
    
    id = Column(Integer, primary_key=True)
    paragraph1_id = Column(Integer, ForeignKey('paragraphs.id', ondelete='CASCADE'), nullable=False)
    paragraph2_id = Column(Integer, ForeignKey('paragraphs.id', ondelete='CASCADE'), nullable=False)
    content_similarity_score = Column(Float, nullable=False)  # Renamed from similarity_score
    text_similarity_score = Column(Float, nullable=True)      # New field for character-based similarity
    similarity_type = Column(String, nullable=False)
    
    # Relationships
    paragraph1 = relationship("Paragraph", foreign_keys=[paragraph1_id], back_populates="similarity_as_para1")
    paragraph2 = relationship("Paragraph", foreign_keys=[paragraph2_id], back_populates="similarity_as_para2")
    
    # Add indices for performance
    __table_args__ = (
        Index('idx_similarity_paragraph1', 'paragraph1_id'),
        Index('idx_similarity_paragraph2', 'paragraph2_id'),
        Index('idx_content_similarity', 'content_similarity_score'),
        Index('idx_text_similarity', 'text_similarity_score'),
    )

class Cluster(Base):
    """Model for storing paragraph clusters."""
    __tablename__ = 'clusters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    creation_date = Column(String, nullable=False)
    similarity_threshold = Column(Float, nullable=False)
    similarity_type = Column(String, nullable=True)  # 'content' or 'text'
    
    # Relationships
    paragraphs = relationship("Paragraph", secondary=cluster_paragraphs, back_populates="clusters")

class DocumentFileMetadata(Base):
    """Model for storing metadata about uploaded document files."""
    __tablename__ = 'document_file_metadata'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    file_size = Column(Integer, nullable=True)
    file_size_formatted = Column(String, nullable=True)
    creation_date = Column(String, nullable=True)
    modification_date = Column(String, nullable=True)
    page_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    image_count = Column(Integer, nullable=True)
    author = Column(String, nullable=True)
    title = Column(String, nullable=True)
    subject = Column(String, nullable=True)
    creator = Column(String, nullable=True)
    producer = Column(String, nullable=True)
    pdf_version = Column(String, nullable=True)
    is_encrypted = Column(Integer, nullable=True)  # Using Integer for boolean in SQLite
    has_signatures = Column(Integer, nullable=True)
    has_forms = Column(Integer, nullable=True)
    has_toc = Column(Integer, nullable=True)
    toc_items = Column(Integer, nullable=True)
    annotation_count = Column(Integer, nullable=True)
    fonts_used = Column(Text, nullable=True)  # Stored as JSON array
    table_count = Column(Integer, nullable=True)
    section_count = Column(Integer, nullable=True)
    has_headers = Column(Integer, nullable=True)
    has_footers = Column(Integer, nullable=True)
    styles_used = Column(Text, nullable=True)  # Stored as JSON array
    
    # Relationship with Document
    document = relationship("Document", back_populates="file_metadata")
    
    # Index definitions
    __table_args__ = (
        Index('idx_file_metadata_document_id', 'document_id'),
        Index('idx_file_metadata_page_count', 'page_count'),
        Index('idx_file_metadata_paragraph_count', 'paragraph_count'),
        Index('idx_file_metadata_file_size', 'file_size'),
    )

class DatabaseManager:
    """
    Manages all database operations for the paragraph analysis system using PostgreSQL with SQLAlchemy.
    """
    def __init__(self, db_url: str, logging_level: str = 'INFO'):
        """
        Initialize the database manager.
        
        Args:
            db_url: Database connection URL (e.g., "postgresql://username:password@localhost/paragraph_analyzer")
            logging_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.db_url = db_url
        self.logger = self._setup_logger(logging_level)
        
        # Create engine and session factory
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._init_db()
    
    def _with_session(self, func: Callable[[Session], T]) -> T:
        """
        Execute a function with a database session, handling session lifecycle.
        
        Args:
            func: Function that takes a session as its argument
            
        Returns:
            The result of the provided function
        """
        session = self.Session()
        try:
            result = func(session)
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
    
    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        self.logger.info(f"Initializing database with SQLAlchemy using {self.db_url}")
        
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            def create_default_tags(session: Session) -> None:
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
                self.logger.info("Database initialized successfully")
            
            self._with_session(create_default_tags)
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise
    
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
            return self._with_session(db_add_document)
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return -1
    
    def add_paragraphs(self, paragraphs: List[ParserParagraph]) -> List[int]:
        """
        Add paragraphs to the database and return their IDs.
        
        Args:
            paragraphs: List of paragraph objects to add
            
        Returns:
            List of paragraph IDs if successful, empty list otherwise
        """
        if not paragraphs:
            return []
            
        self.logger.info(f"Adding {len(paragraphs)} paragraphs to database")
        
        def db_add_paragraphs(session: Session) -> List[int]:
            paragraph_ids = []
            
            for para in paragraphs:
                # Create new paragraph record
                db_paragraph = Paragraph(
                    content=para.content,
                    document_id=para.doc_id,
                    paragraph_type=para.paragraph_type,
                    position=para.position,
                    header_content=para.header_content,
                    column=para.column   # Include column information if available
                )
                
                session.add(db_paragraph)
                # Flush to get the ID but don't commit yet
                session.flush()
                paragraph_ids.append(db_paragraph.id)
            
            # Commit all paragraphs in one transaction
            session.commit()
            
            self.logger.info(f"Added {len(paragraph_ids)} paragraphs")
            return paragraph_ids
        
        try:
            return self._with_session(db_add_paragraphs)
        except Exception as e:
            self.logger.error(f"Error adding paragraphs: {str(e)}")
            return []
    
    def clear_similarity_results(self) -> bool:
        """Clear all similarity results from the database."""
        self.logger.info("Clearing all similarity results")
        
        def db_clear_similarity_results(session: Session) -> int:
            # Delete all similarity results
            count = session.query(SimilarityResult).delete()
            session.commit()
            return count
        
        try:
            count = self._with_session(db_clear_similarity_results)
            self.logger.info(f"Deleted {count} similarity results")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing similarity results: {str(e)}")
            return False
    
    def add_similarity_results(self, results: List[AnalyzerSimilarityResult]) -> bool:
        """
        Add similarity results to the database.
        
        Args:
            results: List of similarity results to add
            
        Returns:
            True if successful, False otherwise
        """
        if not results:
            self.logger.warning("No similarity results to add")
            return True
            
        self.logger.info(f"Adding {len(results)} similarity results to database")
        
        # First, clear existing similarity results
        if not self.clear_similarity_results():
            self.logger.warning("Failed to clear existing similarity results, proceeding with update")
        
        def db_add_similarity_results(session: Session) -> bool:
            for result in results:
                # Log details for debugging
                self.logger.debug(f"Processing similarity result: para1={result.paragraph1_id}, para2={result.paragraph2_id}, "
                                f"type={result.similarity_type}, content_score={result.content_similarity_score}, "
                                f"text_score={result.text_similarity_score}")
                
                # Create new similarity result record with both similarity scores
                db_result = SimilarityResult(
                    paragraph1_id=result.paragraph1_id,
                    paragraph2_id=result.paragraph2_id,
                    content_similarity_score=result.content_similarity_score,
                    text_similarity_score=result.text_similarity_score,
                    similarity_type=result.similarity_type
                )
                session.add(db_result)
            
            # Commit all similarity results in one transaction
            session.commit()
            
            self.logger.info(f"Successfully added {len(results)} similarity results")
            return True
        
        try:
            return self._with_session(db_add_similarity_results)
        except Exception as e:
            self.logger.error(f"Error adding similarity results: {str(e)}")
            return False
    
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
            return self._with_session(db_get_documents)
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_paragraphs(self, document_id: Optional[int] = None, collapse_duplicates: bool = True) -> List[Dict[str, Any]]:
        """
        Get paragraphs, optionally filtered by document.
        
        Args:
            document_id: If provided, filter paragraphs by this document ID
            collapse_duplicates: If True, identical paragraphs across documents will be collapsed
                                into a single entry with document references
                                
        Returns:
            List of paragraph dictionaries with associated metadata
        """
        def db_get_paragraphs(session: Session) -> List[Dict[str, Any]]:
            # Base query for paragraphs and their document data
            query = session.query(
                Paragraph, 
                Document.filename
            ).join(
                Document, 
                Paragraph.document_id == Document.id
            )
            
            if document_id is not None:
                self.logger.info(f"Retrieving paragraphs for document ID: {document_id}")
                query = query.filter(Paragraph.document_id == document_id)
                query = query.order_by(Paragraph.position)
            else:
                self.logger.info("Retrieving all paragraphs")
                query = query.order_by(Paragraph.document_id, Paragraph.position)
            
            results = query.all()
            
            # Process results and get tags for each paragraph
            paragraphs = []
            for para, filename in results:
                # Get tags for this paragraph
                tags = [{
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                } for tag in para.tags]
                
                # Build paragraph dictionary
                para_dict = {
                    'id': para.id,
                    'content': para.content,
                    'document_id': para.document_id,
                    'paragraph_type': para.paragraph_type,
                    'position': para.position,
                    'header_content': para.header_content,
                    'filename': filename,
                    'tags': tags
                }
                
                # Add column information if available
                if para.column is not None:
                    para_dict['column'] = para.column
                
                paragraphs.append(para_dict)
            
            # Handle duplicate collapse if requested and not filtering by document
            if collapse_duplicates and document_id is None:
                # Get exact matching paragraphs based on content
                duplicate_content_query = session.query(
                    Paragraph.content,
                    func.string_agg(func.cast(Paragraph.id, String), ',').label('para_ids'),
                    func.string_agg(func.cast(Document.id, String), ',').label('doc_ids'),
                    func.string_agg(Document.filename, ',').label('filenames')
                ).join(
                    Document, 
                    Paragraph.document_id == Document.id
                ).group_by(
                    Paragraph.content
                ).having(
                    func.count() > 1
                )
                
                duplicate_rows = duplicate_content_query.all()
                
                # Process duplicate paragraphs
                if duplicate_rows:
                    # Create a lookup of paragraph IDs that are duplicates
                    duplicate_ids = set()
                    duplicate_map = {}
                    
                    for content, para_ids_str, doc_ids_str, filenames_str in duplicate_rows:
                        para_ids = [int(pid) for pid in para_ids_str.split(',')]
                        doc_ids = [int(did) for did in doc_ids_str.split(',')]
                        filenames = filenames_str.split(',')
                        
                        # Keep track of all duplicate paragraph IDs (except the one we'll keep)
                        duplicate_ids.update(para_ids[1:])
                        
                        # Map the paragraph we'll keep to its document appearances
                        duplicate_map[para_ids[0]] = {
                            'doc_ids': doc_ids,
                            'filenames': filenames
                        }
                    
                    # Filter out duplicates and add document reference to kept paragraphs
                    filtered_paragraphs = []
                    for para in paragraphs:
                        if para['id'] in duplicate_ids:
                            # Skip this duplicate paragraph
                            continue
                        elif para['id'] in duplicate_map:
                            # This is the representative paragraph from a duplicate group
                            para['document_references'] = duplicate_map[para['id']]['filenames']
                            para['appears_in_multiple'] = True
                        else:
                            # Regular paragraph (no duplicates)
                            para['document_references'] = [para['filename']]
                            para['appears_in_multiple'] = False
                        
                        filtered_paragraphs.append(para)
                    
                    paragraphs = filtered_paragraphs
                else:
                    # No duplicates found, just add default document references
                    for para in paragraphs:
                        para['document_references'] = [para['filename']]
                        para['appears_in_multiple'] = False
            else:
                # Not collapsing duplicates, just add default document references
                for para in paragraphs:
                    para['document_references'] = [para['filename']]
                    para['appears_in_multiple'] = False
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs")
            return paragraphs
        
        try:
            return self._with_session(db_get_paragraphs)
        except Exception as e:
            self.logger.error(f"Error retrieving paragraphs: {str(e)}")
            return []
    
    def get_similar_paragraphs(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get similar paragraphs above the threshold.
        
        Args:
            threshold: Minimum similarity score to include (0.0 to 1.0)
            
        Returns:
            List of similarity result dictionaries with paragraph content
        """
        try:
            def db_get_similar_paragraphs(session: Session) -> List[Dict[str, Any]]:
                self.logger.info(f"Starting similarity query with threshold: {threshold}")
                
                # First, get all similarity results with paragraph1 data
                query = session.query(
                    SimilarityResult,
                    Paragraph.content.label('para1_content'),
                    Paragraph.document_id.label('para1_doc_id'),
                    Document.filename.label('para1_filename')
                ).join(
                    Paragraph,
                    SimilarityResult.paragraph1_id == Paragraph.id
                ).join(
                    Document,
                    Paragraph.document_id == Document.id
                )
                
                # Apply threshold filter if provided - using content_similarity_score (primary metric)
                if threshold is not None:
                    self.logger.info(f"Filtering by threshold: {threshold}")
                    query = query.filter(SimilarityResult.content_similarity_score >= threshold)
                
                # Execute query to get base results
                results = query.all()
                self.logger.info(f"Found {len(results)} similarity results")
                
                # Process results and fetch paragraph2 data separately
                similarities = []
                for sim_result, para1_content, para1_doc_id, para1_filename in results:
                    # Get paragraph2 data with a separate query
                    para2_query = session.query(
                        Paragraph.content,
                        Paragraph.document_id,
                        Document.filename
                    ).join(
                        Document,
                        Paragraph.document_id == Document.id
                    ).filter(
                        Paragraph.id == sim_result.paragraph2_id
                    )
                    
                    # Execute paragraph2 query
                    para2_result = para2_query.first()
                    
                    if para2_result:
                        para2_content, para2_doc_id, para2_filename = para2_result
                        
                        # Build the complete similarity dictionary with both similarity scores
                        similarity_dict = {
                            'id': sim_result.id,
                            'paragraph1_id': sim_result.paragraph1_id,
                            'paragraph2_id': sim_result.paragraph2_id,
                            'content_similarity_score': sim_result.content_similarity_score,  # Renamed field
                            'text_similarity_score': sim_result.text_similarity_score,        # New field
                            'similarity_type': sim_result.similarity_type,
                            'para1_content': para1_content,
                            'para1_doc_id': para1_doc_id,
                            'para1_filename': para1_filename,
                            'para2_content': para2_content,
                            'para2_doc_id': para2_doc_id,
                            'para2_filename': para2_filename
                        }
                        
                        similarities.append(similarity_dict)
                
                self.logger.info(f"Processed {len(similarities)} complete similarity records")
                return similarities
            
            return self._with_session(db_get_similar_paragraphs)
            
        except Exception as e:
            self.logger.error(f"Error retrieving similarity results: {str(e)}")
            return []
    
    def get_tags(self) -> List[Dict[str, Any]]:
        """
        Get all tags with accurate usage counts.
        
        Returns:
            List of tag dictionaries with usage counts
        """
        self.logger.info("Retrieving all tags with usage counts")
        
        def db_get_tags(session: Session) -> List[Dict[str, Any]]:
            # First get all the tags
            tags = session.query(Tag).all()
            
            tag_dicts = []
            # For each tag, count distinct paragraphs
            for tag in tags:
                # Count distinct paragraphs associated with this tag
                count = session.query(paragraph_tags).filter(
                    paragraph_tags.c.tag_id == tag.id
                ).count()
                
                tag_dicts.append({
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color,
                    'usage_count': count
                })
            
            self.logger.info(f"Retrieved {len(tag_dicts)} tags with usage counts")
            return tag_dicts
        
        try:
            return self._with_session(db_get_tags)
        except Exception as e:
            self.logger.error(f"Error retrieving tags: {str(e)}")
            return []
    
    def add_tag(self, name: str, color: str) -> int:
        """
        Add a new tag and return its ID.
        
        Args:
            name: Tag name (must be unique)
            color: Color code (e.g., '#FF0000' for red)
            
        Returns:
            Tag ID if successful, -1 otherwise
        """
        self.logger.info(f"Adding new tag: {name} with color {color}")
        
        def db_add_tag(session: Session) -> int:
            # Create new tag
            tag = Tag(name=name, color=color)
            session.add(tag)
            session.commit()
            
            tag_id = tag.id
            self.logger.info(f"Added tag with ID: {tag_id}")
            return tag_id
        
        try:
            return self._with_session(db_add_tag)
        except Exception as e:
            self.logger.error(f"Error adding tag: {str(e)}")
            return -1
    
    def delete_tag(self, tag_id: int) -> bool:
        """
        Delete a tag and all its associations.
        
        Args:
            tag_id: ID of the tag to delete
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Deleting tag with ID: {tag_id}")
        
        def db_delete_tag(session: Session) -> bool:
            # Get the tag
            tag = session.query(Tag).get(tag_id)
            
            if not tag:
                self.logger.warning(f"Tag with ID {tag_id} not found")
                return False
            
            # Delete the tag (cascade will handle associations)
            session.delete(tag)
            session.commit()
            
            self.logger.info(f"Deleted tag with ID: {tag_id}")
            return True
        
        try:
            return self._with_session(db_delete_tag)
        except Exception as e:
            self.logger.error(f"Error deleting tag: {str(e)}")
            return False
    
    def _find_duplicate_paragraphs(self, paragraph_id: int) -> List[int]:
        """
        Find all paragraph IDs with the same content as the given paragraph.
        
        Args:
            paragraph_id: ID of the paragraph to find duplicates for
            
        Returns:
            List of paragraph IDs with identical content
        """
        self.logger.info(f"Finding duplicate paragraphs for paragraph ID: {paragraph_id}")
        
        def db_find_duplicates(session: Session) -> List[int]:
            # Get the content of the paragraph
            paragraph = session.query(Paragraph).get(paragraph_id)
            if not paragraph:
                self.logger.warning(f"Paragraph with ID {paragraph_id} not found")
                return []
            
            # Find paragraphs with the same content but different IDs
            duplicates = session.query(Paragraph.id).filter(
                Paragraph.content == paragraph.content,
                Paragraph.id != paragraph_id
            ).all()
            
            # Extract IDs
            duplicate_ids = [id for (id,) in duplicates]
            
            self.logger.info(f"Found {len(duplicate_ids)} duplicate paragraphs for paragraph ID {paragraph_id}")
            return duplicate_ids
        
        try:
            return self._with_session(db_find_duplicates)
        except Exception as e:
            self.logger.error(f"Error finding duplicate paragraphs: {str(e)}")
            return []
    
    def tag_paragraph(self, paragraph_id: int, tag_id: int, tag_all_duplicates: bool = False) -> bool:
        """
        Associate a tag with a paragraph, and optionally tag all duplicate paragraphs.
        
        Args:
            paragraph_id: ID of the paragraph to tag
            tag_id: ID of the tag to apply
            tag_all_duplicates: Whether to tag all paragraphs with identical content
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Tagging paragraph {paragraph_id} with tag {tag_id}")
        
        def db_tag_paragraph(session: Session) -> bool:
            # Get paragraph and tag
            paragraph = session.query(Paragraph).get(paragraph_id)
            tag = session.query(Tag).get(tag_id)
            
            if not paragraph or not tag:
                self.logger.warning(f"Paragraph {paragraph_id} or tag {tag_id} not found")
                return False
            
            # List of paragraphs to tag
            paragraphs_to_tag = [paragraph]
            
            # If tagging all duplicates, find paragraphs with the same content
            if tag_all_duplicates:
                duplicate_ids = self._find_duplicate_paragraphs(paragraph_id)
                self.logger.info(f"Found {len(duplicate_ids)} duplicate paragraphs to tag")
                
                # Get the duplicate paragraphs
                if duplicate_ids:
                    duplicates = session.query(Paragraph).filter(Paragraph.id.in_(duplicate_ids)).all()
                    paragraphs_to_tag.extend(duplicates)
            
            # Tag all paragraphs
            for para in paragraphs_to_tag:
                # Check if already tagged
                if tag in para.tags:
                    self.logger.info(f"Paragraph {para.id} already has tag {tag_id}")
                    continue
                
                # Add tag to paragraph
                para.tags.append(tag)
            
            session.commit()
            
            self.logger.info(f"Tagged {len(paragraphs_to_tag)} paragraphs with tag {tag_id}")
            return True
        
        try:
            return self._with_session(db_tag_paragraph)
        except Exception as e:
            self.logger.error(f"Error tagging paragraph: {str(e)}")
            return False
    
    def untag_paragraph(self, paragraph_id: int, tag_id: int, untag_all_duplicates: bool = False) -> bool:
        """
        Remove a tag association from a paragraph, and optionally from all duplicate paragraphs.
        
        Args:
            paragraph_id: ID of the paragraph to untag
            tag_id: ID of the tag to remove
            untag_all_duplicates: Whether to untag all paragraphs with identical content
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Removing tag {tag_id} from paragraph {paragraph_id}")
        
        def db_untag_paragraph(session: Session) -> bool:
            # Get paragraph and tag
            paragraph = session.query(Paragraph).get(paragraph_id)
            tag = session.query(Tag).get(tag_id)
            
            if not paragraph or not tag:
                self.logger.warning(f"Paragraph {paragraph_id} or tag {tag_id} not found")
                return False
            
            # List of paragraphs to untag
            paragraphs_to_untag = [paragraph]
            
            # If untagging all duplicates, find paragraphs with the same content
            if untag_all_duplicates:
                duplicate_ids = self._find_duplicate_paragraphs(paragraph_id)
                self.logger.info(f"Found {len(duplicate_ids)} duplicate paragraphs to untag")
                
                # Get the duplicate paragraphs
                if duplicate_ids:
                    duplicates = session.query(Paragraph).filter(Paragraph.id.in_(duplicate_ids)).all()
                    paragraphs_to_untag.extend(duplicates)
            
            # Untag all paragraphs
            for para in paragraphs_to_untag:
                # Remove tag from paragraph if it exists
                if tag in para.tags:
                    para.tags.remove(tag)
            
            session.commit()
            
            self.logger.info(f"Removed tag {tag_id} from {len(paragraphs_to_untag)} paragraphs")
            return True
        
        try:
            return self._with_session(db_untag_paragraph)
        except Exception as e:
            self.logger.error(f"Error removing tag from paragraph: {str(e)}")
            return False
    
    def create_cluster(self, name: str, description: str, similarity_threshold: float, 
                       similarity_type: str = 'content') -> int:
        """
        Create a new cluster and return its ID.
        
        Args:
            name: Cluster name
            description: Cluster description
            similarity_threshold: Threshold used for similarity clustering
            similarity_type: Type of similarity measure used ('content' or 'text')
            
        Returns:
            Cluster ID if successful, -1 otherwise
        """
        self.logger.info(f"Creating new cluster: {name}")
        
        def db_create_cluster(session: Session) -> int:
            # Create new cluster
            cluster = Cluster(
                name=name,
                description=description,
                creation_date=datetime.now().isoformat(),
                similarity_threshold=similarity_threshold,
                similarity_type=similarity_type
            )
            session.add(cluster)
            session.commit()
            
            cluster_id = cluster.id
            self.logger.info(f"Created cluster with ID: {cluster_id}")
            return cluster_id
        
        try:
            return self._with_session(db_create_cluster)
        except Exception as e:
            self.logger.error(f"Error creating cluster: {str(e)}")
            return -1
    
    def add_paragraphs_to_cluster(self, cluster_id: int, paragraph_ids: List[int]) -> bool:
        """
        Add paragraphs to a cluster.
        
        Args:
            cluster_id: ID of the cluster
            paragraph_ids: List of paragraph IDs to add to the cluster
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Adding {len(paragraph_ids)} paragraphs to cluster {cluster_id}")
        
        def db_add_to_cluster(session: Session) -> bool:
            # Get cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return False
            
            # Get paragraphs
            paragraphs = session.query(Paragraph).filter(Paragraph.id.in_(paragraph_ids)).all()
            
            # Add paragraphs to cluster
            for paragraph in paragraphs:
                if paragraph not in cluster.paragraphs:
                    cluster.paragraphs.append(paragraph)
            
            session.commit()
            
            self.logger.info(f"Added {len(paragraphs)} paragraphs to cluster {cluster_id}")
            return True
        
        try:
            return self._with_session(db_add_to_cluster)
        except Exception as e:
            self.logger.error(f"Error adding paragraphs to cluster: {str(e)}")
            return False
    
    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters.
        
        Returns:
            List of cluster dictionaries
        """
        self.logger.info("Retrieving all clusters")
        
        def db_get_clusters(session: Session) -> List[Dict[str, Any]]:
            # Query all clusters
            clusters = session.query(Cluster).all()
            
            # Convert ORM objects to dictionaries
            cluster_dicts = []
            for cluster in clusters:
                cluster_dicts.append({
                    'id': cluster.id,
                    'name': cluster.name,
                    'description': cluster.description,
                    'creation_date': cluster.creation_date,
                    'similarity_threshold': cluster.similarity_threshold,
                    'similarity_type': cluster.similarity_type,
                    'paragraph_count': len(cluster.paragraphs)
                })
            
            self.logger.info(f"Retrieved {len(cluster_dicts)} clusters")
            return cluster_dicts
        
        try:
            return self._with_session(db_get_clusters)
        except Exception as e:
            self.logger.error(f"Error retrieving clusters: {str(e)}")
            return []
    
    def get_cluster_paragraphs(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get paragraphs in a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of paragraph dictionaries in the cluster
        """
        self.logger.info(f"Retrieving paragraphs for cluster {cluster_id}")
        
        def db_get_cluster_paragraphs(session: Session) -> List[Dict[str, Any]]:
            # Get cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return []
            
            # Get paragraphs in cluster with document info
            paragraphs = []
            for paragraph in cluster.paragraphs:
                document = session.query(Document).get(paragraph.document_id)
                
                # Get tags for this paragraph
                tags = [{
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                } for tag in paragraph.tags]
                
                paragraphs.append({
                    'id': paragraph.id,
                    'content': paragraph.content,
                    'document_id': paragraph.document_id,
                    'paragraph_type': paragraph.paragraph_type,
                    'position': paragraph.position,
                    'header_content': paragraph.header_content,
                    'filename': document.filename if document else 'Unknown',
                    'tags': tags
                })
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs for cluster {cluster_id}")
            return paragraphs
        
        try:
            return self._with_session(db_get_cluster_paragraphs)
        except Exception as e:
            self.logger.error(f"Error retrieving cluster paragraphs: {str(e)}")
            return []
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """
        Delete a cluster.
        
        Args:
            cluster_id: ID of the cluster to delete
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Deleting cluster with ID: {cluster_id}")
        
        def db_delete_cluster(session: Session) -> bool:
            # Get the cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return False
            
            # Delete the cluster (cascade will handle associations)
            session.delete(cluster)
            session.commit()
            
            self.logger.info(f"Deleted cluster with ID: {cluster_id}")
            return True
        
        try:
            return self._with_session(db_delete_cluster)
        except Exception as e:
            self.logger.error(f"Error deleting cluster: {str(e)}")
            return False

    def clear_all_clusters(self) -> bool:
        """
        Delete all clusters in the database.
        
        Returns:
            Boolean indicating success
        """
        self.logger.info("Clearing all clusters")
        
        def db_clear_clusters(session: Session) -> int:
            # Clear cluster_paragraphs associations
            session.execute(cluster_paragraphs.delete())
            
            # Delete all clusters
            count = session.query(Cluster).delete()
            session.commit()
            return count
        
        try:
            count = self._with_session(db_clear_clusters)
            self.logger.info(f"Deleted {count} clusters")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing clusters: {str(e)}")
            return False
    
    def export_to_excel(self, output_path: str) -> bool:
        """
        Export database contents to Excel with updated similarity fields.
        
        Args:
            output_path: Path where the Excel file will be saved
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Exporting data to Excel: {output_path}")
        
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created directory: {output_dir}")
            
            def db_export_to_excel(session: Session) -> tuple:
                # Get paragraphs with document info and tags
                paragraphs_data = []
                
                # Query paragraphs with document info
                results = session.query(
                    Paragraph, 
                    Document.filename, 
                    Document.upload_date
                ).join(
                    Document
                ).order_by(
                    Document.filename, 
                    Paragraph.position
                ).all()
                
                # Process paragraphs and include tags
                for para, filename, upload_date in results:
                    # Get tags as comma-separated string
                    tags_str = ', '.join([tag.name for tag in para.tags]) if para.tags else None
                    
                    paragraphs_data.append({
                        'id': para.id,
                        'content': para.content,
                        'paragraph_type': para.paragraph_type,
                        'header_content': para.header_content,
                        'filename': filename,
                        'upload_date': upload_date,
                        'tags': tags_str
                    })
                
                # Get similarity data with both similarity scores
                similarity_data = []
                
                # Query similarity results with paragraph content and document names
                similarity_results = session.query(
                    SimilarityResult,
                    Paragraph.content.label('paragraph1_content'),
                    Document.filename.label('document1')
                ).join(
                    Paragraph,
                    SimilarityResult.paragraph1_id == Paragraph.id
                ).join(
                    Document,
                    Paragraph.document_id == Document.id
                ).filter(
                    SimilarityResult.content_similarity_score >= 0.8
                ).all()
                
                # Get paragraph2 and document2 data
                for sim, para1_content, doc1_filename in similarity_results:
                    para2 = session.query(Paragraph).get(sim.paragraph2_id)
                    if para2:
                        doc2 = session.query(Document).get(para2.document_id)
                        
                        similarity_data.append({
                            'content_similarity_score': sim.content_similarity_score,  # Renamed field
                            'text_similarity_score': sim.text_similarity_score,        # New field
                            'similarity_type': sim.similarity_type,
                            'paragraph1_content': para1_content,
                            'document1': doc1_filename,
                            'paragraph2_content': para2.content,
                            'document2': doc2.filename if doc2 else 'Unknown'
                        })
                
                return paragraphs_data, similarity_data
            
            paragraphs_data, similarity_data = self._with_session(db_export_to_excel)
            
            # Convert to DataFrames
            paragraphs_df = pd.DataFrame(paragraphs_data)
            similarity_df = pd.DataFrame(similarity_data)
            
            # Write to Excel file
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Paragraphs sheet
                paragraphs_df.to_excel(writer, sheet_name='Paragraphs', index=False)
                
                # Similarities sheet - now with both similarity metrics
                if not similarity_df.empty:
                    similarity_df.to_excel(writer, sheet_name='Similarities', index=False)
                
                # Format the sheets
                workbook = writer.book
                
                # Format for the Paragraphs sheet
                sheet1 = writer.sheets['Paragraphs']
                sheet1.set_column('A:A', 10)  # ID column
                sheet1.set_column('B:B', 80)  # Content column
                sheet1.set_column('C:E', 20)  # Other columns
                
                # Format for the Similarities sheet if it exists
                if not similarity_df.empty:
                    sheet2 = writer.sheets['Similarities']
                    sheet2.set_column('A:A', 15)  # Content Similarity column
                    sheet2.set_column('B:B', 15)  # Text Similarity column
                    sheet2.set_column('C:C', 15)  # Type column
                    sheet2.set_column('D:E', 80)  # Content columns
                    sheet2.set_column('F:G', 30)  # Document columns
            
            self.logger.info(f"Data exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
            return False
    
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
            
            if document:
                file_path = document.file_path
                
                # Delete from database (SQLAlchemy will handle the cascade)
                session.delete(document)
                session.commit()
                
                return file_path
            else:
                self.logger.warning(f"Document with ID {document_id} not found")
                return False
        
        try:
            file_path = self._with_session(db_delete_document)
            
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
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            Boolean indicating success
        """
        self.logger.info("Clearing all data from database")
        
        def db_clear_database(session: Session) -> List[str]:
            # Get all file paths to delete files as well
            documents = session.query(Document).all()
            file_paths = [doc.file_path for doc in documents]
            
            # Delete all records through cascade
            # Start with similarity results
            session.query(SimilarityResult).delete()
            
            # Clear paragraph_tags table
            session.execute(paragraph_tags.delete())
            
            # Clear cluster_paragraphs table
            session.execute(cluster_paragraphs.delete())
            
            # Delete clusters
            session.query(Cluster).delete()
            
            # Delete paragraphs
            session.query(Paragraph).delete()
            
            # Delete documents
            session.query(Document).delete()
            
            # Don't delete tags
            
            session.commit()
            return file_paths
        
        try:
            file_paths = self._with_session(db_clear_database)
            
            # Delete all files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Deleted file: {file_path}")
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Could not delete file {file_path}: {str(e)}")
            
            self.logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def add_document_file_metadata(self, document_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Add file metadata for a document using SQLAlchemy ORM.
        
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
            return self._with_session(db_add_metadata)
        except Exception as e:
            self.logger.error(f"Error adding file metadata: {str(e)}")
            return False

    def get_document_file_metadata(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get file metadata for a document using SQLAlchemy ORM.
        
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
            return self._with_session(db_get_metadata)
        except Exception as e:
            self.logger.error(f"Error retrieving file metadata: {str(e)}")
            return None

    def delete_document_file_metadata(self, document_id: int) -> bool:
        """
        Delete file metadata for a document using SQLAlchemy ORM.
        
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
            return self._with_session(db_delete_metadata)
        except Exception as e:
            self.logger.error(f"Error deleting file metadata: {str(e)}")
            return False
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """
        Set up a logger instance.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f'{__name__}.DatabaseManager')
        
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