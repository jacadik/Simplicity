import os
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table, and_, func, Index, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from document_parser import Paragraph as ParserParagraph
from similarity_analyzer import SimilarityResult as AnalyzerSimilarityResult

# Initialize SQLAlchemy Base
Base = declarative_base()

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

class Paragraph(Base):
    """Paragraph model representing extracted text from documents."""
    __tablename__ = 'paragraphs'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    paragraph_type = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    header_content = Column(Text, nullable=True)
    column = Column(Integer, nullable=True)  # New column to track column position
    
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

# Database migration script to add the column field
# You can add this to a new file or to an existing migration script

def migrate_paragraphs_table():
    """Add column field to paragraphs table if it doesn't exist."""
    import sqlite3
    import os
    from sqlalchemy import create_engine, inspect, text
    
    # Use the DB_URL from your app.py
    # For SQLite:
    # DB_URL = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paragraph_analyzer.db')}"
    # For PostgreSQL:
    DB_URL = "postgresql://paragraph_user:pass@localhost/paragraph_analyzer"
    
    try:
        # Create engine
        engine = create_engine(DB_URL)
        inspector = inspect(engine)
        
        # Check if column exists
        columns = [col['name'] for col in inspector.get_columns('paragraphs')]
        
        if 'column' not in columns:
            # Add column
            with engine.connect() as conn:
                # For SQLite
                if 'sqlite' in DB_URL:
                    conn.execute(text("ALTER TABLE paragraphs ADD COLUMN column INTEGER"))
                # For PostgreSQL
                else:
                    conn.execute(text("ALTER TABLE paragraphs ADD COLUMN column INTEGER"))
                    
                conn.commit()
                
            print("Added 'column' field to paragraphs table")
        else:
            print("'column' field already exists in paragraphs table")
            
    except Exception as e:
        print(f"Error migrating paragraphs table: {str(e)}")
        return False
        
    return True

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
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        self.logger.info(f"Initializing database with SQLAlchemy using {self.db_url}")
        
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Create default tags if they don't exist
            session = self.Session()
            try:
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
                
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise
    
    def add_document(self, filename: str, file_type: str, file_path: str) -> int:
        """Add a document to the database and return its ID."""
        self.logger.info(f"Adding document to database: {filename}")
        
        session = self.Session()
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
            session.commit()
            
            doc_id = document.id
            self.logger.info(f"Document added with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding document: {str(e)}", exc_info=True)
            return -1
        finally:
            session.close()
    
    def add_paragraphs(self, paragraphs: List[ParserParagraph]) -> List[int]:
        """Add paragraphs to the database and return their IDs."""
        if not paragraphs:
            return []
            
        self.logger.info(f"Adding {len(paragraphs)} paragraphs to database")
        
        session = self.Session()
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
            
            # Commit all paragraphs in one transaction
            session.commit()
            
            self.logger.info(f"Added {len(paragraph_ids)} paragraphs")
            return paragraph_ids
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding paragraphs: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def clear_similarity_results(self) -> bool:
        """Clear all similarity results from the database."""
        self.logger.info("Clearing all similarity results")
        
        session = self.Session()
        try:
            # Delete all similarity results
            count = session.query(SimilarityResult).delete()
            session.commit()
            
            self.logger.info(f"Deleted {count} similarity results")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error clearing similarity results: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def add_similarity_results(self, results: List[AnalyzerSimilarityResult]) -> bool:
        """Add similarity results to the database."""
        if not results:
            self.logger.warning("No similarity results to add")
            return True
            
        self.logger.info(f"Adding {len(results)} similarity results to database")
        
        # First, clear existing similarity results
        if not self.clear_similarity_results():
            self.logger.warning("Failed to clear existing similarity results, proceeding with update")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding similarity results: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def get_documents(self) -> List[Dict]:
        """Get all documents."""
        self.logger.info("Retrieving all documents")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def get_paragraphs(self, document_id: Optional[int] = None, collapse_duplicates: bool = True) -> List[Dict]:
        """
        Get paragraphs, optionally filtered by document.
        If collapse_duplicates is True, exact matching paragraphs are displayed only once with document references.
        """
        session = self.Session()
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error retrieving paragraphs: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def get_similar_paragraphs(self, threshold: Optional[float] = None) -> List[Dict]:
        """Get similar paragraphs above the threshold."""
        try:
            session = self.Session()
            
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
            
        except Exception as e:
            self.logger.error(f"Error retrieving similarity results: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def get_tags(self) -> List[Dict]:
        """Get all tags."""
        self.logger.info("Retrieving all tags")
        
        session = self.Session()
        try:
            # Query all tags
            tags = session.query(Tag).all()
            
            # Convert ORM objects to dictionaries
            tag_dicts = [
                {
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                }
                for tag in tags
            ]
            
            self.logger.info(f"Retrieved {len(tag_dicts)} tags")
            return tag_dicts
            
        except Exception as e:
            self.logger.error(f"Error retrieving tags: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def add_tag(self, name: str, color: str) -> int:
        """Add a new tag and return its ID."""
        self.logger.info(f"Adding new tag: {name} with color {color}")
        
        session = self.Session()
        try:
            # Create new tag
            tag = Tag(name=name, color=color)
            session.add(tag)
            session.commit()
            
            tag_id = tag.id
            self.logger.info(f"Added tag with ID: {tag_id}")
            return tag_id
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding tag: {str(e)}", exc_info=True)
            return -1
        finally:
            session.close()
    
    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag and all its associations."""
        self.logger.info(f"Deleting tag with ID: {tag_id}")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting tag: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def tag_paragraph(self, paragraph_id: int, tag_id: int) -> bool:
        """Associate a tag with a paragraph."""
        self.logger.info(f"Tagging paragraph {paragraph_id} with tag {tag_id}")
        
        session = self.Session()
        try:
            # Get paragraph and tag
            paragraph = session.query(Paragraph).get(paragraph_id)
            tag = session.query(Tag).get(tag_id)
            
            if not paragraph or not tag:
                self.logger.warning(f"Paragraph {paragraph_id} or tag {tag_id} not found")
                return False
            
            # Check if already tagged
            if tag in paragraph.tags:
                self.logger.info(f"Paragraph {paragraph_id} already has tag {tag_id}")
                return True
            
            # Add tag to paragraph
            paragraph.tags.append(tag)
            session.commit()
            
            self.logger.info(f"Tagged paragraph {paragraph_id} with tag {tag_id}")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error tagging paragraph: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def untag_paragraph(self, paragraph_id: int, tag_id: int) -> bool:
        """Remove a tag association from a paragraph."""
        self.logger.info(f"Removing tag {tag_id} from paragraph {paragraph_id}")
        
        session = self.Session()
        try:
            # Get paragraph and tag
            paragraph = session.query(Paragraph).get(paragraph_id)
            tag = session.query(Tag).get(tag_id)
            
            if not paragraph or not tag:
                self.logger.warning(f"Paragraph {paragraph_id} or tag {tag_id} not found")
                return False
            
            # Remove tag from paragraph
            if tag in paragraph.tags:
                paragraph.tags.remove(tag)
                session.commit()
            
            self.logger.info(f"Removed tag {tag_id} from paragraph {paragraph_id}")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error removing tag from paragraph: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def create_cluster(self, name: str, description: str, similarity_threshold: float, 
                      similarity_type: str = 'content') -> int:
        """Create a new cluster and return its ID."""
        self.logger.info(f"Creating new cluster: {name}")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating cluster: {str(e)}", exc_info=True)
            return -1
        finally:
            session.close()
    
    def add_paragraphs_to_cluster(self, cluster_id: int, paragraph_ids: List[int]) -> bool:
        """Add paragraphs to a cluster."""
        self.logger.info(f"Adding {len(paragraph_ids)} paragraphs to cluster {cluster_id}")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding paragraphs to cluster: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def get_clusters(self) -> List[Dict]:
        """Get all clusters."""
        self.logger.info("Retrieving all clusters")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error retrieving clusters: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def get_cluster_paragraphs(self, cluster_id: int) -> List[Dict]:
        """Get paragraphs in a specific cluster."""
        self.logger.info(f"Retrieving paragraphs for cluster {cluster_id}")
        
        session = self.Session()
        try:
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
                    'filename': document.filename,
                    'tags': tags
                })
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs for cluster {cluster_id}")
            return paragraphs
            
        except Exception as e:
            self.logger.error(f"Error retrieving cluster paragraphs: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster."""
        self.logger.info(f"Deleting cluster with ID: {cluster_id}")
        
        session = self.Session()
        try:
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
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting cluster: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()

    def clear_all_clusters(self) -> bool:
        """Delete all clusters in the database."""
        self.logger.info("Clearing all clusters")
        
        session = self.Session()
        try:
            # Clear cluster_paragraphs associations
            session.execute(cluster_paragraphs.delete())
            
            # Delete all clusters
            count = session.query(Cluster).delete()
            session.commit()
            
            self.logger.info(f"Deleted {count} clusters")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error clearing clusters: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def export_to_excel(self, output_path: str) -> bool:
        """Export database contents to Excel with updated similarity fields."""
        self.logger.info(f"Exporting data to Excel: {output_path}")
        
        try:
            # Get paragraphs with document info and tags
            paragraphs_data = []
            
            session = self.Session()
            
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
            
            # Convert to DataFrame
            paragraphs_df = pd.DataFrame(paragraphs_data)
            
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
                doc2 = session.query(Document).get(para2.document_id)
                
                similarity_data.append({
                    'content_similarity_score': sim.content_similarity_score,  # Renamed field
                    'text_similarity_score': sim.text_similarity_score,        # New field
                    'similarity_type': sim.similarity_type,
                    'paragraph1_content': para1_content,
                    'document1': doc1_filename,
                    'paragraph2_content': para2.content,
                    'document2': doc2.filename
                })
            
            # Convert to DataFrame
            similarity_df = pd.DataFrame(similarity_data)
            
            session.close()
            
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
        finally:
            if 'session' in locals():
                session.close()
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its paragraphs."""
        self.logger.info(f"Deleting document with ID: {document_id}")
        
        session = self.Session()
        try:
            # Get the document
            document = session.query(Document).get(document_id)
            
            if document:
                file_path = document.file_path
                
                # Delete from database (SQLAlchemy will handle the cascade)
                session.delete(document)
                session.commit()
                
                # Try to delete the file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
                
                self.logger.info(f"Deleted document with ID: {document_id}")
                return True
            else:
                self.logger.warning(f"Document with ID {document_id} not found")
                return False
                
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def clear_database(self) -> bool:
        """Clear all data from the database."""
        self.logger.info("Clearing all data from database")
        
        session = self.Session()
        try:
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
            
            # Delete all files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {file_path}")
            
            self.logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error clearing database: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
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
        
        return logger