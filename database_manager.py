import os
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table, and_, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session, aliased
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
    
    # Relationships
    document = relationship("Document", back_populates="paragraphs")
    tags = relationship("Tag", secondary=paragraph_tags, back_populates="paragraphs")
    clusters = relationship("ParagraphCluster", secondary="paragraph_cluster_items", back_populates="paragraphs")
    
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
    similarity_score = Column(Float, nullable=False)
    similarity_type = Column(String, nullable=False)
    
    # Relationships
    paragraph1 = relationship("Paragraph", foreign_keys=[paragraph1_id], back_populates="similarity_as_para1")
    paragraph2 = relationship("Paragraph", foreign_keys=[paragraph2_id], back_populates="similarity_as_para2")

class ParagraphCluster(Base):
    """Model for storing paragraph clusters."""
    __tablename__ = 'paragraph_clusters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    creation_date = Column(String, nullable=False)
    similarity_threshold = Column(Float, nullable=False)
    
    # Relationships
    paragraphs = relationship("Paragraph", secondary="paragraph_cluster_items", back_populates="clusters")

class ParagraphClusterItem(Base):
    """Association table for paragraphs and clusters."""
    __tablename__ = 'paragraph_cluster_items'
    
    cluster_id = Column(Integer, ForeignKey('paragraph_clusters.id', ondelete='CASCADE'), primary_key=True)
    paragraph_id = Column(Integer, ForeignKey('paragraphs.id', ondelete='CASCADE'), primary_key=True)


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
    
    def add_similarity_results(self, results: List[AnalyzerSimilarityResult]) -> bool:
        """Add similarity results to the database."""
        if not results:
            return True
            
        self.logger.info(f"Adding {len(results)} similarity results to database")
        
        session = self.Session()
        try:
            for result in results:
                # Create new similarity result record
                db_result = SimilarityResult(
                    paragraph1_id=result.paragraph1_id,
                    paragraph2_id=result.paragraph2_id,
                    similarity_score=result.similarity_score,
                    similarity_type=result.similarity_type
                )
                
                session.add(db_result)
            
            # Commit all similarity results in one transaction
            session.commit()
            
            self.logger.info(f"Added {len(results)} similarity results")
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
            
            # Create aliases for second paragraph and document
            Paragraph2 = aliased(Paragraph)
            Document2 = aliased(Document)
            
            # Build query with proper aliases
            query = session.query(
                SimilarityResult,
                Paragraph.content.label('para1_content'),
                Paragraph.document_id.label('para1_doc_id'),
                Document.filename.label('para1_filename'),
                Paragraph2.content.label('para2_content'),
                Paragraph2.document_id.label('para2_doc_id'),
                Document2.filename.label('para2_filename')
            ).join(
                Paragraph,
                SimilarityResult.paragraph1_id == Paragraph.id
            ).join(
                Document,
                Paragraph.document_id == Document.id
            ).join(
                Paragraph2,
                SimilarityResult.paragraph2_id == Paragraph2.id
            ).join(
                Document2,
                Paragraph2.document_id == Document2.id
            )
            
            # Apply threshold filter if provided
            if threshold is not None:
                self.logger.info(f"Retrieving similarity results with threshold: {threshold}")
                query = query.filter(SimilarityResult.similarity_score >= threshold)
            else:
                self.logger.info("Retrieving all similarity results")
            
            # Execute query
            results = query.all()
            
            # Process results into dictionaries
            similarities = []
            for row in results:
                (sim_result, 
                 para1_content, para1_doc_id, para1_filename,
                 para2_content, para2_doc_id, para2_filename) = row
                
                similarity_dict = {
                    'id': sim_result.id,
                    'paragraph1_id': sim_result.paragraph1_id,
                    'paragraph2_id': sim_result.paragraph2_id,
                    'similarity_score': sim_result.similarity_score,
                    'similarity_type': sim_result.similarity_type,
                    'para1_content': para1_content,
                    'para1_doc_id': para1_doc_id,
                    'para1_filename': para1_filename,
                    'para2_content': para2_content,
                    'para2_doc_id': para2_doc_id,
                    'para2_filename': para2_filename
                }
                
                similarities.append(similarity_dict)
            
            self.logger.info(f"Retrieved {len(similarities)} similarity results")
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
    
    def export_to_excel(self, output_path: str) -> bool:
        """Export database contents to Excel."""
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
            
            # Get similarity data
            similarity_data = []
            
            # Query similarity results with paragraph content and document names
            # Create aliases for the second paragraph and document
            Paragraph2 = aliased(Paragraph)
            Document2 = aliased(Document)
            
            # Build query with proper aliases
            similarity_results = session.query(
                SimilarityResult,
                Paragraph.content.label('paragraph1_content'),
                Document.filename.label('document1'),
                Paragraph2.content.label('paragraph2_content'),
                Document2.filename.label('document2')
            ).join(
                Paragraph,
                SimilarityResult.paragraph1_id == Paragraph.id
            ).join(
                Document,
                Paragraph.document_id == Document.id
            ).join(
                Paragraph2,
                SimilarityResult.paragraph2_id == Paragraph2.id
            ).join(
                Document2,
                Paragraph2.document_id == Document2.id
            ).filter(
                SimilarityResult.similarity_score >= 0.8
            ).all()
            
            # Process similarity results
            for sim, para1_content, doc1_filename, para2_content, doc2_filename in similarity_results:
                similarity_data.append({
                    'similarity_score': sim.similarity_score,
                    'similarity_type': sim.similarity_type,
                    'paragraph1_content': para1_content,
                    'document1': doc1_filename,
                    'paragraph2_content': para2_content,
                    'document2': doc2_filename
                })
            
            # Convert to DataFrame
            similarity_df = pd.DataFrame(similarity_data)
            
            session.close()
            
            # Write to Excel file
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Paragraphs sheet
                paragraphs_df.to_excel(writer, sheet_name='Paragraphs', index=False)
                
                # Similarities sheet
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
                    sheet2.set_column('A:A', 15)  # Score column
                    sheet2.set_column('B:B', 15)  # Type column
                    sheet2.set_column('C:D', 80)  # Content columns
                    sheet2.set_column('E:F', 30)  # Document columns
            
            self.logger.info(f"Data exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
            return False
    
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
            
    def create_cluster(self, name: str, description: str, similarity_threshold: float) -> int:
        """Create a new paragraph cluster and return its ID."""
        self.logger.info(f"Creating cluster: {name} with threshold {similarity_threshold}")
        
        session = self.Session()
        try:
            creation_date = datetime.now().isoformat()
            
            cluster = ParagraphCluster(
                name=name,
                description=description,
                creation_date=creation_date,
                similarity_threshold=similarity_threshold
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
            # Get the cluster
            cluster = session.query(ParagraphCluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return False
            
            # Get the paragraphs
            paragraphs = session.query(Paragraph).filter(Paragraph.id.in_(paragraph_ids)).all()
            
            if not paragraphs:
                self.logger.warning(f"No paragraphs found with the given IDs")
                return False
            
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
        """Get all paragraph clusters."""
        self.logger.info("Retrieving all paragraph clusters")
        
        session = self.Session()
        try:
            clusters = session.query(ParagraphCluster).all()
            
            cluster_dicts = []
            for cluster in clusters:
                # Get count of paragraphs in this cluster
                paragraph_count = len(cluster.paragraphs)
                
                cluster_dicts.append({
                    'id': cluster.id,
                    'name': cluster.name,
                    'description': cluster.description,
                    'creation_date': cluster.creation_date,
                    'similarity_threshold': cluster.similarity_threshold,
                    'paragraph_count': paragraph_count
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
            # Get the cluster
            cluster = session.query(ParagraphCluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return []
            
            # Get paragraphs with document information
            paragraphs = []
            for para in cluster.paragraphs:
                document = session.query(Document).get(para.document_id)
                
                # Get tags for this paragraph
                tags = [{
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                } for tag in para.tags]
                
                paragraphs.append({
                    'id': para.id,
                    'content': para.content,
                    'document_id': para.document_id,
                    'paragraph_type': para.paragraph_type,
                    'position': para.position,
                    'header_content': para.header_content,
                    'filename': document.filename if document else 'Unknown',
                    'tags': tags
                })
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs from cluster {cluster_id}")
            return paragraphs
            
        except Exception as e:
            self.logger.error(f"Error retrieving cluster paragraphs: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()

    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a paragraph cluster."""
        self.logger.info(f"Deleting cluster with ID: {cluster_id}")
        
        session = self.Session()
        try:
            # Get the cluster
            cluster = session.query(ParagraphCluster).get(cluster_id)
            
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