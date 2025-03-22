import json
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Table, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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
    content_similarity_score = Column(Float, nullable=False)  
    text_similarity_score = Column(Float, nullable=True)      
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

# New models for inserts
class Insert(Base):
    """Model for template inserts that appear in other documents"""
    __tablename__ = 'inserts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # Custom user-provided name
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # Keep this column in the model
    file_path = Column(String, nullable=False)
    upload_date = Column(String, nullable=False)
    
    # Relationships
    pages = relationship("InsertPage", back_populates="insert", cascade="all, delete-orphan")
    
class InsertPage(Base):
    """Model for individual pages within an insert"""
    __tablename__ = 'insert_pages'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)  # Full page content
    insert_id = Column(Integer, ForeignKey('inserts.id', ondelete='CASCADE'), nullable=False)
    page_number = Column(Integer, nullable=False)
    
    # Relationships
    insert = relationship("Insert", back_populates="pages")
