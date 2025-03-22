"""
Database module for the paragraph analyzer system.
Provides database management and operations for documents, paragraphs, 
similarity analysis, clustering, and exports.
"""

# Import the main DatabaseManager class for backward compatibility
from .manager import DatabaseManager

# Import models for direct access
from .models import (
    Document, 
    Paragraph, 
    Tag, 
    SimilarityResult, 
    Cluster, 
    DocumentFileMetadata,
    Insert,
    InsertPage,
    paragraph_tags,
    cluster_paragraphs
)

# Import specialized managers for advanced use cases
from .document_manager import DocumentManager
from .paragraph_manager import ParagraphManager
from .tag_manager import TagManager
from .similarity_manager import SimilarityManager
from .cluster_manager import ClusterManager
from .export_manager import ExportManager
from .insert_manager import InsertManager
from .base_manager import BaseManager

# Version
__version__ = '1.0.0'
