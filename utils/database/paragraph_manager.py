from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, String  

from .models import Paragraph, Document
from .base_manager import BaseManager
from utils.document_parser import Paragraph as ParserParagraph

class ParagraphManager:
    """
    Manages paragraph-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the paragraph manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
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
            return self.base_manager._with_session(db_add_paragraphs)
        except Exception as e:
            self.logger.error(f"Error adding paragraphs: {str(e)}")
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
            return self.base_manager._with_session(db_get_paragraphs)
        except Exception as e:
            self.logger.error(f"Error retrieving paragraphs: {str(e)}")
            return []
    
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
            return self.base_manager._with_session(db_find_duplicates)
        except Exception as e:
            self.logger.error(f"Error finding duplicate paragraphs: {str(e)}")
            return []
