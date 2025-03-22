from typing import List, Dict, Any

from sqlalchemy.orm import Session

from .models import Tag, Paragraph, paragraph_tags
from .base_manager import BaseManager

class TagManager:
    """
    Manages tag-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager, paragraph_manager=None):
        """
        Initialize the tag manager.
        
        Args:
            base_manager: Base manager instance for database operations
            paragraph_manager: Paragraph manager for duplicate detection
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
        self.paragraph_manager = paragraph_manager
    
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
            return self.base_manager._with_session(db_get_tags)
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
            return self.base_manager._with_session(db_add_tag)
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
            return self.base_manager._with_session(db_delete_tag)
        except Exception as e:
            self.logger.error(f"Error deleting tag: {str(e)}")
            return False
    
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
            if tag_all_duplicates and self.paragraph_manager:
                duplicate_ids = self.paragraph_manager._find_duplicate_paragraphs(paragraph_id)
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
            return self.base_manager._with_session(db_tag_paragraph)
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
            if untag_all_duplicates and self.paragraph_manager:
                duplicate_ids = self.paragraph_manager._find_duplicate_paragraphs(paragraph_id)
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
            return self.base_manager._with_session(db_untag_paragraph)
        except Exception as e:
            self.logger.error(f"Error removing tag from paragraph: {str(e)}")
            return False
