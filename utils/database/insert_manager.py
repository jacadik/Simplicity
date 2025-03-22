from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from .models import Insert, InsertPage
from .base_manager import BaseManager

class InsertManager:
    """
    Manages insert-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the insert manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def add_insert(self, name: str, filename: str, file_type: str, file_path: str) -> int:
        """
        Add an insert to the database and return its ID.
        
        Args:
            name: Custom name for the insert
            filename: The name of the file
            file_type: The file extension
            file_path: The path where the file is stored
            
        Returns:
            The insert ID if successful, -1 otherwise
        """
        self.logger.info(f"Adding insert to database: {name} ({filename})")
        
        def db_add_insert(session: Session) -> int:
            # Create new insert record
            upload_date = datetime.now().isoformat()
            
            insert = Insert(
                name=name,
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                upload_date=upload_date
            )
            
            session.add(insert)
            session.commit()
            
            insert_id = insert.id
            self.logger.info(f"Insert added with ID: {insert_id}")
            return insert_id
        
        try:
            return self.base_manager._with_session(db_add_insert)
        except Exception as e:
            self.logger.error(f"Error adding insert: {str(e)}")
            return -1

    def add_insert_pages(self, pages: List[Dict]) -> List[int]:
        """
        Add insert pages to the database.
        
        Args:
            pages: List of page dictionaries with content, page_number, and insert_id
            
        Returns:
            List of page IDs if successful, empty list otherwise
        """
        if not pages:
            return []
            
        self.logger.info(f"Adding {len(pages)} insert pages to database")
        
        def db_add_insert_pages(session: Session) -> List[int]:
            page_ids = []
            
            for page in pages:
                # Create new insert page record
                db_page = InsertPage(
                    content=page['content'],
                    insert_id=page['insert_id'],
                    page_number=page['page_number']
                )
                
                session.add(db_page)
                # Flush to get the ID but don't commit yet
                session.flush()
                page_ids.append(db_page.id)
            
            # Commit all pages in one transaction
            session.commit()
            
            self.logger.info(f"Added {len(page_ids)} insert pages")
            return page_ids
        
        try:
            return self.base_manager._with_session(db_add_insert_pages)
        except Exception as e:
            self.logger.error(f"Error adding insert pages: {str(e)}")
            return []

    def get_inserts(self) -> List[Dict[str, Any]]:
        """
        Get all inserts with page counts.
        
        Returns:
            List of insert dictionaries
        """
        self.logger.info("Retrieving all inserts")
        
        def db_get_inserts(session: Session) -> List[Dict[str, Any]]:
            # Query inserts with page counts
            results = session.query(
                Insert, 
                func.count(InsertPage.id).label('page_count')
            ).outerjoin(
                InsertPage,
                Insert.id == InsertPage.insert_id
            ).group_by(
                Insert.id
            ).order_by(
                Insert.upload_date.desc()
            ).all()
            
            inserts = []
            for insert, page_count in results:
                inserts.append({
                    'id': insert.id,
                    'name': insert.name,
                    'filename': insert.filename,
                    'file_path': insert.file_path,
                    'upload_date': insert.upload_date,
                    'page_count': page_count
                })
            
            self.logger.info(f"Retrieved {len(inserts)} inserts")
            return inserts
        
        try:
            return self.base_manager._with_session(db_get_inserts)
        except Exception as e:
            self.logger.error(f"Error retrieving inserts: {str(e)}")
            return []

    def get_insert_pages(self, insert_id: int) -> List[Dict[str, Any]]:
        """
        Get pages for a specific insert.
        
        Args:
            insert_id: ID of the insert
            
        Returns:
            List of page dictionaries
        """
        self.logger.info(f"Retrieving pages for insert ID: {insert_id}")
        
        def db_get_insert_pages(session: Session) -> List[Dict[str, Any]]:
            # Get insert to ensure it exists
            insert = session.query(Insert).get(insert_id)
            
            if not insert:
                self.logger.warning(f"Insert with ID {insert_id} not found")
                return []
            
            # Get pages
            pages = session.query(InsertPage).filter(
                InsertPage.insert_id == insert_id
            ).order_by(
                InsertPage.page_number
            ).all()
            
            # Convert to dictionaries
            page_dicts = []
            for page in pages:
                page_dicts.append({
                    'id': page.id,
                    'content': page.content,
                    'page_number': page.page_number,
                    'insert_id': page.insert_id
                })
            
            self.logger.info(f"Retrieved {len(page_dicts)} pages for insert ID: {insert_id}")
            return page_dicts
        
        try:
            return self.base_manager._with_session(db_get_insert_pages)
        except Exception as e:
            self.logger.error(f"Error retrieving insert pages: {str(e)}")
            return []
