from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from .models import SimilarityResult, Paragraph, Document
from .base_manager import BaseManager
from utils.similarity_analyzer import SimilarityResult as AnalyzerSimilarityResult

class SimilarityManager:
    """
    Manages similarity-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the similarity manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def clear_similarity_results(self) -> bool:
        """Clear all similarity results from the database."""
        self.logger.info("Clearing all similarity results")
        
        def db_clear_similarity_results(session: Session) -> int:
            # Delete all similarity results
            count = session.query(SimilarityResult).delete()
            session.commit()
            return count
        
        try:
            count = self.base_manager._with_session(db_clear_similarity_results)
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
            return self.base_manager._with_session(db_add_similarity_results)
        except Exception as e:
            self.logger.error(f"Error adding similarity results: {str(e)}")
            return False
    
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
            
            return self.base_manager._with_session(db_get_similar_paragraphs)
            
        except Exception as e:
            self.logger.error(f"Error retrieving similarity results: {str(e)}")
            return []
