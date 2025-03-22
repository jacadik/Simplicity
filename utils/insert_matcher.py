"""
Insert Matcher module for finding documents containing insert pages.
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass

from utils.similarity_analyzer import SimilarityAnalyzer

@dataclass
class InsertMatchResult:
    """Class for storing insert match results."""
    document_id: int
    document_name: str
    page_matches: List[Dict[str, Any]]
    match_score: float  # Overall match score (0-1)
    match_count: int    # Number of pages matched

class InsertMatcher:
    """
    Finds documents in the collection that contain the given insert,
    with tolerance for variations.
    """
    
    def __init__(self, similarity_analyzer: SimilarityAnalyzer, 
                similarity_threshold: float = 0.8, 
                logging_level: str = 'INFO'):
        """
        Initialize the insert matcher.
        
        Args:
            similarity_analyzer: SimilarityAnalyzer instance
            similarity_threshold: Threshold for considering pages similar (0-1)
            logging_level: Logging level
        """
        self.similarity_analyzer = similarity_analyzer
        self.similarity_threshold = similarity_threshold
        self.logger = self._setup_logger(logging_level)
    
    def find_insert_matches(self, insert_id: int, insert_pages: List[Dict], 
                           documents: List[Dict], document_pages: Dict[int, List[Dict]]) -> List[InsertMatchResult]:
        """
        Find documents containing the given insert.
        
        Args:
            insert_id: ID of the insert to find
            insert_pages: List of insert page dictionaries
            documents: List of document dictionaries
            document_pages: Dictionary mapping document IDs to lists of page dictionaries
            
        Returns:
            List of match results
        """
        self.logger.info(f"Finding matches for insert ID {insert_id} with {len(insert_pages)} pages")
        
        results = []
        
        # Process each document
        for document in documents:
            doc_id = document['id']
            doc_pages = document_pages.get(doc_id, [])
            
            if not doc_pages:
                continue
            
            # Find matches for this document
            match_result = self._match_insert_to_document(insert_id, insert_pages, document, doc_pages)
            
            if match_result:
                results.append(match_result)
        
        # Sort results by match score (descending)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        self.logger.info(f"Found {len(results)} documents containing the insert")
        return results
    
    def _match_insert_to_document(self, insert_id: int, insert_pages: List[Dict], 
                                document: Dict, document_pages: List[Dict]) -> InsertMatchResult:
        """
        Check if a document contains the insert.
        
        Args:
            insert_id: ID of the insert
            insert_pages: List of insert page dictionaries
            document: Document dictionary
            document_pages: List of document page dictionaries
            
        Returns:
            Match result or None if no match
        """
        doc_id = document['id']
        doc_name = document['filename']
        
        self.logger.debug(f"Checking document {doc_id} ({doc_name}) for insert {insert_id}")
        
        # Store matches for each insert page
        page_matches = []
        total_match_score = 0.0
        
        # For each insert page, find the best matching document page
        for insert_page in insert_pages:
            best_match = None
            best_score = 0.0
            
            for doc_page in document_pages:
                # Compare page content using text similarity
                similarity = self._compare_page_content(insert_page['content'], doc_page['content'])
                
                if similarity >= self.similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'insert_page_num': insert_page['page_number'],
                        'doc_page_num': doc_page['page_number'],
                        'similarity': similarity
                    }
            
            if best_match:
                page_matches.append(best_match)
                total_match_score += best_match['similarity']
        
        # Calculate overall match score and determine if this is a match
        if page_matches:
            avg_match_score = total_match_score / len(page_matches)
            match_percentage = len(page_matches) / len(insert_pages)
            
            # Only consider it a match if at least half of the insert pages match
            if match_percentage >= 0.5:
                return InsertMatchResult(
                    document_id=doc_id,
                    document_name=doc_name,
                    page_matches=page_matches,
                    match_score=avg_match_score * match_percentage,  # Weight by coverage
                    match_count=len(page_matches)
                )
        
        return None
    
    def _compare_page_content(self, insert_content: str, doc_content: str) -> float:
        """
        Compare the content of two pages.
        
        Args:
            insert_content: Content of the insert page
            doc_content: Content of the document page
            
        Returns:
            Similarity score (0-1)
        """
        # Use the SimilarityAnalyzer's text similarity function
        return self.similarity_analyzer._calculate_text_similarity(insert_content, doc_content)
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.InsertMatcher')
        
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