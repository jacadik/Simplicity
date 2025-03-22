"""
Insert Matcher module for finding where inserts appear in documents.
Enhanced to support both exact and similar matches with configurable thresholds.
"""

import logging
from typing import List, Dict, Any, Tuple

class InsertMatchResult:
    """Class for storing insert match results."""
    def __init__(self, document_id: int, document_name: str, match_count: int, 
                 match_score: float, page_matches: List[Dict[str, Any]]):
        self.document_id = document_id
        self.document_name = document_name
        self.match_count = match_count
        self.match_score = match_score
        self.page_matches = page_matches

class InsertMatcher:
    """
    Matches inserts against documents to find where they're used.
    Enhanced to find both exact and similar matches based on configurable threshold.
    """
    
    def __init__(self, similarity_analyzer, similarity_threshold=0.7):
        """
        Initialize the insert matcher.
        
        Args:
            similarity_analyzer: Analyzer for calculating text and content similarity
            similarity_threshold: Minimum similarity score to consider a match (0.0 to 1.0)
        """
        self.similarity_analyzer = similarity_analyzer
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
    def find_insert_matches(self, insert_id: int, insert_pages: List[Dict], 
                           documents: List[Dict], document_pages: Dict) -> List[Dict]:
        """
        Find documents that contain the given insert pages.
        
        Args:
            insert_id: ID of the insert to find
            insert_pages: List of insert page data
            documents: List of all documents
            document_pages: Dictionary mapping document IDs to their pages
            
        Returns:
            List of match results with document info and page matches
        """
        self.logger.info(f"Finding matches for insert ID {insert_id} with {len(insert_pages)} pages")
        matches = []
        
        # Ensure we have pages to compare
        if not insert_pages:
            self.logger.warning("No insert pages to compare")
            return matches
        
        # Compare against each document
        for doc in documents:
            doc_id = doc['id']
            doc_pages = document_pages.get(doc_id, [])
            
            if not doc_pages:
                continue
                
            self.logger.debug(f"Comparing insert against document {doc_id}: {doc['filename']}")
            
            # Find matching pages between this insert and document
            page_matches = []
            match_score_sum = 0
            
            for insert_page in insert_pages:
                best_match = None
                best_score = 0
                insert_content = insert_page.get('content', '')
                
                # Skip empty insert pages
                if not insert_content.strip():
                    continue
                
                # Find best matching document page for this insert page
                for doc_page in doc_pages:
                    doc_content = doc_page.get('content', '')
                    
                    # Skip empty document pages
                    if not doc_content.strip():
                        continue
                    
                    # Calculate similarity score using the existing method in your SimilarityAnalyzer
                    # This is the key change - using your actual API
                    similarity = self._compute_similarity(insert_content, doc_content)
                    
                    self.logger.debug(f"Page comparison - Insert page {insert_page.get('page_number', 0)} "
                                    f"vs Doc page {doc_page.get('page_number', 0)}: "
                                    f"Similarity: {similarity:.2f}")
                    
                    # Keep the best match above threshold
                    if similarity >= self.similarity_threshold and similarity > best_score:
                        best_score = similarity
                        best_match = {
                            'insert_page_num': insert_page.get('page_number', 0),
                            'doc_page_num': doc_page.get('page_number', 0),
                            'similarity': similarity
                        }
                
                # Add the best match if found
                if best_match:
                    page_matches.append(best_match)
                    match_score_sum += best_score
            
            # If we have matches for at least one page, consider it a document match
            if page_matches:
                # Calculate overall match quality as average of best page match scores
                match_score = match_score_sum / len(insert_pages)
                
                # Calculate match percentage relative to total insert pages
                match_percentage = len(page_matches) / len(insert_pages)
                
                # Only include documents with a significant portion of the insert
                # (at least one page must match and the overall score must be good)
                if match_percentage > 0 and match_score >= self.similarity_threshold:
                    self.logger.info(f"Found match in document {doc_id}: {doc['filename']} - "
                                   f"Score: {match_score:.2f}, Pages: {len(page_matches)}/{len(insert_pages)}")
                    
                    matches.append({
                        'document_id': doc_id,
                        'document_name': doc['filename'],
                        'match_count': len(page_matches),
                        'match_score': match_score,
                        'match_percentage': match_percentage,
                        'page_matches': sorted(page_matches, key=lambda m: m['insert_page_num'])
                    })
        
        # Sort matches by overall score (best matches first)
        matches.sort(key=lambda m: m['match_score'], reverse=True)
        
        self.logger.info(f"Found {len(matches)} documents containing insert ID {insert_id}")
        return matches
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts using the available methods
        in the SimilarityAnalyzer.
        
        This method adapts to whatever API is available in your SimilarityAnalyzer.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Create special data structures expected by your similarity analyzer
        para1_data = {'content': text1}
        para2_data = {'content': text2}
        
        # Try different approaches based on what methods might be available
        try:
            # If there's a function that accepts two text strings directly
            return self.similarity_analyzer.find_similarity(text1, text2)
        except (AttributeError, TypeError):
            try:
                # If your SimilarityAnalyzer needs to compare paragraph objects
                return self.similarity_analyzer.compare_paragraphs(para1_data, para2_data)
            except (AttributeError, TypeError):
                try:
                    # If your similarity analyzer works like the one in your similarity.html template
                    # where content_similarity_score is the field
                    result = self.similarity_analyzer.compare_text(para1_data, para2_data)
                    if isinstance(result, dict) and 'content_similarity_score' in result:
                        return result['content_similarity_score']
                    return result
                except (AttributeError, TypeError):
                    self.logger.error("Could not find a compatible similarity calculation method")
                    # As a fallback, implement a basic similarity measure
                    return self._basic_similarity(text1, text2)
    
    def _basic_similarity(self, text1: str, text2: str) -> float:
        """
        Provide a basic fallback similarity calculation if the main one fails.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Simple Jaccard similarity using word sets
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)