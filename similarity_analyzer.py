import logging
import re
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityResult:
    """Class for storing similarity analysis results."""
    paragraph1_id: int
    paragraph2_id: int
    paragraph1_content: str
    paragraph2_content: str
    paragraph1_doc_id: int
    paragraph2_doc_id: int
    similarity_score: float
    similarity_type: str  # 'exact' or 'similar'


class SimilarityAnalyzer:
    """
    Analyzes paragraphs to find exact matches and similar paragraphs
    using multiple similarity metrics.
    """
    def __init__(self, threshold: float = 0.8, logging_level: str = 'INFO'):
        """Initialize the similarity analyzer."""
        self.threshold = threshold
        self.logger = self._setup_logger(logging_level)
    
    def find_exact_matches(self, paragraphs: List[Dict]) -> List[SimilarityResult]:
        """Find paragraphs that are exact matches."""
        self.logger.info(f"Finding exact matches among {len(paragraphs)} paragraphs")
        
        results = []
        content_dict = {}
        
        try:
            # Group paragraphs by content
            for para in paragraphs:
                content = self._normalize_text(para['content'])
                if content not in content_dict:
                    content_dict[content] = []
                content_dict[content].append(para)
            
            # Find groups with more than one paragraph
            for content, para_group in content_dict.items():
                if len(para_group) > 1:
                    # Create similarity results for all pairs in group
                    for i in range(len(para_group)):
                        for j in range(i+1, len(para_group)):
                            # Skip if both paragraphs are from the same document
                            if para_group[i]['doc_id'] == para_group[j]['doc_id']:
                                continue
                                
                            result = SimilarityResult(
                                paragraph1_id=para_group[i]['id'],
                                paragraph2_id=para_group[j]['id'],
                                paragraph1_content=para_group[i]['content'],
                                paragraph2_content=para_group[j]['content'],
                                paragraph1_doc_id=para_group[i]['doc_id'],
                                paragraph2_doc_id=para_group[j]['doc_id'],
                                similarity_score=1.0,
                                similarity_type='exact'
                            )
                            results.append(result)
            
            self.logger.info(f"Found {len(results)} exact matches")
            
        except Exception as e:
            self.logger.error(f"Error finding exact matches: {str(e)}", exc_info=True)
        
        return results
    
    def find_similar_paragraphs(self, paragraphs: List[Dict], threshold: float = None) -> List[SimilarityResult]:
        """Find paragraphs that are similar based on the given threshold."""
        threshold = threshold or self.threshold
        self.logger.info(f"Finding similar paragraphs with threshold {threshold}")
        
        results = []
        
        try:
            # Skip if there are too few paragraphs
            if len(paragraphs) < 2:
                return []
                
            # Prepare corpus for TF-IDF
            corpus = [self._preprocess_text(para['content']) for para in paragraphs]
            
            # Calculate TF-IDF and cosine similarity
            tfidf_results = self._calculate_tfidf_similarity(corpus, paragraphs, threshold)
            results.extend(tfidf_results)
            
            # Calculate Jaccard similarity for additional comparison
            jaccard_results = self._calculate_jaccard_similarity(corpus, paragraphs, threshold)
            
            # Merge results, prioritizing higher similarity scores
            results = self._merge_similarity_results(results, jaccard_results)
            
            self.logger.info(f"Found {len(results)} similar paragraph pairs")
            
        except Exception as e:
            self.logger.error(f"Error finding similar paragraphs: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_tfidf_similarity(self, corpus: List[str], paragraphs: List[Dict], threshold: float) -> List[SimilarityResult]:
        """Calculate similarity using TF-IDF and cosine similarity."""
        results = []
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            # Identify pairs above threshold
            for i in range(len(paragraphs)):
                for j in range(i+1, len(paragraphs)):
                    # Skip if both paragraphs are from the same document
                    if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                        continue
                        
                    similarity = cosine_sim[i, j]
                    
                    if similarity >= threshold:
                        result = SimilarityResult(
                            paragraph1_id=paragraphs[i]['id'],
                            paragraph2_id=paragraphs[j]['id'],
                            paragraph1_content=paragraphs[i]['content'],
                            paragraph2_content=paragraphs[j]['content'],
                            paragraph1_doc_id=paragraphs[i]['doc_id'],
                            paragraph2_doc_id=paragraphs[j]['doc_id'],
                            similarity_score=float(similarity),
                            similarity_type='tfidf'
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error calculating TF-IDF similarity: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_jaccard_similarity(self, corpus: List[str], paragraphs: List[Dict], threshold: float) -> List[SimilarityResult]:
        """Calculate similarity using Jaccard similarity."""
        results = []
        
        try:
            # Calculate word sets for each paragraph
            word_sets = [set(text.split()) for text in corpus]
            
            # Calculate Jaccard similarity for each pair
            for i in range(len(paragraphs)):
                for j in range(i+1, len(paragraphs)):
                    # Skip if both paragraphs are from the same document
                    if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                        continue
                    
                    # Calculate Jaccard similarity
                    set_i = word_sets[i]
                    set_j = word_sets[j]
                    
                    if not set_i or not set_j:
                        continue
                        
                    intersection = len(set_i.intersection(set_j))
                    union = len(set_i.union(set_j))
                    
                    if union == 0:
                        similarity = 0
                    else:
                        similarity = intersection / union
                    
                    if similarity >= threshold:
                        result = SimilarityResult(
                            paragraph1_id=paragraphs[i]['id'],
                            paragraph2_id=paragraphs[j]['id'],
                            paragraph1_content=paragraphs[i]['content'],
                            paragraph2_content=paragraphs[j]['content'],
                            paragraph1_doc_id=paragraphs[i]['doc_id'],
                            paragraph2_doc_id=paragraphs[j]['doc_id'],
                            similarity_score=similarity,
                            similarity_type='jaccard'
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error calculating Jaccard similarity: {str(e)}", exc_info=True)
        
        return results
    
    def _merge_similarity_results(self, results1: List[SimilarityResult], results2: List[SimilarityResult]) -> List[SimilarityResult]:
        """Merge similarity results, keeping the highest score for each pair."""
        # Create a dictionary to track highest similarity for each paragraph pair
        pair_dict = {}
        
        # Process first result set
        for result in results1:
            key = self._make_pair_key(result.paragraph1_id, result.paragraph2_id)
            if key not in pair_dict or result.similarity_score > pair_dict[key].similarity_score:
                pair_dict[key] = result
        
        # Process second result set
        for result in results2:
            key = self._make_pair_key(result.paragraph1_id, result.paragraph2_id)
            if key not in pair_dict or result.similarity_score > pair_dict[key].similarity_score:
                pair_dict[key] = result
        
        # Return merged results
        return list(pair_dict.values())
    
    def _make_pair_key(self, id1: int, id2: int) -> tuple:
        """Create a unique key for a paragraph pair."""
        # Always put lower ID first to ensure consistent key
        return (min(id1, id2), max(id1, id2))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for exact matching."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up a logger instance."""
        logger = logging.getLogger(f'{__name__}.SimilarityAnalyzer')
        
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
