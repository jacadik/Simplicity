import logging
import re
import string
from typing import List, Dict, Tuple, Set, Any, Optional, Generator
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from datasketch import MinHash, MinHashLSH
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


@dataclass
class SimilarityResult:
    """Class for storing similarity analysis results."""
    paragraph1_id: int
    paragraph2_id: int
    paragraph1_content: str
    paragraph2_content: str
    paragraph1_doc_id: int
    paragraph2_doc_id: int
    content_similarity_score: float  # Semantic/content similarity (TF-IDF or Jaccard)
    text_similarity_score: float     # Character-based similarity (SequenceMatcher)
    similarity_type: str  # 'exact' or 'similar'


class SimilarityAnalyzer:
    """
    Analyzes paragraphs to find exact matches and similar paragraphs
    using multiple similarity metrics with LSH optimization.
    """
    def __init__(self, threshold: float = 0.8, logging_level: str = 'INFO',
                 num_perm: int = 128, min_length: int = 10):
        """
        Initialize the similarity analyzer.
        
        Args:
            threshold: Similarity threshold for considering paragraphs similar
            logging_level: Logging level
            num_perm: Number of permutations for MinHash (higher = more accurate)
            min_length: Minimum paragraph length for comparison
        """
        self.threshold = float(threshold)  # Ensure threshold is a float
        self.logger = self._setup_logger(logging_level)
        self.num_perm = num_perm  # Number of permutations for MinHash
        self.min_length = min_length  # Minimum paragraph length to consider
        self.stopwords = self._get_stopwords()
        
    def _get_stopwords(self) -> Set[str]:
        """Get common English stopwords."""
        common_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'this', 'that', 'these', 'those', 'of', 'up', 'down'
        }
        return common_stopwords
    
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
        
        # Add console handler if not already added
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison to find exact matches.
        This improved version preserves the essential content while ignoring formatting differences.
        """
        if text is None or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove all whitespace and replace with a single space
        text = ' '.join(text.split())
        
        # Remove common punctuation that doesn't affect meaning
        text = re.sub(r'[,.;:!?"\'\(\)\[\]]', '', text)
        
        # Log normalized text for debugging
        if len(text) > 50:
            self.logger.debug(f"Normalized text: {text[:50]}...")
        
        return text.strip()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF analysis."""
        if text is None or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def _get_paragraph_shingles(self, text: str, k: int = 3) -> List[str]:
        """
        Extract k-shingles (character n-grams) from text for MinHash.
        Using character-level shingles works better for short paragraphs.
        
        Args:
            text: The text to extract shingles from
            k: The size of each shingle (n-gram)
            
        Returns:
            List of shingles
        """
        # Remove punctuation and normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For very short texts, use shorter shingles
        if len(text) < 30 and k > 2:
            k = 2
            
        # Generate character shingles
        shingles = []
        for i in range(len(text) - k + 1):
            shingle = text[i:i+k]
            shingles.append(shingle)
            
        return shingles
    
    def _get_paragraph_tokens(self, text: str) -> List[str]:
        """
        Extract word tokens from text, removing stopwords.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens and remove stopwords
        tokens = [word for word in text.split() if word not in self.stopwords]
        
        return tokens

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create a MinHash signature for a paragraph.
        
        Args:
            text: The paragraph text
            
        Returns:
            MinHash signature
        """
        # For longer paragraphs, use word tokens to capture semantic similarity
        if len(text) > 100:
            tokens = self._get_paragraph_tokens(text)
            shingles = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        else:
            # For shorter paragraphs, use character shingles
            shingles = self._get_paragraph_shingles(text)
        
        # Create MinHash signature
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
            
        return minhash

    def _batch_paragraphs(self, paragraphs: List[Dict], batch_size: int = 1000) -> Generator[List[Dict], None, None]:
        """
        Split paragraphs into batches for processing.
        
        Args:
            paragraphs: List of paragraph dictionaries
            batch_size: Size of each batch
            
        Yields:
            Batches of paragraphs
        """
        for i in range(0, len(paragraphs), batch_size):
            yield paragraphs[i:i+batch_size]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate character-based similarity between two texts using SequenceMatcher.
        This version directly uses the ratio() method for more reliable comparison.
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            # Use SequenceMatcher for direct ratio calculation
            matcher = SequenceMatcher(None, text1, text2)
            similarity = matcher.ratio()
            
            self.logger.debug(f"Text similarity: {similarity:.6f}")
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def find_exact_matches(self, paragraphs: List[Dict]) -> List[SimilarityResult]:
        """Find paragraphs that are exact matches."""
        self.logger.info(f"Finding exact matches among {len(paragraphs)} paragraphs")
        
        results = []
        content_dict = {}
        
        try:
            # Group paragraphs by normalized content
            for para in paragraphs:
                # Extract the content safely
                content = para.get('content', '')
                if not content:
                    self.logger.warning(f"Skipping paragraph with empty content, ID: {para.get('id')}")
                    continue
                
                # Normalize the content for comparison
                normalized = self._normalize_text(content)
                if not normalized:
                    self.logger.warning(f"Normalization resulted in empty string for paragraph ID: {para.get('id')}")
                    continue
                
                # Store with normalized content as key
                if normalized not in content_dict:
                    content_dict[normalized] = []
                content_dict[normalized].append(para)
            
            # Find groups with more than one paragraph
            for normalized, para_group in content_dict.items():
                if len(para_group) > 1:
                    self.logger.info(f"Found group with {len(para_group)} matching paragraphs with normalized content: {normalized[:50]}...")
                    
                    # Create similarity results for all pairs in group
                    for i in range(len(para_group)):
                        for j in range(i+1, len(para_group)):
                            # Get document IDs safely
                            doc_id1 = para_group[i].get('doc_id')
                            doc_id2 = para_group[j].get('doc_id')
                            
                            # Ensure document IDs are valid and different
                            if doc_id1 is None or doc_id2 is None:
                                self.logger.warning(f"Missing document ID in paragraphs: {doc_id1}, {doc_id2}")
                                continue
                            
                            # Skip if both paragraphs are from the same document
                            if doc_id1 == doc_id2:
                                self.logger.debug(f"Skipping comparison of paragraphs from same document ID: {doc_id1}")
                                continue
                            
                            # Log the match for debugging
                            self.logger.info(f"Found exact match between paragraphs {para_group[i]['id']} and {para_group[j]['id']}")
                            
                            # Create similarity result
                            result = SimilarityResult(
                                paragraph1_id=para_group[i]['id'],
                                paragraph2_id=para_group[j]['id'],
                                paragraph1_content=para_group[i]['content'],
                                paragraph2_content=para_group[j]['content'],
                                paragraph1_doc_id=doc_id1,
                                paragraph2_doc_id=doc_id2,
                                content_similarity_score=1.0,  # Perfect content match
                                text_similarity_score=1.0,     # Perfect text match
                                similarity_type='exact'
                            )
                            results.append(result)
            
            self.logger.info(f"Found {len(results)} exact matches")
            
        except Exception as e:
            self.logger.error(f"Error finding exact matches: {str(e)}", exc_info=True)
        
        return results
    
    def fast_similarity_search(self, paragraphs: List[Dict], 
                              threshold: float) -> List[Tuple[int, int, float]]:
        """
        Find similar paragraphs using MinHash LSH.
        
        Args:
            paragraphs: List of paragraph dictionaries
            threshold: Similarity threshold
            
        Returns:
            List of tuples (paragraph1_id, paragraph2_id, estimated_similarity)
        """
        self.logger.info(f"Running fast similarity search on {len(paragraphs)} paragraphs")
        
        # Use a threshold slightly lower for LSH to catch more candidates
        lsh_threshold = max(0.1, threshold - 0.1)
        
        # Initialize LSH index
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=self.num_perm)
        
        # Store paragraph info
        para_info = {}
        
        # Prefilter paragraphs by length
        valid_paragraphs = []
        for para in paragraphs:
            content = para.get('content', '')
            if len(content) >= self.min_length:
                valid_paragraphs.append(para)
            else:
                self.logger.debug(f"Skipping paragraph {para.get('id')} (too short: {len(content)} chars)")
                
        self.logger.info(f"Processing {len(valid_paragraphs)} paragraphs after length filtering")
        
        # Create MinHash signatures for all paragraphs
        try:
            for para in valid_paragraphs:
                para_id = para['id']
                content = para['content']
                doc_id = para['doc_id']
                
                # Create MinHash signature
                minhash = self._create_minhash(content)
                
                # Store paragraph info
                para_info[para_id] = {
                    'content': content,
                    'doc_id': doc_id,
                    'minhash': minhash
                }
                
                # Add to LSH index
                lsh.insert(f"{para_id}", minhash)
                
            # Find similar pairs
            similar_pairs = []
            para_ids = list(para_info.keys())
            
            # For each paragraph, query the LSH index
            for para_id in para_ids:
                # Query the LSH index for similar paragraphs
                minhash = para_info[para_id]['minhash']
                doc_id = para_info[para_id]['doc_id']
                
                # Get candidates from LSH
                candidates = lsh.query(minhash)
                
                # Process candidates
                for candidate in candidates:
                    candidate_id = int(candidate)
                    
                    # Skip self-comparisons
                    if candidate_id == para_id:
                        continue
                        
                    # Skip comparisons from the same document
                    candidate_doc_id = para_info[candidate_id]['doc_id']
                    if candidate_doc_id == doc_id:
                        continue
                        
                    # Ensure sorted order to avoid duplicates
                    if para_id > candidate_id:
                        continue
                        
                    # Calculate estimated similarity
                    candidate_minhash = para_info[candidate_id]['minhash']
                    estimated_similarity = minhash.jaccard(candidate_minhash)
                    
                    # Add pair if above threshold
                    if estimated_similarity >= threshold:
                        similar_pairs.append((para_id, candidate_id, estimated_similarity))
            
            self.logger.info(f"Found {len(similar_pairs)} similar paragraph pairs using LSH")
            return similar_pairs
            
        except Exception as e:
            self.logger.error(f"Error in fast similarity search: {str(e)}", exc_info=True)
            return []

    def find_similar_paragraphs(self, paragraphs: List[Dict], threshold: float = None) -> List[SimilarityResult]:
        """Find paragraphs that are similar based on the given threshold using LSH."""
        # Convert threshold to float and use default if None
        threshold = float(threshold) if threshold is not None else self.threshold
        self.logger.info(f"Finding similar paragraphs with threshold {threshold}")
        
        results = []
        
        try:
            # Skip if there are too few paragraphs
            if len(paragraphs) < 2:
                self.logger.warning("Not enough paragraphs to compare")
                return []
            
            # Filter out paragraphs without content
            valid_paragraphs = []
            for para in paragraphs:
                if 'content' in para and para['content'] and 'id' in para and 'doc_id' in para:
                    valid_paragraphs.append(para)
                else:
                    self.logger.warning(f"Skipping invalid paragraph: {para}")
            
            if len(valid_paragraphs) < 2:
                self.logger.warning("Not enough valid paragraphs to compare")
                return []
            
            # Debug log
            self.logger.debug(f"Processing {len(valid_paragraphs)} valid paragraphs")
            
            # Check for exact matches first (this is more efficient)
            exact_matches = self.find_exact_matches(valid_paragraphs)
            self.logger.debug(f"Found {len(exact_matches)} exact matches from internal method")
            
            # Extract exact match pairs to avoid duplicates
            exact_match_pairs = set()
            for result in exact_matches:
                # Use a consistent order for the pair to avoid duplicates
                pair = tuple(sorted([result.paragraph1_id, result.paragraph2_id]))
                exact_match_pairs.add(pair)
            
            # Add exact matches to results
            results.extend(exact_matches)
                
            # Find similar pairs using LSH
            similar_pairs = self.fast_similarity_search(valid_paragraphs, threshold)
            
            # Filter out pairs that are already exact matches
            filtered_pairs = []
            for para1_id, para2_id, est_similarity in similar_pairs:
                pair = tuple(sorted([para1_id, para2_id]))
                if pair not in exact_match_pairs:
                    filtered_pairs.append((para1_id, para2_id, est_similarity))
            
            # Create a lookup for paragraphs by ID
            para_lookup = {para['id']: para for para in valid_paragraphs}
            
            # Process candidate pairs with more accurate similarity measures
            for para1_id, para2_id, est_similarity in filtered_pairs:
                para1 = para_lookup.get(para1_id)
                para2 = para_lookup.get(para2_id)
                
                if not para1 or not para2:
                    continue
                
                # Calculate more accurate content similarity
                content_similarity = self._calculate_jaccard_similarity(
                    para1['content'],
                    para2['content']
                )
                
                # Calculate text similarity using SequenceMatcher ratio directly
                text_similarity = self._calculate_text_similarity(
                    para1['content'],
                    para2['content']
                )
                
                # Ensure text similarity is not zero
                if text_similarity < 0.01:
                    text_similarity = 0.01  # Set a minimum value to avoid zero
                
                # Only include if actual similarity is above threshold
                if content_similarity >= threshold:
                    result = SimilarityResult(
                        paragraph1_id=para1_id,
                        paragraph2_id=para2_id,
                        paragraph1_content=para1['content'],
                        paragraph2_content=para2['content'],
                        paragraph1_doc_id=para1['doc_id'],
                        paragraph2_doc_id=para2['doc_id'],
                        content_similarity_score=content_similarity,
                        text_similarity_score=text_similarity,
                        similarity_type='similar'
                    )
                    results.append(result)
            
            self.logger.info(f"Found total of {len(results)} similarity results")
            
        except Exception as e:
            self.logger.error(f"Error finding similar paragraphs: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        try:
            # Normalize texts
            text1 = text1.lower()
            text2 = text2.lower()
            
            # Remove punctuation
            text1 = re.sub(r'[^\w\s]', '', text1)
            text2 = re.sub(r'[^\w\s]', '', text2)
            
            # Split into words
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            # Remove stopwords
            words1 = words1.difference(self.stopwords)
            words2 = words2.difference(self.stopwords)
            
            # Calculate Jaccard similarity
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            # Avoid division by zero
            if not union:
                return 0.0
                
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.error(f"Error calculating Jaccard similarity: {str(e)}")
            return 0.0
    
    def _merge_similarity_results(self, results1: List[SimilarityResult], results2: List[SimilarityResult]) -> List[SimilarityResult]:
        """Merge similarity results, keeping the highest score for each pair."""
        # Create a dictionary to track pairs
        pair_map = {}
        
        # Process first result set
        for result in results1:
            key = self._make_pair_key(result.paragraph1_id, result.paragraph2_id)
            pair_map[key] = result
        
        # Process second result set, keeping highest score
        for result in results2:
            key = self._make_pair_key(result.paragraph1_id, result.paragraph2_id)
            if key in pair_map:
                existing = pair_map[key]
                if result.content_similarity_score > existing.content_similarity_score:
                    pair_map[key] = result
            else:
                pair_map[key] = result
        
        # Convert back to list
        return list(pair_map.values())
    
    def _make_pair_key(self, id1: int, id2: int) -> Tuple[int, int]:
        """Make a consistent key for a pair of IDs."""
        return tuple(sorted([id1, id2]))
    
    def cluster_paragraphs(self, similarity_results: List[SimilarityResult], threshold: float = None, 
                          similarity_type: str = 'content') -> List[Dict]:
        """
        Cluster paragraphs using graph-based community detection.
        
        Args:
            similarity_results: List of similarity results
            threshold: Minimum similarity score to consider paragraphs related
            similarity_type: Which similarity metric to use ('content' or 'text')
                
        Returns:
            List of clusters, where each cluster is a dict with name and paragraph IDs
        """
        threshold = float(threshold) if threshold is not None else self.threshold
        self.logger.info(f"Clustering paragraphs with {similarity_type} similarity threshold {threshold}")
        
        try:
            # Create a graph
            G = nx.Graph()
            
            # Add nodes and edges for similarities above threshold
            for result in similarity_results:
                # Determine which similarity score to use based on the requested type
                if similarity_type == 'text':
                    similarity_score = result.text_similarity_score
                    metric_name = "text similarity"
                else:  # Default to content similarity
                    similarity_score = result.content_similarity_score
                    metric_name = "content similarity"
                    
                if similarity_score >= threshold:
                    # Add nodes with paragraph content as attribute
                    G.add_node(result.paragraph1_id, content=result.paragraph1_content)
                    G.add_node(result.paragraph2_id, content=result.paragraph2_content)
                    
                    # Add edge with similarity score as weight
                    G.add_edge(
                        result.paragraph1_id, 
                        result.paragraph2_id, 
                        weight=similarity_score
                    )
            
            # Skip if graph is empty
            if len(G.nodes) == 0:
                self.logger.warning(f"No paragraphs to cluster using {metric_name}")
                return []
                
            # Find communities using Louvain method (optimizes modularity)
            try:
                communities = nx.community.louvain_communities(G)
            except Exception as e:
                self.logger.error(f"Louvain community detection failed: {str(e)}", exc_info=True)
                # Fallback to simpler community detection
                communities = list(nx.connected_components(G))
                self.logger.info(f"Using connected components as fallback, found {len(communities)} communities")
            
            # Convert to list of dicts
            clusters = []
            for i, community in enumerate(communities):
                # Convert community to list
                community_list = list(community)
                
                # Get a representative paragraph for naming the cluster
                if community_list:
                    representative_id = community_list[0]
                    rep_content = G.nodes[representative_id].get('content', '')
                    # Use the first 50 chars of the representative paragraph for the name
                    cluster_name = f"Cluster {i+1}: {rep_content[:50]}..."
                else:
                    cluster_name = f"Cluster {i+1}"
                    
                clusters.append({
                    'name': cluster_name,
                    'description': f"Auto-generated cluster with {len(community_list)} paragraphs using {metric_name}",
                    'creation_date': datetime.now().isoformat(),
                    'similarity_threshold': threshold,
                    'paragraph_ids': community_list,
                    'similarity_type': similarity_type  # Store which similarity metric was used
                })
            
            self.logger.info(f"Created {len(clusters)} clusters using {metric_name}")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering paragraphs: {str(e)}", exc_info=True)
            return []