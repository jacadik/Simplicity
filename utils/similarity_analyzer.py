"""
Optimized SimilarityAnalyzer module with backward compatibility.
Keeps the original class name for compatibility with existing imports.
"""

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
import time
import psutil  # Add this to your requirements.txt


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
    Optimized version of SimilarityAnalyzer with improved performance for large datasets.
    Maintains the original class name for backward compatibility.
    """
    def __init__(self, threshold: float = 0.8, logging_level: str = 'INFO',
                 num_perm: int = 128, min_length: int = 10, batch_size: int = 1000,
                 max_workers: Optional[int] = None):
        """
        Initialize the similarity analyzer with optimized parameters.
        
        Args:
            threshold: Similarity threshold for considering paragraphs similar
            logging_level: Logging level
            num_perm: Number of permutations for MinHash (higher = more accurate)
            min_length: Minimum paragraph length for comparison
            batch_size: Size of batches for parallel processing
            max_workers: Maximum number of worker processes (default: CPU count)
        """
        self.threshold = float(threshold)  # Ensure threshold is a float
        self.logger = self._setup_logger(logging_level)
        self.num_perm = num_perm  # Number of permutations for MinHash
        self.min_length = min_length  # Minimum paragraph length to consider
        self.batch_size = batch_size  # Batch size for processing
        
        # Determine optimal number of workers based on available CPUs
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or max(1, min(cpu_count - 1, 8))  # Leave 1 CPU for system
        self.logger.info(f"Initializing with {self.max_workers} worker processes")
        
        # Initialize stopwords
        self.stopwords = self._get_stopwords()
        
        # Cache for normalized text to avoid repeat processing
        self._normalization_cache = {}
        self._minhash_cache = {}
        
        # Memory monitoring
        self.memory_threshold = 0.85  # 85% memory usage triggers cleanup
        
        # Performance metrics
        self.last_run_stats = {}
    
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
    
    @lru_cache(maxsize=1024)
    def _normalize_text(self, text: str) -> str:
        """
        Normalized and cached text normalization for finding exact matches.
        Using lru_cache for improved performance on repeated texts.
        """
        if text is None or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove all whitespace and replace with a single space
        text = ' '.join(text.split())
        
        # Remove common punctuation that doesn't affect meaning
        text = re.sub(r'[,.;:!?"\'\(\)\[\]]', '', text)
        
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
        """Extract k-shingles (character n-grams) from text for MinHash."""
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
        """Extract word tokens from text, removing stopwords."""
        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens and remove stopwords
        tokens = [word for word in text.split() if word not in self.stopwords]
        
        return tokens

    def _create_minhash(self, text: str) -> MinHash:
        """Create a MinHash signature for a paragraph with caching."""
        # First check cache using text hash to avoid memory bloat
        text_hash = hash(text)
        if text_hash in self._minhash_cache:
            return self._minhash_cache[text_hash]
            
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
        
        # Cache the result (limited by memory monitoring)
        self._check_memory_usage()
        self._minhash_cache[text_hash] = minhash
            
        return minhash
    
    def _check_memory_usage(self):
        """Monitor memory usage and clear caches if needed."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > (self.memory_threshold * 100):
                self.logger.warning(f"Memory usage high ({memory.percent}%). Clearing caches.")
                self._normalization_cache.clear()
                self._minhash_cache.clear()
        except Exception as e:
            self.logger.warning(f"Could not check memory usage: {str(e)}")

    def _batch_paragraphs(self, paragraphs: List[Dict], batch_size: Optional[int] = None) -> Generator[List[Dict], None, None]:
        """Split paragraphs into batches for processing."""
        batch_size = batch_size or self.batch_size
        for i in range(0, len(paragraphs), batch_size):
            yield paragraphs[i:i+batch_size]
    
    def find_exact_matches(self, paragraphs: List[Dict]) -> List[SimilarityResult]:
        """Find paragraphs that are exact matches using an optimized approach."""
        start_time = time.time()
        self.logger.info(f"Finding exact matches among {len(paragraphs)} paragraphs")
        
        results = []
        content_dict = {}
        
        try:
            # Group paragraphs by normalized content using optimized batching
            batch_size = min(self.batch_size, len(paragraphs))
            processed = 0
            
            for batch in self._batch_paragraphs(paragraphs, batch_size):
                # Process each paragraph in the batch
                for para in batch:
                    # Extract the content safely
                    content = para.get('content', '')
                    if not content:
                        continue
                    
                    # Normalize the content for comparison
                    normalized = self._normalize_text(content)
                    if not normalized:
                        continue
                    
                    # Store with normalized content as key
                    if normalized not in content_dict:
                        content_dict[normalized] = []
                    content_dict[normalized].append(para)
                
                # Update progress
                processed += len(batch)
                if processed % 5000 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Processed {processed}/{len(paragraphs)} paragraphs in {elapsed:.2f}s")
                    
                # Check memory usage periodically
                if processed % 10000 == 0:
                    self._check_memory_usage()
            
            # Find groups with more than one paragraph and create similarity results
            duplicate_groups = 0
            for normalized, para_group in content_dict.items():
                if len(para_group) > 1:
                    duplicate_groups += 1
                    
                    # Create similarity results for all pairs in group
                    for i in range(len(para_group)):
                        for j in range(i+1, len(para_group)):
                            # Get document IDs safely
                            doc_id1 = para_group[i].get('doc_id')
                            doc_id2 = para_group[j].get('doc_id')
                            
                            # Ensure document IDs are valid and different
                            if doc_id1 is None or doc_id2 is None or doc_id1 == doc_id2:
                                continue
                            
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
            
            elapsed_time = time.time() - start_time
            avg_time_per_para = elapsed_time / len(paragraphs) if paragraphs else 0
            
            self.logger.info(f"Found {len(results)} exact matches in {duplicate_groups} duplicate groups")
            self.logger.info(f"Exact match search completed in {elapsed_time:.2f}s ({avg_time_per_para:.4f}s per paragraph)")
            
            # Save performance metrics
            self.last_run_stats.update({
                'exact_match_time': elapsed_time,
                'exact_match_count': len(results),
                'duplicate_groups': duplicate_groups,
                'paragraphs_processed': len(paragraphs)
            })
            
        except Exception as e:
            self.logger.error(f"Error finding exact matches: {str(e)}", exc_info=True)
        
        return results
    
    def _process_paragraph_batch(self, batch_data):
        """Process a batch of paragraphs for LSH-based similarity search."""
        batch, lsh_threshold = batch_data
        
        # Initialize LSH index
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=self.num_perm)
        
        # Store paragraph info
        para_info = {}
        
        # Create MinHash signatures and populate LSH index
        for para in batch:
            para_id = para['id']
            content = para['content']
            doc_id = para['doc_id']
            
            # Skip paragraphs that are too short
            if len(content) < self.min_length:
                continue
                
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
        
        # Find similar pairs within this batch
        similar_pairs = []
        para_ids = list(para_info.keys())
        
        # For each paragraph, query the LSH index
        for para_id in para_ids:
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
                if estimated_similarity >= self.threshold:
                    similar_pairs.append((para_id, candidate_id, estimated_similarity))
        
        return similar_pairs
    
    def fast_similarity_search(self, paragraphs: List[Dict], threshold: float) -> List[Tuple[int, int, float]]:
        """
        Find similar paragraphs using MinHash LSH with parallel processing.
        
        Args:
            paragraphs: List of paragraph dictionaries
            threshold: Similarity threshold
            
        Returns:
            List of tuples (paragraph1_id, paragraph2_id, estimated_similarity)
        """
        start_time = time.time()
        self.logger.info(f"Running fast similarity search on {len(paragraphs)} paragraphs")
        
        # Use a slightly lower threshold for LSH to catch more candidates
        lsh_threshold = max(0.1, threshold - 0.1)
        
        # Prefilter paragraphs by length
        valid_paragraphs = []
        for para in paragraphs:
            content = para.get('content', '')
            if len(content) >= self.min_length:
                valid_paragraphs.append(para)
        
        valid_count = len(valid_paragraphs)
        self.logger.info(f"Processing {valid_count} paragraphs after length filtering")
        
        if valid_count == 0:
            return []
            
        # Determine batch size based on paragraph count
        if valid_count > 10000:
            # For very large sets, use smaller batches to control memory
            batch_size = min(self.batch_size, 500)
        else:
            batch_size = min(valid_count, self.batch_size)
            
        self.logger.info(f"Using batch size of {batch_size} paragraphs")
        
        # Create batches for parallel processing
        batches = list(self._batch_paragraphs(valid_paragraphs, batch_size))
        self.logger.info(f"Split data into {len(batches)} batches")
        
        # Process batches with parallel execution
        similar_pairs = []
        try:
            # If we have a very small dataset, just process it directly
            if valid_count < 500:
                self.logger.info("Small dataset, processing in a single batch")
                similar_pairs = self._process_paragraph_batch((valid_paragraphs, lsh_threshold))
            else:
                # Use process pool for parallel execution
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all batches for processing
                    batch_data = [(batch, lsh_threshold) for batch in batches]
                    results = list(executor.map(self._process_paragraph_batch, batch_data))
                    
                    # Collect results
                    for batch_result in results:
                        similar_pairs.extend(batch_result)
        
        except Exception as e:
            self.logger.error(f"Error in parallel similarity search: {str(e)}", exc_info=True)
        
        elapsed_time = time.time() - start_time
        avg_time_per_para = elapsed_time / valid_count if valid_count else 0
        
        self.logger.info(f"Found {len(similar_pairs)} similar paragraph pairs using LSH")
        self.logger.info(f"Fast similarity search completed in {elapsed_time:.2f}s ({avg_time_per_para:.4f}s per paragraph)")
        
        # Save performance metrics
        self.last_run_stats.update({
            'fast_search_time': elapsed_time,
            'similar_pairs': len(similar_pairs),
            'valid_paragraphs': valid_count
        })
        
        return similar_pairs

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Optimized character-based similarity calculation."""
        if not text1 or not text2:
            return 0.0
            
        try:
            # Normalize texts for better comparison
            clean_text1 = text1.strip().lower()
            clean_text2 = text2.strip().lower()
            
            # Quick check for contained text
            if clean_text1 in clean_text2:
                # Text1 is contained in text2
                similarity = len(clean_text1) / len(clean_text2)
                return min(1.0, similarity)  # Cap at 1.0
            
            if clean_text2 in clean_text1:
                # Text2 is contained in text1
                similarity = len(clean_text2) / len(clean_text1)
                return min(1.0, similarity)  # Cap at 1.0
            
            # For longer texts, use a faster approach
            if len(clean_text1) > 1000 or len(clean_text2) > 1000:
                # Use a chunking approach for long texts
                return self._calculate_chunked_similarity(clean_text1, clean_text2)
            
            # For shorter texts, use SequenceMatcher
            matcher = SequenceMatcher(None, clean_text1, clean_text2)
            return matcher.ratio()
                
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def _calculate_chunked_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between long texts using a chunking approach."""
        # Get shorter and longer text
        if len(text1) <= len(text2):
            shorter, longer = text1, text2
        else:
            shorter, longer = text2, text1
            
        chunk_size = min(100, len(shorter) // 2)
        if chunk_size < 20:
            # Text is too short for chunking, use regular approach
            matcher = SequenceMatcher(None, text1, text2)
            return matcher.ratio()
            
        # Extract chunks from shorter text
        chunks = []
        for i in range(0, len(shorter), chunk_size // 2):
            chunk = shorter[i:i+chunk_size]
            if len(chunk) >= 20:  # Only consider substantial chunks
                chunks.append(chunk)
                
        if not chunks:
            # Fallback if no substantial chunks
            return 0.0
                
        # Count matches in longer text
        matched_chars = 0
        for chunk in chunks:
            if chunk in longer:
                matched_chars += len(chunk)
                
        # Calculate similarity based on matched character ratio
        return min(1.0, matched_chars / len(shorter))
    
    @lru_cache(maxsize=1024)
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts with caching."""
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
    
    def find_similar_paragraphs(self, paragraphs: List[Dict], threshold: float = None) -> List[SimilarityResult]:
        """Find paragraphs that are similar based on the given threshold using optimized LSH."""
        total_start_time = time.time()
        
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
            
            if len(valid_paragraphs) < 2:
                self.logger.warning("Not enough valid paragraphs to compare")
                return []
            
            # Check for exact matches first (this is more efficient)
            exact_matches = self.find_exact_matches(valid_paragraphs)
            exact_match_count = len(exact_matches)
            self.logger.info(f"Found {exact_match_count} exact matches")
            
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
            similar_pair_count = len(similar_pairs)
            self.logger.info(f"Found {similar_pair_count} similar paragraph pairs using LSH")
            
            # Filter out pairs that are already exact matches
            filtered_pairs = []
            for para1_id, para2_id, est_similarity in similar_pairs:
                pair = tuple(sorted([para1_id, para2_id]))
                if pair not in exact_match_pairs:
                    filtered_pairs.append((para1_id, para2_id, est_similarity))
            
            filtered_count = len(filtered_pairs)
            self.logger.info(f"After filtering exact matches: {filtered_count} similar pairs remaining")
            
            # Create a lookup for paragraphs by ID
            para_lookup = {para['id']: para for para in valid_paragraphs}
            
            # Process candidate pairs with more accurate similarity measures
            # If we have a large number of pairs, process in batches
            batch_size = 1000  # Smaller batch size for detailed comparison
            
            similar_comparison_start = time.time()
            processed_count = 0
            pair_batches = [filtered_pairs[i:i+batch_size] for i in range(0, len(filtered_pairs), batch_size)]
            
            for batch in pair_batches:
                batch_start = time.time()
                batch_results = []
                
                for para1_id, para2_id, est_similarity in batch:
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
                        batch_results.append(result)
                
                results.extend(batch_results)
                processed_count += len(batch)
                
                batch_time = time.time() - batch_start
                if batch_time > 1.0:  # Only log if batch took significant time
                    progress = (processed_count / len(filtered_pairs)) * 100 if filtered_pairs else 100
                    self.logger.info(f"Processed {processed_count}/{len(filtered_pairs)} pairs ({progress:.1f}%) in {batch_time:.2f}s")
                
                # Check memory usage after each batch
                self._check_memory_usage()
            
            similar_comparison_time = time.time() - similar_comparison_start
            self.logger.info(f"Detailed similarity comparison completed in {similar_comparison_time:.2f}s")
            
            # Final results
            total_time = time.time() - total_start_time
            self.logger.info(f"Total similarity analysis time: {total_time:.2f}s")
            self.logger.info(f"Found total of {len(results)} similarity results")
            
            # Save performance metrics
            self.last_run_stats.update({
                'total_similarity_time': total_time,
                'similar_comparison_time': similar_comparison_time,
                'total_results': len(results),
                'similar_results': len(results) - exact_match_count,
                'paragraphs_compared': len(valid_paragraphs)
            })
            
        except Exception as e:
            self.logger.error(f"Error finding similar paragraphs: {str(e)}", exc_info=True)
        
        return results
    
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
            import networkx as nx
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
                self.logger.error(f"Louvain community detection failed: {str(e)}")
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance statistics from the last run."""
        return self.last_run_stats.copy()
    
    def clear_caches(self) -> None:
        """Manually clear caches."""
        self.logger.info("Clearing analyzer caches")
        self._normalization_cache.clear()
        self._minhash_cache.clear()
        
        # Also clear lru_cache
        self._normalize_text.cache_clear()
        self._calculate_jaccard_similarity.cache_clear()