import logging
import re
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime
from difflib import SequenceMatcher


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
    using multiple similarity metrics.
    """
    def __init__(self, threshold: float = 0.8, logging_level: str = 'INFO'):
        """Initialize the similarity analyzer."""
        self.threshold = float(threshold)  # Ensure threshold is a float
        self.logger = self._setup_logger(logging_level)
    
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate character-based similarity between two texts.
        This implementation matches the JavaScript implementation that shows ~0% when texts differ completely.
        """
        if not text1 or not text2:
            return 0.0
            
        try:
            # Split texts into words for a stricter comparison
            words1 = text1.split()
            words2 = text2.split()
            
            # Calculate exact matches of whole words in same positions
            # This is a much stricter approach that will result in lower similarity scores
            exact_matches = 0
            total_words = max(len(words1), len(words2))
            
            for i in range(min(len(words1), len(words2))):
                if words1[i] == words2[i]:
                    exact_matches += 1
            
            # Calculate character counts for the enhanced comparison view
            from difflib import SequenceMatcher
            matcher = SequenceMatcher(None, text1, text2)
            opcodes = matcher.get_opcodes()
            
            unchanged = 0
            added = 0
            removed = 0
            
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal':
                    # Only count characters as unchanged if they form complete words
                    # or are part of matching word boundaries
                    segment = text1[i1:i2]
                    if segment.strip() and (segment.strip() in words1 and segment.strip() in words2):
                        unchanged += (i2 - i1)
                    else:
                        # If they're just random matching characters, count as both added and removed
                        removed += (i2 - i1)
                        added += (j2 - j1)
                elif tag == 'delete':
                    removed += (i2 - i1)
                elif tag == 'insert':
                    added += (j2 - j1)
                elif tag == 'replace':
                    removed += (i2 - i1)
                    added += (j2 - j1)
            
            # If there are no exact matches, the similarity should be extremely low
            if exact_matches == 0:
                similarity = 0.0
            else:
                # Calculate similarity based on total operations
                total = unchanged + added + removed
                if total == 0:
                    similarity = 0.0
                else:
                    # Use a very stringent measure for similarity that produces lower percentages
                    similarity = unchanged / (total * 3)  # Applying a stricter penalty
            
            self.logger.debug(f"Text similarity: {similarity:.6f}, unchanged: {unchanged}, added: {added}, removed: {removed}")
            
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
    
    def find_similar_paragraphs(self, paragraphs: List[Dict], threshold: float = None) -> List[SimilarityResult]:
        """Find paragraphs that are similar based on the given threshold."""
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
            valid_contents = []
            for para in paragraphs:
                if 'content' in para and para['content'] and 'id' in para and 'doc_id' in para:
                    valid_paragraphs.append(para)
                    valid_contents.append(para['content'])
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
                
            # Prepare corpus for TF-IDF
            corpus = []
            for content in valid_contents:
                try:
                    preprocessed = self._preprocess_text(content)
                    corpus.append(preprocessed)
                except Exception as e:
                    self.logger.error(f"Error preprocessing text: {str(e)}", exc_info=True)
                    corpus.append("")  # Add empty string as fallback
            
            # Skip if all preprocessed texts are empty
            if not any(corpus):
                self.logger.warning("All preprocessed texts are empty")
                return results  # Just return exact matches
            
            # Calculate similar (non-exact) paragraphs
            similar_results = []
            
            # Try TF-IDF first
            try:
                tfidf_results = self._calculate_tfidf_similarity(corpus, valid_paragraphs, threshold)
                similar_results.extend(tfidf_results)
                self.logger.debug(f"Found {len(tfidf_results)} TF-IDF similar results")
            except Exception as e:
                self.logger.error(f"TF-IDF similarity failed: {str(e)}", exc_info=True)
            
            # Try Jaccard as backup
            try:
                jaccard_results = self._calculate_jaccard_similarity(corpus, valid_paragraphs, threshold)
                self.logger.debug(f"Found {len(jaccard_results)} Jaccard similar results")
                similar_results = self._merge_similarity_results(similar_results, jaccard_results)
            except Exception as e:
                self.logger.error(f"Jaccard similarity failed: {str(e)}", exc_info=True)
            
            # Filter out exact matches we've already found
            filtered_similar_results = []
            for result in similar_results:
                pair = tuple(sorted([result.paragraph1_id, result.paragraph2_id]))
                if pair not in exact_match_pairs:
                    result.similarity_type = 'similar'  # Ensure type is set to 'similar'
                    filtered_similar_results.append(result)
            
            results.extend(filtered_similar_results)
            self.logger.info(f"Found total of {len(results)} similarity results")
            
        except Exception as e:
            self.logger.error(f"Error finding similar paragraphs: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_tfidf_similarity(self, corpus: List[str], paragraphs: List[Dict], threshold: float) -> List[SimilarityResult]:
        """Calculate similarity using TF-IDF and cosine similarity."""
        results = []
        
        try:
            # Handle edge cases
            if len(corpus) <= 1 or len(paragraphs) <= 1:
                return []
                
            # Check for empty corpus
            if all(not text for text in corpus):
                return []
                
            # Create TF-IDF vectors - using more robust settings
            vectorizer = TfidfVectorizer(
                min_df=1,  # Include terms that appear in just one document
                strip_accents='unicode',
                use_idf=True,
                smooth_idf=True
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(corpus)
                self.logger.debug(f"Created TF-IDF matrix of shape {tfidf_matrix.shape}")
                
                # Check for zero-length documents
                zero_docs = []
                for i, doc in enumerate(corpus):
                    if not doc.strip() or len(vectorizer.get_feature_names_out()) == 0:
                        zero_docs.append(i)
                        
                if zero_docs:
                    self.logger.warning(f"Found {len(zero_docs)} empty documents after vectorization")
            except ValueError as e:
                self.logger.error(f"TF-IDF vectorization error: {str(e)}")
                return []
            
            # Calculate cosine similarity
            try:
                cosine_sim = cosine_similarity(tfidf_matrix)
                self.logger.debug(f"Cosine similarity matrix shape: {cosine_sim.shape}")
            except Exception as e:
                self.logger.error(f"Error calculating cosine similarity: {str(e)}")
                return []
            
            # Identify pairs above threshold
            pairs_above_threshold = 0
            total_pairs = 0
            
            for i in range(len(paragraphs)):
                for j in range(i+1, len(paragraphs)):
                    total_pairs += 1
                    
                    # Skip if both paragraphs are from the same document
                    if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                        continue
                    
                    # Ensure index is valid
                    if i >= cosine_sim.shape[0] or j >= cosine_sim.shape[1]:
                        self.logger.warning(f"Index out of range: {i}, {j} for shape {cosine_sim.shape}")
                        continue
                    
                    # Get content similarity value as float (TF-IDF)
                    content_similarity = float(cosine_sim[i, j])
                    
                    # Calculate text similarity using our improved method
                    text_similarity = self._calculate_text_similarity(
                        paragraphs[i]['content'],
                        paragraphs[j]['content']
                    )
                    
                    # Check if value is valid
                    if np.isnan(content_similarity) or np.isinf(content_similarity):
                        self.logger.warning(f"Invalid similarity value: {content_similarity}")
                        continue
                    
                    # Count pairs above threshold for debugging (using content similarity for threshold)
                    if content_similarity >= threshold:
                        pairs_above_threshold += 1
                        
                        result = SimilarityResult(
                            paragraph1_id=paragraphs[i]['id'],
                            paragraph2_id=paragraphs[j]['id'],
                            paragraph1_content=paragraphs[i]['content'],
                            paragraph2_content=paragraphs[j]['content'],
                            paragraph1_doc_id=paragraphs[i]['doc_id'],
                            paragraph2_doc_id=paragraphs[j]['doc_id'],
                            content_similarity_score=content_similarity,
                            text_similarity_score=text_similarity,
                            similarity_type='similar'
                        )
                        results.append(result)
            
            self.logger.debug(f"Found {pairs_above_threshold} pairs above threshold out of {total_pairs} total pairs")
            
            # Special case: if threshold is very low but no results, maybe try the highest pairs
            if threshold < 0.3 and len(results) == 0 and total_pairs > 0:
                self.logger.info("Low threshold but no results, getting top 10 similarity pairs instead")
                # Find highest similarity scores
                top_pairs = []
                for i in range(len(paragraphs)):
                    for j in range(i+1, len(paragraphs)):
                        # Skip same document
                        if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                            continue
                            
                        # Skip invalid indices
                        if i >= cosine_sim.shape[0] or j >= cosine_sim.shape[1]:
                            continue
                            
                        content_similarity = float(cosine_sim[i, j])
                        text_similarity = self._calculate_text_similarity(
                            paragraphs[i]['content'],
                            paragraphs[j]['content']
                        )
                        
                        if not np.isnan(content_similarity) and not np.isinf(content_similarity):
                            top_pairs.append((i, j, content_similarity, text_similarity))
                
                # Sort by content similarity descending and take top 10
                top_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, j, content_similarity, text_similarity in top_pairs[:10]:
                    result = SimilarityResult(
                        paragraph1_id=paragraphs[i]['id'],
                        paragraph2_id=paragraphs[j]['id'],
                        paragraph1_content=paragraphs[i]['content'],
                        paragraph2_content=paragraphs[j]['content'],
                        paragraph1_doc_id=paragraphs[i]['doc_id'],
                        paragraph2_doc_id=paragraphs[j]['doc_id'],
                        content_similarity_score=content_similarity,
                        text_similarity_score=text_similarity,
                        similarity_type='similar'
                    )
                    results.append(result)
                
                self.logger.info(f"Added {len(results)} top similarity pairs")
        
        except Exception as e:
            self.logger.error(f"Error calculating TF-IDF similarity: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_jaccard_similarity(self, corpus: List[str], paragraphs: List[Dict], threshold: float) -> List[SimilarityResult]:
        """Calculate similarity using Jaccard similarity (word-based)."""
        results = []
        
        try:
            # Process pairs
            for i in range(len(paragraphs)):
                for j in range(i+1, len(paragraphs)):
                    # Skip if both paragraphs are from the same document
                    if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                        continue
                    
                    # Calculate Jaccard similarity
                    try:
                        text1 = corpus[i].lower()
                        text2 = corpus[j].lower()
                        
                        # Skip empty texts
                        if not text1 or not text2:
                            continue
                        
                        # Tokenize into words
                        words1 = set(text1.split())
                        words2 = set(text2.split())
                        
                        # Calculate Jaccard similarity
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        
                        # Avoid division by zero
                        if not union:
                            continue
                        
                        jaccard_similarity = len(intersection) / len(union)
                        
                        # Calculate text similarity using our improved method
                        text_similarity = self._calculate_text_similarity(
                            paragraphs[i]['content'],
                            paragraphs[j]['content']
                        )
                        
                        # Check if similarity is above threshold
                        if jaccard_similarity >= threshold:
                            result = SimilarityResult(
                                paragraph1_id=paragraphs[i]['id'],
                                paragraph2_id=paragraphs[j]['id'],
                                paragraph1_content=paragraphs[i]['content'],
                                paragraph2_content=paragraphs[j]['content'],
                                paragraph1_doc_id=paragraphs[i]['doc_id'],
                                paragraph2_doc_id=paragraphs[j]['doc_id'],
                                content_similarity_score=jaccard_similarity,
                                text_similarity_score=text_similarity,
                                similarity_type='similar'
                            )
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error calculating Jaccard similarity for pair {i}, {j}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error calculating Jaccard similarity: {str(e)}", exc_info=True)
        
        return results
    
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