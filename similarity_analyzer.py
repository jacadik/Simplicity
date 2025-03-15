import logging
import re
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime


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
        self.threshold = float(threshold)  # Ensure threshold is a float
        self.logger = self._setup_logger(logging_level)
    
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
                    continue
                
                # Normalize the content for comparison
                normalized = self._normalize_text(content)
                if not normalized:
                    continue
                
                if normalized not in content_dict:
                    content_dict[normalized] = []
                content_dict[normalized].append(para)
            
            # Find groups with more than one paragraph
            for normalized, para_group in content_dict.items():
                if len(para_group) > 1:
                    self.logger.debug(f"Found group with {len(para_group)} matching paragraphs")
                    # Create similarity results for all pairs in group
                    for i in range(len(para_group)):
                        for j in range(i+1, len(para_group)):
                            # Skip if both paragraphs are from the same document
                            if para_group[i].get('doc_id') == para_group[j].get('doc_id'):
                                continue
                            
                            # Ensure all required fields are present
                            if not all(k in para_group[i] for k in ['id', 'content', 'doc_id']):
                                self.logger.warning(f"Missing required fields in paragraph: {para_group[i]}")
                                continue
                                
                            if not all(k in para_group[j] for k in ['id', 'content', 'doc_id']):
                                self.logger.warning(f"Missing required fields in paragraph: {para_group[j]}")
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
            
            # First check for exact matches - more efficient and reliable
            exact_matches = []
            content_dict = {}
            
            for i, para in enumerate(valid_paragraphs):
                # Normalize the content
                normalized = self._normalize_text(para['content'])
                if normalized in content_dict:
                    for prev_idx in content_dict[normalized]:
                        # Skip if same document
                        if valid_paragraphs[prev_idx]['doc_id'] == para['doc_id']:
                            continue
                            
                        exact_matches.append(SimilarityResult(
                            paragraph1_id=valid_paragraphs[prev_idx]['id'],
                            paragraph2_id=para['id'],
                            paragraph1_content=valid_paragraphs[prev_idx]['content'],
                            paragraph2_content=para['content'],
                            paragraph1_doc_id=valid_paragraphs[prev_idx]['doc_id'],
                            paragraph2_doc_id=para['doc_id'],
                            similarity_score=1.0,
                            similarity_type='exact'
                        ))
                    content_dict[normalized].append(i)
                else:
                    content_dict[normalized] = [i]
            
            # Add exact matches to results
            results.extend(exact_matches)
            self.logger.debug(f"Found {len(exact_matches)} exact matches")
                
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
            exact_match_pairs = {self._make_pair_key(r.paragraph1_id, r.paragraph2_id) for r in exact_matches}
            
            for result in similar_results:
                pair_key = self._make_pair_key(result.paragraph1_id, result.paragraph2_id)
                if pair_key not in exact_match_pairs:
                    result.similarity_type = 'similar'  # Ensure type is set to 'similar'
                    results.append(result)
            
            self.logger.info(f"Found total of {len(results)} similarity results")
            
        except Exception as e:
            self.logger.error(f"Error finding similar paragraphs: {str(e)}", exc_info=True)
        
        return results
    
    def cluster_paragraphs(self, similarity_results, threshold=None):
        """
        Cluster paragraphs using graph-based community detection.
        
        Args:
            similarity_results: List of similarity results
            threshold: Minimum similarity score to consider paragraphs related
            
        Returns:
            List of clusters, where each cluster is a dict with name and paragraph IDs
        """
        threshold = float(threshold) if threshold is not None else self.threshold
        self.logger.info(f"Clustering paragraphs with threshold {threshold}")
        
        try:
            # Create a graph
            G = nx.Graph()
            
            # Add nodes and edges for similarities above threshold
            for result in similarity_results:
                if result.similarity_score >= threshold:
                    # Add nodes with paragraph content as attribute
                    G.add_node(result.paragraph1_id, content=result.paragraph1_content)
                    G.add_node(result.paragraph2_id, content=result.paragraph2_content)
                    
                    # Add edge with similarity score as weight
                    G.add_edge(
                        result.paragraph1_id, 
                        result.paragraph2_id, 
                        weight=result.similarity_score
                    )
            
            # Skip if graph is empty
            if len(G.nodes) == 0:
                self.logger.warning("No paragraphs to cluster")
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
                    'description': f"Auto-generated cluster with {len(community_list)} paragraphs",
                    'creation_date': datetime.now().isoformat(),
                    'similarity_threshold': threshold,
                    'paragraph_ids': community_list
                })
            
            self.logger.info(f"Created {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering paragraphs: {str(e)}", exc_info=True)
            return []
    
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
                    
                    # Get similarity value as float
                    similarity = float(cosine_sim[i, j])
                    
                    # Check if value is valid
                    if np.isnan(similarity) or np.isinf(similarity):
                        self.logger.warning(f"Invalid similarity value: {similarity}")
                        continue
                    
                    # Count pairs above threshold for debugging
                    if similarity >= threshold:
                        pairs_above_threshold += 1
                        
                        result = SimilarityResult(
                            paragraph1_id=paragraphs[i]['id'],
                            paragraph2_id=paragraphs[j]['id'],
                            paragraph1_content=paragraphs[i]['content'],
                            paragraph2_content=paragraphs[j]['content'],
                            paragraph1_doc_id=paragraphs[i]['doc_id'],
                            paragraph2_doc_id=paragraphs[j]['doc_id'],
                            similarity_score=similarity,
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
                            
                        similarity = float(cosine_sim[i, j])
                        if not np.isnan(similarity) and not np.isinf(similarity):
                            top_pairs.append((i, j, similarity))
                
                # Sort by similarity descending and take top 10
                top_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, j, similarity in top_pairs[:10]:
                    result = SimilarityResult(
                        paragraph1_id=paragraphs[i]['id'],
                        paragraph2_id=paragraphs[j]['id'],
                        paragraph1_content=paragraphs[i]['content'],
                        paragraph2_content=paragraphs[j]['content'],
                        paragraph1_doc_id=paragraphs[i]['doc_id'],
                        paragraph2_doc_id=paragraphs[j]['doc_id'],
                        similarity_score=similarity,
                        similarity_type='similar'
                    )
                    results.append(result)
                
                self.logger.info(f"Added {len(results)} top similarity pairs")
        
        except Exception as e:
            self.logger.error(f"Error calculating TF-IDF similarity: {str(e)}", exc_info=True)
        
        return results
    
    def _calculate_jaccard_similarity(self, corpus: List[str], paragraphs: List[Dict], threshold: float) -> List[SimilarityResult]:
        """Calculate similarity using Jaccard similarity."""
        results = []
        
        try:
            # Calculate word sets for each paragraph
            word_sets = []
            for text in corpus:
                if not text:
                    word_sets.append(set())
                else:
                    words = text.split()
                    # Only use words with 3+ characters to reduce noise
                    word_sets.append(set(w for w in words if len(w) >= 3))
            
            # Calculate Jaccard similarity for each pair
            pairs_above_threshold = 0
            total_pairs = 0
            
            for i in range(len(paragraphs)):
                for j in range(i+1, len(paragraphs)):
                    total_pairs += 1
                    
                    # Skip if both paragraphs are from the same document
                    if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                        continue
                    
                    # Skip if either word set is empty or out of range
                    if i >= len(word_sets) or j >= len(word_sets):
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
                        pairs_above_threshold += 1
                        
                        result = SimilarityResult(
                            paragraph1_id=paragraphs[i]['id'],
                            paragraph2_id=paragraphs[j]['id'],
                            paragraph1_content=paragraphs[i]['content'],
                            paragraph2_content=paragraphs[j]['content'],
                            paragraph1_doc_id=paragraphs[i]['doc_id'],
                            paragraph2_doc_id=paragraphs[j]['doc_id'],
                            similarity_score=similarity,
                            similarity_type='similar'
                        )
                        results.append(result)
            
            self.logger.debug(f"Jaccard: Found {pairs_above_threshold} pairs above threshold out of {total_pairs} total pairs")
            
            # Special case similar to TF-IDF method
            if threshold < 0.3 and len(results) == 0 and total_pairs > 0:
                self.logger.info("Low threshold but no Jaccard results, getting top 10 instead")
                # Find highest similarity scores
                top_pairs = []
                for i in range(len(paragraphs)):
                    for j in range(i+1, len(paragraphs)):
                        # Skip same document
                        if paragraphs[i]['doc_id'] == paragraphs[j]['doc_id']:
                            continue
                            
                        # Skip invalid indices
                        if i >= len(word_sets) or j >= len(word_sets):
                            continue
                            
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
                            
                        top_pairs.append((i, j, similarity))
                
                # Sort by similarity descending and take top 10
                top_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, j, similarity in top_pairs[:10]:
                    result = SimilarityResult(
                        paragraph1_id=paragraphs[i]['id'],
                        paragraph2_id=paragraphs[j]['id'],
                        paragraph1_content=paragraphs[i]['content'],
                        paragraph2_content=paragraphs[j]['content'],
                        paragraph1_doc_id=paragraphs[i]['doc_id'],
                        paragraph2_doc_id=paragraphs[j]['doc_id'],
                        similarity_score=similarity,
                        similarity_type='similar'
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
        if text is None:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity comparison."""
        if text is None:
            return ""
            
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