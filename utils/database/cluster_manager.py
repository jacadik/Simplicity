from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy.orm import Session

from .models import Cluster, Paragraph, Document, cluster_paragraphs
from .base_manager import BaseManager

class ClusterManager:
    """
    Manages cluster-related operations in the database.
    """
    def __init__(self, base_manager: BaseManager):
        """
        Initialize the cluster manager.
        
        Args:
            base_manager: Base manager instance for database operations
        """
        self.base_manager = base_manager
        self.logger = base_manager.logger
    
    def create_cluster(self, name: str, description: str, similarity_threshold: float, 
                       similarity_type: str = 'content') -> int:
        """
        Create a new cluster and return its ID.
        
        Args:
            name: Cluster name
            description: Cluster description
            similarity_threshold: Threshold used for similarity clustering
            similarity_type: Type of similarity measure used ('content' or 'text')
            
        Returns:
            Cluster ID if successful, -1 otherwise
        """
        self.logger.info(f"Creating new cluster: {name}")
        
        def db_create_cluster(session: Session) -> int:
            # Create new cluster
            cluster = Cluster(
                name=name,
                description=description,
                creation_date=datetime.now().isoformat(),
                similarity_threshold=similarity_threshold,
                similarity_type=similarity_type
            )
            session.add(cluster)
            session.commit()
            
            cluster_id = cluster.id
            self.logger.info(f"Created cluster with ID: {cluster_id}")
            return cluster_id
        
        try:
            return self.base_manager._with_session(db_create_cluster)
        except Exception as e:
            self.logger.error(f"Error creating cluster: {str(e)}")
            return -1
    
    def add_paragraphs_to_cluster(self, cluster_id: int, paragraph_ids: List[int]) -> bool:
        """
        Add paragraphs to a cluster.
        
        Args:
            cluster_id: ID of the cluster
            paragraph_ids: List of paragraph IDs to add to the cluster
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Adding {len(paragraph_ids)} paragraphs to cluster {cluster_id}")
        
        def db_add_to_cluster(session: Session) -> bool:
            # Get cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return False
            
            # Get paragraphs
            paragraphs = session.query(Paragraph).filter(Paragraph.id.in_(paragraph_ids)).all()
            
            # Add paragraphs to cluster
            for paragraph in paragraphs:
                if paragraph not in cluster.paragraphs:
                    cluster.paragraphs.append(paragraph)
            
            session.commit()
            
            self.logger.info(f"Added {len(paragraphs)} paragraphs to cluster {cluster_id}")
            return True
        
        try:
            return self.base_manager._with_session(db_add_to_cluster)
        except Exception as e:
            self.logger.error(f"Error adding paragraphs to cluster: {str(e)}")
            return False
    
    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters.
        
        Returns:
            List of cluster dictionaries
        """
        self.logger.info("Retrieving all clusters")
        
        def db_get_clusters(session: Session) -> List[Dict[str, Any]]:
            # Query all clusters
            clusters = session.query(Cluster).all()
            
            # Convert ORM objects to dictionaries
            cluster_dicts = []
            for cluster in clusters:
                cluster_dicts.append({
                    'id': cluster.id,
                    'name': cluster.name,
                    'description': cluster.description,
                    'creation_date': cluster.creation_date,
                    'similarity_threshold': cluster.similarity_threshold,
                    'similarity_type': cluster.similarity_type,
                    'paragraph_count': len(cluster.paragraphs)
                })
            
            self.logger.info(f"Retrieved {len(cluster_dicts)} clusters")
            return cluster_dicts
        
        try:
            return self.base_manager._with_session(db_get_clusters)
        except Exception as e:
            self.logger.error(f"Error retrieving clusters: {str(e)}")
            return []
    
    def get_cluster_paragraphs(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get paragraphs in a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of paragraph dictionaries in the cluster
        """
        self.logger.info(f"Retrieving paragraphs for cluster {cluster_id}")
        
        def db_get_cluster_paragraphs(session: Session) -> List[Dict[str, Any]]:
            # Get cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return []
            
            # Get paragraphs in cluster with document info
            paragraphs = []
            for paragraph in cluster.paragraphs:
                document = session.query(Document).get(paragraph.document_id)
                
                # Get tags for this paragraph
                tags = [{
                    'id': tag.id,
                    'name': tag.name,
                    'color': tag.color
                } for tag in paragraph.tags]
                
                paragraphs.append({
                    'id': paragraph.id,
                    'content': paragraph.content,
                    'document_id': paragraph.document_id,
                    'paragraph_type': paragraph.paragraph_type,
                    'position': paragraph.position,
                    'header_content': paragraph.header_content,
                    'filename': document.filename if document else 'Unknown',
                    'tags': tags
                })
            
            self.logger.info(f"Retrieved {len(paragraphs)} paragraphs for cluster {cluster_id}")
            return paragraphs
        
        try:
            return self.base_manager._with_session(db_get_cluster_paragraphs)
        except Exception as e:
            self.logger.error(f"Error retrieving cluster paragraphs: {str(e)}")
            return []
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """
        Delete a cluster.
        
        Args:
            cluster_id: ID of the cluster to delete
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Deleting cluster with ID: {cluster_id}")
        
        def db_delete_cluster(session: Session) -> bool:
            # Get the cluster
            cluster = session.query(Cluster).get(cluster_id)
            
            if not cluster:
                self.logger.warning(f"Cluster with ID {cluster_id} not found")
                return False
            
            # Delete the cluster (cascade will handle associations)
            session.delete(cluster)
            session.commit()
            
            self.logger.info(f"Deleted cluster with ID: {cluster_id}")
            return True
        
        try:
            return self.base_manager._with_session(db_delete_cluster)
        except Exception as e:
            self.logger.error(f"Error deleting cluster: {str(e)}")
            return False

    def clear_all_clusters(self) -> bool:
        """
        Delete all clusters in the database.
        
        Returns:
            Boolean indicating success
        """
        self.logger.info("Clearing all clusters")
        
        def db_clear_clusters(session: Session) -> int:
            # Clear cluster_paragraphs associations
            session.execute(cluster_paragraphs.delete())
            
            # Delete all clusters
            count = session.query(Cluster).delete()
            session.commit()
            return count
        
        try:
            count = self.base_manager._with_session(db_clear_clusters)
            self.logger.info(f"Deleted {count} clusters")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing clusters: {str(e)}")
            return False
