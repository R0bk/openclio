"""
Module for generating UMAP projections of conversations and clusters
for visualization purposes.
"""

import numpy as np
import umap
import logging
from typing import Any
from models import Conversation, Cluster, Projection

logger = logging.getLogger(__name__)

class Projector:
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        """Initialize the UMAP projector with configurable parameters"""
        self.umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        self.projection_params = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric
        }

    def _scale_coordinates(self, projection: np.ndarray, scale: float = 1000.0) -> np.ndarray:
        """Scale projection coordinates to visualization bounds"""
        x_min, x_max = projection[:, 0].min(), projection[:, 0].max()
        y_min, y_max = projection[:, 1].min(), projection[:, 1].max()
        
        scaled = np.zeros_like(projection)
        scaled[:, 0] = scale * (projection[:, 0] - x_min) / (x_max - x_min)
        scaled[:, 1] = scale * (projection[:, 1] - y_min) / (y_max - y_min)
        
        return scaled

    def _calculate_cluster_position(self, cluster: Cluster, conv_positions: dict[str, dict[str, float]]) -> Projection:
        """Calculate cluster position as mean of its conversations' positions"""
        positions = []
        
        # If cluster has direct conversations
        if cluster.conversations:
            for conv in cluster.conversations:
                if conv.id in conv_positions:
                    positions.append([conv_positions[conv.id]['x'], conv_positions[conv.id]['y']])
        
        # If cluster has children, include their positions too
        if cluster.children:
            for child in cluster.children:
                child_pos = self._calculate_cluster_position(child, conv_positions)
                positions.append([child_pos['x'], child_pos['y']])
                
        if not positions:
            logger.warning(f"No positions found for cluster {cluster.id}")
            return {'x': 0, 'y': 0}
            
        positions = np.array(positions)
        return {
            'x': float(np.mean(positions[:, 0])),
            'y': float(np.mean(positions[:, 1]))
        }

    def project(self, conversations: list[Conversation], clusters: list[Cluster]) -> dict[str, Any]:
        """
        Generate UMAP projections for conversations and clusters
        
        Args:
            conversations: List of Conversation objects
            clusters: List of top-level Cluster objects
            
        Returns:
            Dictionary containing projection data and metadata
        """
        try:
            # Get embeddings and conversation IDs
            embeddings = np.array([conv.metadata['embedding'] for conv in conversations])
            
            # Generate UMAP projection
            projection = self.umap_reducer.fit_transform(embeddings)
            
            # Scale coordinates
            scaled_projection = self._scale_coordinates(projection)
            
            # Create conversation position lookup
            for i, conv in enumerate(conversations):
                conv.metadata["projection"] = {
                    'x': float(scaled_projection[i, 0]),
                    'y': float(scaled_projection[i, 1])
                }
            conv_positions = {conv.id: conv.metadata["projection"] for conv in conversations}

            # Calculate cluster positions recursively
            def process_cluster(cluster: Cluster):
                """Recursively process cluster and its children"""
                cluster.projection = self._calculate_cluster_position(cluster, conv_positions)
                
                for child in cluster.children:
                    process_cluster(child)
            
            # Process all clusters
            for cluster in clusters:
                process_cluster(cluster)
            
            return {
                    'projection_parameters': self.projection_params,
                    'visualization_bounds': {
                        'min_x': 0,
                        'max_x': 1000,
                        'min_y': 0,
                        'max_y': 1000
                    }
                }
                
            
        except Exception as e:
            logger.error(f"Projection generation failed: {e}")
            raise 