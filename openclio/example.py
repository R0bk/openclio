"""
Example script demonstrating the usage of data_loader and openclio together
to analyze conversation data and save results.
"""

import asyncio
from pathlib import Path
import logging
from data_loader import load_conversation_data
import json
import random
from datetime import datetime
from json import JSONEncoder
from openclio import Cluster, ClioSystem
from projector import Projector
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(JSONEncoder):
    """Custom JSON encoder for datetime objects and numpy arrays"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def print_cluster(cluster: Cluster, prefix=""):
    """Print cluster info with minimal but clear formatting"""
    print(f"\n{prefix}{cluster.name}")
    print(f"{' '*len(prefix)}└─ {len(cluster.conversations)} convs: {cluster.description[:100]}...")
    
    if cluster.children:
        for i, child in enumerate(cluster.children, 1):
            print_cluster(child, prefix=f"{prefix}{i}.")

async def process_conversations(input_file: Path, output_file: Path):
    """Load conversations, extract facets, and save results
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output CSV file
    """
    # Load and validate conversation data
    logger.info(f"Loading conversations from {input_file}")
    conversations = load_conversation_data(input_file, ignore_errors=True)
    
    # Initialize Clio system
    clio = ClioSystem(min_cluster_size=10)

    conversation_slice = conversations[:20000] # TEMP
    top_clusters = await clio.process_conversations(conversation_slice, int(len(conversation_slice)**0.5))
        
    # Print summary statistics and hierarchy
    print("\nProcessing Summary:")
    print(f"Total conversations processed: {len(conversation_slice)}")
    
    if top_clusters:
        print(f"\nProcessed {len(conversation_slice)} conversations into {len(top_clusters)} clusters")
        
        # Print hierarchy
        for i, cluster in enumerate(top_clusters, 1):
            print_cluster(cluster, prefix=f"{i}.")
            
        # Show a few examples
        print("\nSample conversations:")
        for cluster in random.sample(top_clusters, min(3, len(top_clusters))):
            convs = [c for child in cluster.children for c in child.conversations]
            if convs:
                print(f"\n{cluster.name}:")
                for conv in random.sample(convs, min(2, len(convs))):
                    print(f"- {conv.metadata.get('request', 'Unknown')}")
    else:
        logger.warning("No clusters were created")

    proj_metadata = Projector().project(conversation_slice, top_clusters)
    
    # Generate analysis JSON
    analysis = {
        'metadata': {
            'total_conversations': len(conversation_slice),
            'analysis_date': datetime.now().isoformat(),
            'min_cluster_size': clio.min_cluster_size,
            'privacy_threshold': 0.8,
            **proj_metadata
        },
        'hierarchy': {
            'clusters': [cluster.to_dict() for cluster in top_clusters]
        },
        'conversations': {
            conv.id: conv.to_dict() for conv in conversation_slice
        }
    }
    
    # Save analysis JSON
    analysis_file = output_file.with_suffix('.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, cls=DateTimeEncoder)
    logger.info(f"Saved analysis to {analysis_file}")

async def main():
    input_file = Path("conversations.xlsx")
    output_file = Path("analysis_results.csv")
    
    await process_conversations(input_file, output_file)
        
if __name__ == "__main__":
    asyncio.run(main()) 