"""
Implementation of Clio: Privacy-Preserving Insights into Real-World AI Use
Based on the paper by Tamkin et al.

Main components:
1. Facet Extraction
2. Semantic Clustering 
3. Cluster Labeling
4. Hierarchy Building
5. Privacy Protection
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from itertools import chain
from random import sample, shuffle
from typing import Literal

import numpy as np
from tqdm.asyncio import tqdm
from sklearn.cluster import KMeans
from llm import embed, llm, split_prompt_into_conversation
from models import Cluster, Conversation, ConversationTurn, MetadataDict
from prompts import (
    CLUSTER_LABELING_PROMPT,
    CONCERNING_SCORE_PROMPT,
    EXTRACTOR_PROMPT_TEMPLATE,
    HIERARCHY_ASSIGNMENT_PROMPT,
    HIERARCHY_DEDUPLICATION_PROMPT,
    HIERARCHY_PROPOSAL_PROMPT,
    HIERARCHY_RENAME_PROMPT,
    PRIVACY_AUDITOR_PROMPT,
    TASK_FACET_CRITERIA,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FacetType = Literal["direct", "model"]

@dataclass
class FacetConfig:
    """Configuration for a facet extraction"""
    name: str
    type: FacetType
    prompt: str | None = None
    prefill: str = ""
    
ids = [0, 0, 0, 0, 0]
class FacetExtractor:
    """Enhanced facet extractor with privacy protection and multilingual support"""
    
    def __init__(self):
        """Initialize extractor with Claude API client"""

        # Configure available facets
        self.facets = {
            "task": FacetConfig(
                name="task",
                type="model",
                prompt="What task is the model being asked to perform in this conversation?",
                prefill="The task is to"
            ),
            "concerning_score": FacetConfig(
                name="concerning_score", 
                type="model",
                prompt=CONCERNING_SCORE_PROMPT
            ),
            "num_turns": FacetConfig(
                name="num_turns",
                type="direct"
            ),
            "request": FacetConfig(
                name="request",
                type="model",
                prompt="What is the user's overall request for the assistant?",
                prefill="The user's overall request for the assistant is to"
            )
        }

    async def _extract_model_facet(self, facet: FacetConfig, conv: Conversation) -> str | float:
        """Extract facets that require model inference"""
        # Build system prompt from template and format messages for the LLM
        prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
            conversation=conv.content,
            question=facet.prompt,
            prefill=facet.prefill
        )
        
        # Get model response
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, model='bedrock.anthropic.claude3-haiku')

        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            raise ValueError("Could not extract answer from response")
        answer = match.group(1).strip()
        
        # Post-process based on facet type
        if facet.name == "concerning_score":
            return float(answer.split('\n')[0])
        
        return answer

    async def extract_facets(self, conv: Conversation, 
        requested_facets: list[str] | None = None) -> Conversation:
        """Extract all requested facets from a conversation
        
        Args:
            conv: Conversation to analyze
            requested_facets: list of facet names to extract, or None for all
            
        Returns:
            Dictionary of facet names to extracted values
        """
        results: MetadataDict = {}
        
        # Determine which facets to extract
        facets_to_extract = (
            [self.facets[f] for f in requested_facets]
            if requested_facets
            else self.facets.values()
        )
        
        # Extract each facet
        for j, facet in enumerate(facets_to_extract):
            ids[j] += 1
            try:
                if facet.type == "direct":

                    if facet.name == "num_turns":
                        value = conv.num_turns
                    else:
                        raise ValueError(f"Unknown direct facet: {facet.name}")

                elif facet.type == 'model':
                    value = await self._extract_model_facet(facet, conv)

                results[facet.name] = value
            except Exception as e:
                logger.error(f"Failed to extract {facet.name}: {e}")
                results[facet.name] = None
                raise e
        ids[4] += 1
        
        conv.metadata = results
        return conv


class ClioSystem:
    """Main Clio system implementation"""
    
    def __init__(self, 
        min_cluster_size: int = 50,
        privacy_threshold: float = 0.8):
        """
        Args:
            min_cluster_size: Minimum number of conversations per cluster
            privacy_threshold: Threshold for privacy screening
        """
        self.min_cluster_size = min_cluster_size
        self.privacy_threshold = privacy_threshold
        self.facet_extractor = FacetExtractor()
        
    async def extract_facets(self, conversations: list[Conversation]) -> list[Conversation]:
        """Extract key facets from a conversation
        
        Args: conversation: Input conversation
        Returns: Dictionary of extracted facets
        """
        return await tqdm.gather(*[
            self.facet_extractor.extract_facets(c) for c in conversations
        ])
        
    async def embed_conversations(self, conversations: list[Conversation]) -> np.ndarray:
        """Generate embeddings for a list of conversations by combining facet embeddings
        
        Args: conversations: list of conversations to embed
        Returns: Array of combined embeddings
        """
        # Get text facets that need embedding
        requests = [c.metadata['request'] for c in conversations]
        tasks = [c.metadata['task'] for c in conversations]
        
        # Embed text facets
        task_embeddings = await embed(tasks)        # Shape: (n_convs, n_embed)

        # Store embeddings in conversation metadata
        for conv, emb in zip(conversations, task_embeddings):
            conv.metadata['embedding'] = emb
        return task_embeddings
        request_embeddings = await embed(requests)  # Shape: (n_convs, n_embed)
        
        # Get numeric facets
        concerning_scores = np.array([
            [float(c.metadata['concerning_score'])] for c in conversations
        ])  # Shape: (n_convs, 1)
        
        num_turns = np.array([
            [float(c.metadata['num_turns'])] for c in conversations
        ])  # Shape: (n_convs, 1)
        
        # Concatenate all features
        combined_embeddings = np.hstack([
            request_embeddings,  # n_embed dims
            task_embeddings,    # n_embed dims
            concerning_scores,  # 1 dim
            num_turns,         # 1 dim
        ])  # Final shape: (n_convs, 1538)
        
        # Normalize the combined embeddings
        norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        normalized_embeddings = combined_embeddings / norms
        
        logger.info(f"Generated embeddings of shape {normalized_embeddings.shape}")
        
        return normalized_embeddings
    
    async def cluster_conversations(self, 
        embeddings: np.ndarray,
        conversations: list[Conversation],
        n_clusters: int | None = None) -> list[Cluster]:
        """Cluster conversations based on embeddings
        
        Args:
            embeddings: Conversation embeddings
            conversations: list of conversations to embed
            n_clusters: Number of clusters to generate. If None, will be set based on dataset size
            
        Returns:
            list of conversation clusters
        """
        # Determine number of clusters - use sqrt scaling heuristic
        n_clusters = n_clusters or max(5, int(np.sqrt(len(embeddings))))
        logger.info(f"Clustering {len(embeddings)} conversations into {n_clusters} clusters")
        
        # Get cluster assignments
        labels = KMeans(n_clusters=n_clusters).fit_predict(embeddings)
        
        # Group conversations by cluster label
        valid_clusters = []
        for i in range(n_clusters):
            idxs = np.where(labels == i)[0]
            convs = [conversations[j] for j in idxs]
            if len(convs) >= self.min_cluster_size:
                valid_clusters.append((convs, idxs))
        
        # Generate all cluster descriptions in parallel
        descriptions = await asyncio.gather(*[
            self._generate_cluster_description(convs, conversations, embeddings, idxs)
            for convs, idxs in valid_clusters
        ])
        
        # Create final cluster objects
        return [
            Cluster(
                id=f"cluster_{i}",
                conversations=convs,
                name=name,
                description=desc
            )
            for i, ((name, desc), (convs, _)) in enumerate(zip(descriptions, valid_clusters))
        ]
    
    async def _propose_neighborhood_clusters(self, 
        neighborhood: list[Cluster],
        all_clusters: list[Cluster],
        embeddings: dict,
        desired_names: int,
        m: int = 10) -> list[str]:
        """Propose new clusters for a neighborhood using nearby clusters for context
        
        Args:
            neighborhood: Clusters in the neighborhood
            all_clusters: All clusters at this level
            embeddings:mapping cluster IDs to their embeddings
            desired_names: Target number of higher-level clusters to propose
            m: Number of nearest clusters to include for context
            
        Returns:
            list of (name, description) tuples for proposed clusters
        """
        # Get embeddings for neighborhood clusters
        neighborhood_embeddings = np.array([embeddings[c.id] for c in neighborhood])
        other_clusters = [c for c in all_clusters if c not in neighborhood]
        
        # Format cluster information with descriptions
        cluster_list = [f"<cluster>{c.name}: {c.description}</cluster>" for c in neighborhood]
        
        # Only add context clusters if there are any
        if other_clusters:
            other_embeddings = np.array([embeddings[c.id] for c in other_clusters])
            
            # Find m nearest clusters
            centroid = np.mean(neighborhood_embeddings, axis=0)
            distances = np.linalg.norm(other_embeddings - centroid, axis=1)
            m = min(m, len(other_clusters))  # Don't try to get more neighbors than exist
            nearest_indices = np.argpartition(distances, m)[:m]
            context_clusters = [other_clusters[i] for i in nearest_indices]
            
            # Add context clusters to list
            cluster_list.extend([
                f"<cluster>{c.name}: {c.description}</cluster>"
                for c in context_clusters
            ])
        
        # Use HIERARCHY_PROPOSAL_PROMPT
        prompt = HIERARCHY_PROPOSAL_PROMPT.format(
            cluster_list="\n".join(cluster_list),
            desired_names=desired_names,
            min_names=int(0.5 * desired_names),
            max_names=int(1.5 * desired_names),
            facet_criteria=TASK_FACET_CRITERIA,  # TODO: Make this configurable per facet type
        )
        messages = split_prompt_into_conversation(prompt)
        
        # Get model response
        response = await llm(messages, temperature=1.0)
        
        # Extract proposed clusters from numbered list
        proposals = []
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            logger.warning("Failed to extract answer from proposal response")
            return []
        
        # Parse numbered list (e.g., "1. Debug Python code")
        for line in answer_match.group(1).strip().split('\n'):
            if line.strip():
                if match := re.match(r'\d+\.\s*(.*)', line.strip()):
                    name = match.group(1).strip()
                    proposals.append(name)
        
        if not proposals:
            logger.warning("No proposals found in response")
        
        return proposals
    
    async def _get_level_embeddings(self, clusters: list[Cluster]) -> dict[str, np.ndarray]:
        """Generate embeddings for cluster texts"""
        texts = [f"{c.name} {c.description}" for c in clusters]
        return {c.id: emb for c, emb in zip(clusters, await embed(texts))}
    
    def _create_neighborhoods(self, 
        clusters: list[Cluster],
        embeddings: dict[str, np.ndarray],
        target_size: int) -> list[list[Cluster]]:
        """Split clusters into neighborhoods using k-means"""
        if len(clusters) <= target_size:
            return [clusters]
            
        n_neighborhoods = max(1, len(clusters) // target_size)
        kmeans = KMeans(n_clusters=n_neighborhoods)
        labels = kmeans.fit_predict([embeddings[c.id] for c in clusters])
        
        neighborhoods = []
        for i in range(n_neighborhoods):
            hood = [c for j, c in enumerate(clusters) if labels[j] == i]
            if hood:
                neighborhoods.append(hood)
        return neighborhoods
    
    async def _generate_proposals(self,
        neighborhoods: list[list[Cluster]],
        all_clusters: list[Cluster],
        embeddings: dict[str, np.ndarray],
        target_n: int) -> list[str]:
        """Generate and deduplicate cluster proposals"""
        # Get proposals from each neighborhood in parallel
        proposal_tasks = [
            self._propose_neighborhood_clusters(
                hood, all_clusters, embeddings, target_n
            )
            for hood in neighborhoods
        ]
        proposals = list(chain.from_iterable(
            await asyncio.gather(*proposal_tasks)
        ))
        
        # Deduplicate proposals
        return await self._deduplicate_clusters(proposals, target_n)

    async def build_hierarchy(self, 
        base_clusters: list[Cluster],
        max_levels: int = 3,
        target_neighborhood_size: int = 40,
        target_top_clusters: int = 10) -> list[Cluster]:
        """Build hierarchical organization of clusters"""
        if not base_clusters:
            return []
        
        # Calculate target number of clusters for each level
        n_base = len(base_clusters)
        level_ratios = (target_top_clusters / n_base) ** (1 / (max_levels - 1))
        target_counts = [int(n_base * (level_ratios ** i)) for i in range(max_levels)]
        logger.info(f"Target clusters per level: {target_counts}")
        
        current_level = base_clusters
        for level in range(max_levels - 1):
            target_n = target_counts[level + 1]
            
            # Create embeddings for current level clusters
            embeddings = await self._get_level_embeddings(current_level)
            
            # Split into neighborhoods if needed
            neighborhoods = self._create_neighborhoods(current_level, embeddings, target_neighborhood_size)
            logger.info(f"Level {level}: Created {len(neighborhoods)} neighborhoods")

            # Generate and deduplicate proposed cluster names across neighborhoods
            proposals = await self._generate_proposals(neighborhoods, current_level, embeddings, target_n)
            if not proposals:
                logger.warning(f"No new clusters generated at level {level}")

            # Assign each cluster to a parent in parallel
            parent_assignments = await asyncio.gather(*[
                self._assign_parent_cluster(cluster, proposals)
                for cluster in current_level
            ])
            
            # Build assignments dictionary
            assignments = {}
            for cluster, parent_name in zip(current_level, parent_assignments):
                if parent_name not in assignments:
                    assignments[parent_name] = []
                assignments[parent_name].append(cluster)
            
            # Create parent clusters and update relationships in parallel
            parents = []
            for i, (name, group) in enumerate(assignments.items()):
                parent = Cluster(
                    id=f"level_{level}_cluster_{i}",
                    level=level,
                    conversations=[],
                    name=name,
                    description='',
                    children=group
                )
                # Update children
                for child in group:
                    child.parent = parent
                parents.append(parent)
            
            # Rename parents in parallel
            renamed = await asyncio.gather(*[
                self._rename_parent_cluster(p) for p in parents
            ])
            
            # Update parent names and descriptions
            for parent, (name, desc) in zip(parents, renamed):
                parent.name = name
                parent.description = desc
            
            current_level = parents
            logger.info(f"Level {level + 1}: Created {len(current_level)} clusters "
                       f"(target: {target_n}±{1.5*target_n:.1f})")
            
            # if len(current_level) <= target_neighborhood_size:
            #     break
        
        return current_level
    
    async def _generate_cluster_description(self, 
        cluster_convos: list[Conversation],
        all_convos: list[Conversation],
        embeddings: np.ndarray,
        cluster_indices: np.ndarray) -> tuple[str, str]:
        """Generate name and description for a cluster
        
        Args:
            cluster_convos: Conversations in the cluster
            all_convos: All conversations (needed for finding nearest neighbors)
            embeddings: Embeddings for all conversations
            cluster_indices: Indices of conversations in this cluster
            
        Returns:
            Tuple of (name, description)
        """
        # Get 50 random samples from cluster
        sample_convos = sample(cluster_convos, 50) if len(cluster_convos) > 50 else cluster_convos
        
        # Get cluster centroid
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distances to centroid for non-cluster conversations
        non_cluster_indices = np.array([i for i in range(len(all_convos)) if i not in cluster_indices])
        non_cluster_embeddings = embeddings[non_cluster_indices]
        distances = np.linalg.norm(non_cluster_embeddings - centroid, axis=1)
        
        # Get 50 nearest neighbors not in cluster
        k = min(50, len(non_cluster_indices))
        nearest_indices = np.argpartition(distances, k)[:k]
        contrast_convos = [all_convos[non_cluster_indices[i]] for i in nearest_indices]
        
        # Format conversation summaries
        sample_summaries = [f"{conv.metadata['request']}" for conv in sample_convos]
        contrast_summaries = [f"{conv.metadata['request']}" for conv in contrast_convos]
        
        # Build prompt
        prompt = CLUSTER_LABELING_PROMPT.format(
            answers="\n".join(sample_summaries),
            contrastive_answers="\n".join(contrast_summaries),
            facet_criteria=TASK_FACET_CRITERIA
        )
        
        # Get Claude response with temperature=1
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, temperature=1.0)
        
        # Extract name and description
        name_match = re.search(r"<name>(.*?)</name>", response, re.DOTALL)
        summary_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        
        if not name_match or not summary_match:
            logger.warning("Failed to extract name or summary from response")
            return "Unknown Cluster", "No description available"
        
        return name_match.group(1).strip(), summary_match.group(1).strip()
    
    async def _generate_parent_description(self, clusters: list[Cluster]) -> tuple[str, str]:
        """Generate name and description for a parent cluster
        
        Args: clusters: Child clusters to summarize
        Returns: Tuple of (name, description) 
        """
        cluster_info = [f"{c.name}: {c.description}" for c in clusters]
        
        # Use the HIERARCHY_RENAME_PROMPT template
        prompt = HIERARCHY_RENAME_PROMPT.format(
            cluster_names="\n".join(cluster_info),
            facet_criteria=TASK_FACET_CRITERIA
        )
        
        # Get Claude response
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, temperature=1.0)
        
        # Extract name and description
        name_match = re.search(r"<name>(.*?)</name>", response, re.DOTALL)
        summary_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        
        if not name_match or not summary_match:
            logger.warning("Failed to extract name or summary from response")
            return "Unknown Group", "No description available"
        
        return name_match.group(1).strip(), summary_match.group(1).strip()

    async def _deduplicate_clusters(self, clusters: list[str], target: int) -> list[str]:
        """Deduplicate and refine clusters across neighborhoods
        
        Args: clusters: list of clusters to deduplicate
        Returns: list of deduplicated clusters
        """
        if len(clusters) <= 1:
            return clusters
                
        # Format cluster names
        cluster_names = [f"<cluster> {c} </cluster>" for c in clusters]
        
        # Use deduplication prompt
        prompt = HIERARCHY_DEDUPLICATION_PROMPT.format(
            cluster_names="\n".join(cluster_names),
            desired_names=target,
            min_names=int(0.5 * target),
            max_names=int(1.5 * target),
            facet_criteria=TASK_FACET_CRITERIA # This is only for task facet clustering
        )
        
        # Get Claude response
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, temperature=1.0)
                
        # Extract deduplicated cluster names
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            logger.warning("Failed to extract answer from deduplication response")
            return clusters
        
        # Parse numbered list of cluster names
        dedup_names = []
        for line in answer_match.group(1).strip().split('\n'):
            if line.strip():
                # Extract name after number (e.g., "1. Debug Python code" -> "Debug Python code")
                if match := re.match(r'\d+\.\s*(.*)', line.strip()):
                    dedup_names.append(match.group(1))
                
        if not dedup_names:
            logger.warning("No deduplicated names found in response")
            return clusters
        return dedup_names

    async def _assign_parent_cluster(self, child_cluster: Cluster, parent_clusters: list[str]) -> str:
        """Assign a cluster to its best-fitting parent cluster
        
        Args:
            child_cluster: Cluster to assign
            parent_clusters: list of potential parent clusters (will be shuffled)
            
        Returns:
            Name of chosen parent cluster
        """
        # Shuffle parent clusters to avoid order bias
        shuffled_parents = parent_clusters.copy()
        shuffle(shuffled_parents)
        
        # Format cluster information and fill prompt
        parent_info = [f"<cluster> {c} </cluster>" for c in shuffled_parents]        
        prompt = HIERARCHY_ASSIGNMENT_PROMPT.format(
            higher_level_clusters="\n".join(parent_info),
            cluster_name=child_cluster.name,
            cluster_description=child_cluster.description
        )
        
        # Get Claude response
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, temperature=1.0)
        
        # Extract chosen parent name
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            logger.warning("Failed to extract parent cluster assignment")
            return shuffled_parents[0]  # Default to first parent if extraction fails
        
        return answer_match.group(1).strip()

    async def _rename_parent_cluster(self, parent_cluster: Cluster) -> tuple[str, str]:
        """Generate new name and description for parent cluster based on its children
        
        Args: parent_cluster: Cluster to rename
        Returns: Tuple of (new_name, new_description)
        """
        # Format child cluster information and fill prompt
        child_info = [f"<cluster> {c.name} </cluster>" for c in parent_cluster.children]
        prompt = HIERARCHY_RENAME_PROMPT.format(
            cluster_names="\n".join(child_info),
            facet_criteria=TASK_FACET_CRITERIA  # This should be parameterized based on facet type
        )
        
        # Get Claude response
        messages = split_prompt_into_conversation(prompt)
        response = await llm(messages, temperature=1.0)
        
        # Extract new name and description
        name_match = re.search(r"<name>(.*?)</name>", response, re.DOTALL)
        summary_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        
        if not name_match or not summary_match:
            logger.warning("Failed to extract new name or summary")
            return parent_cluster.name, parent_cluster.description
        
        return name_match.group(1).strip(), summary_match.group(1).strip()
    
    def verify_privacy(self, cluster: Cluster) -> bool:
        """Verify cluster meets privacy requirements
        
        Args: cluster: Cluster to check
        Returns: True if cluster passes privacy checks
        """
        # Check minimum size
        if len(cluster.conversations) < self.min_cluster_size:
            return False
            
        # Check unique users
        unique_users = len(set(c.metadata['user_id'] for c in cluster.conversations))
        if unique_users < self.min_cluster_size / 2:
            return False
            
        # Use PRIVACY_AUDITOR_PROMPT
        prompt = PRIVACY_AUDITOR_PROMPT.format(
            **{
                "cluster name": cluster.name,
                "cluster description": cluster.description
            }
        )
        
        # In practice, would call Claude API here
        privacy_score = 4
        return privacy_score >= self.privacy_threshold
        
    async def process_conversations(self,
        conversations: list[Conversation],
        n_clusters: int | None = None) -> list[Cluster]:
        """Process a batch of conversations through the full Clio pipeline"""
        logger.info(f"Processing {len(conversations)} conversations")
        
        conversations = await self.extract_facets(conversations) # Extract tasks, request, ...
        embeddings = await self.embed_conversations(conversations) # Generate embeddings
        base_clusters = await self.cluster_conversations(embeddings, conversations, n_clusters) # Create base clusters
        top_clusters = await self.build_hierarchy(base_clusters) # Build hierarchy
        return top_clusters

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        clio = ClioSystem()
        
        # Load sample conversations
        conversations = [
            Conversation(
                id="1", 
                turns=[
                    ConversationTurn(role="user", content="How do I fix this bug in my Python code?"),
                    ConversationTurn(role="assistant", content="I can help with that...")
                ]
            ),
            Conversation(
                id="2", 
                turns=[
                    ConversationTurn(role="user", content="Getting a TypeError in my Django app"),
                    ConversationTurn(role="assistant", content="Let's debug that...")
                ]
            ),
            # ... more conversations
        ]
        
        # Process conversations
        clusters = await clio.process_conversations(conversations)
        
        # Print results
        for cluster in clusters:
            print(f"\nCluster: {cluster.name}")
            print(f"Description: {cluster.description}")
            print(f"Size: {len(cluster.conversations)}")
            if cluster.children:
                print(f"Child clusters: {len(cluster.children)}")
    
    # Run the async main function
    asyncio.run(main())