import { useState } from 'react';
import styled from 'styled-components';
import { ClusterTreeProps, ClusterNodeProps } from '../types/ui';
import { Cluster } from '../types/models';

const TreeLayout = styled.div`
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 1px;
  height: calc(100vh - 60px); // Account for header
  background-color: #1e1e1e;
  min-width: 100vw;
`;

const Sidebar = styled.div`
  padding: 20px;
  overflow-y: auto;
  border-right: 1px solid #333;
`;

const MainPanel = styled.div`
  padding: 20px;
  overflow-y: auto;
`;

const ClusterDetails = styled.div`
  color: #ccc;
`;

const ClusterTitle = styled.h2`
  font-size: 1.1em;
  color: #eee;
  margin-bottom: 12px;
`;

const ClusterId = styled.div`
  color: #666;
  font-size: 0.9em;
  margin-bottom: 16px;
`;

const DetailSection = styled.div`
  margin-bottom: 24px;
`;

const DetailLabel = styled.div`
  color: #888;
  font-size: 0.9em;
  margin-bottom: 4px;
`;

const DetailValue = styled.div`
  color: #eee;
  line-height: 1.4;
`;

const DetailText = styled.div`
  color: #999;
  font-size: 0.9em;
  line-height: 1.6;
  margin-top: 12px;
`;

const StatsSection = styled.div`
  margin-top: 24px;
  display: grid;
  grid-template-columns: auto auto;
  gap: 8px;
  color: #999;
  font-size: 0.9em;
`;

const StatLabel = styled.div`
  color: #666;
`;

const StyledClusterNode = styled.div<{ level: number; selected: boolean; hasChildren: boolean }>`
  margin-left: ${props => props.level * 24}px;
  padding: 4px 12px;
  padding-left: ${props => props.hasChildren ? '12px' : '24px'};
  background-color: ${props => props.selected ? '#3d3522' : '#2a2a2a00'};
  margin-bottom: 1px;
  cursor: pointer;
  border-radius: 6px;
  
  &:hover {
    background-color: ${props => props.selected ? '#3d3522' : '#333'};
  }
`;

const ClusterHeader = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 4px;
`;

const ClusterContent = styled.div`
  flex: 1;
`;

const ClusterMainLine = styled.div`
  display: flex;
  align-items: baseline;
  gap: 8px;
`;

const ClusterName = styled.span<{ selected: boolean }>`
  color: ${props => props.selected ? '#fff' : '#eee'};
`;

const ClusterStats = styled.span`
  color: #888;
  font-size: 0.9em;
`;

const ParentCluster = styled.div`
  font-style: italic;
  color: #666;
  font-size: 0.9em;
  margin-bottom: 2px;
`;

const Description = styled.div`
  color: #888;
  font-size: 0.9em;
  margin-top: 2px;
  line-height: 1.4;
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  color: #666;
  padding: 0;
  margin-right: 4px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  width: 20px;
  text-align: center;
  line-height: 1;
  
  &:hover {
    color: #888;
  }
`;

const ClusterInfo = styled.div`
  flex: 1;
`;

const ParentInfo = styled.div`
  font-style: italic;
  color: #888;
  font-size: 0.9em;
  margin-bottom: 4px;
`;

const ClusterDescription = styled.div`
  color: #888;
  font-size: 0.9em;
  margin-top: 4px;
  line-height: 1.4;
`;

const ClusterNodeComponent: React.FC<ClusterNodeProps> = ({
    cluster,
    level,
    expandedNodes,
    toggleNode,
    selectedCluster,
    onSelect,
}) => {
    const hasChildren = cluster.child_clusters && cluster.child_clusters.length > 0;
    const isSelected = selectedCluster === cluster.id;
    const isExpanded = expandedNodes.has(cluster.id);
    
    // Sort child clusters by size
    const sortedChildren = cluster.child_clusters 
        ? [...cluster.child_clusters].sort((a, b) => b.size - a.size)
        : [];

    return (
        <>
            <StyledClusterNode 
                level={level} 
                selected={isSelected} 
                hasChildren={hasChildren}
                onClick={() => onSelect(cluster.id)}
            >
                {cluster.parent_name && (
                    <ParentCluster>Parent: {cluster.parent_name}</ParentCluster>
                )}
                <ClusterHeader>
                    {hasChildren && (
                        <ToggleButton onClick={(e) => {
                            e.stopPropagation();
                            toggleNode(cluster.id);  // Changed to pass cluster.id directly
                        }}>
                            {isExpanded ? '−' : '+'} 
                        </ToggleButton>
                    )}
                    <ClusterContent>
                        <ClusterMainLine>
                            <ClusterName selected={isSelected}>{cluster.name}</ClusterName>
                            <ClusterStats>
                                ({cluster.percentage?.toFixed(1)}%, {cluster.size.toLocaleString()} records
                                {hasChildren && `, ${cluster.child_clusters!.length} children`})
                            </ClusterStats>
                        </ClusterMainLine>
                    </ClusterContent>
                </ClusterHeader>
            </StyledClusterNode>
            
            {isExpanded && hasChildren && sortedChildren.map(child => (  // Use isExpanded here
                <ClusterNodeComponent
                    key={child.id}
                    cluster={child}
                    level={level + 1}
                    expandedNodes={expandedNodes}
                    toggleNode={toggleNode}
                    selectedCluster={selectedCluster}
                    onSelect={onSelect}
                />
            ))}
        </>
    );
};

export const ClusterTree: React.FC<ClusterTreeProps> = ({ data }) => {
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
    const [selectedCluster, setSelectedCluster] = useState<string>();
    
    const selectedClusterData = selectedCluster ? 
        findClusterById(data.hierarchy.clusters, selectedCluster) : 
        null;

    const toggleNode = (clusterId: string) => {
        setExpandedNodes(prev => {
            const newExpanded = new Set(prev);
            if (newExpanded.has(clusterId)) {
                newExpanded.delete(clusterId);
            } else {
                newExpanded.add(clusterId);
            }
            return newExpanded;
        });
    };
    
    // Sort top-level clusters by size
    const sortedClusters = [...data.hierarchy.clusters].sort((a, b) => b.size - a.size);

    return (
        <TreeLayout>
            <Sidebar>
                {selectedClusterData ? (
                    <ClusterDetails>
                        <ClusterTitle>{selectedClusterData.name}</ClusterTitle>
                        <ClusterId>#{selectedClusterData.id}</ClusterId>
                        
                        <DetailSection>
                            <DetailLabel>Description</DetailLabel>
                            <DetailValue>{selectedClusterData.description}</DetailValue>
                        </DetailSection>
                        
                        <DetailSection>
                            <DetailLabel>Statistics</DetailLabel>
                            <DetailValue>
                                {selectedClusterData.size.toLocaleString()} records
                                <br />
                                {selectedClusterData.percentage?.toFixed(1)}% of total
                                {selectedClusterData.child_clusters && (
                                    <>
                                        <br />
                                        {selectedClusterData.child_clusters.length} child clusters
                                    </>
                                )}
                            </DetailValue>
                        </DetailSection>
                        
                        {selectedClusterData.parent_name && (
                            <DetailSection>
                                <DetailLabel>Parent Cluster</DetailLabel>
                                <DetailValue>{selectedClusterData.parent_name}</DetailValue>
                            </DetailSection>
                        )}
                    </ClusterDetails>
                ) : (
                    <DetailValue>Select a cluster to view details</DetailValue>
                )}
            </Sidebar>

            <MainPanel>
                {sortedClusters.map(cluster => (
                    <ClusterNodeComponent
                        key={cluster.id}
                        cluster={cluster}
                        level={0}
                        expandedNodes={expandedNodes}
                        toggleNode={toggleNode}
                        selectedCluster={selectedCluster}
                        onSelect={setSelectedCluster}
                    />
                ))}
            </MainPanel>
        </TreeLayout>
    );
};

// Helper function to find a cluster by ID
function findClusterById(clusters: Cluster[], id: string): Cluster | null {
    for (const cluster of clusters) {
        if (cluster.id === id) return cluster;
        if (cluster.child_clusters) {
            const found = findClusterById(cluster.child_clusters, id);
            if (found) return found;
        }
    }
    return null;
}