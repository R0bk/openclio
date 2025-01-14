import { useState } from 'react';
import styled from 'styled-components';
import { ClusterTreeProps, ClusterNodeProps } from '../types/ui';

const TreeLayout = styled.div`
  display: grid;
  gap: 1px;
  height: calc(100vh - 60px); // Account for header
  background-color: #1e1e1e;
  min-width: calc(100vw-400px);
`;

const MainPanel = styled.div`
  padding: 20px;
  overflow-y: auto;
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

const ClusterNodeComponent: React.FC<ClusterNodeProps> = ({
  cluster,
  level,
  expandedNodes,
  toggleNode,
  selectedCluster,
  onSelectCluster,
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
        onClick={() => onSelectCluster(cluster)}
      >
        {cluster.parent_name && (
          <ParentCluster>Parent: {cluster.parent_name}</ParentCluster>
        )}
        <ClusterHeader>
          {hasChildren && (
            <ToggleButton onClick={(e) => {
              e.stopPropagation();
              toggleNode(cluster.id);
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
      
      {isExpanded && hasChildren && sortedChildren.map(child => (
        <ClusterNodeComponent
          key={child.id}
          cluster={child}
          level={level + 1}
          expandedNodes={expandedNodes}
          toggleNode={toggleNode}
          selectedCluster={selectedCluster}
          onSelectCluster={onSelectCluster}
        />
      ))}
    </>
  );
};



export const ClusterTree: React.FC<ClusterTreeProps> = ({ 
  data, 
  selectedCluster,
  onSelectCluster 
}) => {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  
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
      <MainPanel>
        {sortedClusters.map(cluster => (
          <ClusterNodeComponent
            key={cluster.id}
            cluster={cluster}
            level={0}
            expandedNodes={expandedNodes}
            toggleNode={toggleNode}
            selectedCluster={selectedCluster}
            onSelectCluster={onSelectCluster}
          />
        ))}
      </MainPanel>
    </TreeLayout>
  );
};
