import React from 'react';
import styled from 'styled-components';
import { Cluster } from '../types/models';

const SidebarContainer = styled.div`
  width: 400px;
  height: 100vh;
  background: #1e1e1e;
  border-right: 1px solid #333;
  overflow-y: auto;
  padding: 20px;
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

const ClearButton = styled.button`
  position: absolute;
  top: 20px;
  right: 20px;
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  padding: 8px;
  font-size: 18px;
  border-radius: 4px;

  &:hover {
    background: #333;
    color: #fff;
  }
`;

const HeaderSection = styled.div`
  position: relative;
  margin-right: 24px;
`;

interface SidebarProps {
  selectedCluster: Cluster | null;
  onClearSelection: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ selectedCluster, onClearSelection }) => {
  return (
    <SidebarContainer>
      {selectedCluster ? (
        <ClusterDetails>
          <HeaderSection>
            <ClusterTitle>{selectedCluster.name}</ClusterTitle>
            <ClusterId>#{selectedCluster.id}</ClusterId>
            <ClearButton onClick={onClearSelection}>×</ClearButton>
          </HeaderSection>
          
          <DetailSection>
            <DetailLabel>Description</DetailLabel>
            <DetailValue>{selectedCluster.description}</DetailValue>
          </DetailSection>
          
          <DetailSection>
            <DetailLabel>Statistics</DetailLabel>
            <DetailValue>
              {selectedCluster.size.toLocaleString()} records
              <br />
              {selectedCluster.percentage?.toFixed(1)}% of total
              {selectedCluster.child_clusters && (
                <>
                  <br />
                  {selectedCluster.child_clusters.length} child clusters
                </>
              )}
            </DetailValue>
          </DetailSection>
          
          {selectedCluster.parent_name && (
            <DetailSection>
              <DetailLabel>Parent Cluster</DetailLabel>
              <DetailValue>{selectedCluster.parent_name}</DetailValue>
            </DetailSection>
          )}
        </ClusterDetails>
      ) : (
        <DetailValue>Select a cluster to view details</DetailValue>
      )}
    </SidebarContainer>
  );
}; 