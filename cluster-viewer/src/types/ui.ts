import { ClusterAnalysis, Cluster } from './models';

export interface ClusterNodeProps {
    cluster: Cluster;
    level: number;
    expandedNodes: Set<string>;
    toggleNode: (clusterId: string) => void;
    selectedCluster?: string;
    onSelect: (clusterId: string) => void;
}

export interface ClusterTreeProps {
    data: ClusterAnalysis;
} 