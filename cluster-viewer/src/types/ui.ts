import { Cluster } from './models';

export interface ClusterNodeProps {
    cluster: Cluster;
    level: number;
    expandedNodes: Set<string>;
    toggleNode: (id: string) => void;
    selectedCluster?: string;
    onSelectCluster: (cluster: Cluster | null) => void;
}

export interface ClusterTreeProps {
    data: {
      hierarchy: {
        clusters: Cluster[];
      };
    };
    selectedCluster?: string;
    onSelectCluster: (cluster: Cluster | null) => void;
}