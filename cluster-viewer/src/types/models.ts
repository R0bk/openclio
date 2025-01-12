// Basic types
type ConversationId = string;
type ClusterId = string;
type ISO8601Date = string;
type CountryCode = string;

// Common cluster properties
export interface ClusterBase {
    id: ClusterId;
    name: string;
    description: string;
    size: number;
    level: number;  // 0 is base level, higher numbers are higher in hierarchy
    concerning_score_avg: number;
    sample_conversations: ConversationId[];
    parent_id?: ClusterId;
    percentage?: number;  // Percentage of total records
    parent_name?: string; // Name of parent cluster
}

// Recursive cluster type that can have n levels
export interface Cluster extends ClusterBase {
    child_clusters?: Cluster[];
    conversation_ids?: ConversationId[];
}

export interface ClusterAnalysis {
    metadata: {
        total_conversations: number;
        analysis_date: ISO8601Date;
        min_cluster_size: number;
    };
    hierarchy: {
        clusters: Cluster[];
    };
} 