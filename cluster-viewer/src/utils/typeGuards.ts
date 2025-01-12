import { ClusterAnalysis } from '../types/models';

export function isClusterAnalysis(data: unknown): data is ClusterAnalysis {
  if (!data || typeof data !== 'object') return false;
  
  const d = data as Partial<ClusterAnalysis>;
  
  return !!(
    d.metadata &&
    typeof d.metadata.total_conversations === 'number' &&
    d.hierarchy &&
    Array.isArray(d.hierarchy.clusters)
  );
} 