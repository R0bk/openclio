import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { ClusterTree } from './components/ClusterTree';
import type { Cluster, ClusterAnalysis } from './types/models';

const AppContainer = styled.div`
  background-color: #1e1e1e;
  min-height: 100vh;
  color: #fff;
`;

const Header = styled.header`
  padding: 20px;
  border-bottom: 1px solid #333;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 1.2em;
  color: #ccc;
`;

const LoadingState = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #666;
`;

const ErrorState = styled.div`
  padding: 20px;
  margin: 20px;
  background-color: #442222;
  border-radius: 4px;
  color: #ff6b6b;
`;

function App() {
  const [data, setData] = useState<ClusterAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const response = await fetch('/analysis_results.json');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();
        
        // Calculate percentages for all clusters
        const totalRecords = jsonData.metadata.total_conversations;
        function addPercentages(clusters: Cluster[]) {
          for (const cluster of clusters) {
            if (cluster.child_clusters) {
              addPercentages(cluster.child_clusters);
              // Sum up sizes from children for parent clusters
              cluster.size = cluster.child_clusters.reduce((sum, child) => sum + child.size, 0);
            }
            cluster.percentage = (cluster.size / totalRecords) * 100;
          }
        }
        addPercentages(jsonData.hierarchy.clusters);
        
        setData(jsonData);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load cluster data');
        console.error('Error loading data:', e);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <AppContainer>
      <Header>
        <Title>Color and sort: Clusters, sorted by size</Title>
      </Header>
      
      {loading && (
        <LoadingState>Loading cluster data...</LoadingState>
      )}
      
      {error && (
        <ErrorState>
          <h3>Error Loading Data</h3>
          <p>{error}</p>
        </ErrorState>
      )}
      
      {data && <ClusterTree data={data} />}
    </AppContainer>
  );
}

export default App;