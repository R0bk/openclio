import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { ClusterTree } from './components/ClusterTree';
import { PlotView } from './components/PlotView';
import { Sidebar } from './components/Sidebar';
import type { Cluster, ClusterAnalysis } from './types/models';

const AppContainer = styled.div`
  display: flex;
  background-color: #1e1e1e;
  min-height: 100vh;
  color: #fff;
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
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

const ViewToggle = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
  display: flex;
  gap: 8px;
  z-index: 1000;
`;

const ToggleButton = styled.button<{ active: boolean }>`
  padding: 8px 16px;
  border-radius: 20px;
  border: none;
  background: ${props => props.active ? '#666' : '#333'};
  color: ${props => props.active ? '#fff' : '#ccc'};
  cursor: pointer;
  font-size: 14px;
  
  &:hover {
    background: ${props => props.active ? '#666' : '#444'};
  }
`;

type ViewType = 'tree' | 'plot';

function App() {
  const [data, setData] = useState<ClusterAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentView, setCurrentView] = useState<ViewType>('tree');
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);

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
      {data && (
        <Sidebar 
          selectedCluster={selectedCluster} 
          onClearSelection={() => setSelectedCluster(null)} 
        />
      )}
      <MainContent>
        {loading && (
          <LoadingState>Loading cluster data...</LoadingState>
        )}
        
        {error && (
          <ErrorState>
            <h3>Error Loading Data</h3>
            <p>{error}</p>
          </ErrorState>
        )}
        
        {data && (
          <>
            {currentView === 'tree' && (
              <ClusterTree 
                data={data} 
                selectedCluster={selectedCluster?.id}
                onSelectCluster={setSelectedCluster}
              />
            )}
            {currentView === 'plot' && (
              <PlotView 
                conversations={data.conversations} 
                clusters={data.hierarchy.clusters}
                selectedCluster={selectedCluster?.id}
                onSelectCluster={setSelectedCluster}
              />
            )}
            
            <ViewToggle>
              <ToggleButton
                active={currentView === 'tree'}
                onClick={() => setCurrentView('tree')}
              >
                Tree View
              </ToggleButton>
              <ToggleButton
                active={currentView === 'plot'}
                onClick={() => setCurrentView('plot')}
              >
                Map View
              </ToggleButton>
            </ViewToggle>
          </>
        )}
      </MainContent>
    </AppContainer>
  );
}

export default App;