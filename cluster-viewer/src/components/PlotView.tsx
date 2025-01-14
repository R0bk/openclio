import React, { useMemo, useRef, useEffect, useCallback, useState } from 'react';
import styled from 'styled-components';
import { Cluster, Conversation } from '../types/models';

const PlotContainer = styled.div`
  width: 100%;
  height: calc(100vh - 60px);
  position: relative;
  background: #1e1e1e;
  overflow: hidden;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: 100%;
  cursor: grab;
  
  &:active {
    cursor: grabbing;
  }
`;

const Tooltip = styled.div`
  position: absolute;
  background: rgba(0, 0, 0, 0.85);
  color: white;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 14px;
  pointer-events: none;
  z-index: 100;
  max-width: 300px;
`;

interface ViewState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

interface PlotViewProps {
  conversations: { [key: string]: Conversation };
  clusters: Cluster[];
  selectedCluster?: string;
  onSelectCluster: (cluster: Cluster | null) => void;
}

const POINT_RADIUS = 2.5;
const HOVER_RADIUS = 4;
const MIN_SCALE = 0.1;
const MAX_SCALE = 10;

export const PlotView: React.FC<PlotViewProps> = ({ 
  conversations, 
  clusters, 
  selectedCluster,
  onSelectCluster 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const conversationsArray = useMemo(() => Object.values(conversations), [conversations]);
  
  // View state for panning and zooming
  const [view, setView] = useState<ViewState>({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  
  // Build conversation to cluster mapping
  const conversationToCluster = useMemo(() => {
    const mapping: { [key: string]: string } = {};
    
    function mapCluster(cluster: Cluster) {
      if (cluster.conversation_ids) {
        cluster.conversation_ids.forEach(convId => {
          mapping[convId] = cluster.id;
        });
      }
      if (cluster.child_clusters) {
        cluster.child_clusters.forEach(mapCluster);
      }
    }
    
    clusters.forEach(mapCluster);
    return mapping;
  }, [clusters]);
  
  // Create a map of cluster IDs to colors
  const clusterColors = useMemo(() => {
    const colors: { [key: string]: string } = {};
    let colorIndex = 0;

    function assignColors(cluster: Cluster) {
      const hue = (colorIndex * 137.508) % 360; // Golden ratio for good color distribution
      colors[cluster.id] = `hsl(${hue}, 70%, 60%)`;
      colorIndex++;

      if (cluster.child_clusters) {
        cluster.child_clusters.forEach(assignColors);
      }
    }

    clusters.forEach(assignColors);
    return colors;
  }, [clusters]);

  // Find the bounds of the projections
  const bounds = useMemo(() => {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    conversationsArray.forEach(conv => {
      if (conv.metadata?.projection) {
        minX = Math.min(minX, conv.metadata.projection.x);
        maxX = Math.max(maxX, conv.metadata.projection.x);
        minY = Math.min(minY, conv.metadata.projection.y);
        maxY = Math.max(maxY, conv.metadata.projection.y);
      }
    });

    return { minX, maxX, minY, maxY };
  }, [conversationsArray]);

  const [tooltip, setTooltip] = React.useState<{
    text: string;
    x: number;
    y: number;
  } | null>(null);

  // Enhanced coordinate conversion with view state
  const toCanvasCoords = useCallback((x: number, y: number, width: number, height: number) => {
    const normalizedX = ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * width;
    const normalizedY = ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * height;
    
    return {
      x: normalizedX * view.scale + view.offsetX,
      y: normalizedY * view.scale + view.offsetY
    };
  }, [bounds, view]);

  // Draw the plot with view state
  const drawPlot = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw points with scaled radius
    const scaledRadius = POINT_RADIUS * Math.sqrt(view.scale);

    conversationsArray.forEach(conv => {
      if (!conv.metadata?.projection) return;
      
      const { x, y } = toCanvasCoords(
        conv.metadata.projection.x,
        conv.metadata.projection.y,
        rect.width,
        rect.height
      );

      // Skip points outside visible area
      if (x < -scaledRadius || x > rect.width + scaledRadius ||
          y < -scaledRadius || y > rect.height + scaledRadius) {
        return;
      }

      const clusterId = conversationToCluster[conv.id];
      const isSelected = selectedCluster === clusterId;
      
      ctx.beginPath();
      ctx.arc(x, y, isSelected ? scaledRadius * 1.5 : scaledRadius, 0, Math.PI * 2);
      ctx.fillStyle = clusterId ? clusterColors[clusterId] || '#666' : '#666';
      if (!selectedCluster || isSelected) {
        ctx.globalAlpha = 1;
      } else {
        ctx.globalAlpha = 0.3;
      }
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }, [conversationsArray, clusterColors, toCanvasCoords, conversationToCluster, view, selectedCluster]);

  // Handle mouse wheel for zooming
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    
    const delta = -e.deltaY;
    const scaleChange = delta > 0 ? 1.1 : 0.9;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.scale * scaleChange));
    
    // Calculate zoom center (mouse position)
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Adjust offset to zoom towards mouse position
    const newOffsetX = mouseX - (mouseX - view.offsetX) * (newScale / view.scale);
    const newOffsetY = mouseY - (mouseY - view.offsetY) * (newScale / view.scale);
    
    setView({
      scale: newScale,
      offsetX: newOffsetX,
      offsetY: newOffsetY
    });
  }, [view]);

  // Function to find cluster by ID
  const findClusterById = useCallback((clusterId: string): Cluster | null => {
    function search(clusters: Cluster[]): Cluster | null {
      for (const cluster of clusters) {
        if (cluster.id === clusterId) return cluster;
        if (cluster.child_clusters) {
          const found = search(cluster.child_clusters);
          if (found) return found;
        }
      }
      return null;
    }
    return search(clusters);
  }, [clusters]);

  // Handle mouse events for panning and clicking
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if we clicked on a point
    const threshold = (POINT_RADIUS * 2) ** 2;
    let clickedPoint: Conversation | null = null;
    let minDistance = Infinity;

    conversationsArray.forEach(conv => {
      if (!conv.metadata?.projection) return;
      
      const coords = toCanvasCoords(
        conv.metadata.projection.x,
        conv.metadata.projection.y,
        rect.width,
        rect.height
      );

      const dx = coords.x - x;
      const dy = coords.y - y;
      const distance = dx * dx + dy * dy;

      if (distance < threshold && distance < minDistance) {
        minDistance = distance;
        clickedPoint = conv;
      }
    });

    if (clickedPoint) {
      const clusterId = conversationToCluster[clickedPoint.id];
      if (clusterId) {
        const cluster = findClusterById(clusterId);
        if (cluster) {
          onSelectCluster(cluster);
          return;
        }
      }
    }

    // If we didn't click a point, start dragging
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  }, [conversationsArray, toCanvasCoords, conversationToCluster, findClusterById, onSelectCluster]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    if (isDragging) {
      const dx = e.clientX - lastMousePos.x;
      const dy = e.clientY - lastMousePos.y;

      setView(prev => ({
        ...prev,
        offsetX: prev.offsetX + dx,
        offsetY: prev.offsetY + dy
      }));

      setLastMousePos({ x: e.clientX, y: e.clientY });
      return;
    }

    // Handle tooltip logic when not dragging
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find nearest point within certain radius
    const threshold = (HOVER_RADIUS * 2) ** 2;

    let nearestPoint: Conversation | null = null;
    let minDistance = Infinity;

    conversationsArray.forEach(conv => {
      if (!conv.metadata?.projection) return;
      
      const coords = toCanvasCoords(
        conv.metadata.projection.x,
        conv.metadata.projection.y,
        rect.width,
        rect.height
      );

      const dx = coords.x - x;
      const dy = coords.y - y;
      const distance = dx * dx + dy * dy;

      if (distance < threshold && distance < minDistance) {
        minDistance = distance;
        nearestPoint = conv;
      }
    });

    if (nearestPoint) {
      setTooltip({
        text: nearestPoint.metadata.task || 'No summary available',
        x: e.clientX,
        y: e.clientY
      });
    } else {
      setTooltip(null);
    }
  }, [isDragging, lastMousePos, conversationsArray, toCanvasCoords]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Set up event listeners
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      canvas.removeEventListener('wheel', handleWheel);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleWheel, handleMouseUp]);

  // Redraw on view changes
  useEffect(() => {
    drawPlot();
  }, [drawPlot, view]);

  return (
    <PlotContainer ref={containerRef}>
      <Canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseUp}
      />
      {tooltip && (
        <Tooltip
          style={{
            left: tooltip.x + 10 -400,
            top: tooltip.y + 10
          }}
        >
          {tooltip.text}
        </Tooltip>
      )}
    </PlotContainer>
  );
}; 