# Dashboard Specialist Implementation Guide

**Version**: 1.0  
**Date**: January 2025  
**For**: Dashboard Specialist  
**From**: Frontend Team Lead  
**Status**: MVP Implementation Ready

---

## Executive Summary

As the Dashboard Specialist for YTEMPIRE, you're responsible for creating the visual intelligence layer that transforms complex data from multiple YouTube channels into actionable insights. This guide provides complete specifications for implementing our dashboard infrastructure within the MVP constraints.

### Critical MVP Constraints (Must Read First)
- **Maximum 5 channels per user** (not 100+)
- **50 beta users total** (not 1000+)
- **Recharts ONLY** for visualizations (no D3.js, no Plotly)
- **60-second polling** for updates (no complex real-time)
- **Desktop-only** (1280px minimum width)
- **Bundle size < 1MB total** (Recharts ~150KB)
- **5-7 charts total** across entire application

---

## 1. Dashboard Layout Architecture

### 1.1 Core Layout Structure

```typescript
// Dashboard Layout Configuration
const DashboardLayout = {
  container: {
    minWidth: 1280,  // Desktop only for MVP
    maxWidth: 1920,
    padding: 16,
    backgroundColor: '#f5f5f5'
  },
  
  grid: {
    columns: 12,
    gutter: 16,
    rowHeight: 80  // Base unit for widget heights
  },
  
  sections: {
    header: {
      height: 64,
      fixed: true,
      zIndex: 1000
    },
    sidebar: {
      width: 240,
      collapsible: false  // MVP: Always visible
    },
    main: {
      padding: 24,
      minHeight: 'calc(100vh - 64px)'
    }
  }
};
```

### 1.2 Dashboard Views Hierarchy

```typescript
// MVP Dashboard Views (Limited Scope)
const DashboardViews = {
  "/dashboard": {
    name: "Overview",
    layout: "grid",
    widgets: [
      { id: "metrics-cards", row: 0, col: 0, width: 12, height: 1 },
      { id: "channel-performance", row: 1, col: 0, width: 8, height: 3 },
      { id: "cost-breakdown", row: 1, col: 8, width: 4, height: 3 },
      { id: "video-queue", row: 4, col: 0, width: 12, height: 2 }
    ]
  },
  
  "/dashboard/channels": {
    name: "Channel Analytics",
    layout: "grid",
    widgets: [
      { id: "channel-selector", row: 0, col: 0, width: 12, height: 1 },
      { id: "channel-metrics", row: 1, col: 0, width: 6, height: 3 },
      { id: "revenue-chart", row: 1, col: 6, width: 6, height: 3 },
      { id: "video-performance", row: 4, col: 0, width: 12, height: 3 }
    ]
  },
  
  "/dashboard/costs": {
    name: "Cost Management",
    layout: "split",
    widgets: [
      { id: "cost-alerts", row: 0, col: 0, width: 12, height: 1 },
      { id: "daily-costs", row: 1, col: 0, width: 6, height: 3 },
      { id: "cost-projection", row: 1, col: 6, width: 6, height: 3 }
    ]
  }
};
```

### 1.3 Responsive Grid System (Desktop-Only)

```tsx
import { Box, Grid } from '@mui/material';

const DashboardGrid: React.FC = ({ children }) => {
  return (
    <Box sx={{ 
      minWidth: 1280,  // Enforce desktop minimum
      p: 3,
      bgcolor: 'background.default' 
    }}>
      <Grid container spacing={2}>
        {React.Children.map(children, (child, index) => {
          const widget = child as React.ReactElement<WidgetProps>;
          return (
            <Grid 
              item 
              xs={widget.props.width || 12}
              key={widget.props.id || index}
            >
              {child}
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};
```

### 1.4 Widget Container Architecture

```tsx
interface WidgetContainerProps {
  id: string;
  title: string;
  loading?: boolean;
  error?: string | null;
  refreshInterval?: number;  // Default: 60000ms
  actions?: React.ReactNode;
  children: React.ReactNode;
}

const WidgetContainer: React.FC<WidgetContainerProps> = ({
  id,
  title,
  loading = false,
  error = null,
  refreshInterval = 60000,
  actions,
  children
}) => {
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  
  // Auto-refresh based on interval
  useEffect(() => {
    const timer = setInterval(() => {
      setLastUpdate(Date.now());
    }, refreshInterval);
    
    return () => clearInterval(timer);
  }, [refreshInterval]);
  
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardHeader
        title={title}
        subheader={`Updated: ${formatRelativeTime(lastUpdate)}`}
        action={actions}
        sx={{ pb: 1 }}
      />
      <CardContent sx={{ flex: 1, position: 'relative' }}>
        {loading && (
          <Box sx={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            right: 0, 
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            zIndex: 1
          }}>
            <CircularProgress />
          </Box>
        )}
        
        {error ? (
          <Alert severity="error">{error}</Alert>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
};
```

---

## 2. Real-time Data Requirements

### 2.1 Data Update Strategy (Polling-Based)

```typescript
// MVP: Polling-based updates (NOT WebSockets for dashboard metrics)
const DataUpdateStrategy = {
  polling: {
    dashboard: 60000,      // 1 minute
    videoStatus: 5000,     // 5 seconds (only during generation)
    costs: 30000,          // 30 seconds
    channels: 60000,       // 1 minute
    queue: 10000          // 10 seconds
  },
  
  websocket: {
    // Only 3 critical events via WebSocket
    events: [
      'video.completed',
      'video.failed', 
      'cost.alert'
    ]
  }
};
```

### 2.2 Data Fetching Implementation

```typescript
// Using React Query for polling
import { useQuery } from '@tanstack/react-query';

const useDashboardData = () => {
  // Main dashboard metrics - 60 second polling
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['dashboard', 'metrics'],
    queryFn: fetchDashboardMetrics,
    refetchInterval: 60000,
    staleTime: 50000  // Consider stale after 50 seconds
  });
  
  // Channel data - 60 second polling
  const { data: channels } = useQuery({
    queryKey: ['channels'],
    queryFn: fetchChannels,
    refetchInterval: 60000
  });
  
  // Cost data - 30 second polling (more critical)
  const { data: costs } = useQuery({
    queryKey: ['costs'],
    queryFn: fetchCostBreakdown,
    refetchInterval: 30000
  });
  
  // Video queue - 10 second polling (user-facing)
  const { data: queue } = useQuery({
    queryKey: ['video-queue'],
    queryFn: fetchVideoQueue,
    refetchInterval: 10000,
    enabled: queue?.processing?.length > 0  // Only poll if videos processing
  });
  
  return {
    metrics,
    channels,
    costs,
    queue,
    isLoading: metricsLoading
  };
};
```

### 2.3 Data Aggregation Layer

```typescript
class DashboardDataAggregator {
  // Aggregate data for dashboard display
  aggregateChannelMetrics(channels: Channel[]): AggregatedMetrics {
    return {
      totalChannels: channels.length,
      activeChannels: channels.filter(c => c.status === 'active').length,
      totalVideos: channels.reduce((sum, c) => sum + c.videoCount, 0),
      totalRevenue: channels.reduce((sum, c) => sum + c.revenue, 0),
      avgVideosPerChannel: channels.length > 0 
        ? channels.reduce((sum, c) => sum + c.videoCount, 0) / channels.length
        : 0,
      performanceByChannel: channels.map(c => ({
        id: c.id,
        name: c.name,
        videos: c.videoCount,
        revenue: c.revenue,
        roi: c.revenue / c.costs
      }))
    };
  }
  
  // Prepare time-series data for charts
  prepareTimeSeriesData(
    rawData: DataPoint[], 
    interval: 'hour' | 'day' | 'week'
  ): ChartData[] {
    // Simplify to max 100 points for performance
    const simplified = this.simplifyDataPoints(rawData, 100);
    
    return simplified.map(point => ({
      timestamp: this.formatTimestamp(point.timestamp, interval),
      value: point.value,
      label: this.generateLabel(point)
    }));
  }
  
  // Calculate real-time metrics
  calculateRealtimeMetrics(data: DashboardData): RealtimeMetrics {
    return {
      videosGeneratingNow: data.queue?.processing?.length || 0,
      estimatedCompletionTime: this.calculateETA(data.queue),
      currentCostRate: this.calculateCostRate(data.costs),
      projectedDailyCost: this.projectDailyCost(data.costs),
      automationPercentage: this.calculateAutomation(data.metrics)
    };
  }
}
```

### 2.4 State Management with Zustand

```typescript
// Dashboard-specific Zustand store
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface DashboardStore {
  // State
  metrics: DashboardMetrics | null;
  channels: Channel[];
  costs: CostBreakdown | null;
  queue: QueueStatus | null;
  selectedChannel: string | null;
  dateRange: DateRange;
  
  // Actions
  updateMetrics: (metrics: DashboardMetrics) => void;
  updateChannels: (channels: Channel[]) => void;
  updateCosts: (costs: CostBreakdown) => void;
  updateQueue: (queue: QueueStatus) => void;
  selectChannel: (channelId: string | null) => void;
  setDateRange: (range: DateRange) => void;
  
  // Computed
  getChannelById: (id: string) => Channel | undefined;
  getFilteredMetrics: () => DashboardMetrics | null;
}

export const useDashboardStore = create<DashboardStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    metrics: null,
    channels: [],
    costs: null,
    queue: null,
    selectedChannel: null,
    dateRange: { start: new Date(), end: new Date() },
    
    // Update actions
    updateMetrics: (metrics) => set({ metrics }),
    updateChannels: (channels) => set({ channels }),
    updateCosts: (costs) => set({ costs }),
    updateQueue: (queue) => set({ queue }),
    
    selectChannel: (channelId) => set({ selectedChannel: channelId }),
    setDateRange: (range) => set({ dateRange: range }),
    
    // Computed getters
    getChannelById: (id) => {
      return get().channels.find(c => c.id === id);
    },
    
    getFilteredMetrics: () => {
      const { metrics, selectedChannel } = get();
      if (!metrics || !selectedChannel) return metrics;
      
      // Filter metrics by selected channel
      return {
        ...metrics,
        channelMetrics: metrics.channelMetrics.filter(
          m => m.channelId === selectedChannel
        )
      };
    }
  }))
);
```

---

## 3. WebSocket Events Specification

### 3.1 Critical Events Only (MVP Scope)

```typescript
// MVP: Only 3 critical WebSocket events
enum CriticalWebSocketEvents {
  VIDEO_COMPLETED = 'video.completed',
  VIDEO_FAILED = 'video.failed',
  COST_ALERT = 'cost.alert'
}

// Event payload interfaces
interface VideoCompletedEvent {
  type: 'video.completed';
  data: {
    videoId: string;
    channelId: string;
    channelName: string;
    title: string;
    url: string;
    cost: number;
    generationTime: number;
  };
  timestamp: string;
}

interface VideoFailedEvent {
  type: 'video.failed';
  data: {
    videoId: string;
    channelId: string;
    error: string;
    stage: string;
    retryable: boolean;
  };
  timestamp: string;
}

interface CostAlertEvent {
  type: 'cost.alert';
  data: {
    alertLevel: 'warning' | 'critical';
    currentCost: number;
    threshold: number;
    projection: number;
    message: string;
  };
  timestamp: string;
}
```

### 3.2 WebSocket Connection Manager

```typescript
class DashboardWebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  connect(userId: string) {
    const wsUrl = `ws://localhost:8000/ws/critical/${userId}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('Dashboard WebSocket connected');
        this.reconnectAttempts = 0;
        this.subscribeToEvents();
      };
      
      this.ws.onmessage = (event) => {
        this.handleWebSocketMessage(event);
      };
      
      this.ws.onerror = (error) => {
        console.error('Dashboard WebSocket error:', error);
      };
      
      this.ws.onclose = () => {
        this.handleReconnect();
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }
  
  private handleWebSocketMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case CriticalWebSocketEvents.VIDEO_COMPLETED:
          this.handleVideoCompleted(data);
          break;
          
        case CriticalWebSocketEvents.VIDEO_FAILED:
          this.handleVideoFailed(data);
          break;
          
        case CriticalWebSocketEvents.COST_ALERT:
          this.handleCostAlert(data);
          break;
          
        default:
          console.warn('Unknown WebSocket event:', data.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }
  
  private handleVideoCompleted(event: VideoCompletedEvent) {
    // Update dashboard immediately
    useDashboardStore.getState().updateQueue(null);
    
    // Show success notification
    toast.success(`Video completed for ${event.data.channelName}`);
    
    // Trigger metrics refresh
    queryClient.invalidateQueries(['dashboard', 'metrics']);
  }
  
  private handleVideoFailed(event: VideoFailedEvent) {
    // Show error notification
    toast.error(`Video generation failed: ${event.data.error}`);
    
    // Update queue status
    queryClient.invalidateQueries(['video-queue']);
  }
  
  private handleCostAlert(event: CostAlertEvent) {
    // Show cost alert based on severity
    if (event.data.alertLevel === 'critical') {
      // Show modal alert for critical
      showCostAlertModal(event.data);
    } else {
      // Show toast for warning
      toast.warning(event.data.message);
    }
    
    // Update cost display immediately
    queryClient.invalidateQueries(['costs']);
  }
  
  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting WebSocket... (attempt ${this.reconnectAttempts})`);
        this.connect(getUserId());
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
```

---

## 4. Performance Optimization

### 4.1 Rendering Optimization

```typescript
// Optimize chart rendering with memoization
const OptimizedChart = React.memo(({ data, type }) => {
  // Only re-render if data actually changes
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Limit to 100 points for performance
    if (data.length > 100) {
      return simplifyDataPoints(data, 100);
    }
    
    return data;
  }, [data]);
  
  // Disable animations for large datasets
  const animationDuration = chartData.length > 50 ? 0 : 1500;
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <Line 
          type="monotone" 
          dataKey="value" 
          animationDuration={animationDuration}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for re-render decision
  return JSON.stringify(prevProps.data) === JSON.stringify(nextProps.data);
});
```

### 4.2 Data Virtualization

```typescript
// Virtualize large lists in dashboard
import { FixedSizeList } from 'react-window';

const VirtualizedVideoList: React.FC<{ videos: Video[] }> = ({ videos }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <VideoListItem video={videos[index]} />
    </div>
  );
  
  return (
    <FixedSizeList
      height={400}  // Fixed height for virtualization
      itemCount={videos.length}
      itemSize={80}  // Height of each row
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
};
```

---

## Next Steps

1. **Implement base dashboard layout** using the grid system
2. **Set up Zustand store** for dashboard state
3. **Configure React Query** for polling
4. **Create WebSocket manager** for critical events
5. **Build first Recharts component** following standards
6. **Test with mock data** before API integration

**Remember**: Focus on MVP scope - 5 channels max, 60-second polling, Recharts only, desktop-only interface.