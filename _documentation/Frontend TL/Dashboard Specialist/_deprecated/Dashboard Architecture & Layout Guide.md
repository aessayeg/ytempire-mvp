# Dashboard Architecture & Layout Guide
## For: Dashboard Specialist | YTEMPIRE Frontend Team

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**Status**: Implementation Ready

---

## Executive Summary

As the Dashboard Specialist for YTEMPIRE, you will be responsible for creating a high-performance, real-time dashboard system that enables users to monitor and manage up to 5 YouTube channels simultaneously. This document outlines the architectural decisions, layout patterns, and implementation standards for the MVP dashboard.

### Key Constraints (MVP)
- **Desktop-only**: 1280px minimum width
- **5 channels max** per user
- **60-second polling** for most updates
- **3 critical WebSocket events** only
- **Bundle size**: <1MB total (including MUI ~300KB)
- **Performance**: Dashboard load <2 seconds

---

## 1. Dashboard Layout Architecture

### 1.1 Core Layout Structure

```typescript
// Dashboard Layout Component Hierarchy
const DashboardLayout = {
  App: {
    AuthProvider: {
      Router: {
        DashboardContainer: {
          Header: {
            Logo: {},
            ChannelSelector: {}, // Dropdown for 5 channels
            UserMenu: {},
            NotificationBadge: {}
          },
          Sidebar: {
            Navigation: {
              DashboardLink: {},
              ChannelsLink: {},
              VideosLink: {},
              AnalyticsLink: {},
              SettingsLink: {}
            },
            QuickStats: {} // Mini metrics
          },
          MainContent: {
            DashboardView: {
              MetricsRow: {},
              ChartsRow: {},
              ActivityFeed: {},
              ChannelGrid: {}
            }
          }
        }
      }
    }
  }
};
```

### 1.2 Grid System Implementation

```typescript
// Material-UI Grid Layout
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';

const DashboardGrid: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Metrics Row - 4 cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={3}>
          <MetricCard title="Active Channels" />
        </Grid>
        <Grid item xs={3}>
          <MetricCard title="Videos Today" />
        </Grid>
        <Grid item xs={3}>
          <MetricCard title="Total Revenue" />
        </Grid>
        <Grid item xs={3}>
          <MetricCard title="Cost per Video" />
        </Grid>
      </Grid>

      {/* Charts Row - 2 charts */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={8}>
          <PerformanceChart />
        </Grid>
        <Grid item xs={4}>
          <CostBreakdownChart />
        </Grid>
      </Grid>

      {/* Channel Overview */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <ChannelOverviewTable />
        </Grid>
      </Grid>
    </Box>
  );
};
```

### 1.3 Responsive Breakpoints (Desktop Only)

```typescript
// MVP: Desktop-only configuration
const layoutConfig = {
  minWidth: 1280,
  minHeight: 720,
  sidebar: {
    width: 240,
    collapsedWidth: 64
  },
  header: {
    height: 64
  },
  content: {
    maxWidth: 1920,
    padding: 24
  }
};

// Viewport enforcement
const ViewportGuard: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [viewportValid, setViewportValid] = useState(true);

  useEffect(() => {
    const checkViewport = () => {
      const valid = window.innerWidth >= 1280 && window.innerHeight >= 720;
      setViewportValid(valid);
    };

    checkViewport();
    window.addEventListener('resize', checkViewport);
    return () => window.removeEventListener('resize', checkViewport);
  }, []);

  if (!viewportValid) {
    return (
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        height="100vh"
        bgcolor="background.default"
      >
        <Typography variant="h6" color="text.secondary">
          YTEMPIRE requires minimum 1280x720 resolution
        </Typography>
      </Box>
    );
  }

  return <>{children}</>;
};
```

### 1.4 State Management Architecture

```typescript
// Zustand store for dashboard state
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface DashboardState {
  // View state
  activeView: 'overview' | 'channels' | 'analytics';
  selectedChannel: string | null;
  dateRange: { start: Date; end: Date };
  
  // Data state
  metrics: DashboardMetrics | null;
  channels: Channel[];
  recentVideos: Video[];
  costBreakdown: CostBreakdown | null;
  
  // UI state
  sidebarCollapsed: boolean;
  refreshInterval: number; // 60000ms default
  
  // Actions
  setActiveView: (view: string) => void;
  selectChannel: (channelId: string | null) => void;
  updateMetrics: (metrics: DashboardMetrics) => void;
  toggleSidebar: () => void;
}

export const useDashboardStore = create<DashboardState>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        activeView: 'overview',
        selectedChannel: null,
        dateRange: {
          start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          end: new Date()
        },
        metrics: null,
        channels: [],
        recentVideos: [],
        costBreakdown: null,
        sidebarCollapsed: false,
        refreshInterval: 60000,

        // Actions
        setActiveView: (view) => set({ activeView: view }),
        selectChannel: (channelId) => set({ selectedChannel: channelId }),
        updateMetrics: (metrics) => set({ metrics }),
        toggleSidebar: () => set((state) => ({ 
          sidebarCollapsed: !state.sidebarCollapsed 
        }))
      }),
      {
        name: 'dashboard-storage',
        partialize: (state) => ({
          sidebarCollapsed: state.sidebarCollapsed,
          dateRange: state.dateRange
        })
      }
    )
  )
);
```

---

## 2. Component Architecture

### 2.1 Base Dashboard Component

```typescript
// Dashboard.tsx - Main dashboard component
import React, { useEffect } from 'react';
import { Box, Container } from '@mui/material';
import { useDashboardStore } from '@/stores/dashboardStore';
import { useDashboardPolling } from '@/hooks/useDashboardPolling';
import { useWebSocketEvents } from '@/hooks/useWebSocketEvents';

export const Dashboard: React.FC = () => {
  const { metrics, channels, updateMetrics } = useDashboardStore();
  
  // Setup 60-second polling
  useDashboardPolling(60000);
  
  // Setup critical WebSocket events
  useWebSocketEvents(['video.completed', 'video.failed', 'cost.alert']);
  
  return (
    <Container maxWidth={false} sx={{ mt: 3, mb: 3 }}>
      <MetricsRow metrics={metrics} />
      <ChartsSection />
      <ChannelsOverview channels={channels} />
      <ActivityFeed />
    </Container>
  );
};
```

### 2.2 Metric Card Component

```typescript
// MetricCard.tsx - Reusable metric display component
import React from 'react';
import { Card, CardContent, Typography, Box, Skeleton } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  format?: 'currency' | 'number' | 'percentage';
  loading?: boolean;
  color?: 'primary' | 'success' | 'warning' | 'error';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  format = 'number',
  loading = false,
  color = 'primary'
}) => {
  const formatValue = (val: string | number): string => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(val);
      case 'percentage':
        return `${val.toFixed(1)}%`;
      default:
        return new Intl.NumberFormat('en-US').format(val);
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography color="text.secondary" gutterBottom variant="overline">
          {title}
        </Typography>
        
        {loading ? (
          <Skeleton variant="rectangular" height={40} />
        ) : (
          <>
            <Typography variant="h4" component="div" color={color}>
              {formatValue(value)}
            </Typography>
            
            {change !== undefined && (
              <Box display="flex" alignItems="center" mt={1}>
                {change > 0 ? (
                  <TrendingUpIcon color="success" fontSize="small" />
                ) : (
                  <TrendingDownIcon color="error" fontSize="small" />
                )}
                <Typography
                  variant="body2"
                  color={change > 0 ? 'success.main' : 'error.main'}
                  ml={0.5}
                >
                  {Math.abs(change).toFixed(1)}%
                </Typography>
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};
```

### 2.3 Channel Selector Component

```typescript
// ChannelSelector.tsx - Channel switching dropdown
import React from 'react';
import { 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Chip,
  Box 
} from '@mui/material';
import { useDashboardStore } from '@/stores/dashboardStore';

export const ChannelSelector: React.FC = () => {
  const { channels, selectedChannel, selectChannel } = useDashboardStore();
  
  return (
    <FormControl size="small" sx={{ minWidth: 200 }}>
      <InputLabel>Channel</InputLabel>
      <Select
        value={selectedChannel || 'all'}
        onChange={(e) => selectChannel(
          e.target.value === 'all' ? null : e.target.value
        )}
        label="Channel"
      >
        <MenuItem value="all">
          <Box display="flex" alignItems="center">
            <Typography>All Channels</Typography>
            <Chip 
              label={channels.length} 
              size="small" 
              sx={{ ml: 1 }} 
            />
          </Box>
        </MenuItem>
        
        {channels.map((channel) => (
          <MenuItem key={channel.id} value={channel.id}>
            <Box display="flex" alignItems="center" width="100%">
              <Typography flex={1}>{channel.name}</Typography>
              <Chip
                label={channel.status}
                size="small"
                color={channel.status === 'active' ? 'success' : 'default'}
              />
            </Box>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};
```

---

## 3. Performance Optimization Strategies

### 3.1 Component Memoization

```typescript
// Optimize re-renders with React.memo and useMemo
import React, { memo, useMemo } from 'react';

// Memoized chart component
export const PerformanceChart = memo<{ data: ChartData[] }>(({ data }) => {
  // Expensive calculations memoized
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      value: Math.round(item.value * 100) / 100
    }));
  }, [data]);

  return <LineChart data={processedData} />;
}, (prevProps, nextProps) => {
  // Custom comparison for shallow equality
  return JSON.stringify(prevProps.data) === JSON.stringify(nextProps.data);
});

PerformanceChart.displayName = 'PerformanceChart';
```

### 3.2 Lazy Loading Dashboard Sections

```typescript
// Lazy load heavy components
import { lazy, Suspense } from 'react';
import { CircularProgress, Box } from '@mui/material';

const AnalyticsSection = lazy(() => import('./AnalyticsSection'));
const VideoQueue = lazy(() => import('./VideoQueue'));

const DashboardSections: React.FC = () => {
  const { activeView } = useDashboardStore();
  
  return (
    <Suspense 
      fallback={
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      }
    >
      {activeView === 'analytics' && <AnalyticsSection />}
      {activeView === 'videos' && <VideoQueue />}
    </Suspense>
  );
};
```

### 3.3 Virtual Scrolling for Lists

```typescript
// Virtual scrolling for large lists
import { FixedSizeList } from 'react-window';

const VideoList: React.FC<{ videos: Video[] }> = ({ videos }) => {
  const Row = ({ index, style }: { index: number; style: CSSProperties }) => (
    <div style={style}>
      <VideoListItem video={videos[index]} />
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={videos.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
};
```

---

## 4. Error Handling & Loading States

### 4.1 Dashboard Error Boundary

```typescript
// Error boundary for dashboard sections
class DashboardErrorBoundary extends React.Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Dashboard error:', error, errorInfo);
    // Send to monitoring service
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert severity="error" sx={{ m: 2 }}>
          <AlertTitle>Dashboard Error</AlertTitle>
          Something went wrong loading this section.
          <Button 
            size="small" 
            onClick={() => window.location.reload()}
            sx={{ mt: 1 }}
          >
            Refresh Page
          </Button>
        </Alert>
      );
    }

    return this.props.children;
  }
}
```

### 4.2 Loading States

```typescript
// Consistent loading states across dashboard
export const DashboardSkeleton: React.FC = () => (
  <Box sx={{ p: 3 }}>
    {/* Metrics skeleton */}
    <Grid container spacing={3} mb={3}>
      {[1, 2, 3, 4].map((i) => (
        <Grid item xs={3} key={i}>
          <Skeleton variant="rectangular" height={120} />
        </Grid>
      ))}
    </Grid>
    
    {/* Charts skeleton */}
    <Grid container spacing={3} mb={3}>
      <Grid item xs={8}>
        <Skeleton variant="rectangular" height={400} />
      </Grid>
      <Grid item xs={4}>
        <Skeleton variant="rectangular" height={400} />
      </Grid>
    </Grid>
    
    {/* Table skeleton */}
    <Skeleton variant="rectangular" height={300} />
  </Box>
);
```

---

## Next Steps

1. Review the [Real-time Data & WebSocket Implementation Guide](dashboard-realtime-guide)
2. Study the [Data Visualization & Recharts Standards](dashboard-visualization-guide)
3. Set up your local development environment
4. Begin implementing the base dashboard layout
5. Coordinate with the React Engineer on shared components

Remember: **Simplicity and performance are key**. Every component should load fast and provide immediate value to the user.

**Questions?** Contact the Frontend Team Lead or review the project architecture documentation.