# YTEMPIRE Documentation - Dashboard Implementation

## 4.1 Dashboard Architecture

### 4.1.1 Layout & Components

#### Dashboard Layout Structure

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
      zIndex: 1000,
      components: ['Logo', 'ChannelSelector', 'NotificationBadge', 'UserMenu']
    },
    sidebar: {
      width: 240,
      collapsible: false,  // MVP: Always visible
      items: ['Dashboard', 'Channels', 'Videos', 'Analytics', 'Settings']
    },
    main: {
      padding: 24,
      minHeight: 'calc(100vh - 64px)'
    }
  }
};
```

#### Component Hierarchy (30-40 Total Components)

```typescript
// Component Structure
const ComponentInventory = {
  // Layout Components (5)
  layout: {
    AppLayout: 'Main application wrapper',
    Header: 'Top navigation bar',
    Sidebar: 'Side navigation menu',
    ContentArea: 'Main content container',
    Footer: 'Footer information'
  },
  
  // Common Components (10)
  common: {
    Button: 'Styled button variants',
    Input: 'Form input with validation',
    Card: 'Content card wrapper',
    Modal: 'Dialog/popup component',
    Dropdown: 'Select dropdown',
    Table: 'Data table with sorting',
    Badge: 'Status/count indicators',
    Avatar: 'User profile image',
    Tooltip: 'Hover information',
    Icon: 'Icon wrapper component'
  },
  
  // Chart Components (5-7)
  charts: {
    LineChart: 'Revenue/growth trends',
    BarChart: 'Channel comparisons',
    PieChart: 'Cost breakdown',
    AreaChart: 'Cumulative metrics',
    SparklineChart: 'Mini trend indicators'
  },
  
  // Form Components (8)
  forms: {
    LoginForm: 'User authentication',
    ChannelSetupForm: 'New channel wizard',
    VideoGenerationForm: 'Video parameters',
    SettingsForm: 'User preferences',
    SearchForm: 'Global search',
    FilterForm: 'Data filtering',
    ExportForm: 'Export options',
    FeedbackForm: 'User feedback'
  },
  
  // Dashboard Widgets (5-7)
  widgets: {
    MetricCard: 'KPI display card',
    ChannelCard: 'Channel summary',
    VideoQueue: 'Processing queue',
    ActivityFeed: 'Recent activities',
    CostAlert: 'Cost warnings',
    PerformanceChart: 'Main dashboard chart'
  },
  
  // Feedback Components (5)
  feedback: {
    Toast: 'Success/error messages',
    ProgressBar: 'Loading indicators',
    Skeleton: 'Loading placeholders',
    ErrorBoundary: 'Error handling',
    EmptyState: 'No data display'
  }
};
```

#### Screen Inventory (20-25 Screens)

```typescript
// Page/Screen Components
const ScreenInventory = {
  // Authentication (3)
  auth: {
    Login: '/login',
    Register: '/register',
    ForgotPassword: '/forgot-password'
  },
  
  // Dashboard (5)
  dashboard: {
    Overview: '/dashboard',
    ChannelAnalytics: '/dashboard/channels',
    RevenueAnalytics: '/dashboard/revenue',
    CostAnalytics: '/dashboard/costs',
    PerformanceMetrics: '/dashboard/performance'
  },
  
  // Channel Management (4)
  channels: {
    ChannelList: '/channels',
    ChannelDetail: '/channels/:id',
    ChannelSetup: '/channels/new',
    ChannelSettings: '/channels/:id/settings'
  },
  
  // Video Management (4)
  videos: {
    VideoQueue: '/videos/queue',
    VideoHistory: '/videos/history',
    VideoDetail: '/videos/:id',
    VideoGeneration: '/videos/generate'
  },
  
  // Analytics (3)
  analytics: {
    Overview: '/analytics',
    Reports: '/analytics/reports',
    Insights: '/analytics/insights'
  },
  
  // Settings (3)
  settings: {
    Account: '/settings/account',
    Preferences: '/settings/preferences',
    Billing: '/settings/billing'
  },
  
  // Utility (3)
  utility: {
    NotFound: '/404',
    ServerError: '/500',
    Maintenance: '/maintenance'
  }
};
```

### 4.1.2 State Management (Zustand)

#### Store Architecture

```typescript
// Main Dashboard Store
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface DashboardState {
  // Data State
  metrics: DashboardMetrics | null;
  channels: Channel[];
  videos: Video[];
  costs: CostBreakdown | null;
  
  // UI State
  selectedChannel: string | null;
  dateRange: { start: Date; end: Date };
  viewMode: 'grid' | 'list';
  sidebarCollapsed: boolean;
  
  // Loading States
  isLoadingMetrics: boolean;
  isLoadingChannels: boolean;
  error: string | null;
  
  // Actions
  fetchDashboardData: () => Promise<void>;
  updateMetrics: (metrics: Partial<DashboardMetrics>) => void;
  selectChannel: (channelId: string | null) => void;
  setDateRange: (range: { start: Date; end: Date }) => void;
  refreshData: () => Promise<void>;
  clearError: () => void;
}

export const useDashboardStore = create<DashboardState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        metrics: null,
        channels: [],
        videos: [],
        costs: null,
        selectedChannel: null,
        dateRange: {
          start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          end: new Date()
        },
        viewMode: 'grid',
        sidebarCollapsed: false,
        isLoadingMetrics: false,
        isLoadingChannels: false,
        error: null,
        
        // Actions
        fetchDashboardData: async () => {
          set({ isLoadingMetrics: true, error: null });
          try {
            const response = await api.dashboard.getOverview();
            set({
              metrics: response.metrics,
              channels: response.channels,
              isLoadingMetrics: false
            });
          } catch (error) {
            set({
              error: error.message,
              isLoadingMetrics: false
            });
          }
        },
        
        updateMetrics: (metrics) => {
          set((state) => ({
            metrics: { ...state.metrics, ...metrics }
          }));
        },
        
        selectChannel: (channelId) => {
          set({ selectedChannel: channelId });
        },
        
        setDateRange: (range) => {
          set({ dateRange: range });
          get().fetchDashboardData(); // Refetch with new range
        },
        
        refreshData: async () => {
          await get().fetchDashboardData();
        },
        
        clearError: () => {
          set({ error: null });
        }
      }),
      {
        name: 'dashboard-storage',
        partialize: (state) => ({
          dateRange: state.dateRange,
          viewMode: state.viewMode,
          sidebarCollapsed: state.sidebarCollapsed
        })
      }
    ),
    { name: 'Dashboard Store' }
  )
);
```

#### Store Patterns

```typescript
// Selectors for Computed Values
export const dashboardSelectors = {
  // Get filtered channels
  getActiveChannels: (state: DashboardState) =>
    state.channels.filter(ch => ch.status === 'active'),
  
  // Calculate total revenue
  getTotalRevenue: (state: DashboardState) =>
    state.channels.reduce((sum, ch) => sum + ch.revenue, 0),
  
  // Get selected channel data
  getSelectedChannel: (state: DashboardState) =>
    state.channels.find(ch => ch.id === state.selectedChannel),
  
  // Calculate ROI
  getROI: (state: DashboardState) => {
    if (!state.metrics) return 0;
    const { revenue, cost } = state.metrics;
    return cost > 0 ? ((revenue - cost) / cost) * 100 : 0;
  }
};

// Action Hooks for Components
export const useDashboardActions = () => {
  const store = useDashboardStore();
  
  return {
    refreshDashboard: useCallback(() => {
      store.fetchDashboardData();
    }, [store]),
    
    selectChannelWithData: useCallback(async (channelId: string) => {
      store.selectChannel(channelId);
      // Fetch channel-specific data
      const channelData = await api.channels.getDetails(channelId);
      store.updateMetrics(channelData.metrics);
    }, [store])
  };
};
```

### 4.1.3 Performance Optimization

#### Bundle Size Optimization

```typescript
// Vite Configuration for Optimal Bundle Size
export default defineConfig({
  build: {
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': [
            'react',
            'react-dom',
            'react-router-dom'
          ],
          'ui': [
            '@mui/material',
            '@emotion/react',
            '@emotion/styled'
          ],
          'charts': ['recharts'],
          'state': ['zustand'],
          'utils': ['axios', 'date-fns']
        }
      }
    },
    chunkSizeWarningLimit: 500 // KB
  }
});

// Component Code Splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => 
  import(/* webpackChunkName: "analytics" */ './pages/Analytics')
);
const Settings = lazy(() => 
  import(/* webpackChunkName: "settings" */ './pages/Settings')
);
```

#### Performance Monitoring

```typescript
// Performance Monitoring Hook
export const usePerformanceMonitor = () => {
  useEffect(() => {
    // First Contentful Paint
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          console.log('FCP:', entry.startTime);
          // Send to analytics
          if (entry.startTime > 2000) {
            console.warn('FCP exceeds 2s target');
          }
        }
      }
    });
    
    observer.observe({ entryTypes: ['paint'] });
    
    // Component render tracking
    const renderStart = performance.now();
    
    return () => {
      const renderTime = performance.now() - renderStart;
      if (renderTime > 100) {
        console.warn(`Component render took ${renderTime}ms`);
      }
    };
  }, []);
};

// Memoization for Expensive Operations
const ExpensiveDashboard = memo(({ data }) => {
  const processedData = useMemo(() => {
    // Heavy data processing
    return data.map(transformData).filter(validateData);
  }, [data]);
  
  const chartData = useMemo(() => {
    return processedData.slice(0, 100); // Limit data points
  }, [processedData]);
  
  return <DashboardContent data={chartData} />;
});
```

## 4.2 Data Visualization

### 4.2.1 Recharts Implementation

#### Chart Configuration Standards

```typescript
// Recharts Base Configuration
const CHART_CONFIG = {
  colors: {
    primary: '#2196F3',
    secondary: '#FF9800',
    success: '#4CAF50',
    error: '#F44336',
    warning: '#FFC107'
  },
  
  margins: {
    top: 5,
    right: 30,
    left: 20,
    bottom: 5
  },
  
  animation: {
    duration: 1500,
    easing: 'ease-out'
  },
  
  responsive: {
    width: '100%',
    minHeight: 300,
    maxHeight: 500
  }
};
```

### 4.2.2 Chart Standards & Types

#### 5-7 Chart Implementations

```typescript
// 1. Revenue Trend Line Chart
export const RevenueTrendChart: React.FC<{ data: any[] }> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={CHART_CONFIG.margins}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="revenue" 
          stroke={CHART_CONFIG.colors.success}
          strokeWidth={2}
          dot={false}
          animationDuration={CHART_CONFIG.animation.duration}
        />
        <Line 
          type="monotone" 
          dataKey="cost" 
          stroke={CHART_CONFIG.colors.error}
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

// 2. Channel Performance Bar Chart
export const ChannelPerformanceChart: React.FC<{ data: any[] }> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={CHART_CONFIG.margins}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="channel" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="videos" fill={CHART_CONFIG.colors.primary} />
        <Bar dataKey="revenue" fill={CHART_CONFIG.colors.success} />
      </BarChart>
    </ResponsiveContainer>
  );
};

// 3. Cost Breakdown Pie Chart
export const CostBreakdownChart: React.FC<{ data: any[] }> = ({ data }) => {
  const COLORS = Object.values(CHART_CONFIG.colors);
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

// 4. Video Generation Timeline Area Chart
export const VideoTimelineChart: React.FC<{ data: any[] }> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={CHART_CONFIG.margins}>
        <defs>
          <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={CHART_CONFIG.colors.primary} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={CHART_CONFIG.colors.primary} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Area type="monotone" dataKey="videos" stroke={CHART_CONFIG.colors.primary} fillOpacity={1} fill="url(#colorGradient)" />
      </AreaChart>
    </ResponsiveContainer>
  );
};

// 5. Success Rate Gauge Chart (Custom Implementation)
export const SuccessRateGauge: React.FC<{ rate: number }> = ({ rate }) => {
  const data = [
    { name: 'Success', value: rate, fill: CHART_CONFIG.colors.success },
    { name: 'Failure', value: 100 - rate, fill: '#f0f0f0' }
  ];
  
  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          startAngle={180}
          endAngle={0}
          innerRadius={60}
          outerRadius={80}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Pie>
        <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="text-2xl font-bold">
          {rate}%
        </text>
      </PieChart>
    </ResponsiveContainer>
  );
};
```

### 4.2.3 Custom Visualizations

```typescript
// Sparkline Component for Inline Trends
export const Sparkline: React.FC<{ data: number[] }> = ({ data }) => {
  const chartData = data.map((value, index) => ({ index, value }));
  
  return (
    <ResponsiveContainer width={100} height={30}>
      <LineChart data={chartData}>
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke={CHART_CONFIG.colors.primary}
          strokeWidth={1}
          dot={false}
          animationDuration={0}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

// Heatmap for Channel Activity
export const ActivityHeatmap: React.FC<{ data: any[] }> = ({ data }) => {
  // Custom heatmap implementation using grid
  return (
    <div className="heatmap-grid">
      {data.map((row, i) => (
        <div key={i} className="heatmap-row">
          {row.map((cell, j) => (
            <div
              key={j}
              className="heatmap-cell"
              style={{
                backgroundColor: getHeatmapColor(cell.value),
                opacity: cell.value / 100
              }}
              title={`${cell.label}: ${cell.value}`}
            />
          ))}
        </div>
      ))}
    </div>
  );
};
```

## 4.3 Real-time Features

### 4.3.1 WebSocket Implementation

```typescript
// WebSocket Manager (3 Critical Events Only)
class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private handlers = new Map<string, Set<Function>>();
  
  connect(userId: string) {
    const wsUrl = `${import.meta.env.VITE_WS_URL}/${userId}`;
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Only handle 3 critical events
      switch(data.type) {
        case 'video.completed':
          this.emit('video.completed', data);
          break;
        case 'video.failed':
          this.emit('video.failed', data);
          break;
        case 'cost.alert':
          this.emit('cost.alert', data);
          break;
        default:
          console.warn('Unexpected WebSocket event:', data.type);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.reconnect(userId);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.reconnect(userId);
    };
  }
  
  private reconnect(userId: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      setTimeout(() => this.connect(userId), delay);
    }
  }
  
  subscribe(event: string, handler: Function) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    
    return () => {
      this.handlers.get(event)?.delete(handler);
    };
  }
  
  private emit(event: string, data: any) {
    this.handlers.get(event)?.forEach(handler => handler(data));
  }
  
  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}

export const wsManager = new WebSocketManager();
```

### 4.3.2 Polling Strategy

```typescript
// 60-Second Polling Implementation
export const usePolling = (
  fetchFunction: () => Promise<void>,
  interval: number = 60000,
  enabled: boolean = true
) => {
  const savedCallback = useRef(fetchFunction);
  
  useEffect(() => {
    savedCallback.current = fetchFunction;
  }, [fetchFunction]);
  
  useEffect(() => {
    if (!enabled) return;
    
    const tick = () => {
      savedCallback.current();
    };
    
    // Initial fetch
    tick();
    
    // Set up interval
    const id = setInterval(tick, interval);
    
    // Pause when tab is not visible
    const handleVisibilityChange = () => {
      if (document.hidden) {
        clearInterval(id);
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [interval, enabled]);
};

// Dashboard Polling Hook
export const useDashboardPolling = () => {
  const store = useDashboardStore();
  
  // Poll dashboard metrics every 60 seconds
  usePolling(
    async () => {
      const metrics = await api.dashboard.getMetrics();
      store.updateMetrics(metrics);
    },
    60000
  );
  
  // Poll video queue every 5 seconds when processing
  const hasProcessingVideos = store.videos.some(v => v.status === 'processing');
  usePolling(
    async () => {
      const queue = await api.videos.getQueue();
      store.updateVideos(queue);
    },
    5000,
    hasProcessingVideos
  );
  
  // Poll costs every 30 seconds
  usePolling(
    async () => {
      const costs = await api.costs.getCurrent();
      store.updateCosts(costs);
    },
    30000
  );
};
```

### 4.3.3 Data Synchronization

```typescript
// Data Sync Manager
class DataSyncManager {
  private lastSync: Map<string, number> = new Map();
  private pendingUpdates: Map<string, any[]> = new Map();
  
  // Merge real-time updates with polled data
  mergeUpdate(dataType: string, realtimeData: any, polledData: any) {
    const lastSyncTime = this.lastSync.get(dataType) || 0;
    const now = Date.now();
    
    // If real-time data is newer, use it
    if (realtimeData.timestamp > lastSyncTime) {
      this.lastSync.set(dataType, now);
      return { ...polledData, ...realtimeData };
    }
    
    return polledData;
  }
  
  // Queue updates for batch processing
  queueUpdate(dataType: string, update: any) {
    if (!this.pendingUpdates.has(dataType)) {
      this.pendingUpdates.set(dataType, []);
    }
    this.pendingUpdates.get(dataType)!.push(update);
    
    // Process batch after 100ms
    setTimeout(() => this.processBatch(dataType), 100);
  }
  
  private processBatch(dataType: string) {
    const updates = this.pendingUpdates.get(dataType) || [];
    if (updates.length === 0) return;
    
    // Combine updates
    const combined = updates.reduce((acc, update) => ({
      ...acc,
      ...update
    }), {});
    
    // Apply to store
    useDashboardStore.getState().updateMetrics(combined);
    
    // Clear pending
    this.pendingUpdates.delete(dataType);
  }
}

export const dataSyncManager = new DataSyncManager();
```

## 4.4 Widget System

### 4.4.1 Widget Architecture

```typescript
// Base Widget Interface
interface Widget {
  id: string;
  type: 'metric' | 'chart' | 'list' | 'alert';
  title: string;
  position: { x: number; y: number; w: number; h: number };
  refreshInterval?: number;
  exportable?: boolean;
  data?: any;
}

// Widget Registry
const widgetRegistry = new Map<string, React.ComponentType<any>>();

widgetRegistry.set('metric-revenue', RevenueMetricWidget);
widgetRegistry.set('chart-performance', PerformanceChartWidget);
widgetRegistry.set('list-channels', ChannelListWidget);
widgetRegistry.set('alert-cost', CostAlertWidget);

// Widget Container Component
export const WidgetContainer: React.FC<{ widget: Widget }> = ({ widget }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(widget.data);
  
  // Auto-refresh if configured
  useEffect(() => {
    if (!widget.refreshInterval) return;
    
    const interval = setInterval(async () => {
      setIsLoading(true);
      const newData = await fetchWidgetData(widget.id);
      setData(newData);
      setIsLoading(false);
    }, widget.refreshInterval);
    
    return () => clearInterval(interval);
  }, [widget.id, widget.refreshInterval]);
  
  const Component = widgetRegistry.get(widget.type);
  if (!Component) return null;
  
  return (
    <Card
      sx={{
        gridColumn: `span ${widget.position.w}`,
        gridRow: `span ${widget.position.h}`,
        position: 'relative'
      }}
    >
      <CardHeader
        title={widget.title}
        action={
          widget.exportable && (
            <IconButton onClick={() => exportWidget(widget)}>
              <DownloadIcon />
            </IconButton>
          )
        }
      />
      <CardContent>
        {isLoading ? (
          <Skeleton variant="rectangular" height={200} />
        ) : (
          <Component data={data} />
        )}
      </CardContent>
    </Card>
  );
};
```

### 4.4.2 Export Functionality

```typescript
// Export Service
class ExportService {
  async exportToCSV(data: any[], filename: string) {
    const csv = this.convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    this.downloadFile(blob, `${filename}.csv`);
  }
  
  async exportToJSON(data: any, filename: string) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    this.downloadFile(blob, `${filename}.json`);
  }
  
  async exportChart(chartElement: HTMLElement, filename: string) {
    const canvas = await html2canvas(chartElement);
    canvas.toBlob((blob) => {
      if (blob) {
        this.downloadFile(blob, `${filename}.png`);
      }
    });
  }
  
  private convertToCSV(data: any[]): string {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(row =>
      headers.map(header => {
        const value = row[header];
        return typeof value === 'string' && value.includes(',')
          ? `"${value}"`
          : value;
      }).join(',')
    );
    
    return [headers.join(','), ...rows].join('\n');
  }
  
  private downloadFile(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
}

export const exportService = new ExportService();

// Export Button Component
export const ExportButton: React.FC<{ data: any; filename: string }> = ({
  data,
  filename
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  const handleExport = (format: 'csv' | 'json' | 'png') => {
    switch(format) {
      case 'csv':
        exportService.exportToCSV(data, filename);
        break;
      case 'json':
        exportService.exportToJSON(data, filename);
        break;
      case 'png':
        // Requires chart element reference
        break;
    }
    setAnchorEl(null);
  };
  
  return (
    <>
      <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
        <DownloadIcon />
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem onClick={() => handleExport('csv')}>Export as CSV</MenuItem>
        <MenuItem onClick={() => handleExport('json')}>Export as JSON</MenuItem>
        <MenuItem onClick={() => handleExport('png')}>Export as Image</MenuItem>
      </Menu>
    </>
  );
};
```