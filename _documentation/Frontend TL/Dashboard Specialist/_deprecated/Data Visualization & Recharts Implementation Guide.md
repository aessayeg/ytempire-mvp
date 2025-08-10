# Data Visualization & Recharts Implementation Guide
## For: Dashboard Specialist | YTEMPIRE Frontend Team

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**Status**: Implementation Ready

---

## Executive Summary

This guide establishes the data visualization standards for YTEMPIRE's dashboard using **Recharts as the exclusive charting library**. We'll create 5-7 total charts across the application that provide clear, actionable insights while maintaining <500ms render times.

### Key Visualization Constraints (MVP)
- **Library**: Recharts ONLY (no D3.js, no Plotly)
- **Total charts**: 5-7 across entire application
- **Data points**: Maximum 100 per chart
- **Animation**: Disabled for >50 data points
- **Update frequency**: 60-second minimum
- **Bundle impact**: ~100KB from Recharts

---

## 1. Data Visualization Standards

### 1.1 Chart Selection Matrix

```typescript
// Chart type selection based on data and purpose
const chartSelectionGuide = {
  // Time-series data
  timeSeries: {
    chartType: 'LineChart',
    useCase: 'Revenue trends, video performance over time',
    maxDataPoints: 100,
    features: ['area fill', 'multiple lines', 'tooltips']
  },
  
  // Comparisons
  comparisons: {
    chartType: 'BarChart',
    useCase: 'Channel performance, daily video counts',
    maxDataPoints: 50,
    features: ['grouped bars', 'stacked option', 'labels']
  },
  
  // Proportions
  proportions: {
    chartType: 'PieChart',
    useCase: 'Cost breakdown, traffic sources',
    maxDataPoints: 8, // Limit for readability
    features: ['labels', 'percentages', 'legend']
  },
  
  // Cumulative/Range
  cumulative: {
    chartType: 'AreaChart',
    useCase: 'Cumulative revenue, view accumulation',
    maxDataPoints: 100,
    features: ['gradient fill', 'stacked areas']
  }
};
```

### 1.2 Color Palette & Theme

```typescript
// Consistent color scheme across all visualizations
export const chartTheme = {
  colors: {
    primary: '#2196F3',    // Blue - Primary metrics
    success: '#4CAF50',    // Green - Positive trends
    warning: '#FF9800',    // Orange - Warnings
    error: '#F44336',      // Red - Errors/failures
    secondary: '#9C27B0',  // Purple - Secondary metrics
    neutral: '#607D8B'     // Blue Grey - Neutral data
  },
  
  gradients: {
    primary: ['#2196F3', '#1976D2'],
    success: ['#4CAF50', '#388E3C'],
    revenue: ['#4CAF50', '#8BC34A']
  },
  
  chart: {
    background: 'transparent',
    gridColor: '#E0E0E0',
    textColor: '#424242',
    fontSize: 12,
    fontFamily: '"Inter", -apple-system, sans-serif'
  }
};

// Chart color assignments
export const getChartColors = (dataKeys: string[]): string[] => {
  const colorOrder = [
    chartTheme.colors.primary,
    chartTheme.colors.success,
    chartTheme.colors.warning,
    chartTheme.colors.secondary,
    chartTheme.colors.neutral
  ];
  
  return dataKeys.map((_, index) => colorOrder[index % colorOrder.length]);
};
```

### 1.3 Responsive Chart Container

```typescript
// ResponsiveChartContainer.tsx - Wrapper for all charts
import React from 'react';
import { Paper, Box, Typography, Skeleton } from '@mui/material';
import { ResponsiveContainer } from 'recharts';

interface ChartContainerProps {
  title: string;
  subtitle?: string;
  height?: number;
  loading?: boolean;
  error?: string;
  children: React.ReactNode;
}

export const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  subtitle,
  height = 400,
  loading = false,
  error,
  children
}) => {
  return (
    <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
      <Box mb={2}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </Box>
      
      {loading ? (
        <Skeleton variant="rectangular" height={height} />
      ) : error ? (
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height={height}
        >
          <Typography color="error">{error}</Typography>
        </Box>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          {children}
        </ResponsiveContainer>
      )}
    </Paper>
  );
};
```

---

## 2. Chart Library Implementation (Recharts)

### 2.1 Revenue Trend Line Chart

```typescript
// RevenueTrendChart.tsx - Primary dashboard chart
import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Area
} from 'recharts';
import { format } from 'date-fns';
import { chartTheme } from '@/theme/charts';

interface RevenueData {
  date: string;
  revenue: number;
  cost: number;
  profit: number;
}

interface RevenueTrendChartProps {
  data: RevenueData[];
  period: 'day' | 'week' | 'month';
}

export const RevenueTrendChart: React.FC<RevenueTrendChartProps> = ({
  data,
  period
}) => {
  // Process data for optimal performance
  const processedData = useMemo(() => {
    // Limit to 100 data points
    if (data.length > 100) {
      const step = Math.ceil(data.length / 100);
      return data.filter((_, index) => index % step === 0);
    }
    return data;
  }, [data]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1.5 }} elevation={3}>
          <Typography variant="body2" gutterBottom>
            {format(new Date(label), 'MMM dd, yyyy')}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              style={{ color: entry.color }}
            >
              {entry.name}: ${entry.value.toFixed(2)}
            </Typography>
          ))}
        </Paper>
      );
    }
    return null;
  };

  // Axis formatters
  const formatXAxis = (tickItem: string) => {
    const date = new Date(tickItem);
    switch (period) {
      case 'day':
        return format(date, 'HH:mm');
      case 'week':
        return format(date, 'EEE');
      case 'month':
        return format(date, 'MMM dd');
      default:
        return format(date, 'MMM dd');
    }
  };

  const formatYAxis = (value: number) => `$${value}`;

  return (
    <LineChart
      data={processedData}
      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
    >
      <defs>
        <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%" stopColor={chartTheme.colors.success} stopOpacity={0.8}/>
          <stop offset="95%" stopColor={chartTheme.colors.success} stopOpacity={0.1}/>
        </linearGradient>
      </defs>
      
      <CartesianGrid 
        strokeDasharray="3 3" 
        stroke={chartTheme.chart.gridColor}
        vertical={false}
      />
      
      <XAxis
        dataKey="date"
        tickFormatter={formatXAxis}
        stroke={chartTheme.chart.textColor}
        style={{ fontSize: chartTheme.chart.fontSize }}
      />
      
      <YAxis
        tickFormatter={formatYAxis}
        stroke={chartTheme.chart.textColor}
        style={{ fontSize: chartTheme.chart.fontSize }}
      />
      
      <Tooltip content={<CustomTooltip />} />
      
      <Legend
        wrapperStyle={{ fontSize: chartTheme.chart.fontSize }}
        iconType="line"
      />
      
      <Line
        type="monotone"
        dataKey="revenue"
        name="Revenue"
        stroke={chartTheme.colors.success}
        strokeWidth={2}
        dot={false}
        animationDuration={processedData.length > 50 ? 0 : 1500}
      />
      
      <Line
        type="monotone"
        dataKey="cost"
        name="Cost"
        stroke={chartTheme.colors.warning}
        strokeWidth={2}
        dot={false}
        animationDuration={processedData.length > 50 ? 0 : 1500}
      />
      
      <Line
        type="monotone"
        dataKey="profit"
        name="Profit"
        stroke={chartTheme.colors.primary}
        strokeWidth={3}
        dot={false}
        animationDuration={processedData.length > 50 ? 0 : 1500}
      />
    </LineChart>
  );
};
```

### 2.2 Channel Performance Bar Chart

```typescript
// ChannelPerformanceChart.tsx - Compare channel metrics
import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from 'recharts';

interface ChannelMetrics {
  channelName: string;
  videos: number;
  revenue: number;
  avgViews: number;
  status: 'active' | 'paused';
}

interface ChannelPerformanceChartProps {
  data: ChannelMetrics[];
}

export const ChannelPerformanceChart: React.FC<ChannelPerformanceChartProps> = ({
  data
}) => {
  // Custom bar colors based on status
  const getBarColor = (status: string) => {
    return status === 'active' 
      ? chartTheme.colors.primary 
      : chartTheme.colors.neutral;
  };

  return (
    <BarChart
      data={data}
      margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
    >
      <CartesianGrid 
        strokeDasharray="3 3" 
        stroke={chartTheme.chart.gridColor}
        vertical={false}
      />
      
      <XAxis
        dataKey="channelName"
        angle={-45}
        textAnchor="end"
        height={100}
        interval={0}
        style={{ fontSize: chartTheme.chart.fontSize }}
      />
      
      <YAxis
        yAxisId="left"
        orientation="left"
        stroke={chartTheme.colors.primary}
        style={{ fontSize: chartTheme.chart.fontSize }}
      />
      
      <YAxis
        yAxisId="right"
        orientation="right"
        stroke={chartTheme.colors.success}
        style={{ fontSize: chartTheme.chart.fontSize }}
      />
      
      <Tooltip
        formatter={(value: any, name: string) => {
          if (name === 'Revenue') return `$${value.toFixed(2)}`;
          return value.toLocaleString();
        }}
      />
      
      <Legend />
      
      <Bar 
        yAxisId="left"
        dataKey="videos" 
        name="Videos"
        fill={chartTheme.colors.primary}
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={getBarColor(entry.status)} />
        ))}
      </Bar>
      
      <Bar 
        yAxisId="right"
        dataKey="revenue" 
        name="Revenue ($)"
        fill={chartTheme.colors.success}
      />
    </BarChart>
  );
};
```

### 2.3 Cost Breakdown Pie Chart

```typescript
// CostBreakdownChart.tsx - Visualize cost distribution
import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface CostCategory {
  name: string;
  value: number;
  percentage: number;
}

interface CostBreakdownChartProps {
  data: CostCategory[];
}

export const CostBreakdownChart: React.FC<CostBreakdownChartProps> = ({
  data
}) => {
  const COLORS = [
    chartTheme.colors.primary,
    chartTheme.colors.success,
    chartTheme.colors.warning,
    chartTheme.colors.secondary,
    chartTheme.colors.neutral
  ];

  // Custom label renderer
  const renderLabel = (entry: any) => {
    return `${entry.percentage}%`;
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <Paper sx={{ p: 1.5 }} elevation={3}>
          <Typography variant="body2">
            {data.name}: ${data.value.toFixed(2)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {data.payload.percentage}% of total
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <PieChart>
      <Pie
        data={data}
        cx="50%"
        cy="50%"
        labelLine={false}
        label={renderLabel}
        outerRadius={120}
        fill="#8884d8"
        dataKey="value"
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
        ))}
      </Pie>
      
      <Tooltip content={<CustomTooltip />} />
      
      <Legend
        verticalAlign="bottom"
        height={36}
        formatter={(value: string, entry: any) => 
          `${value}: $${entry.payload.value.toFixed(2)}`
        }
      />
    </PieChart>
  );
};
```

### 2.4 Video Generation Timeline

```typescript
// VideoGenerationTimeline.tsx - Area chart for video generation over time
import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';

interface GenerationData {
  hour: string;
  completed: number;
  failed: number;
  processing: number;
}

export const VideoGenerationTimeline: React.FC<{ data: GenerationData[] }> = ({
  data
}) => {
  return (
    <AreaChart
      data={data}
      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
    >
      <defs>
        <linearGradient id="completedGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%" stopColor={chartTheme.colors.success} stopOpacity={0.8}/>
          <stop offset="95%" stopColor={chartTheme.colors.success} stopOpacity={0.1}/>
        </linearGradient>
        <linearGradient id="failedGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%" stopColor={chartTheme.colors.error} stopOpacity={0.8}/>
          <stop offset="95%" stopColor={chartTheme.colors.error} stopOpacity={0.1}/>
        </linearGradient>
      </defs>
      
      <CartesianGrid 
        strokeDasharray="3 3" 
        stroke={chartTheme.chart.gridColor}
      />
      
      <XAxis 
        dataKey="hour" 
        stroke={chartTheme.chart.textColor}
      />
      
      <YAxis 
        stroke={chartTheme.chart.textColor}
      />
      
      <Tooltip />
      <Legend />
      
      <Area
        type="monotone"
        dataKey="completed"
        stackId="1"
        stroke={chartTheme.colors.success}
        fill="url(#completedGradient)"
        name="Completed"
      />
      
      <Area
        type="monotone"
        dataKey="failed"
        stackId="1"
        stroke={chartTheme.colors.error}
        fill="url(#failedGradient)"
        name="Failed"
      />
      
      <Area
        type="monotone"
        dataKey="processing"
        stackId="1"
        stroke={chartTheme.colors.warning}
        fill={chartTheme.colors.warning}
        fillOpacity={0.6}
        name="Processing"
      />
    </AreaChart>
  );
};
```

---

## 3. Widget Architecture Design

### 3.1 Dashboard Widget System

```typescript
// DashboardWidget.tsx - Base widget component
import React from 'react';
import { Card, CardContent, CardHeader, IconButton, Menu, MenuItem } from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import RefreshIcon from '@mui/icons-material/Refresh';

export interface WidgetConfig {
  id: string;
  title: string;
  type: 'metric' | 'chart' | 'list' | 'summary';
  refreshable?: boolean;
  exportable?: boolean;
  collapsible?: boolean;
  defaultSize?: { w: number; h: number };
}

interface DashboardWidgetProps extends WidgetConfig {
  loading?: boolean;
  onRefresh?: () => void;
  onExport?: () => void;
  children: React.ReactNode;
}

export const DashboardWidget: React.FC<DashboardWidgetProps> = ({
  title,
  refreshable = true,
  exportable = true,
  loading = false,
  onRefresh,
  onExport,
  children
}) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleRefresh = () => {
    onRefresh?.();
    handleMenuClose();
  };

  const handleExport = () => {
    onExport?.();
    handleMenuClose();
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardHeader
        title={title}
        action={
          <>
            {refreshable && (
              <IconButton onClick={handleRefresh} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            )}
            <IconButton onClick={handleMenuOpen}>
              <MoreVertIcon />
            </IconButton>
          </>
        }
        sx={{ pb: 1 }}
      />
      <CardContent sx={{ flex: 1, pt: 0 }}>
        {children}
      </CardContent>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        {refreshable && (
          <MenuItem onClick={handleRefresh}>Refresh</MenuItem>
        )}
        {exportable && (
          <MenuItem onClick={handleExport}>Export Data</MenuItem>
        )}
      </Menu>
    </Card>
  );
};

### 3.2 Widget Registry

```typescript
// widgetRegistry.ts - Central widget configuration
import { RevenueTrendChart } from './charts/RevenueTrendChart';
import { ChannelPerformanceChart } from './charts/ChannelPerformanceChart';
import { CostBreakdownChart } from './charts/CostBreakdownChart';
import { VideoGenerationTimeline } from './charts/VideoGenerationTimeline';
import { MetricCard } from './widgets/MetricCard';
import { ChannelList } from './widgets/ChannelList';
import { ActivityFeed } from './widgets/ActivityFeed';

export const widgetRegistry = {
  // Metric widgets
  'metric-revenue': {
    component: MetricCard,
    config: {
      id: 'metric-revenue',
      title: 'Total Revenue',
      type: 'metric',
      defaultSize: { w: 3, h: 1 }
    }
  },
  
  'metric-videos': {
    component: MetricCard,
    config: {
      id: 'metric-videos',
      title: 'Videos Generated',
      type: 'metric',
      defaultSize: { w: 3, h: 1 }
    }
  },
  
  'metric-cost': {
    component: MetricCard,
    config: {
      id: 'metric-cost',
      title: 'Cost per Video',
      type: 'metric',
      defaultSize: { w: 3, h: 1 }
    }
  },
  
  'metric-automation': {
    component: MetricCard,
    config: {
      id: 'metric-automation',
      title: 'Automation Rate',
      type: 'metric',
      defaultSize: { w: 3, h: 1 }
    }
  },
  
  // Chart widgets (5 total for MVP)
  'chart-revenue-trend': {
    component: RevenueTrendChart,
    config: {
      id: 'chart-revenue-trend',
      title: 'Revenue Trend',
      type: 'chart',
      defaultSize: { w: 8, h: 3 },
      refreshable: true,
      exportable: true
    }
  },
  
  'chart-channel-performance': {
    component: ChannelPerformanceChart,
    config: {
      id: 'chart-channel-performance',
      title: 'Channel Performance',
      type: 'chart',
      defaultSize: { w: 6, h: 3 },
      refreshable: true,
      exportable: true
    }
  },
  
  'chart-cost-breakdown': {
    component: CostBreakdownChart,
    config: {
      id: 'chart-cost-breakdown',
      title: 'Cost Breakdown',
      type: 'chart',
      defaultSize: { w: 4, h: 3 },
      refreshable: true,
      exportable: true
    }
  },
  
  'chart-video-timeline': {
    component: VideoGenerationTimeline,
    config: {
      id: 'chart-video-timeline',
      title: 'Generation Timeline',
      type: 'chart',
      defaultSize: { w: 12, h: 2 },
      refreshable: true,
      exportable: true
    }
  },
  
  // List widgets
  'list-channels': {
    component: ChannelList,
    config: {
      id: 'list-channels',
      title: 'Channel Overview',
      type: 'list',
      defaultSize: { w: 12, h: 3 },
      refreshable: true
    }
  },
  
  'list-activity': {
    component: ActivityFeed,
    config: {
      id: 'list-activity',
      title: 'Recent Activity',
      type: 'list',
      defaultSize: { w: 4, h: 3 },
      refreshable: false // Real-time via WebSocket
    }
  }
};
```

---

## 4. Performance Metrics Display

### 4.1 Key Performance Indicators

```typescript
// PerformanceMetrics.tsx - Display system performance
import React from 'react';
import { Grid, Box, Typography, LinearProgress, Chip } from '@mui/material';
import { MetricCard } from './MetricCard';

interface SystemMetrics {
  apiLatency: number;
  videoGenerationTime: number;
  uploadSuccessRate: number;
  costPerVideo: number;
  queueDepth: number;
  activeChannels: number;
}

export const PerformanceMetrics: React.FC<{ metrics: SystemMetrics }> = ({
  metrics
}) => {
  const getLatencyColor = (latency: number) => {
    if (latency < 100) return 'success';
    if (latency < 500) return 'warning';
    return 'error';
  };

  const getSuccessRateColor = (rate: number) => {
    if (rate >= 95) return 'success';
    if (rate >= 90) return 'warning';
    return 'error';
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            API Response Time
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <LinearProgress
              variant="determinate"
              value={Math.min((metrics.apiLatency / 1000) * 100, 100)}
              sx={{ flex: 1, height: 8 }}
              color={getLatencyColor(metrics.apiLatency)}
            />
            <Chip
              label={`${metrics.apiLatency}ms`}
              size="small"
              color={getLatencyColor(metrics.apiLatency)}
            />
          </Box>
        </Box>
      </Grid>

      <Grid item xs={12} md={6}>
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Upload Success Rate
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <LinearProgress
              variant="determinate"
              value={metrics.uploadSuccessRate}
              sx={{ flex: 1, height: 8 }}
              color={getSuccessRateColor(metrics.uploadSuccessRate)}
            />
            <Chip
              label={`${metrics.uploadSuccessRate}%`}
              size="small"
              color={getSuccessRateColor(metrics.uploadSuccessRate)}
            />
          </Box>
        </Box>
      </Grid>

      <Grid item xs={12} md={4}>
        <MetricCard
          title="Avg Generation Time"
          value={`${Math.round(metrics.videoGenerationTime)}s`}
          format="string"
          color={metrics.videoGenerationTime < 300 ? 'success' : 'warning'}
        />
      </Grid>

      <Grid item xs={12} md={4}>
        <MetricCard
          title="Queue Depth"
          value={metrics.queueDepth}
          format="number"
          color={metrics.queueDepth < 10 ? 'success' : 'warning'}
        />
      </Grid>

      <Grid item xs={12} md={4}>
        <MetricCard
          title="Cost per Video"
          value={metrics.costPerVideo}
          format="currency"
          color={metrics.costPerVideo < 0.40 ? 'success' : 'error'}
          change={-5.2} // Example: 5.2% decrease
        />
      </Grid>
    </Grid>
  );
};
```

### 4.2 Real-time Performance Monitor

```typescript
// RealtimePerformanceMonitor.tsx - Live performance tracking
import React, { useState, useEffect } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface PerformanceDataPoint {
  timestamp: string;
  apiLatency: number;
  queueSize: number;
  activeVideos: number;
}

export const RealtimePerformanceMonitor: React.FC = () => {
  const [data, setData] = useState<PerformanceDataPoint[]>([]);
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    // Simulate real-time data updates
    const interval = setInterval(() => {
      const newPoint: PerformanceDataPoint = {
        timestamp: new Date().toLocaleTimeString(),
        apiLatency: Math.random() * 200 + 50,
        queueSize: Math.floor(Math.random() * 20),
        activeVideos: Math.floor(Math.random() * 5)
      };

      setData(prev => {
        const updated = [...prev, newPoint];
        // Keep only last 20 points for performance
        return updated.slice(-20);
      });

      // Check for performance issues
      if (newPoint.apiLatency > 500) {
        setAlerts(prev => [...prev, `High API latency: ${newPoint.apiLatency}ms`]);
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <Box>
      {alerts.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setAlerts([])}>
          {alerts[alerts.length - 1]}
        </Alert>
      )}

      <LineChart width={600} height={200} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="timestamp" />
        <YAxis />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="apiLatency"
          stroke={chartTheme.colors.primary}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false} // Disable animation for real-time
        />
      </LineChart>
    </Box>
  );
};
```

---

## 5. Export Functionality Specifications

### 5.1 Data Export Service

```typescript
// exportService.ts - Handle data exports
import { format } from 'date-fns';
import { saveAs } from 'file-saver';

export class ExportService {
  static async exportToCSV(data: any[], filename: string): Promise<void> {
    if (!data || data.length === 0) {
      throw new Error('No data to export');
    }

    // Get headers from first object
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => {
          const value = row[header];
          // Escape quotes and wrap in quotes if contains comma
          if (typeof value === 'string' && value.includes(',')) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, `${filename}_${format(new Date(), 'yyyy-MM-dd_HH-mm')}.csv`);
  }

  static async exportToJSON(data: any, filename: string): Promise<void> {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, `${filename}_${format(new Date(), 'yyyy-MM-dd_HH-mm')}.json`);
  }

  static async exportChartToImage(
    chartElement: HTMLElement,
    filename: string
  ): Promise<void> {
    try {
      const { html2canvas } = await import('html2canvas');
      const canvas = await html2canvas(chartElement);
      canvas.toBlob(blob => {
        if (blob) {
          saveAs(blob, `${filename}_${format(new Date(), 'yyyy-MM-dd_HH-mm')}.png`);
        }
      });
    } catch (error) {
      console.error('Failed to export chart:', error);
      throw new Error('Chart export failed');
    }
  }

  static async exportDashboardReport(dashboardData: any): Promise<void> {
    const report = {
      generatedAt: new Date().toISOString(),
      period: dashboardData.period,
      summary: {
        totalRevenue: dashboardData.metrics.totalRevenue,
        totalVideos: dashboardData.metrics.totalVideos,
        avgCostPerVideo: dashboardData.metrics.avgCostPerVideo,
        activeChannels: dashboardData.channels.filter(ch => ch.status === 'active').length
      },
      channels: dashboardData.channels,
      performance: dashboardData.performance,
      costs: dashboardData.costs
    };

    await this.exportToJSON(report, 'dashboard-report');
  }
}
```

### 5.2 Export Component

```typescript
// ExportButton.tsx - Reusable export button
import React, { useState } from 'react';
import {
  Button,
  Menu,
  MenuItem,
  CircularProgress,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ImageIcon from '@mui/icons-material/Image';
import TableChartIcon from '@mui/icons-material/TableChart';
import DescriptionIcon from '@mui/icons-material/Description';
import { ExportService } from '@/services/exportService';

interface ExportButtonProps {
  data: any;
  filename: string;
  chartRef?: React.RefObject<HTMLElement>;
  formats?: ('csv' | 'json' | 'image' | 'report')[];
}

export const ExportButton: React.FC<ExportButtonProps> = ({
  data,
  filename,
  chartRef,
  formats = ['csv', 'json']
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [exporting, setExporting] = useState(false);

  const handleExport = async (format: string) => {
    setExporting(true);
    try {
      switch (format) {
        case 'csv':
          await ExportService.exportToCSV(data, filename);
          break;
        case 'json':
          await ExportService.exportToJSON(data, filename);
          break;
        case 'image':
          if (chartRef?.current) {
            await ExportService.exportChartToImage(chartRef.current, filename);
          }
          break;
        case 'report':
          await ExportService.exportDashboardReport(data);
          break;
      }
      
      // Show success notification
      showNotification({
        type: 'success',
        message: `Exported as ${format.toUpperCase()}`
      });
    } catch (error) {
      showNotification({
        type: 'error',
        message: 'Export failed'
      });
    } finally {
      setExporting(false);
      setAnchorEl(null);
    }
  };

  return (
    <>
      <Button
        startIcon={exporting ? <CircularProgress size={16} /> : <DownloadIcon />}
        onClick={(e) => setAnchorEl(e.currentTarget)}
        disabled={exporting}
        size="small"
      >
        Export
      </Button>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        {formats.includes('csv') && (
          <MenuItem onClick={() => handleExport('csv')}>
            <ListItemIcon><TableChartIcon /></ListItemIcon>
            <ListItemText>Export as CSV</ListItemText>
          </MenuItem>
        )}
        
        {formats.includes('json') && (
          <MenuItem onClick={() => handleExport('json')}>
            <ListItemIcon><DescriptionIcon /></ListItemIcon>
            <ListItemText>Export as JSON</ListItemText>
          </MenuItem>
        )}
        
        {formats.includes('image') && chartRef && (
          <MenuItem onClick={() => handleExport('image')}>
            <ListItemIcon><ImageIcon /></ListItemIcon>
            <ListItemText>Export as Image</ListItemText>
          </MenuItem>
        )}
        
        {formats.includes('report') && (
          <MenuItem onClick={() => handleExport('report')}>
            <ListItemIcon><DescriptionIcon /></ListItemIcon>
            <ListItemText>Export Full Report</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </>
  );
};
```

---

## 6. Performance Optimization Tips

### 6.1 Chart Performance Guidelines

```typescript
// Chart optimization patterns
const chartOptimizationRules = {
  // Data point limits
  dataLimits: {
    lineChart: 100,
    barChart: 50,
    pieChart: 8,
    areaChart: 100
  },
  
  // Animation rules
  animationRules: {
    enableWhen: 'dataPoints < 50',
    duration: 1500,
    easing: 'ease-out'
  },
  
  // Render optimization
  renderOptimization: {
    useMemo: 'Always for data processing',
    useCallback: 'For event handlers',
    React.memo: 'For chart components'
  },
  
  // Update strategies
  updateStrategies: {
    partial: 'Update only changed data points',
    batch: 'Group updates within 100ms window',
    throttle: 'Limit updates to once per second'
  }
};

// Example: Optimized chart data hook
export const useChartData = (rawData: any[], chartType: string) => {
  return useMemo(() => {
    const limit = chartOptimizationRules.dataLimits[chartType] || 100;
    
    // Reduce data points if necessary
    if (rawData.length > limit) {
      const step = Math.ceil(rawData.length / limit);
      return rawData.filter((_, index) => index % step === 0);
    }
    
    return rawData;
  }, [rawData, chartType]);
};
```

### 6.2 Bundle Size Management

```typescript
// Import only what you need from Recharts
// Good ✅
import { LineChart, Line, XAxis, YAxis } from 'recharts';

// Bad ❌
import * as Recharts from 'recharts';

// Tree-shakeable imports configuration
// vite.config.ts
export default {
  optimizeDeps: {
    include: ['recharts'],
    esbuildOptions: {
      target: 'es2020'
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'recharts': ['recharts']
        }
      }
    }
  }
};
```

---

## 7. Testing Visualizations

### 7.1 Chart Testing Strategy

```typescript
// __tests__/charts/RevenueTrendChart.test.tsx
import { render, screen } from '@testing-library/react';
import { RevenueTrendChart } from '@/components/charts/RevenueTrendChart';

const mockData = [
  { date: '2024-01-01', revenue: 100, cost: 20, profit: 80 },
  { date: '2024-01-02', revenue: 150, cost: 30, profit: 120 },
  { date: '2024-01-03', revenue: 200, cost: 40, profit: 160 }
];

describe('RevenueTrendChart', () => {
  it('renders without crashing', () => {
    render(
      <div style={{ width: 800, height: 400 }}>
        <RevenueTrendChart data={mockData} period="day" />
      </div>
    );
  });

  it('displays correct data series', () => {
    const { container } = render(
      <div style={{ width: 800, height: 400 }}>
        <RevenueTrendChart data={mockData} period="day" />
      </div>
    );
    
    // Check for line elements
    const lines = container.querySelectorAll('.recharts-line');
    expect(lines).toHaveLength(3); // Revenue, Cost, Profit
  });

  it('handles empty data gracefully', () => {
    render(
      <div style={{ width: 800, height: 400 }}>
        <RevenueTrendChart data={[]} period="day" />
      </div>
    );
    
    // Should render empty chart without errors
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  it('limits data points to 100', () => {
    const largeDataset = Array(200).fill(null).map((_, i) => ({
      date: `2024-01-${i + 1}`,
      revenue: Math.random() * 1000,
      cost: Math.random() * 200,
      profit: Math.random() * 800
    }));

    const { container } = render(
      <div style={{ width: 800, height: 400 }}>
        <RevenueTrendChart data={largeDataset} period="month" />
      </div>
    );
    
    // Check that data points are reduced
    const dots = container.querySelectorAll('.recharts-dot');
    expect(dots.length).toBeLessThanOrEqual(300); // 100 points × 3 lines
  });
});
```

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Set up Recharts with proper imports
- [ ] Create base ChartContainer component
- [ ] Implement chart theme and colors
- [ ] Build MetricCard component
- [ ] Create DashboardWidget wrapper

### Week 2: Core Charts
- [ ] Implement RevenueTrendChart (Line)
- [ ] Build ChannelPerformanceChart (Bar)
- [ ] Create CostBreakdownChart (Pie)
- [ ] Add VideoGenerationTimeline (Area)
- [ ] Implement 5th chart (your choice based on needs)

### Week 3: Integration & Polish
- [ ] Connect charts to real data via Zustand
- [ ] Implement export functionality
- [ ] Add loading and error states
- [ ] Optimize performance (memoization)
- [ ] Complete testing suite

### Week 4: Final Testing
- [ ] Performance testing (<500ms render)
- [ ] Bundle size verification (<1MB total)
- [ ] Cross-browser testing
- [ ] Export functionality testing
- [ ] Final polish and documentation

---

## Summary

This guide provides everything needed to implement a performant, user-friendly dashboard visualization system using Recharts. Remember:

1. **Stick to Recharts only** - no D3.js or Plotly
2. **Limit to 5-7 charts total** across the application
3. **Keep data points under 100** per chart
4. **Disable animations** for large datasets
5. **Export functionality** is critical for users
6. **Performance over beauty** - fast renders are key

**Questions?** Reach out to the Frontend Team Lead or refer to the architecture documentation.

**Next Steps**: Review all three documents, set up your development environment, and begin with the base chart components. Focus on performance and simplicity!