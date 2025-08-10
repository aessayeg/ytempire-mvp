# Data Visualization Standards & Recharts Implementation

**Version**: 1.0  
**Date**: January 2025  
**For**: Dashboard Specialist  
**From**: Frontend Team Lead  
**Focus**: Recharts-only implementation for MVP

---

## 1. Data Visualization Standards

### 1.1 Core Visualization Principles

```typescript
const VisualizationPrinciples = {
  // MVP Constraints
  library: "Recharts ONLY",
  totalCharts: "5-7 maximum across entire app",
  dataPoints: "Max 100 per chart",
  animations: "Disable for >50 points",
  
  // Design Principles
  clarity: "Data clarity over decoration",
  consistency: "Same chart type for same data type",
  performance: "Render time < 500ms",
  accessibility: "ARIA labels and keyboard navigation"
};
```

### 1.2 Chart Type Selection Guide

```typescript
const ChartSelectionGuide = {
  // Time-series data → Line Chart
  timeSeriesData: {
    chartType: 'LineChart',
    use_cases: [
      'Revenue over time',
      'Video generation rate',
      'Cost trends',
      'Channel growth'
    ],
    max_series: 3,  // Max 3 lines per chart
    example: 'ChannelPerformanceChart'
  },
  
  // Comparison data → Bar Chart
  comparisonData: {
    chartType: 'BarChart',
    use_cases: [
      'Channel comparison',
      'Daily video count',
      'Cost by category'
    ],
    max_bars: 10,  // Max 10 bars visible
    example: 'ChannelComparisonChart'
  },
  
  // Composition data → Pie Chart
  compositionData: {
    chartType: 'PieChart',
    use_cases: [
      'Cost breakdown',
      'Channel distribution',
      'Revenue sources'
    ],
    max_slices: 6,  // Max 6 slices, rest in "Other"
    example: 'CostBreakdownChart'
  },
  
  // Trends → Area Chart
  trendData: {
    chartType: 'AreaChart',
    use_cases: [
      'Cumulative revenue',
      'Total videos generated',
      'Growth projection'
    ],
    max_areas: 2,  // Max 2 stacked areas
    example: 'RevenueGrowthChart'
  }
};
```

### 1.3 Color Palette & Theming

```typescript
// Consistent color palette for all charts
const ChartColorPalette = {
  primary: '#2196F3',    // Primary blue
  secondary: '#FF9800',  // Orange
  success: '#4CAF50',    // Green
  warning: '#FFC107',    // Amber
  error: '#F44336',      // Red
  info: '#00BCD4',       // Cyan
  
  // Chart-specific colors
  series: [
    '#2196F3',  // Series 1
    '#FF9800',  // Series 2
    '#4CAF50',  // Series 3
    '#9C27B0',  // Series 4
    '#00BCD4'   // Series 5
  ],
  
  // Semantic colors for specific metrics
  revenue: '#4CAF50',    // Green for money
  cost: '#F44336',       // Red for costs
  neutral: '#9E9E9E',    // Gray for neutral
  
  // Gradients for area charts
  gradients: {
    revenue: ['#4CAF50', '#81C784'],
    cost: ['#F44336', '#EF5350'],
    primary: ['#2196F3', '#64B5F6']
  }
};
```

---

## 2. Recharts Implementation Standards

### 2.1 Base Chart Component Template

```tsx
import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { useTheme } from '@mui/material';

interface BaseChartProps {
  data: any[];
  dataKey: string;
  xAxisKey?: string;
  height?: number;
  animate?: boolean;
}

const BaseLineChart: React.FC<BaseChartProps> = ({
  data,
  dataKey,
  xAxisKey = 'timestamp',
  height = 300,
  animate = true
}) => {
  const theme = useTheme();
  
  // Optimize data for performance
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Limit to 100 points
    if (data.length > 100) {
      return simplifyDataPoints(data, 100);
    }
    
    return data;
  }, [data]);
  
  // Disable animation for large datasets
  const animationDuration = chartData.length > 50 || !animate ? 0 : 1500;
  
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={chartData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke={theme.palette.divider}
        />
        <XAxis 
          dataKey={xAxisKey}
          stroke={theme.palette.text.secondary}
          tick={{ fontSize: 12 }}
        />
        <YAxis 
          stroke={theme.palette.text.secondary}
          tick={{ fontSize: 12 }}
        />
        <Tooltip 
          contentStyle={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`
          }}
        />
        <Legend 
          wrapperStyle={{ fontSize: 12 }}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={ChartColorPalette.primary}
          strokeWidth={2}
          dot={chartData.length < 20}
          animationDuration={animationDuration}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
```

### 2.2 MVP Chart Components (5-7 Total)

#### Chart 1: Channel Performance (Line Chart)

```tsx
const ChannelPerformanceChart: React.FC = () => {
  const { channels, dateRange } = useDashboardStore();
  
  const data = useMemo(() => {
    // Transform channel data for chart
    return channels.map(channel => ({
      name: channel.name,
      videos: channel.videoCount,
      revenue: channel.revenue,
      timestamp: channel.lastUpdate
    }));
  }, [channels]);
  
  return (
    <WidgetContainer 
      id="channel-performance" 
      title="Channel Performance"
      refreshInterval={60000}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="videos" 
            stroke={ChartColorPalette.primary}
            name="Videos"
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="revenue" 
            stroke={ChartColorPalette.revenue}
            name="Revenue ($)"
          />
        </LineChart>
      </ResponsiveContainer>
    </WidgetContainer>
  );
};
```

#### Chart 2: Cost Breakdown (Pie Chart)

```tsx
const CostBreakdownChart: React.FC = () => {
  const { costs } = useDashboardStore();
  
  const data = useMemo(() => {
    if (!costs) return [];
    
    return [
      { name: 'AI Generation', value: costs.ai_generation, fill: ChartColorPalette.series[0] },
      { name: 'Voice Synthesis', value: costs.voice_synthesis, fill: ChartColorPalette.series[1] },
      { name: 'Storage', value: costs.storage, fill: ChartColorPalette.series[2] },
      { name: 'API Calls', value: costs.api_calls, fill: ChartColorPalette.series[3] }
    ].filter(item => item.value > 0);  // Only show non-zero costs
  }, [costs]);
  
  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({
    cx, cy, midAngle, innerRadius, outerRadius, percent
  }) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    
    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };
  
  return (
    <WidgetContainer 
      id="cost-breakdown" 
      title="Cost Breakdown"
      refreshInterval={30000}
    >
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomizedLabel}
            outerRadius={80}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </WidgetContainer>
  );
};
```

#### Chart 3: Daily Video Generation (Bar Chart)

```tsx
const DailyVideoGenerationChart: React.FC = () => {
  const { metrics } = useDashboardStore();
  
  const data = useMemo(() => {
    // Last 7 days of video generation
    const last7Days = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      last7Days.push({
        day: date.toLocaleDateString('en', { weekday: 'short' }),
        generated: Math.floor(Math.random() * 15) + 5,  // Mock data
        failed: Math.floor(Math.random() * 3)
      });
    }
    return last7Days;
  }, [metrics]);
  
  return (
    <WidgetContainer 
      id="daily-generation" 
      title="Daily Video Generation"
      refreshInterval={60000}
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="generated" fill={ChartColorPalette.success} name="Completed" />
          <Bar dataKey="failed" fill={ChartColorPalette.error} name="Failed" />
        </BarChart>
      </ResponsiveContainer>
    </WidgetContainer>
  );
};
```

#### Chart 4: Revenue Trend (Area Chart)

```tsx
const RevenueTrendChart: React.FC = () => {
  const { metrics } = useDashboardStore();
  
  const data = useMemo(() => {
    // Generate 30-day revenue trend
    const trend = [];
    let cumulativeRevenue = 0;
    
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const dailyRevenue = Math.random() * 100 + 50;
      cumulativeRevenue += dailyRevenue;
      
      trend.push({
        date: date.toLocaleDateString('en', { month: 'short', day: 'numeric' }),
        daily: dailyRevenue,
        cumulative: cumulativeRevenue
      });
    }
    
    return trend;
  }, [metrics]);
  
  return (
    <WidgetContainer 
      id="revenue-trend" 
      title="Revenue Trend (30 Days)"
      refreshInterval={60000}
    >
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={ChartColorPalette.revenue} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={ChartColorPalette.revenue} stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Area 
            type="monotone" 
            dataKey="cumulative" 
            stroke={ChartColorPalette.revenue}
            fillOpacity={1}
            fill="url(#colorRevenue)"
            name="Cumulative Revenue ($)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </WidgetContainer>
  );
};
```

### 2.3 Chart Utilities & Helpers

```typescript
// Data simplification for performance
function simplifyDataPoints(data: any[], targetPoints: number): any[] {
  if (data.length <= targetPoints) return data;
  
  const interval = Math.floor(data.length / targetPoints);
  const simplified = [];
  
  for (let i = 0; i < data.length; i += interval) {
    simplified.push(data[i]);
  }
  
  // Always include the last point
  if (simplified[simplified.length - 1] !== data[data.length - 1]) {
    simplified.push(data[data.length - 1]);
  }
  
  return simplified;
}

// Format numbers for display
function formatChartValue(value: number, type: 'currency' | 'number' | 'percentage'): string {
  switch (type) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
      }).format(value);
      
    case 'percentage':
      return `${(value * 100).toFixed(1)}%`;
      
    case 'number':
    default:
      return new Intl.NumberFormat('en-US').format(value);
  }
}

// Custom tooltip component
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  
  return (
    <Paper sx={{ p: 1.5 }}>
      <Typography variant="caption" color="textSecondary">
        {label}
      </Typography>
      {payload.map((entry, index) => (
        <Typography key={index} variant="body2" style={{ color: entry.color }}>
          {entry.name}: {formatChartValue(entry.value, 'currency')}
        </Typography>
      ))}
    </Paper>
  );
};
```

---

## 3. Performance Metrics Display

### 3.1 Key Performance Indicators (KPIs)

```tsx
interface KPICardProps {
  title: string;
  value: number | string;
  change?: number;
  changeLabel?: string;
  icon?: React.ReactNode;
  format?: 'currency' | 'number' | 'percentage';
  color?: 'primary' | 'success' | 'warning' | 'error';
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  change,
  changeLabel,
  icon,
  format = 'number',
  color = 'primary'
}) => {
  const formattedValue = useMemo(() => {
    if (typeof value === 'string') return value;
    return formatChartValue(value, format);
  }, [value, format]);
  
  const changeColor = change && change > 0 ? 'success' : 'error';
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box flex={1}>
            <Typography color="textSecondary" variant="subtitle2" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" component="div" color={`${color}.main`}>
              {formattedValue}
            </Typography>
            {change !== undefined && (
              <Box display="flex" alignItems="center" mt={1}>
                <Typography variant="body2" color={`${changeColor}.main`}>
                  {change > 0 ? '+' : ''}{change.toFixed(1)}%
                </Typography>
                {changeLabel && (
                  <Typography variant="body2" color="textSecondary" ml={1}>
                    {changeLabel}
                  </Typography>
                )}
              </Box>
            )}
          </Box>
          {icon && (
            <Box color={`${color}.main`}>
              {icon}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};
```

### 3.2 Real-time Metrics Dashboard

```tsx
const MetricsDashboard: React.FC = () => {
  const { metrics, costs } = useDashboardStore();
  
  return (
    <Grid container spacing={2}>
      {/* Row 1: Primary KPIs */}
      <Grid item xs={3}>
        <KPICard
          title="Active Channels"
          value={metrics?.activeChannels || 0}
          format="number"
          icon={<ChannelIcon />}
          color="primary"
        />
      </Grid>
      
      <Grid item xs={3}>
        <KPICard
          title="Videos Today"
          value={metrics?.videosToday || 0}
          change={12.5}
          changeLabel="vs yesterday"
          format="number"
          icon={<VideoIcon />}
          color="success"
        />
      </Grid>
      
      <Grid item xs={3}>
        <KPICard
          title="Daily Revenue"
          value={metrics?.dailyRevenue || 0}
          change={8.3}
          changeLabel="vs yesterday"
          format="currency"
          icon={<RevenueIcon />}
          color="success"
        />
      </Grid>
      
      <Grid item xs={3}>
        <KPICard
          title="Cost per Video"
          value={costs?.costPerVideo || 0}
          change={-5.2}
          changeLabel="vs average"
          format="currency"
          icon={<CostIcon />}
          color={costs?.costPerVideo > 0.45 ? 'warning' : 'primary'}
        />
      </Grid>
      
      {/* Row 2: Charts */}
      <Grid item xs={8}>
        <ChannelPerformanceChart />
      </Grid>
      
      <Grid item xs={4}>
        <CostBreakdownChart />
      </Grid>
      
      {/* Row 3: Additional Charts */}
      <Grid item xs={6}>
        <DailyVideoGenerationChart />
      </Grid>
      
      <Grid item xs={6}>
        <RevenueTrendChart />
      </Grid>
    </Grid>
  );
};
```

---

## 4. Chart Accessibility Standards

### 4.1 ARIA Labels and Descriptions

```tsx
const AccessibleChart: React.FC = ({ data, title, description }) => {
  return (
    <div 
      role="img" 
      aria-label={title}
      aria-describedby="chart-description"
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          {/* Chart content */}
        </LineChart>
      </ResponsiveContainer>
      
      {/* Screen reader description */}
      <div id="chart-description" className="sr-only">
        {description || `Chart showing ${title} with ${data.length} data points`}
      </div>
      
      {/* Keyboard-accessible data table alternative */}
      <details>
        <summary>View data table</summary>
        <DataTable data={data} />
      </details>
    </div>
  );
};
```

### 4.2 Color Contrast Standards

```typescript
// Ensure WCAG AA compliance (4.5:1 contrast ratio)
const AccessibleColors = {
  // High contrast colors for charts
  primary: '#1976D2',    // Darker blue
  secondary: '#F57C00',  // Darker orange
  success: '#388E3C',    // Darker green
  error: '#D32F2F',      // Darker red
  
  // Patterns for colorblind users
  patterns: [
    'solid',
    'dashed',
    'dotted',
    'dashdot'
  ]
};
```

---

## 5. Testing Standards for Charts

### 5.1 Unit Testing Charts

```tsx
import { render, screen } from '@testing-library/react';
import { ChannelPerformanceChart } from './ChannelPerformanceChart';

describe('ChannelPerformanceChart', () => {
  const mockData = [
    { name: 'Channel 1', videos: 10, revenue: 100 },
    { name: 'Channel 2', videos: 15, revenue: 150 }
  ];
  
  it('renders without crashing', () => {
    render(<ChannelPerformanceChart data={mockData} />);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });
  
  it('displays correct number of data points', () => {
    const { container } = render(<ChannelPerformanceChart data={mockData} />);
    const lines = container.querySelectorAll('.recharts-line');
    expect(lines).toHaveLength(2); // Videos and Revenue lines
  });
  
  it('limits data points to 100', () => {
    const largeData = Array(150).fill(null).map((_, i) => ({
      name: `Point ${i}`,
      value: i
    }));
    
    const { container } = render(<ChannelPerformanceChart data={largeData} />);
    const dots = container.querySelectorAll('.recharts-dot');
    expect(dots.length).toBeLessThanOrEqual(100);
  });
});
```

---

## MVP Chart Summary

### Total Charts: 5-7 (Within MVP Limit)

1. **Channel Performance** (Line) - Overview dashboard
2. **Cost Breakdown** (Pie) - Overview dashboard
3. **Daily Generation** (Bar) - Overview dashboard
4. **Revenue Trend** (Area) - Channel analytics
5. **Video Queue Status** (Bar) - Real-time view

### Optional Charts (if under 7 total):
6. **Channel Comparison** (Bar) - Comparative view
7. **Success Rate** (Line) - Quality metrics

### Performance Targets:
- Initial render: < 500ms
- Data update: < 100ms
- Animation: Disabled for >50 points
- Memory usage: < 50MB for all charts

**Remember**: Stay within Recharts capabilities. No D3.js, no Plotly, no custom Canvas operations. Focus on clear, performant visualizations that help users make decisions quickly.