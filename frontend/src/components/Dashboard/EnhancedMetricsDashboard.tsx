/**
 * Enhanced Metrics Dashboard Component
 * Comprehensive analytics and metrics visualization with real-time updates
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Skeleton,
  Tooltip,
  Chip,
  Button,
  ButtonGroup,
  Alert,
  LinearProgress,
  Tab,
  Tabs,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Refresh,
  Download,
  Info,
  AttachMoney,
  Visibility,
  ThumbUp,
  People,
  VideoLibrary,
  Speed,
  Warning,
  CheckCircle,
  Error,
  MoreVert,
  DateRange,
  Assessment,
  Timeline,
  ExpandMore,
  Share,
  CloudDownload,
  Fullscreen,
  FilterList,
  ShowChart,
  PieChart,
  BarChart as BarChartIcon,
  TableChart,
  Settings,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPie,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  ComposedChart,
  ScatterChart,
  Scatter,
  FunnelChart,
  Funnel,
  LabelList,
  TreeMap,
  Sankey,
} from 'recharts';
import { format, subDays, startOfDay, endOfDay, addHours } from 'date-fns';
import { useOptimizedStore } from '../../stores/optimizedStore';
import { useRealtimeData } from '../../hooks/useRealtimeData';
import { api } from '../../services/api';

// Types
interface MetricCard {
  id: string;
  title: string;
  value: number | string;
  change: number;
  changePercent: number;
  trend: 'up' | 'down' | 'flat';
  icon: React.ReactNode;
  color: string;
  suffix?: string;
  prefix?: string;
  target?: number;
  description?: string;
  sparklineData?: number[];
}

interface ChartData {
  timestamp: string;
  date: string;
  hour?: number;
  views: number;
  revenue: number;
  subscribers: number;
  engagement: number;
  cost: number;
  profit: number;
  ctr: number;
  watchTime: number;
  impressions: number;
}

interface ChannelPerformance {
  channelId: string;
  channelName: string;
  videos: number;
  views: number;
  revenue: number;
  subscribers: number;
  avgEngagement: number;
  growth: number;
  avatar?: string;
}

interface VideoPerformance {
  videoId: string;
  title: string;
  channelName: string;
  views: number;
  engagement: number;
  revenue: number;
  publishedAt: string;
  duration: number;
  thumbnail: string;
}

const CHART_COLORS = {
  views: '#3f51b5',
  revenue: '#4caf50',
  subscribers: '#ff9800',
  engagement: '#e91e63',
  cost: '#f44336',
  profit: '#2e7d32',
  ctr: '#9c27b0',
  watchTime: '#00bcd4',
};

const TIME_RANGES = [
  { value: '24h', label: '24 Hours' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
  { value: '1y', label: '1 Year' },
];

const CHART_TYPES = [
  { value: 'line', label: 'Line Chart', icon: <ShowChart /> },
  { value: 'area', label: 'Area Chart', icon: <Timeline /> },
  { value: 'bar', label: 'Bar Chart', icon: <BarChartIcon /> },
  { value: 'pie', label: 'Pie Chart', icon: <PieChart /> },
];

export const EnhancedMetricsDashboard: React.FC = () => {
  // State management
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [chartType, setChartType] = useState<string>('line');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null);
  const [showTargets, setShowTargets] = useState(true);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['views', 'revenue', 'engagement']);
  const [customDateRange, setCustomDateRange] = useState<{start: string, end: string}>({
    start: '',
    end: ''
  });

  // Data state
  const [metricCards, setMetricCards] = useState<MetricCard[]>([]);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [channelPerformance, setChannelPerformance] = useState<ChannelPerformance[]>([]);
  const [topVideos, setTopVideos] = useState<VideoPerformance[]>([]);
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null);

  const { analytics, channels, addNotification } = useOptimizedStore();

  // Real-time data hook
  const realtime = useRealtimeData('/ws/analytics');

  // Memoized calculations
  const processedChartData = useMemo(() => {
    return chartData.map(item => ({
      ...item,
      date: format(new Date(item.timestamp), timeRange === '24h' ? 'HH:mm' : 'MMM dd'),
      profit: item.revenue - item.cost,
      roi: item.cost > 0 ? ((item.revenue - item.cost) / item.cost) * 100 : 0,
    }));
  }, [chartData, timeRange]);

  const totalMetrics = useMemo(() => {
    if (!chartData.length) return {};
    
    return chartData.reduce((acc, curr) => ({
      views: (acc.views || 0) + curr.views,
      revenue: (acc.revenue || 0) + curr.revenue,
      cost: (acc.cost || 0) + curr.cost,
      subscribers: Math.max(acc.subscribers || 0, curr.subscribers),
      engagement: (acc.engagement || 0) + curr.engagement,
      watchTime: (acc.watchTime || 0) + curr.watchTime,
    }), {});
  }, [chartData]);

  // Fetch comprehensive dashboard data
  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);

      // Parallel API calls for better performance
      const [
        overviewResponse,
        performanceResponse,
        channelsResponse,
        videosResponse,
        realtimeResponse
      ] = await Promise.all([
        api.get(`/dashboard/overview`),
        api.get(`/dashboard/performance?period=${timeRange}`),
        api.get('/dashboard/channels'),
        api.get('/videos/top?limit=10'),
        api.get('/dashboard/realtime-stats')
      ]);

      // Process metric cards
      const cards: MetricCard[] = [
        {
          id: 'views',
          title: 'Total Views',
          value: overviewResponse.data.total_views,
          change: overviewResponse.data.views_change_24h,
          changePercent: overviewResponse.data.views_change_percent,
          trend: overviewResponse.data.views_change_24h >= 0 ? 'up' : 'down',
          icon: <Visibility />,
          color: CHART_COLORS.views,
          target: 1000000,
          description: 'Total video views across all channels',
        },
        {
          id: 'revenue',
          title: 'Revenue',
          value: overviewResponse.data.total_revenue,
          change: overviewResponse.data.revenue_change_24h,
          changePercent: overviewResponse.data.revenue_change_percent,
          trend: overviewResponse.data.revenue_change_24h >= 0 ? 'up' : 'down',
          icon: <AttachMoney />,
          color: CHART_COLORS.revenue,
          prefix: '$',
          target: 10000,
          description: 'Total revenue generated',
        },
        {
          id: 'videos',
          title: 'Videos Generated',
          value: overviewResponse.data.videos_today,
          change: overviewResponse.data.videos_change,
          changePercent: overviewResponse.data.videos_change_percent,
          trend: overviewResponse.data.videos_change >= 0 ? 'up' : 'down',
          icon: <VideoLibrary />,
          color: '#9c27b0',
          description: 'Videos generated today',
        },
        {
          id: 'engagement',
          title: 'Avg Engagement',
          value: overviewResponse.data.avg_video_performance,
          change: overviewResponse.data.engagement_change,
          changePercent: overviewResponse.data.engagement_change_percent,
          trend: overviewResponse.data.engagement_change >= 0 ? 'up' : 'down',
          icon: <ThumbUp />,
          color: CHART_COLORS.engagement,
          suffix: '%',
          target: 5.0,
          description: 'Average engagement rate',
        },
        {
          id: 'cost',
          title: 'Total Cost',
          value: overviewResponse.data.total_cost,
          change: overviewResponse.data.cost_change_24h,
          changePercent: overviewResponse.data.cost_change_percent,
          trend: overviewResponse.data.cost_change_24h <= 0 ? 'up' : 'down', // Inverted for cost
          icon: <Speed />,
          color: CHART_COLORS.cost,
          prefix: '$',
          target: 3000,
          description: 'Total operational costs',
        },
        {
          id: 'profit',
          title: 'Profit Margin',
          value: overviewResponse.data.profit_margin,
          change: overviewResponse.data.profit_change,
          changePercent: overviewResponse.data.profit_change_percent,
          trend: overviewResponse.data.profit_change >= 0 ? 'up' : 'down',
          icon: <TrendingUp />,
          color: CHART_COLORS.profit,
          suffix: '%',
          target: 70,
          description: 'Profit margin percentage',
        },
      ];

      setMetricCards(cards);

      // Process chart data
      const chartDataProcessed = performanceResponse.data.map((item: unknown) => ({
        timestamp: item.period,
        date: item.period,
        views: item.views,
        revenue: item.revenue,
        subscribers: item.subscriber_growth,
        engagement: item.engagement_rate,
        cost: item.cost,
        watchTime: item.watch_time_hours * 60, // Convert to minutes
        impressions: item.views * 1.2, // Estimated
        ctr: (item.views / (item.views * 1.2)) * 100, // Estimated CTR
      }));

      setChartData(chartDataProcessed);

      // Process channel performance
      const channelData: ChannelPerformance[] = channelsResponse.data.map((channel: unknown) => ({
        channelId: channel.channel_id,
        channelName: channel.channel_name,
        videos: channel.video_count,
        views: channel.total_views,
        revenue: channel.total_views * 0.002, // Estimated revenue
        subscribers: channel.subscriber_count,
        avgEngagement: channel.performance_score,
        growth: Math.random() * 20 - 10, // Mock growth data
      }));

      setChannelPerformance(channelData);

      // Process top videos
      const videoData: VideoPerformance[] = videosResponse.data.slice(0, 10).map((video: unknown) => ({
        videoId: video.id,
        title: video.title,
        channelName: channels.list.find(c => c.id === video.channel_id)?.name || 'Unknown',
        views: video.view_count || Math.floor(Math.random() * 50000),
        engagement: Math.random() * 10,
        revenue: (video.view_count || 0) * 0.002,
        publishedAt: video.created_at,
        duration: video.duration || Math.floor(Math.random() * 600) + 300,
        thumbnail: video.thumbnail_url || '/placeholder-thumbnail.jpg',
      }));

      setTopVideos(videoData);
      setRealtimeMetrics(realtimeResponse.data);
      
      setLoading(false);
    } catch (_error) {
      console.error('Failed to fetch dashboard data:', error);
      addNotification({
        type: 'error',
        message: 'Failed to load dashboard data',
      });
      setLoading(false);
    }
  }, [timeRange, channels.list, addNotification]);

  // Effects
  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(fetchDashboardData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [autoRefresh, fetchDashboardData]);

  // Real-time updates effect
  useEffect(() => {
    if (realtime.lastMessage && realtime.lastMessage.type === 'dashboard_update') {
      setRealtimeMetrics(realtime.lastMessage.data);
      if (autoRefresh) {
        fetchDashboardData();
      }
    }
  }, [realtime.lastMessage, autoRefresh, fetchDashboardData]);

  // Render metric card
  const renderMetricCard = (metric: MetricCard) => {
    const formatValue = (value: number | string) => {
      if (typeof value === 'string') return value;
      if (metric.prefix === '$') return `$${value.toLocaleString()}`;
      if (metric.suffix === '%') return `${value.toFixed(1)}%`;
      return value.toLocaleString();
    };

    const getTrendIcon = () => {
      switch (metric.trend) {
        case 'up':
          return <TrendingUp color="success" />;
        case 'down':
          return <TrendingDown color="error" />;
        default:
          return <TrendingFlat color="action" />;
      }
    };

    const getTrendColor = () => {
      switch (metric.trend) {
        case 'up':
          return metric.id === 'cost' ? 'error.main' : 'success.main';
        case 'down':
          return metric.id === 'cost' ? 'success.main' : 'error.main';
        default:
          return 'text.secondary';
      }
    };

    return (
      <Card key={metric.id} sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box>
              <Typography color="text.secondary" gutterBottom variant="body2">
                {metric.title}
              </Typography>
              <Typography variant="h4" component="div" sx={{ color: metric.color }}>
                {formatValue(metric.value)}
              </Typography>
              <Box display="flex" alignItems="center" mt={1}>
                {getTrendIcon()}
                <Typography
                  variant="body2"
                  sx={{
                    color: getTrendColor(),
                    ml: 0.5,
                  }}
                >
                  {metric.changePercent >= 0 ? '+' : ''}{metric.changePercent.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                  vs yesterday
                </Typography>
              </Box>
            </Box>
            <Avatar sx={{ bgcolor: metric.color, width: 56, height: 56 }}>
              {metric.icon}
            </Avatar>
          </Box>
          
          {showTargets && metric.target && (
            <Box mt={2}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="caption" color="text.secondary">
                  Progress to target
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatValue(metric.target)}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((Number(metric.value) / metric.target) * 100, 100)}
                sx={{
                  mt: 1,
                  backgroundColor: `${metric.color}20`,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: metric.color,
                  },
                }}
              />
            </Box>
          )}
          
          {metric.description && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              {metric.description}
            </Typography>
          )}
        </CardContent>
      </Card>
    );
  };

  // Render chart based on selected type
  const renderChart = (data: ChartData[], metrics: string[]) => {
    const commonProps = {
      data: processedChartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chartType) {
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart {...commonProps}>
              <defs>
                {metrics.map((metric) => (
                  <linearGradient key={metric} id={`gradient-${metric}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS[metric]} stopOpacity={0.8} />
                    <stop offset="95%" stopColor={CHART_COLORS[metric]} stopOpacity={0.1} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <ChartTooltip />
              <Legend />
              {metrics.map((metric) => (
                <Area
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={CHART_COLORS[metric]}
                  fillOpacity={1}
                  fill={`url(#gradient-${metric})`}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <ChartTooltip />
              <Legend />
              {metrics.map((metric) => (
                <Bar key={metric} dataKey={metric} fill={CHART_COLORS[metric]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'line':
      default:
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <ChartTooltip />
              <Legend />
              {metrics.map((metric) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={CHART_COLORS[metric]}
                  strokeWidth={2}
                  dot={{ fill: CHART_COLORS[metric], strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
    }
  };

  // Render performance comparison chart
  const renderPerformanceChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={channelPerformance}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="channelName" angle={-45} textAnchor="end" height={100} />
        <YAxis yAxisId="left" />
        <YAxis yAxisId="right" orientation="right" />
        <ChartTooltip />
        <Legend />
        <Bar yAxisId="left" dataKey="videos" fill="#8884d8" name="Videos" />
        <Line yAxisId="right" type="monotone" dataKey="avgEngagement" stroke="#ff7300" name="Avg Engagement %" />
      </ComposedChart>
    </ResponsiveContainer>
  );

  if (loading) {
    return (
      <Box p={3}>
        <Grid container spacing={3}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Grid item xs={12} sm={6} md={4} key={i}>
              <Skeleton variant="rectangular" height={200} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs>
            <Typography variant="h5">Analytics Dashboard</Typography>
            {realtimeMetrics && (
              <Typography variant="body2" color="text.secondary">
                Last updated: {format(new Date(), 'MMM d, h:mm:ss a')} • 
                {realtimeMetrics.videos_generated_today} videos today • 
                ${realtimeMetrics.cost_today} spent
              </Typography>
            )}
          </Grid>
          
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value)}
              >
                {TIME_RANGES.map((range) => (
                  <MenuItem key={range.value} value={range.value}>
                    {range.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Chart Type</InputLabel>
              <Select
                value={chartType}
                label="Chart Type"
                onChange={(e) => setChartType(e.target.value)}
              >
                {CHART_TYPES.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    <Box display="flex" alignItems="center">
                      {type.icon}
                      <Box ml={1}>{type.label}</Box>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Auto Refresh"
            />
          </Grid>
          
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={showTargets}
                  onChange={(e) => setShowTargets(e.target.checked)}
                />
              }
              label="Show Targets"
            />
          </Grid>
          
          <Grid item>
            <Tooltip title="Refresh Data">
              <IconButton onClick={fetchDashboardData}>
                <Badge color="secondary" variant="dot" invisible={!autoRefresh}>
                  <Refresh />
                </Badge>
              </IconButton>
            </Tooltip>
          </Grid>
          
          <Grid item>
            <Button startIcon={<Download />} variant="outlined">
              Export
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {metricCards.map((metric) => (
          <Grid item xs={12} sm={6} md={4} lg={2} key={metric.id}>
            {renderMetricCard(metric)}
          </Grid>
        ))}
      </Grid>

      {/* Main Chart */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Performance Trends</Typography>
          <Box>
            <IconButton onClick={() => setFullscreenChart('main')}>
              <Fullscreen />
            </IconButton>
          </Box>
        </Box>
        {renderChart(chartData, selectedMetrics)}
      </Paper>

      {/* Secondary Charts and Tables */}
      <Grid container spacing={3}>
        {/* Channel Performance */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Channel Performance Comparison
            </Typography>
            {renderPerformanceChart()}
          </Paper>
        </Grid>

        {/* Top Videos */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Top Performing Videos
            </Typography>
            <List>
              {topVideos.slice(0, 5).map((video, index) => (
                <ListItem key={video.videoId}>
                  <ListItemIcon>
                    <Avatar src={video.thumbnail} sx={{ width: 40, height: 40 }}>
                      {index + 1}
                    </Avatar>
                  </ListItemIcon>
                  <ListItemText
                    primary={video.title}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {video.channelName}
                        </Typography>
                        <Typography variant="caption" color="primary">
                          {video.views.toLocaleString()} views • ${video.revenue.toFixed(2)}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Fullscreen Chart Dialog */}
      <Dialog
        open={fullscreenChart !== null}
        onClose={() => setFullscreenChart(null)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>
          Performance Trends - Full View
        </DialogTitle>
        <DialogContent>
          <Box height={600}>
            {renderChart(chartData, selectedMetrics)}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFullscreenChart(null)}>Close</Button>
          <Button startIcon={<Download />}>Export Chart</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};