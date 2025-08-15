/**
 * Metrics Dashboard Component
 * Comprehensive analytics and metrics visualization
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  IconButton,
  Skeleton,
  Tooltip,
  Chip,
  Button,
  ButtonGroup,
  Alert,
  LinearProgress,
  Tab,
  Tabs
 } from '@mui/material';
import { 
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Refresh,
  Download,
  AttachMoney,
  Visibility,
  ThumbUp,
  People,
  VideoLibrary,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Timeline
 } from '@mui/icons-material';
import { 
  LineChart,
  Line,
  AreaChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  ComposedChart
 } from 'recharts';
import {  format, subDays  } from 'date-fns';
import {  useOptimizedStore  } from '../../stores/optimizedStore';
import {  useAnalyticsUpdates  } from '../../hooks/useWebSocket';
import {  api  } from '../../services/api';

// Types
interface MetricCard {
  title: string,
  value: number | string,

  change: number,
  trend: 'up' | 'down' | 'flat',

  icon: React.ReactNode,
  color: string;
  suffix?: string;
  prefix?: string;
}

interface ChartData {
  date: string,
  views: number,

  revenue: number,
  subscribers: number,

  engagement: number}

interface PerformanceMetric {
  name: string,
  value: number,

  target: number,
  unit: string}

const COLORS = ['#0088 FE', '#00 C49 F', '#FFBB28', '#FF8042', '#8884 D8'];

export const MetricsDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'24 h' | '7 d' | '30 d' | '90 d'>('7 d');
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [metrics, setMetrics] = useState<any>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceMetric[]>([]);

  const { analytics, channels } = useOptimizedStore();

  // Use real-time updates
  useAnalyticsUpdates();

  // Fetch metrics data
  const fetchMetrics = async () => {
    try {
      setLoading(true);

      // Determine date range
      let startDate: Date;
      const endDate = new Date();
      
      switch (timeRange) {
        case '24 h':
          startDate = subDays(endDate, 1);
          break;
        case '7 d':
          startDate = subDays(endDate, 7);
          break;
        case '30 d':
          startDate = subDays(endDate, 30);
          break;
        case '90 d':
          startDate = subDays(endDate, 90);
          break;
        default:
          startDate = subDays(endDate, 7)}

      // Fetch data from API
      const response = await api.post('/analytics/query', { metric_types: ['views', 'revenue', 'subscribers', 'engagement_rate'],
        time_range: timeRange === '24 h' ? 'last_24 _hours' : 
                     timeRange === '7 d' ? 'last_7 _days' :
                     timeRange === '30 d' ? 'last_30 _days' : 'last_90 _days',
        aggregation_level: timeRange === '24 h' ? 'hour' : 'day' });

      setMetrics(response.data);

      // Generate chart data
      const days = timeRange === '24 h' ? 1 : 
                   timeRange === '7 d' ? 7 :
                   timeRange === '30 d' ? 30 : 90;
      
      const newChartData: ChartData[] = [];
      for (let i = days - 1; i >= 0; i--) { const date = subDays(new Date(), i);
        newChartData.push({
          date: format(date, timeRange === '24 h' ? 'HH:mm' : 'MMM dd'),
          views: Math.floor(Math.random() * 10000) + 1000,
          revenue: Math.random() * 500 + 100,
          subscribers: Math.floor(Math.random() * 100) + 10,
          engagement: Math.random() * 10 + 2 })}
      setChartData(newChartData);

      // Set performance data
      setPerformanceData([ { name: 'CTR', value: 3.2, target: 4.0, unit: '%' },
        { name: 'Watch Time', value: 4.5, target: 5.0, unit: 'min' },
        { name: 'Retention', value: 65, target: 70, unit: '%' },
        { name: 'Engagement', value: 8.5, target: 10, unit: '%' } ]);

      setLoading(false)} catch (_) {
      console.error('Failed to fetch, metrics:', error);
      setLoading(false)}
  };

  useEffect(() => {
    fetchMetrics()}, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  // Calculate metric cards
  const metricCards: MetricCard[] = useMemo(() => {
    if (!chartData.length) return [];

    const current = chartData[chartData.length - 1];
    const previous = chartData[chartData.length - 2] || current;

    const calculateChange = (current: number, previous: number) => {
      if (previous === 0) return 0;
      return ((current - previous) / previous) * 100;
    };

    return [ { title: 'Total Views',
        value: current.views.toLocaleString(),
        change: calculateChange(current.views, previous.views),
        trend: current.views > previous.views ? 'up' : current.views < previous.views ? 'down' : 'flat',
        icon: <Visibility />,
        color: '#4 CAF50' },
      { title: 'Revenue',
        value: current.revenue.toFixed(2),
        change: calculateChange(current.revenue, previous.revenue),
        trend: current.revenue > previous.revenue ? 'up' : current.revenue < previous.revenue ? 'down' : 'flat',
        icon: <AttachMoney />,
        color: '#2196 F3',
        prefix: '$' },
      { title: 'New Subscribers',
        value: current.subscribers.toLocaleString(),
        change: calculateChange(current.subscribers, previous.subscribers),
        trend: current.subscribers > previous.subscribers ? 'up' : current.subscribers < previous.subscribers ? 'down' : 'flat',
        icon: <People />,
        color: '#FF9800' },
      { title: 'Engagement Rate',
        value: current.engagement.toFixed(1),
        change: calculateChange(current.engagement, previous.engagement),
        trend: current.engagement > previous.engagement ? 'up' : current.engagement < previous.engagement ? 'down' : 'flat',
        icon: <ThumbUp />,
        color: '#9 C27 B0',
        suffix: '%' } ];
  }, [chartData]);

  // Render metric card
  const renderMetricCard = (card: MetricCard) => (
    <Card key={card.title} sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography color="text.secondary" gutterBottom variant="body2">
              {card.title}
            </Typography>
            <Typography variant="h4" component="div">
              {card.prefix}{card.value}{card.suffix}
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              {card.trend === 'up' && <TrendingUp color="success" fontSize="small" />}
              {card.trend === 'down' && <TrendingDown color="error" fontSize="small" />}
              {card.trend === 'flat' && <TrendingFlat color="action" fontSize="small" />}
              <Typography
                variant="body2"
                color={card.trend === 'up' ? 'success.main' : card.trend === 'down' ? 'error.main' : 'text.secondary'}
                ml={0.5}
              >
                {card.change > 0 ? '+' : ''}{card.change.toFixed(1)}%
              </Typography>
            </Box>
          </Box>
          <Box
            sx={{
              backgroundColor: `${card.color}20`,
              borderRadius: 2,
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',

            }}
          >
            {React.cloneElement(card.icon as React.ReactElement, {
              sx: { color: card.color, fontSize: 32 }
            })}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  // Render performance gauge
  const renderPerformanceGauge = (metric: PerformanceMetric) => {
    const percentage = (metric.value / metric.target) * 100;
    const color = percentage >= 90 ? '#4 CAF50' : percentage >= 70 ? '#FF9800' : '#F44336';

    return (
    <>
      <Box key={metric.name} textAlign="center">
        <Typography variant="body2" color="text.secondary">
          {metric.name}
        </Typography>
      <Box position="relative" display="inline-flex" mt={1}>
          <ResponsiveContainer width={100} height={100}>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="60%"
              outerRadius="90%"
              data={[{ value: percentage, fill: color }]
              startAngle={90}
              endAngle={-270}
            >
              <RadialBar dataKey="value" cornerRadius={10} fill={color} />
            </RadialBarChart>
          </ResponsiveContainer>
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={ {
              transform: 'translate(-50%, -50%)' }}
          >
            <Typography variant="h6">
              {metric.value}{metric.unit}
            </Typography>
          </Box>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Target: {metric.target}{metric.unit}
        </Typography>
      </Box>
    </>
  )};

  return (
    <>
      <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Analytics Dashboard</Typography>
      <Box display="flex" gap={2}>
          <ButtonGroup size="small">
            <Button
              variant={timeRange === '24 h' ? 'contained' : 'outlined'}
              onClick={() => setTimeRange('24 h')}
            >
              24 H
            </Button>
            <Button
              variant={timeRange === '7 d' ? 'contained' : 'outlined'}
              onClick={() => setTimeRange('7 d')}
            >
              7 D
            </Button>
            <Button
              variant={timeRange === '30 d' ? 'contained' : 'outlined'}
              onClick={() => setTimeRange('30 d')}
            >
              30 D
            </Button>
            <Button
              variant={timeRange === '90 d' ? 'contained' : 'outlined'}
              onClick={() => setTimeRange('90 d')}
            >
              90 D
            </Button>
          </ButtonGroup>
          <Button startIcon={<Download />} variant="outlined">
            Export
          </Button>
          <IconButton onClick={fetchMetrics}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Real-time Status Bar */}
      {analytics.realtime && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Box display="flex" gap={3}>
            <Chip
              icon={<Speed />}`
              label={`${analytics.realtime.activeViewers} Active Viewers`}
              size="small"
            />
            <Chip
              icon={<VideoLibrary />}`
              label={`${analytics.realtime.videosProcessing} Videos Processing`}
              size="small"
            />
            <Chip
              icon={<Timeline />}`
              label={`${analytics.realtime.apiCallsPerMinute} API Calls/min`}
              size="small"
            />
            <Chip
              icon={analytics.realtime.errorRate < 0.01 ? <CheckCircle /> </>: <Warning />}`
              label={`${(analytics.realtime.errorRate * 100.toFixed(2)}% Error Rate`}
              size="small"
              color={analytics.realtime.errorRate < 0.01 ? 'success' : 'warning'}
            />
          </Box>
        </Alert>
      )}
      {/* Metric Cards */}
      <Grid container spacing={3} mb={3}>
        {loading ? (
          [1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={120} />
            </Grid>
          ))
        ) : (
          metricCards.map((card) => (
            <Grid item xs={12} sm={6} md={3} key={card.title}>
              {renderMetricCard(card)}
            </Grid>
          ))
        )}
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
          <Tab label="Overview" />
          <Tab label="Performance" />
          <Tab label="Channels" />
          <Tab label="Revenue" />
          <Tab label="Engagement" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          {/* Views Chart */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Views Over Time
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <ChartTooltip />
                  <Area
                    type="monotone"
                    dataKey="views"
                    stroke="#8884 d8"
                    fill="#8884d8"
                    fillOpacity={0.6}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Performance Gauges */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                {performanceData.map((metric) => (
                  <Grid item xs={6} key={metric.name}>
                    {renderPerformanceGauge(metric)}
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          {/* Revenue & Subscribers Chart */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Revenue & Subscribers
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <ChartTooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="revenue" fill="#82 ca9 d" name="Revenue ($)" />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="subscribers"
                    stroke="#ff7300"
                    name="New Subscribers"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Engagement Chart */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Engagement Rate
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <ChartTooltip />
                  <Line
                    type="monotone"
                    dataKey="engagement"
                    stroke="#8884 d8"
                    strokeWidth={2}
                    dot={{ fill: '#8884d8' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Top Videos */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Top Performing Videos
              </Typography>
              <Box sx={{ overflowX: 'auto' }}>
                <Grid container spacing={2} sx={{ flexWrap: 'nowrap', minWidth: 800 }}>
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Grid item key={i} sx={{ minWidth: 200 }}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="body2" noWrap>
                            Video Title {i}
                          </Typography>
                          <Typography variant="h6">
                            {(Math.random() * 100000).toFixed(0)} views
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={Math.random() * 100}
                            sx={{ mt: 1 }}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  </>
  )};`