/**
 * Real-time Dashboard with WebSocket Integration
 * P0 Task: Dashboard Implementation with real-time updates
 */
import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert,
  Badge,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  VideoLibrary,
  AttachMoney,
  People,
  Schedule,
  Refresh,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  PlayCircle,
  Pause,
  Stop,
  FiberManualRecord,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

import { useDashboardWebSocket, useMetricsWebSocket } from '../../hooks/useWebSocket';
import { useAuthStore } from '../../stores/authStore';
import { DashboardHeader } from '../../components/Dashboard/DashboardHeader';
import { MetricCard } from '../../components/Dashboard/MetricCard';
import { api } from '../../services/api';

interface RealTimeMetrics {
  totalVideos: number;
  totalViews: number;
  totalRevenue: number;
  totalCosts: number;
  activeChannels: number;
  queuedVideos: number;
  processingVideos: number;
  completedToday: number;
  averageCostPerVideo: number;
  engagementRate: number;
  activeGenerations: Array<{
    videoId: string;
    channelId: string;
    phase: string;
    progress: number;
    startTime: string;
    estimatedCompletion: string;
  }>;
  recentCosts: Array<{
    service: string;
    amount: number;
    timestamp: string;
  }>;
  quotaStatus: {
    used: number;
    total: number;
    percentage: number;
  };
}

interface VideoGenerationStatus {
  videoId: string;
  phase: string;
  progress: number;
  currentCost: number;
  estimatedTime: string;
  quality?: number;
}

export const DashboardRealtime: React.FC = () => {
  const theme = useTheme();
  const { user } = useAuthStore();
  
  // WebSocket connections
  const dashboardWs = useDashboardWebSocket();
  const metricsWs = useMetricsWebSocket();
  
  const [metrics, setMetrics] = useState<RealTimeMetrics>({
    totalVideos: 0,
    totalViews: 0,
    totalRevenue: 0,
    totalCosts: 0,
    activeChannels: 0,
    queuedVideos: 0,
    processingVideos: 0,
    completedToday: 0,
    averageCostPerVideo: 0,
    engagementRate: 0,
    activeGenerations: [],
    recentCosts: [],
    quotaStatus: { used: 0, total: 150000, percentage: 0 },
  });
  
  const [videoGenerations, setVideoGenerations] = useState<VideoGenerationStatus[]>([]);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [chartData, setChartData] = useState<any>({
    views: [],
    costs: [],
    revenue: [],
  });
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Fetch initial metrics
  useEffect(() => {
    fetchInitialMetrics();
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (dashboardWs.lastMessage) {
      handleWebSocketMessage(dashboardWs.lastMessage);
    }
  }, [dashboardWs.lastMessage]);

  useEffect(() => {
    if (metricsWs.lastMessage) {
      handleMetricsUpdate(metricsWs.lastMessage);
    }
  }, [metricsWs.lastMessage]);

  // Subscribe to specific WebSocket events
  useEffect(() => {
    const unsubscribeVideoUpdate = dashboardWs.subscribe('video-update', (data) => {
      handleVideoUpdate(data);
    });

    const unsubscribeCostUpdate = dashboardWs.subscribe('cost-update', (data) => {
      handleCostUpdate(data);
    });

    const unsubscribeGenerationStatus = dashboardWs.subscribe('generation-status', (data) => {
      handleGenerationStatus(data);
    });

    return () => {
      unsubscribeVideoUpdate?.();
      unsubscribeCostUpdate?.();
      unsubscribeGenerationStatus?.();
    };
  }, [dashboardWs]);

  const fetchInitialMetrics = async () => {
    setLoading(true);
    try {
      // Fetch dashboard metrics
      const response = await api.get('/dashboard/metrics');
      setMetrics(response.data);
      
      // Fetch chart data
      const chartResponse = await api.get('/dashboard/charts');
      setChartData(chartResponse.data);
      
      // Fetch active generations
      const generationsResponse = await api.get('/test/test-generation-status');
      if (generationsResponse.data.active_generations) {
        const generations = Object.entries(generationsResponse.data.active_generations).map(
          ([videoId, phase]) => ({
            videoId,
            phase: phase as string,
            progress: calculateProgressFromPhase(phase as string),
            currentCost: 0,
            estimatedTime: '5 min',
          })
        );
        setVideoGenerations(generations);
      }
      
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleWebSocketMessage = (message: any) => {
    console.log('Dashboard WebSocket message:', message);
    
    if (message.type === 'metrics-update') {
      setMetrics(prev => ({ ...prev, ...message.data }));
    } else if (message.type === 'notification') {
      setNotifications(prev => [message, ...prev].slice(0, 10));
    }
    
    setLastUpdated(new Date());
  };

  const handleMetricsUpdate = (data: any) => {
    if (data.metrics) {
      setMetrics(prev => ({ ...prev, ...data.metrics }));
    }
    if (data.chartData) {
      setChartData(prev => ({ ...prev, ...data.chartData }));
    }
  };

  const handleVideoUpdate = (data: any) => {
    setMetrics(prev => ({
      ...prev,
      totalVideos: prev.totalVideos + (data.increment || 0),
      processingVideos: data.processingCount || prev.processingVideos,
      queuedVideos: data.queuedCount || prev.queuedVideos,
      completedToday: data.completedToday || prev.completedToday,
    }));
  };

  const handleCostUpdate = (data: any) => {
    setMetrics(prev => ({
      ...prev,
      totalCosts: prev.totalCosts + (data.amount || 0),
      recentCosts: [
        { 
          service: data.service, 
          amount: data.amount, 
          timestamp: new Date().toISOString() 
        },
        ...prev.recentCosts
      ].slice(0, 10),
    }));
  };

  const handleGenerationStatus = (data: any) => {
    setVideoGenerations(prev => {
      const existing = prev.find(g => g.videoId === data.videoId);
      if (existing) {
        return prev.map(g => 
          g.videoId === data.videoId 
            ? { ...g, ...data } 
            : g
        );
      } else {
        return [...prev, data];
      }
    });
  };

  const calculateProgressFromPhase = (phase: string): number => {
    const phaseProgress: { [key: string]: number } = {
      'initialization': 5,
      'trend_analysis': 15,
      'script_generation': 30,
      'voice_synthesis': 45,
      'visual_generation': 60,
      'video_assembly': 75,
      'quality_check': 85,
      'publishing': 95,
      'completed': 100,
    };
    return phaseProgress[phase] || 0;
  };

  const getPhaseColor = (phase: string): string => {
    if (phase === 'completed') return theme.palette.success.main;
    if (phase === 'failed') return theme.palette.error.main;
    if (phase.includes('processing') || phase.includes('synthesis')) return theme.palette.warning.main;
    return theme.palette.info.main;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header with WebSocket Status */}
      <DashboardHeader
        title="Real-time Dashboard"
        subtitle={`Welcome back, ${user?.name || 'User'}`}
        lastUpdated={lastUpdated}
        onRefresh={fetchInitialMetrics}
        loading={loading}
        extra={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              icon={<FiberManualRecord />}
              label={dashboardWs.connected ? 'Live' : 'Offline'}
              color={dashboardWs.connected ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
            {metrics.processingVideos > 0 && (
              <Badge badgeContent={metrics.processingVideos} color="warning">
                <CircularProgress size={20} />
              </Badge>
            )}
          </Box>
        }
      />

      {/* Cost Alert */}
      {metrics.totalCosts > metrics.totalRevenue * 0.3 && (
        <Alert
          severity="warning"
          icon={<Warning />}
          sx={{ mb: 3 }}
        >
          Cost ratio high: {((metrics.totalCosts / metrics.totalRevenue) * 100).toFixed(1)}% of revenue
        </Alert>
      )}

      {/* Real-time Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Videos Today"
            value={metrics.completedToday}
            icon={<VideoLibrary />}
            trend={metrics.processingVideos > 0 ? `+${metrics.processingVideos} processing` : ''}
            trendDirection="up"
            color="#8884d8"
            realtime
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Live Cost"
            value={formatCurrency(metrics.totalCosts)}
            icon={<AttachMoney />}
            trend={`Avg: ${formatCurrency(metrics.averageCostPerVideo)}/video`}
            trendDirection={metrics.averageCostPerVideo < 3 ? 'down' : 'up'}
            color={metrics.averageCostPerVideo < 3 ? '#82ca9d' : '#ff7c7c'}
            realtime
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Queue Status"
            value={metrics.queuedVideos}
            icon={<Schedule />}
            trend={`${metrics.processingVideos} processing`}
            color="#ffc658"
            realtime
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="API Quota"
            value={`${metrics.quotaStatus.percentage.toFixed(1)}%`}
            icon={<TrendingUp />}
            trend={`${formatNumber(metrics.quotaStatus.used)}/${formatNumber(metrics.quotaStatus.total)}`}
            trendDirection={metrics.quotaStatus.percentage > 80 ? 'up' : 'down'}
            color={metrics.quotaStatus.percentage > 80 ? '#ff7c7c' : '#82ca9d'}
            realtime
          />
        </Grid>
      </Grid>

      {/* Active Video Generations */}
      {videoGenerations.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Active Video Generations
          </Typography>
          <List>
            {videoGenerations.map((generation, index) => (
              <React.Fragment key={generation.videoId}>
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: getPhaseColor(generation.phase) }}>
                      {generation.progress < 100 ? <PlayCircle /> : <CheckCircle />}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle1">
                          {generation.videoId}
                        </Typography>
                        <Chip 
                          label={generation.phase.replace('_', ' ')} 
                          size="small" 
                          color={generation.progress === 100 ? 'success' : 'primary'}
                        />
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={generation.progress} 
                          sx={{ mb: 1 }}
                          color={generation.progress === 100 ? 'success' : 'primary'}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="caption" color="text.secondary">
                            Cost: {formatCurrency(generation.currentCost)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ETA: {generation.estimatedTime}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {generation.progress}%
                          </Typography>
                        </Box>
                      </Box>
                    }
                  />
                </ListItem>
                {index < videoGenerations.length - 1 && <Divider variant="inset" component="li" />}
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}

      {/* Real-time Cost Breakdown */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Live Cost Tracking
            </Typography>
            <List sx={{ maxHeight: 320, overflow: 'auto' }}>
              {metrics.recentCosts.map((cost, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={cost.service}
                    secondary={new Date(cost.timestamp).toLocaleTimeString()}
                  />
                  <Typography variant="subtitle2" color={cost.amount > 1 ? 'error' : 'text.primary'}>
                    {formatCurrency(cost.amount)}
                  </Typography>
                </ListItem>
              ))}
            </List>
            <Divider sx={{ my: 2 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="subtitle1">Total Session Cost:</Typography>
              <Typography variant="h6" color="primary">
                {formatCurrency(metrics.recentCosts.reduce((sum, c) => sum + c.amount, 0))}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Notifications */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Live Notifications
            </Typography>
            <List sx={{ maxHeight: 350, overflow: 'auto' }}>
              {notifications.map((notif, index) => (
                <ListItem key={index}>
                  <ListItemAvatar>
                    <Avatar sx={{ 
                      bgcolor: notif.severity === 'error' ? 'error.main' : 
                               notif.severity === 'warning' ? 'warning.main' : 
                               'success.main' 
                    }}>
                      {notif.severity === 'error' ? <ErrorIcon /> : 
                       notif.severity === 'warning' ? <Warning /> : 
                       <CheckCircle />}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={notif.message}
                    secondary={new Date(notif.timestamp).toLocaleString()}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Real-time Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Real-time Performance Metrics
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={chartData.views || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="views" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                />
                <Line 
                  type="monotone" 
                  dataKey="engagement" 
                  stroke="#82ca9d" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                />
                <Line 
                  type="monotone" 
                  dataKey="cost" 
                  stroke="#ff7c7c" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};