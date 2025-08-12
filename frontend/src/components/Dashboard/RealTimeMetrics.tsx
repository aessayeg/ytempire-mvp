import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  CircularProgress,
  IconButton,
  Tooltip,
  Badge,
  Avatar,
  useTheme,
  Paper,
  Skeleton,
  Fade,
  Grow,
  Alert,
  Button,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  AttachMoney,
  Visibility,
  ThumbUp,
  Schedule,
  CloudQueue,
  CheckCircle,
  Error,
  Warning,
  Refresh,
  FiberManualRecord,
  Speed,
  Timer,
  Update,
  NotificationImportant,
} from '@mui/icons-material';
import { format, formatDistanceToNow } from 'date-fns';
import { useWebSocket } from '../../hooks/useWebSocket';
import { AnimatePresence, motion } from 'framer-motion';
import CountUp from 'react-countup';

interface MetricData {
  id: string;
  label: string;
  value: number;
  previousValue: number;
  unit?: string;
  trend: 'up' | 'down' | 'flat';
  changePercent: number;
  icon: React.ReactNode;
  color: string;
  sparklineData?: number[];
  lastUpdated: Date;
  isLive?: boolean;
}

interface LiveEvent {
  id: string;
  type: 'video_published' | 'revenue_earned' | 'milestone_reached' | 'error' | 'warning';
  title: string;
  description: string;
  timestamp: Date;
  severity: 'info' | 'success' | 'warning' | 'error';
}

interface RealTimeMetricsProps {
  channelId?: string;
  refreshInterval?: number;
  showSparklines?: boolean;
  compactMode?: boolean;
}

export const RealTimeMetrics: React.FC<RealTimeMetricsProps> = ({
  channelId,
  refreshInterval = 5000,
  showSparklines = true,
  compactMode = false,
}) => {
  const theme = useTheme();
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const animationRef = useRef<number>();

  // WebSocket connection for real-time updates
  const { sendMessage, lastMessage, readyState } = useWebSocket('/ws/metrics', {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
  });

  // Initialize metrics
  useEffect(() => {
    const initialMetrics: MetricData[] = [
      {
        id: 'revenue',
        label: 'Revenue Today',
        value: 127.50,
        previousValue: 112.30,
        unit: '$',
        trend: 'up',
        changePercent: 13.5,
        icon: <AttachMoney />,
        color: theme.palette.success.main,
        sparklineData: [100, 105, 110, 108, 115, 120, 127],
        lastUpdated: new Date(),
        isLive: true,
      },
      {
        id: 'views',
        label: 'Total Views',
        value: 15234,
        previousValue: 14500,
        trend: 'up',
        changePercent: 5.1,
        icon: <Visibility />,
        color: theme.palette.primary.main,
        sparklineData: [14000, 14200, 14500, 14600, 14900, 15000, 15234],
        lastUpdated: new Date(),
        isLive: true,
      },
      {
        id: 'engagement',
        label: 'Engagement Rate',
        value: 4.7,
        previousValue: 4.2,
        unit: '%',
        trend: 'up',
        changePercent: 11.9,
        icon: <ThumbUp />,
        color: theme.palette.secondary.main,
        sparklineData: [4.0, 4.1, 4.2, 4.3, 4.5, 4.6, 4.7],
        lastUpdated: new Date(),
        isLive: true,
      },
      {
        id: 'processing',
        label: 'Videos Processing',
        value: 3,
        previousValue: 5,
        trend: 'down',
        changePercent: -40,
        icon: <CloudQueue />,
        color: theme.palette.warning.main,
        lastUpdated: new Date(),
        isLive: true,
      },
      {
        id: 'scheduled',
        label: 'Scheduled Today',
        value: 8,
        previousValue: 6,
        trend: 'up',
        changePercent: 33.3,
        icon: <Schedule />,
        color: theme.palette.info.main,
        lastUpdated: new Date(),
      },
      {
        id: 'health',
        label: 'System Health',
        value: 98,
        previousValue: 95,
        unit: '%',
        trend: 'up',
        changePercent: 3.2,
        icon: <Speed />,
        color: theme.palette.success.main,
        lastUpdated: new Date(),
        isLive: true,
      },
    ];

    setMetrics(initialMetrics);
    setLoading(false);
  }, [theme]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        
        if (data.type === 'metric_update') {
          setMetrics(prev => prev.map(metric =>
            metric.id === data.metricId
              ? {
                  ...metric,
                  previousValue: metric.value,
                  value: data.value,
                  trend: data.value > metric.value ? 'up' : data.value < metric.value ? 'down' : 'flat',
                  changePercent: ((data.value - metric.value) / metric.value) * 100,
                  lastUpdated: new Date(),
                  sparklineData: metric.sparklineData ? [...metric.sparklineData.slice(1), data.value] : undefined,
                }
              : metric
          ));
          setLastUpdate(new Date());
        }
        
        if (data.type === 'live_event') {
          const newEvent: LiveEvent = {
            id: `event-${Date.now()}`,
            type: data.eventType,
            title: data.title,
            description: data.description,
            timestamp: new Date(),
            severity: data.severity || 'info',
          };
          setLiveEvents(prev => [newEvent, ...prev].slice(0, 10));
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  // Connection status monitoring
  useEffect(() => {
    if (readyState === WebSocket.OPEN) {
      setConnectionStatus('connected');
    } else if (readyState === WebSocket.CONNECTING) {
      setConnectionStatus('connecting');
    } else {
      setConnectionStatus('disconnected');
    }
  }, [readyState]);

  // Simulate real-time updates (for demo)
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => {
        if (metric.isLive) {
          const change = (Math.random() - 0.5) * 10;
          const newValue = Math.max(0, metric.value + change);
          return {
            ...metric,
            previousValue: metric.value,
            value: newValue,
            trend: newValue > metric.value ? 'up' : newValue < metric.value ? 'down' : 'flat',
            changePercent: ((newValue - metric.value) / metric.value) * 100,
            lastUpdated: new Date(),
            sparklineData: metric.sparklineData 
              ? [...metric.sparklineData.slice(1), newValue]
              : undefined,
          };
        }
        return metric;
      }));
      setLastUpdate(new Date());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  // Render trend icon
  const getTrendIcon = (trend: 'up' | 'down' | 'flat', color: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp sx={{ color, fontSize: 16 }} />;
      case 'down':
        return <TrendingDown sx={{ color: theme.palette.error.main, fontSize: 16 }} />;
      default:
        return <TrendingFlat sx={{ color: theme.palette.text.secondary, fontSize: 16 }} />;
    }
  };

  // Render sparkline
  const renderSparkline = (data?: number[]) => {
    if (!data || !showSparklines) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const width = 60;
    const height = 30;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width={width} height={height} style={{ marginLeft: 'auto' }}>
        <polyline
          points={points}
          fill="none"
          stroke={theme.palette.primary.main}
          strokeWidth="2"
        />
      </svg>
    );
  };

  // Render metric card
  const renderMetricCard = (metric: MetricData) => (
    <Grid item xs={12} sm={6} md={4} lg={2} key={metric.id}>
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Card
          sx={{
            height: '100%',
            position: 'relative',
            overflow: 'visible',
            ...(metric.isLive && {
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 8,
                right: 8,
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: theme.palette.success.main,
                animation: 'pulse 2s infinite',
              },
            }),
          }}
        >
          <CardContent sx={{ p: compactMode ? 1.5 : 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  backgroundColor: `${metric.color}20`,
                  color: metric.color,
                }}
              >
                {metric.icon}
              </Avatar>
              {metric.isLive && (
                <Tooltip title="Live data">
                  <FiberManualRecord
                    sx={{
                      fontSize: 10,
                      color: theme.palette.success.main,
                      ml: 'auto',
                      animation: 'pulse 2s infinite',
                    }}
                  />
                </Tooltip>
              )}
            </Box>
            
            <Typography variant="caption" color="text.secondary" display="block">
              {metric.label}
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5, my: 1 }}>
              <Typography variant="h5" fontWeight="bold">
                {metric.unit && metric.unit === '$' && metric.unit}
                <CountUp
                  start={metric.previousValue}
                  end={metric.value}
                  duration={1}
                  decimals={metric.unit === '$' || metric.unit === '%' ? 2 : 0}
                  preserveValue
                />
                {metric.unit && metric.unit !== '$' && metric.unit}
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {getTrendIcon(metric.trend, metric.color)}
              <Typography
                variant="caption"
                sx={{
                  color: metric.trend === 'up' 
                    ? theme.palette.success.main 
                    : metric.trend === 'down'
                    ? theme.palette.error.main
                    : theme.palette.text.secondary,
                }}
              >
                {metric.changePercent > 0 ? '+' : ''}{metric.changePercent.toFixed(1)}%
              </Typography>
              {renderSparkline(metric.sparklineData)}
            </Box>
            
            {!compactMode && (
              <Typography variant="caption" color="text.disabled" display="block" sx={{ mt: 1 }}>
                {formatDistanceToNow(metric.lastUpdated, { addSuffix: true })}
              </Typography>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </Grid>
  );

  // Render live events ticker
  const renderLiveEvents = () => (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" fontWeight="bold">
          Live Activity
        </Typography>
        <Badge
          badgeContent={liveEvents.length}
          color="primary"
          sx={{ ml: 2 }}
        >
          <NotificationImportant />
        </Badge>
      </Box>
      
      <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
        <AnimatePresence>
          {liveEvents.map((event) => (
            <motion.div
              key={event.id}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 20, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Alert
                severity={event.severity}
                sx={{ mb: 1 }}
                onClose={() => {
                  setLiveEvents(prev => prev.filter(e => e.id !== event.id));
                }}
              >
                <Typography variant="subtitle2" fontWeight="medium">
                  {event.title}
                </Typography>
                <Typography variant="caption" display="block">
                  {event.description}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatDistanceToNow(event.timestamp, { addSuffix: true })}
                </Typography>
              </Alert>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {liveEvents.length === 0 && (
          <Typography variant="body2" color="text.secondary" align="center">
            No recent activity
          </Typography>
        )}
      </Box>
    </Paper>
  );

  if (loading) {
    return (
      <Grid container spacing={2}>
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <Grid item xs={12} sm={6} md={4} lg={2} key={i}>
            <Card>
              <CardContent>
                <Skeleton variant="circular" width={32} height={32} />
                <Skeleton variant="text" width="60%" sx={{ mt: 1 }} />
                <Skeleton variant="text" width="40%" height={32} />
                <Skeleton variant="text" width="30%" />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  return (
    <Box>
      {/* Connection Status */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 2 }}>
        <Chip
          icon={
            connectionStatus === 'connected' ? (
              <FiberManualRecord sx={{ fontSize: 12 }} />
            ) : connectionStatus === 'connecting' ? (
              <CircularProgress size={12} />
            ) : (
              <Error sx={{ fontSize: 12 }} />
            )
          }
          label={
            connectionStatus === 'connected'
              ? 'Live'
              : connectionStatus === 'connecting'
              ? 'Connecting...'
              : 'Disconnected'
          }
          color={
            connectionStatus === 'connected'
              ? 'success'
              : connectionStatus === 'connecting'
              ? 'warning'
              : 'error'
          }
          size="small"
        />
        
        <Typography variant="caption" color="text.secondary">
          Last updated: {format(lastUpdate, 'HH:mm:ss')}
        </Typography>
        
        <IconButton size="small" onClick={() => window.location.reload()}>
          <Refresh fontSize="small" />
        </IconButton>
      </Box>

      {/* Metrics Grid */}
      <Grid container spacing={2}>
        {metrics.map(renderMetricCard)}
      </Grid>

      {/* Live Events */}
      {!compactMode && renderLiveEvents()}

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </Box>
  );
};