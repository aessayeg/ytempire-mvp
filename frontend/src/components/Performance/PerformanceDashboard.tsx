import React, { useState, useEffect } from 'react';
import { 
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Alert,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper,
  IconButton,
  Tooltip
 } from '@mui/material';
import { 
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
 } from 'recharts';
import RefreshIcon from '@mui/icons-material/Refresh';
import SpeedIcon from '@mui/icons-material/Speed';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import {  authStore  } from '../../stores/authStore';

interface PerformanceMetrics {
  current: {,
  request_rate: number,

    average_latency: number,
  error_rate: number,

    throughput: number};
  historical: Array<{,
  timestamp: string,

    request_rate: number,
  average_latency: number,

    error_rate: number}>;
  slow_endpoints: Array<{,
  endpoint: string,

    method: string,
  avg_duration: number,

    count: number}>;
  error_rates: {
    '4 xx_errors': number;
    '5 xx_errors': number;
    timeout_errors: number,
  total_errors: number};
  database: {,
  average_query_time: number,

    slow_query_count: number,
  connection_pool_usage: number,

    deadlock_count: number};
  system: {,
  cpu_usage: number,

    memory_usage: number,
  disk_usage: number,

    network_io: {,
  bytes_sent: number,

      bytes_recv: number};
  };
}

interface PerformanceAlert {
  type: string,
  severity: 'warning' | 'critical' | 'info',

  message: string,
  timestamp: string}

export const PerformanceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const { token } = authStore();

  const fetchPerformanceData = async () => {
    try {
      const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      };

      // Fetch performance overview
      const overviewResponse = await fetch(`
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/performance/overview`,
        { headers }
      );
      
      if (overviewResponse.ok) {
        const data = await overviewResponse.json();
        setMetrics(data)}

      // Fetch alerts
      const alertsResponse = await fetch(`
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/performance/alerts`,
        { headers }
      );
      
      if (alertsResponse.ok) {
        const alertData = await alertsResponse.json();
        setAlerts(alertData)}
    } catch (_) {
      console.error('Error fetching performance, data:', error)} finally {
      setLoading(false)}
  };

  useEffect(() => {
    
    fetchPerformanceData();
    
    // Auto-refresh every 30 seconds
    const interval = autoRefresh ? setInterval(fetchPerformanceData, 30000) : null;
    
    return () => {
      if (interval) clearInterval(interval)}, [token, autoRefresh]);

  const getStatusColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value < thresholds.good) return '#4 caf50';
    if (value < thresholds.warning) return '#ff9800';
    return '#f44336';
  };

  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  if (loading) {
    return (
    <>
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
      </Box>
    )}

  if (!metrics) {
    return (
    <Alert severity="error">
        Failed to load performance metrics. Please try again later.
      </Alert>
    )}

  // Prepare chart data
  const errorPieData = [
    { name: '4 xx Errors', value: metrics.error_rates['4 xx_errors'], color: '#ff9800' },
    { name: '5 xx Errors', value: metrics.error_rates['5 xx_errors'], color: '#f44336' },
    { name: 'Timeouts', value: metrics.error_rates.timeout_errors, color: '#9 c27 b0' }];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h2">
          Performance Monitoring
        </Typography>
      <Box>
          <Tooltip title="Refresh">
            <IconButton onClick={fetchPerformanceData}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Box sx={{ mb: 3 }}>
          {alerts.map((alert, index) => (
            <Alert 
              key={index}
              severity={alert.severity === 'critical' ? 'error' : alert.severity}
              sx={{ mb: 1 }}
              icon={
                alert.severity === 'critical' ? <ErrorIcon /> :
                alert.severity === 'warning' ? <WarningIcon /> </>:
                <CheckCircleIcon />
              }
            >
              {alert.message}
            </Alert>
          ))}
        </Box>
      )}
      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Request Rate
              </Typography>
              <Typography variant="h4">
                {metrics.current.request_rate.toFixed(0)}
                <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                  req/s
                </Typography>
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Average Latency
              </Typography>
              <Typography 
                variant="h4"
                sx={{ 
                  color: getStatusColor(
                    metrics.current.average_latency,
                    { good: 200, warning: 500 }
                  )
                }}
              >
                {metrics.current.average_latency.toFixed(0)}
                <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                  ms
                </Typography>
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Error Rate
              </Typography>
              <Typography 
                variant="h4"
                sx={{ 
                  color: getStatusColor(
                    metrics.current.error_rate,
                    { good: 1, warning: 5 }
                  )
                }}
              >
                {metrics.current.error_rate.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Throughput
              </Typography>
              <Typography variant="h4">
                {metrics.current.throughput.toFixed(0)}
                <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                  ops/s
                </Typography>
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Trends */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Latency Trend
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics.historical}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="average_latency" 
                  stroke="#8884 d8" 
                  name="Latency (ms)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Request Rate Trend
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={metrics.historical}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="request_rate" 
                  stroke="#82 ca9 d" 
                  fill="#82 ca9 d"
                  name="Requests/s"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* System Resources */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              System Resources
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                CPU Usage
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <LinearProgress 
                  variant="determinate" 
                  value={metrics.system.cpu_usage} 
                  sx={{ flexGrow: 1, mr: 2, height: 8 }}
                  color={metrics.system.cpu_usage > 80 ? 'error' : 'primary'}
                />
                <Typography>{metrics.system.cpu_usage.toFixed(1)}%</Typography>
              </Box>
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                Memory Usage
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <LinearProgress 
                  variant="determinate" 
                  value={metrics.system.memory_usage} 
                  sx={{ flexGrow: 1, mr: 2, height: 8 }}
                  color={metrics.system.memory_usage > 90 ? 'error' : 'primary'}
                />
                <Typography>{metrics.system.memory_usage.toFixed(1)}%</Typography>
              </Box>
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                Disk Usage
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <LinearProgress 
                  variant="determinate" 
                  value={metrics.system.disk_usage} 
                  sx={{ flexGrow: 1, mr: 2, height: 8 }}
                  color={metrics.system.disk_usage > 85 ? 'error' : 'primary'}
                />
                <Typography>{metrics.system.disk_usage.toFixed(1)}%</Typography>
              </Box>
            </Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2" color="textSecondary">
              Network I/O
            </Typography>
            <Typography variant="body2">
              ↑ {formatBytes(metrics.system.network_io.bytes_sent)} | 
              ↓ {formatBytes(metrics.system.network_io.bytes_recv)}
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Database Performance
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Avg Query Time"`
                  secondary={`${metrics.database.average_query_time.toFixed(2)} ms`}
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Slow Queries"
                  secondary={
                    <Chip 
                      label={metrics.database.slow_query_count}
                      color={metrics.database.slow_query_count > 5 ? 'error' : 'default'}
                      size="small"
                    />
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Connection Pool"
                  secondary={
                    <LinearProgress 
                      variant="determinate" 
                      value={metrics.database.connection_pool_usage}
                    />
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Deadlocks"
                  secondary={
                    <Chip 
                      label={metrics.database.deadlock_count}
                      color={metrics.database.deadlock_count > 0 ? 'warning' : 'success'}
                      size="small"
                    />
                  }
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Error Breakdown
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={errorPieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {errorPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Slow Endpoints */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Slowest Endpoints
        </Typography>
        <List>
          {metrics.slow_endpoints.map((endpoint, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip 
                        label={endpoint.method} 
                        size="small" 
                        sx={{ mr: 1 }}
                        color={endpoint.method === 'GET' ? 'success' : 'primary'}
                      />
                      <Typography>{endpoint.endpoint}</Typography>
                    </Box>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                      <Typography variant="body2">
                        Avg: {endpoint.avg_duration.toFixed(2)}s
                      </Typography>
                      <Typography variant="body2">
                        Count: {endpoint.count}
                      </Typography>
                    </Box>
                  }
                />
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <SpeedIcon 
                    sx={{ 
                      color: endpoint.avg_duration > 2 ? '#f44336' : 
                             endpoint.avg_duration > 1 ? '#ff9800' : '#4 caf50'
                    }}
                  />
                </Box>
              </ListItem>
              {index < metrics.slow_endpoints.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      </Paper>
    </Box>
  </>
  )};`