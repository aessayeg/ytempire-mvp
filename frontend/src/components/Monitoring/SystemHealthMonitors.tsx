import React, { useState, useEffect, useRef } from 'react';
import { 
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Alert,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Avatar,
  useTheme,
  FormControl,
  Switch,
  FormControlLabel
 } from '@mui/material';
import { 
  CheckCircle,
  Error,
  Warning,
  Info,
  Speed,
  Memory,
  Storage,
  CloudQueue,
  Security,
  Api,
  Database,
  NetworkCheck,
  VpnKey,
  Refresh,
  Build
 } from '@mui/icons-material';
import { 
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip
 } from 'recharts';
import {  format, subMinutes  } from 'date-fns';

interface Service {
  
id: string;
name: string;

category: string;
status: 'operational' | 'degraded' | 'outage' | 'maintenance';

uptime: number;
responseTime: number;

lastChecked: Date;
icon: React.ReactNode;
endpoint?: string;
errorRate?: number;
dependencies?: string[];


}

interface SystemMetric {
  
timestamp: Date;
cpu: number;

memory: number;
disk: number;

network: number;
requests: number;

errors: number;

}

interface Incident {
  
id: string;
severity: 'low' | 'medium' | 'high' | 'critical';

service: string;
title: string;

description: string;
startTime: Date;
endTime?: Date;
status: 'investigating' | 'identified' | 'monitoring' | 'resolved';
impact: string;

}

export const SystemHealthMonitors: React.FC = () => { const theme = useTheme();
  const [services, setServices] = useState<Service[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Initialize services
    setServices([ {
        id: 'api',
        name: 'API Gateway',
        category: 'Core',
        status: 'operational',
        uptime: 99.99,
        responseTime: 125,
        lastChecked: new Date(),
        icon: <Api />,
        endpoint: 'https://api.ytempire.com',
        errorRate: 0.01 },
      { id: 'database',
        name: 'PostgreSQL',
        category: 'Database',
        status: 'operational',
        uptime: 99.95,
        responseTime: 15,
        lastChecked: new Date(),
        icon: <Database />,
        errorRate: 0.02 },
      { id: 'redis',
        name: 'Redis Cache',
        category: 'Cache',
        status: 'operational',
        uptime: 99.98,
        responseTime: 2,
        lastChecked: new Date(),
        icon: <Memory />,
        errorRate: 0.001 },
      { id: 'celery',
        name: 'Celery Workers',
        category: 'Queue',
        status: 'operational',
        uptime: 99.90,
        responseTime: 500,
        lastChecked: new Date(),
        icon: <CloudQueue />,
        errorRate: 0.05 },
      { id: 'openai',
        name: 'OpenAI API',
        category: 'External',
        status: 'operational',
        uptime: 99.5,
        responseTime: 1200,
        lastChecked: new Date(),
        icon: <Memory />,
        errorRate: 0.1 },
      { id: 'elevenlabs',
        name: 'ElevenLabs API',
        category: 'External',
        status: 'degraded',
        uptime: 98.5,
        responseTime: 2500,
        lastChecked: new Date(),
        icon: <CloudQueue />,
        errorRate: 0.8 },
      { id: 'youtube',
        name: 'YouTube API',
        category: 'External',
        status: 'operational',
        uptime: 99.9,
        responseTime: 450,
        lastChecked: new Date(),
        icon: <Api />,
        errorRate: 0.02 },
      { id: 'storage',
        name: 'S3 Storage',
        category: 'Storage',
        status: 'operational',
        uptime: 99.999,
        responseTime: 200,
        lastChecked: new Date(),
        icon: <Storage />,
        errorRate: 0.001 },
      { id: 'cdn',
        name: 'CloudFront CDN',
        category: 'Network',
        status: 'operational',
        uptime: 99.99,
        responseTime: 50,
        lastChecked: new Date(),
        icon: <NetworkCheck />,
        errorRate: 0.001 },
      { id: 'auth',
        name: 'Authentication',
        category: 'Security',
        status: 'operational',
        uptime: 99.99,
        responseTime: 100,
        lastChecked: new Date(),
        icon: <VpnKey />,
        errorRate: 0.01 } ]);

    // Initialize system metrics
    const initialMetrics = Array.from({ length: 60 }, (_, i) => ({ timestamp: subMinutes(new Date(), 59 - i),
      cpu: 40 + Math.random() * 30,
      memory: 50 + Math.random() * 20,
      disk: 60 + Math.random() * 10,
      network: 30 + Math.random() * 40,
      requests: Math.floor(100 + Math.random() * 50),
      errors: Math.floor(Math.random() * 5) }));
    setSystemMetrics(initialMetrics);

    // Initialize incidents
    setIncidents([ { id: '1',
        severity: 'medium',
        service: 'ElevenLabs API',
        title: 'Increased Response Times',
        description: 'ElevenLabs API experiencing higher than normal response times',
        startTime: new Date(Date.now() - 1000 * 60 * 30),
        status: 'monitoring',
        impact: 'Voice synthesis may take longer than usual' },
      { id: '2',
        severity: 'low',
        service: 'Celery Workers',
        title: 'Scheduled Maintenance',
        description: 'Routine maintenance window for worker updates',
        startTime: new Date(Date.now() + 1000 * 60 * 60 * 2),
        status: 'identified',
        impact: 'Video processing capacity reduced by 20% during maintenance' } ])}, []);

  useEffect(() => { if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        // Update services with random changes
        setServices(prev => prev.map(service => ({
          ...service,
          responseTime: Math.max(1, service.responseTime + (Math.random() - 0.5) * 20),
          lastChecked: new Date(),
          errorRate: Math.max(0, (service.errorRate || 0) + (Math.random() - 0.5) * 0.01) })));

        // Add new metric point
        setSystemMetrics(prev => { const newMetric = {
            timestamp: new Date(),
            cpu: 40 + Math.random() * 30,
            memory: 50 + Math.random() * 20,
            disk: 60 + Math.random() * 10,
            network: 30 + Math.random() * 40,
            requests: Math.floor(100 + Math.random() * 50),
            errors: Math.floor(Math.random() * 5)
};
          return [...prev.slice(1), newMetric]})}, refreshInterval * 1000)}

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)}
    }
  }, [autoRefresh, refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return 'success';
      case 'degraded': return 'warning';
      case 'outage': return 'error';
      case 'maintenance': return 'info';
      default: return 'default'}
  };
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return <CheckCircle color="success" />;
      case 'degraded': return <Warning color="warning" />;
      case 'outage': return <Error color="error" />;
      case 'maintenance': return <Build color="info" />;
      default: return <Info />}
  };
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default'}
  };
  const getUptimeColor = (uptime: number) => {
    if (uptime >= 99.9) return theme.palette.success.main;
    if (uptime >= 99) return theme.palette.warning.main;
    return theme.palette.error.main
  };
  const operationalCount = services.filter(s => s.status === 'operational').length;
  const overallStatus = services.every(s => s.status === 'operational')
    ? 'operational'
    : services.some(s => s.status === 'outage')
    ? 'partial outage'
    : 'degraded';

  const latestMetric = systemMetrics[systemMetrics.length - 1] || { cpu: 0;
    memory: 0;
    disk: 0;
    network: 0;
    requests: 0;
    errors: 0 };
  return (
    <>
      <Box>
      {/* Overall Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {overallStatus === 'operational' ? (
                <CheckCircle color="success" sx={{ fontSize: 48 }} />
              ) : overallStatus === 'partial outage' ? (
                <Error color="error" sx={{ fontSize: 48 }} />
              ) : (
                <Warning color="warning" sx={{ fontSize: 48 }} />
              )}
              <Box>
                <Typography variant="h5" fontWeight="bold">
                  System Status: {overallStatus === 'operational' ? 'All Systems Operational' : overallStatus === 'degraded' ? 'Degraded Performance' : 'Partial Outage'}
                </Typography>
      <Typography variant="body2" color="text.secondary">
                  {operationalCount} of {services.length} services operational
                </Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked}
                  />
                }
                label={`Auto-refresh (${refreshInterval}s)`}
              />
              <Button variant="outlined" startIcon={<Refresh />}>
                Refresh Now
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* System Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Speed fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="subtitle2">CPU Usage</Typography>
            </Box>
            <Typography variant="h4" fontWeight="bold">
              {latestMetric.cpu.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={latestMetric.cpu}
              color={latestMetric.cpu > 80 ? 'error' : latestMetric.cpu > 60 ? 'warning' : 'primary'}
            />
          </Paper>
        </Grid>

        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Memory fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Memory</Typography>
            </Box>
            <Typography variant="h4" fontWeight="bold">
              {latestMetric.memory.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={latestMetric.memory}
              color={latestMetric.memory > 80 ? 'error' : latestMetric.memory > 60 ? 'warning' : 'primary'}
            />
          </Paper>
        </Grid>

        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Storage fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Disk Usage</Typography>
            </Box>
            <Typography variant="h4" fontWeight="bold">
              {latestMetric.disk.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={latestMetric.disk}
              color={latestMetric.disk > 80 ? 'error' : latestMetric.disk > 60 ? 'warning' : 'primary'}
            />
          </Paper>
        </Grid>

        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <NetworkCheck fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Network I/O</Typography>
            </Box>
            <Typography variant="h4" fontWeight="bold">
              {latestMetric.network.toFixed(1)} Mbps
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min(100, latestMetric.network)}
              color="primary"
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Services Status Grid */}
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Service Status
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {services.map(service => (
          <Grid item xs={12} sm={6} md={4} key={service.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: 'action.selected', mr: 2 }}>
                    {service.icon}
                  </Avatar>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle1" fontWeight="medium">
                      {service.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {service.category}
                    </Typography>
                  </Box>
                  {getStatusIcon(service.status)}
                </Box>

                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Uptime
                    </Typography>
                    <Typography variant="body2" fontWeight="bold" color={getUptimeColor(service.uptime)}>
                      {service.uptime.toFixed(2)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Response Time
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {service.responseTime}ms
                    </Typography>
                  </Grid>
                  {service.errorRate !== undefined && (
                    <>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Error Rate
                        </Typography>
                        <Typography variant="body2" fontWeight="bold" color={service.errorRate > 0.5 ? 'error' : 'inherit'}>
                          {service.errorRate.toFixed(2)}%
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Last Check
                        </Typography>
                        <Typography variant="body2">
                          {format(service.lastChecked, 'HH:mm:ss')}
                        </Typography>
                      </Grid>
                    </>
                  )}
                </Grid>

                {service.status !== 'operational' && (
                  <Alert severity={getStatusColor(service.status) as any} sx={{ mt: 2 }}>
                    <Typography variant="caption">
                      {service.status === 'degraded' ? 'Service experiencing degraded performance' :
                       service.status === 'outage' ? 'Service is currently unavailable' :
                       'Service under maintenance'}
                    </Typography>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* System Metrics Chart */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            System Performance (Last 60, Minutes)
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={systemMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={timestamp) => format(timestamp, 'HH:mm'}
              />
              <YAxis />
              <RechartsTooltip
                labelFormatter={timestamp) => format(timestamp, 'HH:mm:ss'}
              />
              <Line type="monotone" dataKey="cpu" stroke={theme.palette.primary.main} name="CPU %" dot={false} />
              <Line type="monotone" dataKey="memory" stroke={theme.palette.secondary.main} name="Memory %" dot={false} />
              <Line type="monotone" dataKey="disk" stroke={theme.palette.success.main} name="Disk %" dot={false} />
              <Line type="monotone" dataKey="network" stroke={theme.palette.warning.main} name="Network Mbps" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Active Incidents */}
      {incidents.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Active Incidents
            </Typography>
            <List>
              {incidents.map(incident => (
                <ListItem key={incident.id}>
                  <ListItemIcon>
                    <Badge
                      badgeContent={incident.severity}
                      color={getSeverityColor(incident.severity) as any}
                    >
                      <Warning />
                    </Badge>
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1" fontWeight="medium">
                          {incident.title}
                        </Typography>
                        <Chip
                          label={incident.status}
                          size="small"
                          color={incident.status === 'resolved' ? 'success' : 'warning'}
                        />
                      </Box>}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Service: {incident.service} • {incident.description}
                        </Typography>
                        <Typography variant="caption" color="warning.main">
                          Impact: {incident.impact}
                        </Typography>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Started: {format(incident.startTime, 'PPp')}
                        </Typography>
                      </Box>}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}
    </Box>
  </>
  )};
