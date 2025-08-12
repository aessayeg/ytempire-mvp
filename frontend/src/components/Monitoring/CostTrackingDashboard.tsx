import React, { useState, useEffect } from 'react';
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
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Tooltip,
  Avatar,
  useTheme,
  Divider,
} from '@mui/material';
import {
  AttachMoney,
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Settings,
  Download,
  Refresh,
  NotificationImportant,
  Speed,
  CloudQueue,
  Memory,
  Storage,
  Timeline,
  PieChart,
  BarChart,
  ShowChart,
  ArrowUpward,
  ArrowDownward,
  Info,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  BarChart as RechartsBarChart,
  Bar,
} from 'recharts';
import { format, subDays, startOfDay, endOfDay } from 'date-fns';

interface CostMetric {
  service: string;
  icon: React.ReactNode;
  currentCost: number;
  previousCost: number;
  budget: number;
  usage: number;
  trend: 'up' | 'down' | 'stable';
  color: string;
}

interface ServiceCost {
  id: string;
  name: string;
  provider: string;
  category: string;
  costToday: number;
  costYesterday: number;
  costThisMonth: number;
  callsToday: number;
  avgCostPerCall: number;
  status: 'normal' | 'warning' | 'critical';
}

interface CostAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  timestamp: Date;
  action?: string;
}

interface BudgetLimit {
  service: string;
  daily: number;
  monthly: number;
  alertThreshold: number;
}

export const CostTrackingDashboard: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('today');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedService, setSelectedService] = useState<ServiceCost | null>(null);
  
  const [costMetrics] = useState<CostMetric[]>([
    {
      service: 'OpenAI',
      icon: <Memory />,
      currentCost: 45.67,
      previousCost: 38.92,
      budget: 50.00,
      usage: 91.34,
      trend: 'up',
      color: theme.palette.primary.main,
    },
    {
      service: 'ElevenLabs',
      icon: <CloudQueue />,
      currentCost: 18.23,
      previousCost: 20.15,
      budget: 20.00,
      usage: 91.15,
      trend: 'down',
      color: theme.palette.secondary.main,
    },
    {
      service: 'Google Cloud',
      icon: <Storage />,
      currentCost: 8.45,
      previousCost: 7.82,
      budget: 10.00,
      usage: 84.50,
      trend: 'up',
      color: theme.palette.success.main,
    },
    {
      service: 'YouTube API',
      icon: <CloudQueue />,
      currentCost: 2.15,
      previousCost: 2.15,
      budget: 5.00,
      usage: 43.00,
      trend: 'stable',
      color: theme.palette.warning.main,
    },
  ]);

  const [serviceCosts] = useState<ServiceCost[]>([
    {
      id: '1',
      name: 'GPT-4 Turbo',
      provider: 'OpenAI',
      category: 'Script Generation',
      costToday: 32.45,
      costYesterday: 28.90,
      costThisMonth: 945.67,
      callsToday: 234,
      avgCostPerCall: 0.14,
      status: 'warning',
    },
    {
      id: '2',
      name: 'GPT-3.5 Turbo',
      provider: 'OpenAI',
      category: 'Script Generation',
      costToday: 13.22,
      costYesterday: 10.02,
      costThisMonth: 312.45,
      callsToday: 567,
      avgCostPerCall: 0.02,
      status: 'normal',
    },
    {
      id: '3',
      name: 'ElevenLabs Voice',
      provider: 'ElevenLabs',
      category: 'Voice Synthesis',
      costToday: 18.23,
      costYesterday: 20.15,
      costThisMonth: 523.89,
      callsToday: 89,
      avgCostPerCall: 0.20,
      status: 'warning',
    },
    {
      id: '4',
      name: 'DALL-E 3',
      provider: 'OpenAI',
      category: 'Image Generation',
      costToday: 5.67,
      costYesterday: 4.23,
      costThisMonth: 156.78,
      callsToday: 45,
      avgCostPerCall: 0.13,
      status: 'normal',
    },
  ]);

  const [alerts] = useState<CostAlert[]>([
    {
      id: '1',
      severity: 'high',
      title: 'OpenAI Daily Budget Alert',
      description: 'OpenAI costs at 91% of daily budget ($45.67/$50.00)',
      timestamp: new Date(),
      action: 'Consider switching to GPT-3.5 for remaining videos today',
    },
    {
      id: '2',
      severity: 'medium',
      title: 'Cost Optimization Opportunity',
      description: 'Switching 30% of GPT-4 calls to GPT-3.5 could save $15/day',
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      action: 'Review script generation settings',
    },
    {
      id: '3',
      severity: 'low',
      title: 'Monthly Projection Update',
      description: 'Current spending trend projects $2,850 for this month',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2),
    },
  ]);

  // Mock data for charts
  const dailyCostData = Array.from({ length: 7 }, (_, i) => ({
    date: format(subDays(new Date(), 6 - i), 'MM/dd'),
    openai: 35 + Math.random() * 15,
    elevenlabs: 15 + Math.random() * 8,
    google: 5 + Math.random() * 5,
    total: 0,
  })).map(d => ({ ...d, total: d.openai + d.elevenlabs + d.google }));

  const costBreakdown = [
    { name: 'Script Generation', value: 45.67, percentage: 45 },
    { name: 'Voice Synthesis', value: 18.23, percentage: 18 },
    { name: 'Image Generation', value: 8.45, percentage: 8 },
    { name: 'Video Rendering', value: 12.34, percentage: 12 },
    { name: 'API Calls', value: 5.67, percentage: 6 },
    { name: 'Storage', value: 10.64, percentage: 11 },
  ];

  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
  ];

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp color="error" fontSize="small" />;
      case 'down': return <TrendingDown color="success" fontSize="small" />;
      default: return <ArrowUpward color="action" fontSize="small" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const totalCostToday = costMetrics.reduce((sum, m) => sum + m.currentCost, 0);
  const totalBudget = costMetrics.reduce((sum, m) => sum + m.budget, 0);
  const budgetUsagePercent = (totalCostToday / totalBudget) * 100;

  return (
    <Box>
      {/* Header Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  <AttachMoney />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    ${totalCostToday.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Cost Today
                  </Typography>
                </Box>
                {getTrendIcon('up')}
              </Box>
              <LinearProgress
                variant="determinate"
                value={budgetUsagePercent}
                color={budgetUsagePercent > 90 ? 'error' : budgetUsagePercent > 70 ? 'warning' : 'primary'}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {budgetUsagePercent.toFixed(1)}% of daily budget
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                  <Speed />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    $0.97
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Cost per Video
                  </Typography>
                </Box>
                <Chip label="-12%" size="small" color="success" />
              </Box>
              <Typography variant="caption" color="success.main">
                Optimized from $1.10 yesterday
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                  <Timeline />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    $2,850
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monthly Projection
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary">
                Based on current usage trends
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'error.main', mr: 2 }}>
                  <NotificationImportant />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    {alerts.filter(a => a.severity === 'high' || a.severity === 'critical').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Alerts
                  </Typography>
                </Box>
              </Box>
              <Button size="small" variant="outlined">
                View All
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Service Cost Breakdown */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {costMetrics.map((metric) => (
          <Grid item xs={12} sm={6} md={3} key={metric.service}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar sx={{ bgcolor: `${metric.color}20`, color: metric.color, width: 32, height: 32 }}>
                    {metric.icon}
                  </Avatar>
                  <Typography variant="subtitle2" sx={{ ml: 1, flex: 1 }}>
                    {metric.service}
                  </Typography>
                  {metric.usage > 90 && (
                    <Warning color="warning" fontSize="small" />
                  )}
                </Box>
                
                <Typography variant="h6" fontWeight="bold">
                  ${metric.currentCost.toFixed(2)}
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                  {getTrendIcon(metric.trend)}
                  <Typography variant="caption" color={metric.trend === 'up' ? 'error.main' : 'success.main'}>
                    {metric.trend === 'up' ? '+' : '-'}
                    {Math.abs(metric.currentCost - metric.previousCost).toFixed(2)} from yesterday
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 0.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption">Budget</Typography>
                    <Typography variant="caption" fontWeight="bold">
                      ${metric.budget.toFixed(2)}
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={metric.usage}
                    color={metric.usage > 90 ? 'error' : metric.usage > 70 ? 'warning' : 'primary'}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Charts and Tables */}
      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 2 }}>
        <Tab label="Overview" />
        <Tab label="Service Details" />
        <Tab label="Trends" />
        <Tab label="Optimization" />
      </Tabs>

      {/* Overview Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  7-Day Cost Trend
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={dailyCostData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Area type="monotone" dataKey="openai" stackId="1" stroke={COLORS[0]} fill={COLORS[0]} />
                    <Area type="monotone" dataKey="elevenlabs" stackId="1" stroke={COLORS[1]} fill={COLORS[1]} />
                    <Area type="monotone" dataKey="google" stackId="1" stroke={COLORS[2]} fill={COLORS[2]} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  Cost Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={costBreakdown}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry) => `${entry.percentage}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {costBreakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
                
                <Box sx={{ mt: 2 }}>
                  {costBreakdown.map((item, index) => (
                    <Box key={item.name} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          bgcolor: COLORS[index % COLORS.length],
                          mr: 1,
                        }}
                      />
                      <Typography variant="caption" sx={{ flex: 1 }}>
                        {item.name}
                      </Typography>
                      <Typography variant="caption" fontWeight="bold">
                        ${item.value.toFixed(2)}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Alerts */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  Cost Alerts
                </Typography>
                {alerts.map(alert => (
                  <Alert
                    key={alert.id}
                    severity={getSeverityColor(alert.severity) as any}
                    sx={{ mb: 1 }}
                    action={
                      alert.action && (
                        <Button size="small">
                          Take Action
                        </Button>
                      )
                    }
                  >
                    <Typography variant="subtitle2" fontWeight="bold">
                      {alert.title}
                    </Typography>
                    <Typography variant="body2">
                      {alert.description}
                    </Typography>
                    {alert.action && (
                      <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                        Recommendation: {alert.action}
                      </Typography>
                    )}
                  </Alert>
                ))}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Service Details Tab */}
      {tabValue === 1 && (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Service</TableCell>
                <TableCell>Provider</TableCell>
                <TableCell>Category</TableCell>
                <TableCell align="right">Cost Today</TableCell>
                <TableCell align="right">Yesterday</TableCell>
                <TableCell align="right">This Month</TableCell>
                <TableCell align="right">Calls Today</TableCell>
                <TableCell align="right">Avg Cost/Call</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {serviceCosts.map(service => (
                <TableRow key={service.id}>
                  <TableCell>{service.name}</TableCell>
                  <TableCell>{service.provider}</TableCell>
                  <TableCell>{service.category}</TableCell>
                  <TableCell align="right">${service.costToday.toFixed(2)}</TableCell>
                  <TableCell align="right">${service.costYesterday.toFixed(2)}</TableCell>
                  <TableCell align="right">${service.costThisMonth.toFixed(2)}</TableCell>
                  <TableCell align="right">{service.callsToday}</TableCell>
                  <TableCell align="right">${service.avgCostPerCall.toFixed(3)}</TableCell>
                  <TableCell>
                    <Chip
                      label={service.status}
                      size="small"
                      color={service.status === 'warning' ? 'warning' : service.status === 'critical' ? 'error' : 'success'}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};