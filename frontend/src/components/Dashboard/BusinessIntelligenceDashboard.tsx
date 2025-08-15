/**
 * Business Intelligence Dashboard
 * Executive-level dashboard with comprehensive business metrics, financial reports, and strategic insights
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Paper,
  Tabs,
  Tab,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  IconButton,
  Chip,
  Alert,
  AlertTitle,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  CircularProgress,
  Avatar,
  Tooltip,
  Switch,
  FormControlLabel
 } from '@mui/material';
import { 
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  AttachMoney,
  People,
  VideoLibrary,
  Analytics,
  Speed,
  Info,
  Refresh,
  Download,
  PieChart,
  BarChart as BarChartIcon,
  Star,
  AccountBalance,
  Business,
  Growth,
  Lightbulb
 } from '@mui/icons-material';
import { 
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  LabelList
 } from 'recharts';
import {  useOptimizedStore  } from '../../stores/optimizedStore';
import {  api  } from '../../services/api';

interface ExecutiveMetric {
  id: string,
  name: string,

  value: number | string,
  previousValue: number | string,

  change: number,
  changePercent: number,

  trend: 'up' | 'down' | 'flat',
  status: 'excellent' | 'good' | 'warning' | 'critical';
  target?: number;
  unit: string,
  category: 'revenue' | 'growth' | 'efficiency' | 'quality',

  description: string,
  icon: React.ReactNode,

  color: string}

interface BusinessKPI {
  id: string,
  name: string,

  current: number,
  target: number,

  benchmark: number,
  trend: number[],

  status: 'on_track' | 'at_risk' | 'critical',
  category: string,

  unit: string}

interface FinancialMetric {
  period: string,
  revenue: number,

  costs: number,
  profit: number,

  margin: number,
  users: number,

  arpu: number,
  ltv: number,

  cac: number}

interface UserSegment {
  segment: string,
  count: number,

  percentage: number,
  revenue: number,

  avgLifetime: number,
  churnRate: number,

  growthRate: number}

interface CompetitiveMetric {
  metric: string,
  ourValue: number,

  industry: number,
  leader: number,

  position: string}

const CHART_COLORS = { primary: '#1976 d2',
  secondary: '#dc004 e',
  success: '#2 e7 d32',
  warning: '#ed6 c02',
  error: '#d32 f2 f',
  info: '#0288 d1',
  revenue: '#4 caf50',
  costs: '#f44336',
  profit: '#2 e7 d32',
  users: '#3 f51 b5' };

const TIME_PERIODS = [ { value: '24 h', label: '24 Hours' },
  { value: '7 d', label: '7 Days' },
  { value: '30 d', label: '30 Days' },
  { value: '90 d', label: '90 Days' },
  { value: '1 y', label: '1 Year' } ];

const DASHBOARD_TABS = [ 'Executive Overview',
  'Financial Performance',
  'Growth Analytics',
  'User Intelligence',
  'Operational Metrics',
  'Strategic Insights' ];

export const BusinessIntelligenceDashboard: React.FC = () => {
  // State management
  const [selectedTab, setSelectedTab] = useState(0);
  const [timePeriod, setTimePeriod] = useState('30 d');
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showTargets, setShowTargets] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  
  // Data state
  const [executiveMetrics, setExecutiveMetrics] = useState<ExecutiveMetric[]>([]);
  const [businessKPIs, setBusinessKPIs] = useState<BusinessKPI[]>([]);
  const [financialData, setFinancialData] = useState<FinancialMetric[]>([]);
  const [userSegments, setUserSegments] = useState<UserSegment[]>([]);
  const [competitiveMetrics, setCompetitiveMetrics] = useState<CompetitiveMetric[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [insights, setInsights] = useState<any[]>([]);

  const { addNotification } = useOptimizedStore();

  // Fetch comprehensive business intelligence data
  const fetchBIData = useCallback(_async () => {
    try {
      setLoading(true);

      // Parallel API calls for executive dashboard
      const [ executiveResponse,
        kpiResponse,
        financialResponse,
        segmentResponse,
        competitiveResponse,
        alertsResponse,
        insightsResponse ] = await Promise.all([ api.get(`/bi/executive-metrics?period=${timePeriod}`),`
        api.get(`/bi/business-kpis?period=${timePeriod}`),`
        api.get(`/bi/financial-performance?period=${timePeriod}`),`
        api.get(`/bi/user-segments?period=${timePeriod}`),`
        api.get(`/bi/competitive-analysis`),`
        api.get(`/bi/alerts`),`
        api.get(`/bi/insights?period=${timePeriod}`) ]);

      // Process executive metrics
      const metrics: ExecutiveMetric[] = [ { id: 'mrr',
          name: 'Monthly Recurring Revenue',
          value: executiveResponse.data.mrr || 0,
          previousValue: executiveResponse.data.previous_mrr || 0,
          change: executiveResponse.data.mrr_change || 0,
          changePercent: executiveResponse.data.mrr_change_percent || 0,
          trend: executiveResponse.data.mrr_change >= 0 ? 'up' : 'down',
          status: executiveResponse.data.mrr >= 10000 ? 'excellent' : 'good',
          target: 10000,
          unit: '$',
          category: 'revenue',
          description: 'Monthly recurring revenue from subscriptions',
          icon: <AttachMoney />,
          color: CHART_COLORS.revenue },
        { id: 'arr',
          name: 'Annual Recurring Revenue',
          value: executiveResponse.data.arr || 0,
          previousValue: executiveResponse.data.previous_arr || 0,
          change: executiveResponse.data.arr_change || 0,
          changePercent: executiveResponse.data.arr_change_percent || 0,
          trend: executiveResponse.data.arr_change >= 0 ? 'up' : 'down',
          status: executiveResponse.data.arr >= 120000 ? 'excellent' : 'good',
          target: 120000,
          unit: '$',
          category: 'revenue',
          description: 'Annual recurring revenue projection',
          icon: <AccountBalance />,
          color: CHART_COLORS.success },
        { id: 'active_users',
          name: 'Monthly Active Users',
          value: executiveResponse.data.mau || 0,
          previousValue: executiveResponse.data.previous_mau || 0,
          change: executiveResponse.data.mau_change || 0,
          changePercent: executiveResponse.data.mau_change_percent || 0,
          trend: executiveResponse.data.mau_change >= 0 ? 'up' : 'down',
          status: executiveResponse.data.mau >= 100 ? 'excellent' : 'good',
          target: 100,
          unit: 'users',
          category: 'growth',
          description: 'Monthly active user count',
          icon: <People />,
          color: CHART_COLORS.users },
        { id: 'videos_generated',
          name: 'Videos Generated',
          value: executiveResponse.data.videos_generated || 0,
          previousValue: executiveResponse.data.previous_videos || 0,
          change: executiveResponse.data.videos_change || 0,
          changePercent: executiveResponse.data.videos_change_percent || 0,
          trend: executiveResponse.data.videos_change >= 0 ? 'up' : 'down',
          status: executiveResponse.data.videos_generated >= 1000 ? 'excellent' : 'good',
          target: 1000,
          unit: 'videos',
          category: 'efficiency',
          description: 'Total videos generated by platform',
          icon: <VideoLibrary />,
          color: CHART_COLORS.primary },
        { id: 'cost_per_video',
          name: 'Average Cost per Video',
          value: executiveResponse.data.avg_cost_per_video || 0,
          previousValue: executiveResponse.data.previous_cost_per_video || 0,
          change: executiveResponse.data.cost_change || 0,
          changePercent: executiveResponse.data.cost_change_percent || 0,
          trend: executiveResponse.data.cost_change <= 0 ? 'up' : 'down', // Lower cost is better
          status: executiveResponse.data.avg_cost_per_video <= 2 ? 'excellent' : 'warning',
          target: 2,
          unit: '$',
          category: 'efficiency',
          description: 'Average cost to generate one video',
          icon: <Speed />,
          color: CHART_COLORS.warning },
        { id: 'quality_score',
          name: 'Content Quality Score',
          value: executiveResponse.data.quality_score || 0,
          previousValue: executiveResponse.data.previous_quality || 0,
          change: executiveResponse.data.quality_change || 0,
          changePercent: executiveResponse.data.quality_change_percent || 0,
          trend: executiveResponse.data.quality_change >= 0 ? 'up' : 'down',
          status: executiveResponse.data.quality_score >= 8 ? 'excellent' : 'good',
          target: 8,
          unit: '/10',
          category: 'quality',
          description: 'Average quality score of generated content',
          icon: <Star />,
          color: CHART_COLORS.secondary } ];

      setExecutiveMetrics(metrics);
      setBusinessKPIs(kpiResponse.data || []);
      setFinancialData(financialResponse.data || []);
      setUserSegments(segmentResponse.data || []);
      setCompetitiveMetrics(competitiveResponse.data || []);
      setAlerts(alertsResponse.data || []);
      setInsights(insightsResponse.data || []);
      
      setLoading(false)} catch (_) { console.error('Failed to fetch BI, data:', error);
      addNotification({
        type: 'error',
        message: 'Failed to load business intelligence data' });
      setLoading(false)}
  }, [timePeriod, addNotification]);

  // Effects
  useEffect(() => {
    fetchBIData()}, [fetchBIData]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(fetchBIData, 5 * 60 * 1000); // 5 minutes
    return () => clearInterval(interval)}, [autoRefresh, fetchBIData]);

  // Render executive metric card
  const renderExecutiveMetric = (metric: ExecutiveMetric) => {
    const getStatusColor = () => {
      switch (metric.status) {
        case 'excellent': return 'success';
        case 'good': return 'info';
        case 'warning': return 'warning';
        case 'critical': return 'error';
        default: return 'info'}
    };

    const formatValue = (value: number | string) => {
      if (typeof value === 'string') return value;`
      if (metric.unit === '$') return `$${value.toLocaleString()}`;`
      return `${value.toLocaleString()}${metric.unit}`;
    };

    const getTrendIcon = () => {
      switch (metric.trend) {
        case 'up':
          return <TrendingUp sx={{ color: metric.category === 'efficiency' && metric.id === 'cost_per_video' ? 'error.main' : 'success.main' }} />;
        case 'down':
          return <TrendingDown sx={{ color: metric.category === 'efficiency' && metric.id === 'cost_per_video' ? 'success.main' : 'error.main' }} />;
        default:
          return <TrendingFlat color="action" />}
    };

    return (
    <>
      <Card key={metric.id} sx={{ height: '100%' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Box display="flex" alignItems="center">
              <Avatar sx={{ bgcolor: metric.color, width: 40, height: 40, mr: 2 }}>
                {metric.icon}
              </Avatar>
      <Box>
                <Typography variant="body2" color="text.secondary">
                  {metric.name}
                </Typography>
                <Typography variant="h5" fontWeight="bold" sx={{ color: metric.color }}>
                  {formatValue(metric.value)}
                </Typography>
              </Box>
            </Box>
            <Chip 
              label={metric.status.toUpperCase()}
              color={getStatusColor()}
              size="small"
              variant="outlined"
            />
          </Box>
          
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Box display="flex" alignItems="center">
              {getTrendIcon()}
              <Typography 
                variant="body2" 
                sx={{ 
                  ml: 0.5,
                  color: metric.changePercent >= 0 ? 'success.main' : 'error.main',

                }}
              >
                {metric.changePercent >= 0 ? '+' : ''}{metric.changePercent.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                vs prev period
              </Typography>
            </Box>
          </Box>

          {showTargets && metric.target && (
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                <Typography variant="caption" color="text.secondary">
                  Target: {formatValue(metric.target)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {Math.round((Number(metric.value) / metric.target) * 100)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((Number(metric.value) / metric.target) * 100, 100)}
                sx={{
                  height: 6,
                  borderRadius: 3,`
                  backgroundColor: `${metric.color}20`,
                  '& .MuiLinearProgress-bar': { backgroundColor: metric.color }
                }}
              />
            </Box>
          )}
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            {metric.description}
          </Typography>
        </CardContent>
      </Card>
    </>
  )};

  // Render financial performance chart
  const renderFinancialChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={financialData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="period" />
        <YAxis yAxisId="left" />
        <YAxis yAxisId="right" orientation="right" />
        <RechartsTooltip />
        <Legend />
        <Bar yAxisId="left" dataKey="revenue" fill={CHART_COLORS.revenue} name="Revenue ($)" />
        <Bar yAxisId="left" dataKey="costs" fill={CHART_COLORS.costs} name="Costs ($)" />
        <Line yAxisId="right" type="monotone" dataKey="margin" stroke={CHART_COLORS.profit} strokeWidth={3} name="Profit Margin (%)" />
      </ComposedChart>
    </ResponsiveContainer>
  );

  // Render user segments chart
  const renderUserSegmentsChart = () => { const segmentData = userSegments.map(segment => ({
      name: segment.segment,
      value: segment.percentage,
      count: segment.count,
      revenue: segment.revenue }));

    const COLORS = ['#0088 FE', '#00 C49 F', '#FFBB28', '#FF8042', '#8884 D8'];

    return (
    <>
      <ResponsiveContainer width="100%" height={300}>
        <RechartsPie data={segmentData} cx="50%" cy="50%" outerRadius={100} dataKey="value">
          {segmentData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
          <LabelList dataKey="name" position="outside" />
        </RechartsPie>
      </ResponsiveContainer>
    </>
  )};

  // Render KPI dashboard
  const renderKPIDashboard = () => (
    <Grid container spacing={2}>
      {businessKPIs.map((kpi) => (
        <Grid item xs={12} sm={6} md={4} key={kpi.id}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {kpi.name}
              </Typography>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="h6" fontWeight="bold">
                  {kpi.current.toLocaleString()}{kpi.unit}
                </Typography>
                <Chip 
                  label={kpi.status.replace('_', ' ').toUpperCase()}
                  color={kpi.status === 'on_track' ? 'success' : kpi.status === 'at_risk' ? 'warning' : 'error'}
                  size="small"
                />
              </Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="caption" color="text.secondary">
                  Target: {kpi.target.toLocaleString()}{kpi.unit}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Industry: {kpi.benchmark.toLocaleString()}{kpi.unit}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={(kpi.current / kpi.target) * 100}
                color={kpi.status === 'on_track' ? 'success' : kpi.status === 'at_risk' ? 'warning' : 'error'}
                sx={{ mb: 1 }}
              />
              <ResponsiveContainer width="100%" height={60}>
                <LineChart data={kpi.trend.map((value, index) => ({ index, value }))}>
                  <Line type="monotone" dataKey="value" stroke="#8884 d8" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  // Render competitive analysis
  const renderCompetitiveAnalysis = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Metric</TableCell>
            <TableCell align="right">Our Value</TableCell>
            <TableCell align="right">Industry Average</TableCell>
            <TableCell align="right">Market Leader</TableCell>
            <TableCell>Position</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {competitiveMetrics.map((metric, index) => (
            <TableRow key={index}>
              <TableCell>{metric.metric}</TableCell>
              <TableCell align="right">{metric.ourValue.toLocaleString()}</TableCell>
              <TableCell align="right">{metric.industry.toLocaleString()}</TableCell>
              <TableCell align="right">{metric.leader.toLocaleString()}</TableCell>
              <TableCell>
                <Chip 
                  label={metric.position} 
                  color={
                    metric.position === 'Leading' ? 'success' : 
                    metric.position === 'Competitive' ? 'info' : 'warning'
                  }
                  size="small"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  // Render strategic insights
  const renderStrategicInsights = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        Strategic Insights & Recommendations
      </Typography>
      <Grid container spacing={2}>
        {insights.map((insight, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <Lightbulb color="primary" sx={{ mr: 1 }} />
                  <Typography variant="subtitle1" fontWeight="bold">
                    {insight.title}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {insight.description}
                </Typography>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Impact: {insight.impact} • Effort: {insight.effort}
                  </Typography>
                </Box>
                <Box mt={1}>
                  {insight.tags?.map((tag: string, tagIndex: number) => (
                    <Chip key={tagIndex} label={tag} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // Render alerts section
  const renderAlerts = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Business Alerts
      </Typography>
      {alerts.length === 0 ? (
        <Alert severity="success">
          <AlertTitle>All Systems Operational</AlertTitle>
          No critical business alerts at this time.
        </Alert>
      ) : (
        alerts.map((alert, index) => (
          <Alert 
            key={index} 
            severity={alert.severity || 'info'} 
            sx={{ mb: 1 }}
            action={
              <IconButton size="small">
                <Info />
              </IconButton>
            }
          >
            <AlertTitle>{alert.title}</AlertTitle>
            {alert.message}
          </Alert>
        ))
      )}
    </Box>
  );

  if (loading) {
    return (
    <>
      <Box p={3}>
        <Grid container spacing={3}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Grid item xs={12} md={6} lg={4} key={i}>
              <Skeleton variant="rectangular" height={200} />
            </Grid>
          ))}
        </Grid>
      </Box>
    )}

  return (
    <Box p={3}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs>
            <Typography variant="h4" fontWeight="bold">
              Business Intelligence Dashboard
            </Typography>
      <Typography variant="body2" color="text.secondary">
              Executive, insights, financial, performance, and strategic analytics
            </Typography>
          </Grid>
          
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Period</InputLabel>
              <Select
                value={timePeriod}
                label="Time Period"
                onChange={(e) => setTimePeriod(e.target.value)}
              >
                {TIME_PERIODS.map((period) => (
                  <MenuItem key={period.value} value={period.value}>
                    {period.label}
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
            <Button onClick={fetchBIData} startIcon={<Refresh />}>
              Refresh
            </Button>
          </Grid>
          
          <Grid item>
            <Button startIcon={<Download />} variant="outlined">
              Export Report
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Alerts */}
      {renderAlerts()}
      {/* Executive Metrics */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Executive Metrics
        </Typography>
        <Grid container spacing={3}>
          {executiveMetrics.map((metric) => (
            <Grid item xs={12} sm={6} lg={4} key={metric.id}>
              {renderExecutiveMetric(metric)}
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Tabbed Content */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={selectedTab} 
          onChange={(e, newValue) => setSelectedTab(newValue)}
          scrollButtons="auto"
          variant="scrollable"
        >
          {DASHBOARD_TABS.map((tab, index) => (
            <Tab key={index} label={tab} />
          ))}
        </Tabs>

        <Box p={3}>
          {selectedTab === 0 && renderKPIDashboard()}
          {selectedTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Financial Performance
              </Typography>
              {renderFinancialChart()}
            </Box>
          )}
          {selectedTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Growth Analytics
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="User Growth Trend" />
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={financialData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="period" />
                          <YAxis />
                          <RechartsTooltip />
                          <Area type="monotone" dataKey="users" stroke={CHART_COLORS.users} fill={CHART_COLORS.users} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="Revenue Growth" />
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={financialData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="period" />
                          <YAxis />
                          <RechartsTooltip />
                          <Line type="monotone" dataKey="revenue" stroke={CHART_COLORS.revenue} strokeWidth={3} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}
          {selectedTab === 3 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                User Intelligence
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="User Segments" />
                    <CardContent>
                      {renderUserSegmentsChart()}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardHeader title="Segment Details" />
                    <CardContent>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Segment</TableCell>
                              <TableCell align="right">Users</TableCell>
                              <TableCell align="right">Revenue</TableCell>
                              <TableCell align="right">Churn %</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {userSegments.map((segment, index) => (
                              <TableRow key={index}>
                                <TableCell>{segment.segment}</TableCell>
                                <TableCell align="right">{segment.count.toLocaleString()}</TableCell>
                                <TableCell align="right">${segment.revenue.toLocaleString()}</TableCell>
                                <TableCell align="right">{segment.churnRate.toFixed(1)}%</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}
          {selectedTab === 4 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Operational Metrics
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Card>
                    <CardHeader title="System Performance" />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={3}>
                          <Box textAlign="center">
                            <CircularProgress 
                              variant="determinate" 
                              value={95} 
                              size={60}
                              color="success"
                            />
                            <Typography variant="h6">95%</Typography>
                            <Typography variant="caption">Uptime</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} sm={3}>
                          <Box textAlign="center">
                            <CircularProgress 
                              variant="determinate" 
                              value={87} 
                              size={60}
                              color="info"
                            />
                            <Typography variant="h6">87%</Typography>
                            <Typography variant="caption">Quality Score</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} sm={3}>
                          <Box textAlign="center">
                            <CircularProgress 
                              variant="determinate" 
                              value={92} 
                              size={60}
                              color="primary"
                            />
                            <Typography variant="h6">92%</Typography>
                            <Typography variant="caption">User Satisfaction</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12} sm={3}>
                          <Box textAlign="center">
                            <CircularProgress 
                              variant="determinate" 
                              value={78} 
                              size={60}
                              color="warning"
                            />
                            <Typography variant="h6">78%</Typography>
                            <Typography variant="caption">Cost Efficiency</Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}
          {selectedTab === 5 && renderStrategicInsights()}
        </Box>
      </Paper>

      {/* Competitive Analysis */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Competitive Analysis
        </Typography>
        {renderCompetitiveAnalysis()}
      </Paper>
    </Box>
  </>
  )};`