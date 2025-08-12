import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip,
  LinearProgress,
  Alert,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  People,
  TouchApp,
  Assessment,
  FilterList,
  Info,
  Warning,
  CheckCircle,
  Cancel,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Funnel,
  FunnelChart,
  Sankey,
  HeatMapGrid,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { format, subDays } from 'date-fns';
import { useBehaviorAnalytics } from '../../hooks/useBehaviorAnalytics';
import { formatNumber, formatPercentage, formatDuration } from '../../utils/formatters';

interface UserBehaviorDashboardProps {
  userId?: number;
  dateRange?: {
    start: Date;
    end: Date;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export const UserBehaviorDashboard: React.FC<UserBehaviorDashboardProps> = ({
  userId,
  dateRange,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedFunnel, setSelectedFunnel] = useState('signup');
  const [cohortType, setCohortType] = useState('signup');
  const [segmentView, setSegmentView] = useState('overview');

  const {
    overview,
    funnelData,
    cohortData,
    heatmapData,
    segments,
    loading,
    error,
    refetch,
  } = useBehaviorAnalytics({
    userId,
    dateRange,
    funnelSteps: getFunnelSteps(selectedFunnel),
    cohortType,
  });

  function getFunnelSteps(funnelType: string): string[] {
    switch (funnelType) {
      case 'signup':
        return ['page_view', 'signup_start', 'signup_complete', 'first_video'];
      case 'video':
        return ['dashboard_view', 'video_create_click', 'video_generate', 'video_publish'];
      case 'upgrade':
        return ['pricing_view', 'plan_select', 'checkout', 'payment_complete'];
      default:
        return [];
    }
  }

  const renderMetricCard = (title: string, value: number | string, icon: React.ReactNode, color: string = 'primary') => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography color="textSecondary" variant="body2">
            {title}
          </Typography>
          <Box color={`${color}.main`}>{icon}</Box>
        </Box>
        <Typography variant="h4" fontWeight="bold">
          {typeof value === 'number' ? formatNumber(value) : value}
        </Typography>
      </CardContent>
    </Card>
  );

  const renderEventBreakdown = () => {
    if (!overview?.event_breakdown) return null;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={overview.event_breakdown}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="event_type" angle={-45} textAnchor="end" height={80} />
          <YAxis />
          <RechartsTooltip />
          <Bar dataKey="count" fill="#8884d8">
            {overview.event_breakdown.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderFunnelChart = () => {
    if (!funnelData?.steps) return null;

    const funnelChartData = funnelData.steps.map(step => ({
      name: step.step,
      value: step.users,
      rate: step.conversion_rate,
    }));

    return (
      <Box>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Funnel Type</InputLabel>
            <Select
              value={selectedFunnel}
              onChange={(e) => setSelectedFunnel(e.target.value)}
              label="Funnel Type"
            >
              <MenuItem value="signup">Signup Flow</MenuItem>
              <MenuItem value="video">Video Creation</MenuItem>
              <MenuItem value="upgrade">Upgrade Flow</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="body2" color="textSecondary">
            Overall Conversion: {formatPercentage(funnelData.overall_conversion)}
          </Typography>
        </Box>

        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={funnelChartData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={120} />
            <RechartsTooltip
              formatter={(value: number) => formatNumber(value)}
              labelFormatter={(label) => `${label}: ${formatPercentage(funnelChartData.find(d => d.name === label)?.rate || 0)}`}
            />
            <Bar dataKey="value" fill="#8884d8">
              {funnelChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Funnel Steps Table */}
        <TableContainer component={Paper} sx={{ mt: 2 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Step</TableCell>
                <TableCell align="right">Users</TableCell>
                <TableCell align="right">Conversion Rate</TableCell>
                <TableCell align="right">Drop-off Rate</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {funnelData.steps.map((step) => (
                <TableRow key={step.step}>
                  <TableCell>{step.step}</TableCell>
                  <TableCell align="right">{formatNumber(step.users)}</TableCell>
                  <TableCell align="right">{formatPercentage(step.conversion_rate)}</TableCell>
                  <TableCell align="right">
                    <Chip
                      label={formatPercentage(step.drop_off_rate)}
                      size="small"
                      color={step.drop_off_rate > 50 ? 'error' : 'default'}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderCohortAnalysis = () => {
    if (!cohortData?.cohorts) return null;

    return (
      <Box>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Cohort Type</InputLabel>
            <Select
              value={cohortType}
              onChange={(e) => setCohortType(e.target.value)}
              label="Cohort Type"
            >
              <MenuItem value="signup">By Signup Date</MenuItem>
              <MenuItem value="first_video">By First Video</MenuItem>
              <MenuItem value="upgrade">By Upgrade Date</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Cohort</TableCell>
                <TableCell align="right">Size</TableCell>
                <TableCell align="center">Week 0</TableCell>
                <TableCell align="center">Week 1</TableCell>
                <TableCell align="center">Week 2</TableCell>
                <TableCell align="center">Week 3</TableCell>
                <TableCell align="center">Week 4</TableCell>
                <TableCell align="center">Week 5</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {cohortData.cohorts.map((cohort) => (
                <TableRow key={cohort.cohort}>
                  <TableCell>{cohort.cohort}</TableCell>
                  <TableCell align="right">{cohort.size}</TableCell>
                  {[0, 1, 2, 3, 4, 5].map((week) => {
                    const retention = cohort.retention.find(r => r.period === week);
                    const rate = retention?.retention_rate || 0;
                    return (
                      <TableCell key={week} align="center">
                        <Chip
                          label={formatPercentage(rate)}
                          size="small"
                          sx={{
                            backgroundColor: `rgba(0, 136, 254, ${rate / 100})`,
                            color: rate > 50 ? 'white' : 'inherit',
                          }}
                        />
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderHeatmap = () => {
    if (!heatmapData?.heatmap) return null;

    // Group data by hour and day
    const heatmapMatrix: number[][] = Array(7).fill(null).map(() => Array(24).fill(0));
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

    heatmapData.heatmap.forEach((item) => {
      const date = new Date(item.date);
      const dayOfWeek = date.getDay();
      heatmapMatrix[dayOfWeek][item.hour] = item.intensity;
    });

    return (
      <Box>
        <Typography variant="h6" mb={2}>Feature Usage Heatmap</Typography>
        <Box sx={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ padding: 8, textAlign: 'left' }}>Day</th>
                {Array.from({ length: 24 }, (_, i) => (
                  <th key={i} style={{ padding: 4, textAlign: 'center', fontSize: '10px' }}>
                    {i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dayNames.map((day, dayIndex) => (
                <tr key={day}>
                  <td style={{ padding: 8 }}>{day}</td>
                  {heatmapMatrix[dayIndex].map((intensity, hour) => (
                    <td
                      key={hour}
                      style={{
                        padding: 4,
                        backgroundColor: `rgba(0, 136, 254, ${intensity})`,
                        border: '1px solid #f0f0f0',
                        width: 20,
                        height: 20,
                      }}
                      title={`${day} ${hour}:00 - Intensity: ${(intensity * 100).toFixed(0)}%`}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Box>
        <Box display="flex" alignItems="center" mt={2}>
          <Typography variant="body2" color="textSecondary" mr={2}>
            Low Activity
          </Typography>
          <Box sx={{ flexGrow: 1, height: 10, background: 'linear-gradient(to right, rgba(0,136,254,0.1), rgba(0,136,254,1))' }} />
          <Typography variant="body2" color="textSecondary" ml={2}>
            High Activity
          </Typography>
        </Box>
      </Box>
    );
  };

  const renderUserSegments = () => {
    if (!segments?.segments) return null;

    const segmentData = Object.entries(segments.segments).map(([name, data]) => ({
      name: name.replace('_', ' ').toUpperCase(),
      count: data.count,
      percentage: (data.count / segments.total_users) * 100,
    }));

    return (
      <Box>
        <Typography variant="h6" mb={2}>User Segments</Typography>
        <Grid container spacing={2}>
          {segmentData.map((segment) => (
            <Grid item xs={12} sm={6} md={3} key={segment.name}>
              <Card>
                <CardContent>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    {segment.name}
                  </Typography>
                  <Typography variant="h4">{formatNumber(segment.count)}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    {formatPercentage(segment.percentage)} of users
                  </Typography>
                  {segment.name === 'AT RISK' && (
                    <Chip label="Action Required" color="warning" size="small" sx={{ mt: 1 }} />
                  )}
                  {segment.name === 'POWER USERS' && (
                    <Chip label="VIP" color="success" size="small" sx={{ mt: 1 }} />
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  if (loading) {
    return <LinearProgress />;
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          User Behavior Analytics
        </Typography>
        <Button variant="outlined" onClick={refetch}>
          Refresh
        </Button>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard('Total Events', overview?.total_events || 0, <Timeline />, 'primary')}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard('Unique Users', overview?.unique_users || 0, <People />, 'secondary')}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard('Avg Session Duration', 
            formatDuration(overview?.session_stats?.avg_duration || 0), 
            <Assessment />, 'success'
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderMetricCard('Bounce Rate', 
            formatPercentage(overview?.session_stats?.bounce_rate || 0), 
            <TouchApp />, 
            overview?.session_stats?.bounce_rate > 50 ? 'error' : 'info'
          )}
        </Grid>
      </Grid>

      {/* Tabs */}
      <Card>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab label="Overview" />
          <Tab label="Funnels" />
          <Tab label="Cohorts" />
          <Tab label="Heatmap" />
          <Tab label="Segments" />
        </Tabs>

        <CardContent>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" mb={2}>Event Breakdown</Typography>
              {renderEventBreakdown()}
              
              {overview?.journey_stats && (
                <Box mt={4}>
                  <Typography variant="h6" mb={2}>User Journey Insights</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="body2" color="textSecondary">
                            Total Sessions
                          </Typography>
                          <Typography variant="h5">
                            {formatNumber(overview.journey_stats.total_sessions)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="body2" color="textSecondary">
                            Avg Events/Session
                          </Typography>
                          <Typography variant="h5">
                            {overview.journey_stats.avg_events_per_session?.toFixed(1)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="body2" color="textSecondary">
                            Top Pattern
                          </Typography>
                          <Typography variant="body1">
                            {overview.journey_stats.top_patterns?.[0]?.pattern}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </Box>
          )}
          
          {activeTab === 1 && renderFunnelChart()}
          {activeTab === 2 && renderCohortAnalysis()}
          {activeTab === 3 && renderHeatmap()}
          {activeTab === 4 && renderUserSegments()}
        </CardContent>
      </Card>
    </Box>
  );
};