/**
 * Main Dashboard Page Component
 * MVP Screen Design - Dashboard mockup
 */
import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  VideoLibrary,
  AttachMoney,
  People,
  Schedule,
  Refresh,
  NotificationsActive,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
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

import { useAuthStore } from '../../stores/authStore';
import { useVideoStore } from '../../stores/videoStore';
import { DashboardHeader } from '../../components/Dashboard/DashboardHeader';
import { MetricCard } from '../../components/Dashboard/MetricCard';
import { VideoQueue } from '../../components/Dashboard/VideoQueue';
import { RecentActivity } from '../../components/Dashboard/RecentActivity';
import { CostBreakdown } from '../../components/Dashboard/CostBreakdown';

// Mock data for charts
const mockViewsData = [
  { date: 'Mon', views: 12000, engagement: 850 },
  { date: 'Tue', views: 15000, engagement: 1200 },
  { date: 'Wed', views: 13500, engagement: 980 },
  { date: 'Thu', views: 18000, engagement: 1500 },
  { date: 'Fri', views: 22000, engagement: 1800 },
  { date: 'Sat', views: 19000, engagement: 1600 },
  { date: 'Sun', views: 21000, engagement: 1900 },
];

const mockCostData = [
  { name: 'Script Generation', value: 35, color: '#8884d8' },
  { name: 'Voice Synthesis', value: 25, color: '#82ca9d' },
  { name: 'Thumbnail Creation', value: 15, color: '#ffc658' },
  { name: 'Video Processing', value: 20, color: '#ff7c7c' },
  { name: 'Infrastructure', value: 5, color: '#8dd1e1' },
];

const mockRevenueData = [
  { month: 'Jan', revenue: 4500, costs: 1200 },
  { month: 'Feb', revenue: 5200, costs: 1400 },
  { month: 'Mar', revenue: 6800, costs: 1600 },
  { month: 'Apr', revenue: 7500, costs: 1800 },
  { month: 'May', revenue: 8900, costs: 2000 },
  { month: 'Jun', revenue: 9800, costs: 2200 },
];

interface DashboardMetrics {
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
}

export const Dashboard: React.FC = () => {
  const theme = useTheme();
  const { user } = useAuthStore();
  const { videos, fetchVideos } = useVideoStore();
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalVideos: 156,
    totalViews: 2450000,
    totalRevenue: 9800,
    totalCosts: 2200,
    activeChannels: 5,
    queuedVideos: 8,
    processingVideos: 3,
    completedToday: 12,
    averageCostPerVideo: 2.85,
    engagementRate: 8.5,
  });
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    // Fetch initial data
    handleRefresh();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  const handleRefresh = async () => {
    setLoading(true);
    try {
      // Fetch latest metrics
      await fetchVideos();
      // Update metrics from API
      setLastUpdated(new Date());
    } catch (_error) {
      console.error('Failed to refresh dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
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
      {/* Header */}
      <DashboardHeader
        title="Dashboard"
        subtitle={`Welcome back, ${user?.name || 'User'}`}
        lastUpdated={lastUpdated}
        onRefresh={handleRefresh}
        loading={loading}
      />

      {/* Alert for cost threshold */}
      {metrics.totalCosts > 2000 && (
        <Alert
          severity="warning"
          icon={<Warning />}
          sx={{ mb: 3 }}
          action={
            <Chip label="View Details" size="small" clickable color="warning" />
          }
        >
          Monthly cost threshold approaching. Current: {formatCurrency(metrics.totalCosts)} / {formatCurrency(2500)}
        </Alert>
      )}

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Videos"
            value={metrics.totalVideos}
            icon={<VideoLibrary />}
            trend="+12%"
            trendDirection="up"
            color="#8884d8"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Views"
            value={formatNumber(metrics.totalViews)}
            icon={<TrendingUp />}
            trend="+25%"
            trendDirection="up"
            color="#82ca9d"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Revenue"
            value={formatCurrency(metrics.totalRevenue)}
            icon={<AttachMoney />}
            trend="+18%"
            trendDirection="up"
            color="#ffc658"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Cost/Video"
            value={`$${metrics.averageCostPerVideo}`}
            icon={<AttachMoney />}
            trend="-5%"
            trendDirection="down"
            color="#ff7c7c"
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Views & Engagement Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Views & Engagement Trend
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <AreaChart data={mockViewsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="views"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="engagement"
                  stackId="2"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Cost Breakdown Pie Chart */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Cost Breakdown
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <PieChart>
                <Pie
                  data={mockCostData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {mockCostData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Video Queue and Activity */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Video Queue */}
        <Grid item xs={12} md={6}>
          <VideoQueue
            queuedCount={metrics.queuedVideos}
            processingCount={metrics.processingVideos}
            completedCount={metrics.completedToday}
          />
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <RecentActivity />
        </Grid>
      </Grid>

      {/* Revenue vs Costs Chart */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Revenue vs Costs Trend
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={mockRevenueData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="revenue" fill="#82ca9d" />
                <Bar dataKey="costs" fill="#ff7c7c" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Quick Stats Footer */}
      <Grid container spacing={2} sx={{ mt: 3 }}>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <People color="primary" />
              <Typography variant="h6">{metrics.activeChannels}</Typography>
              <Typography variant="body2" color="text.secondary">
                Active Channels
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Schedule color="warning" />
              <Typography variant="h6">{metrics.queuedVideos}</Typography>
              <Typography variant="body2" color="text.secondary">
                In Queue
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <CheckCircle color="success" />
              <Typography variant="h6">{metrics.completedToday}</Typography>
              <Typography variant="body2" color="text.secondary">
                Completed Today
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <TrendingUp color="info" />
              <Typography variant="h6">{metrics.engagementRate}%</Typography>
              <Typography variant="body2" color="text.secondary">
                Engagement Rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};