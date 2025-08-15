/**
 * Analytics Screen Component
 * Comprehensive analytics dashboard for YouTube channel performance
 */
import React, { useState, useEffect } from 'react';
import { 
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  IconButton,
  Tooltip,
  useTheme,
  ToggleButton,
  ToggleButtonGroup
 } from '@mui/material';
import { 
  TrendingUp,
  TrendingDown,
  Visibility,
  ThumbUp,
  AttachMoney,
  PlayArrow,
  Download,
  Refresh,
  YouTube,
  Analytics as AnalyticsIcon,
  CompareArrows,
  StarRate
 } from '@mui/icons-material';
import { 
  Line,
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
  ComposedChart
 } from 'recharts';
import {  format  } from 'date-fns';

// Mock data for analytics
const mockViewsData = [ { date: '2024-01-01', views: 12000, subscribers: 850, revenue: 120, engagement: 8.2 },
  { date: '2024-01-02', views: 15000, subscribers: 1200, revenue: 180, engagement: 9.1 },
  { date: '2024-01-03', views: 13500, subscribers: 980, revenue: 145, engagement: 7.8 },
  { date: '2024-01-04', views: 18000, subscribers: 1500, revenue: 220, engagement: 8.9 },
  { date: '2024-01-05', views: 22000, subscribers: 1800, revenue: 280, engagement: 9.5 },
  { date: '2024-01-06', views: 19000, subscribers: 1600, revenue: 245, engagement: 8.7 },
  { date: '2024-01-07', views: 21000, subscribers: 1900, revenue: 265, engagement: 9.2 },
  { date: '2024-01-08', views: 24000, subscribers: 2100, revenue: 320, engagement: 9.8 },
  { date: '2024-01-09', views: 26500, subscribers: 2300, revenue: 355, engagement: 10.1 },
  { date: '2024-01-10', views: 23000, subscribers: 2000, revenue: 295, engagement: 8.9 },
  { date: '2024-01-11', views: 28000, subscribers: 2500, revenue: 385, engagement: 10.4 },
  { date: '2024-01-12', views: 31000, subscribers: 2800, revenue: 420, engagement: 11.2 },
  { date: '2024-01-13', views: 29500, subscribers: 2650, revenue: 395, engagement: 10.8 },
  { date: '2024-01-14', views: 33000, subscribers: 3000, revenue: 450, engagement: 11.5 } ];

const mockTopVideos = [ { id: 1,
    title: "10 AI Tools Every YouTuber Needs in 2024",
    thumbnail: "/api/placeholder/120/68",
    views: 145000,
    likes: 8900,
    comments: 1200,
    duration: "12:45",
    uploadDate: "2024-01-10",
    revenue: 1250,
    ctr: 12.5,
    avgViewDuration: 8.2,
    engagement: 6.8 },
  { id: 2,
    title: "How I Automated My Entire YouTube Channel",
    thumbnail: "/api/placeholder/120/68",
    views: 98000,
    likes: 5600,
    comments: 890,
    duration: "15:30",
    uploadDate: "2024-01-08",
    revenue: 890,
    ctr: 9.8,
    avgViewDuration: 10.2,
    engagement: 7.2 },
  { id: 3,
    title: "AI vs Human: Content Creation Showdown",
    thumbnail: "/api/placeholder/120/68",
    views: 87000,
    likes: 4800,
    comments: 650,
    duration: "18:20",
    uploadDate: "2024-01-05",
    revenue: 750,
    ctr: 11.2,
    avgViewDuration: 12.1,
    engagement: 6.1 },
  { id: 4,
    title: "Building a Million Dollar YouTube Empire",
    thumbnail: "/api/placeholder/120/68",
    views: 156000,
    likes: 12000,
    comments: 2100,
    duration: "22:15",
    uploadDate: "2024-01-03",
    revenue: 1680,
    ctr: 8.9,
    avgViewDuration: 15.8,
    engagement: 8.9 },
  { id: 5,
    title: "YouTube Algorithm Secrets Revealed",
    thumbnail: "/api/placeholder/120/68",
    views: 203000,
    likes: 15600,
    comments: 2800,
    duration: "16:40",
    uploadDate: "2024-01-01",
    revenue: 2180,
    ctr: 14.2,
    avgViewDuration: 11.5,
    engagement: 9.4 } ];

const mockChannelData = [ { name: 'Tech Reviews', subscribers: 125000, views: 2400000, revenue: 8500, growth: 12.5, color: '#8884d8' },
  { name: 'Gaming Central', subscribers: 89000, views: 1800000, revenue: 6200, growth: 8.3, color: '#82ca9d' },
  { name: 'Lifestyle Vlog', subscribers: 67000, views: 980000, revenue: 3400, growth: -2.1, color: '#ffc658' },
  { name: 'Educational Hub', subscribers: 156000, views: 3200000, revenue: 12000, growth: 18.9, color: '#ff7c7c' },
  { name: 'Music Covers', subscribers: 34000, views: 450000, revenue: 1200, growth: 5.7, color: '#8 dd1 e1' } ];

const mockAudienceData = [ { ageGroup: '13-17', percentage: 15, male: 8, female: 7 },
  { ageGroup: '18-24', percentage: 35, male: 20, female: 15 },
  { ageGroup: '25-34', percentage: 28, male: 16, female: 12 },
  { ageGroup: '35-44', percentage: 15, male: 9, female: 6 },
  { ageGroup: '45-54', percentage: 5, male: 3, female: 2 },
  { ageGroup: '55+', percentage: 2, male: 1, female: 1 } ];

interface AnalyticsMetrics {
  totalViews: number,
  totalSubscribers: number,

  totalRevenue: number,
  avgEngagement: number,

  totalVideos: number,
  avgCTR: number,

  avgViewDuration: number,
  totalWatchTime: number}

export const Analytics: React.FC = () => { const theme = useTheme();
  const [selectedChannel, setSelectedChannel] = useState<string>('all');
  const [dateRange, setDateRange] = useState<string>('30 d');
  const [viewType, setViewType] = useState<string>('views');
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<AnalyticsMetrics>({
    totalViews: 8850000,
    totalSubscribers: 471000,
    totalRevenue: 31300,
    avgEngagement: 8.7,
    totalVideos: 156,
    avgCTR: 11.3,
    avgViewDuration: 11.6,
    totalWatchTime: 1250000 });

  const handleRefresh = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false)}, 2000)};

  const formatCurrency = (_value: number) => { return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0 }).format(value)};

  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000.toFixed(1)}M`;
    } else if (value >= 1000) {`
      return `${(value / 1000.toFixed(1)}K`;
    }
    return value.toString()};

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);`
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'views': return <Visibility />;
      case 'subscribers': return <YouTube />;
      case 'revenue': return <AttachMoney />;
      case 'engagement': return <ThumbUp />;
      default: return <AnalyticsIcon />}
  };

  const getChangeColor = (change: number) => {
    return change > 0 ? 'success.main' : change < 0 ? 'error.main' : 'text.secondary';
  };

  const getChangeIcon = (change: number) => {
    return change > 0 ? <TrendingUp /> : change < 0 ? <TrendingDown /> : null;
  };

  return (
    <>
      <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Analytics Dashboard
          </Typography>
      <Typography variant="body2" color="text.secondary">
            Track performance across all your YouTube channels
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Channel</InputLabel>
            <Select
              value={selectedChannel}
              onChange={(e) => setSelectedChannel(e.target.value as string)}
              label="Channel"
            >
              <MenuItem value="all">All Channels</MenuItem>
              {mockChannelData.map((channel) => (
                <MenuItem key={channel.name} value={channel.name}>
                  {channel.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <ToggleButtonGroup
            value={dateRange}
            exclusive
            onChange={(e, value) => value && setDateRange(value)}
            size="small"
          >
            <ToggleButton value="7 d">7 D</ToggleButton>
            <ToggleButton value="30 d">30 D</ToggleButton>
            <ToggleButton value="90 d">90 D</ToggleButton>
            <ToggleButton value="1 y">1 Y</ToggleButton>
          </ToggleButtonGroup>

          <IconButton onClick={handleRefresh} disabled={loading}>
            <Refresh />
          </IconButton>

          <Button variant="outlined" startIcon={<Download />}>
            Export
          </Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {[ { 
            title: 'Total Views', 
            value: formatNumber(metrics.totalViews), 
            change: +15.2, 
            metric: 'views',

          },
          { 
            title: 'Subscribers', 
            value: formatNumber(metrics.totalSubscribers), 
            change: +8.7, 
            metric: 'subscribers',

          },
          { 
            title: 'Revenue', 
            value: formatCurrency(metrics.totalRevenue), 
            change: +12.3, 
            metric: 'revenue',

          },
          { 
            title: 'Avg Engagement', `
            value: `${metrics.avgEngagement}%`, 
            change: +2.1, 
            metric: 'engagement',

          } ].map((item, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {item.title}
                  </Typography>
                  {getMetricIcon(item.metric)}
                </Box>
                <Typography variant="h4" fontWeight="bold" gutterBottom>
                  {item.value}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {getChangeIcon(item.change)}
                  <Typography 
                    variant="body2" 
                    color={getChangeColor(item.change)}
                    sx={{ ml: 0.5 }}
                  >
                    {item.change > 0 ? '+' : ''}{item.change}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                    vs last period
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Main Analytics Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Performance Trends */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Performance Trends</Typography>
              <ToggleButtonGroup
                value={viewType}
                exclusive
                onChange={(e, value) => value && setViewType(value)}
                size="small"
              >
                <ToggleButton value="views">Views</ToggleButton>
                <ToggleButton value="subscribers">Subscribers</ToggleButton>
                <ToggleButton value="revenue">Revenue</ToggleButton>
                <ToggleButton value="engagement">Engagement</ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <ResponsiveContainer width="100%" height="90%">
              <ComposedChart data={mockViewsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={(value) => format(new Date(value), 'MMM dd')} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <RechartsTooltip 
                  labelFormatter={(value) => format(new Date(value), 'PPP')}
                  formatter={(value: React.ChangeEvent<HTMLInputElement>, name: string) => [ name === 'revenue' ? formatCurrency(value) : formatNumber(value),
                    name.charAt(0).toUpperCase() + name.slice(1)
                   ]
                />
                <Legend />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey={viewType}
                  fill={theme.palette.primary.main}
                  fillOpacity={0.3}
                  stroke={theme.palette.primary.main}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="engagement"
                  stroke={theme.palette.secondary.main}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Channel Performance */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Channel Performance
            </Typography>
            <Box sx={{ height: '90%', overflowY: 'auto' }}>
              {mockChannelData.map((channel, index) => (
                <Box key={index} sx={{ mb: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {channel.name}
                    </Typography>
                    <Chip
                      label={`${channel.growth > 0 ? '+' : ''}${channel.growth}%`}
                      color={channel.growth > 0 ? 'success' : channel.growth < 0 ? 'error' : 'default'}
                      size="small"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {formatNumber(channel.subscribers)} subscribers • {formatNumber(channel.views)} views
                  </Typography>
                  <Typography variant="body2" color="primary" fontWeight="bold">
                    {formatCurrency(channel.revenue)} revenue
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min((channel.revenue / 15000) * 100, 100)}
                    sx={{ mt: 1, height: 6, borderRadius: 3 }}
                    color={channel.growth > 0 ? 'success' : 'error'}
                  />
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Top Performing Videos */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">Top Performing Videos</Typography>
              <Button variant="outlined" size="small" startIcon={<CompareArrows />}>
                Compare
              </Button>
            </Box>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Video</TableCell>
                    <TableCell align="right">Views</TableCell>
                    <TableCell align="right">Engagement</TableCell>
                    <TableCell align="right">CTR</TableCell>
                    <TableCell align="right">Avg Duration</TableCell>
                    <TableCell align="right">Revenue</TableCell>
                    <TableCell align="right">Upload Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {mockTopVideos.map((video) => (
                    <TableRow key={video.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box
                            sx={ {
                              width: 80,
                              height: 45,
                              bgcolor: 'grey.200',
                              borderRadius: 1,
                              mr: 2,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              position: 'relative' }}
                          >
                            <PlayArrow />
                            <Typography
                              variant="caption"
                              sx={ {
                                position: 'absolute',
                                bottom: 2,
                                right: 2,
                                bgcolor: 'rgba(0,0,0,0.8)',
                                color: 'white',
                                px: 0.5,
                                borderRadius: 0.5,
                                fontSize: '0.7 rem' }}
                            >
                              {video.duration}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="body2" fontWeight="bold" noWrap sx={{ maxWidth: 300 }}>
                              {video.title}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                              <Typography variant="caption" color="text.secondary">
                                {formatNumber(video.likes)} likes
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatNumber(video.comments)} comments
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="bold">
                          {formatNumber(video.views)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                          <Typography variant="body2">
                            {video.engagement}%
                          </Typography>
                          {video.engagement > 7 && <StarRate color="primary" fontSize="small" sx={{ ml: 0.5 }} />}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {video.ctr}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {formatDuration(video.avgViewDuration * 60)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="success.main" fontWeight="bold">
                          {formatCurrency(video.revenue)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="text.secondary">
                          {format(new Date(video.uploadDate), 'MMM dd')}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Audience Demographics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Audience Demographics
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={mockAudienceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ageGroup" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="male" stackId="gender" fill="#8884d8" name="Male" />
                <Bar dataKey="female" stackId="gender" fill="#82 ca9 d" name="Female" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Revenue Sources
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <PieChart>
                <Pie
                  data={[ { name: 'Ad Revenue', value: 65, color: '#8884d8' },
                    { name: 'Sponsorships', value: 25, color: '#82ca9d' },
                    { name: 'Merchandise', value: 7, color: '#ffc658' },
                    { name: 'Channel Memberships', value: 3, color: '#ff7c7c' }  ]
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100.toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {[ { name: 'Ad Revenue', value: 65, color: '#8884d8' },
                    { name: 'Sponsorships', value: 25, color: '#82ca9d' },
                    { name: 'Merchandise', value: 7, color: '#ffc658' },
                    { name: 'Channel Memberships', value: 3, color: '#ff7c7c' } ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  </>
  )};

export default Analytics;`