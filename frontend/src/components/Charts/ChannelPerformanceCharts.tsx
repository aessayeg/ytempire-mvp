/**
 * Channel Performance Charts Component
 * Advanced analytics visualization for individual channel performance
 */
import React, { useState, useEffect } from 'react';
import { 
  Box,
  Card,
  CardContent,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Chip,
  Tooltip,
  IconButton,
  CircularProgress,
  Alert,
  Button
 } from '@mui/material';
import {  Share as ShareIcon  } from '@mui/icons-material';
import { 
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
  Scatter,
  ScatterChart,
  ZAxis
 } from 'recharts';

interface ChannelMetrics {
  
date: string;
views: number;

subscribers: number;
revenue: number;

watchTime: number;
engagement: number;

ctr: number; // Click-through rate;
avgViewDuration: number;

likes: number;
comments: number;

shares: number;

}

interface VideoPerformance {
  
id: string;
title: string;

views: number;
revenue: number;

engagement: number;
publishedAt: string;

category: string;

}

interface ChannelPerformanceChartsProps {
  
channelId: string;
timeRange: '7 d' | '30 d' | '90 d' | '1 y';

onTimeRangeChange: (range: '7 d' | '30 d' | '90 d' | '1 y') => void;

}
const ChannelPerformanceCharts: React.FC<ChannelPerformanceChartsProps> = ({
  channelId, timeRange, onTimeRangeChange
}) => {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<ChannelMetrics[]>([]);
  const [videos, setVideos] = useState<VideoPerformance[]>([]);
  const [selectedMetric, setSelectedMetric] = useState('views');
  const [comparisonMode, setComparisonMode] = useState(false);

  // Sample data generation
  useEffect(() => {
    setLoading(true);
    
    // Generate sample data based on time range
    const generateSampleData = () => {
      const days = timeRange === '7 d' ? 7 : timeRange === '30 d' ? 30 : timeRange === '90 d' ? 90 : 365;
const data: ChannelMetrics[] = [];
      
      for (let i = 0; i < days; i++) {
        const date = new Date();
        date.setDate(date.getDate() - (days - i));
        
        data.push({
          date: date.toISOString().split('T')[0],
          views: Math.floor(Math.random() * 50000) + 10000,
          subscribers: Math.floor(Math.random() * 1000) + 100,
          revenue: Math.floor(Math.random() * 500) + 50,
          watchTime: Math.floor(Math.random() * 10000) + 2000,
          engagement: Math.random() * 10 + 2,
          ctr: Math.random() * 5 + 1,
          avgViewDuration: Math.random() * 300 + 60,
          likes: Math.floor(Math.random() * 2000) + 100,
          comments: Math.floor(Math.random() * 500) + 20,
          shares: Math.floor(Math.random() * 200) + 10

        })}
      return data
    };
    // Generate sample video data
    const generateVideoData = (): VideoPerformance[] => {
      return [
        { id: '1', title: 'Gaming Tutorial: Advanced Strategies', views: 45000, revenue: 234, engagement: 8.5, publishedAt: '2024-01-08', category: 'Gaming' },
        { id: '2', title: 'Tech Review: Latest Smartphone', views: 32000, revenue: 189, engagement: 7.2, publishedAt: '2024-01-07', category: 'Tech' },
        { id: '3', title: 'Cooking Masterclass: Italian Cuisine', views: 28000, revenue: 156, engagement: 9.1, publishedAt: '2024-01-06', category: 'Food' },
        { id: '4', title: 'Fitness Journey: 30-Day Challenge', views: 52000, revenue: 298, engagement: 8.8, publishedAt: '2024-01-05', category: 'Fitness' },
        { id: '5', title: 'Travel Vlog: Hidden Gems in Tokyo', views: 38000, revenue: 221, engagement: 7.9, publishedAt: '2024-01-04', category: 'Travel' }
      ]
    };
    setTimeout(() => {
      setMetrics(generateSampleData());
      setVideos(generateVideoData());
      setLoading(false)}, 1000)}, [channelId, timeRange]);

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000.toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000.toFixed(1)}K`;
    return num.toString()};
  const formatCurrency = (num: number) => `$${num.toFixed(2)}`;

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}
  };
  const getMetricColor = (value: number, baseline: number) => {
    return value >= baseline ? '#10 b981' : '#ef4444'};
  const CustomTooltip = ({ active, payload, label }: React.ChangeEvent<HTMLInputElement>) => {
    if (active && payload && payload.length) {
      return (
    <>
      <Box sx={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 1,
          p: 2,
          boxShadow: 2}}>
          <Typography variant="subtitle2">{label}</Typography>
          {payload.map((entry: React.ChangeEvent<HTMLInputElement>, index: number) => (
            <Typography key={index} sx={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.dataKey === 'revenue' ? formatCurrency(entry.value) : formatNumber(entry.value)}`}
            </Typography>
          ))}
        </Box>
      )}
    return null};
  const renderOverviewMetrics = () => {
    if (loading</>
  ) return <CircularProgress />;

    const totalViews = metrics.reduce((sum, m) => sum + m.views, 0);
    const totalRevenue = metrics.reduce((sum, m) => sum + m.revenue, 0);
    const avgEngagement = metrics.reduce((sum, m) => sum + m.engagement, 0) / metrics.length;
    const avgCTR = metrics.reduce((sum, m) => sum + m.ctr, 0) / metrics.length;

    return (
    <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total Views', value: formatNumber(totalViews), color: '#3 b82 f6', trend: 12.5 },
          { label: 'Revenue', value: formatCurrency(totalRevenue), color: '#10 b981', trend: 8.3 },
          { label: 'Avg. Engagement', value: `${avgEngagement.toFixed(1)}%`, color: '#f59e0 b', trend: -2.1 },
          { label: 'Avg. CTR', value: `${avgCTR.toFixed(2)}%`, color: '#8b5cf6', trend: 5.7 }
        ].map((metric, index) => (
          <Grid item xs={6} sm={3} key={index}>
            <Card>
              <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" color="textSecondary">
                    {metric.label}
                  </Typography>
      <Chip
                    size="small"
                    icon={metric.trend >= 0 ? <TrendingUpIcon /> </>: <TrendingDownIcon />}
                    label={`${metric.trend >= 0 ? '+' : ''}${metric.trend}%`}
                    color={metric.trend >= 0 ? 'success' : 'error'}
                    sx={{ height: 20, fontSize: '0.75rem' }}
                  />
                </Box>
                <Typography variant="h6" sx={{ color: metric.color, fontWeight: 'bold' }}>
                  {metric.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    )};
  const renderMainPerformanceChart = () => {
    if (loading) {
      return (
    <>
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      </>
      );
    }

    return (
    <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Performance Trends</Typography>
      <Box sx={{ display: 'flex', gap: 1 }}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <Select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  <MenuItem value="views">Views</MenuItem>
                  <MenuItem value="revenue">Revenue</MenuItem>
                  <MenuItem value="engagement">Engagement</MenuItem>
                  <MenuItem value="subscribers">Subscribers</MenuItem>
                </Select>
              </FormControl>
              <IconButton size="small">
                <RefreshIcon />
              </IconButton>
            </Box>
          </Box>
          
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                stroke="#666"
                fontSize={12}
                tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickFormatter={selectedMetric === 'revenue' ? formatCurrency : formatNumber}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey={selectedMetric}
                fill="#3 b82 f6"
                fillOpacity={0.1}
                stroke="#3 b82 f6"
                strokeWidth={2}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </>
  )
};
  const renderEngagementRadar = () => {
    if (loading) return <CircularProgress />;

    const radarData = [
      { subject: 'Views', A: 85, fullMark: 100 },
      { subject: 'Engagement', A: 92, fullMark: 100 },
      { subject: 'CTR', A: 78, fullMark: 100 },
      { subject: 'Watch Time', A: 88, fullMark: 100 },
      { subject: 'Subscribers', A: 81, fullMark: 100 },
      { subject: 'Revenue', A: 95, fullMark: 100 }
    ];

    return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Channel Performance Radar
          </Typography>
      <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" fontSize={12} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} fontSize={10} />
              <Radar
                name="Performance"
                dataKey="A"
                stroke="#3 b82 f6"
                fill="#3 b82 f6"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </>
  )};
  const renderVideoPerformanceScatter = () => {
    if (loading) return <CircularProgress />;

    return (_<Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Video Performance Matrix
          </Typography>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart data={videos}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="views" name="Views" tickFormatter={formatNumber} fontSize={12} />
              <YAxis dataKey="revenue" name="Revenue" tickFormatter={formatCurrency} fontSize={12} />
              <ZAxis dataKey="engagement" name="Engagement" range={[50, 200]} />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value: React.ChangeEvent<HTMLInputElement>, name: string) => {
                  if (name === 'views') return [formatNumber(value), 'Views'],
                  if (name === 'revenue') return [formatCurrency(value), 'Revenue'],
                  if (name === 'engagement') return [`${value}%`, 'Engagement'];
                  return [value, name]
                }}
                labelFormatter={() => ''}
              />
              <Scatter 
                name="Videos" 
                dataKey="revenue" 
                fill="#10 b981"
                fillOpacity={0.8}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    )
};
  const renderTopVideosTable = () => {
    if (loading) return <CircularProgress />;

    return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Top Performing Videos
          </Typography>
      <Box sx={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: '12px 8px', fontWeight: 600 }}>Video</th>
                  <th style={{ textAlign: 'right', padding: '12px 8px', fontWeight: 600 }}>Views</th>
                  <th style={{ textAlign: 'right', padding: '12px 8px', fontWeight: 600 }}>Revenue</th>
                  <th style={{ textAlign: 'right', padding: '12px 8px', fontWeight: 600 }}>Engagement</th>
                  <th style={{ textAlign: 'center', padding: '12px 8px', fontWeight: 600 }}>Category</th>
                </tr>
              </thead>
              <tbody>
                {videos.slice(0, 5).map((video, index) => (
                  <tr key={video.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '12px 8px' }}>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {video.title}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {new Date(video.publishedAt).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </td>
                    <td style={{ textAlign: 'right', padding: '12px 8px' }}>
                      <Typography variant="body2">{formatNumber(video.views)}</Typography>
                    </td>
                    <td style={{ textAlign: 'right', padding: '12px 8px' }}>
                      <Typography variant="body2">{formatCurrency(video.revenue)}</Typography>
                    </td>
                    <td style={{ textAlign: 'right', padding: '12px 8px' }}>
                      <Typography 
                        variant="body2" 
                        sx={{ color: getMetricColor(video.engagement, 7) }}
                      >
                        {video.engagement}%
                      </Typography>
                    </td>
                    <td style={{ textAlign: 'center', padding: '12px 8px' }}>
                      <Chip 
                        label={video.category} 
                        size="small" 
                        variant="outlined"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </CardContent>
      </Card>
    </>
  )};
  const renderRevenueTrendChart = () => {
    if (loading) return <CircularProgress />;

    return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Revenue Trend
          </Typography>
      <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={metrics}>
              <defs>
                <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10 b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10 b981" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                fontSize={12}
                tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
              />
              <YAxis fontSize={12} tickFormatter={formatCurrency} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="revenue"
                stroke="#10 b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#revenueGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </>
  )
};
  return (
    <>
      <Box>
      {/* Header Controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          Channel Performance Analytics
        </Typography>
      <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small">
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => onTimeRangeChange(e.target.value as any)}
              sx={{ minWidth: 120 }}
            >
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
              <MenuItem value="90d">Last 90 Days</MenuItem>
              <MenuItem value="1y">Last Year</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            size="small"
          >
            Export
          </Button>
          <Button
            variant="outlined"
            startIcon={<ShareIcon />}
            size="small"
          >
            Share
          </Button>
        </Box>
      </Box>

      {/* Overview Metrics */}
      {renderOverviewMetrics()}
      {/* Main Performance Chart */}
      {renderMainPerformanceChart()}
      {/* Secondary Charts Grid */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          {renderEngagementRadar()}
        </Grid>
        <Grid item xs={12} md={6}>
          {renderVideoPerformanceScatter()}
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          {renderTopVideosTable()}
        </Grid>
        <Grid item xs={12} lg={4}>
          {renderRevenueTrendChart()}
        </Grid>
      </Grid>

      {/* Performance Insights */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Insights
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Alert severity="success" sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Top Performing Category</Typography>
                <Typography variant="body2">Fitness videos show 23% higher engagement</Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} md={4}>
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Optimal Upload Time</Typography>
                <Typography variant="body2">Tuesdays at 2 PM show peak performance</Typography>
              </Alert>
            </Grid>
            <Grid item xs={12} md={4}>
              <Alert severity="warning" sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Revenue Opportunity</Typography>
                <Typography variant="body2">Tech videos have potential for 15% more revenue</Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  </>
  )};
export default ChannelPerformanceCharts;
