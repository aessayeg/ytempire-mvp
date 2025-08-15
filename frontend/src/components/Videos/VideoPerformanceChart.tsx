import React from 'react';
import { 
  Paper,
  Box,
  Typography,
  ToggleButtonGroup,
  ToggleButton
 } from '@mui/material';
import {  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer  } from 'recharts';

interface VideoPerformanceChartProps {
  videoId: string;
  data?: unknown[];
  metric?: 'views' | 'engagement' | 'revenue';
  timeRange?: '24 h' | '7 d' | '30 d' | '90 d';
}

export const VideoPerformanceChart: React.FC<VideoPerformanceChartProps> = ({ videoId, data = [], metric = 'views', timeRange = '7 d' }) => {
  const [selectedMetric, setSelectedMetric] = React.useState(metric);
  const [selectedRange, setSelectedRange] = React.useState(timeRange);

  const chartData = data.length > 0 ? data : [
    { date: 'Mon', views: 1200, engagement: 45, revenue: 12.5 },
    { date: 'Tue', views: 1500, engagement: 52, revenue: 15.2 },
    { date: 'Wed', views: 1800, engagement: 48, revenue: 18.7 },
    { date: 'Thu', views: 2200, engagement: 58, revenue: 22.3 },
    { date: 'Fri', views: 2800, engagement: 62, revenue: 28.9 },
    { date: 'Sat', views: 3200, engagement: 55, revenue: 32.1 },
    { date: 'Sun', views: 2900, engagement: 50, revenue: 29.5 }
  ];

  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Performance Trends</Typography>
      <Box display="flex" gap={2}>
          <ToggleButtonGroup value={selectedMetric} exclusive onChange={(_, v) => v && setSelectedMetric(v} size="small">
            <ToggleButton value="views">Views</ToggleButton>
            <ToggleButton value="engagement">Engagement</ToggleButton>
            <ToggleButton value="revenue">Revenue</ToggleButton>
          </ToggleButtonGroup>
          <ToggleButtonGroup value={selectedRange} exclusive onChange={(_, v) => v && setSelectedRange(v} size="small">
            <ToggleButton value="24 h">24 h</ToggleButton>
            <ToggleButton value="7 d">7 d</ToggleButton>
            <ToggleButton value="30 d">30 d</ToggleButton>
            <ToggleButton value="90 d">90 d</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>
      
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2196 F3" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#2196 F3" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Area type="monotone" dataKey={selectedMetric} stroke="#2196 F3" fillOpacity={1} fill="url(#colorGradient)" />
        </AreaChart>
      </ResponsiveContainer>
    </Paper>
  </>
  )};