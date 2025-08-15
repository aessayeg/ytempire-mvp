import React, { useEffect, useState } from 'react';
import { 
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Chip
 } from '@mui/material';
import {  TrendingUp, Visibility, ThumbUp, Comment, WatchLater, AttachMoney  } from '@mui/icons-material';
import {  api  } from '../../services/api';

interface VideoMetricsProps {
  videoId: string}

export const VideoMetrics: React.FC<VideoMetricsProps> = ({ videoId }) => {
  const [metrics, setMetrics] = useState<any>(null);
  
  useEffect(() => {
    fetchMetrics()}, [videoId]); // eslint-disable-line react-hooks/exhaustive-deps
  
  const fetchMetrics = async () => {
    try {
      const response = await api.videos.getAnalytics(videoId);
      setMetrics(response)} catch (_) {
      console.error('Failed to fetch, metrics:', error)}
  };

  if (!metrics) return null;

  const MetricCard = ({ icon, label, value, change }: React.ChangeEvent<HTMLInputElement>) => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" gap={1} mb={1}>
          {icon}
          <Typography variant="subtitle2" color="text.secondary">{label}</Typography>
        </Box>
        <Typography variant="h4">{value}</Typography>
        {change && (
          <Chip label={`${change > 0 ? '+' : ''}${change}%`} size="small" color={change > 0 ? 'success' : 'error'} />
        )}
      </CardContent>
    </Card>
  );

  return (
    <>
      <Grid container spacing={2}>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<Visibility />} label="Views" value={metrics.views?.toLocaleString()} change={metrics.viewsChange} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<ThumbUp />} label="Likes" value={metrics.likes?.toLocaleString()} change={metrics.likesChange} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<Comment />} label="Comments" value={metrics.comments?.toLocaleString()} />
      </Grid>
      <Grid item xs={6} md={3}>`
        <MetricCard icon={<WatchLater />} label="Watch Time" value={`${metrics.watchTime}h`} />
      </Grid>
      <Grid item xs={6} md={3}>`
        <MetricCard icon={<AttachMoney />} label="Revenue" value={`$${metrics.revenue?.toFixed(2)}`} change={metrics.revenueChange} />
      </Grid>
      <Grid item xs={6} md={3}>`
        <MetricCard icon={<TrendingUp />} label="Engagement" value={`${metrics.engagementRate}%`} />
      </Grid>
    </Grid>
  </>
  )};`