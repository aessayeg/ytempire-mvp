import React from 'react';
import { 
  Paper,
  Box,
  Typography,
  LinearProgress,
  Chip,
  Grid
 } from '@mui/material';
import {  PieChart, Pie, Cell, ResponsiveContainer, Tooltip  } from 'recharts';

interface VideoEngagementStatsProps {
  stats: {,
  likeRatio: number,

    commentRate: number,
  shareRate: number,

    avgViewDuration: number,
  clickThroughRate: number,

    audienceRetention: number[]};
}

export const VideoEngagementStats: React.FC<VideoEngagementStatsProps> = ({ stats }) => {
  const engagementData = [
    { name: 'Likes', value: stats.likeRatio || 85, color: '#4 CAF50' },
    { name: 'Dislikes', value: 100 - (stats.likeRatio || 85), color: '#F44336' }
  ];

  const StatItem = ({ label, value, max = 100, color = 'primary' }: React.ChangeEvent<HTMLInputElement>) => (
    <Box mb={2}>
      <Box display="flex" justifyContent="space-between" mb={1}>
        <Typography variant="body2">{label}</Typography>
        <Typography variant="body2" fontWeight="bold">{value}%</Typography>
      </Box>
      <LinearProgress variant="determinate" value={(value / max) * 100} color={color} />
    </Box>
  );

  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Engagement Statistics</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <StatItem label="Like/Dislike Ratio" value={stats.likeRatio || 85} />
          <StatItem label="Comment Rate" value={stats.commentRate || 12} max={50} />
          <StatItem label="Share Rate" value={stats.shareRate || 8} max={20} />
          <StatItem label="Avg View Duration" value={stats.avgViewDuration || 65} />
          <StatItem label="Click-through Rate" value={stats.clickThroughRate || 4.5} max={10} />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Like/Dislike Distribution</Typography>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={engagementData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                {engagementData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          
          <Box display="flex" justifyContent="center" gap={2} mt={2}>
            {engagementData.map((entry) => (
              <Chip key={entry.name} label={`${entry.name}: ${entry.value}%`} size="small" style={{ backgroundColor: entry.color, color: 'white' }} />
            ))}
          </Box>
        </Grid>
      </Grid>
    </Paper>
  </>
  )};`