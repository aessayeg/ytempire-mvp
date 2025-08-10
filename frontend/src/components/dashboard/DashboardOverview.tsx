/**
 * Dashboard Overview Component
 * Owner: Frontend Team Lead
 * 
 * Main dashboard component demonstrating Zustand state management
 */

import React, { useEffect } from 'react'
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Button,
  LinearProgress,
  Alert
} from '@mui/material'
import {
  TrendingUp,
  VideoLibrary,
  Subscriptions,
  AttachMoney,
  Refresh
} from '@mui/icons-material'
import { useDashboard, useUI } from '@/hooks/useStores'

const MetricCard: React.FC<{
  title: string
  value: string | number
  change?: number
  icon: React.ReactNode
  color: string
}> = ({ title, value, change, icon, color }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography color="textSecondary" gutterBottom variant="body2">
            {title}
          </Typography>
          <Typography variant="h4" component="h2">
            {value}
          </Typography>
          {change !== undefined && (
            <Typography
              variant="body2"
              color={change >= 0 ? 'success.main' : 'error.main'}
            >
              {change >= 0 ? '+' : ''}{change}%
            </Typography>
          )}
        </Box>
        <Box sx={{ color }}>
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
)

export const DashboardOverview: React.FC = () => {
  const {
    user,
    dashboard,
    metrics,
    channels,
    videos,
    queue,
    isLoading,
    actions
  } = useDashboard()
  
  const { actions: uiActions } = useUI()
  
  useEffect(() => {
    // Load dashboard data on mount
    actions.refreshAll()
  }, [])
  
  const handleRefresh = async () => {
    try {
      await actions.refreshAll()
      uiActions.showSuccess('Dashboard data refreshed successfully')
    } catch (error) {
      uiActions.showError('Failed to refresh dashboard data')
    }
  }
  
  if (isLoading && !dashboard) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Loading Dashboard...
        </Typography>
        <LinearProgress />
      </Box>
    )
  }
  
  if (!dashboard) {
    return (
      <Alert severity="error">
        Failed to load dashboard data. Please try refreshing the page.
      </Alert>
    )
  }
  
  const activeQueue = queue.filter(item => item.status === 'processing')
  
  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Welcome back, {user?.fullName || user?.username}!
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Here's what's happening with your YouTube empire
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={handleRefresh}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>
      
      {/* Key Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Channels"
            value={dashboard.overview.total_channels}
            icon={<VideoLibrary fontSize="large" />}
            color="primary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Videos"
            value={dashboard.overview.total_videos}
            icon={<VideoLibrary fontSize="large" />}
            color="secondary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Subscribers"
            value={dashboard.overview.total_subscribers.toLocaleString()}
            icon={<Subscriptions fontSize="large" />}
            color="success.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Monthly Revenue"
            value={`$${dashboard.overview.monthly_revenue.toFixed(2)}`}
            icon={<AttachMoney fontSize="large" />}
            color="warning.main"
          />
        </Grid>
      </Grid>
      
      {/* Calculated Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total ROI"
            value={`${metrics.totalROI}%`}
            change={metrics.totalROI > 100 ? metrics.totalROI - 100 : undefined}
            icon={<TrendingUp fontSize="large" />}
            color="success.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Cost per Video"
            value={`$${metrics.costPerVideo}`}
            icon={<AttachMoney fontSize="large" />}
            color="info.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg. Engagement"
            value={`${metrics.averageEngagement}%`}
            icon={<TrendingUp fontSize="large" />}
            color="secondary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Revenue per Subscriber"
            value={`$${metrics.revenuePerSubscriber}`}
            icon={<AttachMoney fontSize="large" />}
            color="primary.main"
          />
        </Grid>
      </Grid>
      
      {/* Active Generation Queue */}
      {activeQueue.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Currently Processing ({activeQueue.length} videos)
            </Typography>
            {activeQueue.map(item => (
              <Box key={item.id} mb={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">
                    Video ID: {item.video_id}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {item.progress}% - {item.current_stage}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={item.progress}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            ))}
          </CardContent>
        </Card>
      )}
      
      {/* Quick Stats */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <Box>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Active Channels: {channels.filter(c => c.status === 'active').length}
                </Typography>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Processing Videos: {videos.filter(v => v.status === 'processing').length}
                </Typography>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Completed Today: {videos.filter(v => 
                    v.status === 'completed' && 
                    new Date(v.completed_at || '').toDateString() === new Date().toDateString()
                  ).length}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Top Performing Channels
              </Typography>
              {dashboard.performance.top_performing_channels.slice(0, 3).map((channel, index) => (
                <Box
                  key={channel.id}
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  mb={1}
                >
                  <Typography variant="body2">
                    #{index + 1} {channel.name}
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    ${channel.revenue.toFixed(2)}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DashboardOverview