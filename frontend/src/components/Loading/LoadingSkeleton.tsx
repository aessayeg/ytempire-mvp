import React from 'react';
import { 
  Skeleton,
  Box,
  Card,
  CardContent,
  Grid
 } from '@mui/material';

export type SkeletonVariant = 'text' | 'card' | 'table' | 'chart' | 'metric' | 'list' | 'form';

interface LoadingSkeletonProps {
  variant?: SkeletonVariant;
  rows?: number;
  columns?: number;
  height?: number | string;
  width?: number | string;
  animation?: 'pulse' | 'wave' | false;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ variant = 'text', rows = 3, columns = 1, height, width = '100%', animation = 'wave' }) => {
  // Text skeleton
  if (variant === 'text') {
    return (
    <>
      <Box sx={{ width }}>
        {Array.from({ length: rows }).map((_, index) => (
          <Skeleton
            key={index}
            animation={animation}
            height={height || 24}
            width={index === rows - 1 ? '60%' : '100%'}
            sx={{ mb: 1 }}
          />
        ))}
      </Box>
    )}

  // Card skeleton
  if (variant === 'card') {
    return (
    <Card sx={{ width, height: height || 'auto' }}>
        <CardContent>
          <Skeleton animation={animation} variant="text" width="40%" height={32} sx={{ mb: 2 }} />
          <Skeleton animation={animation} variant="rectangular" height={height || 200} sx={{ mb: 2 }} />
          <Skeleton animation={animation} variant="text" width="80%" />
          <Skeleton animation={animation} variant="text" width="60%" />
        </CardContent>
      </Card>
    )}

  // Table skeleton
  if (variant === 'table') {
    return (
    <Box sx={{ width }}>
        {/* Table header */}
        <Box sx={{ display: 'flex', mb: 2, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
          {Array.from({ length: columns }).map((_, index) => (
            <Skeleton
              key={`header-${index}`}
              animation={animation}
              variant="text"
              width={`${100 / columns}%`}
              height={24}
              sx={{ mx: 1 }}
            />
          ))}
        </Box>
        {/* Table rows */}
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <Box
            key={`row-${rowIndex}`}
            sx={{ display: 'flex', mb: 1, pb: 1, borderBottom: 1, borderColor: 'divider' }}
          >
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Skeleton
                key={`cell-${rowIndex}-${colIndex}`}
                animation={animation}
                variant="text"
                width={`${100 / columns}%`}
                height={20}
                sx={{ mx: 1 }}
              />
            ))}
          </Box>
        ))}
      </Box>
    )}

  // Chart skeleton
  if (variant === 'chart') {
    return (
    <Box sx={{ width, height: height || 300, position: 'relative' }}>
        <Skeleton animation={animation} variant="rectangular" width="100%" height="100%" />
        <Box
          sx={ {
            position: 'absolute',
            bottom: 20,
            left: 20,
            right: 20,
            display: 'flex',
            justifyContent: 'space-between' }}
        >
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton
              key={index}
              animation={animation}
              variant="rectangular"
              width="15%"
              height={`${(index + 1) * 20}%`}
              sx={{ backgroundColor: 'rgba(0, 0, 0, 0.11)' }}
            />
          ))}
        </Box>
      </Box>
    )}

  // Metric skeleton
  if (variant === 'metric') {
    return (
    <Card sx={{ width, height: height || 'auto' }}>
        <CardContent>
          <Skeleton animation={animation} variant="text" width="60%" height={20} sx={{ mb: 1 }} />
          <Skeleton animation={animation} variant="text" width="40%" height={48} sx={{ mb: 1 }} />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Skeleton animation={animation} variant="circular" width={20} height={20} />
            <Skeleton animation={animation} variant="text" width="30%" height={16} />
          </Box>
        </CardContent>
      </Card>
    )}

  // List skeleton
  if (variant === 'list') {
    return (
    <Box sx={{ width }}>
        {Array.from({ length: rows }).map((_, index) => (
          <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Skeleton animation={animation} variant="circular" width={40} height={40} sx={{ mr: 2 }} />
            <Box sx={{ flex: 1 }}>
              <Skeleton animation={animation} variant="text" width="70%" height={20} />
              <Skeleton animation={animation} variant="text" width="40%" height={16} />
            </Box>
          </Box>
        ))}
      </Box>
    )}

  // Form skeleton
  if (variant === 'form') {
    return (
    <Box sx={{ width }}>
        {Array.from({ length: rows }).map((_, index) => (
          <Box key={index} sx={{ mb: 3 }}>
            <Skeleton animation={animation} variant="text" width="30%" height={16} sx={{ mb: 1 }} />
            <Skeleton animation={animation} variant="rectangular" width="100%" height={56} />
          </Box>
        ))}
        <Skeleton animation={animation} variant="rectangular" width="30%" height={40} />
      </Box>
    )}

  // Default rectangular skeleton
  return <Skeleton animation={animation} variant="rectangular" width={width} height={height || 200} />;
};

// Dashboard skeleton composition
export const DashboardSkeleton: React.FC = () => {
  return (
    <Grid container spacing={3}>
      {/* Metric cards */}
      <Grid item xs={12} sm={6} md={3}>
        <LoadingSkeleton variant="metric" />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <LoadingSkeleton variant="metric" />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <LoadingSkeleton variant="metric" />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <LoadingSkeleton variant="metric" />
      </Grid>
      
      {/* Chart */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
            <LoadingSkeleton variant="chart" height={300} />
          </CardContent>
        </Card>
      </Grid>
      
      {/* List */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Skeleton variant="text" width="40%" height={32} sx={{ mb: 2 }} />
            <LoadingSkeleton variant="list" rows={4} />
          </CardContent>
        </Card>
      </Grid>
      
      {/* Table */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Skeleton variant="text" width="25%" height={32} sx={{ mb: 2 }} />
            <LoadingSkeleton variant="table" rows={5} columns={4} />
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  </>
  )};

// Video Queue skeleton
export const VideoQueueSkeleton: React.FC = () => {
  return (
    <>
      <Grid container spacing={2}>
      {Array.from({ length: 6 }).map((_, index) => (
        <Grid item xs={12} sm={6} md={4} key={index}>
          <Card>
            <Skeleton variant="rectangular" height={180} />
            <CardContent>
              <Skeleton variant="text" width="80%" height={24} />
              <Skeleton variant="text" width="60%" height={20} />
              <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                <Skeleton variant="rectangular" width={80} height={32} />
                <Skeleton variant="rectangular" width={80} height={32} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  </>
  )};

// Channel List skeleton
export const ChannelListSkeleton: React.FC = () => {
  return (
    <Box>
      {Array.from({ length: 5 }).map((_, index) => (
        <Card key={index} sx={{ mb: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Skeleton variant="circular" width={60} height={60} sx={{ mr: 2 }} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="40%" height={28} />
                <Skeleton variant="text" width="60%" height={20} />
                <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                  <Skeleton variant="text" width={100} height={20} />
                  <Skeleton variant="text" width={100} height={20} />
                  <Skeleton variant="text" width={100} height={20} />
                </Box>
              </Box>
      <Box sx={{ display: 'flex', gap: 1 }}>
                <Skeleton variant="circular" width={40} height={40} />
                <Skeleton variant="circular" width={40} height={40} />
              </Box>
            </Box>
          </CardContent>
        </Card>
      ))}
    </Box>
  )};