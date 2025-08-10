import React from 'react'
import { Card, CardContent, Typography, Box, Chip } from '@mui/material'
import { TrendingUp, TrendingDown } from '@mui/icons-material'

interface MetricCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  trend?: string
  trendDirection?: 'up' | 'down' | 'neutral'
  color?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  icon,
  trend,
  trendDirection = 'neutral',
  color = '#8884d8',
}) => {
  const getTrendColor = () => {
    switch (trendDirection) {
      case 'up':
        return 'success'
      case 'down':
        return 'error'
      default:
        return 'default'
    }
  }

  const getTrendIcon = () => {
    switch (trendDirection) {
      case 'up':
        return <TrendingUp fontSize="small" />
      case 'down':
        return <TrendingDown fontSize="small" />
      default:
        return null
    }
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ color, display: 'flex', alignItems: 'center' }}>
            {icon}
          </Box>
          {trend && (
            <Chip
              label={trend}
              size="small"
              color={getTrendColor()}
              icon={getTrendIcon()}
            />
          )}
        </Box>
        <Typography variant="h4" component="div" gutterBottom>
          {value}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {title}
        </Typography>
      </CardContent>
    </Card>
  )
}