import React from 'react'
import { Box, Typography, IconButton, Tooltip, CircularProgress } from '@mui/material'
import { Refresh } from '@mui/icons-material'
import { format } from 'date-fns'

interface DashboardHeaderProps {
  title: string
  subtitle?: string
  lastUpdated?: Date
  onRefresh?: () => void
  loading?: boolean
}

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  title,
  subtitle,
  lastUpdated,
  onRefresh,
  loading = false,
}) => {
  return (
    <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Box>
        <Typography variant="h4" gutterBottom>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body1" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        {lastUpdated && (
          <Typography variant="body2" color="text.secondary">
            Last updated: {format(lastUpdated, 'MMM dd, HH:mm')}
          </Typography>
        )}
        {onRefresh && (
          <Tooltip title="Refresh data">
            <IconButton onClick={onRefresh} disabled={loading}>
              {loading ? <CircularProgress size={24} /> : <Refresh />}
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </Box>
  )
}