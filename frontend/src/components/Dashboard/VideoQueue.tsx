import React from 'react'
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Box,
  LinearProgress,
} from '@mui/material'
import {
  Schedule,
  PlayCircleOutline,
  CheckCircle,
  HourglassEmpty,
} from '@mui/icons-material'

interface VideoQueueProps {
  queuedCount: number
  processingCount: number
  completedCount: number
}

const mockQueueItems = [
  { id: 1, title: 'Top 10 AI Tools for 2024', status: 'processing', progress: 65 },
  { id: 2, title: 'How to Start a YouTube Channel', status: 'queued', progress: 0 },
  { id: 3, title: 'Best Productivity Apps Review', status: 'processing', progress: 30 },
  { id: 4, title: 'Machine Learning Basics', status: 'queued', progress: 0 },
  { id: 5, title: 'Web Development Trends', status: 'completed', progress: 100 },
]

export const VideoQueue: React.FC<VideoQueueProps> = ({
  queuedCount,
  processingCount,
  completedCount,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <PlayCircleOutline color="warning" />
      case 'completed':
        return <CheckCircle color="success" />
      case 'queued':
        return <Schedule color="action" />
      default:
        return <HourglassEmpty />
    }
  }

  const getStatusColor = (status: string): any => {
    switch (status) {
      case 'processing':
        return 'warning'
      case 'completed':
        return 'success'
      case 'queued':
        return 'default'
      default:
        return 'default'
    }
  }

  return (
    <Paper sx={{ p: 3, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Video Queue</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip label={`Queued: ${queuedCount}`} size="small" />
          <Chip label={`Processing: ${processingCount}`} size="small" color="warning" />
          <Chip label={`Completed: ${completedCount}`} size="small" color="success" />
        </Box>
      </Box>
      <List>
        {mockQueueItems.map((item) => (
          <ListItem key={item.id} sx={{ px: 0 }}>
            <ListItemIcon>{getStatusIcon(item.status)}</ListItemIcon>
            <ListItemText
              primary={item.title}
              secondary={
                <Box sx={{ mt: 1 }}>
                  {item.status === 'processing' && (
                    <LinearProgress
                      variant="determinate"
                      value={item.progress}
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  )}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                    <Chip
                      label={item.status}
                      size="small"
                      color={getStatusColor(item.status)}
                    />
                    {item.status === 'processing' && (
                      <Typography variant="caption">{item.progress}%</Typography>
                    )}
                  </Box>
                </Box>
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  )
}