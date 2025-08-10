import React from 'react'
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Box,
  Chip,
} from '@mui/material'
import {
  VideoLibrary,
  TrendingUp,
  AttachMoney,
  ThumbUp,
  Comment,
} from '@mui/icons-material'
import { formatDistanceToNow } from 'date-fns'

const mockActivities = [
  {
    id: 1,
    type: 'video_published',
    title: 'New video published',
    description: 'Top 10 AI Tools for 2024',
    time: new Date(Date.now() - 1000 * 60 * 30),
    icon: <VideoLibrary />,
    color: '#8884d8',
  },
  {
    id: 2,
    type: 'milestone',
    title: 'Milestone reached',
    description: '1M total views achieved',
    time: new Date(Date.now() - 1000 * 60 * 60 * 2),
    icon: <TrendingUp />,
    color: '#82ca9d',
  },
  {
    id: 3,
    type: 'revenue',
    title: 'Revenue earned',
    description: '$245 from ad revenue',
    time: new Date(Date.now() - 1000 * 60 * 60 * 4),
    icon: <AttachMoney />,
    color: '#ffc658',
  },
  {
    id: 4,
    type: 'engagement',
    title: 'High engagement',
    description: '500+ likes on recent video',
    time: new Date(Date.now() - 1000 * 60 * 60 * 6),
    icon: <ThumbUp />,
    color: '#ff7c7c',
  },
  {
    id: 5,
    type: 'comment',
    title: 'New comments',
    description: '25 new comments to review',
    time: new Date(Date.now() - 1000 * 60 * 60 * 8),
    icon: <Comment />,
    color: '#8dd1e1',
  },
]

export const RecentActivity: React.FC = () => {
  const getActivityChip = (type: string) => {
    switch (type) {
      case 'video_published':
        return <Chip label="Video" size="small" color="primary" />
      case 'milestone':
        return <Chip label="Milestone" size="small" color="success" />
      case 'revenue':
        return <Chip label="Revenue" size="small" color="warning" />
      case 'engagement':
        return <Chip label="Engagement" size="small" color="info" />
      case 'comment':
        return <Chip label="Comment" size="small" color="default" />
      default:
        return null
    }
  }

  return (
    <Paper sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Recent Activity
      </Typography>
      <List>
        {mockActivities.map((activity) => (
          <ListItem key={activity.id} alignItems="flex-start" sx={{ px: 0 }}>
            <ListItemAvatar>
              <Avatar sx={{ bgcolor: activity.color }}>
                {activity.icon}
              </Avatar>
            </ListItemAvatar>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="subtitle2">{activity.title}</Typography>
                  {getActivityChip(activity.type)}
                </Box>
              }
              secondary={
                <>
                  <Typography variant="body2" color="text.secondary">
                    {activity.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatDistanceToNow(activity.time, { addSuffix: true })}
                  </Typography>
                </>
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  )
}