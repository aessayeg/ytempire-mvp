/**
 * Channel List Component
 * Owner: Frontend Team Lead
 * 
 * Channel list with filtering and sorting using Zustand
 */

import React, { useState } from 'react'
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Avatar,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Box,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  Button,
  LinearProgress
} from '@mui/material'
import {
  Search,
  MoreVert,
  Add,
  YouTube,
  TrendingUp,
  VideoLibrary,
  Subscribers
} from '@mui/icons-material'
import { useChannels, useUI } from '@/hooks/useStores'
import type { Channel } from '@/services/channelService'

const ChannelCard: React.FC<{ channel: Channel }> = ({ channel }) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const { actions } = useChannels()
  const { actions: uiActions } = useUI()
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }
  
  const handleMenuClose = () => {
    setAnchorEl(null)
  }
  
  const handleEdit = () => {
    actions.select(channel)
    uiActions.openModal({
      component: 'ChannelEditModal',
      props: { channelId: channel.id }
    })
    handleMenuClose()
  }
  
  const handleDelete = () => {
    uiActions.showConfirm({
      title: 'Delete Channel',
      message: `Are you sure you want to delete "${channel.name}"? This action cannot be undone.`,
      variant: 'danger',
      onConfirm: async () => {
        const success = await actions.delete(channel.id)
        if (success) {
          uiActions.showSuccess('Channel deleted successfully')
        } else {
          uiActions.showError('Failed to delete channel')
        }
      }
    })
    handleMenuClose()
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'connecting': return 'warning'
      case 'failed': return 'error'
      default: return 'default'
    }
  }
  
  const getStatusIcon = (status: string) => {
    if (channel.youtube_channel_id) {
      return <YouTube fontSize="small" color="error" />
    }
    return null
  }
  
  return (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        {/* Header */}
        <Box display="flex" alignItems="flex-start" justifyContent="space-between" mb={2}>
          <Box display="flex" alignItems="center" gap={2} flex={1}>
            <Avatar
              src={channel.avatar_url}
              alt={channel.name}
              sx={{ width: 48, height: 48 }}
            >
              {channel.name.charAt(0).toUpperCase()}
            </Avatar>
            <Box flex={1}>
              <Typography variant="h6" noWrap>
                {channel.name}
              </Typography>
              <Typography variant="body2" color="textSecondary" noWrap>
                {channel.category} • {channel.language}
              </Typography>
            </Box>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            {getStatusIcon(channel.status)}
            <IconButton size="small" onClick={handleMenuOpen}>
              <MoreVert />
            </IconButton>
          </Box>
        </Box>
        
        {/* Status */}
        <Box mb={2}>
          <Chip
            label={channel.status}
            color={getStatusColor(channel.status) as any}
            size="small"
          />
        </Box>
        
        {/* Description */}
        <Typography variant="body2" color="textSecondary" mb={2}>
          {channel.description.length > 100
            ? `${channel.description.substring(0, 100)}...`
            : channel.description}
        </Typography>
        
        {/* Statistics */}
        <Grid container spacing={2}>
          <Grid item xs={4}>
            <Box textAlign="center">
              <Typography variant="h6" component="div">
                {channel.subscriber_count.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Subscribers
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box textAlign="center">
              <Typography variant="h6" component="div">
                {channel.video_count}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Videos
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box textAlign="center">
              <Typography variant="h6" component="div">
                {(channel.total_views / 1000).toFixed(1)}K
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Views
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
      
      {/* Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEdit}>Edit Channel</MenuItem>
        <MenuItem onClick={() => {
          // Navigate to channel details
          handleMenuClose()
        }}>
          View Analytics
        </MenuItem>
        {channel.youtube_channel_id && (
          <MenuItem onClick={() => {
            // Sync from YouTube
            handleMenuClose()
          }}>
            Sync from YouTube
          </MenuItem>
        )}
        <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
          Delete Channel
        </MenuItem>
      </Menu>
    </Card>
  )
}

export const ChannelList: React.FC = () => {
  const {
    channels,
    isLoading,
    error,
    filters,
    actions
  } = useChannels()
  
  const { actions: uiActions } = useUI()
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    actions.setFilters({ searchQuery: event.target.value })
  }
  
  const handleStatusFilter = (event: any) => {
    actions.setFilters({ status: event.target.value })
  }
  
  const handleCategoryFilter = (event: any) => {
    actions.setFilters({ category: event.target.value || null })
  }
  
  const handleCreateChannel = () => {
    uiActions.openModal({
      component: 'ChannelCreateModal'
    })
  }
  
  if (isLoading && channels.length === 0) {
    return (
      <Box>
        <Typography variant="h5" gutterBottom>
          Loading Channels...
        </Typography>
        <LinearProgress />
      </Box>
    )
  }
  
  if (error) {
    return (
      <Box textAlign="center" py={4}>
        <Typography variant="h6" color="error" gutterBottom>
          Failed to load channels
        </Typography>
        <Typography variant="body2" color="textSecondary" gutterBottom>
          {error}
        </Typography>
        <Button variant="outlined" onClick={() => actions.clearError()}>
          Retry
        </Button>
      </Box>
    )
  }
  
  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Your Channels ({channels.length})
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleCreateChannel}
        >
          Create Channel
        </Button>
      </Box>
      
      {/* Filters */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            placeholder="Search channels..."
            value={filters.searchQuery}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              )
            }}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Status</InputLabel>
            <Select
              value={filters.status}
              label="Status"
              onChange={handleStatusFilter}
            >
              <MenuItem value="all">All Statuses</MenuItem>
              <MenuItem value="active">Active</MenuItem>
              <MenuItem value="connecting">Connecting</MenuItem>
              <MenuItem value="inactive">Inactive</MenuItem>
              <MenuItem value="failed">Failed</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Category</InputLabel>
            <Select
              value={filters.category || ''}
              label="Category"
              onChange={handleCategoryFilter}
            >
              <MenuItem value="">All Categories</MenuItem>
              <MenuItem value="Entertainment">Entertainment</MenuItem>
              <MenuItem value="Education">Education</MenuItem>
              <MenuItem value="Technology">Technology</MenuItem>
              <MenuItem value="Gaming">Gaming</MenuItem>
              <MenuItem value="Music">Music</MenuItem>
              <MenuItem value="Sports">Sports</MenuItem>
              <MenuItem value="News">News</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={2}>
          <Button
            fullWidth
            variant="outlined"
            onClick={actions.resetFilters}
            sx={{ height: '56px' }}
          >
            Clear Filters
          </Button>
        </Grid>
      </Grid>
      
      {/* Channel Grid */}
      {channels.length === 0 ? (
        <Box textAlign="center" py={8}>
          <Typography variant="h6" gutterBottom>
            No channels found
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            {filters.searchQuery || filters.status !== 'all' || filters.category
              ? 'Try adjusting your filters or search query'
              : 'Create your first channel to get started'}
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleCreateChannel}
            sx={{ mt: 2 }}
          >
            Create Your First Channel
          </Button>
        </Box>
      ) : (
        <Grid container spacing={3}>
          {channels.map(channel => (
            <Grid item xs={12} md={6} lg={4} key={channel.id}>
              <ChannelCard channel={channel} />
            </Grid>
          ))}
        </Grid>
      )}
      
      {/* Loading indicator for refresh */}
      {isLoading && channels.length > 0 && (
        <Box mt={2}>
          <LinearProgress />
        </Box>
      )}
    </Box>
  )
}

export default ChannelList