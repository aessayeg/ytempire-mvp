import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Chip,
  Avatar,
  IconButton,
  Menu,
  MenuItem,
  Skeleton,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  LinearProgress,
  Tooltip
} from '@mui/material';
import Grid from '@mui/material/Grid2';
import {
  MoreVert,
  Edit,
  Delete,
  Visibility,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  YouTube,
  TrendingUp,
  TrendingDown,
  Schedule
} from '@mui/icons-material';
import { api } from '../../services/api';
import { format } from 'date-fns';

interface Channel {
  id: string;
  name: string;
  channel_id: string;
  description?: string;
  subscriber_count: number;
  video_count: number;
  status: 'active' | 'inactive' | 'error';
  last_upload?: string;
  upload_schedule?: string;
  created_at: string;
  health_score: number;
  quota_usage: number;
  monetization_enabled: boolean;
  analytics?: {
    views_last_30_days: number;
    revenue_last_30_days: number;
    average_view_duration: number;
    engagement_rate: number;
  };
}

const ChannelList: React.FC = () => {
  const [channels, setChannels] = useState<Channel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedChannel, setSelectedChannel] = useState<Channel | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editForm, setEditForm] = useState<Partial<Channel>>({});
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  useEffect(() => {
    fetchChannels();
  }, []);

  const fetchChannels = async () => {
    try {
      setLoading(true);
      const response = await api.get('/channels');
      setChannels(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load channels');
      console.error('Error fetching channels:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, channel: Channel) => {
    setAnchorEl(event.currentTarget);
    setSelectedChannel(channel);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleEdit = () => {
    if (selectedChannel) {
      setEditForm(selectedChannel);
      setEditDialogOpen(true);
    }
    handleMenuClose();
  };

  const handleDelete = () => {
    setDeleteDialogOpen(true);
    handleMenuClose();
  };

  const handleEditSubmit = async () => {
    if (!selectedChannel) return;
    
    try {
      await api.patch(`/channels/${selectedChannel.id}`, editForm);
      await fetchChannels();
      setEditDialogOpen(false);
      setEditForm({});
    } catch (err) {
      console.error('Error updating channel:', err);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedChannel) return;
    
    try {
      await api.delete(`/channels/${selectedChannel.id}`);
      await fetchChannels();
      setDeleteDialogOpen(false);
      setSelectedChannel(null);
    } catch (err) {
      console.error('Error deleting channel:', err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getHealthIcon = (score: number) => {
    if (score >= 80) return <CheckCircle color="success" />;
    if (score >= 50) return <Warning color="warning" />;
    return <ErrorIcon color="error" />;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  if (loading) {
    return (
      <Box>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid key={i} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card>
                <CardContent>
                  <Skeleton variant="circular" width={40} height={40} />
                  <Skeleton variant="text" sx={{ mt: 2 }} />
                  <Skeleton variant="text" />
                  <Skeleton variant="rectangular" height={60} sx={{ mt: 2 }} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" action={
        <Button color="inherit" size="small" onClick={fetchChannels}>
          Retry
        </Button>
      }>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">YouTube Channels</Typography>
        <Button variant="contained" startIcon={<YouTube />}>
          Connect New Channel
        </Button>
      </Box>

      <Grid container spacing={3}>
        {channels.map((channel) => (
          <Grid key={channel.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Avatar sx={{ bgcolor: 'primary.main', width: 48, height: 48 }}>
                    <YouTube />
                  </Avatar>
                  <IconButton size="small" onClick={(e) => handleMenuOpen(e, channel)}>
                    <MoreVert />
                  </IconButton>
                </Box>

                <Typography variant="h6" gutterBottom>
                  {channel.name}
                </Typography>

                <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                  <Chip
                    label={channel.status}
                    size="small"
                    color={getStatusColor(channel.status)}
                  />
                  <Tooltip title="Health Score">
                    <Chip
                      icon={getHealthIcon(channel.health_score)}
                      label={`${channel.health_score}%`}
                      size="small"
                      variant="outlined"
                    />
                  </Tooltip>
                  {channel.monetization_enabled && (
                    <Chip label="Monetized" size="small" color="success" variant="outlined" />
                  )}
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Subscribers:</strong> {formatNumber(channel.subscriber_count)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Videos:</strong> {channel.video_count}
                  </Typography>
                  {channel.last_upload && (
                    <Typography variant="body2" color="text.secondary">
                      <strong>Last Upload:</strong> {format(new Date(channel.last_upload), 'MMM d, yyyy')}
                    </Typography>
                  )}
                </Box>

                {channel.analytics && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Last 30 Days
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Visibility fontSize="small" color="action" />
                        <Typography variant="caption">
                          {formatNumber(channel.analytics.views_last_30_days)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {channel.analytics.engagement_rate > 5 ? (
                          <TrendingUp fontSize="small" color="success" />
                        ) : (
                          <TrendingDown fontSize="small" color="error" />
                        )}
                        <Typography variant="caption">
                          {channel.analytics.engagement_rate.toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>
                    {channel.analytics.revenue_last_30_days > 0 && (
                      <Typography variant="caption" color="success.main">
                        Revenue: ${channel.analytics.revenue_last_30_days.toFixed(2)}
                      </Typography>
                    )}
                  </Box>
                )}

                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Quota Usage
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={channel.quota_usage}
                    color={channel.quota_usage > 80 ? 'error' : channel.quota_usage > 50 ? 'warning' : 'primary'}
                    sx={{ mt: 0.5 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {channel.quota_usage}% of daily limit
                  </Typography>
                </Box>
              </CardContent>

              <CardActions>
                <Button size="small" startIcon={<Visibility />}>
                  View Details
                </Button>
                <Button size="small" startIcon={<Schedule />}>
                  Schedule
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEdit}>
          <Edit sx={{ mr: 1 }} /> Edit
        </MenuItem>
        <MenuItem onClick={() => {/* View analytics */}}>
          <Visibility sx={{ mr: 1 }} /> View Analytics
        </MenuItem>
        <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Channel</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Channel Name"
              fullWidth
              value={editForm.name || ''}
              onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
            />
            <TextField
              label="Description"
              fullWidth
              multiline
              rows={3}
              value={editForm.description || ''}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
            />
            <FormControl fullWidth>
              <InputLabel>Upload Schedule</InputLabel>
              <Select
                value={editForm.upload_schedule || ''}
                label="Upload Schedule"
                onChange={(e: SelectChangeEvent) => setEditForm({ ...editForm, upload_schedule: e.target.value })}
              >
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="biweekly">Bi-weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
                <MenuItem value="custom">Custom</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEditSubmit} variant="contained">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Channel</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedChannel?.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChannelList;