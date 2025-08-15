import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Grid,
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
} from '@mui/material';
import {
  YouTube,
  MoreVert,
  Edit,
  Delete,
  Analytics,
  Link as LinkIcon,
  Add,
  CheckCircle,
  Warning,
  TrendingUp,
  VideoLibrary,
  Visibility,
  MonetizationOn,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { channelApi } from '../../services/api';
import { formatNumber, formatCurrency } from '../../utils/formatters';

interface Channel {
  id: string;
  name: string;
  description?: string;
  category: string;
  target_audience: string;
  upload_schedule: string;
  language: string;
  youtube_channel_id?: string;
  youtube_channel_url?: string;
  is_active: boolean;
  is_verified: boolean;
  total_videos: number;
  total_views: number;
  total_revenue: number;
  subscriber_count?: number;
  created_at: string;
  updated_at?: string;
}

export const ChannelList: React.FC = () => {
  const navigate = useNavigate();
  const [channels, setChannels] = useState<Channel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedChannel, setSelectedChannel] = useState<Channel | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editForm, setEditForm] = useState<Partial<Channel>>({});

  useEffect(() => {
    fetchChannels();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  const fetchChannels = async () => {
    try {
      setLoading(true);
      const response = await channelApi.getChannels();
      setChannels(response.data);
      setError(null);
    } catch (error: unknown) {
      setError('Failed to load channels. Please try again.');
      console.error('Error fetching channels:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, channel: Channel) => {
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

  const handleViewAnalytics = () => {
    if (selectedChannel) {
      navigate(`/channels/${selectedChannel.id}/analytics`);
    }
    handleMenuClose();
  };

  const handleConnectYouTube = () => {
    if (selectedChannel) {
      navigate(`/channels/${selectedChannel.id}/connect`);
    }
    handleMenuClose();
  };

  const confirmDelete = async () => {
    if (selectedChannel) {
      try {
        await channelApi.deleteChannel(selectedChannel.id);
        setChannels(channels.filter(c => c.id !== selectedChannel.id));
        setDeleteDialogOpen(false);
        setSelectedChannel(null);
      } catch (_error) {
        console.error('Error deleting channel:', error);
      }
    }
  };

  const handleEditSubmit = async () => {
    if (selectedChannel && editForm) {
      try {
        const response = await channelApi.updateChannel(selectedChannel.id, editForm);
        setChannels(channels.map(c => 
          c.id === selectedChannel.id ? { ...c, ...response.data } : c
        ));
        setEditDialogOpen(false);
        setSelectedChannel(null);
        setEditForm({});
      } catch (_error) {
        console.error('Error updating channel:', error);
      }
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      gaming: '#e91e63',
      education: '#2196f3',
      technology: '#9c27b0',
      entertainment: '#ff9800',
      music: '#4caf50',
      sports: '#f44336',
      news: '#607d8b',
      howto: '#00bcd4',
      travel: '#8bc34a',
      food: '#ff5722',
    };
    return colors[category.toLowerCase()] || '#757575';
  };

  if (loading) {
    return (
      <Grid container spacing={3}>
        {[1, 2, 3].map((i) => (
          <Grid item xs={12} md={6} lg={4} key={i}>
            <Card>
              <CardContent>
                <Skeleton variant="text" width="60%" height={32} />
                <Skeleton variant="text" width="100%" />
                <Skeleton variant="text" width="80%" />
                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                  <Skeleton variant="rectangular" width={60} height={24} />
                  <Skeleton variant="rectangular" width={60} height={24} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h5" fontWeight="bold">
              My Channels
            </Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => navigate('/channels/create')}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                },
              }}
            >
              Create Channel
            </Button>
          </Box>
        </Grid>

        {channels.length === 0 ? (
          <Grid item xs={12}>
            <Card sx={{ textAlign: 'center', py: 6 }}>
              <YouTube sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                No channels yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Create your first channel to start generating videos
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => navigate('/channels/create')}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                  },
                }}
              >
                Create Your First Channel
              </Button>
            </Card>
          </Grid>
        ) : (
          channels.map((channel) => (
            <Grid item xs={12} md={6} lg={4} key={channel.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Avatar
                        sx={{
                          bgcolor: getCategoryColor(channel.category),
                          width: 40,
                          height: 40,
                        }}
                      >
                        <YouTube />
                      </Avatar>
                      <Box>
                        <Typography variant="h6" component="div">
                          {channel.name}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          {channel.is_verified ? (
                            <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                          ) : (
                            <Warning sx={{ fontSize: 16, color: 'warning.main' }} />
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {channel.is_verified ? 'Connected' : 'Not connected'}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuClick(e, channel)}
                    >
                      <MoreVert />
                    </IconButton>
                  </Box>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ mb: 2, minHeight: 40 }}
                  >
                    {channel.description || 'No description provided'}
                  </Typography>

                  <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                    <Chip
                      label={channel.category}
                      size="small"
                      sx={{
                        bgcolor: getCategoryColor(channel.category),
                        color: 'white',
                      }}
                    />
                    <Chip
                      label={channel.upload_schedule}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={channel.language.toUpperCase()}
                      size="small"
                      variant="outlined"
                    />
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <VideoLibrary sx={{ fontSize: 18, color: 'text.secondary' }} />
                        <Typography variant="body2" color="text.secondary">
                          {formatNumber(channel.total_videos)} videos
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Visibility sx={{ fontSize: 18, color: 'text.secondary' }} />
                        <Typography variant="body2" color="text.secondary">
                          {formatNumber(channel.total_views)} views
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <TrendingUp sx={{ fontSize: 18, color: 'text.secondary' }} />
                        <Typography variant="body2" color="text.secondary">
                          {formatNumber(channel.subscriber_count || 0)} subs
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <MonetizationOn sx={{ fontSize: 18, color: 'text.secondary' }} />
                        <Typography variant="body2" color="text.secondary">
                          {formatCurrency(channel.total_revenue)}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>

                <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
                  <Button
                    size="small"
                    startIcon={<Analytics />}
                    onClick={() => navigate(`/channels/${channel.id}/analytics`)}
                  >
                    Analytics
                  </Button>
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() => navigate(`/channels/${channel.id}`)}
                    sx={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                      },
                    }}
                  >
                    Manage
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))
        )}
      </Grid>

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEdit}>
          <Edit sx={{ mr: 1, fontSize: 20 }} />
          Edit Channel
        </MenuItem>
        <MenuItem onClick={handleViewAnalytics}>
          <Analytics sx={{ mr: 1, fontSize: 20 }} />
          View Analytics
        </MenuItem>
        {selectedChannel && !selectedChannel.is_verified && (
          <MenuItem onClick={handleConnectYouTube}>
            <LinkIcon sx={{ mr: 1, fontSize: 20 }} />
            Connect YouTube
          </MenuItem>
        )}
        <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1, fontSize: 20 }} />
          Delete Channel
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Channel</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedChannel?.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Edit Channel</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
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
                onChange={(e: SelectChangeEvent) => 
                  setEditForm({ ...editForm, upload_schedule: e.target.value })
                }
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
    </>
  );
};