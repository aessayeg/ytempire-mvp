import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Tooltip,
  Badge,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  YouTube as YouTubeIcon,
  Settings as SettingsIcon,
  Analytics as AnalyticsIcon,
  SwapHoriz as SwapIcon
} from '@mui/icons-material';
import { useChannelStore } from '../../stores/channelStore';
import { Channel, ChannelHealth } from '../../types/channel';

interface ChannelManagerProps {
  onChannelSelect?: (channel: Channel) => void;
  maxChannels?: number;
}

const ChannelManager: React.FC<ChannelManagerProps> = ({ 
  onChannelSelect,
  maxChannels = 15 
}) => {
  const [channels, setChannels] = useState<Channel[]>([]);
  const [selectedChannel, setSelectedChannel] = useState<Channel | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [newChannel, setNewChannel] = useState({
    name: '',
    youtubeId: '',
    apiKey: '',
    category: '',
    description: ''
  });

  // Mock channel health status
  const getHealthIcon = (health: ChannelHealth) => {
    switch (health) {
      case 'healthy':
        return <CheckCircleIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'critical':
        return <ErrorIcon color="error" />;
      default:
        return <CheckCircleIcon color="disabled" />;
    }
  };

  const getHealthColor = (health: ChannelHealth) => {
    switch (health) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  useEffect(() => {
    loadChannels();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  const loadChannels = async () => {
    setIsLoading(true);
    try {
      // TODO: Fetch channels from API
      const mockChannels: Channel[] = [
        {
          id: '1',
          name: 'Tech Reviews Pro',
          youtubeId: 'UC_tech_reviews',
          category: 'Technology',
          health: 'healthy',
          quota: { used: 3500, limit: 10000 },
          subscribers: 15420,
          videoCount: 145,
          isActive: true,
          lastSync: new Date().toISOString()
        },
        {
          id: '2',
          name: 'Gaming Central',
          youtubeId: 'UC_gaming_central',
          category: 'Gaming',
          health: 'warning',
          quota: { used: 8500, limit: 10000 },
          subscribers: 28300,
          videoCount: 312,
          isActive: true,
          lastSync: new Date().toISOString()
        },
        {
          id: '3',
          name: 'DIY Crafts Hub',
          youtubeId: 'UC_diy_crafts',
          category: 'Lifestyle',
          health: 'critical',
          quota: { used: 9800, limit: 10000 },
          subscribers: 8900,
          videoCount: 89,
          isActive: false,
          lastSync: new Date().toISOString()
        }
      ];
      setChannels(mockChannels);
    } catch (_error) {
      console.error('Failed to load channels:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddChannel = async () => {
    if (channels.length >= maxChannels) {
      alert(`Maximum of ${maxChannels} channels allowed`);
      return;
    }
    
    // TODO: Add channel via API
    const newChannelData: Channel = {
      id: Date.now().toString(),
      name: newChannel.name,
      youtubeId: newChannel.youtubeId,
      category: newChannel.category,
      health: 'healthy',
      quota: { used: 0, limit: 10000 },
      subscribers: 0,
      videoCount: 0,
      isActive: true,
      lastSync: new Date().toISOString()
    };
    
    setChannels([...channels, newChannelData]);
    setIsAddDialogOpen(false);
    setNewChannel({ name: '', youtubeId: '', apiKey: '', category: '', description: '' });
  };

  const handleDeleteChannel = async (channelId: string) => {
    if (confirm('Are you sure you want to delete this channel?')) {
      setChannels(channels.filter(c => c.id !== channelId));
    }
  };

  const handleToggleChannel = async (channelId: string) => {
    setChannels(channels.map(c => 
      c.id === channelId ? { ...c, isActive: !c.isActive } : c
    ));
  };

  const handleRefreshChannel = async (channelId: string) => {
    // TODO: Refresh channel data from YouTube API
    console.log('Refreshing channel:', channelId);
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Multi-Channel Manager
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <FormControlLabel
            control={
              <Switch
                checked={autoRotate}
                onChange={(e) => setAutoRotate(e.target.checked)}
                color="primary"
              />
            }
            label="Auto-Rotate"
          />
          <Chip
            label={`${channels.length} / ${maxChannels} Channels`}
            color={channels.length >= maxChannels ? 'error' : 'primary'}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setIsAddDialogOpen(true)}
            disabled={channels.length >= maxChannels}
          >
            Add Channel
          </Button>
        </Box>
      </Box>

      {/* Alert for quota warnings */}
      {channels.some(c => c.health === 'critical') && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Some channels have critical quota usage. Consider rotating to other channels.
        </Alert>
      )}

      {/* Channels Grid */}
      {isLoading ? (
        <LinearProgress />
      ) : (
        <Grid container spacing={3}>
          {channels.map((channel) => (
            <Grid item xs={12} md={6} lg={4} key={channel.id}>
              <Card 
                sx={{ 
                  position: 'relative',
                  opacity: channel.isActive ? 1 : 0.6,
                  border: selectedChannel?.id === channel.id ? 2 : 0,
                  borderColor: 'primary.main'
                }}
              >
                <CardContent>
                  {/* Channel Header */}
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <YouTubeIcon color="error" />
                      <Typography variant="h6" component="h2">
                        {channel.name}
                      </Typography>
                    </Box>
                    <Tooltip title={`Health: ${channel.health}`}>
                      {getHealthIcon(channel.health)}
                    </Tooltip>
                  </Box>

                  {/* Channel Info */}
                  <Box mb={2}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      ID: {channel.youtubeId}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Category: {channel.category}
                    </Typography>
                    <Box display="flex" gap={2} mt={1}>
                      <Chip 
                        label={`${channel.subscribers.toLocaleString()} subs`} 
                        size="small" 
                        variant="outlined"
                      />
                      <Chip 
                        label={`${channel.videoCount} videos`} 
                        size="small" 
                        variant="outlined"
                      />
                    </Box>
                  </Box>

                  {/* Quota Usage */}
                  <Box mb={2}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">Quota Usage</Typography>
                      <Typography variant="body2" color={getHealthColor(channel.health)}>
                        {channel.quota.used.toLocaleString()} / {channel.quota.limit.toLocaleString()}
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={(channel.quota.used / channel.quota.limit) * 100}
                      color={getHealthColor(channel.health) as any}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>

                  {/* Actions */}
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Switch
                        checked={channel.isActive}
                        onChange={() => handleToggleChannel(channel.id)}
                        size="small"
                      />
                      <Typography variant="caption" ml={1}>
                        {channel.isActive ? 'Active' : 'Inactive'}
                      </Typography>
                    </Box>
                    <Box>
                      <IconButton 
                        size="small" 
                        onClick={() => handleRefreshChannel(channel.id)}
                        title="Refresh"
                      >
                        <RefreshIcon />
                      </IconButton>
                      <IconButton 
                        size="small"
                        onClick={() => {
                          setSelectedChannel(channel);
                          onChannelSelect?.(channel);
                        }}
                        title="Analytics"
                      >
                        <AnalyticsIcon />
                      </IconButton>
                      <IconButton 
                        size="small"
                        onClick={() => {
                          setSelectedChannel(channel);
                          setIsEditDialogOpen(true);
                        }}
                        title="Edit"
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton 
                        size="small"
                        onClick={() => handleDeleteChannel(channel.id)}
                        color="error"
                        title="Delete"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Add Channel Dialog */}
      <Dialog open={isAddDialogOpen} onClose={() => setIsAddDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Channel</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} mt={2}>
            <TextField
              label="Channel Name"
              value={newChannel.name}
              onChange={(e) => setNewChannel({ ...newChannel, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="YouTube Channel ID"
              value={newChannel.youtubeId}
              onChange={(e) => setNewChannel({ ...newChannel, youtubeId: e.target.value })}
              fullWidth
              required
              helperText="Found in YouTube Studio > Settings"
            />
            <TextField
              label="API Key"
              value={newChannel.apiKey}
              onChange={(e) => setNewChannel({ ...newChannel, apiKey: e.target.value })}
              fullWidth
              required
              type="password"
              helperText="YouTube Data API v3 key"
            />
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={newChannel.category}
                onChange={(e) => setNewChannel({ ...newChannel, category: e.target.value })}
                label="Category"
              >
                <MenuItem value="Technology">Technology</MenuItem>
                <MenuItem value="Gaming">Gaming</MenuItem>
                <MenuItem value="Education">Education</MenuItem>
                <MenuItem value="Entertainment">Entertainment</MenuItem>
                <MenuItem value="Lifestyle">Lifestyle</MenuItem>
                <MenuItem value="News">News</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Description"
              value={newChannel.description}
              onChange={(e) => setNewChannel({ ...newChannel, description: e.target.value })}
              fullWidth
              multiline
              rows={3}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsAddDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleAddChannel} 
            variant="contained"
            disabled={!newChannel.name || !newChannel.youtubeId || !newChannel.apiKey}
          >
            Add Channel
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChannelManager;