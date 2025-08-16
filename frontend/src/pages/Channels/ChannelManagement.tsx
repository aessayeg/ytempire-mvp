/**
 * Channel Management UI Component
 * MVP Screen Design - Channel management interface
 */
import React, { useState, useEffect } from 'react';
import { 
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Avatar,
  Chip,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel
 } from '@mui/material';
import { 
  Add,
  Edit,
  Delete,
  YouTube,
  Settings,
  Analytics,
  Schedule,
  CheckCircle,
  PlayCircle,
  Pause,
  MoreVert
 } from '@mui/icons-material';

interface Channel {
  
id: string;
name: string;

youtubeId: string;
thumbnail: string;

status: 'active' | 'paused' | 'pending' | 'error';
subscribers: number;

totalVideos: number;
totalViews: number;

isMonetized: boolean;
autoUpload: boolean;

uploadSchedule: string;
category: string;

apiQuota: {;
used: number;

limit: number;

};
  lastSync: Date;
  created: Date}

interface TabPanelProps {
  
children?: React.ReactNode;
index: number;
value: number;

}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <>
      <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  )}

export const ChannelManagement: React.FC = () => {
  const [channels, setChannels] = useState<Channel[]>([ {
      id: '1',
      name: 'Tech Reviews Pro',
      youtubeId: 'UCxxxxxxxxxxxxx',
      thumbnail: '/api/placeholder/80/80',
      status: 'active',
      subscribers: 125000,
      totalVideos: 342,
      totalViews: 15600000,
      isMonetized: true,
      autoUpload: true,
      uploadSchedule: 'Daily at, 2:00 PM',
      category: 'Technology',
      apiQuota: { used: 8500, limit: 10000 },
      lastSync: new Date(),
      created: new Date('2024-01-15')

    },
    {
      id: '2',
      name: 'Gaming Highlights',
      youtubeId: 'UCyyyyyyyyyyyyy',
      thumbnail: '/api/placeholder/80/80',
      status: 'active',
      subscribers: 89000,
      totalVideos: 256,
      totalViews: 9800000,
      isMonetized: true,
      autoUpload: false,
      uploadSchedule: '3 times per week',
      category: 'Gaming',
      apiQuota: { used: 6200, limit: 10000 },
      lastSync: new Date(),
      created: new Date('2024-02-20')

    },
    {
      id: '3',
      name: 'Cooking Adventures',
      youtubeId: 'UCzzzzzzzzzzzzz',
      thumbnail: '/api/placeholder/80/80',
      status: 'paused',
      subscribers: 45000,
      totalVideos: 128,
      totalViews: 3200000,
      isMonetized: false,
      autoUpload: true,
      uploadSchedule: 'Twice weekly',
      category: 'Food & Cooking',
      apiQuota: { used: 3100, limit: 10000 },
      lastSync: new Date(),
      created: new Date('2024-03-10')

    } ]</>
  );

  const [tabValue, setTabValue] = useState(0);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingChannel, setEditingChannel] = useState<Channel | null>(null);
  const [formData, setFormData] = useState({ name: '',
    youtubeId: '',
    category: '',
    autoUpload: false,
    uploadSchedule: '' });

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
};
  const handleAddChannel = () => { setEditingChannel(null);
    setFormData({
      name: '',
      youtubeId: '',
      category: '',
      autoUpload: false,
      uploadSchedule: '' });
    setOpenDialog(true)
};
  const handleEditChannel = (channel: Channel) => { setEditingChannel(channel);
    setFormData({
      name: channel.name,
      youtubeId: channel.youtubeId,
      category: channel.category,
      autoUpload: channel.autoUpload,
      uploadSchedule: channel.uploadSchedule });
    setOpenDialog(true)
};
  const handleSaveChannel = () => {
    
    // Save channel logic
    setOpenDialog(false)
};
  const handleToggleStatus = (channelId: string) => {
    setChannels(channels.map(c =>
      c.id === channelId 
        ? { ...c, status: c.status === 'active' ? 'paused' : 'active' as any }
        : c
    ))
};
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'pending': return 'info';
      case 'error': return 'error';
      default: return 'default'}
  };
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle />;
      case 'paused': return <Pause />;
      case 'pending': return <Schedule />;
      case 'error': return <ErrorIcon />;
      default: return null}
  };
  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${num / 1000000.toFixed(1}M`;
    if (num >= 1000) return `${num / 1000.toFixed(1}K`;
    return num.toString()};
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Channel Management
          </Typography>
      <Typography variant="body2" color="text.secondary">
            Manage your YouTube channels and automation settings
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleAddChannel}
          size="large"
        >
          Add Channel
        </Button>
      </Box>

      {/* Stats Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Channels
              </Typography>
              <Typography variant="h4">
                {channels.length}
              </Typography>
              <Chip 
                label={`${channels.filter(c => c.status === 'active').length} Active`}
                color="success"
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Subscribers
              </Typography>
              <Typography variant="h4">
                {formatNumber(channels.reduce((sum, c) => sum + c.subscribers, 0))}
              </Typography>
              <Typography variant="body2" color="success.main">
                +12.5% this month
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Videos
              </Typography>
              <Typography variant="h4">
                {channels.reduce((sum, c) => sum + c.totalVideos, 0)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Across all channels
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Views
              </Typography>
              <Typography variant="h4">
                {formatNumber(channels.reduce((sum, c) => sum + c.totalViews, 0))}
              </Typography>
              <Typography variant="body2" color="success.main">
                +25% this month
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="All Channels" />
          <Tab label="Active" />
          <Tab label="Paused" />
          <Tab label="Settings" />
        </Tabs>
      </Paper>

      {/* Channel List */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {channels.map((channel) => (
            <Grid item xs={12} md={6} lg={4} key={channel.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Avatar
                      src={channel.thumbnail}
                      sx={{ width: 60, height: 60, mr: 2 }}
                    />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6">
                        {channel.name}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          icon={getStatusIcon(channel.status)}
                          label={channel.status}
                          color={getStatusColor(channel.status) as any}
                          size="small"
                        />
                        {channel.isMonetized && (
                          <Chip
                            label="Monetized"
                            color="success"
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    </Box>
                    <IconButton onClick={() => handleEditChannel(channel}>
                      <MoreVert />
                    </IconButton>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  {/* Channel Stats */}
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Subscribers
                      </Typography>
                      <Typography variant="h6">
                        {formatNumber(channel.subscribers)}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Videos
                      </Typography>
                      <Typography variant="h6">
                        {channel.totalVideos}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2" color="text.secondary">
                        Views
                      </Typography>
                      <Typography variant="h6">
                        {formatNumber(channel.totalViews)}
                      </Typography>
                    </Grid>
                  </Grid>

                  {/* API Quota */}
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        API Quota
                      </Typography>
                      <Typography variant="body2">
                        {channel.apiQuota.used} / {channel.apiQuota.limit}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={(channel.apiQuota.used / channel.apiQuota.limit) * 100}
                      color={
                        channel.apiQuota.used / channel.apiQuota.limit > 0.8
                          ? 'warning'
                          : 'primary'
                      }
                    />
                  </Box>

                  {/* Auto Upload Status */}
                  <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Auto Upload
                    </Typography>
                    <Switch
                      checked={channel.autoUpload}
                      onChange={() =>
                      size="small"
                    />
                  </Box>
                </CardContent>

                <CardActions>
                  <Button
                    size="small"
                    startIcon={<Analytics />}
                  >
                    Analytics
                  </Button>
                  <Button
                    size="small"
                    startIcon={<Settings />}
                  >
                    Settings
                  </Button>
                  <IconButton
                    size="small"
                    onClick={() => handleToggleStatus(channel.id}
                    color={channel.status === 'active' ? 'warning' : 'success'}
                  >
                    {channel.status === 'active' ? <Pause /> </>: <PlayCircle />}
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      {/* Add/Edit Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingChannel ? 'Edit Channel' : 'Add New Channel'}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Channel Name"
            fullWidth
            variant="outlined"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value)})}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="YouTube Channel ID"
            fullWidth
            variant="outlined"
            value={formData.youtubeId}
            onChange={(e) => setFormData({ ...formData, youtubeId: e.target.value)})}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={formData.category}
              label="Category"
              onChange={(e) => setFormData({ ...formData, category: e.target.value)})}
            >
              <MenuItem value="Technology">Technology</MenuItem>
              <MenuItem value="Gaming">Gaming</MenuItem>
              <MenuItem value="Education">Education</MenuItem>
              <MenuItem value="Entertainment">Entertainment</MenuItem>
              <MenuItem value="Food & Cooking">Food & Cooking</MenuItem>
              <MenuItem value="Travel">Travel</MenuItem>
              <MenuItem value="Other">Other</MenuItem>
            </Select>
          </FormControl>
          <FormControlLabel
            control={
              <Switch
                checked={formData.autoUpload}
                onChange={(e) => setFormData({ ...formData, autoUpload: e.target.checked })}
              />}
            label="Enable Auto Upload"
          />
          {formData.autoUpload && (
            <TextField
              margin="dense"
              label="Upload Schedule"
              fullWidth
              variant="outlined"
              value={formData.uploadSchedule}
              onChange={(e) => setFormData({ ...formData, uploadSchedule: e.target.value)})}
              placeholder="e.g., Daily at 2:00 PM"
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false}>Cancel</Button>
          <Button onClick={handleSaveChannel} variant="contained">
            {editingChannel ? 'Save Changes' : 'Add Channel'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}}
