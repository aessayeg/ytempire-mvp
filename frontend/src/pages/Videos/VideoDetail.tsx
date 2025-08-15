import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Grid,
  Typography,
  Button,
  Chip,
  Divider,
  Tab,
  Tabs,
  CircularProgress,
  Alert,
  IconButton,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ArrowBack,
  Edit,
  Delete,
  Publish,
  Schedule,
  PlayArrow,
  ContentCopy,
  OpenInNew,
  Download,
  Refresh,
  CheckCircle,
  Error as ErrorIcon,
  AttachMoney,
  TrendingUp,
  Speed,
  AccessTime,
} from '@mui/icons-material';
import { api } from '../../services/api';
import { VideoPlayer } from '../../components/Videos/VideoPlayer';
import { VideoMetrics } from '../../components/Videos/VideoMetrics';
import { formatDistanceToNow } from 'date-fns';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`video-tabpanel-${index}`}
      aria-labelledby={`video-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const VideoDetail: React.FC = () => {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();
  const [video, setVideo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [publishDialogOpen, setPublishDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editForm, setEditForm] = useState({
    title: '',
    description: '',
    tags: [],
  });
  const [publishSchedule, setPublishSchedule] = useState('');

  useEffect(() => {
    if (videoId) {
      fetchVideoDetails();
    }
  }, [videoId]);

  const fetchVideoDetails = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.videos.get(videoId!);
      setVideo(response);
      setEditForm({
        title: response.title,
        description: response.description || '',
        tags: response.tags || [],
      });
    } catch (err: unknown) {
      setError(err.message || 'Failed to load video details');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleEdit = async () => {
    try {
      await api.videos.update(videoId!, editForm);
      await fetchVideoDetails();
      setEditDialogOpen(false);
    } catch (err: unknown) {
      setError(err.message || 'Failed to update video');
    }
  };

  const handlePublish = async () => {
    try {
      const scheduledTime = publishSchedule ? new Date(publishSchedule) : undefined;
      await api.videos.publish(videoId!, scheduledTime);
      await fetchVideoDetails();
      setPublishDialogOpen(false);
    } catch (err: unknown) {
      setError(err.message || 'Failed to publish video');
    }
  };

  const handleDelete = async () => {
    try {
      await api.videos.delete(videoId!);
      navigate('/videos');
    } catch (err: unknown) {
      setError(err.message || 'Failed to delete video');
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(window.location.href);
  };

  const handleOpenYouTube = () => {
    if (video?.youtube_url) {
      window.open(video.youtube_url, '_blank');
    }
  };

  const getStatusColor = (status: string): unknown => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getPublishStatusColor = (status: string): unknown => {
    switch (status) {
      case 'published':
        return 'success';
      case 'scheduled':
        return 'info';
      case 'publishing':
        return 'warning';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!video) {
    return (
      <Box p={3}>
        <Alert severity="warning">Video not found</Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box display="flex" alignItems="center" gap={2}>
          <IconButton onClick={() => navigate('/videos')}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h4">{video.title}</Typography>
        </Box>
        <Box display="flex" gap={1}>
          <IconButton onClick={fetchVideoDetails}>
            <Refresh />
          </IconButton>
          <IconButton onClick={handleCopyLink}>
            <ContentCopy />
          </IconButton>
          {video.youtube_url && (
            <IconButton onClick={handleOpenYouTube}>
              <OpenInNew />
            </IconButton>
          )}
        </Box>
      </Box>

      {/* Status and Actions */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box display="flex" gap={2}>
          <Chip
            label={video.generation_status}
            color={getStatusColor(video.generation_status)}
            icon={video.generation_status === 'completed' ? <CheckCircle /> : undefined}
          />
          <Chip
            label={video.publish_status}
            color={getPublishStatusColor(video.publish_status)}
          />
          {video.quality_score && (
            <Chip
              label={`Quality: ${video.quality_score.toFixed(0)}%`}
              color="primary"
              variant="outlined"
            />
          )}
        </Box>
        <Box display="flex" gap={1}>
          {video.generation_status === 'completed' && (
            <>
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={() => setTabValue(1)}
              >
                Preview
              </Button>
              {video.publish_status === 'draft' && (
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<Publish />}
                  onClick={() => setPublishDialogOpen(true)}
                >
                  Publish
                </Button>
              )}
            </>
          )}
          <Button
            variant="outlined"
            startIcon={<Edit />}
            onClick={() => setEditDialogOpen(true)}
          >
            Edit
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={<Delete />}
            onClick={() => setDeleteDialogOpen(true)}
          >
            Delete
          </Button>
        </Box>
      </Box>

      {/* Main Content */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Details" />
              <Tab label="Preview" disabled={video.generation_status !== 'completed'} />
              <Tab label="Analytics" disabled={!video.published_at} />
              <Tab label="History" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              {/* Video Details */}
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Description
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {video.description || 'No description available'}
                  </Typography>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Tags
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    {video.tags?.length > 0 ? (
                      video.tags.map((tag: string, index: number) => (
                        <Chip key={index} label={tag} size="small" />
                      ))
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No tags
                      </Typography>
                    )}
                  </Box>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Script
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography
                      variant="body2"
                      sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}
                    >
                      {video.script || 'Script not available'}
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {/* Video Preview */}
              {video.video_url ? (
                <VideoPlayer videoUrl={video.video_url} thumbnail={video.thumbnail_url} />
              ) : (
                <Alert severity="info">Video preview not available</Alert>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              {/* Analytics */}
              <VideoMetrics videoId={video.id} />
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              {/* History */}
              <List>
                <ListItem>
                  <ListItemText
                    primary="Created"
                    secondary={new Date(video.created_at).toLocaleString()}
                  />
                </ListItem>
                {video.generation_started_at && (
                  <ListItem>
                    <ListItemText
                      primary="Generation Started"
                      secondary={new Date(video.generation_started_at).toLocaleString()}
                    />
                  </ListItem>
                )}
                {video.generation_completed_at && (
                  <ListItem>
                    <ListItemText
                      primary="Generation Completed"
                      secondary={new Date(video.generation_completed_at).toLocaleString()}
                    />
                  </ListItem>
                )}
                {video.published_at && (
                  <ListItem>
                    <ListItemText
                      primary="Published"
                      secondary={new Date(video.published_at).toLocaleString()}
                    />
                  </ListItem>
                )}
              </List>
            </TabPanel>
          </Paper>
        </Grid>

        {/* Sidebar */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {/* Cost Breakdown */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <AttachMoney /> Cost Breakdown
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Script Generation" />
                      <Typography variant="body2">
                        ${video.script_cost?.toFixed(2) || '0.00'}
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Voice Synthesis" />
                      <Typography variant="body2">
                        ${video.voice_cost?.toFixed(2) || '0.00'}
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Video Processing" />
                      <Typography variant="body2">
                        ${video.video_cost?.toFixed(2) || '0.00'}
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Thumbnail" />
                      <Typography variant="body2">
                        ${video.thumbnail_cost?.toFixed(2) || '0.00'}
                      </Typography>
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemText primary={<strong>Total</strong>} />
                      <Typography variant="h6" color="primary">
                        ${video.total_cost?.toFixed(2) || '0.00'}
                      </Typography>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Performance Metrics */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Speed /> Performance
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Quality Score" />
                      <Typography variant="body2" color="primary">
                        {video.quality_score?.toFixed(0) || 'N/A'}%
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Trend Score" />
                      <Typography variant="body2" color="primary">
                        {video.trend_score?.toFixed(0) || 'N/A'}%
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Engagement Prediction" />
                      <Typography variant="body2" color="primary">
                        {video.engagement_prediction?.toFixed(0) || 'N/A'}%
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Generation Time" />
                      <Typography variant="body2">
                        {video.generation_time_seconds
                          ? `${Math.round(video.generation_time_seconds / 60)} min`
                          : 'N/A'}
                      </Typography>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Technical Details */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Technical Details
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Video ID" />
                      <Typography variant="caption">{video.id}</Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Channel ID" />
                      <Typography variant="caption">{video.channel_id}</Typography>
                    </ListItem>
                    {video.youtube_video_id && (
                      <ListItem>
                        <ListItemText primary="YouTube ID" />
                        <Typography variant="caption">{video.youtube_video_id}</Typography>
                      </ListItem>
                    )}
                    <ListItem>
                      <ListItemText primary="Duration" />
                      <Typography variant="body2">
                        {video.duration_seconds
                          ? `${Math.floor(video.duration_seconds / 60)}:${(
                              video.duration_seconds % 60
                            )
                              .toString()
                              .padStart(2, '0')}`
                          : 'N/A'}
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Category" />
                      <Typography variant="body2">{video.category}</Typography>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Video</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Title"
            value={editForm.title}
            onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Description"
            value={editForm.description}
            onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
            multiline
            rows={4}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Tags (comma separated)"
            value={editForm.tags.join(', ')}
            onChange={(e) =>
              setEditForm({ ...editForm, tags: e.target.value.split(',').map((t) => t.trim()) })
            }
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEdit} variant="contained">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Publish Dialog */}
      <Dialog open={publishDialogOpen} onClose={() => setPublishDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Publish Video</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            Choose when to publish this video:
          </Typography>
          <Box mt={2}>
            <Button
              fullWidth
              variant="contained"
              onClick={() => {
                setPublishSchedule('');
                handlePublish();
              }}
              sx={{ mb: 2 }}
            >
              Publish Now
            </Button>
            <Typography variant="body2" align="center" sx={{ my: 2 }}>
              OR
            </Typography>
            <TextField
              fullWidth
              label="Schedule for later"
              type="datetime-local"
              value={publishSchedule}
              onChange={(e) => setPublishSchedule(e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPublishDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handlePublish}
            variant="contained"
            disabled={!publishSchedule && publishSchedule !== ''}
          >
            Schedule
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Video</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this video? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};