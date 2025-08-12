import os

# Base directory for components
base_dir = r"C:\Users\Hp\projects\ytempire-mvp\frontend\src\components\Videos"

# Component definitions
components = {
    "VideoPreview.tsx": """import React from 'react';
import { Box, Paper, Typography, Button, Grid, Chip, Alert } from '@mui/material';
import { PlayArrow, Edit, Publish } from '@mui/icons-material';
import { VideoPlayer } from './VideoPlayer';

interface VideoPreviewProps {
  video: any;
  onEdit?: () => void;
  onPublish?: () => void;
  onApprove?: () => void;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({ video, onEdit, onPublish, onApprove }) => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Video Preview</Typography>
      <VideoPlayer videoUrl={video.video_url} thumbnail={video.thumbnail_url} />
      <Box mt={2}>
        <Typography variant="h6">{video.title}</Typography>
        <Typography variant="body2" color="text.secondary">{video.description}</Typography>
        <Box display="flex" gap={1} mt={2}>
          <Button variant="contained" startIcon={<PlayArrow />}>Preview</Button>
          <Button variant="outlined" startIcon={<Edit />} onClick={onEdit}>Edit</Button>
          <Button variant="contained" color="success" startIcon={<Publish />} onClick={onPublish}>Publish</Button>
        </Box>
      </Box>
    </Paper>
  );
};""",

    "VideoApproval.tsx": """import React, { useState } from 'react';
import { Box, Paper, Typography, Button, TextField, Rating, FormControl, RadioGroup, FormControlLabel, Radio, Alert } from '@mui/material';
import { CheckCircle, Cancel, Edit } from '@mui/icons-material';

interface VideoApprovalProps {
  video: any;
  onApprove: (feedback: any) => void;
  onReject: (reason: string) => void;
  onRequestChanges: (changes: string) => void;
}

export const VideoApproval: React.FC<VideoApprovalProps> = ({ video, onApprove, onReject, onRequestChanges }) => {
  const [decision, setDecision] = useState<'approve' | 'reject' | 'changes'>('approve');
  const [quality, setQuality] = useState(4);
  const [feedback, setFeedback] = useState('');
  
  const handleSubmit = () => {
    if (decision === 'approve') {
      onApprove({ quality, feedback });
    } else if (decision === 'reject') {
      onReject(feedback);
    } else {
      onRequestChanges(feedback);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Video Approval</Typography>
      <Box mb={2}>
        <Typography>Quality Rating</Typography>
        <Rating value={quality} onChange={(e, v) => setQuality(v || 0)} />
      </Box>
      <FormControl component="fieldset">
        <RadioGroup value={decision} onChange={(e) => setDecision(e.target.value as any)}>
          <FormControlLabel value="approve" control={<Radio />} label="Approve for Publishing" />
          <FormControlLabel value="changes" control={<Radio />} label="Request Changes" />
          <FormControlLabel value="reject" control={<Radio />} label="Reject" />
        </RadioGroup>
      </FormControl>
      <TextField fullWidth multiline rows={4} label="Feedback" value={feedback} onChange={(e) => setFeedback(e.target.value)} margin="normal" />
      <Box display="flex" gap={2} mt={2}>
        <Button variant="contained" color={decision === 'approve' ? 'success' : decision === 'reject' ? 'error' : 'warning'} onClick={handleSubmit} startIcon={decision === 'approve' ? <CheckCircle /> : decision === 'reject' ? <Cancel /> : <Edit />}>
          {decision === 'approve' ? 'Approve' : decision === 'reject' ? 'Reject' : 'Request Changes'}
        </Button>
      </Box>
    </Paper>
  );
};""",

    "VideoUploadProgress.tsx": """import React from 'react';
import { Box, Paper, Typography, LinearProgress, List, ListItem, ListItemText, Chip, Alert } from '@mui/material';
import { CloudUpload, CheckCircle, Error } from '@mui/icons-material';

interface UploadProgressProps {
  progress: number;
  status: 'preparing' | 'uploading' | 'processing' | 'completed' | 'failed';
  fileName?: string;
  fileSize?: number;
  uploadSpeed?: number;
  timeRemaining?: number;
  error?: string;
}

export const VideoUploadProgress: React.FC<UploadProgressProps> = ({ progress, status, fileName, fileSize, uploadSpeed, timeRemaining, error }) => {
  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" gap={2} mb={2}>
        <CloudUpload color="primary" />
        <Typography variant="h6">Upload Progress</Typography>
        <Chip label={status} size="small" color={status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'primary'} />
      </Box>
      
      <LinearProgress variant="determinate" value={progress} sx={{ height: 10, borderRadius: 5, mb: 2 }} />
      <Typography variant="body2" align="center">{progress}%</Typography>
      
      {fileName && <Typography variant="body2">File: {fileName}</Typography>}
      {fileSize && <Typography variant="body2">Size: {formatBytes(fileSize)}</Typography>}
      {uploadSpeed && <Typography variant="body2">Speed: {formatBytes(uploadSpeed)}/s</Typography>}
      {timeRemaining && <Typography variant="body2">Time remaining: {Math.ceil(timeRemaining / 60)} minutes</Typography>}
      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
    </Paper>
  );
};""",

    "PublishingControls.tsx": """import React, { useState } from 'react';
import { Box, Paper, Typography, Button, TextField, Select, MenuItem, FormControl, InputLabel, Switch, FormControlLabel, DateTimePicker } from '@mui/material';
import { Publish, Schedule, Visibility, VisibilityOff } from '@mui/icons-material';

interface PublishingControlsProps {
  onPublish: (settings: any) => void;
  channels: any[];
}

export const PublishingControls: React.FC<PublishingControlsProps> = ({ onPublish, channels }) => {
  const [settings, setSettings] = useState({
    publishNow: true,
    scheduledTime: null,
    visibility: 'public',
    notifySubscribers: true,
    premiere: false,
    ageRestriction: false,
    comments: true,
    likes: true,
    channelId: '',
    playlist: '',
    tags: [],
    category: ''
  });

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Publishing Settings</Typography>
      
      <FormControlLabel control={<Switch checked={settings.publishNow} onChange={(e) => setSettings({...settings, publishNow: e.target.checked})} />} label="Publish Immediately" />
      
      {!settings.publishNow && (
        <TextField type="datetime-local" label="Schedule Time" fullWidth margin="normal" InputLabelProps={{ shrink: true }} />
      )}
      
      <FormControl fullWidth margin="normal">
        <InputLabel>Visibility</InputLabel>
        <Select value={settings.visibility} onChange={(e) => setSettings({...settings, visibility: e.target.value})}>
          <MenuItem value="public">Public</MenuItem>
          <MenuItem value="unlisted">Unlisted</MenuItem>
          <MenuItem value="private">Private</MenuItem>
        </Select>
      </FormControl>
      
      <FormControlLabel control={<Switch checked={settings.notifySubscribers} />} label="Notify Subscribers" />
      <FormControlLabel control={<Switch checked={settings.comments} />} label="Allow Comments" />
      <FormControlLabel control={<Switch checked={settings.likes} />} label="Show Likes" />
      
      <Box display="flex" gap={2} mt={3}>
        <Button variant="contained" startIcon={settings.publishNow ? <Publish /> : <Schedule />} onClick={() => onPublish(settings)}>
          {settings.publishNow ? 'Publish Now' : 'Schedule'}
        </Button>
      </Box>
    </Paper>
  );
};""",

    "YouTubeUploadStatus.tsx": """import React from 'react';
import { Box, Paper, Typography, Stepper, Step, StepLabel, Alert, Link, Button, Chip } from '@mui/material';
import { YouTube, CheckCircle, Error, HourglassEmpty } from '@mui/icons-material';

interface YouTubeUploadStatusProps {
  status: 'uploading' | 'processing' | 'published' | 'failed';
  videoId?: string;
  youtubeUrl?: string;
  currentStep: number;
  error?: string;
}

export const YouTubeUploadStatus: React.FC<YouTubeUploadStatusProps> = ({ status, videoId, youtubeUrl, currentStep, error }) => {
  const steps = ['Upload Video', 'Process on YouTube', 'Set Metadata', 'Publish'];
  
  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" gap={2} mb={3}>
        <YouTube sx={{ color: '#FF0000', fontSize: 32 }} />
        <Typography variant="h6">YouTube Upload Status</Typography>
        <Chip label={status} size="small" color={status === 'published' ? 'success' : status === 'failed' ? 'error' : 'primary'} />
      </Box>
      
      <Stepper activeStep={currentStep}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      {status === 'published' && youtubeUrl && (
        <Alert severity="success" sx={{ mt: 3 }}>
          Video published successfully!
          <Link href={youtubeUrl} target="_blank" sx={{ ml: 1 }}>View on YouTube</Link>
        </Alert>
      )}
      
      {status === 'failed' && error && (
        <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>
      )}
      
      {videoId && (
        <Typography variant="body2" sx={{ mt: 2 }}>YouTube Video ID: {videoId}</Typography>
      )}
    </Paper>
  );
};""",

    "VideoMetrics.tsx": """import React, { useEffect, useState } from 'react';
import { Box, Paper, Grid, Typography, Card, CardContent, Chip } from '@mui/material';
import { TrendingUp, Visibility, ThumbUp, Comment, WatchLater, AttachMoney } from '@mui/icons-material';
import { api } from '../../services/api';

interface VideoMetricsProps {
  videoId: string;
}

export const VideoMetrics: React.FC<VideoMetricsProps> = ({ videoId }) => {
  const [metrics, setMetrics] = useState<any>(null);
  
  useEffect(() => {
    fetchMetrics();
  }, [videoId]);
  
  const fetchMetrics = async () => {
    try {
      const response = await api.videos.getAnalytics(videoId);
      setMetrics(response);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  if (!metrics) return null;

  const MetricCard = ({ icon, label, value, change }: any) => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" gap={1} mb={1}>
          {icon}
          <Typography variant="subtitle2" color="text.secondary">{label}</Typography>
        </Box>
        <Typography variant="h4">{value}</Typography>
        {change && (
          <Chip label={`${change > 0 ? '+' : ''}${change}%`} size="small" color={change > 0 ? 'success' : 'error'} />
        )}
      </CardContent>
    </Card>
  );

  return (
    <Grid container spacing={2}>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<Visibility />} label="Views" value={metrics.views?.toLocaleString()} change={metrics.viewsChange} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<ThumbUp />} label="Likes" value={metrics.likes?.toLocaleString()} change={metrics.likesChange} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<Comment />} label="Comments" value={metrics.comments?.toLocaleString()} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<WatchLater />} label="Watch Time" value={`${metrics.watchTime}h`} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<AttachMoney />} label="Revenue" value={`$${metrics.revenue?.toFixed(2)}`} change={metrics.revenueChange} />
      </Grid>
      <Grid item xs={6} md={3}>
        <MetricCard icon={<TrendingUp />} label="Engagement" value={`${metrics.engagementRate}%`} />
      </Grid>
    </Grid>
  );
};""",

    "VideoPerformanceChart.tsx": """import React from 'react';
import { Paper, Box, Typography, ToggleButtonGroup, ToggleButton } from '@mui/material';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface VideoPerformanceChartProps {
  videoId: string;
  data?: any[];
  metric?: 'views' | 'engagement' | 'revenue';
  timeRange?: '24h' | '7d' | '30d' | '90d';
}

export const VideoPerformanceChart: React.FC<VideoPerformanceChartProps> = ({ videoId, data = [], metric = 'views', timeRange = '7d' }) => {
  const [selectedMetric, setSelectedMetric] = React.useState(metric);
  const [selectedRange, setSelectedRange] = React.useState(timeRange);

  const chartData = data.length > 0 ? data : [
    { date: 'Mon', views: 1200, engagement: 45, revenue: 12.5 },
    { date: 'Tue', views: 1500, engagement: 52, revenue: 15.2 },
    { date: 'Wed', views: 1800, engagement: 48, revenue: 18.7 },
    { date: 'Thu', views: 2200, engagement: 58, revenue: 22.3 },
    { date: 'Fri', views: 2800, engagement: 62, revenue: 28.9 },
    { date: 'Sat', views: 3200, engagement: 55, revenue: 32.1 },
    { date: 'Sun', views: 2900, engagement: 50, revenue: 29.5 }
  ];

  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Performance Trends</Typography>
        <Box display="flex" gap={2}>
          <ToggleButtonGroup value={selectedMetric} exclusive onChange={(e, v) => v && setSelectedMetric(v)} size="small">
            <ToggleButton value="views">Views</ToggleButton>
            <ToggleButton value="engagement">Engagement</ToggleButton>
            <ToggleButton value="revenue">Revenue</ToggleButton>
          </ToggleButtonGroup>
          <ToggleButtonGroup value={selectedRange} exclusive onChange={(e, v) => v && setSelectedRange(v)} size="small">
            <ToggleButton value="24h">24h</ToggleButton>
            <ToggleButton value="7d">7d</ToggleButton>
            <ToggleButton value="30d">30d</ToggleButton>
            <ToggleButton value="90d">90d</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>
      
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2196F3" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#2196F3" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Area type="monotone" dataKey={selectedMetric} stroke="#2196F3" fillOpacity={1} fill="url(#colorGradient)" />
        </AreaChart>
      </ResponsiveContainer>
    </Paper>
  );
};""",

    "VideoEngagementStats.tsx": """import React from 'react';
import { Paper, Box, Typography, LinearProgress, List, ListItem, ListItemText, Chip, Grid } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface VideoEngagementStatsProps {
  stats: {
    likeRatio: number;
    commentRate: number;
    shareRate: number;
    avgViewDuration: number;
    clickThroughRate: number;
    audienceRetention: number[];
  };
}

export const VideoEngagementStats: React.FC<VideoEngagementStatsProps> = ({ stats }) => {
  const engagementData = [
    { name: 'Likes', value: stats.likeRatio || 85, color: '#4CAF50' },
    { name: 'Dislikes', value: 100 - (stats.likeRatio || 85), color: '#F44336' }
  ];

  const StatItem = ({ label, value, max = 100, color = 'primary' }: any) => (
    <Box mb={2}>
      <Box display="flex" justifyContent="space-between" mb={1}>
        <Typography variant="body2">{label}</Typography>
        <Typography variant="body2" fontWeight="bold">{value}%</Typography>
      </Box>
      <LinearProgress variant="determinate" value={(value / max) * 100} color={color} />
    </Box>
  );

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Engagement Statistics</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <StatItem label="Like/Dislike Ratio" value={stats.likeRatio || 85} />
          <StatItem label="Comment Rate" value={stats.commentRate || 12} max={50} />
          <StatItem label="Share Rate" value={stats.shareRate || 8} max={20} />
          <StatItem label="Avg View Duration" value={stats.avgViewDuration || 65} />
          <StatItem label="Click-through Rate" value={stats.clickThroughRate || 4.5} max={10} />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Like/Dislike Distribution</Typography>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={engagementData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                {engagementData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          
          <Box display="flex" justifyContent="center" gap={2} mt={2}>
            {engagementData.map((entry) => (
              <Chip key={entry.name} label={`${entry.name}: ${entry.value}%`} size="small" style={{ backgroundColor: entry.color, color: 'white' }} />
            ))}
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};""",

    "VideoSearch.tsx": """import React, { useState } from 'react';
import { Box, TextField, InputAdornment, IconButton, Menu, MenuItem, Chip, Button } from '@mui/material';
import { Search, FilterList, Clear, CalendarToday, Sort } from '@mui/icons-material';

interface VideoSearchProps {
  onSearch: (query: string, filters: any) => void;
  onClear: () => void;
}

export const VideoSearch: React.FC<VideoSearchProps> = ({ onSearch, onClear }) => {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({ dateRange: 'all', sortBy: 'relevance', status: 'all' });
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleSearch = () => {
    onSearch(query, filters);
  };

  const handleClear = () => {
    setQuery('');
    setFilters({ dateRange: 'all', sortBy: 'relevance', status: 'all' });
    onClear();
  };

  return (
    <Box display="flex" gap={2} alignItems="center">
      <TextField
        fullWidth
        placeholder="Search videos..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Search />
            </InputAdornment>
          ),
          endAdornment: query && (
            <InputAdornment position="end">
              <IconButton size="small" onClick={handleClear}>
                <Clear />
              </IconButton>
            </InputAdornment>
          )
        }}
      />
      
      <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
        <FilterList />
      </IconButton>
      
      <Button variant="contained" onClick={handleSearch}>Search</Button>
      
      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
        <MenuItem>Date Range</MenuItem>
        <MenuItem>Sort By</MenuItem>
        <MenuItem>Status</MenuItem>
      </Menu>
      
      {Object.values(filters).some(v => v !== 'all') && (
        <Box display="flex" gap={1}>
          {filters.dateRange !== 'all' && <Chip label={filters.dateRange} size="small" onDelete={() => setFilters({...filters, dateRange: 'all'})} />}
          {filters.sortBy !== 'relevance' && <Chip label={filters.sortBy} size="small" onDelete={() => setFilters({...filters, sortBy: 'relevance'})} />}
          {filters.status !== 'all' && <Chip label={filters.status} size="small" onDelete={() => setFilters({...filters, status: 'all'})} />}
        </Box>
      )}
    </Box>
  );
};""",

    "VideoFilters.tsx": """import React, { useState } from 'react';
import { Box, Paper, Typography, FormControl, InputLabel, Select, MenuItem, Slider, Chip, Button, Accordion, AccordionSummary, AccordionDetails, FormGroup, FormControlLabel, Checkbox } from '@mui/material';
import { ExpandMore, FilterList, Clear } from '@mui/icons-material';

interface VideoFiltersProps {
  onApplyFilters: (filters: any) => void;
  onClearFilters: () => void;
}

export const VideoFilters: React.FC<VideoFiltersProps> = ({ onApplyFilters, onClearFilters }) => {
  const [filters, setFilters] = useState({
    status: 'all',
    dateRange: 'all',
    channel: 'all',
    qualityScore: [0, 100],
    duration: 'all',
    hasRevenue: false,
    minViews: 0,
    categories: []
  });

  const handleApply = () => {
    onApplyFilters(filters);
  };

  const handleClear = () => {
    setFilters({
      status: 'all',
      dateRange: 'all',
      channel: 'all',
      qualityScore: [0, 100],
      duration: 'all',
      hasRevenue: false,
      minViews: 0,
      categories: []
    });
    onClearFilters();
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          <FilterList sx={{ mr: 1, verticalAlign: 'middle' }} />
          Filters
        </Typography>
        <Button size="small" startIcon={<Clear />} onClick={handleClear}>Clear All</Button>
      </Box>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Status</InputLabel>
        <Select value={filters.status} onChange={(e) => setFilters({...filters, status: e.target.value})}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="draft">Draft</MenuItem>
          <MenuItem value="published">Published</MenuItem>
          <MenuItem value="scheduled">Scheduled</MenuItem>
          <MenuItem value="processing">Processing</MenuItem>
        </Select>
      </FormControl>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Date Range</InputLabel>
        <Select value={filters.dateRange} onChange={(e) => setFilters({...filters, dateRange: e.target.value})}>
          <MenuItem value="all">All Time</MenuItem>
          <MenuItem value="today">Today</MenuItem>
          <MenuItem value="week">This Week</MenuItem>
          <MenuItem value="month">This Month</MenuItem>
          <MenuItem value="year">This Year</MenuItem>
        </Select>
      </FormControl>

      <Box mt={2} mb={1}>
        <Typography variant="body2">Quality Score</Typography>
        <Slider value={filters.qualityScore} onChange={(e, v) => setFilters({...filters, qualityScore: v as number[]})} valueLabelDisplay="auto" min={0} max={100} />
      </Box>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Duration</InputLabel>
        <Select value={filters.duration} onChange={(e) => setFilters({...filters, duration: e.target.value})}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="short">Short (< 3 min)</MenuItem>
          <MenuItem value="medium">Medium (3-10 min)</MenuItem>
          <MenuItem value="long">Long (> 10 min)</MenuItem>
        </Select>
      </FormControl>

      <FormControlLabel control={<Checkbox checked={filters.hasRevenue} onChange={(e) => setFilters({...filters, hasRevenue: e.target.checked})} />} label="Has Revenue" />

      <Box mt={2}>
        <Button fullWidth variant="contained" onClick={handleApply}>Apply Filters</Button>
      </Box>
    </Paper>
  );
};"""
}

# Create all component files
for filename, content in components.items():
    filepath = os.path.join(base_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Created: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to create {filename}: {e}")

print(f"\nCreated {len(components)} video components successfully!")