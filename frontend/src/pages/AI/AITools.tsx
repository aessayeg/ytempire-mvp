/**
 * AI Tools Screen Component
 * Comprehensive AI tools dashboard for content creation and automation
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  Avatar,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Slider,
  CircularProgress,
  Tabs,
  Tab,
  Badge,
  Tooltip
} from '@mui/material';
import {
  SmartToy,
  Psychology,
  AutoFixHigh,
  VideoLibrary,
  Image,
  MusicNote,
  Translate,
  Description,
  TrendingUp,
  PlayArrow,
  Pause,
  Stop,
  Download,
  Share,
  Settings,
  History,
  Star,
  AttachMoney,
  Speed,
  AccessTime,
  CheckCircle,
  Error,
  Warning
} from '@mui/icons-material';

interface AITool {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: React.ReactNode;
  provider: string;
  costPerUse: number;
  averageTime: string;
  accuracy: number;
  popular: boolean;
  status: 'available' | 'busy' | 'maintenance';
}

interface AIJob {
  id: string;
  tool: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  startTime: Date;
  endTime?: Date;
  input: string;
  output?: string;
  cost: number;
}

const AITools: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [aiTools, setAiTools] = useState<AITool[]>([]);
  const [activeJobs, setActiveJobs] = useState<AIJob[]>([]);
  const [completedJobs, setCompletedJobs] = useState<AIJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [newJobDialogOpen, setNewJobDialogOpen] = useState(false);
  const [selectedTool, setSelectedTool] = useState<AITool | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  
  const [jobSettings, setJobSettings] = useState({
    tool: '',
    input: '',
    language: 'en',
    style: 'professional',
    creativity: 50,
    maxLength: 1000
  });

  const categories = [
    { id: 'all', name: 'All Tools', icon: <SmartToy /> },
    { id: 'content', name: 'Content Creation', icon: <Description /> },
    { id: 'video', name: 'Video Generation', icon: <VideoLibrary /> },
    { id: 'image', name: 'Image Generation', icon: <Image /> },
    { id: 'audio', name: 'Audio & Voice', icon: <MusicNote /> },
    { id: 'analysis', name: 'Analysis & Research', icon: <Psychology /> },
    { id: 'optimization', name: 'Optimization', icon: <TrendingUp /> }
  ];

  useEffect(() => {
    loadAITools();
    loadJobs();
  }, []);

  const loadAITools = () => {
    // Mock AI tools data
    const tools: AITool[] = [
      {
        id: '1',
        name: 'GPT-4 Script Writer',
        description: 'Advanced script generation for YouTube videos',
        category: 'content',
        icon: <Description />,
        provider: 'OpenAI',
        costPerUse: 0.10,
        averageTime: '30s',
        accuracy: 95,
        popular: true,
        status: 'available'
      },
      {
        id: '2',
        name: 'DALL-E 3 Thumbnail',
        description: 'AI-powered thumbnail generation',
        category: 'image',
        icon: <Image />,
        provider: 'OpenAI',
        costPerUse: 0.20,
        averageTime: '45s',
        accuracy: 92,
        popular: true,
        status: 'available'
      },
      {
        id: '3',
        name: 'ElevenLabs Voice',
        description: 'Natural voice synthesis for narration',
        category: 'audio',
        icon: <MusicNote />,
        provider: 'ElevenLabs',
        costPerUse: 0.15,
        averageTime: '60s',
        accuracy: 98,
        popular: false,
        status: 'available'
      },
      {
        id: '4',
        name: 'Claude Analyzer',
        description: 'Content analysis and optimization',
        category: 'analysis',
        icon: <Psychology />,
        provider: 'Anthropic',
        costPerUse: 0.08,
        averageTime: '20s',
        accuracy: 94,
        popular: false,
        status: 'busy'
      },
      {
        id: '5',
        name: 'Trend Predictor',
        description: 'AI-powered trend analysis',
        category: 'analysis',
        icon: <TrendingUp />,
        provider: 'Custom',
        costPerUse: 0.05,
        averageTime: '15s',
        accuracy: 87,
        popular: true,
        status: 'available'
      },
      {
        id: '6',
        name: 'Video Compiler',
        description: 'Automated video assembly',
        category: 'video',
        icon: <VideoLibrary />,
        provider: 'Custom',
        costPerUse: 0.25,
        averageTime: '120s',
        accuracy: 90,
        popular: false,
        status: 'maintenance'
      }
    ];
    setAiTools(tools);
  };

  const loadJobs = () => {
    // Mock jobs data
    const active: AIJob[] = [
      {
        id: '1',
        tool: 'GPT-4 Script Writer',
        status: 'processing',
        progress: 65,
        startTime: new Date(),
        input: 'Create a script about AI trends in 2024',
        cost: 0.10
      }
    ];
    
    const completed: AIJob[] = [
      {
        id: '2',
        tool: 'DALL-E 3 Thumbnail',
        status: 'completed',
        progress: 100,
        startTime: new Date(Date.now() - 3600000),
        endTime: new Date(Date.now() - 3540000),
        input: 'Tech tutorial thumbnail',
        output: 'thumbnail_url_here',
        cost: 0.20
      }
    ];
    
    setActiveJobs(active);
    setCompletedJobs(completed);
  };

  const handleCreateJob = () => {
    if (!selectedTool) return;
    
    const newJob: AIJob = {
      id: Date.now().toString(),
      tool: selectedTool.name,
      status: 'pending',
      progress: 0,
      startTime: new Date(),
      input: jobSettings.input,
      cost: selectedTool.costPerUse
    };
    
    setActiveJobs([...activeJobs, newJob]);
    setNewJobDialogOpen(false);
    setJobSettings({
      tool: '',
      input: '',
      language: 'en',
      style: 'professional',
      creativity: 50,
      maxLength: 1000
    });
  };

  const handleToolSelect = (tool: AITool) => {
    if (tool.status !== 'available') {
      return;
    }
    setSelectedTool(tool);
    setJobSettings(prev => ({ ...prev, tool: tool.id }));
    setNewJobDialogOpen(true);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'available':
        return <CheckCircle color="success" />;
      case 'busy':
        return <Warning color="warning" />;
      case 'maintenance':
        return <Error color="error" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'primary';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const filteredTools = selectedCategory === 'all' 
    ? aiTools 
    : aiTools.filter(tool => tool.category === selectedCategory);

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          AI Tools Suite
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Powerful AI tools for content creation and automation
        </Typography>
      </Box>

      {/* Category Filter */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {categories.map(category => (
            <Chip
              key={category.id}
              icon={category.icon}
              label={category.name}
              onClick={() => setSelectedCategory(category.id)}
              color={selectedCategory === category.id ? 'primary' : 'default'}
              variant={selectedCategory === category.id ? 'filled' : 'outlined'}
            />
          ))}
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
          <Tab label="Available Tools" />
          <Tab 
            label={
              <Badge badgeContent={activeJobs.length} color="primary">
                Active Jobs
              </Badge>
            } 
          />
          <Tab label="History" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {filteredTools.map(tool => (
            <Grid item xs={12} md={6} lg={4} key={tool.id}>
              <Card 
                sx={{ 
                  cursor: tool.status === 'available' ? 'pointer' : 'not-allowed',
                  opacity: tool.status === 'available' ? 1 : 0.6,
                  '&:hover': {
                    boxShadow: tool.status === 'available' ? 4 : 1
                  }
                }}
                onClick={() => handleToolSelect(tool)}
              >
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Avatar sx={{ bgcolor: 'primary.main' }}>
                        {tool.icon}
                      </Avatar>
                      <Box>
                        <Typography variant="h6">
                          {tool.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          by {tool.provider}
                        </Typography>
                      </Box>
                    </Box>
                    <Tooltip title={`Status: ${tool.status}`}>
                      {getStatusIcon(tool.status)}
                    </Tooltip>
                  </Box>

                  <Typography variant="body2" color="text.secondary" mb={2}>
                    {tool.description}
                  </Typography>

                  <Box display="flex" gap={1} mb={2}>
                    <Chip 
                      icon={<AttachMoney />} 
                      label={`$${tool.costPerUse.toFixed(2)}`} 
                      size="small" 
                      variant="outlined" 
                    />
                    <Chip 
                      icon={<AccessTime />} 
                      label={tool.averageTime} 
                      size="small" 
                      variant="outlined" 
                    />
                    <Chip 
                      icon={<Speed />} 
                      label={`${tool.accuracy}%`} 
                      size="small" 
                      variant="outlined" 
                    />
                  </Box>

                  {tool.popular && (
                    <Chip 
                      icon={<Star />} 
                      label="Popular" 
                      color="warning" 
                      size="small" 
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={3}>
          {activeJobs.length === 0 ? (
            <Grid item xs={12}>
              <Alert severity="info">No active jobs</Alert>
            </Grid>
          ) : (
            activeJobs.map(job => (
              <Grid item xs={12} key={job.id}>
                <Paper sx={{ p: 3 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">{job.tool}</Typography>
                    <Chip 
                      label={job.status} 
                      color={getStatusColor(job.status) as any}
                      size="small"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" mb={2}>
                    {job.input}
                  </Typography>
                  <Box mb={2}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">Progress</Typography>
                      <Typography variant="body2">{job.progress}%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={job.progress} />
                  </Box>
                  <Box display="flex" gap={1}>
                    <Button size="small" startIcon={<Pause />}>Pause</Button>
                    <Button size="small" startIcon={<Stop />} color="error">Cancel</Button>
                  </Box>
                </Paper>
              </Grid>
            ))
          )}
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={3}>
          {completedJobs.length === 0 ? (
            <Grid item xs={12}>
              <Alert severity="info">No completed jobs</Alert>
            </Grid>
          ) : (
            completedJobs.map(job => (
              <Grid item xs={12} key={job.id}>
                <Paper sx={{ p: 3 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">{job.tool}</Typography>
                    <Box display="flex" gap={1}>
                      <Chip 
                        label={`$${job.cost.toFixed(2)}`} 
                        size="small"
                        variant="outlined"
                      />
                      <Chip 
                        label={job.status} 
                        color={getStatusColor(job.status) as any}
                        size="small"
                      />
                    </Box>
                  </Box>
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    Input: {job.input}
                  </Typography>
                  {job.output && (
                    <Typography variant="body2" color="text.secondary" mb={2}>
                      Output: {job.output}
                    </Typography>
                  )}
                  <Box display="flex" gap={1}>
                    <Button size="small" startIcon={<Download />}>Download</Button>
                    <Button size="small" startIcon={<Share />}>Share</Button>
                  </Box>
                </Paper>
              </Grid>
            ))
          )}
        </Grid>
      )}

      {/* New Job Dialog */}
      <Dialog open={newJobDialogOpen} onClose={() => setNewJobDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {selectedTool ? `Configure ${selectedTool.name}` : 'New AI Job'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Input/Prompt"
                multiline
                rows={4}
                value={jobSettings.input}
                onChange={(e) => setJobSettings(prev => ({ ...prev, input: e.target.value }))}
                placeholder="Enter your prompt or upload content..."
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={jobSettings.language}
                  onChange={(e) => setJobSettings(prev => ({ ...prev, language: e.target.value }))}
                  label="Language"
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Spanish</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                  <MenuItem value="ja">Japanese</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Style</InputLabel>
                <Select
                  value={jobSettings.style}
                  onChange={(e) => setJobSettings(prev => ({ ...prev, style: e.target.value }))}
                  label="Style"
                >
                  <MenuItem value="professional">Professional</MenuItem>
                  <MenuItem value="casual">Casual</MenuItem>
                  <MenuItem value="energetic">Energetic</MenuItem>
                  <MenuItem value="educational">Educational</MenuItem>
                  <MenuItem value="entertaining">Entertaining</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Typography variant="body2" gutterBottom>
                Creativity Level: {jobSettings.creativity}%
              </Typography>
              <Slider
                value={jobSettings.creativity}
                onChange={(_, value) => setJobSettings(prev => ({ ...prev, creativity: value as number }))}
                min={0}
                max={100}
                marks={[
                  { value: 0, label: 'Conservative' },
                  { value: 50, label: 'Balanced' },
                  { value: 100, label: 'Creative' }
                ]}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewJobDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            variant="contained" 
            onClick={handleCreateJob}
            disabled={loading || !jobSettings.tool || !jobSettings.input}
            startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          >
            {loading ? 'Creating...' : 'Start Job'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AITools;