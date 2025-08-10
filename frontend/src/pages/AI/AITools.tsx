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
  CardMedia,
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
  Slider,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Alert,
  Divider,
  useTheme,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  SmartToy,
  AutoAwesome,
  Psychology,
  Mic,
  Image as ImageIcon,
  VideoLibrary,
  Article,
  Translate,
  TrendingUp,
  Schedule,
  PlayArrow,
  Pause,
  Stop,
  Download,
  Upload,
  Edit,
  Delete,
  Add,
  Settings,
  Speed,
  Tune,
  Lightbulb,
  ChatBubble,
  VolumeUp,
  Palette,
  Movie,
  TextFields,
  ExpandMore,
  CheckCircle,
  Warning,
  Info,
  Refresh,
  Launch,
} from '@mui/icons-material';

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
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const aiToolsData = [
  {
    id: 1,
    name: 'Script Generator',
    description: 'Generate engaging YouTube scripts using AI',
    icon: <Article />,
    category: 'Content',
    status: 'active',
    usage: 85,
    lastUsed: '2 hours ago',
    model: 'GPT-4',
    features: ['SEO optimized', 'Multi-language', 'Custom tone', 'Hook generation'],
    pricing: '$0.02/request',
  },
  {
    id: 2,
    name: 'Voice Synthesis',
    description: 'Convert text to natural sounding speech',
    icon: <Mic />,
    category: 'Audio',
    status: 'active',
    usage: 72,
    lastUsed: '1 hour ago',
    model: 'ElevenLabs',
    features: ['Voice cloning', 'Multiple accents', 'Emotion control', 'Speed adjustment'],
    pricing: '$0.18/minute',
  },
  {
    id: 3,
    name: 'Thumbnail Creator',
    description: 'AI-powered thumbnail generation and optimization',
    icon: <ImageIcon />,
    category: 'Visual',
    status: 'active',
    usage: 68,
    lastUsed: '3 hours ago',
    model: 'DALL-E 3',
    features: ['A/B testing', 'Style templates', 'Brand consistency', 'CTR optimization'],
    pricing: '$0.04/image',
  },
  {
    id: 4,
    name: 'Video Editor',
    description: 'Automated video editing and assembly',
    icon: <VideoLibrary />,
    category: 'Video',
    status: 'active',
    usage: 45,
    lastUsed: '5 hours ago',
    model: 'Custom AI',
    features: ['Auto-cut', 'Scene detection', 'Music sync', 'Subtitle generation'],
    pricing: '$0.25/minute',
  },
  {
    id: 5,
    name: 'Trend Analyzer',
    description: 'Analyze YouTube trends and suggest topics',
    icon: <TrendingUp />,
    category: 'Analytics',
    status: 'active',
    usage: 91,
    lastUsed: '30 minutes ago',
    model: 'GPT-4 + Analytics',
    features: ['Real-time trends', 'Niche analysis', 'Competition research', 'Viral prediction'],
    pricing: '$0.15/analysis',
  },
  {
    id: 6,
    name: 'Translation Service',
    description: 'Multi-language content translation',
    icon: <Translate />,
    category: 'Content',
    status: 'beta',
    usage: 23,
    lastUsed: '1 day ago',
    model: 'Google Translate + AI',
    features: ['50+ languages', 'Context aware', 'Cultural adaptation', 'Subtitle sync'],
    pricing: '$0.08/1000 words',
  },
  {
    id: 7,
    name: 'SEO Optimizer',
    description: 'Optimize titles, descriptions, and tags',
    icon: <AutoAwesome />,
    category: 'Marketing',
    status: 'active',
    usage: 77,
    lastUsed: '2 hours ago',
    model: 'GPT-4 + SEO API',
    features: ['Keyword research', 'Competitor analysis', 'Trend integration', 'Performance tracking'],
    pricing: '$0.12/optimization',
  },
  {
    id: 8,
    name: 'Content Scheduler',
    description: 'AI-powered optimal publishing schedule',
    icon: <Schedule />,
    category: 'Analytics',
    status: 'active',
    usage: 56,
    lastUsed: '6 hours ago',
    model: 'Analytics AI',
    features: ['Audience analysis', 'Time optimization', 'Global scheduling', 'Performance prediction'],
    pricing: '$0.05/schedule',
  },
];

const mockActiveJobs = [
  {
    id: 1,
    tool: 'Script Generator',
    task: 'Generating script for "AI Tools for Creators"',
    progress: 75,
    status: 'processing',
    estimatedTime: '2 minutes',
  },
  {
    id: 2,
    tool: 'Voice Synthesis',
    task: 'Converting script to speech (Professional voice)',
    progress: 45,
    status: 'processing',
    estimatedTime: '5 minutes',
  },
  {
    id: 3,
    tool: 'Thumbnail Creator',
    task: 'Generating 3 thumbnail variations',
    progress: 90,
    status: 'processing',
    estimatedTime: '30 seconds',
  },
];

const mockTemplates = [
  {
    id: 1,
    name: 'Tech Review Template',
    category: 'Script',
    usage: 45,
    lastModified: '2 days ago',
    description: 'Template for technology product reviews',
  },
  {
    id: 2,
    name: 'Tutorial Format',
    category: 'Script',
    usage: 67,
    lastModified: '1 week ago',
    description: 'Step-by-step tutorial structure',
  },
  {
    id: 3,
    name: 'Modern Gaming Thumbnail',
    category: 'Visual',
    usage: 89,
    lastModified: '3 days ago',
    description: 'High-contrast gaming thumbnail style',
  },
  {
    id: 4,
    name: 'Professional Voice',
    category: 'Audio',
    usage: 34,
    lastModified: '1 day ago',
    description: 'Clear, professional narration voice',
  },
];

export const AITools: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTool, setSelectedTool] = useState<any>(null);
  const [toolDialogOpen, setToolDialogOpen] = useState(false);
  const [newJobDialogOpen, setNewJobDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const [jobSettings, setJobSettings] = useState({
    tool: '',
    input: '',
    model: '',
    creativity: 70,
    quality: 'high',
    language: 'en',
    style: 'professional',
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleToolClick = (tool: any) => {
    setSelectedTool(tool);
    setToolDialogOpen(true);
  };

  const handleStartJob = () => {
    setNewJobDialogOpen(true);
  };

  const handleCreateJob = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setNewJobDialogOpen(false);
    }, 2000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'beta': return 'warning';
      case 'processing': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Content': return <Article />;
      case 'Audio': return <VolumeUp />;
      case 'Visual': return <Palette />;
      case 'Video': return <Movie />;
      case 'Analytics': return <TrendingUp />;
      case 'Marketing': return <AutoAwesome />;
      default: return <SmartToy />;
    }
  };

  const toolCategories = ['All', 'Content', 'Audio', 'Visual', 'Video', 'Analytics', 'Marketing'];
  const [selectedCategory, setSelectedCategory] = useState('All');

  const filteredTools = selectedCategory === 'All' 
    ? aiToolsData 
    : aiToolsData.filter(tool => tool.category === selectedCategory);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            AI Tools Studio
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Harness the power of AI to automate your YouTube content creation
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button variant="outlined" startIcon={<Settings />}>
            Configure AI
          </Button>
          <Button variant="contained" startIcon={<Add />} onClick={handleStartJob}>
            New Job
          </Button>
        </Box>
      </Box>

      {/* Active Jobs Alert */}
      {mockActiveJobs.length > 0 && (
        <Alert 
          severity="info" 
          sx={{ mb: 3 }}
          action={
            <Button size="small" color="inherit">
              View Queue
            </Button>
          }
        >
          {mockActiveJobs.length} AI job(s) currently processing
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Left Panel - Tools and Categories */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            {/* Category Filter */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Tool Categories
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {toolCategories.map((category) => (
                  <Chip
                    key={category}
                    label={category}
                    onClick={() => setSelectedCategory(category)}
                    color={selectedCategory === category ? 'primary' : 'default'}
                    variant={selectedCategory === category ? 'filled' : 'outlined'}
                    icon={getCategoryIcon(category)}
                  />
                ))}
              </Box>
            </Box>

            {/* Tools Grid */}
            <Grid container spacing={2}>
              {filteredTools.map((tool) => (
                <Grid item xs={12} sm={6} md={4} key={tool.id}>
                  <Card 
                    sx={{ 
                      height: '100%', 
                      cursor: 'pointer',
                      '&:hover': {
                        boxShadow: theme.shadows[4],
                        transform: 'translateY(-2px)',
                      },
                      transition: 'all 0.2s ease-in-out',
                    }}
                    onClick={() => handleToolClick(tool)}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                          {tool.icon}
                        </Avatar>
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="h6" noWrap>
                            {tool.name}
                          </Typography>
                          <Chip 
                            label={tool.status} 
                            size="small" 
                            color={getStatusColor(tool.status) as any}
                            variant="outlined"
                          />
                        </Box>
                      </Box>

                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, height: 40 }}>
                        {tool.description}
                      </Typography>

                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2">Usage</Typography>
                          <Typography variant="body2">{tool.usage}%</Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={tool.usage} 
                          color={tool.usage > 80 ? 'warning' : 'primary'}
                          sx={{ height: 6, borderRadius: 3 }}
                        />
                      </Box>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="caption" color="text.secondary">
                          {tool.lastUsed}
                        </Typography>
                        <Typography variant="body2" fontWeight="bold" color="primary">
                          {tool.pricing}
                        </Typography>
                      </Box>

                      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {tool.features.slice(0, 2).map((feature, idx) => (
                          <Chip key={idx} label={feature} size="small" variant="outlined" />
                        ))}
                        {tool.features.length > 2 && (
                          <Chip label={`+${tool.features.length - 2} more`} size="small" />
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Right Panel - Active Jobs & Templates */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Jobs
            </Typography>
            
            {mockActiveJobs.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <SmartToy sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  No active jobs
                </Typography>
              </Box>
            ) : (
              <List dense>
                {mockActiveJobs.map((job) => (
                  <ListItem key={job.id} sx={{ px: 0 }}>
                    <ListItemIcon>
                      <CircularProgress size={24} variant="determinate" value={job.progress} />
                    </ListItemIcon>
                    <ListItemText
                      primary={job.tool}
                      secondary={
                        <Box>
                          <Typography variant="caption" display="block">
                            {job.task}
                          </Typography>
                          <Typography variant="caption" color="primary">
                            {job.estimatedTime} remaining
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton size="small">
                        <Pause />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>

          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Templates</Typography>
              <Button size="small" startIcon={<Add />}>
                New
              </Button>
            </Box>

            <List dense>
              {mockTemplates.map((template) => (
                <ListItem key={template.id} sx={{ px: 0 }}>
                  <ListItemIcon>
                    {getCategoryIcon(template.category)}
                  </ListItemIcon>
                  <ListItemText
                    primary={template.name}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {template.description}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Used {template.usage} times • {template.lastModified}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton size="small">
                      <Launch />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Tool Details Dialog */}
      <Dialog 
        open={toolDialogOpen} 
        onClose={() => setToolDialogOpen(false)} 
        maxWidth="md" 
        fullWidth
      >
        {selectedTool && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  {selectedTool.icon}
                </Avatar>
                <Box>
                  <Typography variant="h6">{selectedTool.name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedTool.description}
                  </Typography>
                </Box>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Features
                  </Typography>
                  <List dense>
                    {selectedTool.features.map((feature: string, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemIcon>
                          <CheckCircle color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    Usage Statistics
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Model: {selectedTool.model}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Pricing: {selectedTool.pricing}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Last used: {selectedTool.lastUsed}
                    </Typography>
                  </Box>
                  
                  <Box>
                    <Typography variant="body2" gutterBottom>
                      Monthly Usage: {selectedTool.usage}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={selectedTool.usage} 
                      color={selectedTool.usage > 80 ? 'warning' : 'primary'}
                    />
                  </Box>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography>Advanced Settings</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={<Switch defaultChecked />}
                        label="Auto-process outputs"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={<Switch />}
                        label="Save as template"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2" gutterBottom>
                        Quality Level
                      </Typography>
                      <Slider
                        defaultValue={70}
                        marks={[
                          { value: 0, label: 'Fast' },
                          { value: 50, label: 'Balanced' },
                          { value: 100, label: 'High Quality' },
                        ]}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setToolDialogOpen(false)}>
                Close
              </Button>
              <Button variant="contained" startIcon={<PlayArrow />}>
                Start Job
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* New Job Dialog */}
      <Dialog 
        open={newJobDialogOpen} 
        onClose={() => setNewJobDialogOpen(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>Create New AI Job</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>AI Tool</InputLabel>
                <Select
                  value={jobSettings.tool}
                  onChange={(e) => setJobSettings(prev => ({ ...prev, tool: e.target.value }))}
                  label="AI Tool"
                >
                  {aiToolsData.map((tool) => (
                    <MenuItem key={tool.id} value={tool.name}>
                      {tool.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Quality</InputLabel>
                <Select
                  value={jobSettings.quality}
                  onChange={(e) => setJobSettings(prev => ({ ...prev, quality: e.target.value }))}
                  label="Quality"
                >
                  <MenuItem value="draft">Draft (Fast)</MenuItem>
                  <MenuItem value="standard">Standard</MenuItem>
                  <MenuItem value="high">High Quality</MenuItem>
                  <MenuItem value="premium">Premium</MenuItem>
                </Select>
              </FormControl>
            </Grid>

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
                onChange={(e, value) => setJobSettings(prev => ({ ...prev, creativity: value as number }))}
                min={0}
                max={100}
                marks={[
                  { value: 0, label: 'Conservative' },
                  { value: 50, label: 'Balanced' },
                  { value: 100, label: 'Creative' },
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