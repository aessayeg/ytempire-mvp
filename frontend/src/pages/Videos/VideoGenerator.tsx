import React, { useState, useEffect } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  Card,
  CardContent,
  Grid,
  Slider,
  Switch,
  FormControlLabel,
  RadioGroup,
  Radio,
  Autocomplete,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  VideoLibrary,
  Settings,
  Preview,
  Publish,
  AttachMoney,
  TrendingUp,
  Speed,
  ExpandMore,
  ExpandLess,
  Info,
  AutoAwesome,
  Psychology,
  RecordVoiceOver,
  Image,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useChannelStore } from '../../stores/channelStore';
import { api } from '../../services/api';
import { GenerationProgress } from '../../components/Videos/GenerationProgress';

const steps = ['Select Channel & Topic', 'Configure Settings', 'Review & Generate'];

interface GenerationConfig {
  channel_id: string;
  title: string;
  topic: string;
  style: string;
  duration: string;
  voice_style: string;
  language: string;
  use_trending: boolean;
  quality_preset: string;
  thumbnail_style: string;
  music_style: string;
  target_audience: string;
  keywords: string[];
  tone: string;
  pacing: string;
}

export const VideoGenerator: React.FC = () => {
  const navigate = useNavigate();
  const { channels, fetchChannels } = useChannelStore();
  const [activeStep, setActiveStep] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [estimatedCost, setEstimatedCost] = useState(0);
  const [trendingTopics, setTrendingTopics] = useState<any[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const [config, setConfig] = useState<GenerationConfig>({
    channel_id: '',
    title: '',
    topic: '',
    style: 'informative',
    duration: 'short',
    voice_style: 'natural',
    language: 'en',
    use_trending: true,
    quality_preset: 'balanced',
    thumbnail_style: 'modern',
    music_style: 'none',
    target_audience: 'general',
    keywords: [],
    tone: 'professional',
    pacing: 'medium',
  });

  useEffect(() => {
    fetchChannels();
    fetchTrendingTopics();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  useEffect(() => {
    calculateEstimatedCost();
  }, [config]);

  const fetchTrendingTopics = async () => {
    try {
      const response = await api.ai.getTrendingTopics();
      setTrendingTopics(response.topics);
    } catch (_error) {
      console.error('Failed to fetch trending topics:', error);
    }
  };

  const calculateEstimatedCost = () => {
    let cost = 0.1; // Base cost
    
    // Duration cost
    if (config.duration === 'short') cost += 0.5;
    else if (config.duration === 'medium') cost += 1.0;
    else if (config.duration === 'long') cost += 2.0;
    
    // Quality cost
    if (config.quality_preset === 'fast') cost += 0.2;
    else if (config.quality_preset === 'balanced') cost += 0.5;
    else if (config.quality_preset === 'quality') cost += 1.0;
    
    // Voice cost
    if (config.voice_style !== 'none') cost += 0.3;
    
    // Thumbnail cost
    if (config.thumbnail_style !== 'none') cost += 0.1;
    
    setEstimatedCost(cost);
  };

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      handleGenerate();
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    
    try {
      const response = await api.videos.generate(config);
      setGenerationId(response.id);
      // Navigate to generation progress
      navigate(`/videos/generation/${response.id}`);
    } catch (err: unknown) {
      setError(err.message || 'Failed to start video generation');
      setIsGenerating(false);
    }
  };

  const handleUseTrendingTopic = (topic: unknown) => {
    setConfig({
      ...config,
      topic: topic.title,
      keywords: topic.keywords || [],
      use_trending: true,
    });
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <VideoLibrary /> Select Channel
                  </Typography>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Channel</InputLabel>
                    <Select
                      value={config.channel_id}
                      onChange={(e) => setConfig({ ...config, channel_id: e.target.value })}
                      label="Channel"
                    >
                      {channels.map((channel) => (
                        <MenuItem key={channel.id} value={channel.id}>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography>{channel.name}</Typography>
                            <Chip
                              label={channel.category}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <TextField
                    fullWidth
                    label="Video Title (Optional)"
                    value={config.title}
                    onChange={(e) => setConfig({ ...config, title: e.target.value })}
                    margin="normal"
                    helperText="Leave empty to auto-generate based on topic"
                  />

                  <TextField
                    fullWidth
                    label="Topic"
                    value={config.topic}
                    onChange={(e) => setConfig({ ...config, topic: e.target.value })}
                    margin="normal"
                    required
                    multiline
                    rows={2}
                    helperText="What should the video be about?"
                  />

                  <Autocomplete
                    multiple
                    options={[]}
                    freeSolo
                    value={config.keywords}
                    onChange={(_, newValue) => setConfig({ ...config, keywords: newValue })}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Keywords"
                        margin="normal"
                        helperText="Press Enter to add keywords"
                      />
                    )}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip
                          variant="outlined"
                          label={option}
                          size="small"
                          {...getTagProps({ index })}
                        />
                      ))
                    }
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Typography variant="h6">
                      <TrendingUp /> Trending Topics
                    </Typography>
                    <IconButton onClick={fetchTrendingTopics} size="small">
                      <AutoAwesome />
                    </IconButton>
                  </Box>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.use_trending}
                        onChange={(e) => setConfig({ ...config, use_trending: e.target.checked })}
                      />
                    }
                    label="Use trending topics"
                  />

                  <List dense>
                    {trendingTopics.slice(0, 5).map((topic, index) => (
                      <ListItem
                        key={index}
                        button
                        onClick={() => handleUseTrendingTopic(topic)}
                        sx={{
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                          mb: 1,
                          '&:hover': { bgcolor: 'action.hover' },
                        }}
                      >
                        <ListItemIcon>
                          <TrendingUp color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary={topic.title}
                          secondary={
                            <Box>
                              <Typography variant="caption">
                                Score: {topic.score}% • Views: {topic.potential_views}
                              </Typography>
                              <Box display="flex" gap={0.5} mt={0.5}>
                                {topic.keywords?.slice(0, 3).map((keyword: string, i: number) => (
                                  <Chip key={i} label={keyword} size="small" />
                                ))}
                              </Box>
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Settings /> Video Settings
                  </Typography>

                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Style</InputLabel>
                        <Select
                          value={config.style}
                          onChange={(e) => setConfig({ ...config, style: e.target.value })}
                          label="Style"
                        >
                          <MenuItem value="informative">Informative</MenuItem>
                          <MenuItem value="entertaining">Entertaining</MenuItem>
                          <MenuItem value="tutorial">Tutorial</MenuItem>
                          <MenuItem value="review">Review</MenuItem>
                          <MenuItem value="news">News</MenuItem>
                          <MenuItem value="story">Story</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Duration</InputLabel>
                        <Select
                          value={config.duration}
                          onChange={(e) => setConfig({ ...config, duration: e.target.value })}
                          label="Duration"
                        >
                          <MenuItem value="short">Short (1-3 min)</MenuItem>
                          <MenuItem value="medium">Medium (5-10 min)</MenuItem>
                          <MenuItem value="long">Long (10+ min)</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Voice Style</InputLabel>
                        <Select
                          value={config.voice_style}
                          onChange={(e) => setConfig({ ...config, voice_style: e.target.value })}
                          label="Voice Style"
                        >
                          <MenuItem value="natural">Natural</MenuItem>
                          <MenuItem value="energetic">Energetic</MenuItem>
                          <MenuItem value="calm">Calm</MenuItem>
                          <MenuItem value="professional">Professional</MenuItem>
                          <MenuItem value="conversational">Conversational</MenuItem>
                          <MenuItem value="none">No Voice</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Quality Preset</InputLabel>
                        <Select
                          value={config.quality_preset}
                          onChange={(e) => setConfig({ ...config, quality_preset: e.target.value })}
                          label="Quality Preset"
                        >
                          <MenuItem value="fast">Fast (Lower Quality)</MenuItem>
                          <MenuItem value="balanced">Balanced</MenuItem>
                          <MenuItem value="quality">High Quality</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Thumbnail Style</InputLabel>
                        <Select
                          value={config.thumbnail_style}
                          onChange={(e) => setConfig({ ...config, thumbnail_style: e.target.value })}
                          label="Thumbnail Style"
                        >
                          <MenuItem value="modern">Modern</MenuItem>
                          <MenuItem value="minimalist">Minimalist</MenuItem>
                          <MenuItem value="bold">Bold</MenuItem>
                          <MenuItem value="professional">Professional</MenuItem>
                          <MenuItem value="custom">Custom</MenuItem>
                          <MenuItem value="none">No Thumbnail</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>Target Audience</InputLabel>
                        <Select
                          value={config.target_audience}
                          onChange={(e) => setConfig({ ...config, target_audience: e.target.value })}
                          label="Target Audience"
                        >
                          <MenuItem value="general">General</MenuItem>
                          <MenuItem value="young">Young Adults (18-24)</MenuItem>
                          <MenuItem value="adults">Adults (25-44)</MenuItem>
                          <MenuItem value="professionals">Professionals</MenuItem>
                          <MenuItem value="students">Students</MenuItem>
                          <MenuItem value="tech">Tech Enthusiasts</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>

                  <Box mt={2}>
                    <Button
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      endIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
                    >
                      Advanced Settings
                    </Button>
                    <Collapse in={showAdvanced}>
                      <Grid container spacing={2} mt={1}>
                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth margin="normal">
                            <InputLabel>Tone</InputLabel>
                            <Select
                              value={config.tone}
                              onChange={(e) => setConfig({ ...config, tone: e.target.value })}
                              label="Tone"
                            >
                              <MenuItem value="professional">Professional</MenuItem>
                              <MenuItem value="casual">Casual</MenuItem>
                              <MenuItem value="humorous">Humorous</MenuItem>
                              <MenuItem value="serious">Serious</MenuItem>
                              <MenuItem value="inspirational">Inspirational</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth margin="normal">
                            <InputLabel>Pacing</InputLabel>
                            <Select
                              value={config.pacing}
                              onChange={(e) => setConfig({ ...config, pacing: e.target.value })}
                              label="Pacing"
                            >
                              <MenuItem value="slow">Slow</MenuItem>
                              <MenuItem value="medium">Medium</MenuItem>
                              <MenuItem value="fast">Fast</MenuItem>
                              <MenuItem value="dynamic">Dynamic</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth margin="normal">
                            <InputLabel>Music Style</InputLabel>
                            <Select
                              value={config.music_style}
                              onChange={(e) => setConfig({ ...config, music_style: e.target.value })}
                              label="Music Style"
                            >
                              <MenuItem value="none">No Music</MenuItem>
                              <MenuItem value="upbeat">Upbeat</MenuItem>
                              <MenuItem value="calm">Calm</MenuItem>
                              <MenuItem value="corporate">Corporate</MenuItem>
                              <MenuItem value="cinematic">Cinematic</MenuItem>
                              <MenuItem value="electronic">Electronic</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth margin="normal">
                            <InputLabel>Language</InputLabel>
                            <Select
                              value={config.language}
                              onChange={(e) => setConfig({ ...config, language: e.target.value })}
                              label="Language"
                            >
                              <MenuItem value="en">English</MenuItem>
                              <MenuItem value="es">Spanish</MenuItem>
                              <MenuItem value="fr">French</MenuItem>
                              <MenuItem value="de">German</MenuItem>
                              <MenuItem value="ja">Japanese</MenuItem>
                              <MenuItem value="zh">Chinese</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                      </Grid>
                    </Collapse>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Info /> Generation Info
                  </Typography>
                  
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <AttachMoney />
                      </ListItemIcon>
                      <ListItemText
                        primary="Estimated Cost"
                        secondary={`$${estimatedCost.toFixed(2)}`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Speed />
                      </ListItemIcon>
                      <ListItemText
                        primary="Generation Time"
                        secondary={
                          config.quality_preset === 'fast'
                            ? '3-5 minutes'
                            : config.quality_preset === 'balanced'
                            ? '5-10 minutes'
                            : '10-15 minutes'
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Psychology />
                      </ListItemIcon>
                      <ListItemText
                        primary="AI Model"
                        secondary={config.quality_preset === 'quality' ? 'GPT-4' : 'GPT-3.5'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <RecordVoiceOver />
                      </ListItemIcon>
                      <ListItemText
                        primary="Voice Synthesis"
                        secondary={config.voice_style !== 'none' ? 'ElevenLabs' : 'Disabled'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Image />
                      </ListItemIcon>
                      <ListItemText
                        primary="Thumbnail"
                        secondary={config.thumbnail_style !== 'none' ? 'DALL-E 3' : 'Disabled'}
                      />
                    </ListItem>
                  </List>

                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="caption">
                      Higher quality settings will increase generation time and cost but produce better results.
                    </Typography>
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Preview /> Review Configuration
                  </Typography>

                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Channel
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        {channels.find((c) => c.id === config.channel_id)?.name || 'Not selected'}
                      </Typography>
                    </Grid>

                    <Grid item xs={12}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Topic
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        {config.topic || 'Not specified'}
                      </Typography>
                    </Grid>

                    {config.title && (
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Title
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {config.title}
                        </Typography>
                      </Grid>
                    )}

                    <Grid item xs={12}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Keywords
                      </Typography>
                      <Box display="flex" gap={1} flexWrap="wrap" mt={1}>
                        {config.keywords.length > 0 ? (
                          config.keywords.map((keyword, index) => (
                            <Chip key={index} label={keyword} size="small" />
                          ))
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No keywords specified
                          </Typography>
                        )}
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Style
                      </Typography>
                      <Typography variant="body1">{config.style}</Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Duration
                      </Typography>
                      <Typography variant="body1">{config.duration}</Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Voice Style
                      </Typography>
                      <Typography variant="body1">{config.voice_style}</Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Quality
                      </Typography>
                      <Typography variant="body1">{config.quality_preset}</Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Target Audience
                      </Typography>
                      <Typography variant="body1">{config.target_audience}</Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Language
                      </Typography>
                      <Typography variant="body1">{config.language.toUpperCase()}</Typography>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 3 }} />

                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="h6">
                        Total Estimated Cost: ${estimatedCost.toFixed(2)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Estimated time: {
                          config.quality_preset === 'fast'
                            ? '3-5 minutes'
                            : config.quality_preset === 'balanced'
                            ? '5-10 minutes'
                            : '10-15 minutes'
                        }
                      </Typography>
                    </Box>
                    <Chip
                      label={config.use_trending ? 'Using Trending Topics' : 'Custom Topic'}
                      color={config.use_trending ? 'success' : 'default'}
                    />
                  </Box>

                  {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {error}
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Info /> What happens next?
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="1. Script Generation"
                        secondary="AI creates an engaging script based on your topic"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="2. Voice Synthesis"
                        secondary="Convert script to natural-sounding speech"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="3. Visual Creation"
                        secondary="Generate relevant visuals and animations"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="4. Video Assembly"
                        secondary="Combine all elements into final video"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="5. Thumbnail Generation"
                        secondary="Create eye-catching thumbnail"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="6. Quality Check"
                        secondary="Automated quality assessment"
                      />
                    </ListItem>
                  </List>

                  <Alert severity="success" sx={{ mt: 2 }}>
                    You'll be notified when your video is ready!
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      default:
        return 'Unknown step';
    }
  };

  if (generationId) {
    return <GenerationProgress generationId={generationId} />;
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Generate New Video
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {renderStepContent(activeStep)}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        <Box>
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleGenerate}
              disabled={isGenerating || !config.channel_id || !config.topic}
              startIcon={isGenerating ? <CircularProgress size={20} /> : <Publish />}
            >
              {isGenerating ? 'Generating...' : 'Generate Video'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={
                (activeStep === 0 && (!config.channel_id || !config.topic))
              }
            >
              Next
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
};