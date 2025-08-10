import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Slider,
  RadioGroup,
  Radio,
  Autocomplete,
  IconButton,
  Tooltip,
  LinearProgress,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  SmartToy,
  TrendingUp,
  Psychology,
  RecordVoiceOver,
  Image,
  Schedule,
  MonetizationOn,
  Info,
  CheckCircle,
  Warning,
  Speed,
  HighQuality,
  Balance,
  AutoAwesome,
  Refresh,
  Preview,
  Help,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { videoApi, aiApi } from '../../services/api';
import { useChannelStore } from '../../stores/channelStore';
import { formatCurrency } from '../../utils/formatters';

interface GenerationConfig {
  channelId: string;
  title: string;
  topic: string;
  style: string;
  duration: string;
  voiceStyle: string;
  language: string;
  useTrending: boolean;
  qualityPreset: string;
  targetAudience: string;
  tone: string;
  keywords: string[];
  thumbnailStyle: string;
  musicStyle: string;
  autoPublish: boolean;
  scheduledTime?: string;
}

interface CostEstimate {
  script: number;
  voice: number;
  thumbnail: number;
  processing: number;
  total: number;
}

interface TrendingSuggestion {
  topic: string;
  score: number;
  keywords: string[];
  competitionLevel: 'low' | 'medium' | 'high';
}

const steps = [
  'Channel & Topic',
  'Content Style',
  'Voice & Audio',
  'Visuals',
  'Publishing',
  'Review & Generate'
];

export const VideoGenerationForm: React.FC = () => {
  const navigate = useNavigate();
  const { channels, fetchChannels } = useChannelStore();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [config, setConfig] = useState<GenerationConfig>({
    channelId: '',
    title: '',
    topic: '',
    style: 'informative',
    duration: 'short',
    voiceStyle: 'natural',
    language: 'en',
    useTrending: true,
    qualityPreset: 'balanced',
    targetAudience: 'general',
    tone: 'professional',
    keywords: [],
    thumbnailStyle: 'modern',
    musicStyle: 'none',
    autoPublish: false,
    scheduledTime: undefined,
  });

  const [costEstimate, setCostEstimate] = useState<CostEstimate>({
    script: 0,
    voice: 0,
    thumbnail: 0,
    processing: 0,
    total: 0,
  });

  const [trendingSuggestions, setTrendingSuggestions] = useState<TrendingSuggestion[]>([]);
  const [selectedTrending, setSelectedTrending] = useState<TrendingSuggestion | null>(null);
  const [titleSuggestions, setTitleSuggestions] = useState<string[]>([]);

  useEffect(() => {
    fetchChannels();
  }, []);

  useEffect(() => {
    // Update cost estimate when config changes
    updateCostEstimate();
  }, [config]);

  const updateCostEstimate = () => {
    const costs = {
      script: config.qualityPreset === 'quality' ? 0.50 : config.qualityPreset === 'fast' ? 0.20 : 0.35,
      voice: config.duration === 'long' ? 0.80 : config.duration === 'medium' ? 0.50 : 0.30,
      thumbnail: config.thumbnailStyle === 'custom' ? 0.10 : 0.05,
      processing: 0.10,
      total: 0,
    };
    costs.total = costs.script + costs.voice + costs.thumbnail + costs.processing;
    setCostEstimate(costs);
  };

  const fetchTrendingSuggestions = async () => {
    if (!config.channelId) return;
    
    try {
      setLoading(true);
      const channel = channels.find(c => c.id === config.channelId);
      if (!channel) return;
      
      const response = await aiApi.getTrendingTopics(channel.category);
      setTrendingSuggestions(response.data);
    } catch (error) {
      console.error('Error fetching trending topics:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateTitleSuggestions = async () => {
    if (!config.topic) return;
    
    try {
      const response = await aiApi.generateTitles(config.topic, config.style);
      setTitleSuggestions(response.data);
    } catch (error) {
      console.error('Error generating titles:', error);
    }
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    setGenerationProgress(0);
    
    try {
      // Start generation
      const response = await videoApi.generateVideo(config);
      const videoId = response.data.id;
      
      // Simulate progress updates (in real app, use WebSocket)
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => {
          if (prev >= 100) {
            clearInterval(progressInterval);
            return 100;
          }
          return prev + 10;
        });
      }, 2000);
      
      // Poll for status
      const checkStatus = async () => {
        const statusResponse = await videoApi.getGenerationStatus(videoId);
        if (statusResponse.data.status === 'completed') {
          clearInterval(progressInterval);
          setGenerationProgress(100);
          setSuccess('Video generated successfully!');
          setTimeout(() => {
            navigate(`/videos/${videoId}`);
          }, 2000);
        } else if (statusResponse.data.status === 'failed') {
          clearInterval(progressInterval);
          setError('Video generation failed. Please try again.');
          setGenerating(false);
        } else {
          setTimeout(checkStatus, 5000);
        }
      };
      
      setTimeout(checkStatus, 5000);
      
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Failed to generate video');
      setGenerating(false);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0: // Channel & Topic
        return (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Select Channel</InputLabel>
              <Select
                value={config.channelId}
                label="Select Channel"
                onChange={(e) => setConfig({ ...config, channelId: e.target.value })}
              >
                {channels.map((channel) => (
                  <MenuItem key={channel.id} value={channel.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {channel.name}
                      {channel.isVerified && (
                        <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                      )}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Switch
                  checked={config.useTrending}
                  onChange={(e) => setConfig({ ...config, useTrending: e.target.checked })}
                />
              }
              label="Use trending topics"
              sx={{ mb: 2 }}
            />

            {config.useTrending && (
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="subtitle2">Trending Topics</Typography>
                  <Button
                    size="small"
                    startIcon={<Refresh />}
                    onClick={fetchTrendingSuggestions}
                    disabled={!config.channelId || loading}
                  >
                    Refresh
                  </Button>
                </Box>
                {loading ? (
                  <CircularProgress size={24} />
                ) : (
                  <Grid container spacing={1}>
                    {trendingSuggestions.map((suggestion, index) => (
                      <Grid item xs={12} sm={6} key={index}>
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: selectedTrending?.topic === suggestion.topic ? 2 : 1,
                            borderColor: selectedTrending?.topic === suggestion.topic ? 'primary.main' : 'divider',
                          }}
                          onClick={() => {
                            setSelectedTrending(suggestion);
                            setConfig({ ...config, topic: suggestion.topic });
                          }}
                        >
                          <CardContent sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Typography variant="body2" fontWeight="bold">
                                {suggestion.topic}
                              </Typography>
                              <Chip
                                label={`${suggestion.score}%`}
                                size="small"
                                color="primary"
                              />
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              <Chip
                                label={suggestion.competitionLevel}
                                size="small"
                                variant="outlined"
                                color={
                                  suggestion.competitionLevel === 'low' ? 'success' :
                                  suggestion.competitionLevel === 'medium' ? 'warning' : 'error'
                                }
                              />
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Box>
            )}

            <TextField
              fullWidth
              label="Topic"
              value={config.topic}
              onChange={(e) => setConfig({ ...config, topic: e.target.value })}
              placeholder="Enter your video topic or select from trending"
              sx={{ mb: 3 }}
              multiline
              rows={2}
            />

            <Autocomplete
              freeSolo
              options={titleSuggestions}
              value={config.title}
              onChange={(e, value) => setConfig({ ...config, title: value || '' })}
              onInputChange={(e, value) => setConfig({ ...config, title: value })}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Video Title"
                  placeholder="Enter title or use AI suggestions"
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {params.InputProps.endAdornment}
                        <IconButton
                          size="small"
                          onClick={generateTitleSuggestions}
                          disabled={!config.topic}
                        >
                          <AutoAwesome />
                        </IconButton>
                      </>
                    ),
                  }}
                />
              )}
            />
          </Box>
        );

      case 1: // Content Style
        return (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Content Style</InputLabel>
              <Select
                value={config.style}
                label="Content Style"
                onChange={(e) => setConfig({ ...config, style: e.target.value })}
              >
                <MenuItem value="informative">Informative</MenuItem>
                <MenuItem value="entertaining">Entertaining</MenuItem>
                <MenuItem value="tutorial">Tutorial</MenuItem>
                <MenuItem value="review">Review</MenuItem>
                <MenuItem value="news">News</MenuItem>
                <MenuItem value="story">Story</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Video Duration</InputLabel>
              <Select
                value={config.duration}
                label="Video Duration"
                onChange={(e) => setConfig({ ...config, duration: e.target.value })}
              >
                <MenuItem value="short">Short (1-3 min)</MenuItem>
                <MenuItem value="medium">Medium (5-10 min)</MenuItem>
                <MenuItem value="long">Long (10+ min)</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Target Audience</InputLabel>
              <Select
                value={config.targetAudience}
                label="Target Audience"
                onChange={(e) => setConfig({ ...config, targetAudience: e.target.value })}
              >
                <MenuItem value="general">General</MenuItem>
                <MenuItem value="kids">Kids</MenuItem>
                <MenuItem value="teens">Teens</MenuItem>
                <MenuItem value="adults">Adults</MenuItem>
                <MenuItem value="professionals">Professionals</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Tone</InputLabel>
              <Select
                value={config.tone}
                label="Tone"
                onChange={(e) => setConfig({ ...config, tone: e.target.value })}
              >
                <MenuItem value="professional">Professional</MenuItem>
                <MenuItem value="casual">Casual</MenuItem>
                <MenuItem value="humorous">Humorous</MenuItem>
                <MenuItem value="serious">Serious</MenuItem>
                <MenuItem value="inspirational">Inspirational</MenuItem>
              </Select>
            </FormControl>

            <Autocomplete
              multiple
              freeSolo
              options={[]}
              value={config.keywords}
              onChange={(e, value) => setConfig({ ...config, keywords: value })}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Keywords"
                  placeholder="Add keywords for SEO"
                  helperText="Press Enter to add keywords"
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip label={option} {...getTagProps({ index })} />
                ))
              }
            />
          </Box>
        );

      case 2: // Voice & Audio
        return (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Voice Style</InputLabel>
              <Select
                value={config.voiceStyle}
                label="Voice Style"
                onChange={(e) => setConfig({ ...config, voiceStyle: e.target.value })}
              >
                <MenuItem value="natural">Natural</MenuItem>
                <MenuItem value="energetic">Energetic</MenuItem>
                <MenuItem value="calm">Calm</MenuItem>
                <MenuItem value="professional">Professional</MenuItem>
                <MenuItem value="friendly">Friendly</MenuItem>
                <MenuItem value="authoritative">Authoritative</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Language</InputLabel>
              <Select
                value={config.language}
                label="Language"
                onChange={(e) => setConfig({ ...config, language: e.target.value })}
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="es">Spanish</MenuItem>
                <MenuItem value="fr">French</MenuItem>
                <MenuItem value="de">German</MenuItem>
                <MenuItem value="it">Italian</MenuItem>
                <MenuItem value="pt">Portuguese</MenuItem>
                <MenuItem value="ja">Japanese</MenuItem>
                <MenuItem value="ko">Korean</MenuItem>
                <MenuItem value="zh">Chinese</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Background Music</InputLabel>
              <Select
                value={config.musicStyle}
                label="Background Music"
                onChange={(e) => setConfig({ ...config, musicStyle: e.target.value })}
              >
                <MenuItem value="none">None</MenuItem>
                <MenuItem value="upbeat">Upbeat</MenuItem>
                <MenuItem value="calm">Calm</MenuItem>
                <MenuItem value="dramatic">Dramatic</MenuItem>
                <MenuItem value="corporate">Corporate</MenuItem>
                <MenuItem value="cinematic">Cinematic</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                Voice synthesis will use ElevenLabs for natural voices or Google TTS as fallback.
                Music will be royalty-free from our library.
              </Typography>
            </Alert>
          </Box>
        );

      case 3: // Visuals
        return (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Thumbnail Style</InputLabel>
              <Select
                value={config.thumbnailStyle}
                label="Thumbnail Style"
                onChange={(e) => setConfig({ ...config, thumbnailStyle: e.target.value })}
              >
                <MenuItem value="modern">Modern</MenuItem>
                <MenuItem value="minimalist">Minimalist</MenuItem>
                <MenuItem value="bold">Bold</MenuItem>
                <MenuItem value="custom">Custom (AI Generated)</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info">
              <Typography variant="body2">
                Thumbnails will be automatically generated using DALL-E 3 based on your video content.
                Visual elements will be optimized for maximum click-through rate.
              </Typography>
            </Alert>
          </Box>
        );

      case 4: // Publishing
        return (
          <Box sx={{ mt: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.autoPublish}
                  onChange={(e) => setConfig({ ...config, autoPublish: e.target.checked })}
                />
              }
              label="Auto-publish after generation"
              sx={{ mb: 3 }}
            />

            {config.autoPublish && (
              <TextField
                fullWidth
                label="Schedule Publishing Time"
                type="datetime-local"
                value={config.scheduledTime || ''}
                onChange={(e) => setConfig({ ...config, scheduledTime: e.target.value })}
                InputLabelProps={{ shrink: true }}
                helperText="Leave empty to publish immediately"
                sx={{ mb: 3 }}
              />
            )}

            <Alert severity="warning">
              <Typography variant="body2">
                Make sure your YouTube channel is connected and verified before enabling auto-publish.
              </Typography>
            </Alert>
          </Box>
        );

      case 5: // Review & Generate
        return (
          <Box sx={{ mt: 2 }}>
            <Paper sx={{ p: 3, mb: 3, bgcolor: 'grey.50' }}>
              <Typography variant="h6" gutterBottom>
                Generation Summary
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemIcon><SmartToy /></ListItemIcon>
                  <ListItemText primary="Topic" secondary={config.topic} />
                </ListItem>
                <ListItem>
                  <ListItemIcon><Psychology /></ListItemIcon>
                  <ListItemText primary="Style" secondary={`${config.style} - ${config.duration}`} />
                </ListItem>
                <ListItem>
                  <ListItemIcon><RecordVoiceOver /></ListItemIcon>
                  <ListItemText primary="Voice" secondary={`${config.voiceStyle} (${config.language})`} />
                </ListItem>
                <ListItem>
                  <ListItemIcon><Image /></ListItemIcon>
                  <ListItemText primary="Thumbnail" secondary={config.thumbnailStyle} />
                </ListItem>
              </List>
            </Paper>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Cost Estimate
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Script Generation</Typography>
                  <Typography variant="body2">{formatCurrency(costEstimate.script)}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Voice Synthesis</Typography>
                  <Typography variant="body2">{formatCurrency(costEstimate.voice)}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Thumbnail</Typography>
                  <Typography variant="body2">{formatCurrency(costEstimate.thumbnail)}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Processing</Typography>
                  <Typography variant="body2">{formatCurrency(costEstimate.processing)}</Typography>
                </Box>
                <Box sx={{ borderTop: 1, borderColor: 'divider', pt: 1, mt: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="subtitle1" fontWeight="bold">Total Cost</Typography>
                    <Typography variant="subtitle1" fontWeight="bold" color="primary">
                      {formatCurrency(costEstimate.total)}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Paper>

            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Quality Preset
              </Typography>
              <RadioGroup
                row
                value={config.qualityPreset}
                onChange={(e) => setConfig({ ...config, qualityPreset: e.target.value })}
              >
                <FormControlLabel
                  value="fast"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Speed />
                      Fast
                    </Box>
                  }
                />
                <FormControlLabel
                  value="balanced"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Balance />
                      Balanced
                    </Box>
                  }
                />
                <FormControlLabel
                  value="quality"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <HighQuality />
                      Quality
                    </Box>
                  }
                />
              </RadioGroup>
            </FormControl>

            {generating && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Generating video... {generationProgress}%
                </Typography>
                <LinearProgress variant="determinate" value={generationProgress} />
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {success && (
              <Alert severity="success" sx={{ mb: 2 }}>
                {success}
              </Alert>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          Generate New Video
        </Typography>
        
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((label, index) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
              <StepContent>
                {renderStepContent(index)}
                
                <Box sx={{ mt: 3 }}>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    sx={{ mr: 1 }}
                  >
                    Back
                  </Button>
                  {index === steps.length - 1 ? (
                    <Button
                      variant="contained"
                      onClick={handleGenerate}
                      disabled={generating || !config.channelId || !config.topic}
                      sx={{
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                        },
                      }}
                    >
                      {generating ? (
                        <>
                          <CircularProgress size={20} sx={{ mr: 1 }} />
                          Generating...
                        </>
                      ) : (
                        'Generate Video'
                      )}
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      sx={{
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                        },
                      }}
                    >
                      Next
                    </Button>
                  )}
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </Paper>
    </Box>
  );
};