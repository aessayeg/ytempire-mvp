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
  Alert,
  Chip,
  IconButton,
  Autocomplete,
  FormControlLabel,
  Switch,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  RadioGroup,
  Radio,
  LinearProgress,
  CircularProgress
} from '@mui/material';
import {
  AutoAwesome,
  SmartToy,
  Psychology,
  RecordVoiceOver,
  Image,
  Speed,
  Balance,
  HighQuality
} from '@mui/icons-material';

interface VideoGenerationConfig {
  channelId: string;
  topic: string;
  title: string;
  style: string;
  duration: string;
  targetAudience: string;
  tone: string;
  keywords: string[];
  voiceStyle: string;
  language: string;
  musicStyle: string;
  thumbnailStyle: string;
  autoPublish: boolean;
  scheduledTime: string;
  qualityPreset: string;
}

interface VideoGenerationFormProps {
  channelId?: string;
  onGenerate?: (config: VideoGenerationConfig) => void;
}

const steps = [
  'Topic & Title',
  'Content Style',
  'Voice & Audio',
  'Visuals',
  'Publishing',
  'Review & Generate'
];

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};

export const VideoGenerationForm: React.FC<VideoGenerationFormProps> = ({ 
  channelId = '', 
  onGenerate 
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [config, setConfig] = useState<VideoGenerationConfig>({
    channelId,
    topic: '',
    title: '',
    style: 'informative',
    duration: 'medium',
    targetAudience: 'general',
    tone: 'professional',
    keywords: [],
    voiceStyle: 'natural',
    language: 'en',
    musicStyle: 'none',
    thumbnailStyle: 'modern',
    autoPublish: false,
    scheduledTime: '',
    qualityPreset: 'balanced'
  });
  
  const [titleSuggestions, setTitleSuggestions] = useState<string[]>([]);
  const [generating, setGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [costEstimate, setCostEstimate] = useState({
    script: 0.10,
    voice: 0.30,
    thumbnail: 0.10,
    processing: 0.05,
    total: 0.55
  });

  useEffect(() => {
    // Calculate cost based on configuration
    const calculateCost = () => {
      let script = 0.10;
      let voice = 0.30;
      let thumbnail = 0.10;
      const processing = 0.05;
      
      // Adjust based on duration
      if (config.duration === 'short') {
        script *= 0.5;
        voice *= 0.5;
      } else if (config.duration === 'long') {
        script *= 2;
        voice *= 2;
      }
      
      // Adjust based on quality
      if (config.qualityPreset === 'quality') {
        script *= 1.5;
        voice *= 1.5;
        thumbnail *= 1.5;
      } else if (config.qualityPreset === 'fast') {
        script *= 0.7;
        voice *= 0.7;
        thumbnail *= 0.7;
      }
      
      setCostEstimate({
        script,
        voice,
        thumbnail,
        processing,
        total: script + voice + thumbnail + processing
      });
    };
    
    calculateCost();
  }, [config]);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate generation progress
      const interval = setInterval(() => {
        setGenerationProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 10;
        });
      }, 1000);
      
      if (onGenerate) {
        await onGenerate(config);
      }
      
      setSuccess('Video generation started successfully!');
    } catch (err: any) {
      setError(err.message || 'Failed to generate video');
    } finally {
      setGenerating(false);
    }
  };

  const generateTitleSuggestions = () => {
    // Generate AI-powered title suggestions based on topic
    if (config.topic) {
      setTitleSuggestions([
        `The Ultimate Guide to ${config.topic}`,
        `${config.topic}: Everything You Need to Know`,
        `Why ${config.topic} Matters in 2024`,
        `${config.topic} Explained in Simple Terms`,
        `Top 10 Facts About ${config.topic}`
      ]);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0: // Topic & Title
        return (
          <Box sx={{ mt: 2 }}>
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
                    )
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
                  <Chip label={option} {...getTagProps({ index })} key={index} />
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
                          background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)'
                        }
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
                          background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)'
                        }
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