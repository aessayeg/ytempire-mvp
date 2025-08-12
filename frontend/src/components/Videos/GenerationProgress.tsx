import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Divider,
} from '@mui/material';
import {
  CheckCircle,
  Error as ErrorIcon,
  HourglassEmpty,
  Psychology,
  RecordVoiceOver,
  VideoLibrary,
  Image,
  AutoAwesome,
  Speed,
  AttachMoney,
  Timer,
  PlayArrow,
  Cancel,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import { useWebSocket } from '../../hooks/useWebSocket';

interface GenerationProgressProps {
  generationId: string;
  onComplete?: (videoId: string) => void;
  onError?: (error: string) => void;
}

interface GenerationStep {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
  startTime?: Date;
  endTime?: Date;
  cost?: number;
  details?: string;
}

export const GenerationProgress: React.FC<GenerationProgressProps> = ({
  generationId,
  onComplete,
  onError,
}) => {
  const navigate = useNavigate();
  const { subscribe, unsubscribe } = useWebSocket();
  const [overallProgress, setOverallProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [status, setStatus] = useState<'processing' | 'completed' | 'failed' | 'cancelled'>('processing');
  const [error, setError] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [totalCost, setTotalCost] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const [steps, setSteps] = useState<GenerationStep[]>([
    {
      id: 'script',
      label: 'Script Generation',
      description: 'Creating engaging script content',
      icon: <Psychology />,
      status: 'pending',
      progress: 0,
    },
    {
      id: 'voice',
      label: 'Voice Synthesis',
      description: 'Converting script to natural speech',
      icon: <RecordVoiceOver />,
      status: 'pending',
      progress: 0,
    },
    {
      id: 'visuals',
      label: 'Visual Generation',
      description: 'Creating video frames and animations',
      icon: <VideoLibrary />,
      status: 'pending',
      progress: 0,
    },
    {
      id: 'assembly',
      label: 'Video Assembly',
      description: 'Combining all elements',
      icon: <AutoAwesome />,
      status: 'pending',
      progress: 0,
    },
    {
      id: 'thumbnail',
      label: 'Thumbnail Creation',
      description: 'Generating eye-catching thumbnail',
      icon: <Image />,
      status: 'pending',
      progress: 0,
    },
    {
      id: 'quality',
      label: 'Quality Check',
      description: 'Verifying video quality',
      icon: <CheckCircle />,
      status: 'pending',
      progress: 0,
    },
  ]);

  useEffect(() => {
    // Subscribe to WebSocket updates
    const unsubscribeWs = subscribe(`video.generation.${generationId}`, handleWebSocketUpdate);
    
    // Start polling for status
    const pollInterval = setInterval(fetchGenerationStatus, 2000);
    
    // Start elapsed time counter
    const timeInterval = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    
    // Initial fetch
    fetchGenerationStatus();
    
    return () => {
      unsubscribeWs();
      clearInterval(pollInterval);
      clearInterval(timeInterval);
    };
  }, [generationId]);

  const fetchGenerationStatus = async () => {
    try {
      const response = await api.videos.getGenerationStatus(generationId);
      updateProgress(response);
    } catch (error) {
      console.error('Failed to fetch generation status:', error);
    }
  };

  const handleWebSocketUpdate = (data: any) => {
    if (data.generationId === generationId) {
      updateProgress(data);
    }
  };

  const updateProgress = (data: any) => {
    // Update overall progress
    setOverallProgress(data.progress || 0);
    
    // Update current step
    if (data.currentStep !== undefined) {
      setCurrentStep(data.currentStep);
    }
    
    // Update steps
    if (data.steps) {
      setSteps((prevSteps) =>
        prevSteps.map((step, index) => {
          const updatedStep = data.steps[index];
          if (updatedStep) {
            return {
              ...step,
              status: updatedStep.status,
              progress: updatedStep.progress,
              startTime: updatedStep.startTime,
              endTime: updatedStep.endTime,
              cost: updatedStep.cost,
              details: updatedStep.details,
            };
          }
          return step;
        })
      );
    }
    
    // Update status
    if (data.status) {
      setStatus(data.status);
      
      if (data.status === 'completed' && data.videoId) {
        setVideoId(data.videoId);
        onComplete?.(data.videoId);
      } else if (data.status === 'failed' && data.error) {
        setError(data.error);
        onError?.(data.error);
      }
    }
    
    // Update costs and time
    if (data.totalCost !== undefined) {
      setTotalCost(data.totalCost);
    }
    if (data.estimatedTime !== undefined) {
      setEstimatedTime(data.estimatedTime);
    }
  };

  const handleCancel = async () => {
    try {
      await api.videos.cancelGeneration(generationId);
      setStatus('cancelled');
      navigate('/videos');
    } catch (error) {
      console.error('Failed to cancel generation:', error);
    }
  };

  const handleViewVideo = () => {
    if (videoId) {
      navigate(`/videos/${videoId}`);
    }
  };

  const handleRetry = () => {
    // Implement retry logic
    window.location.reload();
  };

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getStepIcon = (step: GenerationStep) => {
    if (step.status === 'completed') {
      return <CheckCircle color="success" />;
    } else if (step.status === 'error') {
      return <ErrorIcon color="error" />;
    } else if (step.status === 'processing') {
      return <CircularProgress size={24} />;
    }
    return step.icon;
  };

  return (
    <Box p={3}>
      <Paper sx={{ p: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h5">
            Video Generation Progress
          </Typography>
          <Box display="flex" gap={2}>
            {status === 'processing' && (
              <Button
                variant="outlined"
                color="error"
                onClick={handleCancel}
                startIcon={<Cancel />}
              >
                Cancel
              </Button>
            )}
            {status === 'completed' && videoId && (
              <Button
                variant="contained"
                onClick={handleViewVideo}
                startIcon={<PlayArrow />}
              >
                View Video
              </Button>
            )}
            {status === 'failed' && (
              <Button
                variant="contained"
                onClick={handleRetry}
                color="error"
              >
                Retry
              </Button>
            )}
          </Box>
        </Box>

        {/* Overall Progress */}
        <Box mb={4}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Overall Progress
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {overallProgress}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={overallProgress}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>

        {/* Status Alert */}
        {status === 'completed' && (
          <Alert severity="success" sx={{ mb: 3 }}>
            Video generation completed successfully!
          </Alert>
        )}
        {status === 'failed' && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error || 'Video generation failed. Please try again.'}
          </Alert>
        )}
        {status === 'cancelled' && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Video generation was cancelled.
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Steps Progress */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Generation Steps
                </Typography>
                <Stepper activeStep={currentStep} orientation="vertical">
                  {steps.map((step, index) => (
                    <Step key={step.id} completed={step.status === 'completed'}>
                      <StepLabel
                        icon={getStepIcon(step)}
                        error={step.status === 'error'}
                      >
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography>{step.label}</Typography>
                          {step.status === 'processing' && step.progress !== undefined && (
                            <Chip
                              label={`${step.progress}%`}
                              size="small"
                              color="primary"
                            />
                          )}
                          {step.cost !== undefined && (
                            <Chip
                              label={`$${step.cost.toFixed(2)}`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      </StepLabel>
                      <StepContent>
                        <Typography variant="body2" color="text.secondary">
                          {step.description}
                        </Typography>
                        {step.details && (
                          <Typography variant="caption" color="text.secondary">
                            {step.details}
                          </Typography>
                        )}
                        {step.status === 'processing' && step.progress !== undefined && (
                          <LinearProgress
                            variant="determinate"
                            value={step.progress}
                            sx={{ mt: 1, mb: 1 }}
                          />
                        )}
                        {step.startTime && step.endTime && (
                          <Typography variant="caption" color="text.secondary">
                            Duration: {Math.round((step.endTime.getTime() - step.startTime.getTime()) / 1000)}s
                          </Typography>
                        )}
                      </StepContent>
                    </Step>
                  ))}
                </Stepper>
              </CardContent>
            </Card>
          </Grid>

          {/* Statistics */}
          <Grid item xs={12} md={4}>
            <Grid container spacing={2}>
              {/* Time Stats */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <Timer /> Time
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <HourglassEmpty />
                        </ListItemIcon>
                        <ListItemText
                          primary="Elapsed Time"
                          secondary={formatTime(elapsedTime)}
                        />
                      </ListItem>
                      {estimatedTime && (
                        <ListItem>
                          <ListItemIcon>
                            <Speed />
                          </ListItemIcon>
                          <ListItemText
                            primary="Estimated Remaining"
                            secondary={formatTime(Math.max(0, estimatedTime - elapsedTime))}
                          />
                        </ListItem>
                      )}
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Cost Breakdown */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <AttachMoney /> Cost Breakdown
                    </Typography>
                    <List dense>
                      {steps.map((step) => (
                        step.cost !== undefined && (
                          <ListItem key={step.id}>
                            <ListItemText primary={step.label} />
                            <Typography variant="body2">
                              ${step.cost.toFixed(3)}
                            </Typography>
                          </ListItem>
                        )
                      ))}
                      <Divider />
                      <ListItem>
                        <ListItemText primary={<strong>Total</strong>} />
                        <Typography variant="h6" color="primary">
                          ${totalCost.toFixed(2)}
                        </Typography>
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Generation Info */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Generation Info
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Generation ID"
                          secondary={generationId}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Status"
                          secondary={
                            <Chip
                              label={status}
                              size="small"
                              color={
                                status === 'completed'
                                  ? 'success'
                                  : status === 'failed'
                                  ? 'error'
                                  : status === 'cancelled'
                                  ? 'warning'
                                  : 'primary'
                              }
                            />
                          }
                        />
                      </ListItem>
                      {videoId && (
                        <ListItem>
                          <ListItemText
                            primary="Video ID"
                            secondary={videoId}
                          />
                        </ListItem>
                      )}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};