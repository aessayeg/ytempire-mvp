import React, { useState, useEffect, useRef } from 'react';
import { 
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  CircularProgress,
  Chip,
  Alert,
  Button,
  IconButton,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Collapse,
  Avatar,
  useTheme,
  Divider,
  TextField
 } from '@mui/material';
import { 
  PlayCircle,
  Pause,
  CheckCircle,
  Error,
  Movie,
  Mic,
  Image,
  TextFields,
  Upload,
  Cancel,
  ExpandMore,
  ExpandLess,
  Speed,
  Timer,
  AttachMoney,
  Memory,
  TrendingUp,
  VideoCall
 } from '@mui/icons-material';
import {  format, formatDistanceToNow, addMinutes  } from 'date-fns';
import {  motion  } from 'framer-motion';

interface VideoGenerationTask {
  id: string,
  title: string,

  channelId: string,
  channelName: string,

  status: 'queued' | 'processing' | 'completed' | 'failed' | 'paused',
  currentStep: number,

  totalSteps: number,
  progress: number,

  startTime: Date,
  estimatedCompletion: Date,

  steps: GenerationStep[],
  metrics: {,

    costSoFar: number,
  estimatedTotalCost: number,

    processingTime: number,
  gpuUsage: number,

    memoryUsage: number};
  errors?: string[];
  warnings?: string[];
}

interface GenerationStep {
  id: string,
  name: string,

  description: string,
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped',

  progress: number;
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  cost?: number;
  output?: string;
  error?: string;
  retryCount?: number;
  maxRetries?: number;
}

const generationSteps: GenerationStep[] = [ { id: 'analyze',
    name: 'Trend Analysis',
    description: 'Analyzing trending topics and keywords',
    status: 'pending',
    progress: 0 },
  { id: 'script',
    name: 'Script Generation',
    description: 'Creating optimized script with AI',
    status: 'pending',
    progress: 0 },
  { id: 'voice',
    name: 'Voice Synthesis',
    description: 'Converting script to natural speech',
    status: 'pending',
    progress: 0 },
  { id: 'visuals',
    name: 'Visual Generation',
    description: 'Creating thumbnail and visual assets',
    status: 'pending',
    progress: 0 },
  { id: 'render',
    name: 'Video Rendering',
    description: 'Assembling final video file',
    status: 'pending',
    progress: 0 },
  { id: 'quality',
    name: 'Quality Check',
    description: 'Validating content quality and compliance',
    status: 'pending',
    progress: 0 },
  { id: 'upload',
    name: 'YouTube Upload',
    description: 'Publishing to YouTube channel',
    status: 'pending',
    progress: 0 } ];

export const LiveVideoGenerationMonitor: React.FC = () => { const [tasks, setTasks] = useState<VideoGenerationTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<VideoGenerationTask | null>(null);
  const [expandedTasks, setExpandedTasks] = useState<string[]>([]);
  const [detailsDialog, setDetailsDialog] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState({
    activeJobs: 3,
    queuedJobs: 8,
    completedToday: 45,
    failedToday: 2,
    avgProcessingTime: 8.5,
    totalCostToday: 127.50,
    gpuUtilization: 75,
    memoryUsage: 62 });
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => { // Initialize with mock data
    setTasks([ {
        id: '1',
        title: '10 AI Tools That Will Change Your Life',
        channelId: 'ch1',
        channelName: 'Tech Insights',
        status: 'processing',
        currentStep: 3,
        totalSteps: 7,
        progress: 42,
        startTime: new Date(Date.now() - 1000 * 60 * 5),
        estimatedCompletion: addMinutes(new Date(), 3),
        steps: generationSteps.map((step, index) => ({
          ...step,
          status: index < 3 ? 'completed' : index === 3 ? 'processing' : 'pending',
          progress: index < 3 ? 100 : index === 3 ? 60 : 0 })),
        metrics: { costSoFar: 0.45,
          estimatedTotalCost: 1.20,
          processingTime: 300,
          gpuUsage: 82,
          memoryUsage: 4096 }
      },
      { id: '2',
        title: 'Quantum Computing Explained Simply',
        channelId: 'ch2',
        channelName: 'Science Daily',
        status: 'processing',
        currentStep: 1,
        totalSteps: 7,
        progress: 15,
        startTime: new Date(Date.now() - 1000 * 60 * 2),
        estimatedCompletion: addMinutes(new Date(), 6),
        steps: generationSteps.map((step, index) => ({
          ...step,
          status: index === 0 ? 'processing' : 'pending',
          progress: index === 0 ? 75 : 0 })),
        metrics: { costSoFar: 0.12,
          estimatedTotalCost: 1.15,
          processingTime: 120,
          gpuUsage: 45,
          memoryUsage: 2048 }
      },
      {
        id: '3',
        title: 'Top 5 Productivity Apps 2024',
        channelId: 'ch1',
        channelName: 'Tech Insights',
        status: 'queued',
        currentStep: 0,
        totalSteps: 7,
        progress: 0,
        startTime: new Date(),
        estimatedCompletion: addMinutes(new Date(), 12),
        steps: generationSteps.map(step => ({ ...step })),
        metrics: { costSoFar: 0,
          estimatedTotalCost: 1.10,
          processingTime: 0,
          gpuUsage: 0,
          memoryUsage: 0 }
      } ]);

    // Simulate real-time updates
    intervalRef.current = setInterval(() => {
      setTasks(prev => prev.map(task => {
        if (task.status === 'processing') {
          const newProgress = Math.min(task.progress + Math.random() * 5, 100);
          const currentStepIndex = task.currentStep - 1;
          const updatedSteps = [...task.steps];
          
          if (currentStepIndex >= 0 && currentStepIndex < updatedSteps.length) {
            updatedSteps[currentStepIndex].progress = Math.min(
              updatedSteps[currentStepIndex].progress + Math.random() * 10,
              100
            );
            
            if (updatedSteps[currentStepIndex].progress >= 100) {
              updatedSteps[currentStepIndex].status = 'completed';
              if (currentStepIndex + 1 < updatedSteps.length) {
                updatedSteps[currentStepIndex + 1].status = 'processing';
              }
            }
          }

          return { ...task,
            progress: newProgress,
            currentStep: updatedSteps.filter(s => s.status === 'completed').length + 1,
            steps: updatedSteps,
            metrics: {
              ...task.metrics,
              costSoFar: task.metrics.costSoFar + Math.random() * 0.01,
              processingTime: task.metrics.processingTime + 1 },
            status: newProgress >= 100 ? 'completed' : 'processing',

          };
        }
        return task}))}, 1000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)}
    };
  }, []);

  const handleToggleExpand = (taskId: string) => {
    setExpandedTasks(prev => {}
      prev.includes(taskId)
        ? prev.filter(id => id !== taskId)
        : [...prev, taskId]
    )};

  const handlePauseTask = (taskId: string) => {
    setTasks(prev => prev.map(task => {}
      task.id === taskId ? { ...task, status: 'paused' } : task
    ))};

  const handleResumeTask = (taskId: string) => {
    setTasks(prev => prev.map(task => {}
      task.id === taskId ? { ...task, status: 'processing' } : task
    ))};

  const handleCancelTask = (taskId: string) => {
    setTasks(prev => prev.filter(task => task.id !== taskId))};

  const handleRetryStep = (taskId: string, stepId: string) => {
    // Retry logic
    console.log('Retrying step', stepId, 'for task', taskId)};

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'primary';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      case 'queued': return 'default';
      default: return 'default'}
  };

  const getStepIcon = (stepId: string) => {
    switch (stepId) {
      case 'analyze': return <TrendingUp />;
      case 'script': return <TextFields />;
      case 'voice': return <Mic />;
      case 'visuals': return <Image />;
      case 'render': return <Movie />;
      case 'quality': return <CheckCircle />;
      case 'upload': return <Upload />;
      default: return <PlayCircle />}
  };

  const renderTaskCard = (task: VideoGenerationTask) => {
    const isExpanded = expandedTasks.includes(task.id);

    return (
    <>
      <Card key={task.id} sx={{ mb: 2 }}>
        <CardContent>
          {/* Task Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
              <VideoCall />
            </Avatar>
      <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle1" fontWeight="bold">
                {task.title}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {task.channelName}
                </Typography>
                <Chip
                  label={task.status}
                  size="small"
                  color={getStatusColor(task.status) as any}
                />
                <Typography variant="caption" color="text.secondary">
                  Started {formatDistanceToNow(task.startTime, { addSuffix: true });
}
                </Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {task.status === 'processing' && (
                <IconButton size="small" onClick={() => handlePauseTask(task.id}>
                  <Pause />
                </IconButton>
              )}
              {task.status === 'paused' && (
                <IconButton size="small" onClick={() => handleResumeTask(task.id}>
                  <PlayCircle />
                </IconButton>
              )}
              <IconButton size="small" onClick={() => handleCancelTask(task.id}>
                <Cancel />
              </IconButton>
              <IconButton
                size="small"
                onClick={() => handleToggleExpand(task.id}
              >
                {isExpanded ? <ExpandLess /> </>: <ExpandMore />}
              </IconButton>
            </Box>
          </Box>

          {/* Progress Bar */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">
                Step {task.currentStep} of {task.totalSteps}
              </Typography>
              <Typography variant="body2" fontWeight="bold">
                {task.progress.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={task.progress}
              sx={{ height: 8, borderRadius: 1 }}
            />
          </Box>

          {/* Quick Metrics */}
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Timer fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Est. Time
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {format(task.estimatedCompletion, 'HH:mm')}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AttachMoney fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Cost
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    ${task.metrics.costSoFar.toFixed(2)} / ${task.metrics.estimatedTotalCost.toFixed(2)}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Speed fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    GPU Usage
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {task.metrics.gpuUsage}%
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Memory fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Memory
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {task.metrics.memoryUsage / 1024).toFixed(1} GB
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>

          {/* Detailed Steps */}
          <Collapse in={isExpanded}>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Generation Pipeline
            </Typography>
            <Stepper activeStep={task.currentStep - 1} orientation="vertical">
              {task.steps.map((step, index) => (
                <Step key={step.id} completed={step.status === 'completed'}>
                  <StepLabel
                    error={step.status === 'failed'}
                    icon={getStepIcon(step.id)}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {step.name}
                      {step.status === 'processing' && (
                        <CircularProgress size={16} />
                      )}
                      {step.status === 'completed' && (
                        <CheckCircle fontSize="small" color="success" />
                      )}
                      {step.status === 'failed' && (
                        <Error fontSize="small" color="error" />
                      )}
                    </Box>
                  </StepLabel>
                  <StepContent>
                    <Typography variant="caption" color="text.secondary">
                      {step.description}
                    </Typography>
                    {step.status === 'processing' && (
                      <Box sx={{ mt: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={step.progress}
                          sx={{ height: 4 }}
                        />
                      </Box>
                    )}
                    {step.error && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {step.error}
                        <Button
                          size="small"
                          onClick={() => handleRetryStep(task.id, step.id}
                          sx={{ ml: 1 }}
                        >
                          Retry
                        </Button>
                      </Alert>
                    )}
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </Collapse>
        </CardContent>
      </Card>
    </>
  )};

  return (
    <>
      <Box>
      {/* System Metrics */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold" color="primary">
              {systemMetrics.activeJobs}
            </Typography>
      <Typography variant="body2" color="text.secondary">
              Active Jobs
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold" color="warning.main">
              {systemMetrics.queuedJobs}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Queued
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold" color="success.main">
              {systemMetrics.completedToday}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Completed Today
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold">
              ${systemMetrics.totalCostToday.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Cost Today
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Resource Usage */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            System Resources
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">GPU Utilization</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {systemMetrics.gpuUtilization}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemMetrics.gpuUtilization}
                  color={systemMetrics.gpuUtilization > 80 ? 'warning' : 'primary'}
                />
              </Box>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Memory Usage</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {systemMetrics.memoryUsage}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={systemMetrics.memoryUsage}
                  color={systemMetrics.memoryUsage > 80 ? 'warning' : 'primary'}
                />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Active Tasks */}
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Video Generation Pipeline
      </Typography>
      {tasks.map(renderTaskCard)}
    </Box>
  </>
  )};