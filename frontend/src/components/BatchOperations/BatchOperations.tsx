import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  MenuItem,
  Alert,
  Checkbox,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Schedule as ScheduleIcon,
  VideoLibrary as VideoIcon,
  Settings as SettingsIcon,
  Queue as QueueIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';

interface BatchJob {
  id: string;
  name: string;
  type: 'video_generation' | 'thumbnail_update' | 'metadata_update' | 'analytics_sync';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  totalItems: number;
  processedItems: number;
  failedItems: number;
  startTime?: string;
  endTime?: string;
  estimatedCompletion?: string;
  channels: string[];
  priority: 'low' | 'medium' | 'high';
}

interface BatchOperationsProps {
  maxConcurrent?: number;
}

const BatchOperations: React.FC<BatchOperationsProps> = ({ 
  maxConcurrent = 10,
}) => {
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [newBatch, setNewBatch] = useState({
    name: '',
    type: 'video_generation',
    videoCount: 50,
    channels: [] as string[],
    priority: 'medium',
    schedule: 'immediate',
    options: {
      generateThumbnails: true,
      autoUpload: true,
      qualityCheck: true,
      costOptimization: true
    }
  });

  const steps = ['Select Type', 'Configure Options', 'Select Channels', 'Review & Start'];

  useEffect(() => {
    loadJobs();
    const interval = setInterval(updateJobProgress, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadJobs = () => {
    // Mock data - replace with API call
    const mockJobs: BatchJob[] = [
      {
        id: '1',
        name: 'Daily Tech Videos',
        type: 'video_generation',
        status: 'running',
        totalItems: 50,
        processedItems: 23,
        failedItems: 2,
        startTime: new Date(Date.now() - 3600000).toISOString(),
        estimatedCompletion: new Date(Date.now() + 7200000).toISOString(),
        channels: ['Tech Reviews Pro', 'Gaming Central'],
        priority: 'high'
      },
      {
        id: '2',
        name: 'Thumbnail Refresh',
        type: 'thumbnail_update',
        status: 'pending',
        totalItems: 100,
        processedItems: 0,
        failedItems: 0,
        channels: ['DIY Crafts Hub'],
        priority: 'low'
      },
      {
        id: '3',
        name: 'Weekly Gaming Content',
        type: 'video_generation',
        status: 'completed',
        totalItems: 75,
        processedItems: 75,
        failedItems: 3,
        startTime: new Date(Date.now() - 86400000).toISOString(),
        endTime: new Date(Date.now() - 3600000).toISOString(),
        channels: ['Gaming Central'],
        priority: 'medium'
      }
    ];
    setJobs(mockJobs);
  };

  const updateJobProgress = () => {
    setJobs(prevJobs => 
      prevJobs.map(job => {
        if (job.status === 'running' && job.processedItems < job.totalItems) {
          return {
            ...job,
            processedItems: Math.min(job.processedItems + Math.floor(Math.random() * 3), job.totalItems),
            status: job.processedItems + 1 >= job.totalItems ? 'completed' : 'running'
          };
        }
        return job;
      })
    );
  };

  const getStatusIcon = (status: BatchJob['status']) => {
    switch (status) {
      case 'running':
        return <CircularProgress size={20} />;
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'paused':
        return <PauseIcon color="warning" />;
      default:
        return <ScheduleIcon color="action" />;
    }
  };

  const getStatusColor = (status: BatchJob['status']) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'paused':
        return 'warning';
      default:
        return 'default';
    }
  };

  const handleStartJob = (jobId: string) => {
    setJobs(jobs.map(job => 
      job.id === jobId ? { ...job, status: 'running', startTime: new Date().toISOString() } : job
    ));
  };

  const handlePauseJob = (jobId: string) => {
    setJobs(jobs.map(job => 
      job.id === jobId ? { ...job, status: 'paused' } : job
    ));
  };

  const handleStopJob = (jobId: string) => {
    setJobs(jobs.map(job => 
      job.id === jobId ? { ...job, status: 'failed', endTime: new Date().toISOString() } : job
    ));
  };

  const handleDeleteJob = (jobId: string) => {
    if (confirm('Are you sure you want to delete this batch job?')) {
      setJobs(jobs.filter(job => job.id !== jobId));
    }
  };

  const handleCreateBatch = () => {
    const newJob: BatchJob = {
      id: Date.now().toString(),
      name: newBatch.name,
      type: newBatch.type as BatchJob['type'],
      status: 'pending',
      totalItems: newBatch.videoCount,
      processedItems: 0,
      failedItems: 0,
      channels: newBatch.channels,
      priority: newBatch.priority as BatchJob['priority']
    };
    
    setJobs([newJob, ...jobs]);
    setIsCreateDialogOpen(false);
    setActiveStep(0);
    setNewBatch({
      name: '',
      type: 'video_generation',
      videoCount: 50,
      channels: [],
      priority: 'medium',
      schedule: 'immediate',
      options: {
        generateThumbnails: true,
        autoUpload: true,
        qualityCheck: true,
        costOptimization: true
      }
    });
  };

  const runningJobs = jobs.filter(j => j.status === 'running').length;
  const totalProcessed = jobs.reduce((sum, job) => sum + job.processedItems, 0);
  const totalFailed = jobs.reduce((sum, job) => sum + job.failedItems, 0);

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Batch Operations
        </Typography>
        <Box display="flex" gap={2}>
          <Chip
            icon={<SpeedIcon />}
            label={`${runningJobs} / ${maxConcurrent} Running`}
            color={runningJobs >= maxConcurrent ? 'error' : 'primary'}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setIsCreateDialogOpen(true)}
          >
            New Batch Job
          </Button>
        </Box>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Active Jobs
                  </Typography>
                  <Typography variant="h4">
                    {runningJobs}
                  </Typography>
                </Box>
                <PlayIcon color="primary" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Videos Processed
                  </Typography>
                  <Typography variant="h4">
                    {totalProcessed}
                  </Typography>
                </Box>
                <VideoIcon color="success" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Failed Items
                  </Typography>
                  <Typography variant="h4">
                    {totalFailed}
                  </Typography>
                </Box>
                <ErrorIcon color="error" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Capacity Used
                  </Typography>
                  <Typography variant="h4">
                    {Math.round((runningJobs / maxConcurrent) * 100)}%
                  </Typography>
                </Box>
                <QueueIcon color="action" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Jobs Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  checked={selectedJobs.length === jobs.length}
                  onChange={(_e) => {
                    if (e.target.checked) {
                      setSelectedJobs(jobs.map(j => j.id));
                    } else {
                      setSelectedJobs([]);
                    }
                  }}
                />
              </TableCell>
              <TableCell>Job Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Progress</TableCell>
              <TableCell>Channels</TableCell>
              <TableCell>Priority</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.map((job) => (
              <TableRow key={job.id}>
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedJobs.includes(job.id)}
                    onChange={(_e) => {
                      if (e.target.checked) {
                        setSelectedJobs([...selectedJobs, job.id]);
                      } else {
                        setSelectedJobs(selectedJobs.filter(id => id !== job.id));
                      }
                    }}
                  />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getStatusIcon(job.status)}
                    <Typography variant="body2">{job.name}</Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip label={job.type.replace('_', ' ')} size="small" variant="outlined" />
                </TableCell>
                <TableCell>
                  <Chip 
                    label={job.status} 
                    size="small" 
                    color={getStatusColor(job.status) as any}
                  />
                </TableCell>
                <TableCell>
                  <Box>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="caption">
                        {job.processedItems} / {job.totalItems}
                      </Typography>
                      <Typography variant="caption">
                        {Math.round((job.processedItems / job.totalItems) * 100)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={(job.processedItems / job.totalItems) * 100}
                      sx={{ height: 6, borderRadius: 1 }}
                    />
                    {job.failedItems > 0 && (
                      <Typography variant="caption" color="error">
                        {job.failedItems} failed
                      </Typography>
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  <Box display="flex" flexDirection="column" gap={0.5}>
                    {job.channels.slice(0, 2).map((channel, idx) => (
                      <Chip key={idx} label={channel} size="small" />
                    ))}
                    {job.channels.length > 2 && (
                      <Typography variant="caption">
                        +{job.channels.length - 2} more
                      </Typography>
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip 
                    label={job.priority} 
                    size="small"
                    color={job.priority === 'high' ? 'error' : job.priority === 'medium' ? 'warning' : 'default'}
                  />
                </TableCell>
                <TableCell align="right">
                  <Box>
                    {job.status === 'pending' && (
                      <IconButton size="small" onClick={() => handleStartJob(job.id)}>
                        <PlayIcon />
                      </IconButton>
                    )}
                    {job.status === 'running' && (
                      <IconButton size="small" onClick={() => handlePauseJob(job.id)}>
                        <PauseIcon />
                      </IconButton>
                    )}
                    {job.status === 'paused' && (
                      <IconButton size="small" onClick={() => handleStartJob(job.id)}>
                        <PlayIcon />
                      </IconButton>
                    )}
                    {(job.status === 'running' || job.status === 'paused') && (
                      <IconButton size="small" onClick={() => handleStopJob(job.id)}>
                        <StopIcon />
                      </IconButton>
                    )}
                    <IconButton 
                      size="small" 
                      onClick={() => handleDeleteJob(job.id)}
                      color="error"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Create Batch Dialog */}
      <Dialog 
        open={isCreateDialogOpen} 
        onClose={() => setIsCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Batch Job</DialogTitle>
        <DialogContent>
          <Stepper activeStep={activeStep} sx={{ mb: 3, mt: 2 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          {activeStep === 0 && (
            <Box>
              <TextField
                label="Batch Name"
                value={newBatch.name}
                onChange={(_e) => setNewBatch({ ...newBatch, name: e.target.value })}
                fullWidth
                margin="normal"
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Batch Type</InputLabel>
                <Select
                  value={newBatch.type}
                  onChange={(_e) => setNewBatch({ ...newBatch, type: e.target.value })}
                  label="Batch Type"
                >
                  <MenuItem value="video_generation">Video Generation</MenuItem>
                  <MenuItem value="thumbnail_update">Thumbnail Update</MenuItem>
                  <MenuItem value="metadata_update">Metadata Update</MenuItem>
                  <MenuItem value="analytics_sync">Analytics Sync</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Number of Items"
                type="number"
                value={newBatch.videoCount}
                onChange={(_e) => setNewBatch({ ...newBatch, videoCount: parseInt(e.target.value) })}
                fullWidth
                margin="normal"
                helperText="Maximum 100 items per batch"
                inputProps={{ min: 1, max: 100 }}
              />
            </Box>
          )}

          {activeStep === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>Processing Options</Typography>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newBatch.options.generateThumbnails}
                    onChange={(_e) => setNewBatch({
                      ...newBatch,
                      options: { ...newBatch.options, generateThumbnails: e.target.checked }
                    })}
                  />
                }
                label="Generate Thumbnails"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newBatch.options.autoUpload}
                    onChange={(_e) => setNewBatch({
                      ...newBatch,
                      options: { ...newBatch.options, autoUpload: e.target.checked }
                    })}
                  />
                }
                label="Auto Upload to YouTube"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newBatch.options.qualityCheck}
                    onChange={(_e) => setNewBatch({
                      ...newBatch,
                      options: { ...newBatch.options, qualityCheck: e.target.checked }
                    })}
                  />
                }
                label="Enable Quality Check"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={newBatch.options.costOptimization}
                    onChange={(_e) => setNewBatch({
                      ...newBatch,
                      options: { ...newBatch.options, costOptimization: e.target.checked }
                    })}
                  />
                }
                label="Cost Optimization Mode"
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Priority</InputLabel>
                <Select
                  value={newBatch.priority}
                  onChange={(_e) => setNewBatch({ ...newBatch, priority: e.target.value })}
                  label="Priority"
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                </Select>
              </FormControl>
            </Box>
          )}

          {activeStep === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>Select Channels</Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                Select channels to distribute the batch job across multiple YouTube accounts
              </Alert>
              {/* Mock channel selection */}
              <List>
                {['Tech Reviews Pro', 'Gaming Central', 'DIY Crafts Hub'].map((channel) => (
                  <ListItem key={channel}>
                    <ListItemIcon>
                      <Checkbox
                        checked={newBatch.channels.includes(channel)}
                        onChange={(_e) => {
                          if (e.target.checked) {
                            setNewBatch({ ...newBatch, channels: [...newBatch.channels, channel] });
                          } else {
                            setNewBatch({ 
                              ...newBatch, 
                              channels: newBatch.channels.filter(c => c !== channel)
                            });
                          }
                        }}
                      />
                    </ListItemIcon>
                    <ListItemText primary={channel} />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {activeStep === 3 && (
            <Box>
              <Typography variant="h6" gutterBottom>Review Batch Job</Typography>
              <List>
                <ListItem>
                  <ListItemText primary="Name" secondary={newBatch.name} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Type" secondary={newBatch.type} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Items" secondary={newBatch.videoCount} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Channels" secondary={newBatch.channels.join(', ') || 'None selected'} />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Priority" secondary={newBatch.priority} />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Estimated Cost" 
                    secondary={`$${(newBatch.videoCount * 2.04).toFixed(2)}`}
                  />
                </ListItem>
              </List>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsCreateDialogOpen(false)}>Cancel</Button>
          {activeStep > 0 && (
            <Button onClick={() => setActiveStep(activeStep - 1)}>Back</Button>
          )}
          {activeStep < steps.length - 1 && (
            <Button 
              onClick={() => setActiveStep(activeStep + 1)}
              variant="contained"
              disabled={activeStep === 0 && !newBatch.name}
            >
              Next
            </Button>
          )}
          {activeStep === steps.length - 1 && (
            <Button 
              onClick={handleCreateBatch}
              variant="contained"
              disabled={!newBatch.name || newBatch.channels.length === 0}
            >
              Create Batch
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BatchOperations;