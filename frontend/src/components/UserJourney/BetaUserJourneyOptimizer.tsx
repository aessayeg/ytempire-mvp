import React, { useState, useEffect } from 'react';
import { 
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  IconButton,
  Chip,
  Alert,
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Avatar,
  LinearProgress,
  Rating,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Tabs,
  Tab,
  useTheme,
  Divider
 } from '@mui/material';
import { 
  PlayCircle,
  Pause,
  FastForward,
  FastRewind,
  BugReport,
  CheckCircle,
  Warning,
  Error,
  Mouse,
  TouchApp,
  Visibility,
  Timeline as TimelineIcon,
  EmojiEmotions,
  SentimentDissatisfied,
  SentimentNeutral,
  NavigateNext,
  Person,
  Edit
 } from '@mui/icons-material';
import {  format  } from 'date-fns';

interface SessionRecording {
  id: string,
  userId: string,

  userName: string,
  startTime: Date,

  endTime: Date,
  duration: number,

  pages: string[],
  actions: UserAction[],

  painPoints: PainPoint[],
  sentiment: 'positive' | 'neutral' | 'negative',

  completionRate: number,
  device: 'desktop' | 'mobile' | 'tablet'}

interface UserAction {
  id: string,
  timestamp: Date,

  type: 'click' | 'scroll' | 'input' | 'navigation' | 'error' | 'rage_click' | 'dead_click',
  element: string,

  page: string;
  value?: string;
  duration?: number;
  coordinates?: { x: number; y: number };
  success: boolean}

interface PainPoint {
  id: string,
  type: 'confusion' | 'frustration' | 'error' | 'abandonment' | 'slow_task',

  severity: 'low' | 'medium' | 'high' | 'critical',
  page: string,

  element: string,
  description: string,

  timestamp: Date,
  duration: number,

  recommendation: string,
  impactedUsers: number}

interface JourneyStep {
  id: string,
  name: string,

  expectedDuration: number,
  actualDuration: number,

  completionRate: number,
  dropoffRate: number,

  painPoints: number,
  satisfaction: number}

interface WireframeImprovement {
  id: string,
  page: string,

  component: string,
  currentDesign: string,

  proposedDesign: string,
  reason: string,

  impact: 'high' | 'medium' | 'low',
  effort: 'high' | 'medium' | 'low',

  status: 'pending' | 'in_progress' | 'completed' | 'tested'}

export const BetaUserJourneyOptimizer: React.FC = () => { const theme = useTheme();
  const [selectedSession, setSelectedSession] = useState<SessionRecording | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentActionIndex, setCurrentActionIndex] = useState(0);
  const [tabValue, setTabValue] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [selectedPainPoint, setSelectedPainPoint] = useState<PainPoint | null>(null);

  // Mock data
  const [sessions] = useState<SessionRecording[]>([
    {
      id: '1',
      userId: 'beta_user_1',
      userName: 'John Doe',
      startTime: new Date('2024-01-15 T10:30:00'),
      endTime: new Date('2024-01-15 T10:45:00'),
      duration: 900,
      pages: ['/dashboard', '/videos/create', '/channels', '/analytics'],
      actions: [ {,
  id: 'a1',
          timestamp: new Date('2024-01-15 T10:30:00'),
          type: 'navigation',
          element: 'dashboard',
          page: '/dashboard',
          success: true },
        { id: 'a2',
          timestamp: new Date('2024-01-15 T10:31:00'),
          type: 'click',
          element: 'create_video_button',
          page: '/dashboard',
          success: true },
        { id: 'a3',
          timestamp: new Date('2024-01-15 T10:32:00'),
          type: 'rage_click',
          element: 'generate_script_button',
          page: '/videos/create',
          success: false },
        { id: 'a4',
          timestamp: new Date('2024-01-15 T10:33:00'),
          type: 'error',
          element: 'script_generation',
          page: '/videos/create',
          success: false } ],
      painPoints: [ { id: 'pp1',
          type: 'frustration',
          severity: 'high',
          page: '/videos/create',
          element: 'generate_script_button',
          description: 'User clicked multiple times on generate button with no response',
          timestamp: new Date('2024-01-15 T10:32:00'),
          duration: 15,
          recommendation: 'Add loading indicator and disable button during processing',
          impactedUsers: 12 },
        { id: 'pp2',
          type: 'confusion',
          severity: 'medium',
          page: '/dashboard',
          element: 'metrics_section',
          description: 'User hovered over metrics without understanding what they mean',
          timestamp: new Date('2024-01-15 T10:30:30'),
          duration: 8,
          recommendation: 'Add tooltips explaining each metric',
          impactedUsers: 8 } ],
      sentiment: 'negative',
      completionRate: 65,
      device: 'desktop',

    },
    { id: '2',
      userId: 'beta_user_2',
      userName: 'Jane Smith',
      startTime: new Date('2024-01-15 T11:00:00'),
      endTime: new Date('2024-01-15 T11:20:00'),
      duration: 1200,
      pages: ['/onboarding', '/dashboard', '/channels', '/videos/create', '/analytics'],
      actions: [],
      painPoints: [{,
  id: 'pp3',
          type: 'slow_task',
          severity: 'medium',
          page: '/channels',
          element: 'channel_setup',
          description: 'Channel connection took longer than expected',
          timestamp: new Date('2024-01-15 T11:05:00'),
          duration: 45,
          recommendation: 'Optimize API calls and add progress indicator',
          impactedUsers: 15 }],
      sentiment: 'positive',
      completionRate: 85,
      device: 'desktop',

    }]);

  const [painPointsSummary] = useState<PainPoint[]>([ { id: 'pp_summary_1',
      type: 'frustration',
      severity: 'critical',
      page: '/videos/create',
      element: 'script_generation',
      description: 'Script generation fails without clear error message',
      timestamp: new Date(),
      duration: 0,
      recommendation: 'Implement proper error handling with actionable messages',
      impactedUsers: 25 },
    { id: 'pp_summary_2',
      type: 'confusion',
      severity: 'high',
      page: '/onboarding',
      element: 'channel_connection',
      description: 'Users confused about YouTube authorization process',
      timestamp: new Date(),
      duration: 0,
      recommendation: 'Add step-by-step visual guide for authorization',
      impactedUsers: 18 },
    { id: 'pp_summary_3',
      type: 'abandonment',
      severity: 'high',
      page: '/analytics',
      element: 'complex_charts',
      description: 'Users leave analytics page due to overwhelming information',
      timestamp: new Date(),
      duration: 0,
      recommendation: 'Simplify initial view with progressive disclosure',
      impactedUsers: 15 } ]);

  const [journeySteps] = useState<JourneyStep[]>([ { id: 'step1',
      name: 'Sign Up & Onboarding',
      expectedDuration: 300,
      actualDuration: 480,
      completionRate: 92,
      dropoffRate: 8,
      painPoints: 3,
      satisfaction: 3.5 },
    { id: 'step2',
      name: 'Channel Connection',
      expectedDuration: 120,
      actualDuration: 240,
      completionRate: 85,
      dropoffRate: 15,
      painPoints: 5,
      satisfaction: 3.0 },
    { id: 'step3',
      name: 'First Video Creation',
      expectedDuration: 600,
      actualDuration: 900,
      completionRate: 72,
      dropoffRate: 28,
      painPoints: 8,
      satisfaction: 2.8 },
    { id: 'step4',
      name: 'Video Publishing',
      expectedDuration: 180,
      actualDuration: 200,
      completionRate: 95,
      dropoffRate: 5,
      painPoints: 1,
      satisfaction: 4.2 },
    { id: 'step5',
      name: 'Analytics Review',
      expectedDuration: 300,
      actualDuration: 180,
      completionRate: 88,
      dropoffRate: 12,
      painPoints: 4,
      satisfaction: 3.8 } ]);

  const [wireframeImprovements] = useState<WireframeImprovement[]>([ { id: 'wi1',
      page: '/videos/create',
      component: 'Script Generation Button',
      currentDesign: 'Single button with no feedback',
      proposedDesign: 'Button with loading, state, progress, indicator, and disable on click',
      reason: 'Users clicking multiple times due to no feedback',
      impact: 'high',
      effort: 'low',
      status: 'in_progress' },
    { id: 'wi2',
      page: '/dashboard',
      component: 'Metrics Cards',
      currentDesign: 'Numbers only without context',
      proposedDesign: 'Add, tooltips, trend, indicators, and comparison to previous period',
      reason: 'Users confused about metric meanings',
      impact: 'medium',
      effort: 'low',
      status: 'pending' },
    { id: 'wi3',
      page: '/onboarding',
      component: 'Channel Authorization',
      currentDesign: 'Text instructions only',
      proposedDesign: 'Visual step-by-step guide with screenshots',
      reason: 'High dropout rate during authorization',
      impact: 'high',
      effort: 'medium',
      status: 'pending' },
    { id: 'wi4',
      page: '/analytics',
      component: 'Initial View',
      currentDesign: 'All charts visible at once',
      proposedDesign: 'Summary view with expandable sections',
      reason: 'Information overload causing abandonment',
      impact: 'high',
      effort: 'medium',
      status: 'tested' } ]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return theme.palette.error.main;
      case 'high': return theme.palette.warning.main;
      case 'medium': return theme.palette.info.main;
      case 'low': return theme.palette.success.main;
      default: return theme.palette.grey[500]}
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <EmojiEmotions color="success" />;
      case 'negative': return <SentimentDissatisfied color="error" />;
      default: return <SentimentNeutral color="action" />}
  };

  const getActionIcon = (type: string) => {
    switch (type) {
      case 'click': return <Mouse />;
      case 'rage_click': return <Warning color="error" />;
      case 'error': return <Error color="error" />;
      case 'navigation': return <NavigateNext />;
      default: return <TouchApp />}
  };

  const renderSessionRecordings = () => (
    <Grid container spacing={3}>
      {/* Session List */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardHeader title="Session Recordings" />
          <CardContent>
            <List>
              {sessions.map((session) => (
                <ListItem
                  key={session.id}
                  button
                  selected={selectedSession?.id === session.id}
                  onClick={() => setSelectedSession(session}
                >
                  <ListItemIcon>
                    <Avatar>
                      <Person />
                    </Avatar>
                  </ListItemIcon>
                  <ListItemText
                    primary={session.userName}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {format(session.startTime, 'PPp')}
                        </Typography>
                        <Typography variant="caption">
                          Duration: {Math.floor(session.duration / 60)}m {session.duration % 60}s
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Box sx={{ textAlign: 'right' }}>
                      {getSentimentIcon(session.sentiment)}
                      <Typography variant="caption" display="block">
                        {session.completionRate}%
                      </Typography>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Grid>

      {/* Session Playback */}
      <Grid item xs={12} md={8}>
        {selectedSession ? (
          <Card>
            <CardHeader
              title={`Session: ${selectedSession.userName}`}`
              subheader={`${format(selectedSession.startTime, 'PPP')} • ${selectedSession.device}`}
              action={
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <IconButton onClick={() => setPlaybackSpeed(0.5}>
                    <FastRewind />
                  </IconButton>
                  <IconButton onClick={() => setIsPlaying(!isPlaying}>
                    {isPlaying ? <Pause /> </>: <PlayCircle />}
                  </IconButton>
                  <IconButton onClick={() => setPlaybackSpeed(2}>
                    <FastForward />
                  </IconButton>
                </Box>
              }
            />
            <CardContent>
              {/* Session Timeline */}
              <Timeline position="alternate">
                {selectedSession.actions.map((action, index) => (
                  <TimelineItem key={action.id}>
                    <TimelineOppositeContent>
                      <Typography variant="caption" color="text.secondary">
                        {format(action.timestamp, 'HH:mm:ss')}
                      </Typography>
                    </TimelineOppositeContent>
                    <TimelineSeparator>
                      <TimelineDot color={action.success ? 'success' : 'error'}>
                        {getActionIcon(action.type)}
                      </TimelineDot>
                      {index < selectedSession.actions.length - 1 && <TimelineConnector />}
                    </TimelineSeparator>
                    <TimelineContent>
                      <Paper elevation={3} sx={{ p: 1 }}>
                        <Typography variant="subtitle2">
                          {action.type.replace('_', ' ').toUpperCase()}
                        </Typography>
                        <Typography variant="caption" display="block">
                          {action.element}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Page: {action.page}
                        </Typography>
                      </Paper>
                    </TimelineContent>
                  </TimelineItem>
                ))}
              </Timeline>

              {/* Pain Points in Session */}
              {selectedSession.painPoints.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Identified Pain Points
                  </Typography>
                  {selectedSession.painPoints.map((painPoint) => (
                    <Alert
                      key={painPoint.id}
                      severity={painPoint.severity === 'critical' ? 'error' : painPoint.severity === 'high' ? 'warning' : 'info'}
                      sx={{ mb: 1 }}
                    >
                      <Typography variant="subtitle2">
                        {painPoint.type.replace('_', ' ').toUpperCase()}
                      </Typography>
                      <Typography variant="body2">
                        {painPoint.description}
                      </Typography>
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Recommendation: {painPoint.recommendation}
                      </Typography>
                    </Alert>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent>
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h6" color="text.secondary">
                  Select a session to view details
                </Typography>
              </Box>
            </CardContent>
          </Card>
        )}
      </Grid>
    </Grid>
  );

  const renderPainPoints = () => (
    <Grid container spacing={3}>
      {/* Pain Points Summary */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            title="Pain Points Analysis"
            subheader="Aggregated from all beta user sessions"
          />
          <CardContent>
            {painPointsSummary.map((painPoint) => (
              <Paper key={painPoint.id} sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flex: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Chip
                        label={painPoint.type.replace('_', ' ').toUpperCase()}
                        size="small"
                        sx={ {
                          backgroundColor: getSeverityColor(painPoint.severity),
                          color: 'white' }}
                      />
                      <Chip
                        label={`${painPoint.impactedUsers} users affected`}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="caption" color="text.secondary">
                        {painPoint.page}
                      </Typography>
                    </Box>
                    <Typography variant="body1" gutterBottom>
                      {painPoint.description}
                    </Typography>
                    <Alert severity="success" icon={<CheckCircle />} sx={{ mt: 1 }}>
                      <Typography variant="body2">
                        <strong>Recommendation:</strong> {painPoint.recommendation}
                      </Typography>
                    </Alert>
                  </Box>
                  <Box sx={{ ml: 2 }}>
                    <Button variant="outlined" size="small" startIcon={<BugReport />}>
                      Create Issue
                    </Button>
                  </Box>
                </Box>
              </Paper>
            ))}
          </CardContent>
        </Card>
      </Grid>

      {/* Journey Funnel */}
      <Grid item xs={12}>
        <Card>
          <CardHeader title="User Journey Funnel" />
          <CardContent>
            <Stepper orientation="vertical">
              {journeySteps.map((step, index) => (
                <Step key={step.id} active expanded>
                  <StepLabel>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Typography variant="subtitle1">{step.name}</Typography>
                      <Chip
                        label={`${step.completionRate}% completion`}
                        size="small"
                        color={step.completionRate > 80 ? 'success' : step.completionRate > 60 ? 'warning' : 'error'}
                      />
                      <Rating value={step.satisfaction} readOnly size="small" />
                    </Box>
                  </StepLabel>
                  <StepContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={3}>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Expected Duration
                        </Typography>
                        <Typography variant="body2">
                          {Math.floor(step.expectedDuration / 60)}m {step.expectedDuration % 60}s
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={3}>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Actual Duration
                        </Typography>
                        <Typography
                          variant="body2"
                          color={step.actualDuration > step.expectedDuration * 1.5 ? 'error.main' : 'text.primary'}
                        >
                          {Math.floor(step.actualDuration / 60)}m {step.actualDuration % 60}s
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={3}>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Dropoff Rate
                        </Typography>
                        <Typography
                          variant="body2"
                          color={step.dropoffRate > 20 ? 'error.main' : step.dropoffRate > 10 ? 'warning.main' : 'success.main'}
                        >
                          {step.dropoffRate}%
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={3}>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Pain Points
                        </Typography>
                        <Typography variant="body2">
                          {step.painPoints} identified
                        </Typography>
                      </Grid>
                    </Grid>
                    <LinearProgress
                      variant="determinate"
                      value={step.completionRate}
                      sx={{ mt: 2, height: 8, borderRadius: 1 }}
                      color={step.completionRate > 80 ? 'success' : step.completionRate > 60 ? 'warning' : 'error'}
                    />
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderWireframeImprovements = () => (
    <Grid container spacing={3}>
      {wireframeImprovements.map((improvement) => (
        <Grid item xs={12} md={6} key={improvement.id}>
          <Card>
            <CardHeader
              title={improvement.component}
              subheader={improvement.page}
              action={
                <Chip
                  label={improvement.status.replace('_', ' ').toUpperCase()}
                  size="small"
                  color={
                    improvement.status === 'completed' ? 'success' :
                    improvement.status === 'tested' ? 'info' :
                    improvement.status === 'in_progress' ? 'warning' : 'default'
                  }
                />
              }
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" color="error.main" gutterBottom>
                  Current Design Issue:
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  {improvement.currentDesign}
                </Typography>
                
                <Typography variant="subtitle2" color="success.main" gutterBottom>
                  Proposed Solution:
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  {improvement.proposedDesign}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  Reason for Change:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {improvement.reason}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                <Chip
                  label={`Impact: ${improvement.impact}`}
                  size="small"
                  color={improvement.impact === 'high' ? 'error' : improvement.impact === 'medium' ? 'warning' : 'default'}
                />
                <Chip
                  label={`Effort: ${improvement.effort}`}
                  size="small"
                  variant="outlined"
                />
              </Box>
            </CardContent>
            <Divider />
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between' }}>
              <Button size="small" startIcon={<Visibility />}>
                View Mockup
              </Button>
              <Button size="small" startIcon={<Edit />} variant="contained">
                Implement
              </Button>
            </Box>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <>
      <Box>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Beta User Journey Optimization
      </Typography>
      <Tabs value={tabValue} onChange={(_, v) => setTabValue(v} sx={{ mb: 3 }>
        <Tab label="Session Recordings" />
        <Tab label="Pain Points" />
        <Tab label="Wireframe Improvements" />
      </Tabs>

      {tabValue === 0 && renderSessionRecordings()}
      {tabValue === 1 && renderPainPoints()}
      {tabValue === 2 && renderWireframeImprovements()}
    </Box>
  </>
  )};`