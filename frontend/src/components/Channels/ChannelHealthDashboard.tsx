import React, { useState, useEffect } from 'react';
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
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Badge,
  useTheme,
  Accordion,
  AccordionSummary,
  AccordionDetails
 } from '@mui/material';
import { 
  CheckCircle,
  Warning,
  TrendingUp,
  TrendingDown,
  Info,
  Refresh,
  ExpandMore,
  Schedule,
  AutoFixHigh
 } from '@mui/icons-material';
import {  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip  } from 'recharts';
import {  format, subDays  } from 'date-fns';

interface HealthMetric {
  category: string,
  score: number,
  maxScore: 100,
  status: 'healthy' | 'warning' | 'critical',
  factors: {,
  name: string,
    value: number,
  impact: 'positive' | 'negative' | 'neutral';
    recommendation?: string;
  }[];
}

interface HealthIssue {
  id: string,
  severity: 'low' | 'medium' | 'high' | 'critical',
  category: string,
  title: string,
  description: string,
  impact: string,
  solution: string,
  autoFixAvailable: boolean}

interface ChannelHealthData {
  channelId: string,
  channelName: string,
  overallHealth: number,
  trend: 'improving' | 'stable' | 'declining',
  lastChecked: Date,
  metrics: HealthMetric[],
  issues: HealthIssue[],
  history: {  date:  Date; score: number  }[];
}

export const ChannelHealthDashboard: React.FC<{  channelId?:  string  }> = ({ channelId }) => {
  const theme = useTheme();
  const [healthData, setHealthData] = useState<ChannelHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoFixing, setAutoFixing] = useState(false);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  useEffect(() => {
    // Simulate fetching health data
    setTimeout(() => {
      setHealthData({
        channelId: channelId || '1',
        channelName: 'Tech Insights Daily',
        overallHealth: 85,
        trend: 'improving',
        lastChecked: new Date(),
        metrics: [ {,
  category: 'Content Performance',
            score: 88,
            maxScore: 100,
            status: 'healthy',
            factors: [
              {  name:  'View Count', value: 92, impact: 'positive'  },
              {  name:  'Engagement Rate', value: 85, impact: 'positive'  },
              {  name:  'Average View Duration', value: 78, impact: 'neutral'  },
              {  name:  'Click-Through Rate', value: 95, impact: 'positive'  } ]
          },
          {
            category: 'Upload Consistency',
            score: 75,
            maxScore: 100,
            status: 'warning',
            factors: [ {  name:  'Upload Frequency', value: 70, impact: 'negative', recommendation: 'Increase upload frequency to daily'  },
              {  name:  'Schedule Adherence', value: 80, impact: 'neutral'  },
              {  name:  'Content Gaps', value: 65, impact: 'negative', recommendation: 'Fill content gaps on weekends'  } ]
          },
          {
            category: 'Audience Growth',
            score: 92,
            maxScore: 100,
            status: 'healthy',
            factors: [ {  name:  'Subscriber Growth', value: 95, impact: 'positive'  },
              {  name:  'Retention Rate', value: 88, impact: 'positive'  },
              {  name:  'New Viewer Acquisition', value: 93, impact: 'positive'  } ]
          },
          {
            category: 'Monetization',
            score: 82,
            maxScore: 100,
            status: 'healthy',
            factors: [ {  name:  'RPM', value: 85, impact: 'positive'  },
              {  name:  'Ad Revenue', value: 80, impact: 'neutral'  },
              {  name:  'Channel Memberships', value: 75, impact: 'neutral'  } ]
          },
          {
            category: 'Technical Health',
            score: 95,
            maxScore: 100,
            status: 'healthy',
            factors: [ {  name:  'API Quota Usage', value: 60, impact: 'positive'  },
              {  name:  'Upload Success Rate', value: 100, impact: 'positive'  },
              {  name:  'Processing Errors', value: 5, impact: 'positive'  } ]
          },
          {
            category: 'Compliance',
            score: 100,
            maxScore: 100,
            status: 'healthy',
            factors: [ {  name:  'Community Guidelines', value: 100, impact: 'positive'  },
              {  name:  'Copyright Status', value: 100, impact: 'positive'  },
              {  name:  'Monetization Policies', value: 100, impact: 'positive'  } ]
          }],
        issues: [ { id: '1',
            severity: 'medium',
            category: 'Upload Consistency',
            title: 'Irregular Upload Schedule',
            description: 'Videos are not being uploaded at consistent times',
            impact: 'May reduce audience retention by 15%',
            solution: 'Enable auto-scheduling with optimal time slots',
            autoFixAvailable: true },
          { id: '2',
            severity: 'low',
            category: 'Content Performance',
            title: 'Low Weekend Engagement',
            description: 'Weekend videos receive 30% less engagement',
            impact: 'Missing potential revenue opportunities',
            solution: 'Adjust content strategy for weekends',
            autoFixAvailable: false },
          { id: '3',
            severity: 'low',
            category: 'Monetization',
            title: 'Underutilized Membership Features',
            description: 'Channel memberships could be better promoted',
            impact: 'Potential 20% increase in membership revenue',
            solution: 'Add membership perks and promote in videos',
            autoFixAvailable: false } ],
        history: Array.from({  length:  30  }, (_, i) => ({ date: subDays(new Date(), 29 - i),
          score: 75 + Math.random() * 15 + (i * 0.3) }))
      });
      setLoading(false)}, 1000)}, [channelId]);

  const getHealthColor = (score: number) => {
    if (score >= 80) return theme.palette.success.main;
    if (score >= 60) return theme.palette.warning.main;
    return theme.palette.error.main;
  };

  const getHealthLabel = (score: number) => {
    if (score >= 80) return 'Healthy';
    if (score >= 60) return 'Needs Attention';
    return 'Critical';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default'}
  };

  const handleAutoFix = async (issue: HealthIssue) => { setAutoFixing(true);
    // Simulate auto-fix
    await new Promise(resolve => setTimeout(resolve, 2000));
    setHealthData(prev => prev ? {
      ...prev,
      issues: prev.issues.filter(i => i.id !== issue.id) } : null);
    setAutoFixing(false)};

  const radarData = healthData?.metrics.map(metric => () { category: metric.category,
    score: metric.score,
    fullMark: 100 })) || [];

  if (loading) {
    return (
    <>
      <Box sx={ { display:  'flex', justifyContent: 'center', p: 4  }}>
        <CircularProgress />
      </Box>
    )}

  return (
    <Box>
      {/* Overall Health Score */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={ { display:  'flex', alignItems: 'center', justifyContent: 'space-between'  }}>
                <Box sx={ { display:  'flex', alignItems: 'center', gap: 3  }}>
                  <Box sx={ { position:  'relative', display: 'inline-flex'  }}>
                    <CircularProgress
                      variant="determinate"
                      value={healthData?.overallHealth || 0}
                      size={120}
                      thickness={4}
                      sx={ { color:  getHealthColor(healthData?.overallHealth || 0)  }}
                    />
                    <Box
                      sx={ {
                        top: 0,
                        left: 0,
                        bottom: 0,
                        right: 0,
                        position: 'absolute',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center' }}
                    >
                      <Box sx={ { textAlign:  'center'  }}>
                        <Typography variant="h3" fontWeight="bold">
                          {healthData?.overallHealth}%
                        </Typography>
      <Typography variant="caption" color="text.secondary">
                          Health Score
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  <Box>
                    <Typography variant="h5" fontWeight="bold" gutterBottom>
                      {healthData?.channelName}
                    </Typography>
                    <Box sx={ { display:  'flex', gap: 1, alignItems: 'center'  }}>
                      <Chip
                        label={getHealthLabel(healthData?.overallHealth || 0)}
                        color={ healthData?.overallHealth! >= 80 ? 'success' :  healthData?.overallHealth! >= 60 ? 'warning' : 'error' }
                        size="small"
                      />
                      <Chip
                        icon={ healthData?.trend === 'improving' ? <TrendingUp /> :  healthData?.trend === 'declining' ? <TrendingDown /> </>: <TrendingUp /> }
                        label={healthData?.trend}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="caption" color="text.secondary">
                        Last checked: { healthData?.lastChecked ? format(healthData.lastChecked, 'HH: mm') : '' }
                      </Typography>
                    </Box>
                  </Box>
                </Box>

                <Button
                  variant="contained"
                  startIcon={<Refresh />}
                  onClick={() => setLoading(true)}
                >
                  Refresh
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Health Radar Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Health Overview
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="category" tick={ { fontSize:  10  }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar
                    name="Health Score"
                    dataKey="score"
                    stroke={theme.palette.primary.main}
                    fill={theme.palette.primary.main}
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Health Trend */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                30-Day Health Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={healthData?.history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(date) => format(date, 'MM/dd')}
                    tick={ { fontSize:  10  }}
                  />
                  <YAxis domain={[0, 100]} />
                  <RechartsTooltip
                    labelFormatter={(date) => format(date, 'PPP')}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke={theme.palette.primary.main}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Metrics */}
        <Grid item xs={12}>
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Detailed Health Metrics
          </Typography>
          {healthData?.metrics.map((metric) => (
            <Accordion
              key={metric.category}
              expanded={expandedMetric === metric.category}
              onChange={(() => setExpandedMetric(
                expandedMetric === metric.category ? null : metric.category
              )}
            >
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={ { display:  'flex', alignItems: 'center', width: '100%', gap: 2  }}>
                  <Box sx={ { flex:  1  }}>
                    <Typography variant="subtitle1" fontWeight="medium">
                      {metric.category}
                    </Typography>
                  </Box>
                  <Chip
                    label={`${metric.score}%`}
                    color={ metric.status === 'healthy' ? 'success' :  metric.status === 'warning' ? 'warning' : 'error' }
                    size="small"
                  />
                  <LinearProgress
                    variant="determinate"
                    value={metric.score}
                    sx={ {
                      width: 100,
                      height: 6,
                      borderRadius: 1,
                      backgroundColor: 'grey.200',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getHealthColor(metric.score) }
                    }}
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  {metric.factors.map((factor, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {factor.impact === 'positive' ? (
                          <CheckCircle color="success" />
                        ) : factor.impact === 'negative' ? (
                          <Warning color="warning" />
                        ) : (
                          <Info color="info" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={factor.name}
                        secondary={factor.recommendation}
                      />
                      <ListItemSecondaryAction>
                        <Typography variant="body2" fontWeight="bold">
                          {factor.value}%
                        </Typography>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          ))}
        </Grid>

        {/* Issues & Recommendations */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Issues & Recommendations
              </Typography>
              
              {healthData?.issues.length === 0 ? (
                <Alert severity="success">
                  No issues detected. Your channel is healthy!
                </Alert>
              ) : (
                <List>
                  {healthData?.issues.map((issue) => (
                    <ListItem key={issue.id}>
                      <ListItemIcon>
                        <Badge
                          badgeContent={issue.severity}
                          color={getSeverityColor(issue.severity) as any}
                        >
                          <Warning />
                        </Badge>
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={ { display:  'flex', alignItems: 'center', gap: 1  }}>
                            <Typography variant="subtitle1" fontWeight="medium">
                              {issue.title}
                            </Typography>
                            <Chip
                              label={issue.category}
                              size="small"
                              variant="outlined"
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {issue.description}
                            </Typography>
                            <Typography variant="caption" color="warning.main">
                              Impact: {issue.impact}
                            </Typography>
                            <Typography variant="body2" color="primary.main" sx={ { mt:  1  }}>
                              Solution: {issue.solution}
                            </Typography>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        {issue.autoFixAvailable && (
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<AutoFixHigh />}
                            onClick={() => handleAutoFix(issue)}
                            disabled={autoFixing}
                          >
                            Auto Fix
                          </Button>
                        )}
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  </>
  )};`