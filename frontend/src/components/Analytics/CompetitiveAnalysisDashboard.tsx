/**
 * Competitive Analysis Dashboard
 * Provides insights into competitor performance and market positioning
 */

import React, { useState, useMemo } from 'react';
import { 
  Box,
  Paper,
  Checkbox,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  Avatar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Rating,
  Alert,
  AlertTitle,
  Tabs,
  Tab,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  TextField,
  Stack
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Group as GroupIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { 
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  Tooltip as RechartsTooltip
} from 'recharts';
import { format, subDays } from 'date-fns';

// Types
interface Competitor {
  id: string;
  channelName: string;
  channelId: string;
  thumbnailUrl: string;
  subscriberCount: number;
  videoCount: number;
  viewCount: number;
  category: string;
  country: string;
  joinedDate: Date;
  isTracking: boolean;
  lastUpdated: Date;
}

interface CompetitorMetrics {
  channelId: string;
  avgViews: number;
  avgLikes: number;
  avgComments: number;
  engagementRate: number;
  uploadFrequency: number; // videos per week
  estimatedRevenue: number;
  growthRate: number; // % per month
  contentQualityScore: number; // 0-100
  audienceRetention: number; // percentage
  clickThroughRate: number;
}

interface MarketInsight {
  trend: string;
  impact: 'high' | 'medium' | 'low';
  description: string;
  recommendedAction: string;
}

interface ContentGap {
  topic: string;
  competitorsCovering: number;
  potentialViews: number;
  difficulty: 'easy' | 'medium' | 'hard';
  recommendedApproach: string;
}

const MOCK_COMPETITORS: Competitor[] = [
  {
    id: '1',
    channelName: 'TechVision Pro',
    channelId: 'UC_channel1',
    thumbnailUrl: 'https://via.placeholder.com/50',
    subscriberCount: 1250000,
    videoCount: 542,
    viewCount: 125000000,
    category: 'Technology',
    country: 'US',
    joinedDate: new Date('2019-03-15'),
    isTracking: true,
    lastUpdated: new Date()
  },
  {
    id: '2',
    channelName: 'Digital Insights',
    channelId: 'UC_channel2',
    thumbnailUrl: 'https://via.placeholder.com/50',
    subscriberCount: 850000,
    videoCount: 312,
    viewCount: 75000000,
    category: 'Technology',
    country: 'UK',
    joinedDate: new Date('2020-01-10'),
    isTracking: true,
    lastUpdated: new Date()
  },
  {
    id: '3',
    channelName: 'Future Tech Today',
    channelId: 'UC_channel3',
    thumbnailUrl: 'https://via.placeholder.com/50',
    subscriberCount: 2100000,
    videoCount: 723,
    viewCount: 310000000,
    category: 'Technology',
    country: 'US',
    joinedDate: new Date('2018-06-20'),
    isTracking: true,
    lastUpdated: new Date()
  }
];

export const CompetitiveAnalysisDashboard: React.FC = () => {
  // State
  const [competitors, setCompetitors] = useState<Competitor[]>(MOCK_COMPETITORS);
  const [selectedCompetitors, setSelectedCompetitors] = useState<string[]>(['1', '2']);
  const [currentTab, setCurrentTab] = useState(0);
  const [timeRange, setTimeRange] = useState('30d');
  const [isLoading, setIsLoading] = useState(false);
  const [newCompetitorUrl, setNewCompetitorUrl] = useState('');

  // Mock data generation
  const competitorMetrics = useMemo<Record<string, CompetitorMetrics>>(() => {
const metrics: Record<string, CompetitorMetrics> = {};
    competitors.forEach(comp => {
    
      metrics[comp.id] = {
        channelId: comp.channelId,
        avgViews: Math.floor(comp.viewCount / comp.videoCount),
        avgLikes: Math.floor(comp.viewCount / comp.videoCount * 0.05),
        avgComments: Math.floor(comp.viewCount / comp.videoCount * 0.002),
        engagementRate: 5.2 + Math.random() * 3,
        uploadFrequency: 2 + Math.random() * 3,
        estimatedRevenue: Math.floor(comp.viewCount * 0.002),
        growthRate: 5 + Math.random() * 15,
        contentQualityScore: 70 + Math.random() * 25,
        audienceRetention: 40 + Math.random() * 30,
        clickThroughRate: 3 + Math.random() * 7,

      };
    });
    return metrics;
  }, [competitors]);

  const marketInsights: MarketInsight[] = [
    {
      trend: 'AI-generated content gaining traction',
      impact: 'high',
      description: 'Competitors are increasingly using AI for content creation',
      recommendedAction: 'Differentiate with unique human insights and storytelling',

    },
    {
      trend: 'Short-form content outperforming',
      impact: 'high',
      description: 'Videos under 60 seconds seeing 2x engagement',
      recommendedAction: 'Create YouTube Shorts versions of main content',

    },
    {
      trend: 'Tutorial content saturated',
      impact: 'medium',
      description: 'Market oversaturated with basic tutorials',
      recommendedAction: 'Focus on advanced topics and case studies',

    }
  ];

  const contentGaps: ContentGap[] = [
    {
      topic: 'Quantum Computing Basics',
      competitorsCovering: 1,
      potentialViews: 250000,
      difficulty: 'hard',
      recommendedApproach: 'Create beginner-friendly series with animations',

    },
    {
      topic: 'Web3 Development Tools',
      competitorsCovering: 2,
      potentialViews: 180000,
      difficulty: 'medium',
      recommendedApproach: 'Hands-on tutorials with real projects',

    },
    {
      topic: 'AI Ethics Discussion',
      competitorsCovering: 0,
      potentialViews: 150000,
      difficulty: 'easy',
      recommendedApproach: 'Interview series with experts',

    }
  ];

  // Generate comparison chart data
  const comparisonData = useMemo(() => {
    const selected = competitors.filter(c => selectedCompetitors.includes(c.id));
    return selected.map(comp => ({
      name: comp.channelName,
      subscribers: comp.subscriberCount / 1000,
      views: comp.viewCount / 1000000,
      videos: comp.videoCount,
      engagement: competitorMetrics[comp.id]?.engagementRate || 0,
      revenue: competitorMetrics[comp.id]?.estimatedRevenue / 1000 || 0,

    }))}, [competitors, selectedCompetitors, competitorMetrics]);

  // Generate radar chart data for competitive positioning
  const radarData = useMemo(() => {
    const metrics = ['Content Quality', 'Upload Frequency', 'Engagement', 'Growth Rate', 'Revenue', 'Retention'];
    return metrics.map(metric => {
const dataPoint: Record<string, number | string> = { metric };
      selectedCompetitors.forEach(compId => {
        const comp = competitors.find(c => c.id === compId);
        const compMetrics = competitorMetrics[compId];
        if (comp && compMetrics) {
          switch (metric) {
            case 'Content Quality':
              dataPoint[comp.channelName] = compMetrics.contentQualityScore;
              break;
            case 'Upload Frequency':
              dataPoint[comp.channelName] = compMetrics.uploadFrequency * 20;
              break;
            case 'Engagement':
              dataPoint[comp.channelName] = compMetrics.engagementRate * 10;
              break;
            case 'Growth Rate':
              dataPoint[comp.channelName] = compMetrics.growthRate * 5;
              break;
            case 'Revenue':
              dataPoint[comp.channelName] = Math.min(100, compMetrics.estimatedRevenue / 1000);
              break;
            case 'Retention':
              dataPoint[comp.channelName] = compMetrics.audienceRetention;
              break;
          }
        }
      });
      return dataPoint;
    });
  }, [competitors, selectedCompetitors, competitorMetrics]);

  // Generate trend data
  const trendData = useMemo(() => {
    return Array.from({ length: 30 }, (_, i) => {
      const date = subDays(new Date(), 30 - i);
      const dataPoint: Record<string, string | number> = { date: format(date, 'MMM dd') };
      
      selectedCompetitors.forEach(compId => {
        const comp = competitors.find(c => c.id === compId);
        if (comp) {
          // Simulate growth trend
          const baseValue = comp.subscriberCount / 1000;
          const growth = (i / 30) * (competitorMetrics[compId]?.growthRate || 5) / 100;
          dataPoint[comp.channelName] = Math.floor(baseValue * (1 - 0.1 + growth));
        }
      });
      
      return dataPoint;
    });
  }, [competitors, selectedCompetitors, competitorMetrics]);

  // Add competitor
  const addCompetitor = () => {
    if (!newCompetitorUrl) return;
    
    // Extract channel ID from URL (mock, implementation)
    const newCompetitor: Competitor = {
  id: Date.now().toString(),
      channelName: 'New Competitor',
      channelId: 'UC_new',
      thumbnailUrl: 'https://via.placeholder.com/50',
      subscriberCount: Math.floor(Math.random() * 1000000),
      videoCount: Math.floor(Math.random() * 500),
      viewCount: Math.floor(Math.random() * 100000000),
      category: 'Technology',
      country: 'US',
      joinedDate: new Date(),
      isTracking: false,
      lastUpdated: new Date()
    };
    
    setCompetitors([...competitors, newCompetitor]);
    setNewCompetitorUrl('');
  };

  // Remove competitor
  const removeCompetitor = (id: string) => {
    setCompetitors(competitors.filter(c => c.id !== id));
    setSelectedCompetitors(selectedCompetitors.filter(cId => cId !== id));
  };

  // Toggle competitor selection
  const toggleCompetitorSelection = (id: string) => {
    if (selectedCompetitors.includes(id)) {
      setSelectedCompetitors(selectedCompetitors.filter(cId => cId !== id));
    } else {
      setSelectedCompetitors([...selectedCompetitors, id]);
    }
  };

  // Refresh data
  const refreshData = async () => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    setIsLoading(false);
  };

  // Export data
  const exportData = () => {
    const data = {
      competitors: competitors.filter(c => selectedCompetitors.includes(c.id)),
      metrics: selectedCompetitors.map(id => ({ id, ...competitorMetrics[id] })),
      insights: marketInsights,
      gaps: contentGaps,
      exportDate: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `competitive_analysis_${format(new Date(), 'yyyy-MM-dd')}.json`;
    a.click();
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Competitive Analysis</Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={refreshData}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={exportData}
          >
            Export
          </Button>
        </Stack>
      </Box>

      {/* Key Metrics Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <GroupIcon color="primary" sx={{ mr: 1 }} />
                <Typography color="textSecondary" variant="body2">
                  Tracked Competitors
                </Typography>
              </Box>
              <Typography variant="h4">{competitors.length}</Typography>
              <Typography variant="body2" color="success.main">
                {competitors.filter(c => c.isTracking).length} active
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                <Typography color="textSecondary" variant="body2">
                  Market Position
                </Typography>
              </Box>
              <Typography variant="h4">#4</Typography>
              <Typography variant="body2" color="success.main">
                ↑ 2 positions this month
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <SpeedIcon color="warning" sx={{ mr: 1 }} />
                <Typography color="textSecondary" variant="body2">
                  Performance Gap
                </Typography>
              </Box>
              <Typography variant="h4">-15%</Typography>
              <Typography variant="body2" color="textSecondary">
                vs. top competitor
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <InfoIcon color="info" sx={{ mr: 1 }} />
                <Typography color="textSecondary" variant="body2">
                  Content Gaps
                </Typography>
              </Box>
              <Typography variant="h4">{contentGaps.length}</Typography>
              <Typography variant="body2" color="info.main">
                Opportunities identified
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Tabs value={currentTab} onChange={(e, newValue) => setCurrentTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Competitors" />
        <Tab label="Comparison" />
        <Tab label="Market Insights" />
        <Tab label="Content Gaps" />
        <Tab label="Trends" />
      </Tabs>

      {/* Competitors Tab */}
      {currentTab === 0 && (
        <>
          {/* Add Competitor */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Stack direction="row" spacing={2}>
              <TextField
                fullWidth
                placeholder="Enter YouTube channel URL or ID"
                value={newCompetitorUrl}
                onChange={(e) => setNewCompetitorUrl(e.target.value)}
                size="small"
              />
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={addCompetitor}
                disabled={!newCompetitorUrl}
              >
                Add Competitor
              </Button>
            </Stack>
          </Paper>

          {/* Competitors List */}
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      indeterminate={
                        selectedCompetitors.length > 0 && 
                        selectedCompetitors.length < competitors.length
                      }
                      checked={selectedCompetitors.length === competitors.length}
                      onChange={(_) => {
                        if (_.target.checked) {
                          setSelectedCompetitors(competitors.map(c => c.id));
                        } else {
                          setSelectedCompetitors([]);
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell>Channel</TableCell>
                  <TableCell align="right">Subscribers</TableCell>
                  <TableCell align="right">Videos</TableCell>
                  <TableCell align="right">Total Views</TableCell>
                  <TableCell align="right">Engagement</TableCell>
                  <TableCell align="right">Growth</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {competitors.map((competitor) => {
                  const metrics = competitorMetrics[competitor.id];
                  return (
                    <TableRow key={competitor.id}>
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={selectedCompetitors.includes(competitor.id)}
                          onChange={() => toggleCompetitorSelection(competitor.id)}
                        />
                      </TableCell>
      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Avatar src={competitor.thumbnailUrl} sx={{ mr: 2 }} />
                          <Box>
                            <Typography variant="body1">
                              {competitor.channelName}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {competitor.category} • {competitor.country}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {(competitor.subscriberCount / 1000000).toFixed(2)}M
                      </TableCell>
                      <TableCell align="right">
                        {competitor.videoCount}
                      </TableCell>
                      <TableCell align="right">
                        {(competitor.viewCount / 1000000).toFixed(1)}M
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                          {metrics?.engagementRate.toFixed(1)}%
                          {metrics && metrics.engagementRate > 5 ? (
                            <TrendingUpIcon color="success" fontSize="small" sx={{ ml: 0.5 }} />
                          ) : (
                            <TrendingDownIcon color="error" fontSize="small" sx={{ ml: 0.5 }} />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`+${metrics?.growthRate.toFixed(1)}%`}
                          color={metrics && metrics.growthRate > 10 ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="center">
                        {competitor.isTracking ? (
                          <Chip label="Tracking" color="success" size="small" />
                        ) : (
                          <Chip label="Not Tracking" color="default" size="small" />
                        )}
                      </TableCell>
                      <TableCell align="center">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => removeCompetitor(competitor.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
      {/* Comparison Tab */}
      {currentTab === 1 && (
        <Grid container spacing={3}>
          {/* Bar Chart Comparison */}
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Channel Metrics Comparison
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="subscribers" fill="#8884d8" name="Subscribers (K)" />
                  <Bar dataKey="views" fill="#82ca9d" name="Views (M)" />
                  <Bar dataKey="engagement" fill="#ffc658" name="Engagement (%)" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Radar Chart - Competitive Positioning */}
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Competitive Positioning
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  {selectedCompetitors.map((compId, index) => {
                    const comp = competitors.find(c => c.id === compId);
                    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];
                    return comp ? (
                      <Radar
                        key={compId}
                        name={comp.channelName}
                        dataKey={comp.channelName}
                        stroke={colors[index % colors.length]}
                        fill={colors[index % colors.length]}
                        fillOpacity={0.3}
                      />
                    ) : null;
                  })}
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Performance Matrix */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Performance Matrix
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      {selectedCompetitors.map(compId => {
                        const comp = competitors.find(c => c.id === compId);
                        return (
                          <TableCell key={compId} align="center">
                            {comp?.channelName}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  </TableHead>
      <TableBody>
                    <TableRow>
                      <TableCell>Avg Views per Video</TableCell>
                      {selectedCompetitors.map(compId => (
                        <TableCell key={compId} align="center">
                          {competitorMetrics[compId]?.avgViews.toLocaleString()}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Upload Frequency (per, week)</TableCell>
                      {selectedCompetitors.map(compId => (
                        <TableCell key={compId} align="center">
                          {competitorMetrics[compId]?.uploadFrequency.toFixed(1)}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Content Quality Score</TableCell>
                      {selectedCompetitors.map(compId => (
                        <TableCell key={compId} align="center">
                          <Rating
                            value={competitorMetrics[compId]?.contentQualityScore / 20}
                            readOnly
                            precision={0.5}
                            size="small"
                          />
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Estimated Monthly Revenue</TableCell>
                      {selectedCompetitors.map(compId => (
                        <TableCell key={compId} align="center">
                          ${competitorMetrics[compId]?.estimatedRevenue.toLocaleString()}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
      {/* Market Insights Tab */}
      {currentTab === 2 && (
        <Grid container spacing={3}>
          {marketInsights.map((insight, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Badge
                      badgeContent={insight.impact}
                      color={
                        insight.impact === 'high' ? 'error' : 
                        insight.impact === 'medium' ? 'warning' : 'info'
                      }
                    >
                      <AnalyticsIcon />
                    </Badge>
                    <Typography variant="h6" sx={{ ml: 2 }}>
                      {insight.trend}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    {insight.description}
                  </Typography>
                  <Alert severity="info" icon={<InfoIcon />}>
                    <AlertTitle>Recommended Action</AlertTitle>
                    {insight.recommendedAction}
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      {/* Content Gaps Tab */}
      {currentTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Alert severity="success" sx={{ mb: 3 }}>
              <AlertTitle>Content Opportunities</AlertTitle>
              We've identified {contentGaps.length} content gaps based on competitor analysis and market demand.
            </Alert>
          </Grid>
          
          {contentGaps.map((gap, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {gap.topic}
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={`Difficulty: ${gap.difficulty}`}
                      color={
                        gap.difficulty === 'easy' ? 'success' : 
                        gap.difficulty === 'medium' ? 'warning' : 'error'
                      }
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <Chip
                      label={`${gap.competitorsCovering} competitors`}
                      size="small"
                    />
                  </Box>
                  
                  <Typography variant="body2" color="textSecondary" paragraph>
                    Potential, Views: <strong>{gap.potentialViews.toLocaleString()}</strong>
                  </Typography>
                  
                  <Typography variant="body2">
                    <strong>Approach:</strong> {gap.recommendedApproach}
                  </Typography>
                  
                  <Box sx={{ mt: 2 }}>
                    <Button variant="outlined" size="small" fullWidth>
                      Create Content Plan
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      {/* Trends Tab */}
      {currentTab === 4 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Subscriber Growth Trends
                </Typography>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Time Range</InputLabel>
                  <Select
                    value={timeRange}
                    onChange={(e) => setTimeRange(e.target.value)}
                    label="Time Range"
                  >
                    <MenuItem value="7 d">Last 7 Days</MenuItem>
                    <MenuItem value="30 d">Last 30 Days</MenuItem>
                    <MenuItem value="90 d">Last 90 Days</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  {selectedCompetitors.map((compId, index) => {
                    const comp = competitors.find(c => c.id === compId);
                    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];
                    return comp ? (
                      <Area
                        key={compId}
                        type="monotone"
                        dataKey={comp.channelName}
                        stroke={colors[index % colors.length]}
                        fill={colors[index % colors.length]}
                        fillOpacity={0.3}
                      />
                    ) : null;
                  })}
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};