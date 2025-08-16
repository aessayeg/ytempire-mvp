import React, { useState, useEffect } from 'react';
import Grid from '@mui/material/Grid2';
import { 
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Chip,
  Avatar,
  LinearProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  ToggleButton,
  ToggleButtonGroup,
  useTheme
} from '@mui/material';
import { 
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Visibility,
  Comment,
  ThumbUp,
  PlayCircle,
  Download,
  Refresh,
  CompareArrows,
  BarChart as BarChartIcon,
  ShowChart,
  Timeline,
  YouTube
} from '@mui/icons-material';
import { 
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Scatter,
  ScatterChart,
  ZAxis
} from 'recharts';
import { format, subDays, parseISO } from 'date-fns';

interface RevenueData {
  date: string;
  revenue: number;
  adRevenue: number;
  membershipRevenue: number;
  sponsorshipRevenue: number;
  views: number;
  rpm: number;
}

interface VideoPerformance {
  id: string;
  title: string;
  thumbnail: string;
  publishDate: Date;
  views: number;
  watchTime: number;
  likes: number;
  comments: number;
  shares: number;
  ctr: number;
  avd: number;
  revenue: number;
  impressions: number;
  retention: number[];
}

interface ChannelComparison {
  channelId: string;
  channelName: string;
  subscribers: number;
  totalViews: number;
  totalRevenue: number;
  avgViews: number;
  avgEngagement: number;
  videosPublished: number;
  growthRate: number;
  health: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <React.Fragment>
      <div hidden={value !== index}>
        {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
      </div>
    </React.Fragment>
  );
};

export const AnalyticsDashboard: React.FC = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('7d');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area'>('line');
  const [comparisonMode, setComparisonMode] = useState(false);

  // Mock data - replace with API calls
  const [revenueData, setRevenueData] = useState<RevenueData[]>([]);
  const [videoPerformance, setVideoPerformance] = useState<VideoPerformance[]>([]);
  const [channelComparison, setChannelComparison] = useState<ChannelComparison[]>([]);

  useEffect(() => {
    // Generate mock revenue data
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    const revenue = Array.from({ length: days }, (_, i) => {
      const date = subDays(new Date(), days - 1 - i);
      const baseRevenue = 100 + Math.random() * 50;
      return {
        date: format(date, 'yyyy-MM-dd'),
        revenue: baseRevenue,
        adRevenue: baseRevenue * 0.6,
        membershipRevenue: baseRevenue * 0.25,
        sponsorshipRevenue: baseRevenue * 0.15,
        views: Math.floor(5000 + Math.random() * 10000),
        rpm: 3 + Math.random() * 2
      };
    });
    setRevenueData(revenue);

    // Generate mock video performance data
    const videos: VideoPerformance[] = Array.from({ length: 20 }, (_, i) => ({
      id: `video-${i}`,
      title: `Video Title ${i + 1}: Amazing Content That Gets Views`,
      thumbnail: `/thumbnail-${i}.jpg`,
      publishDate: subDays(new Date(), Math.floor(Math.random() * 30)),
      views: Math.floor(1000 + Math.random() * 50000),
      watchTime: Math.floor(100 + Math.random() * 5000),
      likes: Math.floor(50 + Math.random() * 2000),
      comments: Math.floor(10 + Math.random() * 500),
      shares: Math.floor(5 + Math.random() * 200),
      ctr: 2 + Math.random() * 8,
      avd: 30 + Math.random() * 40,
      revenue: 10 + Math.random() * 200,
      impressions: Math.floor(10000 + Math.random() * 100000),
      retention: Array.from({ length: 10 }, () => 100 - Math.random() * 50)
    }));
    setVideoPerformance(videos);

    // Generate mock channel comparison data
    const channels: ChannelComparison[] = [
      {
        channelId: 'ch1',
        channelName: 'Tech Insights Daily',
        subscribers: 125000,
        totalViews: 8500000,
        totalRevenue: 15000,
        avgViews: 25000,
        avgEngagement: 4.5,
        videosPublished: 342,
        growthRate: 12.5,
        health: 92
      },
      {
        channelId: 'ch2',
        channelName: 'AI Explained',
        subscribers: 89000,
        totalViews: 5200000,
        totalRevenue: 9800,
        avgViews: 18000,
        avgEngagement: 5.2,
        videosPublished: 289,
        growthRate: 15.3,
        health: 88
      },
      {
        channelId: 'ch3',
        channelName: 'Future Tech',
        subscribers: 67000,
        totalViews: 3100000,
        totalRevenue: 6500,
        avgViews: 12000,
        avgEngagement: 3.8,
        videosPublished: 258,
        growthRate: 8.7,
        health: 75
      },
      {
        channelId: 'ch4',
        channelName: 'Coding Masters',
        subscribers: 234000,
        totalViews: 12000000,
        totalRevenue: 28000,
        avgViews: 35000,
        avgEngagement: 6.1,
        videosPublished: 342,
        growthRate: 18.9,
        health: 95
      }
    ];
    setChannelComparison(channels);

    setLoading(false);
  }, [timeRange]);

  // Calculate summary metrics
  const totalRevenue = revenueData.reduce((sum, d) => sum + d.revenue, 0);
  const totalViews = revenueData.reduce((sum, d) => sum + d.views, 0);
  const avgRPM = totalViews > 0 ? (totalRevenue / totalViews) * 1000 : 0;
  const revenueGrowth = revenueData.length > 1
    ? ((revenueData[revenueData.length - 1].revenue - revenueData[0].revenue) / revenueData[0].revenue) * 100
    : 0;

  // Chart colors
  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main
  ];

  // Revenue breakdown for pie chart
  const revenueBreakdown = [
    { name: 'Ad Revenue', value: totalRevenue * 0.6, percentage: 60 },
    { name: 'Memberships', value: totalRevenue * 0.25, percentage: 25 },
    { name: 'Sponsorships', value: totalRevenue * 0.15, percentage: 15 }
  ];

  // Channel comparison radar data
  const radarData = channelComparison.map(channel => ({
    channel: channel.channelName,
    subscribers: (channel.subscribers / 250000) * 100,
    views: (channel.avgViews / 40000) * 100,
    engagement: (channel.avgEngagement / 10) * 100,
    revenue: (channel.totalRevenue / 30000) * 100,
    growth: (channel.growthRate / 20) * 100,
    health: channel.health
  }));

  const renderRevenueTab = () => (
    <Grid container spacing={3}>
      {/* Summary Cards */}
      <Grid size={{ xs: 12, sm: 6, md: 3 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                <AttachMoney />
              </Avatar>
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  ${totalRevenue.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Revenue
                </Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {revenueGrowth > 0 ? (
                <TrendingUp color="success" fontSize="small" />
              ) : (
                <TrendingDown color="error" fontSize="small" />
              )}
              <Typography
                variant="body2"
                color={revenueGrowth > 0 ? 'success.main' : 'error.main'}
              >
                {revenueGrowth > 0 ? '+' : ''}{revenueGrowth.toFixed(1)}% vs previous period
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, sm: 6, md: 3 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                <Visibility />
              </Avatar>
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  {(totalViews / 1000).toFixed(1)}K
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Views
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Avg per day: {(totalViews / revenueData.length).toFixed(0)}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, sm: 6, md: 3 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                <Timeline />
              </Avatar>
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  ${avgRPM.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average RPM
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Revenue per 1000 views
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, sm: 6, md: 3 }}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'info.main', mr: 2 }}>
                <BarChartIcon />
              </Avatar>
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  ${(totalRevenue / revenueData.length).toFixed(2)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Daily Average
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Based on {timeRange} data
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Revenue Chart */}
      <Grid size={{ xs: 12, md: 8 }}>
        <Card>
          <CardHeader
            title="Revenue Trend"
            action={
              <ToggleButtonGroup
                value={chartType}
                exclusive
                onChange={(_, v) => v && setChartType(v)}
                size="small"
              >
                <ToggleButton value="line">
                  <ShowChart />
                </ToggleButton>
                <ToggleButton value="bar">
                  <BarChartIcon />
                </ToggleButton>
                <ToggleButton value="area">
                  <Timeline />
                </ToggleButton>
              </ToggleButtonGroup>
            }
          />
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              {chartType === 'line' ? (
                <LineChart data={revenueData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(date) => format(parseISO(date), 'MM/dd')} />
                  <YAxis />
                  <RechartsTooltip
                    labelFormatter={(date) => format(parseISO(date as string), 'PPP')}
                    formatter={(value: number) => `$${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="revenue" stroke={COLORS[0]} name="Total Revenue" strokeWidth={2} />
                  <Line type="monotone" dataKey="adRevenue" stroke={COLORS[1]} name="Ad Revenue" />
                  <Line type="monotone" dataKey="membershipRevenue" stroke={COLORS[2]} name="Memberships" />
                  <Line type="monotone" dataKey="sponsorshipRevenue" stroke={COLORS[3]} name="Sponsorships" />
                </LineChart>
              ) : chartType === 'bar' ? (
                <BarChart data={revenueData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(date) => format(parseISO(date), 'MM/dd')} />
                  <YAxis />
                  <RechartsTooltip
                    labelFormatter={(date) => format(parseISO(date as string), 'PPP')}
                    formatter={(value: number) => `$${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Bar dataKey="adRevenue" stackId="a" fill={COLORS[1]} name="Ad Revenue" />
                  <Bar dataKey="membershipRevenue" stackId="a" fill={COLORS[2]} name="Memberships" />
                  <Bar dataKey="sponsorshipRevenue" stackId="a" fill={COLORS[3]} name="Sponsorships" />
                </BarChart>
              ) : (
                <AreaChart data={revenueData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(date) => format(parseISO(date), 'MM/dd')} />
                  <YAxis />
                  <RechartsTooltip
                    labelFormatter={(date) => format(parseISO(date as string), 'PPP')}
                    formatter={(value: number) => `$${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="adRevenue" stackId="1" stroke={COLORS[1]} fill={COLORS[1]} name="Ad Revenue" />
                  <Area type="monotone" dataKey="membershipRevenue" stackId="1" stroke={COLORS[2]} fill={COLORS[2]} name="Memberships" />
                  <Area type="monotone" dataKey="sponsorshipRevenue" stackId="1" stroke={COLORS[3]} fill={COLORS[3]} name="Sponsorships" />
                </AreaChart>
              )}
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Revenue Breakdown Pie Chart */}
      <Grid size={{ xs: 12, md: 4 }}>
        <Card>
          <CardHeader title="Revenue Breakdown" />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={revenueBreakdown}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.percentage}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {revenueBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
              </PieChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 2 }}>
              {revenueBreakdown.map((item, index) => (
                <Box key={item.name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: COLORS[index % COLORS.length],
                      mr: 1
                    }}
                  />
                  <Typography variant="body2" sx={{ flex: 1 }}>
                    {item.name}
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    ${item.value.toFixed(2)}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* RPM Trend */}
      <Grid size={12}>
        <Card>
          <CardHeader title="RPM Trend" />
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={revenueData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={(date) => format(parseISO(date), 'MM/dd')} />
                <YAxis />
                <RechartsTooltip
                  labelFormatter={(date) => format(parseISO(date as string), 'PPP')}
                  formatter={(value: number) => `$${value.toFixed(2)}`}
                />
                <Line type="monotone" dataKey="rpm" stroke={theme.palette.primary.main} name="RPM" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderVideoPerformanceTab = () => (
    <Grid container spacing={3}>
      {/* Top Performing Videos Table */}
      <Grid size={12}>
        <Card>
          <CardHeader
            title="Video Performance Metrics"
            action={
              <Button startIcon={<Download />} variant="outlined" size="small">
                Export
              </Button>
            }
          />
          <CardContent>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Video</TableCell>
                    <TableCell align="right">
                      <TableSortLabel active direction="desc">
                        Views
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">Watch Time (hrs)</TableCell>
                    <TableCell align="right">CTR %</TableCell>
                    <TableCell align="right">AVD %</TableCell>
                    <TableCell align="right">Engagement</TableCell>
                    <TableCell align="right">Revenue</TableCell>
                    <TableCell align="right">Performance</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {videoPerformance.slice(0, 10).map((video) => {
                    const engagementRate = ((video.likes + video.comments + video.shares) / video.views) * 100;
                    const performanceScore = (video.ctr * 0.3 + video.avd * 0.3 + engagementRate * 0.2 + (video.revenue / 100) * 0.2) * 10;
                    
                    return (
                      <TableRow key={video.id}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Avatar variant="rounded" sx={{ width: 60, height: 34 }}>
                              <PlayCircle />
                            </Avatar>
                            <Box>
                              <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                                {video.title}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {format(video.publishDate, 'MMM dd, yyyy')}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight="bold">
                            {video.views.toLocaleString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          {(video.watchTime / 60).toFixed(1)}
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${video.ctr.toFixed(1)}%`}
                            size="small"
                            color={video.ctr > 5 ? 'success' : video.ctr > 3 ? 'warning' : 'error'}
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${video.avd.toFixed(0)}%`}
                            size="small"
                            color={video.avd > 50 ? 'success' : video.avd > 30 ? 'warning' : 'error'}
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <ThumbUp fontSize="small" />
                            <Typography variant="caption">{video.likes}</Typography>
                            <Comment fontSize="small" />
                            <Typography variant="caption">{video.comments}</Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight="bold" color="success.main">
                            ${video.revenue.toFixed(2)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={Math.min(100, performanceScore)}
                              sx={{ width: 60, height: 6 }}
                              color={performanceScore > 70 ? 'success' : performanceScore > 40 ? 'warning' : 'error'}
                            />
                            <Typography variant="caption">
                              {performanceScore.toFixed(0)}
                            </Typography>
                          </Box>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Performance Distribution Charts */}
      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardHeader title="Views vs Revenue Correlation" />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="views" name="Views" />
                <YAxis dataKey="revenue" name="Revenue" />
                <ZAxis dataKey="engagement" range={[50, 400]} />
                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter
                  name="Videos"
                  data={videoPerformance.map(v => ({
                    views: v.views,
                    revenue: v.revenue,
                    engagement: (v.likes + v.comments) / v.views * 100
                  }))}
                  fill={theme.palette.primary.main}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardHeader title="Engagement Metrics Distribution" />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={[
                  { range: '0-10%', videos: videoPerformance.filter(v => v.ctr < 2).length },
                  { range: '2-4%', videos: videoPerformance.filter(v => v.ctr >= 2 && v.ctr < 4).length },
                  { range: '4-6%', videos: videoPerformance.filter(v => v.ctr >= 4 && v.ctr < 6).length },
                  { range: '6-8%', videos: videoPerformance.filter(v => v.ctr >= 6 && v.ctr < 8).length },
                  { range: '8%+', videos: videoPerformance.filter(v => v.ctr >= 8).length }
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="videos" fill={theme.palette.primary.main} name="Number of Videos" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderChannelComparisonTab = () => (
    <Grid container spacing={3}>
      {/* Channel Comparison Table */}
      <Grid size={12}>
        <Card>
          <CardHeader
            title="Channel Performance Comparison"
            action={
              <FormControlLabel
                control={
                  <Switch
                    checked={comparisonMode}
                    onChange={(e) => setComparisonMode(e.target.checked)}
                  />
                }
                label="Compare Mode"
              />
            }
          />
          <CardContent>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Channel</TableCell>
                    <TableCell align="right">Subscribers</TableCell>
                    <TableCell align="right">Total Views</TableCell>
                    <TableCell align="right">Avg Views</TableCell>
                    <TableCell align="right">Revenue</TableCell>
                    <TableCell align="right">Engagement</TableCell>
                    <TableCell align="right">Growth Rate</TableCell>
                    <TableCell align="right">Health Score</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {channelComparison.map((channel) => (
                    <TableRow key={channel.channelId}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                            <YouTube fontSize="small" />
                          </Avatar>
                          <Typography variant="body2" fontWeight="medium">
                            {channel.channelName}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {(channel.subscribers / 1000).toFixed(1)}K
                      </TableCell>
                      <TableCell align="right">
                        {(channel.totalViews / 1000000).toFixed(1)}M
                      </TableCell>
                      <TableCell align="right">
                        {(channel.avgViews / 1000).toFixed(1)}K
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="bold" color="success.main">
                          ${channel.totalRevenue.toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${channel.avgEngagement.toFixed(1)}%`}
                          size="small"
                          color={channel.avgEngagement > 5 ? 'success' : channel.avgEngagement > 3 ? 'warning' : 'error'}
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          {channel.growthRate > 0 ? (
                            <TrendingUp color="success" fontSize="small" />
                          ) : (
                            <TrendingDown color="error" fontSize="small" />
                          )}
                          <Typography
                            variant="body2"
                            color={channel.growthRate > 0 ? 'success.main' : 'error.main'}
                          >
                            {channel.growthRate > 0 ? '+' : ''}{channel.growthRate.toFixed(1)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={channel.health}
                            sx={{ width: 60, height: 6 }}
                            color={channel.health > 80 ? 'success' : channel.health > 60 ? 'warning' : 'error'}
                          />
                          <Typography variant="caption">
                            {channel.health}%
                          </Typography>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Channel Comparison Radar Chart */}
      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardHeader title="Channel Performance Radar" />
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="channel" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                {channelComparison.map((channel, index) => (
                  <Radar
                    key={channel.channelId}
                    name={channel.channelName}
                    dataKey={channel.channelName.toLowerCase().replace(/\s+/g, '')}
                    stroke={COLORS[index % COLORS.length]}
                    fill={COLORS[index % COLORS.length]}
                    fillOpacity={0.3}
                  />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Channel Growth Comparison */}
      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardHeader title="Channel Growth Comparison" />
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={channelComparison}
                layout="horizontal"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="channelName" type="category" width={100} />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="growthRate" fill={theme.palette.primary.main} name="Growth Rate %" />
                <Bar dataKey="health" fill={theme.palette.success.main} name="Health Score" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Revenue per Channel */}
      <Grid size={12}>
        <Card>
          <CardHeader title="Revenue Distribution by Channel" />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={channelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channelName" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <RechartsTooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="totalRevenue" fill={theme.palette.primary.main} name="Total Revenue ($)" />
                <Line yAxisId="right" type="monotone" dataKey="videosPublished" stroke={theme.palette.secondary.main} name="Videos Published" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Analytics Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              label="Time Range"
            >
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
              <MenuItem value="90d">Last 90 Days</MenuItem>
            </Select>
          </FormControl>
          <Button startIcon={<Refresh />} variant="outlined">
            Refresh
          </Button>
          <Button startIcon={<Download />} variant="contained">
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Tabs */}
      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 3 }}>
        <Tab label="Revenue Analytics" icon={<AttachMoney />} iconPosition="start" />
        <Tab label="Video Performance" icon={<PlayCircle />} iconPosition="start" />
        <Tab label="Channel Comparison" icon={<CompareArrows />} iconPosition="start" />
      </Tabs>

      {/* Tab Panels */}
      <TabPanel value={tabValue} index={0}>
        {renderRevenueTab()}
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        {renderVideoPerformanceTab()}
      </TabPanel>
      <TabPanel value={tabValue} index={2}>
        {renderChannelComparisonTab()}
      </TabPanel>
    </Box>
  );
};