import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Tooltip,
  Chip,
  LinearProgress,
  Alert,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Assessment,
  Download,
  Refresh,
  Info,
  CalendarToday,
  ShowChart,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
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
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format, subDays, startOfMonth, endOfMonth } from 'date-fns';
import { useRevenueData } from '../../hooks/useRevenueData';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

interface RevenueDashboardProps {
  userId?: number;
  channelId?: number;
  dateRange?: {
    start: Date;
    end: Date;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

export const RevenueDashboard: React.FC<RevenueDashboardProps> = ({
  userId,
  channelId,
  dateRange,
}) => {
  const [selectedPeriod, setSelectedPeriod] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area'>('line');
  const [breakdownType, setBreakdownType] = useState<string>('source');
  const [isExporting, setIsExporting] = useState(false);

  const {
    overview,
    trends,
    forecast,
    breakdown,
    channelRevenue,
    loading,
    error,
    refetch,
    exportData,
  } = useRevenueData({
    userId,
    channelId,
    dateRange,
    period: selectedPeriod,
    breakdownBy: breakdownType,
  });

  const handleExport = async (format: 'csv' | 'json') => {
    setIsExporting(true);
    try {
      await exportData(format);
    } finally {
      setIsExporting(false);
    }
  };

  const handlePeriodChange = (event: React.MouseEvent<HTMLElement>, newPeriod: string | null) => {
    if (newPeriod) {
      setSelectedPeriod(newPeriod as 'daily' | 'weekly' | 'monthly');
    }
  };

  const renderRevenueCard = (title: string, value: number | string, trend?: number, icon?: React.ReactNode) => (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography color="textSecondary" gutterBottom variant="body2">
            {title}
          </Typography>
          {icon}
        </Box>
        <Typography variant="h4" component="div" fontWeight="bold">
          {typeof value === 'number' ? formatCurrency(value) : value}
        </Typography>
        {trend !== undefined && (
          <Box display="flex" alignItems="center" mt={1}>
            {trend > 0 ? (
              <TrendingUp color="success" fontSize="small" />
            ) : (
              <TrendingDown color="error" fontSize="small" />
            )}
            <Typography
              variant="body2"
              color={trend > 0 ? 'success.main' : 'error.main'}
              ml={0.5}
            >
              {formatPercentage(Math.abs(trend))}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderTrendsChart = () => {
    if (!trends || trends.length === 0) return null;

    const ChartComponent = chartType === 'bar' ? BarChart : chartType === 'area' ? AreaChart : LineChart;
    const DataComponent = chartType === 'bar' ? Bar : chartType === 'area' ? Area : Line;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <ChartComponent data={trends}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="period"
            tickFormatter={(value) => format(new Date(value), 'MMM dd')}
          />
          <YAxis tickFormatter={(value) => `$${value}`} />
          <RechartsTooltip
            formatter={(value: number) => formatCurrency(value)}
            labelFormatter={(label) => format(new Date(label), 'PPP')}
          />
          <Legend />
          <DataComponent
            type="monotone"
            dataKey="revenue"
            stroke="#8884d8"
            fill="#8884d8"
            strokeWidth={2}
            name="Revenue"
          />
          {chartType === 'area' && (
            <Area
              type="monotone"
              dataKey="revenue"
              stroke="#8884d8"
              fillOpacity={0.3}
              fill="#8884d8"
            />
          )}
        </ChartComponent>
      </ResponsiveContainer>
    );
  };

  const renderBreakdownChart = () => {
    if (!breakdown || breakdown.length === 0) return null;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={breakdown}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="revenue"
            nameKey={breakdownType === 'source' ? 'source' : breakdownType === 'content_type' ? 'content_type' : 'time_period'}
          >
            {breakdown.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <RechartsTooltip formatter={(value: number) => formatCurrency(value)} />
        </PieChart>
      </ResponsiveContainer>
    );
  };

  const renderForecastChart = () => {
    if (!forecast || forecast.length === 0) return null;

    return (
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={forecast}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={(value) => format(new Date(value), 'MMM dd')}
          />
          <YAxis tickFormatter={(value) => `$${value}`} />
          <RechartsTooltip
            formatter={(value: number) => formatCurrency(value)}
            labelFormatter={(label) => format(new Date(label), 'PPP')}
          />
          <Area
            type="monotone"
            dataKey="predicted_revenue"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.6}
            name="Predicted Revenue"
          />
          <Area
            type="monotone"
            dataKey="confidence_upper"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.2}
            strokeDasharray="3 3"
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="confidence_lower"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.2}
            strokeDasharray="3 3"
            name="Lower Bound"
          />
        </AreaChart>
      </ResponsiveContainer>
    );
  };

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Revenue Dashboard
        </Typography>
        <Box display="flex" gap={2}>
          <Button
            startIcon={<Download />}
            variant="outlined"
            onClick={() => handleExport('csv')}
            disabled={isExporting}
          >
            Export CSV
          </Button>
          <Button
            startIcon={<Download />}
            variant="outlined"
            onClick={() => handleExport('json')}
            disabled={isExporting}
          >
            Export JSON
          </Button>
          <IconButton onClick={refetch} disabled={loading}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          {renderRevenueCard(
            'Total Revenue',
            overview?.total_revenue || 0,
            overview?.revenue_growth,
            <AttachMoney color="primary" />
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderRevenueCard(
            'Avg. Revenue/Video',
            overview?.average_revenue_per_video || 0,
            undefined,
            <Assessment color="secondary" />
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderRevenueCard(
            'CPM',
            `$${overview?.cpm?.toFixed(2) || '0.00'}`,
            overview?.cpm_trend,
            <ShowChart color="success" />
          )}
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          {renderRevenueCard(
            'RPM',
            `$${overview?.rpm?.toFixed(2) || '0.00'}`,
            overview?.rpm_trend,
            <BarChartIcon color="info" />
          )}
        </Grid>
      </Grid>

      {/* Revenue Trends */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Revenue Trends</Typography>
            <Box display="flex" gap={2}>
              <ToggleButtonGroup
                value={selectedPeriod}
                exclusive
                onChange={handlePeriodChange}
                size="small"
              >
                <ToggleButton value="daily">Daily</ToggleButton>
                <ToggleButton value="weekly">Weekly</ToggleButton>
                <ToggleButton value="monthly">Monthly</ToggleButton>
              </ToggleButtonGroup>
              <ToggleButtonGroup
                value={chartType}
                exclusive
                onChange={(e, val) => val && setChartType(val)}
                size="small"
              >
                <ToggleButton value="line">
                  <ShowChart />
                </ToggleButton>
                <ToggleButton value="bar">
                  <BarChartIcon />
                </ToggleButton>
                <ToggleButton value="area">
                  <Assessment />
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          </Box>
          {renderTrendsChart()}
        </CardContent>
      </Card>

      {/* Revenue Breakdown and Forecast */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">Revenue Breakdown</Typography>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <Select
                    value={breakdownType}
                    onChange={(e) => setBreakdownType(e.target.value)}
                  >
                    <MenuItem value="source">By Source</MenuItem>
                    <MenuItem value="content_type">By Content Type</MenuItem>
                    <MenuItem value="video_length">By Video Length</MenuItem>
                    <MenuItem value="time_of_day">By Time of Day</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              {renderBreakdownChart()}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">Revenue Forecast</Typography>
                <Tooltip title="Machine learning-based forecast with confidence intervals">
                  <IconButton size="small">
                    <Info />
                  </IconButton>
                </Tooltip>
              </Box>
              {renderForecastChart()}
              {forecast && (
                <Box mt={2}>
                  <Typography variant="body2" color="textSecondary">
                    Confidence: {formatPercentage(forecast.confidence || 0)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Estimated 7-day revenue: {formatCurrency(forecast.estimated_total || 0)}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Channel Revenue Table */}
      {channelRevenue && channelRevenue.length > 0 && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" mb={2}>Channel Performance</Typography>
            <Box sx={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e0e0e0' }}>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Channel</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>Revenue</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>Videos</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>Avg/Video</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>Views</th>
                  </tr>
                </thead>
                <tbody>
                  {channelRevenue.map((channel) => (
                    <tr key={channel.channel_id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '12px' }}>{channel.channel_name}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {formatCurrency(channel.revenue)}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>{channel.video_count}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {formatCurrency(channel.revenue / channel.video_count)}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {channel.total_views.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};