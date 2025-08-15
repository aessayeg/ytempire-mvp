/**
 * Cost Tracking Visualization Component
 * Comprehensive cost tracking and budget management interface
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Alert,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Stack
 } from '@mui/material';
import { 
  AttachMoney,
  TrendingUp,
  TrendingDown,
  Add,
  Edit,
  Refresh,
  Download,
  ShowChart,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon
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
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
 } from 'recharts';
import {  format, startOfMonth, subMonths  } from 'date-fns';
import {  api  } from '../../services/api';
import {  useOptimizedStore  } from '../../stores/optimizedStore';

// Types
interface CostCategory {
  name: string,
  amount: number,

  percentage: number,
  trend: number,

  color: string}

interface Budget {
  id: string,
  name: string,

  amount: number,
  spent: number,

  remaining: number,
  percentage: number,

  period: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  category?: string;
  alertThresholds: number[],
  status: 'healthy' | 'warning' | 'critical' | 'exceeded'}

interface CostItem {
  id: string,
  date: Date,

  category: string,
  service: string,

  amount: number,
  description: string;
  userId?: string;
  channelId?: string;
  videoId?: string;
}

interface CostForecast {
  date: string,
  predicted: number;
  actual?: number;
  confidenceLow: number,
  confidenceHigh: number}

const CATEGORY_COLORS = { 'openai_api': '#4 CAF50',
  'youtube_api': '#2196 F3',
  'elevenlabs_tts': '#FF9800',
  'storage': '#9 C27 B0',
  'compute': '#F44336',
  'bandwidth': '#00 BCD4',
  'database': '#795548',
  'third_party': '#607 D8 B',
  'other': '#9 E9 E9 E' };

export const CostVisualization: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'daily' | 'weekly' | 'monthly' | 'quarterly'>('monthly');
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'forecast'>('overview');
  const [loading, setLoading] = useState(true);
  const [categories, setCategories] = useState<CostCategory[]>([]);
  const [budgets, setBudgets] = useState<Budget[]>([]);
  const [costHistory, setCostHistory] = useState<any[]>([]);
  const [forecasts, setForecasts] = useState<CostForecast[]>([]);
  const [totalCost, setTotalCost] = useState(0);
  const [costTrend, setCostTrend] = useState(0);
  const [budgetDialogOpen, setBudgetDialogOpen] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<Budget | null>(null);
  const [alerts, setAlerts] = useState<any[]>([]);

  const { addNotification } = useOptimizedStore();

  // Fetch cost data
  const fetchCostData = async () => {
    try {
      setLoading(true);

      // Determine date range
      const endDate = new Date();
      let startDate: Date;
      
      switch (timeRange) {
        case 'daily':
          startDate = new Date();
          startDate.setHours(0, 0, 0, 0);
          break;
        case 'weekly':
          startDate = new Date();
          startDate.setDate(startDate.getDate() - 7);
          break;
        case 'monthly':
          startDate = startOfMonth(new Date());
          break;
        case 'quarterly':
          startDate = subMonths(new Date(), 3);
          break;
      }

      // Fetch cost summary
      const summaryResponse = await api.get('/costs/summary', { params: {,
  start_date: format(startDate, 'yyyy-MM-dd'),
          end_date: format(endDate, 'yyyy-MM-dd') }
      });

      const summary = summaryResponse.data;
      setTotalCost(summary.total_cost);
      setCostTrend(summary.trend === 'increasing' ? 15 : summary.trend === 'decreasing' ? -10 : 0);

      // Process categories
      const categoryData: CostCategory[] = Object.entries(summary.category_breakdown || {}).map(
        ([name, amount]: [string, any]) => ({ name,
          amount: Number(amount),
          percentage: (Number(amount) / summary.total_cost) * 100,
          trend: Math.random() * 20 - 10, // Mock trend, color: CATEGORY_COLORS[name] || '#9 E9 E9 E' })
      );
      setCategories(categoryData);

      // Fetch budgets
      const budgetsResponse = await api.get('/costs/budgets');
      setBudgets(budgetsResponse.data.map((b: React.ChangeEvent<HTMLInputElement>) => ({ ...b,
        spent: Math.random() * b.amount, // Mock spent amount
        remaining: b.amount - (Math.random() * b.amount),
        percentage: (Math.random() * b.amount / b.amount) * 100,
        status: b.percentage > 100 ? 'exceeded' : b.percentage > 90 ? 'critical' : b.percentage > 75 ? 'warning' : 'healthy' })));

      // Generate mock cost history
      const history = [];
      for (let i = 29; i >= 0; i--) { const date = new Date();
        date.setDate(date.getDate() - i);
        history.push({
          date: format(date, 'MMM dd'),
          cost: Math.random() * 200 + 50,
          openai: Math.random() * 80 + 20,
          youtube: Math.random() * 30 + 10,
          storage: Math.random() * 20 + 5,
          compute: Math.random() * 40 + 10 });
}
      setCostHistory(history);

      // Generate mock forecasts
      const forecastData: CostForecast[] = [];
      for (let i = 0; i < 7; i++) { const date = new Date();
        date.setDate(date.getDate() + i);
        const predicted = Math.random() * 200 + 100;
        forecastData.push({
          date: format(date, 'MMM dd'),
          predicted,
          actual: i === 0 ? predicted * 0.95 : undefined,
          confidenceLow: predicted * 0.8,
          confidenceHigh: predicted * 1.2 });
}
      setForecasts(forecastData);

      // Fetch alerts
      const alertsResponse = await api.get('/costs/alerts');
      setAlerts(alertsResponse.data || []);

      setLoading(false)} catch (error) { console.error('Failed to fetch cost, data:', error);
      addNotification({
        type: 'error',
        message: 'Failed to load cost data' });
      setLoading(false)}
  };

  useEffect(() => {
    fetchCostData()}, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  // Calculate total budget status
  const budgetStatus = useMemo(() => {
     if (!budgets.length) return null;

    const totalBudget = budgets.reduce((sum, b) => sum + b.amount, 0);
    const totalSpent = budgets.reduce((sum, b) => sum + b.spent, 0);
    const percentage = (totalSpent / totalBudget) * 100;

    return {
      totalBudget,
      totalSpent,
      totalRemaining: totalBudget - totalSpent,
      percentage,
      status: percentage > 100 ? 'exceeded' : percentage > 90 ? 'critical' : percentage > 75 ? 'warning' : 'healthy',

  }, [budgets]);

  // Handle budget creation/edit
  const handleSaveBudget = async (budget: Partial<Budget>) => {
    try {
      if (selectedBudget) {
        await api.put(`/costs/budgets/${selectedBudget.id}`, budget)} else {
        await api.post('/costs/budgets', budget)}
      
      fetchCostData();
      setBudgetDialogOpen(false);
      setSelectedBudget(null);
      
      addNotification({ type: 'success',
        message: selectedBudget ? 'Budget updated' : 'Budget created' });
} catch (error) { addNotification({
        type: 'error',
        message: 'Failed to save budget' });
}
  };

  // Render category card
  const renderCategoryCard = (category: CostCategory) => (
    <Card key={category.name} variant="outlined">
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="body2" color="text.secondary">
            {category.name.replace('_', ' ').toUpperCase()}
          </Typography>
          <Box display="flex" alignItems="center">
            {category.trend > 0 ? (
              <TrendingUp color="error" fontSize="small" />
            ) : (
              <TrendingDown color="success" fontSize="small" />
            )}
            <Typography variant="caption" color={category.trend > 0 ? 'error' : 'success'}>
              {category.trend > 0 ? '+' : ''}{category.trend.toFixed(1)}%
            </Typography>
          </Box>
        </Box>
        <Typography variant="h5" gutterBottom>
          ${category.amount.toFixed(2)}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={category.percentage}
          sx={ {
            height: 6,
            borderRadius: 3,
            backgroundColor: '#e0 e0 e0',
            '& .MuiLinearProgress-bar': {
              backgroundColor: category.color }
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
          {category.percentage.toFixed(1)}% of total
        </Typography>
      </CardContent>
    </Card>
  );

  // Render budget card
  const renderBudgetCard = (budget: Budget) => {
    const getStatusColor = () => {
      switch (budget.status) {
        case 'healthy': return 'success';
        case 'warning': return 'warning';
        case 'critical': return 'error';
        case 'exceeded': return 'error';
        default: return 'info'}
    };

    return (
    <>
      <Card key={budget.id} variant="outlined">
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">{budget.name}</Typography>
      <Chip
              label={budget.status}
              color={getStatusColor()}
              size="small"
            />
          </Box>
          <Box mb={2}>
            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2">
                ${budget.spent.toFixed(2)} / ${budget.amount.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {budget.percentage.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={Math.min(budget.percentage, 100)}
              color={getStatusColor()}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
          <Box display="flex" justifyContent="space-between">
            <Typography variant="caption" color="text.secondary">
              Remaining: ${budget.remaining.toFixed(2)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {budget.period}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </>
  )};

  return (
    <>
      <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4">Cost Tracking & Budgets</Typography>
      <Typography variant="body2" color="text.secondary">
            Monitor and manage your platform costs
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Period</InputLabel>
            <Select
              value={timeRange}
              label="Period"
              onChange={(e) => setTimeRange(e.target.value as any)}
            >
              <MenuItem value="daily">Daily</MenuItem>
              <MenuItem value="weekly">Weekly</MenuItem>
              <MenuItem value="monthly">Monthly</MenuItem>
              <MenuItem value="quarterly">Quarterly</MenuItem>
            </Select>
          </FormControl>
          <Button
            startIcon={<Add />}
            variant="contained"
            onClick={() => {
              setSelectedBudget(null</>
  );
              setBudgetDialogOpen(true)}}
          >
            Add Budget
          </Button>
          <Button startIcon={<Download />} variant="outlined">
            Export
          </Button>
          <IconButton onClick={fetchCostData}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Cost Alerts */}
      {alerts.length > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Cost Alerts
          </Typography>
          {alerts.map((alert, index) => (
            <Typography key={index} variant="body2">
              • {alert.message}
            </Typography>
          ))}
        </Alert>
      )}
      {/* Summary Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Total Cost ({timeRange})
                  </Typography>
                  <Typography variant="h4">
                    ${totalCost.toFixed(2)}
                  </Typography>
                  <Box display="flex" alignItems="center" mt={1}>
                    {costTrend > 0 ? (
                      <TrendingUp color="error" fontSize="small" />
                    ) : costTrend < 0 ? (
                      <TrendingDown color="success" fontSize="small" />
                    ) : null}
                    <Typography
                      variant="body2"
                      color={costTrend > 0 ? 'error' : 'success'}
                    >
                      {costTrend > 0 ? '+' : ''}{costTrend.toFixed(1)}% vs last period
                    </Typography>
                  </Box>
                </Box>
                <AttachMoney sx={{ fontSize: 40, color: 'primary.main', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {budgetStatus && (
          <>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Budget Utilized
                  </Typography>
                  <Typography variant="h4">
                    {budgetStatus.percentage.toFixed(0)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(budgetStatus.percentage, 100)}
                    color={
                      budgetStatus.status === 'healthy' ? 'success' :
                      budgetStatus.status === 'warning' ? 'warning' : 'error'
                    }
                    sx={{ mt: 2, height: 8, borderRadius: 4 }}
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total Budget
                  </Typography>
                  <Typography variant="h4">
                    ${budgetStatus.totalBudget.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" mt={1}>
                    {budgets.length} active budgets
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Remaining Budget
                  </Typography>
                  <Typography variant="h4" color={budgetStatus.totalRemaining < 0 ? 'error' : 'inherit'}>
                    ${budgetStatus.totalRemaining.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" mt={1}>
                    {Math.ceil(budgetStatus.totalRemaining / (totalCost / 30))} days at current rate
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>

      {/* View Mode Tabs */}
      <Paper sx={{ mb: 3 }}>
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, value) => value && setViewMode(value)}
          sx={{ p: 1 }}
        >
          <ToggleButton value="overview">
            <PieChartIcon sx={{ mr: 1 }} />
            Overview
          </ToggleButton>
          <ToggleButton value="detailed">
            <BarChartIcon sx={{ mr: 1 }} />
            Detailed
          </ToggleButton>
          <ToggleButton value="forecast">
            <ShowChart sx={{ mr: 1 }} />
            Forecast
          </ToggleButton>
        </ToggleButtonGroup>
      </Paper>

      {/* Content based on view mode */}
      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {viewMode === 'overview' && (
            <Grid container spacing={3}>
              {/* Category Breakdown */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Cost by Category
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={categories}
                        dataKey="amount"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label={(entry) => `${entry.name}: $${entry.amount.toFixed(0)}`}
                      >
                        {categories.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <ChartTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* Cost Trend */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Cost Trend
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={costHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <ChartTooltip />
                      <Area
                        type="monotone"
                        dataKey="cost"
                        stroke="#8884 d8"
                        fill="#8884d8"
                        fillOpacity={0.6}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* Category Cards */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Categories
                </Typography>
                <Grid container spacing={2}>
                  {categories.map((category) => (
                    <Grid item xs={12} sm={6} md={3} key={category.name}>
                      {renderCategoryCard(category)}
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              {/* Budget Cards */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Budgets
                </Typography>
                <Grid container spacing={2}>
                  {budgets.map((budget) => (
                    <Grid item xs={12} sm={6} md={4} key={budget.id}>
                      {renderBudgetCard(budget)}
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </Grid>
          )}
          {viewMode === 'detailed' && (
            <Grid container spacing={3}>
              {/* Stacked Cost Chart */}
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Cost Breakdown Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={costHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <ChartTooltip />
                      <Legend />
                      <Bar dataKey="openai" stackId="a" fill="#4 CAF50" />
                      <Bar dataKey="youtube" stackId="a" fill="#2196 F3" />
                      <Bar dataKey="storage" stackId="a" fill="#9 C27 B0" />
                      <Bar dataKey="compute" stackId="a" fill="#F44336" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* Service Comparison Radar */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Service Utilization
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={categories}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="name" />
                      <PolarRadiusAxis />
                      <Radar
                        name="Cost"
                        dataKey="amount"
                        stroke="#8884 d8"
                        fill="#8884d8"
                        fillOpacity={0.6}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* Cost Table */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Recent Costs
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Service</TableCell>
                          <TableCell>Category</TableCell>
                          <TableCell align="right">Amount</TableCell>
                          <TableCell>Date</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {[1, 2, 3, 4, 5].map((i) => (
                          <TableRow key={i}>
                            <TableCell>Service {i}</TableCell>
                            <TableCell>
                              <Chip
                                label="API"
                                size="small"
                                sx={{ backgroundColor: CATEGORY_COLORS.openai_api + '20' }}
                              />
                            </TableCell>
                            <TableCell align="right">
                              ${(Math.random() * 50.toFixed(2)}
                            </TableCell>
                            <TableCell>
                              {format(new Date(), 'MMM dd, HH:mm')}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>
              </Grid>
            </Grid>
          )}
          {viewMode === 'forecast' && (
            <Grid container spacing={3}>
              {/* Forecast Chart */}
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Cost Forecast
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={forecasts}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <ChartTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#8884 d8"
                        strokeWidth={2}
                        name="Predicted"
                      />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#82 ca9 d"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        name="Actual"
                      />
                      <Area
                        type="monotone"
                        dataKey="confidenceHigh"
                        stroke="none"
                        fill="#8884d8"
                        fillOpacity={0.1}
                      />
                      <Area
                        type="monotone"
                        dataKey="confidenceLow"
                        stroke="none"
                        fill="#ffffff"
                        fillOpacity={1}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              {/* Forecast Summary */}
              <Grid item xs={12}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Next 7 Days Forecast
                        </Typography>
                        <Typography variant="h4">
                          ${forecasts.reduce((sum, f) => sum + f.predicted, 0.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          ± ${((forecasts[0]?.predicted || 0) * 0.2.toFixed(2)} confidence interval
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Monthly Projection
                        </Typography>
                        <Typography variant="h4">
                          ${(totalCost * 30 / 7.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Based on current usage patterns
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Cost Optimization Potential
                        </Typography>
                        <Typography variant="h4" color="success.main">
                          ${(totalCost * 0.15.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          ~15% possible savings identified
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          )}
        </>
      )}
      {/* Budget Dialog */}
      <Dialog open={budgetDialogOpen} onClose={() => setBudgetDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedBudget ? 'Edit Budget' : 'Create Budget'}
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              label="Budget Name"
              fullWidth
              defaultValue={selectedBudget?.name}
            />
            <TextField
              label="Amount"
              type="number"
              fullWidth
              defaultValue={selectedBudget?.amount}
              InputProps={ {
                startAdornment: '$' }}
            />
            <FormControl fullWidth>
              <InputLabel>Period</InputLabel>
              <Select defaultValue={selectedBudget?.period || 'monthly'}>
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
                <MenuItem value="quarterly">Quarterly</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Category (Optional)</InputLabel>
              <Select defaultValue={selectedBudget?.category || ''}>
                <MenuItem value="">All Categories</MenuItem>
                {Object.keys(CATEGORY_COLORS).map((cat) => (
                  <MenuItem key={cat} value={cat}>
                    {cat.replace('_', ' ').toUpperCase()}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Alert Thresholds (%)"
              fullWidth
              defaultValue="50,75,90,100"
              helperText="Comma-separated percentages for alerts"
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBudgetDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => handleSaveBudget({});
} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )};