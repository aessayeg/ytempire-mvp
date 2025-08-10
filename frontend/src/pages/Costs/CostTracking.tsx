/**
 * Cost Tracking Screen Component
 * Comprehensive cost monitoring and budget management for YouTube operations
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
  Tooltip,
} from '@mui/material';
import {
  AttachMoney,
  TrendingUp,
  TrendingDown,
  Warning,
  Add,
  Edit,
  Delete,
  Refresh,
  Download,
  Upload,
  Notifications,
  AccountBalance,
  CreditCard,
  Receipt,
  Analytics,
  SmartToy,
  CloudUpload,
  VideoLibrary,
  Mic,
  Image as ImageIcon,
  Storage,
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
  ComposedChart,
} from 'recharts';
import { format, subDays, startOfMonth, endOfMonth, isWithinInterval } from 'date-fns';

// Mock data for cost tracking
const mockCostData = [
  { date: '2024-01-01', aiCosts: 120, infrastructure: 45, storage: 15, voiceover: 85, total: 265 },
  { date: '2024-01-02', aiCosts: 135, infrastructure: 48, storage: 16, voiceover: 92, total: 291 },
  { date: '2024-01-03', aiCosts: 145, infrastructure: 50, storage: 18, voiceover: 88, total: 301 },
  { date: '2024-01-04', aiCosts: 158, infrastructure: 52, storage: 17, voiceover: 95, total: 322 },
  { date: '2024-01-05', aiCosts: 167, infrastructure: 55, storage: 19, voiceover: 110, total: 351 },
  { date: '2024-01-06', aiCosts: 142, infrastructure: 47, storage: 16, voiceover: 78, total: 283 },
  { date: '2024-01-07', aiCosts: 189, infrastructure: 58, storage: 22, voiceover: 125, total: 394 },
];

const mockCostCategories = [
  {
    id: 1,
    name: 'AI Content Generation',
    icon: <SmartToy color="primary" />,
    currentSpend: 3250,
    budgetLimit: 4000,
    usage: 81.25,
    change: +12.5,
    subcategories: [
      { name: 'GPT-4 API', cost: 1850, usage: '2.5M tokens' },
      { name: 'Claude API', cost: 980, usage: '1.8M tokens' },
      { name: 'DALL-E Image Gen', cost: 420, usage: '1.2K images' },
    ],
  },
  {
    id: 2,
    name: 'Voice Synthesis',
    icon: <Mic color="secondary" />,
    currentSpend: 1450,
    budgetLimit: 2000,
    usage: 72.5,
    change: +8.7,
    subcategories: [
      { name: 'ElevenLabs API', cost: 890, usage: '45 hours' },
      { name: 'Azure Speech', cost: 380, usage: '32 hours' },
      { name: 'Google TTS', cost: 180, usage: '18 hours' },
    ],
  },
  {
    id: 3,
    name: 'Infrastructure',
    icon: <CloudUpload color="info" />,
    currentSpend: 890,
    budgetLimit: 1200,
    usage: 74.17,
    change: +5.2,
    subcategories: [
      { name: 'GCP Compute', cost: 450, usage: '850 hours' },
      { name: 'AWS Storage', cost: 280, usage: '2.5 TB' },
      { name: 'Cloudflare CDN', cost: 160, usage: '8.2 TB transfer' },
    ],
  },
  {
    id: 4,
    name: 'Video Processing',
    icon: <VideoLibrary color="success" />,
    currentSpend: 680,
    budgetLimit: 1000,
    usage: 68,
    change: -2.1,
    subcategories: [
      { name: 'FFmpeg Cloud', cost: 420, usage: '156 videos' },
      { name: 'Thumbnail Gen', cost: 160, usage: '312 thumbnails' },
      { name: 'Video Compression', cost: 100, usage: '89 GB processed' },
    ],
  },
  {
    id: 5,
    name: 'Storage & Backup',
    icon: <Storage color="warning" />,
    currentSpend: 320,
    budgetLimit: 500,
    usage: 64,
    change: +3.8,
    subcategories: [
      { name: 'AWS S3', cost: 180, usage: '1.8 TB' },
      { name: 'Google Drive', cost: 90, usage: '2 TB backup' },
      { name: 'Dropbox Business', cost: 50, usage: '5 TB sync' },
    ],
  },
];

const mockTransactions = [
  {
    id: 1,
    date: '2024-01-14',
    description: 'OpenAI API Usage - GPT-4',
    category: 'AI Content Generation',
    amount: 245.67,
    type: 'expense',
    status: 'completed',
    usage: '850K tokens',
  },
  {
    id: 2,
    date: '2024-01-14',
    description: 'ElevenLabs Voice Synthesis',
    category: 'Voice Synthesis',
    amount: 89.32,
    type: 'expense',
    status: 'completed',
    usage: '12 hours',
  },
  {
    id: 3,
    date: '2024-01-13',
    description: 'GCP Compute Engine',
    category: 'Infrastructure',
    amount: 156.78,
    type: 'expense',
    status: 'completed',
    usage: '120 hours',
  },
  {
    id: 4,
    date: '2024-01-13',
    description: 'YouTube Ad Revenue',
    category: 'Revenue',
    amount: 1250.00,
    type: 'income',
    status: 'completed',
    usage: 'Tech Reviews channel',
  },
  {
    id: 5,
    date: '2024-01-12',
    description: 'AWS S3 Storage',
    category: 'Storage & Backup',
    amount: 45.21,
    type: 'expense',
    status: 'completed',
    usage: '890 GB',
  },
];

const mockBudgetAlerts = [
  {
    id: 1,
    category: 'AI Content Generation',
    message: 'Approaching 80% of monthly budget',
    severity: 'warning' as const,
    threshold: 80,
    current: 81.25,
  },
  {
    id: 2,
    category: 'Infrastructure',
    message: 'Cost spike detected - 25% above average',
    severity: 'info' as const,
    threshold: 100,
    current: 125,
  },
];

interface CostMetrics {
  totalSpent: number;
  monthlyBudget: number;
  projectedSpend: number;
  costPerVideo: number;
  efficiency: number;
  savings: number;
}

export const CostTracking: React.FC = () => {
  const theme = useTheme();
  const [selectedPeriod, setSelectedPeriod] = useState<string>('current');
  const [viewMode, setViewMode] = useState<string>('categories');
  const [loading, setLoading] = useState(false);
  const [budgetDialogOpen, setBudgetDialogOpen] = useState(false);
  const [editingCategory, setEditingCategory] = useState<any>(null);

  const [metrics, setMetrics] = useState<CostMetrics>({
    totalSpent: 6590,
    monthlyBudget: 8700,
    projectedSpend: 7850,
    costPerVideo: 42.25,
    efficiency: 87.3,
    savings: 1250,
  });

  const handleRefresh = async () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 2000);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const getBudgetColor = (usage: number) => {
    if (usage >= 90) return 'error';
    if (usage >= 75) return 'warning';
    if (usage >= 50) return 'info';
    return 'success';
  };

  const getChangeColor = (change: number) => {
    return change > 0 ? 'error.main' : change < 0 ? 'success.main' : 'text.secondary';
  };

  const getChangeIcon = (change: number) => {
    return change > 0 ? <TrendingUp /> : change < 0 ? <TrendingDown /> : null;
  };

  const handleEditBudget = (category: any) => {
    setEditingCategory(category);
    setBudgetDialogOpen(true);
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Cost Tracking
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Monitor and optimize your YouTube automation costs
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <ToggleButtonGroup
            value={selectedPeriod}
            exclusive
            onChange={(e, value) => value && setSelectedPeriod(value)}
            size="small"
          >
            <ToggleButton value="current">Current</ToggleButton>
            <ToggleButton value="last">Last Month</ToggleButton>
            <ToggleButton value="quarter">Quarter</ToggleButton>
          </ToggleButtonGroup>

          <IconButton onClick={handleRefresh} disabled={loading}>
            <Refresh />
          </IconButton>

          <Button variant="outlined" startIcon={<Download />}>
            Export
          </Button>

          <Button variant="contained" startIcon={<Add />}>
            Add Expense
          </Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Budget Alerts */}
      {mockBudgetAlerts.map((alert) => (
        <Alert
          key={alert.id}
          severity={alert.severity}
          sx={{ mb: 2 }}
          action={
            <Button size="small" color="inherit">
              Adjust Budget
            </Button>
          }
          icon={alert.severity === 'warning' ? <Warning /> : <Notifications />}
        >
          <strong>{alert.category}:</strong> {alert.message}
        </Alert>
      ))}

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {[
          { 
            title: 'Total Spent', 
            value: formatCurrency(metrics.totalSpent), 
            subtitle: `of ${formatCurrency(metrics.monthlyBudget)} budget`,
            progress: (metrics.totalSpent / metrics.monthlyBudget) * 100,
            icon: <AttachMoney />,
            color: 'primary'
          },
          { 
            title: 'Projected Spend', 
            value: formatCurrency(metrics.projectedSpend), 
            subtitle: `${metrics.projectedSpend > metrics.monthlyBudget ? 'Over' : 'Under'} budget`,
            progress: (metrics.projectedSpend / metrics.monthlyBudget) * 100,
            icon: <Analytics />,
            color: metrics.projectedSpend > metrics.monthlyBudget ? 'error' : 'success'
          },
          { 
            title: 'Cost per Video', 
            value: formatCurrency(metrics.costPerVideo), 
            subtitle: '15% improvement',
            progress: metrics.efficiency,
            icon: <VideoLibrary />,
            color: 'info'
          },
          { 
            title: 'Monthly Savings', 
            value: formatCurrency(metrics.savings), 
            subtitle: 'vs manual creation',
            progress: 85,
            icon: <AccountBalance />,
            color: 'success'
          },
        ].map((item, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {item.title}
                  </Typography>
                  <Box sx={{ color: `${item.color}.main` }}>
                    {item.icon}
                  </Box>
                </Box>
                <Typography variant="h4" fontWeight="bold" gutterBottom>
                  {item.value}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {item.subtitle}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(item.progress, 100)}
                  color={item.color as any}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Cost Trends Chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Cost Trends</Typography>
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={(e, value) => value && setViewMode(value)}
                size="small"
              >
                <ToggleButton value="categories">By Category</ToggleButton>
                <ToggleButton value="total">Total</ToggleButton>
                <ToggleButton value="efficiency">Efficiency</ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <ResponsiveContainer width="100%" height="90%">
              <ComposedChart data={mockCostData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={(value) => format(new Date(value), 'MMM dd')} />
                <YAxis />
                <RechartsTooltip 
                  labelFormatter={(value) => format(new Date(value), 'PPP')}
                  formatter={(value: any, name: string) => [formatCurrency(value), name]}
                />
                <Legend />
                {viewMode === 'categories' ? (
                  <>
                    <Area dataKey="aiCosts" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} name="AI Costs" />
                    <Area dataKey="infrastructure" stackId="1" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} name="Infrastructure" />
                    <Area dataKey="voiceover" stackId="1" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} name="Voice Synthesis" />
                    <Area dataKey="storage" stackId="1" stroke="#ff7c7c" fill="#ff7c7c" fillOpacity={0.6} name="Storage" />
                  </>
                ) : (
                  <Line
                    type="monotone"
                    dataKey="total"
                    stroke={theme.palette.primary.main}
                    strokeWidth={3}
                    dot={{ r: 6 }}
                    name="Total Cost"
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Budget Overview */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Budget Overview
            </Typography>
            <ResponsiveContainer width="100%" height="90%">
              <PieChart>
                <Pie
                  data={mockCostCategories.map(cat => ({
                    name: cat.name,
                    value: cat.currentSpend,
                    color: theme.palette.mode === 'dark' ? 
                      ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'][cat.id - 1] :
                      ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'][cat.id - 1]
                  }))}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {mockCostCategories.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={
                      ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'][index]
                    } />
                  ))}
                </Pie>
                <RechartsTooltip formatter={(value: any) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Cost Categories */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Cost Categories
            </Typography>
            <Grid container spacing={2}>
              {mockCostCategories.map((category) => (
                <Grid item xs={12} md={6} lg={4} key={category.id}>
                  <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {category.icon}
                          <Typography variant="h6" sx={{ ml: 1 }}>
                            {category.name}
                          </Typography>
                        </Box>
                        <IconButton size="small" onClick={() => handleEditBudget(category)}>
                          <Edit />
                        </IconButton>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h5" fontWeight="bold">
                          {formatCurrency(category.currentSpend)}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {getChangeIcon(category.change)}
                          <Typography 
                            variant="body2" 
                            color={getChangeColor(category.change)}
                            sx={{ ml: 0.5 }}
                          >
                            {category.change > 0 ? '+' : ''}{category.change}%
                          </Typography>
                        </Box>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        of {formatCurrency(category.budgetLimit)} budget
                      </Typography>
                      
                      <LinearProgress
                        variant="determinate"
                        value={category.usage}
                        color={getBudgetColor(category.usage)}
                        sx={{ mb: 2, height: 8, borderRadius: 4 }}
                      />
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Breakdown:
                      </Typography>
                      {category.subcategories.map((sub, idx) => (
                        <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="caption" color="text.secondary">
                            {sub.name}
                          </Typography>
                          <Typography variant="caption" fontWeight="bold">
                            {formatCurrency(sub.cost)}
                          </Typography>
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      {/* Recent Transactions */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">Recent Transactions</Typography>
              <Button variant="outlined" size="small" startIcon={<Receipt />}>
                View All
              </Button>
            </Box>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Date</TableCell>
                    <TableCell>Description</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Usage</TableCell>
                    <TableCell align="right">Amount</TableCell>
                    <TableCell align="center">Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {mockTransactions.map((transaction) => (
                    <TableRow key={transaction.id} hover>
                      <TableCell>
                        <Typography variant="body2">
                          {format(new Date(transaction.date), 'MMM dd, yyyy')}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {transaction.description}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={transaction.category}
                          size="small"
                          color={transaction.type === 'income' ? 'success' : 'default'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {transaction.usage}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography 
                          variant="body2" 
                          fontWeight="bold"
                          color={transaction.type === 'income' ? 'success.main' : 'text.primary'}
                        >
                          {transaction.type === 'income' ? '+' : '-'}{formatCurrency(transaction.amount)}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={transaction.status}
                          size="small"
                          color="success"
                          variant="outlined"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Budget Edit Dialog */}
      <Dialog open={budgetDialogOpen} onClose={() => setBudgetDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Budget - {editingCategory?.name}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Budget Limit"
                type="number"
                defaultValue={editingCategory?.budgetLimit}
                InputProps={{
                  startAdornment: '$',
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Enable budget alerts"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Alert Threshold (%)"
                type="number"
                defaultValue={80}
                InputProps={{
                  endAdornment: '%',
                }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBudgetDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setBudgetDialogOpen(false)}>
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CostTracking;