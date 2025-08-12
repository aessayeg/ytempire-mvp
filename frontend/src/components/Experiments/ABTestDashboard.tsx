import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import AddIcon from '@mui/icons-material/Add';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ScienceIcon from '@mui/icons-material/Science';
import { authStore } from '../../stores/authStore';

interface Variant {
  name: string;
  allocation: number;
  config: Record<string, any>;
  is_control: boolean;
}

interface Experiment {
  experiment_id: number;
  name: string;
  description: string;
  status: string;
  variants: any[];
  target_metric: string;
  start_date: string | null;
  end_date: string | null;
  winner_variant: string | null;
}

interface ExperimentResults {
  experiment_id: number;
  name: string;
  status: string;
  variants: Array<{
    name: string;
    sample_size: number;
    conversions: number;
    conversion_rate: number;
    confidence_interval: [number, number];
    revenue: number;
    avg_revenue_per_user: number;
    p_value: number | null;
    is_significant: boolean;
    lift: number | null;
  }>;
  winner: string | null;
  required_sample_size: number;
  can_conclude: boolean;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const ABTestDashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    hypothesis: '',
    variants: [
      { name: 'control', allocation: 50, config: {}, is_control: true },
      { name: 'variant_a', allocation: 50, config: {}, is_control: false },
    ],
    target_metric: 'conversion_rate',
    duration_days: 14,
    min_sample_size: 100,
  });
  const { token } = authStore();

  const fetchExperiments = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/experiments/`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setExperiments(data);
      }
    } catch (error) {
      console.error('Error fetching experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchExperimentResults = async (experimentId: number) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/experiments/${experimentId}/results`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setSelectedExperiment(data);
      }
    } catch (error) {
      console.error('Error fetching experiment results:', error);
    }
  };

  const createExperiment = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/experiments/`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(newExperiment),
        }
      );
      
      if (response.ok) {
        setCreateDialogOpen(false);
        fetchExperiments();
      }
    } catch (error) {
      console.error('Error creating experiment:', error);
    }
  };

  const startExperiment = async (experimentId: number) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/experiments/${experimentId}/start`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );
      
      if (response.ok) {
        fetchExperiments();
      }
    } catch (error) {
      console.error('Error starting experiment:', error);
    }
  };

  const concludeExperiment = async (experimentId: number, winner?: string) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/experiments/${experimentId}/conclude`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ winner_variant: winner }),
        }
      );
      
      if (response.ok) {
        fetchExperiments();
      }
    } catch (error) {
      console.error('Error concluding experiment:', error);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, [token]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'default';
      case 'running': return 'primary';
      case 'paused': return 'warning';
      case 'completed': return 'success';
      case 'archived': return 'default';
      default: return 'default';
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;

  if (loading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h2">
          A/B Testing Dashboard
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          New Experiment
        </Button>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab label="Active Experiments" />
          <Tab label="Results & Analysis" />
          <Tab label="All Experiments" />
        </Tabs>
      </Paper>

      {/* Active Experiments */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          {experiments
            .filter(exp => exp.status === 'running')
            .map(experiment => (
              <Grid item xs={12} md={6} lg={4} key={experiment.experiment_id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">
                        {experiment.name}
                      </Typography>
                      <Chip 
                        label={experiment.status}
                        color={getStatusColor(experiment.status)}
                        size="small"
                      />
                    </Box>
                    <Typography variant="body2" color="textSecondary" paragraph>
                      {experiment.description}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Target Metric:</strong> {experiment.target_metric}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Variants:</strong> {experiment.variants.length}
                    </Typography>
                    {experiment.start_date && (
                      <Typography variant="body2">
                        <strong>Started:</strong> {new Date(experiment.start_date).toLocaleDateString()}
                      </Typography>
                    )}
                    <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                      <Button
                        size="small"
                        startIcon={<VisibilityIcon />}
                        onClick={() => fetchExperimentResults(experiment.experiment_id)}
                      >
                        View Results
                      </Button>
                      <IconButton
                        size="small"
                        onClick={() => concludeExperiment(experiment.experiment_id)}
                      >
                        <StopIcon />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
        </Grid>
      </TabPanel>

      {/* Results & Analysis */}
      <TabPanel value={activeTab} index={1}>
        {selectedExperiment ? (
          <Box>
            <Typography variant="h5" gutterBottom>
              {selectedExperiment.name}
            </Typography>
            
            {/* Statistical Significance Alert */}
            {selectedExperiment.can_conclude ? (
              <Alert severity="success" sx={{ mb: 3 }}>
                This experiment has reached statistical significance and can be concluded.
                {selectedExperiment.winner && ` Winner: ${selectedExperiment.winner}`}
              </Alert>
            ) : (
              <Alert severity="info" sx={{ mb: 3 }}>
                More data needed for statistical significance. 
                Required sample size: {selectedExperiment.required_sample_size} per variant
              </Alert>
            )}

            {/* Variant Performance Chart */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Conversion Rate by Variant
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={selectedExperiment.variants}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <RechartsTooltip formatter={(value: number) => formatPercentage(value)} />
                  <Legend />
                  <Bar dataKey="conversion_rate" fill="#8884d8">
                    {selectedExperiment.variants.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.is_significant ? '#82ca9d' : '#8884d8'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Paper>

            {/* Detailed Results Table */}
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Variant</TableCell>
                    <TableCell align="right">Sample Size</TableCell>
                    <TableCell align="right">Conversions</TableCell>
                    <TableCell align="right">Conversion Rate</TableCell>
                    <TableCell align="right">Confidence Interval</TableCell>
                    <TableCell align="right">Lift</TableCell>
                    <TableCell align="right">P-Value</TableCell>
                    <TableCell align="right">Revenue</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {selectedExperiment.variants.map((variant) => (
                    <TableRow key={variant.name}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {variant.name}
                          {variant.name === 'control' && (
                            <Chip label="Control" size="small" />
                          )}
                          {variant.is_significant && (
                            <Chip label="Significant" color="success" size="small" />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{variant.sample_size}</TableCell>
                      <TableCell align="right">{variant.conversions}</TableCell>
                      <TableCell align="right">{formatPercentage(variant.conversion_rate)}</TableCell>
                      <TableCell align="right">
                        [{formatPercentage(variant.confidence_interval[0])}, {formatPercentage(variant.confidence_interval[1])}]
                      </TableCell>
                      <TableCell align="right">
                        {variant.lift !== null ? (
                          <Chip 
                            label={`${variant.lift > 0 ? '+' : ''}${variant.lift.toFixed(2)}%`}
                            color={variant.lift > 0 ? 'success' : 'error'}
                            size="small"
                          />
                        ) : '-'}
                      </TableCell>
                      <TableCell align="right">
                        {variant.p_value !== null ? variant.p_value.toFixed(4) : '-'}
                      </TableCell>
                      <TableCell align="right">${variant.revenue.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        ) : (
          <Alert severity="info">
            Select an experiment to view detailed results
          </Alert>
        )}
      </TabPanel>

      {/* All Experiments */}
      <TabPanel value={activeTab} index={2}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Target Metric</TableCell>
                <TableCell>Variants</TableCell>
                <TableCell>Start Date</TableCell>
                <TableCell>End Date</TableCell>
                <TableCell>Winner</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {experiments.map((experiment) => (
                <TableRow key={experiment.experiment_id}>
                  <TableCell>{experiment.name}</TableCell>
                  <TableCell>
                    <Chip 
                      label={experiment.status}
                      color={getStatusColor(experiment.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{experiment.target_metric}</TableCell>
                  <TableCell>{experiment.variants.length}</TableCell>
                  <TableCell>
                    {experiment.start_date ? new Date(experiment.start_date).toLocaleDateString() : '-'}
                  </TableCell>
                  <TableCell>
                    {experiment.end_date ? new Date(experiment.end_date).toLocaleDateString() : '-'}
                  </TableCell>
                  <TableCell>
                    {experiment.winner_variant ? (
                      <Chip label={experiment.winner_variant} color="success" size="small" />
                    ) : '-'}
                  </TableCell>
                  <TableCell>
                    {experiment.status === 'draft' && (
                      <IconButton
                        size="small"
                        onClick={() => startExperiment(experiment.experiment_id)}
                      >
                        <PlayArrowIcon />
                      </IconButton>
                    )}
                    <IconButton
                      size="small"
                      onClick={() => fetchExperimentResults(experiment.experiment_id)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Create Experiment Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New A/B Test</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Experiment Name"
                value={newExperiment.name}
                onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={newExperiment.description}
                onChange={(e) => setNewExperiment({ ...newExperiment, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Hypothesis"
                value={newExperiment.hypothesis}
                onChange={(e) => setNewExperiment({ ...newExperiment, hypothesis: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Target Metric</InputLabel>
                <Select
                  value={newExperiment.target_metric}
                  onChange={(e) => setNewExperiment({ ...newExperiment, target_metric: e.target.value })}
                >
                  <MenuItem value="conversion_rate">Conversion Rate</MenuItem>
                  <MenuItem value="revenue">Revenue</MenuItem>
                  <MenuItem value="engagement">Engagement</MenuItem>
                  <MenuItem value="retention">Retention</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                type="number"
                label="Duration (days)"
                value={newExperiment.duration_days}
                onChange={(e) => setNewExperiment({ ...newExperiment, duration_days: parseInt(e.target.value) })}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                type="number"
                label="Min Sample Size"
                value={newExperiment.min_sample_size}
                onChange={(e) => setNewExperiment({ ...newExperiment, min_sample_size: parseInt(e.target.value) })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={createExperiment} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};