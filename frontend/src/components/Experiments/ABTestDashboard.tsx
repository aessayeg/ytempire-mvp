import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
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
  LinearProgress,
  Alert
} from '@mui/material';
import Grid from '@mui/material/Grid2';
import { Science, TrendingUp, CheckCircle } from '@mui/icons-material';

interface Experiment {
  id: string;
  name: string;
  description: string;
  hypothesis: string;
  status: 'draft' | 'running' | 'completed' | 'paused';
  startDate?: Date;
  endDate?: Date;
  variants: Variant[];
  metrics: Metric[];
}

interface Variant {
  id: string;
  name: string;
  description: string;
  allocation: number;
  conversions?: number;
  impressions?: number;
}

interface Metric {
  name: string;
  value: number;
  target: number;
  unit: string;
}

const ABTestDashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    hypothesis: '',
    status: 'draft' as const
  });

  // Sample data
  useEffect(() => {
    setExperiments([
      {
        id: '1',
        name: 'Thumbnail Style Test',
        description: 'Testing different thumbnail styles',
        hypothesis: 'Brighter thumbnails will increase CTR by 15%',
        status: 'running',
        startDate: new Date('2024-01-01'),
        variants: [
          { id: 'a', name: 'Control', description: 'Current style', allocation: 50, conversions: 1250, impressions: 10000 },
          { id: 'b', name: 'Bright', description: 'Brighter colors', allocation: 50, conversions: 1450, impressions: 10000 }
        ],
        metrics: [
          { name: 'CTR', value: 14.5, target: 15, unit: '%' },
          { name: 'Watch Time', value: 4.2, target: 4, unit: 'min' }
        ]
      },
      {
        id: '2',
        name: 'Video Length Optimization',
        description: 'Testing optimal video duration',
        hypothesis: 'Shorter videos will increase completion rate',
        status: 'completed',
        startDate: new Date('2023-12-01'),
        endDate: new Date('2023-12-31'),
        variants: [
          { id: 'a', name: '5 min', description: 'Standard length', allocation: 33, conversions: 800, impressions: 5000 },
          { id: 'b', name: '3 min', description: 'Short format', allocation: 33, conversions: 950, impressions: 5000 },
          { id: 'c', name: '8 min', description: 'Long format', allocation: 34, conversions: 650, impressions: 5000 }
        ],
        metrics: [
          { name: 'Completion Rate', value: 78, target: 75, unit: '%' },
          { name: 'Engagement', value: 8.5, target: 8, unit: 'score' }
        ]
      }
    ]);
  }, []);

  const createExperiment = () => {
    const experiment: Experiment = {
      id: Date.now().toString(),
      name: newExperiment.name,
      description: newExperiment.description,
      hypothesis: newExperiment.hypothesis,
      status: 'draft',
      variants: [],
      metrics: []
    };
    setExperiments([...experiments, experiment]);
    setCreateDialogOpen(false);
    setNewExperiment({ name: '', description: '', hypothesis: '', status: 'draft' });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const getWinningVariant = (variants: Variant[]) => {
    if (!variants || variants.length === 0) return null;
    return variants.reduce((best, current) => {
      const currentRate = (current.conversions || 0) / (current.impressions || 1);
      const bestRate = (best.conversions || 0) / (best.impressions || 1);
      return currentRate > bestRate ? current : best;
    });
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">
          <Science sx={{ mr: 1, verticalAlign: 'middle' }} />
          A/B Test Dashboard
        </Typography>
        <Button
          variant="contained"
          startIcon={<Science />}
          onClick={() => setCreateDialogOpen(true)}
        >
          New Experiment
        </Button>
      </Box>

      <Grid container spacing={3}>
        {experiments.map(experiment => {
          const winner = getWinningVariant(experiment.variants);
          return (
            <Grid item xs={12} md={6} key={experiment.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6">{experiment.name}</Typography>
                    <Chip
                      label={experiment.status}
                      color={getStatusColor(experiment.status)}
                      size="small"
                    />
                  </Box>
                  
                  <Typography variant="body2" color="textSecondary" paragraph>
                    {experiment.description}
                  </Typography>
                  
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Hypothesis:</strong> {experiment.hypothesis}
                    </Typography>
                  </Alert>

                  {experiment.variants.length > 0 && (
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Variant</TableCell>
                            <TableCell align="right">Allocation</TableCell>
                            <TableCell align="right">Conversions</TableCell>
                            <TableCell align="right">Rate</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {experiment.variants.map(variant => (
                            <TableRow key={variant.id}>
                              <TableCell>
                                {variant.name}
                                {winner?.id === variant.id && (
                                  <CheckCircle color="success" sx={{ ml: 1, fontSize: 16 }} />
                                )}
                              </TableCell>
                              <TableCell align="right">{variant.allocation}%</TableCell>
                              <TableCell align="right">{variant.conversions || 0}</TableCell>
                              <TableCell align="right">
                                {((variant.conversions || 0) / (variant.impressions || 1) * 100).toFixed(1)}%
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )}

                  {experiment.metrics && experiment.metrics.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      {experiment.metrics.map((metric, index) => (
                        <Box key={index} sx={{ mb: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography variant="body2">{metric.name}</Typography>
                            <Typography variant="body2">
                              {metric.value}{metric.unit} / {metric.target}{metric.unit}
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={Math.min((metric.value / metric.target) * 100, 100)}
                            color={metric.value >= metric.target ? 'success' : 'primary'}
                          />
                        </Box>
                      ))}
                    </Box>
                  )}

                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button size="small" variant="outlined">View Details</Button>
                    {experiment.status === 'running' && (
                      <Button size="small" color="warning">Pause</Button>
                    )}
                    {experiment.status === 'paused' && (
                      <Button size="small" color="primary">Resume</Button>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Experiment</DialogTitle>
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

export default ABTestDashboard;