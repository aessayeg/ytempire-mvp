import React, { useState } from 'react';
import { Box, Paper, Typography, FormControl, InputLabel, Select, MenuItem, Slider, Chip, Button, Accordion, AccordionSummary, AccordionDetails, FormGroup, FormControlLabel, Checkbox } from '@mui/material';
import { ExpandMore, FilterList, Clear } from '@mui/icons-material';

interface VideoFiltersProps {
  onApplyFilters: (filters: any) => void;
  onClearFilters: () => void;
}

export const VideoFilters: React.FC<VideoFiltersProps> = ({ onApplyFilters, onClearFilters }) => {
  const [filters, setFilters] = useState({
    status: 'all',
    dateRange: 'all',
    channel: 'all',
    qualityScore: [0, 100],
    duration: 'all',
    hasRevenue: false,
    minViews: 0,
    categories: []
  });

  const handleApply = () => {
    onApplyFilters(filters);
  };

  const handleClear = () => {
    setFilters({
      status: 'all',
      dateRange: 'all',
      channel: 'all',
      qualityScore: [0, 100],
      duration: 'all',
      hasRevenue: false,
      minViews: 0,
      categories: []
    });
    onClearFilters();
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          <FilterList sx={{ mr: 1, verticalAlign: 'middle' }} />
          Filters
        </Typography>
        <Button size="small" startIcon={<Clear />} onClick={handleClear}>Clear All</Button>
      </Box>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Status</InputLabel>
        <Select value={filters.status} onChange={(e) => setFilters({...filters, status: e.target.value})}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="draft">Draft</MenuItem>
          <MenuItem value="published">Published</MenuItem>
          <MenuItem value="scheduled">Scheduled</MenuItem>
          <MenuItem value="processing">Processing</MenuItem>
        </Select>
      </FormControl>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Date Range</InputLabel>
        <Select value={filters.dateRange} onChange={(e) => setFilters({...filters, dateRange: e.target.value})}>
          <MenuItem value="all">All Time</MenuItem>
          <MenuItem value="today">Today</MenuItem>
          <MenuItem value="week">This Week</MenuItem>
          <MenuItem value="month">This Month</MenuItem>
          <MenuItem value="year">This Year</MenuItem>
        </Select>
      </FormControl>

      <Box mt={2} mb={1}>
        <Typography variant="body2">Quality Score</Typography>
        <Slider value={filters.qualityScore} onChange={(e, v) => setFilters({...filters, qualityScore: v as number[]})} valueLabelDisplay="auto" min={0} max={100} />
      </Box>

      <FormControl fullWidth margin="normal" size="small">
        <InputLabel>Duration</InputLabel>
        <Select value={filters.duration} onChange={(e) => setFilters({...filters, duration: e.target.value})}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="short">Short (< 3 min)</MenuItem>
          <MenuItem value="medium">Medium (3-10 min)</MenuItem>
          <MenuItem value="long">Long (> 10 min)</MenuItem>
        </Select>
      </FormControl>

      <FormControlLabel control={<Checkbox checked={filters.hasRevenue} onChange={(e) => setFilters({...filters, hasRevenue: e.target.checked})} />} label="Has Revenue" />

      <Box mt={2}>
        <Button fullWidth variant="contained" onClick={handleApply}>Apply Filters</Button>
      </Box>
    </Paper>
  );
};