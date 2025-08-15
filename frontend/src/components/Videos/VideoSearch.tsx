import React, { useState } from 'react';
import { Box, TextField, InputAdornment, IconButton, Menu, MenuItem, Chip, Button } from '@mui/material';
import { Search, FilterList, Clear, CalendarToday, Sort } from '@mui/icons-material';

interface VideoSearchProps {
  onSearch: (query: string, filters: unknown) => void;
  onClear: () => void;
}

export const VideoSearch: React.FC<VideoSearchProps> = ({ onSearch, onClear }) => {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({ dateRange: 'all', sortBy: 'relevance', status: 'all' });
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleSearch = () => {
    onSearch(query, filters);
  };

  const handleClear = () => {
    setQuery('');
    setFilters({ dateRange: 'all', sortBy: 'relevance', status: 'all' });
    onClear();
  };

  return (
    <Box display="flex" gap={2} alignItems="center">
      <TextField
        fullWidth
        placeholder="Search videos..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Search />
            </InputAdornment>
          ),
          endAdornment: query && (
            <InputAdornment position="end">
              <IconButton size="small" onClick={handleClear}>
                <Clear />
              </IconButton>
            </InputAdornment>
          )
        }}
      />
      
      <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
        <FilterList />
      </IconButton>
      
      <Button variant="contained" onClick={handleSearch}>Search</Button>
      
      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
        <MenuItem>Date Range</MenuItem>
        <MenuItem>Sort By</MenuItem>
        <MenuItem>Status</MenuItem>
      </Menu>
      
      {Object.values(filters).some(v => v !== 'all') && (
        <Box display="flex" gap={1}>
          {filters.dateRange !== 'all' && <Chip label={filters.dateRange} size="small" onDelete={() => setFilters({...filters, dateRange: 'all'})} />}
          {filters.sortBy !== 'relevance' && <Chip label={filters.sortBy} size="small" onDelete={() => setFilters({...filters, sortBy: 'relevance'})} />}
          {filters.status !== 'all' && <Chip label={filters.status} size="small" onDelete={() => setFilters({...filters, status: 'all'})} />}
        </Box>
      )}
    </Box>
  );
};