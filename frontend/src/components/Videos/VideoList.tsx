import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Pagination,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Skeleton,
} from '@mui/material';
import { ViewModule, ViewList } from '@mui/icons-material';
import { VideoCard } from './VideoCard';
import { useVideoStore } from '../../stores/videoStore';
import { api } from '../../services/api';

interface VideoListProps {
  channelId?: string;
  status?: 'all' | 'generated' | 'published' | 'draft';
  onVideoSelect?: (videoId: string) => void;
  onVideoEdit?: (videoId: string) => void;
  onVideoDelete?: (videoId: string) => void;
  onVideoPublish?: (videoId: string) => void;
  onVideoPreview?: (videoId: string) => void;
}

export const VideoList: React.FC<VideoListProps> = ({
  channelId,
  status = 'all',
  onVideoSelect,
  onVideoEdit,
  onVideoDelete,
  onVideoPublish,
  onVideoPreview,
}) => {
  const [page, setPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(12);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'created' | 'published' | 'views' | 'cost'>('created');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [videos, setVideos] = useState<any[]>([]);
  const [totalCount, setTotalCount] = useState(0);

  // Fetch videos from API
  useEffect(() => {
    fetchVideos();
  }, [channelId, status, page, itemsPerPage, sortBy, sortOrder]);

  const fetchVideos = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.videos.list({
        channel_id: channelId,
        status: status === 'all' ? undefined : status,
        skip: (page - 1) * itemsPerPage,
        limit: itemsPerPage,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      setVideos(response.data);
      setTotalCount(response.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load videos');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleItemsPerPageChange = (event: any) => {
    setItemsPerPage(event.target.value);
    setPage(1);
  };

  const handleSortChange = (event: any) => {
    setSortBy(event.target.value);
    setPage(1);
  };

  const handleSortOrderToggle = () => {
    setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    setPage(1);
  };

  const handleViewModeChange = (
    event: React.MouseEvent<HTMLElement>,
    newMode: 'grid' | 'list' | null
  ) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };

  const totalPages = Math.ceil(totalCount / itemsPerPage);

  // Sort options
  const sortOptions = [
    { value: 'created', label: 'Date Created' },
    { value: 'published', label: 'Date Published' },
    { value: 'views', label: 'View Count' },
    { value: 'cost', label: 'Generation Cost' },
    { value: 'quality', label: 'Quality Score' },
    { value: 'trend', label: 'Trend Score' },
  ];

  // Loading skeleton
  const renderSkeleton = () => (
    <Grid container spacing={3}>
      {[...Array(itemsPerPage)].map((_, index) => (
        <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
          <Paper sx={{ p: 2 }}>
            <Skeleton variant="rectangular" height={180} />
            <Box pt={2}>
              <Skeleton variant="text" />
              <Skeleton variant="text" width="60%" />
              <Box display="flex" gap={1} mt={1}>
                <Skeleton variant="rectangular" width={60} height={20} />
                <Skeleton variant="rectangular" width={60} height={20} />
              </Box>
            </Box>
          </Paper>
        </Grid>
      ))}
    </Grid>
  );

  // Empty state
  const renderEmptyState = () => (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight={400}
      p={4}
    >
      <Typography variant="h6" color="text.secondary" gutterBottom>
        No videos found
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {channelId
          ? 'No videos have been generated for this channel yet.'
          : 'Start generating videos to see them here.'}
      </Typography>
    </Box>
  );

  // List view component
  const VideoListItem = ({ video }: { video: any }) => (
    <Paper
      sx={{
        p: 2,
        mb: 2,
        cursor: 'pointer',
        transition: 'box-shadow 0.2s',
        '&:hover': { boxShadow: 3 },
      }}
      onClick={() => onVideoSelect?.(video.id)}
    >
      <Box display="flex" alignItems="center" gap={2}>
        <Box
          component="img"
          src={video.thumbnail_url || '/placeholder-video.png'}
          alt={video.title}
          sx={{ width: 120, height: 67, borderRadius: 1, objectFit: 'cover' }}
        />
        <Box flex={1}>
          <Typography variant="subtitle1" fontWeight={500}>
            {video.title}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {video.channel_name} • {new Date(video.created_at).toLocaleDateString()}
          </Typography>
          <Box display="flex" gap={2}>
            <Typography variant="caption">
              Status: {video.generation_status}
            </Typography>
            <Typography variant="caption">
              Cost: ${video.total_cost.toFixed(2)}
            </Typography>
            {video.view_count > 0 && (
              <Typography variant="caption">
                Views: {video.view_count.toLocaleString()}
              </Typography>
            )}
          </Box>
        </Box>
      </Box>
    </Paper>
  );

  return (
    <Box>
      {/* Controls */}
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={3}
        flexWrap="wrap"
        gap={2}
      >
        <Box display="flex" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Sort By</InputLabel>
            <Select value={sortBy} onChange={handleSortChange} label="Sort By">
              {sortOptions.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <ToggleButton
            value="sort"
            selected={sortOrder === 'desc'}
            onChange={handleSortOrderToggle}
            size="small"
          >
            {sortOrder === 'desc' ? '↓' : '↑'}
          </ToggleButton>

          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel>Show</InputLabel>
            <Select
              value={itemsPerPage}
              onChange={handleItemsPerPageChange}
              label="Show"
            >
              <MenuItem value={12}>12</MenuItem>
              <MenuItem value={24}>24</MenuItem>
              <MenuItem value={48}>48</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <Box display="flex" gap={2} alignItems="center">
          <Typography variant="body2" color="text.secondary">
            {totalCount} videos
          </Typography>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewModeChange}
            size="small"
          >
            <ToggleButton value="grid">
              <ViewModule />
            </ToggleButton>
            <ToggleButton value="list">
              <ViewList />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Error state */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Loading state */}
      {isLoading && renderSkeleton()}

      {/* Empty state */}
      {!isLoading && videos.length === 0 && renderEmptyState()}

      {/* Video grid/list */}
      {!isLoading && videos.length > 0 && (
        <>
          {viewMode === 'grid' ? (
            <Grid container spacing={3}>
              {videos.map((video) => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={video.id}>
                  <VideoCard
                    video={video}
                    onEdit={onVideoEdit}
                    onDelete={onVideoDelete}
                    onPublish={onVideoPublish}
                    onPreview={onVideoPreview}
                  />
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box>
              {videos.map((video) => (
                <VideoListItem key={video.id} video={video} />
              ))}
            </Box>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <Box display="flex" justifyContent="center" mt={4}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
                size="large"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}
    </Box>
  );
};