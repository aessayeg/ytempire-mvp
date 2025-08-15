import React from 'react';
import { 
  Card,
  CardContent,
  CardMedia,
  CardActions,
  Typography,
  Chip,
  IconButton,
  Box,
  LinearProgress,
  Menu,
  MenuItem
 } from '@mui/material';
import { 
  PlayArrow,
  Edit,
  Delete,
  MoreVert,
  Schedule,
  CheckCircle,
  Error,
  HourglassEmpty,
  TrendingUp,
  AttachMoney,
  Visibility,
  ThumbUp,
  Comment
 } from '@mui/icons-material';
import {  formatDistanceToNow  } from 'date-fns';
import {  useNavigate  } from 'react-router-dom';

interface VideoCardProps {
  video: {,
  id: string,

    title: string;
    description?: string;
    thumbnail_url?: string;
    channel_id: string;
    channel_name?: string;
    generation_status: 'pending' | 'processing' | 'completed' | 'failed',
  publish_status: 'draft' | 'scheduled' | 'published' | 'publishing';
    quality_score?: number;
    trend_score?: number;
    total_cost: number,
  view_count: number,

    like_count: number,
  comment_count: number,

    created_at: string;
    published_at?: string;
    scheduled_publish_time?: string;
    duration_seconds?: number;
    youtube_url?: string;
    progress?: number;
    error_?: string;
  };
  onEdit?: (videoId: string) => void;
  onDelete?: (videoId: string) => void;
  onPublish?: (videoId: string) => void;
  onPreview?: (videoId: string) => void}

export const VideoCard: React.FC<VideoCardProps> = ({ video, onEdit, onDelete, onPublish, onPreview }) => {
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (_: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)};

  const handleMenuClose = () => {
    setAnchorEl(null)};

  const handleCardClick = () => {
    navigate(`/videos/${video.id}`)};

  const getStatusIcon = () => {
    switch (video.generation_status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'processing':
        return <HourglassEmpty color="warning" />;
      case 'failed':
        return <Error color="error" />;
        return <Schedule color="action" />}
  };

  const getStatusColor = (): unknown => {
    switch (video.generation_status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
        return 'default'}
  };

  const getPublishStatusColor = (): unknown => {
    switch (video.publish_status) {
      case 'published':
        return 'success';
      case 'scheduled':
        return 'info';
      case 'publishing':
        return 'warning';
        return 'default'}
  };

  const formatDuration = (_seconds?: number) => {
    if (!seconds) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${num / 1000000.toFixed(1}M`;
    if (num >= 1000) return `${num / 1000.toFixed(1}K`;
    return num.toString()};

  return (
    <>
      <Card
      sx={ {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        cursor: 'pointer',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 4 }
      }}
    >
      <Box position="relative">
        <CardMedia
          component="img"
          height="180"
          image={video.thumbnail_url || '/placeholder-video.png'}
          alt={video.title}
          onClick={handleCardClick}
          sx={{ cursor: 'pointer' }}
        />
        {video.duration_seconds && (
          <Chip
            label={formatDuration(video.duration_seconds)}
            size="small"
            sx={ {
              position: 'absolute',
              bottom: 8,
              right: 8,
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              color: 'white' }}
          />
        )}
        {video.generation_status === 'processing' && video.progress && (
          <LinearProgress
            variant="determinate"
            value={video.progress}
            sx={ {
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0 }}
          />
        )}
      </Box>
      <CardContent sx={{ flexGrow: 1, pb: 1 }} onClick={handleCardClick}>
        <Box display="flex" alignItems="flex-start" justifyContent="space-between" mb={1}>
          <Typography
            variant="subtitle1"
            component="h3"
            sx={ {
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              fontWeight: 500,
              flex: 1 }}
          >
            {video.title}
          </Typography>
          <Tooltip title={video.generation_status}>
            <Box ml={1}>{getStatusIcon()}</Box>
          </Tooltip>
        </Box>

        {video.channel_name && (
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {video.channel_name}
          </Typography>
        )}
        <Box display="flex" gap={1} flexWrap="wrap" mb={1}>
          <Chip
            label={video.generation_status}
            color={getStatusColor()}
            size="small"
          />
          <Chip
            label={video.publish_status}
            color={getPublishStatusColor()}
            size="small"
          />
        </Box>

        {video.generation_status === 'completed' && (
          <Box display="flex" gap={2} mb={1}>
            {video.quality_score !== undefined && (
              <Box display="flex" alignItems="center" gap={0.5}>
                <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                <Typography variant="caption">
                  Quality: {video.quality_score.toFixed(0)}%
                </Typography>
              </Box>
            )}
            {video.trend_score !== undefined && (
              <Box display="flex" alignItems="center" gap={0.5}>
                <TrendingUp sx={{ fontSize: 16, color: 'info.main' }} />
                <Typography variant="caption">
                  Trend: {video.trend_score.toFixed(0)}%
                </Typography>
              </Box>
            )}
          </Box>
        )}
        <Box display="flex" alignItems="center" gap={2} mb={1}>
          <Box display="flex" alignItems="center" gap={0.5}>
            <AttachMoney sx={{ fontSize: 16 }} />
            <Typography variant="caption">${video.total_cost.toFixed(2)}</Typography>
          </Box>
          {video.published_at && (
            <>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Visibility sx={{ fontSize: 16 }} />
                <Typography variant="caption">{formatNumber(video.view_count)}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <ThumbUp sx={{ fontSize: 16 }} />
                <Typography variant="caption">{formatNumber(video.like_count)}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Comment sx={{ fontSize: 16 }} />
                <Typography variant="caption">{formatNumber(video.comment_count)}</Typography>
              </Box>
            </>
          )}
        </Box>

        {video.error_ && (
          <Typography variant="caption" color="error" sx={{ display: 'block', mt: 1 }}>
            Error: {video.error_}
          </Typography>
        )}
        <Typography variant="caption" color="text.secondary">
          Created {formatDistanceToNow(new Date(video.created_at), { addSuffix: true });
}
        </Typography>

        {video.scheduled_publish_time && video.publish_status === 'scheduled' && (
          <Typography variant="caption" color="info.main" sx={{ display: 'block' }}>
            Scheduled for {new Date(video.scheduled_publish_time).toLocaleString()}
          </Typography>
        )}
      </CardContent>

      <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
        <Box>
          {video.generation_status === 'completed' && (_<>
              <Tooltip title="Preview">
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation(</>
  );
                    onPreview?.(video.id)}}
                >
                  <PlayArrow />
                </IconButton>
              </Tooltip>
              {video.publish_status === 'draft' && (_<Tooltip title="Publish">
                  <IconButton
                    size="small"
                    color="primary"
                    onClick={(e) => {
                      e.stopPropagation(</>
  );
                      onPublish?.(video.id)}}
                  >
                    <Schedule />
                  </IconButton>
                </Tooltip>
              )}
            </>
          )}
          {video.generation_status !== 'processing' && (_<Tooltip title="Edit">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit?.(video.id)}}
              >
                <Edit />
              </IconButton>
            </Tooltip>
          )}
        </Box>

        <IconButton
          size="small"
          onClick={(e) => {
            e.stopPropagation();
            handleMenuOpen(e)}}
        >
          <MoreVert />
        </IconButton>

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          onClick={(e) => e.stopPropagation()}
        >
          <MenuItem
            onClick={() => {
              handleMenuClose();
              navigate(`/videos/${video.id}`)}}
          >
            View Details
          </MenuItem>
          {video.youtube_url && (_<MenuItem
              onClick={() => {
                handleMenuClose();
                window.open(video.youtube_url, '_blank')}}
            >
              View on YouTube
            </MenuItem>
          )}
          <MenuItem
            onClick={() => {
              handleMenuClose();
              onEdit?.(video.id)}}
          >
            Edit
          </MenuItem>
          <MenuItem
            onClick={() => {
              handleMenuClose();
              onDelete?.(video.id)}}
            sx={{ color: 'error.main' }}
          >
            Delete
          </MenuItem>
        </Menu>
      </CardActions>
    </Card>
  )};