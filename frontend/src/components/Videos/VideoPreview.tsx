import React from 'react';
import { 
  Box,
  Paper,
  Typography,
  Button
 } from '@mui/material';
import {  PlayArrow, Edit, Publish  } from '@mui/icons-material';
import {  VideoPlayer  } from './VideoPlayer';

interface VideoPreviewProps {
  video: unknown;
  onEdit?: () => void;
  onPublish?: () => void;
  onApprove?: () => void;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({ video, onEdit, onPublish, onApprove }) => {
  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Video Preview</Typography>
      <VideoPlayer videoUrl={video.video_url} thumbnail={video.thumbnail_url} />
      <Box mt={2}>
        <Typography variant="h6">{video.title}</Typography>
        <Typography variant="body2" color="text.secondary">{video.description}</Typography>
        <Box display="flex" gap={1} mt={2}>
          <Button variant="contained" startIcon={<PlayArrow />}>Preview</Button>
          <Button variant="outlined" startIcon={<Edit />} onClick={onEdit}>Edit</Button>
          <Button variant="contained" color="success" startIcon={<Publish />} onClick={onPublish}>Publish</Button>
        </Box>
      </Box>
    </Paper>
  </>
  )};