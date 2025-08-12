import React from 'react';
import { Box, Paper, Typography, LinearProgress, List, ListItem, ListItemText, Chip, Alert } from '@mui/material';
import { CloudUpload, CheckCircle, Error } from '@mui/icons-material';

interface UploadProgressProps {
  progress: number;
  status: 'preparing' | 'uploading' | 'processing' | 'completed' | 'failed';
  fileName?: string;
  fileSize?: number;
  uploadSpeed?: number;
  timeRemaining?: number;
  error?: string;
}

export const VideoUploadProgress: React.FC<UploadProgressProps> = ({ progress, status, fileName, fileSize, uploadSpeed, timeRemaining, error }) => {
  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" gap={2} mb={2}>
        <CloudUpload color="primary" />
        <Typography variant="h6">Upload Progress</Typography>
        <Chip label={status} size="small" color={status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'primary'} />
      </Box>
      
      <LinearProgress variant="determinate" value={progress} sx={{ height: 10, borderRadius: 5, mb: 2 }} />
      <Typography variant="body2" align="center">{progress}%</Typography>
      
      {fileName && <Typography variant="body2">File: {fileName}</Typography>}
      {fileSize && <Typography variant="body2">Size: {formatBytes(fileSize)}</Typography>}
      {uploadSpeed && <Typography variant="body2">Speed: {formatBytes(uploadSpeed)}/s</Typography>}
      {timeRemaining && <Typography variant="body2">Time remaining: {Math.ceil(timeRemaining / 60)} minutes</Typography>}
      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
    </Paper>
  );
};