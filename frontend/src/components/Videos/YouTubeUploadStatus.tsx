import React from 'react';
import { 
  Box,
  Paper,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Link,
  Chip
 } from '@mui/material';
import {  YouTube  } from '@mui/icons-material';

interface YouTubeUploadStatusProps {
  
status: 'uploading' | 'processing' | 'published' | 'failed';
videoId?: string;
youtubeUrl?: string;
currentStep: number;
error?: string;


}

export const YouTubeUploadStatus: React.FC<YouTubeUploadStatusProps> = ({ status, videoId, youtubeUrl, currentStep, error }) => {
  const steps = ['Upload Video', 'Process on YouTube', 'Set Metadata', 'Publish'];
  
  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" gap={2} mb={3}>
        <YouTube sx={{ color: '#FF0000', fontSize: 32 }} />
        <Typography variant="h6">YouTube Upload Status</Typography>
      <Chip label={status} size="small" color={status === 'published' ? 'success' : status === 'failed' ? 'error' : 'primary'} />
      </Box>
      
      <Stepper activeStep={currentStep}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      {status === 'published' && youtubeUrl && (
        <Alert severity="success" sx={{ mt: 3 }}>
          Video published successfully!
          <Link href={youtubeUrl} target="_blank" sx={{ ml: 1 }}>View on YouTube</Link>
        </Alert>
      )}
      {status === 'failed' && error && (
        <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>
      )}
      {videoId && (
        <Typography variant="body2" sx={{ mt: 2 }}>YouTube Video, ID: {videoId}</Typography>
      )}
    </Paper>
  </>
  )
};
