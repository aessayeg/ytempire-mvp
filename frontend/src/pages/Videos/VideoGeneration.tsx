import React from 'react';
import { Container, Box, Typography, Button } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from 'react-router-dom';
import { VideoGenerationForm } from '../../components/Videos/VideoGenerationForm';

const VideoGenerationPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 3 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/videos')}
          sx={{ mb: 2 }}
        >
          Back to Videos
        </Button>
        
        <Typography variant="h4" component="h1" gutterBottom>
          Generate New Video
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Use AI to generate professional videos automatically
        </Typography>
        
        <VideoGenerationForm />
      </Box>
    </Container>
  );
};

export default VideoGenerationPage;