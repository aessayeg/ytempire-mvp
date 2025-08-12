import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Box, Container, Button } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { VideoEditor } from '../../components/VideoEditor/VideoEditor';

const VideoEditorPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const handleSave = (editedVideo: any) => {
    console.log('Saving edited video:', editedVideo);
    // TODO: Implement API call to save edited video
    navigate(`/videos/${id}`);
  };

  const handleExport = (format: string) => {
    console.log('Exporting video in format:', format);
    // TODO: Implement export functionality
  };

  return (
    <Container maxWidth={false} sx={{ py: 3 }}>
      <Box sx={{ mb: 2 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(-1)}
          sx={{ mb: 2 }}
        >
          Back to Videos
        </Button>
      </Box>
      
      <VideoEditor
        videoId={id}
        videoUrl={`/api/v1/videos/${id}/stream`} // TODO: Get actual video URL
        onSave={handleSave}
        onExport={handleExport}
      />
    </Container>
  );
};

export default VideoEditorPage;