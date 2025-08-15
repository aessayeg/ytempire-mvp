import React, { useState } from 'react';
import { 
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Rating,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio
 } from '@mui/material';
import {  CheckCircle, Cancel, Edit  } from '@mui/icons-material';

interface VideoApprovalProps {
  video: unknown,
  onApprove: (feedback: React.ChangeEvent<HTMLInputElement>) => void,
  onReject: (reason: string) => void,
  onRequestChanges: (changes: string) => void}

export const VideoApproval: React.FC<VideoApprovalProps> = ({ video, onApprove, onReject, onRequestChanges }) => {
  const [decision, setDecision] = useState<'approve' | 'reject' | 'changes'>('approve');
  const [quality, setQuality] = useState(4);
  const [feedback, setFeedback] = useState('');
  
  const handleSubmit = () => {
    if (decision === 'approve') {
      onApprove({ quality, feedback });
} else if (decision === 'reject') {
      onReject(feedback)} else {
      onRequestChanges(feedback)}
  };

  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Video Approval</Typography>
      <Box mb={2}>
        <Typography>Quality Rating</Typography>
        <Rating value={quality} onChange={(e, v) => setQuality(v || 0} />
      </Box>
      <FormControl component="fieldset">
        <RadioGroup value={decision} onChange={(e) => setDecision(e.target.value as any}>
          <FormControlLabel value="approve" control={<Radio />} label="Approve for Publishing" />
          <FormControlLabel value="changes" control={<Radio />} label="Request Changes" />
          <FormControlLabel value="reject" control={<Radio />} label="Reject" />
        </RadioGroup>
      </FormControl>
      <TextField fullWidth multiline rows={4} label="Feedback" value={feedback} onChange={(e) => setFeedback(e.target.value)} margin="normal" />
      <Box display="flex" gap={2} mt={2}>
        <Button variant="contained" color={decision === 'approve' ? 'success' : decision === 'reject' ? 'error' : 'warning'} onClick={handleSubmit} startIcon={decision === 'approve' ? <CheckCircle /> : decision === 'reject' ? <Cancel /> </>: <Edit />}>
          {decision === 'approve' ? 'Approve' : decision === 'reject' ? 'Reject' : 'Request Changes'}
        </Button>
      </Box>
    </Paper>
  </>
  )};