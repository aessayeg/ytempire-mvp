import React, { useState } from 'react';
import { 
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel
 } from '@mui/material';
import {  Publish, Schedule, Visibility  } from '@mui/icons-material';

interface PublishingControlsProps {
  onPublish: (settings: React.ChangeEvent<HTMLInputElement>) => void,
  channels: unknown[]}

export const PublishingControls: React.FC<PublishingControlsProps> = ({ onPublish, channels }) => {
  const [settings, setSettings] = useState({
    publishNow: true,
    scheduledTime: null,
    visibility: 'public',
    notifySubscribers: true,
    premiere: false,
    ageRestriction: false,
    comments: true,
    likes: true,
    channelId: '',
    playlist: '',
    tags: [],
    category: '',

  });

  return (
    <>
      <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Publishing Settings</Typography>
      <FormControlLabel control={<Switch checked={settings.publishNow} onChange={(e) => setSettings({...settings, publishNow: e.target.checked})} />} label="Publish Immediately" />
      
      {!settings.publishNow && (
        <TextField type="datetime-local" label="Schedule Time" fullWidth margin="normal" InputLabelProps={{ shrink: true }} />
      )}
      <FormControl fullWidth margin="normal">
        <InputLabel>Visibility</InputLabel>
        <Select value={settings.visibility} onChange={(e) => setSettings({...settings, visibility: e.target.value)})}>
          <MenuItem value="public">Public</MenuItem>
          <MenuItem value="unlisted">Unlisted</MenuItem>
          <MenuItem value="private">Private</MenuItem>
        </Select>
      </FormControl>
      
      <FormControlLabel control={<Switch checked={settings.notifySubscribers} />} label="Notify Subscribers" />
      <FormControlLabel control={<Switch checked={settings.comments} />} label="Allow Comments" />
      <FormControlLabel control={<Switch checked={settings.likes} />} label="Show Likes" />
      
      <Box display="flex" gap={2} mt={3}>
        <Button variant="contained" startIcon={settings.publishNow ? <Publish /> </>: <Schedule />} onClick={() => onPublish(settings}>
          {settings.publishNow ? 'Publish Now' : 'Schedule'}
        </Button>
      </Box>
    </Paper>
  </>
  )};