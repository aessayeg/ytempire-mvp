import React, { useState } from 'react';
import { 
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Checkbox,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  LinearProgress,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  Avatar
 } from '@mui/material';
import { 
  YouTube,
  Edit,
  Delete,
  Pause,
  PlayArrow,
  Schedule,
  Settings,
  CloudUpload,
  ContentCopy,
  Label,
  MonetizationOn,
  Update
 } from '@mui/icons-material';

interface Channel {
  id: string,
  name: string,

  status: 'active' | 'paused' | 'error',
  videos: number,

  subscribers: number,
  health: number;
  selected?: boolean;
}

interface BulkAction {
  id: string,
  label: string,

  icon: React.ReactNode,
  action: (channels: string[]) => void;
  requiresConfirmation?: boolean;
  dangerous?: boolean;
}

export const BulkOperations: React.FC = () => {
  const [channels, setChannels] = useState<Channel[]>([ { id: '1', name: 'Tech Insights', status: 'active', videos: 234, subscribers: 125000, health: 85 },
    { id: '2', name: 'AI Daily', status: 'active', videos: 189, subscribers: 89000, health: 92 },
    { id: '3', name: 'Future Tech', status: 'paused', videos: 156, subscribers: 67000, health: 78 },
    { id: '4', name: 'Coding Tips', status: 'active', videos: 342, subscribers: 234000, health: 95 },
    { id: '5', name: 'Tech Reviews', status: 'error', videos: 89, subscribers: 45000, health: 45 } ]);

  const [selectedChannels, setSelectedChannels] = useState<string[]>([]);
  const [actionDialog, setActionDialog] = useState(false);
  const [currentAction, setCurrentAction] = useState<BulkAction | null>(null);
  const [processing, setProcessing] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  // Bulk Actions
  const bulkActions: BulkAction[] = [ { id: 'activate',
      label: 'Activate Channels',
      icon: <PlayArrow />,
      action: (channelIds) => handleBulkStatusChange(channelIds, 'active') },
    { id: 'pause',
      label: 'Pause Channels',
      icon: <Pause />,
      action: (channelIds) => handleBulkStatusChange(channelIds, 'paused') },
    { id: 'schedule',
      label: 'Bulk Schedule Videos',
      icon: <Schedule />,
      action: (channelIds) => handleBulkSchedule(channelIds) },
    { id: 'upload',
      label: 'Bulk Upload Settings',
      icon: <CloudUpload />,
      action: (channelIds) => handleBulkUploadSettings(channelIds) },
    { id: 'monetization',
      label: 'Update Monetization',
      icon: <MonetizationOn />,
      action: (channelIds) => handleBulkMonetization(channelIds) },
    { id: 'tags',
      label: 'Update Tags & Categories',
      icon: <Label />,
      action: (channelIds) => handleBulkTags(channelIds) },
    { id: 'duplicate',
      label: 'Duplicate Settings',
      icon: <ContentCopy />,
      action: (channelIds) => handleDuplicateSettings(channelIds) },
    { id: 'delete',
      label: 'Delete Channels',
      icon: <Delete />,
      action: (channelIds) => handleBulkDelete(channelIds),
      requiresConfirmation: true,
      dangerous: true } ];

  const handleSelectAll = (_: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      setSelectedChannels(channels.map(c => c.id))} else {
      setSelectedChannels([])}
  };

  const handleSelectChannel = (channelId: string) => {
    setSelectedChannels(prev => {}
      prev.includes(channelId)
        ? prev.filter(id => id !== channelId)
        : [...prev, channelId]
    )};

  const handleBulkAction = (_action: BulkAction) => {
    if (selectedChannels.length === 0) {
      alert('Please select at least one channel');
      return;
    }
    setCurrentAction(action);
    setActionDialog(true)};

  const handleBulkStatusChange = async (channelIds: string[], status: string) => {
    setProcessing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setChannels(prev => prev.map(c => {}
      channelIds.includes(c.id) ? { ...c, status: status as any } : c
    ));
    setProcessing(false);
    setActionDialog(false);
    setSelectedChannels([])};

  const handleBulkSchedule = async (_channelIds: string[]) => {
    setProcessing(true);
    // Implementation for bulk scheduling
    await new Promise(resolve => setTimeout(resolve, 2000));
    setProcessing(false);
    setActionDialog(false)};

  const handleBulkUploadSettings = async (_channelIds: string[]) => {
    setProcessing(true);
    // Implementation for bulk upload settings
    await new Promise(resolve => setTimeout(resolve, 2000));
    setProcessing(false);
    setActionDialog(false)};

  const handleBulkMonetization = async (_channelIds: string[]) => {
    setProcessing(true);
    // Implementation for bulk monetization
    await new Promise(resolve => setTimeout(resolve, 2000));
    setProcessing(false);
    setActionDialog(false)};

  const handleBulkTags = async (_channelIds: string[]) => {
    setProcessing(true);
    // Implementation for bulk tags
    await new Promise(resolve => setTimeout(resolve, 2000));
    setProcessing(false);
    setActionDialog(false)};

  const handleDuplicateSettings = async (_channelIds: string[]) => {
    setProcessing(true);
    // Implementation for duplicate settings
    await new Promise(resolve => setTimeout(resolve, 2000));
    setProcessing(false);
    setActionDialog(false)};

  const handleBulkDelete = async (channelIds: string[]) => {
    setProcessing(true);
    // Implementation for bulk delete
    await new Promise(resolve => setTimeout(resolve, 2000));
    setChannels(prev => prev.filter(c => !channelIds.includes(c.id)));
    setProcessing(false);
    setActionDialog(false);
    setSelectedChannels([])};

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'error': return 'error';
      default: return 'default'}
  };

  return (
    <>
      <Box>
      {/* Header with Actions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5" fontWeight="bold">
              Channel Bulk Operations
            </Typography>
      <Badge badgeContent={selectedChannels.length} color="primary">
              <Chip
                label={`${selectedChannels.length} selected`}
                color={selectedChannels.length > 0 ? 'primary' : 'default'}
              />
            </Badge>
          </Box>

          {selectedChannels.length > 0 && (
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {bulkActions.map(action => (
                <Button
                  key={action.id}
                  variant={action.dangerous ? 'outlined' : 'contained'}
                  color={action.dangerous ? 'error' : 'primary'}
                  size="small"
                  startIcon={action.icon}
                  onClick={() => handleBulkAction(action}
                >
                  {action.label}
                </Button>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Channels Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={selectedChannels.length > 0 && selectedChannels.length < channels.length}
                  checked={channels.length > 0 && selectedChannels.length === channels.length}
                  onChange={handleSelectAll}
                />
              </TableCell>
              <TableCell>Channel</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Videos</TableCell>
              <TableCell align="right">Subscribers</TableCell>
              <TableCell align="right">Health</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {channels.map(channel => (
              <TableRow
                key={channel.id}
                selected={selectedChannels.includes(channel.id)}
                hover
              >
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedChannels.includes(channel.id)}
                    onChange={() => handleSelectChannel(channel.id}
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Avatar sx={{ width: 32, height: 32 }}>
                      <YouTube />
                    </Avatar>
                    <Typography variant="body2" fontWeight="medium">
                      {channel.name}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip
                    label={channel.status}
                    size="small"
                    color={getStatusColor(channel.status)}
                  />
                </TableCell>
                <TableCell align="right">{channel.videos}</TableCell>
                <TableCell align="right">{channel.subscribers.toLocaleString()}</TableCell>
                <TableCell align="right">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={channel.health}
                      sx={{ width: 60, height: 6, borderRadius: 1 }}
                      color={channel.health > 80 ? 'success' : channel.health > 50 ? 'warning' : 'error'}
                    />
                    <Typography variant="caption">{channel.health}%</Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <IconButton size="small">
                    <Edit fontSize="small" />
                  </IconButton>
                  <IconButton size="small">
                    <Settings fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Bulk Action Dialog */}
      <Dialog
        open={actionDialog}
        onClose={() => !processing && setActionDialog(false}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {currentAction?.label}
        </DialogTitle>
        <DialogContent>
          {currentAction?.requiresConfirmation && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              This action cannot be undone. Are you sure you want to proceed?
            </Alert>
          )}
          <Typography variant="body2" gutterBottom>
            This action will be applied to {selectedChannels.length} channel(s):
          </Typography>
          
          <List dense>
            {selectedChannels.map(id => {
              const channel = channels.find(c => c.id === id</>
  );
              return channel ? (
                <ListItem key={id}>
                  <ListItemIcon>
                    <YouTube />
                  </ListItemIcon>
                  <ListItemText primary={channel.name} />
                </ListItem>
              ) : null});
}
          </List>

          {processing && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                Processing bulk operation...
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActionDialog(false} disabled={processing}>
            Cancel
          </Button>
          <Button
            variant="contained"
            color={currentAction?.dangerous ? 'error' : 'primary'}
            onClick={() => currentAction?.action(selectedChannels}
            disabled={processing}
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )};