import React, { useState, useEffect } from 'react';
import { 
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  IconButton,
  Alert,
  Chip
 } from '@mui/material';
import { 
  GetApp,
  Close,
  PhoneIphone,
  OfflinePin,
  Speed,
  Notifications
 } from '@mui/icons-material';
import {  usePWA  } from '../../contexts/PWAContext';

export const InstallPrompt: React.FC = () => {
  const { isInstallable, installApp, isInstalled } = usePWA();
  const [open, setOpen] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Check if user has previously dismissed the prompt
    const wasDismissed = localStorage.getItem('pwa-prompt-dismissed');
    if (wasDismissed) {
      setDismissed(true)}

    // Show prompt after 30 seconds if installable and not dismissed
    const timer = setTimeout(() => {
      if (isInstallable && !wasDismissed && !isInstalled) {
        setOpen(true)}
    }, 30000);

    return () => clearTimeout(timer)}, [isInstallable, isInstalled]);

  const handleInstall = async () => {
    await installApp();
    setOpen(false)
};
  const handleDismiss = () => {
    setOpen(false);
    setDismissed(true);
    localStorage.setItem('pwa-prompt-dismissed', 'true')
};
  const handleRemindLater = () => {
    setOpen(false);
    // Show again in next session
  };
  if (!isInstallable || dismissed || isInstalled) {
    return null
  }

  return (
    <>
      {/* Floating install button */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          zIndex: 1000,
          display: { xs: 'block', md: 'none' }
        }}
      >
        <Button
          variant="contained"
          color="primary"
          startIcon={<GetApp />}
          onClick={() => setOpen(true}
          sx={ {
            borderRadius: 20,
            boxShadow: 3 }}
        >
          Install App
        </Button>
      </Box>

      {/* Install dialog */}
      <Dialog
        open={open}
        onClose={handleRemindLater}
        maxWidth="sm"
        fullWidth
        PaperProps={ {
          sx: {
  borderRadius: 2,
            backgroundImage: 'linear-gradient(135 deg, #667eea0%, #764ba2100%)',
            color: 'white' }
        }}
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h5" fontWeight="bold">
              Install YTEmpire
            </Typography>
            <IconButton
              edge="end"
              color="inherit"
              onClick={handleDismiss}
              aria-label="close"
            >
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>

        <DialogContent>
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Box
              sx={ {
                width: 80,
                height: 80,
                margin: '0 auto',
                mb: 3,
                backgroundColor: 'white',
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center' }}
            >
              <Typography variant="h3" sx={{ color: '#667eea' }}>
                YT
              </Typography>
            </Box>

            <Typography variant="body1" sx={{ mb: 3 }}>
              Install YTEmpire on your device for a better experience
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <OfflinePin />
                <Typography variant="body2">Work offline with cached data</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Speed />
                <Typography variant="body2">Faster loading and performance</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Notifications />
                <Typography variant="body2">Get push notifications</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <PhoneIphone />
                <Typography variant="body2">Add to home screen</Typography>
              </Box>
            </Box>

            <Alert
              severity="info"
              sx={ {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                color: 'white',
                '& .MuiAlert-icon': {
                  color: 'white' }
              }}
            >
              No app store needed. Installs directly from your browser!
            </Alert>
          </Box>
        </DialogContent>

        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button
            onClick={handleRemindLater}
            sx={{ color: 'white' }}
          >
            Maybe Later
          </Button>
          <Button
            onClick={handleInstall}
            variant="contained"
            sx={ {
              backgroundColor: 'white',
              color: '#667eea',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.9)' }
            }}
            startIcon={<GetApp />}
          >
            Install Now
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
};
// Offline indicator component
export const OfflineIndicator: React.FC = () => {
  const { isOnline, offlineReady } = usePWA();

  if (isOnline) {
    return null
  }

  return (
    <Chip
      label={offlineReady ? 'Offline Mode' : 'No Connection'}
      color={offlineReady ? 'warning' : 'error'}
      size="small"
      sx={ {
        position: 'fixed',
        top: 70,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1300 }}
    />
  )
};
