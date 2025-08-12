import React from 'react';
import {
  Backdrop,
  CircularProgress,
  Box,
  Typography,
  LinearProgress,
  Fade,
  Paper,
} from '@mui/material';
import { keyframes } from '@emotion/react';

interface LoadingOverlayProps {
  open: boolean;
  message?: string;
  progress?: number;
  variant?: 'circular' | 'linear' | 'dots' | 'pulse';
  fullScreen?: boolean;
  transparent?: boolean;
  showCancel?: boolean;
  onCancel?: () => void;
}

// Animated dots
const dotPulse = keyframes`
  0%, 60%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  30% {
    transform: scale(1.5);
    opacity: 0.7;
  }
`;

// Pulse animation
const pulseAnimation = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  open,
  message,
  progress,
  variant = 'circular',
  fullScreen = true,
  transparent = false,
  showCancel = false,
  onCancel,
}) => {
  const renderLoader = () => {
    switch (variant) {
      case 'linear':
        return (
          <Box sx={{ width: '100%', maxWidth: 400 }}>
            <LinearProgress
              variant={progress !== undefined ? 'determinate' : 'indeterminate'}
              value={progress}
              sx={{ mb: message ? 2 : 0 }}
            />
            {progress !== undefined && (
              <Typography variant="body2" color="text.secondary" align="center">
                {Math.round(progress)}%
              </Typography>
            )}
          </Box>
        );

      case 'dots':
        return (
          <Box sx={{ display: 'flex', gap: 1 }}>
            {[0, 1, 2].map((index) => (
              <Box
                key={index}
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: 'primary.main',
                  animation: `${dotPulse} 1.4s ease-in-out infinite`,
                  animationDelay: `${index * 0.16}s`,
                }}
              />
            ))}
          </Box>
        );

      case 'pulse':
        return (
          <Box
            sx={{
              width: 80,
              height: 80,
              borderRadius: '50%',
              backgroundColor: 'primary.main',
              animation: `${pulseAnimation} 2s ease-in-out infinite`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <CircularProgress
              size={60}
              sx={{ color: 'white' }}
              variant={progress !== undefined ? 'determinate' : 'indeterminate'}
              value={progress}
            />
          </Box>
        );

      default:
        return (
          <CircularProgress
            size={60}
            variant={progress !== undefined ? 'determinate' : 'indeterminate'}
            value={progress}
          />
        );
    }
  };

  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
      }}
    >
      {renderLoader()}
      {message && (
        <Fade in={true} timeout={500}>
          <Typography
            variant="body1"
            color={fullScreen ? 'white' : 'text.primary'}
            align="center"
            sx={{ maxWidth: 300 }}
          >
            {message}
          </Typography>
        </Fade>
      )}
      {showCancel && onCancel && (
        <Fade in={true} timeout={500}>
          <Typography
            variant="body2"
            color={fullScreen ? 'white' : 'primary.main'}
            sx={{ cursor: 'pointer', textDecoration: 'underline', mt: 1 }}
            onClick={onCancel}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onCancel()}
          >
            Cancel
          </Typography>
        </Fade>
      )}
    </Box>
  );

  if (!fullScreen) {
    return (
      <Fade in={open}>
        <Paper
          elevation={3}
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            p: 4,
            zIndex: 1300,
            backgroundColor: transparent ? 'transparent' : 'background.paper',
          }}
        >
          {content}
        </Paper>
      </Fade>
    );
  }

  return (
    <Backdrop
      sx={{
        color: '#fff',
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: transparent ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.7)',
      }}
      open={open}
    >
      {content}
    </Backdrop>
  );
};

// Inline loading indicator
export const InlineLoader: React.FC<{
  loading: boolean;
  size?: 'small' | 'medium' | 'large';
  message?: string;
}> = ({ loading, size = 'medium', message }) => {
  if (!loading) return null;

  const sizeMap = {
    small: 16,
    medium: 24,
    large: 32,
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <CircularProgress size={sizeMap[size]} />
      {message && (
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
      )}
    </Box>
  );
};

// Loading placeholder for images
export const ImageLoadingPlaceholder: React.FC<{
  width?: number | string;
  height?: number | string;
  borderRadius?: number | string;
}> = ({ width = '100%', height = 200, borderRadius = 0 }) => {
  return (
    <Box
      sx={{
        width,
        height,
        borderRadius,
        backgroundColor: 'grey.200',
        backgroundImage: `linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)`,
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s infinite',
        '@keyframes shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      }}
    />
  );
};