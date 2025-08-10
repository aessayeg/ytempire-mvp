/**
 * Component Library Expansion
 * Reusable UI components for consistent design across the application
 */

import React, { ReactNode, useState, useEffect, useRef } from 'react';
import {
  Box,
  Button as MuiButton,
  ButtonProps,
  Card as MuiCard,
  CardProps,
  CircularProgress,
  Skeleton,
  Typography,
  IconButton,
  Paper,
  Alert,
  AlertProps,
  Chip,
  ChipProps,
  Badge,
  BadgeProps,
  Avatar,
  AvatarProps,
  Tooltip,
  TooltipProps,
  Fade,
  Grow,
  Zoom,
  Slide,
  Modal,
  Backdrop,
  styled,
  keyframes,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Check,
  Close,
  Info,
  Warning,
  Error as ErrorIcon,
  ContentCopy,
  CheckCircle,
  RadioButtonUnchecked,
  CloudUpload,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';

// ============= Animations =============
const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(33, 150, 243, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(33, 150, 243, 0);
  }
`;

const shimmer = keyframes`
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
`;

const rotate = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

// ============= Loading Components =============
export const LoadingButton: React.FC<ButtonProps & { loading?: boolean }> = ({
  loading,
  children,
  disabled,
  startIcon,
  ...props
}) => {
  return (
    <MuiButton
      {...props}
      disabled={disabled || loading}
      startIcon={loading ? <CircularProgress size={20} /> : startIcon}
    >
      {children}
    </MuiButton>
  );
};

export const SkeletonCard: React.FC<{ height?: number; animate?: boolean }> = ({
  height = 200,
  animate = true,
}) => {
  return (
    <MuiCard sx={{ height, overflow: 'hidden' }}>
      <Skeleton
        variant="rectangular"
        height={height}
        animation={animate ? 'pulse' : false}
        sx={{
          background: animate
            ? `linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)`
            : undefined,
          animation: animate ? `${shimmer} 2s infinite` : undefined,
        }}
      />
    </MuiCard>
  );
};

export const LoadingOverlay: React.FC<{ open: boolean; message?: string }> = ({
  open,
  message = 'Loading...',
}) => {
  return (
    <Backdrop
      sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
      open={open}
    >
      <Box textAlign="center">
        <CircularProgress color="inherit" />
        {message && (
          <Typography variant="body1" sx={{ mt: 2 }}>
            {message}
          </Typography>
        )}
      </Box>
    </Backdrop>
  );
};

// ============= Status Components =============
export interface StatusBadgeProps extends BadgeProps {
  status: 'online' | 'offline' | 'busy' | 'away';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, children, ...props }) => {
  const colors = {
    online: '#44b700',
    offline: '#757575',
    busy: '#f44336',
    away: '#ff9800',
  };

  const StyledBadge = styled(Badge)(({ theme }) => ({
    '& .MuiBadge-badge': {
      backgroundColor: colors[status],
      color: colors[status],
      boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
      '&::after': {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        borderRadius: '50%',
        animation: status === 'online' ? `${pulse} 1.5s infinite` : undefined,
        border: '1px solid currentColor',
        content: '""',
      },
    },
  }));

  return (
    <StyledBadge
      overlap="circular"
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      variant="dot"
      {...props}
    >
      {children}
    </StyledBadge>
  );
};

export const StatusChip: React.FC<ChipProps & { status: string }> = ({ status, ...props }) => {
  const getColor = () => {
    switch (status.toLowerCase()) {
      case 'active':
      case 'success':
      case 'completed':
        return 'success';
      case 'pending':
      case 'processing':
      case 'warning':
        return 'warning';
      case 'error':
      case 'failed':
      case 'cancelled':
        return 'error';
      default:
        return 'default';
    }
  };

  return <Chip size="small" label={status} color={getColor()} {...props} />;
};

// ============= Card Components =============
export interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: ReactNode;
  color?: string;
  loading?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  icon,
  color = '#2196F3',
  loading,
}) => {
  const theme = useTheme();

  if (loading) {
    return (
      <MuiCard>
        <Box p={2}>
          <Skeleton variant="text" width="60%" />
          <Skeleton variant="text" width="40%" height={40} />
          <Skeleton variant="text" width="30%" />
        </Box>
      </MuiCard>
    );
  }

  return (
    <MuiCard>
      <Box p={2}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography color="text.secondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4">{value}</Typography>
            {change !== undefined && (
              <Typography
                variant="body2"
                color={change >= 0 ? 'success.main' : 'error.main'}
                sx={{ mt: 1 }}
              >
                {change >= 0 ? '+' : ''}{change}%
              </Typography>
            )}
          </Box>
          {icon && (
            <Box
              sx={{
                backgroundColor: alpha(color, 0.1),
                borderRadius: 2,
                p: 1.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {React.cloneElement(icon as React.ReactElement, {
                sx: { color, fontSize: 32 },
              })}
            </Box>
          )}
        </Box>
      </Box>
    </MuiCard>
  );
};

export const GlassCard = styled(MuiCard)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

// ============= Input Components =============
export interface CopyableTextProps {
  text: string;
  variant?: 'body1' | 'body2' | 'caption';
  showIcon?: boolean;
}

export const CopyableText: React.FC<CopyableTextProps> = ({
  text,
  variant = 'body2',
  showIcon = true,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Typography variant={variant} sx={{ fontFamily: 'monospace' }}>
        {text}
      </Typography>
      {showIcon && (
        <IconButton size="small" onClick={handleCopy}>
          {copied ? <Check fontSize="small" /> : <ContentCopy fontSize="small" />}
        </IconButton>
      )}
    </Box>
  );
};

export interface FileUploadProps {
  onFileSelect: (files: FileList) => void;
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // in MB
  disabled?: boolean;
}

export const FileUploadButton: React.FC<FileUploadProps> = ({
  onFileSelect,
  accept,
  multiple,
  maxSize,
  disabled,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    if (maxSize) {
      const oversizedFiles = Array.from(files).filter(
        (file) => file.size > maxSize * 1024 * 1024
      );
      if (oversizedFiles.length > 0) {
        setError(`File size must be less than ${maxSize}MB`);
        return;
      }
    }

    setError(null);
    onFileSelect(files);
  };

  return (
    <Box>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <MuiButton
        variant="outlined"
        startIcon={<CloudUpload />}
        onClick={handleClick}
        disabled={disabled}
      >
        Upload File
      </MuiButton>
      {error && (
        <Typography variant="caption" color="error" sx={{ mt: 1 }}>
          {error}
        </Typography>
      )}
    </Box>
  );
};

// ============= Notification Components =============
export interface ToastProps extends AlertProps {
  open: boolean;
  message: string;
  autoHideDuration?: number;
  onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({
  open,
  message,
  autoHideDuration = 6000,
  onClose,
  severity = 'info',
  ...props
}) => {
  useEffect(() => {
    if (open && autoHideDuration) {
      const timer = setTimeout(onClose, autoHideDuration);
      return () => clearTimeout(timer);
    }
  }, [open, autoHideDuration, onClose]);

  return (
    <Slide direction="up" in={open} mountOnEnter unmountOnExit>
      <Alert
        severity={severity}
        onClose={onClose}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 9999,
          minWidth: 300,
        }}
        {...props}
      >
        {message}
      </Alert>
    </Slide>
  );
};

// ============= Progress Components =============
export interface StepperProps {
  steps: string[];
  activeStep: number;
  completed?: number[];
}

export const ProgressStepper: React.FC<StepperProps> = ({ steps, activeStep, completed = [] }) => {
  return (
    <Box display="flex" alignItems="center">
      {steps.map((step, index) => (
        <React.Fragment key={step}>
          <Box display="flex" flexDirection="column" alignItems="center">
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor:
                  completed.includes(index) || index < activeStep
                    ? 'primary.main'
                    : index === activeStep
                    ? 'primary.light'
                    : 'grey.300',
                color: 'white',
                fontWeight: 'bold',
              }}
            >
              {completed.includes(index) ? (
                <Check fontSize="small" />
              ) : (
                <Typography variant="caption">{index + 1}</Typography>
              )}
            </Box>
            <Typography
              variant="caption"
              sx={{
                mt: 0.5,
                color: index <= activeStep ? 'text.primary' : 'text.secondary',
              }}
            >
              {step}
            </Typography>
          </Box>
          {index < steps.length - 1 && (
            <Box
              sx={{
                flex: 1,
                height: 2,
                mx: 1,
                backgroundColor: index < activeStep ? 'primary.main' : 'grey.300',
              }}
            />
          )}
        </React.Fragment>
      ))}
    </Box>
  );
};

export const CircularProgressWithLabel: React.FC<{ value: number; size?: number }> = ({
  value,
  size = 60,
}) => {
  return (
    <Box position="relative" display="inline-flex">
      <CircularProgress variant="determinate" value={value} size={size} />
      <Box
        position="absolute"
        top="50%"
        left="50%"
        sx={{
          transform: 'translate(-50%, -50%)',
        }}
      >
        <Typography variant="caption" component="div" color="text.secondary">
          {`${Math.round(value)}%`}
        </Typography>
      </Box>
    </Box>
  );
};

// ============= Data Display Components =============
export interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, description, icon, action }) => {
  return (
    <Paper
      sx={{
        p: 4,
        textAlign: 'center',
        backgroundColor: 'background.default',
      }}
    >
      {icon && (
        <Box sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }}>
          {icon}
        </Box>
      )}
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      {description && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {description}
        </Typography>
      )}
      {action && <Box mt={2}>{action}</Box>}
    </Paper>
  );
};

export interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  icon,
  trend,
  trendValue,
}) => {
  const getTrendColor = () => {
    switch (trend) {
      case 'up': return 'success.main';
      case 'down': return 'error.main';
      default: return 'text.secondary';
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="flex-start">
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {label}
          </Typography>
          <Typography variant="h5">{value}</Typography>
          {trend && trendValue && (
            <Typography variant="caption" color={getTrendColor()}>
              {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'} {trendValue}
            </Typography>
          )}
        </Box>
        {icon && (
          <Box sx={{ color: 'primary.main', opacity: 0.6 }}>
            {icon}
          </Box>
        )}
      </Box>
    </Paper>
  );
};

// ============= Interactive Components =============
export interface ToggleButtonGroupProps {
  options: { value: string; label: string; icon?: ReactNode }[];
  value: string;
  onChange: (value: string) => void;
  exclusive?: boolean;
}

export const StyledToggleButtonGroup: React.FC<ToggleButtonGroupProps> = ({
  options,
  value,
  onChange,
  exclusive = true,
}) => {
  return (
    <Paper sx={{ p: 0.5, display: 'inline-flex' }}>
      {options.map((option) => (
        <MuiButton
          key={option.value}
          size="small"
          variant={value === option.value ? 'contained' : 'text'}
          onClick={() => onChange(option.value)}
          startIcon={option.icon}
          sx={{ mx: 0.5 }}
        >
          {option.label}
        </MuiButton>
      ))}
    </Paper>
  );
};

// ============= Animation Wrappers =============
export const FadeIn: React.FC<{ children: ReactNode; delay?: number }> = ({
  children,
  delay = 0,
}) => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShow(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <Fade in={show} timeout={1000}>
      <Box>{children}</Box>
    </Fade>
  );
};

export const SlideIn: React.FC<{
  children: ReactNode;
  direction?: 'left' | 'right' | 'up' | 'down';
}> = ({ children, direction = 'up' }) => {
  return (
    <Slide direction={direction} in={true} timeout={500}>
      <Box>{children}</Box>
    </Slide>
  );
};

// ============= Utility Components =============
export const Divider: React.FC<{ text?: string }> = ({ text }) => {
  return (
    <Box display="flex" alignItems="center" my={2}>
      <Box flex={1} height={1} bgcolor="divider" />
      {text && (
        <>
          <Typography variant="body2" color="text.secondary" sx={{ mx: 2 }}>
            {text}
          </Typography>
          <Box flex={1} height={1} bgcolor="divider" />
        </>
      )}
    </Box>
  );
};

export const PasswordField: React.FC<{
  value: string;
  onChange: (value: string) => void;
  label?: string;
}> = ({ value, onChange, label = 'Password' }) => {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <Box position="relative">
      <input
        type={showPassword ? 'text' : 'password'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={label}
        style={{
          width: '100%',
          padding: '12px',
          paddingRight: '40px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          fontSize: '16px',
        }}
      />
      <IconButton
        size="small"
        onClick={() => setShowPassword(!showPassword)}
        sx={{
          position: 'absolute',
          right: 8,
          top: '50%',
          transform: 'translateY(-50%)',
        }}
      >
        {showPassword ? <VisibilityOff /> : <Visibility />}
      </IconButton>
    </Box>
  );
};