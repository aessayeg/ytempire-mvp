import React, { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Collapse,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Link,
} from '@mui/material';
import {
  Close,
  ExpandMore,
  ExpandLess,
  Refresh,
  ContentCopy,
  BugReport,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Info,
  HelpOutline,
  AutoFixHigh,
} from '@mui/icons-material';

interface ErrorAction {
  label: string;
  action: () => void;
  icon?: React.ReactNode;
}

interface ErrorSolution {
  description: string;
  action?: () => void;
}

interface ErrorMessageProps {
  severity?: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  errorCode?: string;
  details?: string;
  solutions?: ErrorSolution[];
  actions?: ErrorAction[];
  onClose?: () => void;
  autoHideDuration?: number;
  showReportButton?: boolean;
  retryable?: boolean;
  onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  severity = 'error',
  title,
  message,
  errorCode,
  details,
  solutions,
  actions,
  onClose,
  autoHideDuration,
  showReportButton = true,
  retryable = false,
  onRetry,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);
  const [visible, setVisible] = useState(true);

  React.useEffect(() => {
    if (autoHideDuration) {
      const timer = setTimeout(() => {
        setVisible(false);
        onClose?.();
      }, autoHideDuration);
      return () => clearTimeout(timer);
    }
  }, [autoHideDuration, onClose]);

  const handleCopyError = () => {
    const errorInfo = `
Error: ${title}
Message: ${message}
Code: ${errorCode || 'N/A'}
Details: ${details || 'N/A'}
Time: ${new Date().toISOString()}
    `.trim();
    
    navigator.clipboard.writeText(errorInfo);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReportBug = () => {
    const errorInfo = encodeURIComponent(`
**Error Report**
- Title: ${title}
- Message: ${message}
- Code: ${errorCode || 'N/A'}
- Time: ${new Date().toISOString()}
    `);
    window.open(`/feedback?type=bug&details=${errorInfo}`, '_blank');
  };

  const getSeverityIcon = () => {
    switch (severity) {
      case 'error':
        return <ErrorIcon />;
      case 'warning':
        return <Warning />;
      case 'info':
        return <Info />;
      case 'success':
        return <CheckCircle />;
    }
  };

  if (!visible) return null;

  return (
    <Collapse in={visible}>
      <Alert
        severity={severity}
        icon={getSeverityIcon()}
        action={
          onClose && (
            <IconButton
              size="small"
              onClick={() => {
                setVisible(false);
                onClose();
              }}
            >
              <Close fontSize="small" />
            </IconButton>
          )
        }
        sx={{
          mb: 2,
          '.MuiAlert-message': { width: '100%' },
        }}
      >
        <AlertTitle sx={{ fontWeight: 'bold' }}>
          {title}
          {errorCode && (
            <Typography
              component="span"
              variant="caption"
              sx={{ ml: 1, opacity: 0.7 }}
            >
              ({errorCode})
            </Typography>
          )}
        </AlertTitle>
        
        <Typography variant="body2" sx={{ mb: 1 }}>
          {message}
        </Typography>

        {/* Quick Actions */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
          {retryable && onRetry && (
            <Button
              size="small"
              startIcon={<Refresh />}
              onClick={onRetry}
              variant="outlined"
              color={severity === 'error' ? 'error' : severity}
            >
              Retry
            </Button>
          )}
          
          {actions?.map((action, index) => (
            <Button
              key={index}
              size="small"
              startIcon={action.icon}
              onClick={action.action}
              variant="outlined"
              color={severity === 'error' ? 'error' : severity}
            >
              {action.label}
            </Button>
          ))}
        </Box>

        {/* Solutions */}
        {solutions && solutions.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1.5, mb: 1, backgroundColor: 'action.hover' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
              <AutoFixHigh fontSize="small" color="primary" />
              <Typography variant="subtitle2" fontWeight="medium">
                Suggested Solutions:
              </Typography>
            </Box>
            <List dense sx={{ p: 0 }}>
              {solutions.map((solution, index) => (
                <ListItem key={index} sx={{ pl: 0 }}>
                  <ListItemIcon sx={{ minWidth: 28 }}>
                    <CheckCircle fontSize="small" color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary={solution.description}
                    primaryTypographyProps={{ variant: 'caption' }}
                  />
                  {solution.action && (
                    <Button
                      size="small"
                      onClick={solution.action}
                      sx={{ ml: 1 }}
                    >
                      Apply
                    </Button>
                  )}
                </ListItem>
              ))}
            </List>
          </Paper>
        )}

        {/* Details Section */}
        {details && (
          <Box>
            <Button
              size="small"
              startIcon={showDetails ? <ExpandLess /> : <ExpandMore />}
              onClick={() => setShowDetails(!showDetails)}
              sx={{ mb: 1 }}
            >
              {showDetails ? 'Hide' : 'Show'} Details
            </Button>
            
            <Collapse in={showDetails}>
              <Paper
                variant="outlined"
                sx={{
                  p: 1.5,
                  backgroundColor: 'grey.50',
                  fontFamily: 'monospace',
                  fontSize: 12,
                  overflowX: 'auto',
                }}
              >
                <pre style={{ margin: 0 }}>{details}</pre>
              </Paper>
            </Collapse>
          </Box>
        )}

        {/* Bottom Actions */}
        <Box sx={{ display: 'flex', gap: 1, mt: 2, pt: 1, borderTop: 1, borderColor: 'divider' }}>
          <Button
            size="small"
            startIcon={copied ? <CheckCircle /> : <ContentCopy />}
            onClick={handleCopyError}
            disabled={copied}
          >
            {copied ? 'Copied!' : 'Copy Error'}
          </Button>
          
          {showReportButton && (
            <Button
              size="small"
              startIcon={<BugReport />}
              onClick={handleReportBug}
            >
              Report Issue
            </Button>
          )}
          
          <Link
            href="/help/troubleshooting"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              ml: 'auto',
              fontSize: 12,
              textDecoration: 'none',
            }}
          >
            <HelpOutline fontSize="small" />
            Troubleshooting Guide
          </Link>
        </Box>
      </Alert>
    </Collapse>
  );
};