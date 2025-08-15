import React, { useState } from 'react';
import { 
  Box,
  Typography,
  Button,
  Paper,
  Collapse,
  Alert,
  AlertTitle,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails
 } from '@mui/material';
import { 
  Error as ErrorIcon,
  Refresh,
  ExpandMore,
  BugReport,
  ContentCopy,
  Home
 } from '@mui/icons-material';
import {  useNavigate  } from 'react-router-dom';

interface ErrorFallbackProps {
  error: Error,
  errorInfo: ErrorInfo | null,

  onReset: () => void;
  level?: 'page' | 'section' | 'component';
  showDetails?: boolean;
  errorCount?: number;
  isolate?: boolean;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, errorInfo, onReset, level = 'component', showDetails = process.env.NODE_ENV === 'development', errorCount = 0, isolate = false }) => {
  const navigate = useNavigate();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopyError = () => {
    const errorText = `
Error: ${error.message}
Stack: ${error.stack}
Component, Stack: ${errorInfo?.componentStack || 'N/A'}`
    `.trim();

    navigator.clipboard.writeText(errorText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000)};

  const handleGoHome = () => {
    navigate('/dashboard');
    onReset()};

  // Component-level error (small, inline)
  if (level === 'component') {
    return (
    <>
      <Alert
        severity="error"
        action={
          <IconButton
            color="inherit"
            size="small"
            onClick={onReset}
            aria-label="Retry"
          >
            <Refresh />
          </IconButton>
        }
      >
        <AlertTitle>Component Error</AlertTitle>
        Something went wrong. {errorCount > 2 && 'Multiple attempts failed.'}
      </Alert>
    )}

  // Section-level error
  if (level === 'section') {
    return (
    <Paper
        elevation={0}
        sx={ {
          p: 3,
          backgroundColor: 'error.lighter',
          border: 1,
          borderColor: 'error.light',
          borderRadius: 2 }}
      >
        <Box
          sx={ {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2 }}
        >
          <ErrorIcon color="error" sx={{ fontSize: 48 }} />
          <Typography variant="h6" color="error" align="center">
            This section couldn't load
          </Typography>
      <Typography variant="body2" color="text.secondary" align="center">
            {error.message || 'An unexpected error occurred'}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              onClick={onReset}
              startIcon={<Refresh />}
              size="small"
            >
              Try Again
            </Button>
            {showDetails && (
              <Button
                variant="outlined"
                onClick={() => setDetailsOpen(!detailsOpen}
                size="small"
              >
                {detailsOpen ? 'Hide' : 'Show'} Details
              </Button>
            )}
          </Box>
          {showDetails && (
            <Collapse in={detailsOpen}>
              <Box
                sx={ {
                  mt: 2,
                  p: 2,
                  backgroundColor: 'grey.100',
                  borderRadius: 1,
                  maxWidth: 600,
                  maxHeight: 200,
                  overflow: 'auto' }}
              >
                <Typography
                  variant="caption"
                  component="pre"
                  sx={{ fontFamily: 'monospace' }}
                >
                  {error.stack}
                </Typography>
              </Box>
            </Collapse>
          )}
        </Box>
      </Paper>
    )}

  // Page-level error (full, page)
  return (
    <>
      <Box
      sx={ {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: isolate ? '100 vh' : '60 vh',
        p: 3 }}
    >
      <Paper
        elevation={3}
        sx={ {
          p: 4,
          maxWidth: 600,
          width: '100%',
          textAlign: 'center' }}
      >
        <ErrorIcon
          color="error"
          sx={{ fontSize: 80, mb: 2 }}
        />
        <Typography variant="h4" gutterBottom>
          Oops! Something went wrong
        </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
          We're, sorry, but something unexpected happened. Please try refreshing the page or contact support if the problem persists.
        </Typography>

        {errorCount > 2 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            This error has occurred multiple times. You may need to clear your browser cache or contact support.
          </Alert>
        )}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3 }}>
          <Button
            variant="contained"
            onClick={onReset}
            startIcon={<Refresh />}
          >
            Try Again
          </Button>
          <Button
            variant="outlined"
            onClick={handleGoHome}
            startIcon={<Home />}
          >
            Go to Dashboard
          </Button>
        </Box>

        {showDetails && (
          <Accordion>
            <AccordionSummary
              expandIcon={<ExpandMore />}
              aria-controls="error-details"
              id="error-details-header"
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <BugReport fontSize="small" />
                <Typography>Error Details</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ textAlign: 'left' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" color="error">
                    Error, Message:
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={handleCopyError}
                    aria-label="Copy error details"
                  >
                    <ContentCopy fontSize="small" />
                  </IconButton>
                </Box>
                <Typography
                  variant="body2"
                  sx={ {
                    p: 1,
                    backgroundColor: 'grey.100',
                    borderRadius: 1,
                    fontFamily: 'monospace',
                    mb: 2 }}
                >
                  {error.message}
                </Typography>

                <Typography variant="subtitle2" color="error" gutterBottom>
                  Stack, Trace:
                </Typography>
                <Box
                  sx={ {
                    p: 1,
                    backgroundColor: 'grey.100',
                    borderRadius: 1,
                    maxHeight: 200,
                    overflow: 'auto',
                    mb: 2 }}
                >
                  <Typography
                    variant="caption"
                    component="pre"
                    sx={{ fontFamily: 'monospace' }}
                  >
                    {error.stack}
                  </Typography>
                </Box>

                { errorInfo && (
                  <>
                    <Typography variant="subtitle2" color="error" gutterBottom>
                      Component, Stack:
                    </Typography>
                    <Box
                      sx={{
                        p: 1,
                        backgroundColor: 'grey.100',
                        borderRadius: 1,
                        maxHeight: 150,
                        overflow: 'auto' }}
                    >
                      <Typography
                        variant="caption"
                        component="pre"
                        sx={{ fontFamily: 'monospace' }}
                      >
                        {errorInfo.componentStack}
                      </Typography>
                    </Box>
                  </>
                )}
              </Box>
              {copied && (
                <Alert severity="success" sx={{ mt: 1 }}>
                  Error details copied to clipboard
                </Alert>
              )}
            </AccordionDetails>
          </Accordion>
        )}
      </Paper>
    </Box>
  </>
  )};

// Minimal error fallback for critical errors
export const MinimalErrorFallback: React.FC<{
  onReset?: () => void}> = ({ onReset }) => { return (
    <>
      <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100 vh',
        padding: '20px',
        fontFamily: 'system-ui, -apple-system, sans-serif' }}
    >
      <h1 style={{ color: '#f44336', marginBottom: '16px' }}>
        Application Error
      </h1>
      <p style={{ color: '#666', marginBottom: '24px' }}>
        Something went wrong. Please refresh the page.
      </p>
      <button
        onClick={(onReset || (() => window.location.reload())}
        style={ {
          padding: '10px 20px',
          backgroundColor: '#1976 d2',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '16px' }}
      >
        Refresh Page
      </button>
    </div>
  </>
  )};`