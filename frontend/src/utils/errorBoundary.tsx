/**
 * Error Boundary Components - React error boundaries
 * Owner: Frontend Team Lead
 */

import React, { Component, ReactNode } from 'react'
import { Box, Typography, Button, Card, CardContent, Alert, Accordion, AccordionSummary, AccordionDetails } from '@mui/material'
import { ErrorOutline, ExpandMore, Refresh, Home } from '@mui/icons-material'
import { errorHandler, ErrorDetails } from './errorHandler'
import { env } from '@/config/env'

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
  errorDetails?: ErrorDetails
  eventId?: string
}

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: React.ComponentType<ErrorFallbackProps>
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
  level?: 'page' | 'component' | 'feature'
}

interface ErrorFallbackProps {
  error: Error
  errorDetails: ErrorDetails
  retry?: () => void
  eventId?: string
}

/**
 * Main Error Boundary Component
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Handle the error with our error handler
    const errorDetails = errorHandler.handle(error, {
      showNotification: false, // Don't show notification in error boundary
      reportToService: true,
      logToConsole: true,
      customMessage: undefined
    })

    // Generate event ID for user reference
    const eventId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    this.setState({ 
      errorDetails,
      eventId
    })

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo)

    // Log error details
    console.error('React Error Boundary caught an error:', {
      error,
      errorInfo,
      errorDetails,
      eventId
    })
  }

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || this.getDefaultFallback()
      
      return (
        <FallbackComponent
          error={this.state.error!}
          errorDetails={this.state.errorDetails!}
          eventId={this.state.eventId}
          retry={() => this.setState({ hasError: false, error: undefined, errorDetails: undefined })}
        />
      )
    }

    return this.props.children
  }

  private getDefaultFallback(): React.ComponentType<ErrorFallbackProps> {
    switch (this.props.level) {
      case 'page':
        return PageErrorFallback
      case 'component':
        return ComponentErrorFallback
      case 'feature':
        return FeatureErrorFallback
      default:
        return PageErrorFallback
    }
  }
}

/**
 * Page-level Error Fallback
 */
export const PageErrorFallback: React.FC<ErrorFallbackProps> = ({ error, errorDetails, eventId, retry }) => (
  <Box
    display="flex"
    flexDirection="column"
    alignItems="center"
    justifyContent="center"
    minHeight="100vh"
    padding={3}
    bgcolor="background.default"
  >
    <Card sx={{ maxWidth: 600, width: '100%' }}>
      <CardContent sx={{ textAlign: 'center', p: 4 }}>
        <ErrorOutline sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
        
        <Typography variant="h4" gutterBottom>
          Oops! Something went wrong
        </Typography>
        
        <Typography variant="body1" color="textSecondary" paragraph>
          We're sorry, but something unexpected happened. Our team has been notified and is working on a fix.
        </Typography>

        {eventId && (
          <Typography variant="body2" color="textSecondary" paragraph>
            Error ID: <code>{eventId}</code>
          </Typography>
        )}

        <Box display="flex" gap={2} justifyContent="center" mt={3}>
          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={() => window.location.reload()}
          >
            Reload Page
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<Home />}
            onClick={() => window.location.href = '/'}
          >
            Go Home
          </Button>
          
          {retry && (
            <Button
              variant="text"
              onClick={retry}
            >
              Try Again
            </Button>
          )}
        </Box>

        {/* Development Error Details */}
        {env.IS_DEVELOPMENT && (
          <Accordion sx={{ mt: 3, textAlign: 'left' }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">Developer Details</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" component="div" sx={{ mb: 2 }}>
                <strong>Error:</strong> {error.message}
              </Typography>
              <Typography variant="body2" component="div" sx={{ mb: 2 }}>
                <strong>Code:</strong> {errorDetails.code}
              </Typography>
              <Typography variant="body2" component="div" sx={{ mb: 2 }}>
                <strong>Category:</strong> {errorDetails.category}
              </Typography>
              <Typography variant="body2" component="div" sx={{ mb: 2 }}>
                <strong>Severity:</strong> {errorDetails.severity}
              </Typography>
              {error.stack && (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Stack Trace:</strong>
                  </Typography>
                  <Box
                    component="pre"
                    sx={{
                      fontSize: '0.75rem',
                      overflow: 'auto',
                      backgroundColor: 'grey.100',
                      p: 1,
                      borderRadius: 1,
                      maxHeight: 300
                    }}
                  >
                    {error.stack}
                  </Box>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        )}
      </CardContent>
    </Card>
  </Box>
)

/**
 * Component-level Error Fallback
 */
export const ComponentErrorFallback: React.FC<ErrorFallbackProps> = ({ error, errorDetails, eventId, retry }) => (
  <Alert 
    severity="error" 
    action={
      retry && (
        <Button color="inherit" size="small" onClick={retry}>
          Retry
        </Button>
      )
    }
    sx={{ m: 2 }}
  >
    <Typography variant="subtitle2" gutterBottom>
      Component Error
    </Typography>
    <Typography variant="body2">
      This component encountered an error and couldn't render properly.
    </Typography>
    {eventId && (
      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
        Error ID: {eventId}
      </Typography>
    )}
  </Alert>
)

/**
 * Feature-level Error Fallback
 */
export const FeatureErrorFallback: React.FC<ErrorFallbackProps> = ({ error, errorDetails, eventId, retry }) => (
  <Card sx={{ m: 2 }}>
    <CardContent>
      <Box display="flex" alignItems="center" gap={2} mb={2}>
        <ErrorOutline color="error" />
        <Typography variant="h6">
          Feature Unavailable
        </Typography>
      </Box>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        This feature is temporarily unavailable due to a technical issue. 
        Please try again or contact support if the problem persists.
      </Typography>

      {eventId && (
        <Typography variant="caption" color="textSecondary" paragraph>
          Reference: {eventId}
        </Typography>
      )}

      {retry && (
        <Button variant="outlined" size="small" onClick={retry}>
          Try Again
        </Button>
      )}
    </CardContent>
  </Card>
)

/**
 * Hook for handling errors in functional components
 */
export const useErrorHandler = () => {
  const handleError = React.useCallback((error: any, context?: Record<string, any>) => {
    errorHandler.handle(error, {
      showNotification: true,
      reportToService: true,
      logToConsole: env.IS_DEVELOPMENT
    })
  }, [])

  const handleAsyncError = React.useCallback((asyncFn: () => Promise<any>, context?: Record<string, any>) => {
    return asyncFn().catch(error => {
      handleError(error, context)
      throw error // Re-throw to allow component to handle if needed
    })
  }, [handleError])

  return {
    handleError,
    handleAsyncError
  }
}

/**
 * HOC for wrapping components with error boundary
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  )

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`
  
  return WrappedComponent
}

/**
 * Async Error Boundary for handling async errors
 */
interface AsyncErrorBoundaryState {
  error?: Error
  hasError: boolean
}

export class AsyncErrorBoundary extends Component<
  { 
    children: ReactNode
    fallback?: React.ComponentType<{ error: Error; retry: () => void }>
  },
  AsyncErrorBoundaryState
> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    errorHandler.handle(error, {
      showNotification: true,
      reportToService: true
    })
  }

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || ComponentErrorFallback
      return (
        <FallbackComponent 
          error={this.state.error!}
          errorDetails={{} as ErrorDetails}
          retry={() => this.setState({ hasError: false, error: undefined })}
        />
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary