import React, { Component, ErrorInfo, ReactNode } from 'react';
import { ErrorFallback } from './ErrorFallback';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKeys?: Array<string | number>;
  resetOnPropsChange?: boolean;
  isolate?: boolean;
  level?: 'page' | 'section' | 'component';
  showDetails?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  private resetTimeoutId: NodeJS.Timeout | null = null;
  private previousResetKeys: Array<string | number> = [];

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const { onError } = this.props;
    
    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // Call custom error handler if provided
    if (onError) {
      onError(error, errorInfo);
    }

    // Log to error reporting service
    this.logErrorToService(error, errorInfo);

    // Update state with error details
    this.setState((prevState) => ({
      errorInfo,
      errorCount: prevState.errorCount + 1,
    }));

    // Auto-reset after 10 seconds for component-level errors
    if (this.props.level === 'component' && this.state.errorCount < 3) {
      this.resetTimeoutId = setTimeout(() => {
        this.resetErrorBoundary();
      }, 10000);
    }
  }

  componentDidUpdate(prevProps: Props) {
    const { resetKeys, resetOnPropsChange } = this.props;
    const { hasError } = this.state;

    // Reset on prop changes if enabled
    if (hasError && resetOnPropsChange && prevProps.children !== this.props.children) {
      this.resetErrorBoundary();
    }

    // Reset when resetKeys change
    if (hasError && resetKeys) {
      const hasResetKeyChanged = resetKeys.some(
        (key, index) => key !== this.previousResetKeys[index]
      );
      if (hasResetKeyChanged) {
        this.resetErrorBoundary();
        this.previousResetKeys = resetKeys;
      }
    }
  }

  componentWillUnmount() {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }
  }

  logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // In production, this would send to an error tracking service
    const errorData = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      level: this.props.level || 'component',
    };

    // Store in localStorage for debugging
    const errors = JSON.parse(localStorage.getItem('app_errors') || '[]');
    errors.push(errorData);
    // Keep only last 10 errors
    if (errors.length > 10) {
      errors.shift();
    }
    localStorage.setItem('app_errors', JSON.stringify(errors));

    // In production, send to error tracking service
    // Example: Sentry, LogRocket, etc.
    if (process.env.NODE_ENV === 'production') {
      // sendToErrorTrackingService(errorData);
    }
  };

  resetErrorBoundary = () => {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
      this.resetTimeoutId = null;
    }

    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    });
  };

  render() {
    const { hasError, error, errorInfo, errorCount } = this.state;
    const { children, fallback, isolate, level = 'component', showDetails } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (fallback) {
        return <>{fallback}</>;
      }

      // Use default error fallback
      return (
        <ErrorFallback
          error={error}
          errorInfo={errorInfo}
          onReset={this.resetErrorBoundary}
          level={level}
          showDetails={showDetails}
          errorCount={errorCount}
          isolate={isolate}
        />
      );
    }

    return children;
  }
}

// HOC for wrapping components with error boundary
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;

  return WrappedComponent;
}