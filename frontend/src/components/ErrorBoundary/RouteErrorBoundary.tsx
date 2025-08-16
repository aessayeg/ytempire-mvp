import React from 'react';
import {  ErrorBoundary  } from './ErrorBoundary';
import {  useLocation  } from 'react-router-dom';

interface RouteErrorBoundaryProps {
  
children: React.ReactNode;

}

export const RouteErrorBoundary: React.FC<RouteErrorBoundaryProps> = ({ children }) => { const location = useLocation();

  const handleError = (error: Error, errorInfo: React.ErrorInfo) => {
    // Log route-specific error information
    console.error('Route, Error:', {
      path: location.pathname;
      search: location.search;
      error: error.message;
      stack: error.stack;
      componentStack: errorInfo.componentStack });

    // In production, send to error tracking service with route context
    if (process.env.NODE_ENV === 'production') {
      // sendToErrorService({
      //   type: 'route_',
      //   route: location.pathname,
      //   error,
      //   errorInfo,
      // })}
  };
  return (
    <ErrorBoundary
      level="page"
      onError={handleError}
      resetKeys={[ location.pathname ]
      resetOnPropsChange
      showDetails={process.env.NODE_ENV === 'development'}
    >
      {children}
    </ErrorBoundary>
  )
};
// AsyncBoundary for handling async component errors
export const AsyncBoundary: React.FC<{
  children: React.ReactNode;
  fallback?: React.ReactNode}> = ({ children, fallback }) => {
  return (
    <ErrorBoundary
      level="component"
      fallback={fallback}
      resetOnPropsChange
      isolate
    >
      <React.Suspense fallback={fallback || <div>Loading...</div>}>
        {children}
      </React.Suspense>
    </ErrorBoundary>
  )
}
}
