/**
 * Error Handler - Centralized error handling and reporting
 * Owner: Frontend Team Lead
 */

import React from 'react'
import { useUI } from '@/hooks/useStores'
import { env } from '@/config/env'

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum ErrorCategory {
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  NETWORK = 'network',
  VALIDATION = 'validation',
  BUSINESS_LOGIC = 'business_logic',
  SYSTEM = 'system',
  USER_INPUT = 'user_input',
  EXTERNAL_SERVICE = 'external_service'
}

export interface ErrorDetails {
  code: string
  message: string
  category: ErrorCategory
  severity: ErrorSeverity
  context?: Record<string, any>
  timestamp: string
  userAgent?: string
  url?: string
  userId?: string
  sessionId?: string
  stackTrace?: string
  originalError?: any
}

export interface ErrorHandlerOptions {
  showNotification?: boolean
  reportToService?: boolean
  logToConsole?: boolean
  redirectOnAuth?: boolean
  retryable?: boolean
  customMessage?: string
}

class ErrorHandler {
  private errorQueue: ErrorDetails[] = []
  private isOnline: boolean = navigator.onLine
  private reportingEndpoint: string = '/api/v1/errors'

  constructor() {
    // Monitor online status
    window.addEventListener('online', () => {
      this.isOnline = true
      this.flushErrorQueue()
    })
    
    window.addEventListener('offline', () => {
      this.isOnline = false
    })

    // Global error handlers
    this.setupGlobalHandlers()
  }

  /**
   * Handle application errors
   */
  handle(error: any, options: ErrorHandlerOptions = {}): ErrorDetails {
    const errorDetails = this.createErrorDetails(error, options)
    
    // Default options
    const opts = {
      showNotification: true,
      reportToService: true,
      logToConsole: env.IS_DEVELOPMENT,
      redirectOnAuth: true,
      retryable: false,
      ...options
    }

    // Log to console in development
    if (opts.logToConsole) {
      console.error('🚨 Application Error:', errorDetails)
    }

    // Show user notification
    if (opts.showNotification) {
      this.showUserNotification(errorDetails, opts.customMessage)
    }

    // Report to error service
    if (opts.reportToService) {
      this.reportError(errorDetails)
    }

    // Handle authentication errors
    if (errorDetails.category === ErrorCategory.AUTHENTICATION && opts.redirectOnAuth) {
      this.handleAuthenticationError(errorDetails)
    }

    // Handle authorization errors
    if (errorDetails.category === ErrorCategory.AUTHORIZATION) {
      this.handleAuthorizationError(errorDetails)
    }

    return errorDetails
  }

  /**
   * Handle API errors
   */
  handleApiError(error: any, options: ErrorHandlerOptions = {}): ErrorDetails {
    let category = ErrorCategory.SYSTEM
    let severity = ErrorSeverity.MEDIUM
    let code = 'API_ERROR'
    let message = 'An unexpected error occurred'

    if (error.response) {
      const status = error.response.status
      
      // Categorize by status code
      if (status === 401) {
        category = ErrorCategory.AUTHENTICATION
        severity = ErrorSeverity.HIGH
        code = 'AUTHENTICATION_FAILED'
        message = 'Authentication required'
      } else if (status === 403) {
        category = ErrorCategory.AUTHORIZATION
        severity = ErrorSeverity.HIGH
        code = 'ACCESS_DENIED'
        message = 'Access denied'
      } else if (status === 404) {
        category = ErrorCategory.BUSINESS_LOGIC
        severity = ErrorSeverity.LOW
        code = 'RESOURCE_NOT_FOUND'
        message = 'Resource not found'
      } else if (status === 422) {
        category = ErrorCategory.VALIDATION
        severity = ErrorSeverity.LOW
        code = 'VALIDATION_ERROR'
        message = 'Invalid input data'
      } else if (status === 429) {
        category = ErrorCategory.EXTERNAL_SERVICE
        severity = ErrorSeverity.MEDIUM
        code = 'RATE_LIMIT_EXCEEDED'
        message = 'Rate limit exceeded'
        options.retryable = true
      } else if (status >= 500) {
        category = ErrorCategory.SYSTEM
        severity = ErrorSeverity.HIGH
        code = 'SERVER_ERROR'
        message = 'Server error'
        options.retryable = true
      }

      // Extract error details from response
      if (error.response.data) {
        message = error.response.data.message || error.response.data.detail || message
      }
    } else if (error.request) {
      category = ErrorCategory.NETWORK
      severity = ErrorSeverity.HIGH
      code = 'NETWORK_ERROR'
      message = 'Network connection error'
      options.retryable = true
    }

    const enhancedError = {
      ...error,
      code,
      message,
      category,
      severity
    }

    return this.handle(enhancedError, options)
  }

  /**
   * Handle validation errors
   */
  handleValidationError(
    errors: Record<string, string[]> | string,
    options: ErrorHandlerOptions = {}
  ): ErrorDetails {
    let message: string
    let context: Record<string, any> = {}

    if (typeof errors === 'string') {
      message = errors
    } else {
      // Combine field errors
      const fieldErrors = Object.entries(errors)
        .map(([field, fieldErrors]) => `${field}: ${fieldErrors.join(', ')}`)
        .join('; ')
      
      message = `Validation failed: ${fieldErrors}`
      context = { fieldErrors: errors }
    }

    const error = {
      code: 'VALIDATION_ERROR',
      message,
      category: ErrorCategory.VALIDATION,
      severity: ErrorSeverity.LOW,
      context
    }

    return this.handle(error, {
      ...options,
      showNotification: true,
      reportToService: false // Validation errors are usually user input issues
    })
  }

  /**
   * Handle business logic errors
   */
  handleBusinessError(
    code: string,
    message: string,
    context?: Record<string, any>,
    options: ErrorHandlerOptions = {}
  ): ErrorDetails {
    const error = {
      code,
      message,
      category: ErrorCategory.BUSINESS_LOGIC,
      severity: ErrorSeverity.MEDIUM,
      context
    }

    return this.handle(error, options)
  }

  /**
   * Handle user input errors
   */
  handleUserError(message: string, options: ErrorHandlerOptions = {}): ErrorDetails {
    const error = {
      code: 'USER_INPUT_ERROR',
      message,
      category: ErrorCategory.USER_INPUT,
      severity: ErrorSeverity.LOW
    }

    return this.handle(error, {
      ...options,
      reportToService: false,
      showNotification: true
    })
  }

  /**
   * Handle system errors
   */
  handleSystemError(
    error: any,
    context?: Record<string, any>,
    options: ErrorHandlerOptions = {}
  ): ErrorDetails {
    const enhancedError = {
      ...error,
      category: ErrorCategory.SYSTEM,
      severity: ErrorSeverity.HIGH,
      context
    }

    return this.handle(enhancedError, {
      ...options,
      reportToService: true,
      showNotification: true
    })
  }

  /**
   * Create standardized error details
   */
  private createErrorDetails(error: any, options: ErrorHandlerOptions): ErrorDetails {
    const now = new Date().toISOString()
    
    return {
      code: error.code || 'UNKNOWN_ERROR',
      message: error.message || 'An unexpected error occurred',
      category: error.category || ErrorCategory.SYSTEM,
      severity: error.severity || ErrorSeverity.MEDIUM,
      context: error.context || {},
      timestamp: now,
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: this.getCurrentUserId(),
      sessionId: this.getSessionId(),
      stackTrace: error.stack,
      originalError: error
    }
  }

  /**
   * Show user notification
   */
  private showUserNotification(errorDetails: ErrorDetails, customMessage?: string): void {
    try {
      const { actions } = useUI()
      const message = customMessage || this.getUserFriendlyMessage(errorDetails)
      
      switch (errorDetails.severity) {
        case ErrorSeverity.LOW:
          actions.showWarning(message)
          break
        case ErrorSeverity.MEDIUM:
        case ErrorSeverity.HIGH:
        case ErrorSeverity.CRITICAL:
          actions.showError(message)
          break
      }
    } catch (error) {
      console.error('Failed to show error notification:', error)
    }
  }

  /**
   * Get user-friendly error message
   */
  private getUserFriendlyMessage(errorDetails: ErrorDetails): string {
    switch (errorDetails.category) {
      case ErrorCategory.AUTHENTICATION:
        return 'Please log in to continue'
      case ErrorCategory.AUTHORIZATION:
        return 'You do not have permission to perform this action'
      case ErrorCategory.NETWORK:
        return 'Connection error. Please check your internet connection'
      case ErrorCategory.VALIDATION:
        return errorDetails.message
      case ErrorCategory.BUSINESS_LOGIC:
        return errorDetails.message
      case ErrorCategory.USER_INPUT:
        return errorDetails.message
      case ErrorCategory.EXTERNAL_SERVICE:
        return 'Service temporarily unavailable. Please try again'
      default:
        return 'Something went wrong. Please try again'
    }
  }

  /**
   * Report error to monitoring service
   */
  private async reportError(errorDetails: ErrorDetails): Promise<void> {
    if (!this.isOnline) {
      this.errorQueue.push(errorDetails)
      return
    }

    try {
      await fetch(this.reportingEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorDetails)
      })
    } catch (error) {
      // Queue error for later reporting
      this.errorQueue.push(errorDetails)
    }
  }

  /**
   * Flush queued errors when back online
   */
  private async flushErrorQueue(): Promise<void> {
    while (this.errorQueue.length > 0 && this.isOnline) {
      const errorDetails = this.errorQueue.shift()!
      await this.reportError(errorDetails)
    }
  }

  /**
   * Handle authentication errors
   */
  private handleAuthenticationError(errorDetails: ErrorDetails): void {
    // Clear auth state and redirect to login
    import('@/stores/authStore').then(({ useAuthStore }) => {
      useAuthStore.getState().logout()
    })

    // Redirect to login after a short delay
    setTimeout(() => {
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }, 1000)
  }

  /**
   * Handle authorization errors
   */
  private handleAuthorizationError(errorDetails: ErrorDetails): void {
    // Show access denied message
    // Optionally redirect to dashboard or previous page
    console.warn('Authorization error:', errorDetails)
  }

  /**
   * Setup global error handlers
   */
  private setupGlobalHandlers(): void {
    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.handleSystemError(event.reason, {
        type: 'unhandled_promise_rejection',
        promise: event.promise
      })
    })

    // JavaScript errors
    window.addEventListener('error', (event) => {
      this.handleSystemError(event.error || event, {
        type: 'javascript_error',
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      })
    })

    // Resource loading errors
    window.addEventListener('error', (event) => {
      if (event.target !== window) {
        this.handleSystemError(new Error(`Resource failed to load: ${(event.target as any).src || (event.target as any).href}`), {
          type: 'resource_error',
          element: event.target
        })
      }
    }, true)
  }

  /**
   * Get current user ID
   */
  private getCurrentUserId(): string | undefined {
    try {
      const authStore = import('@/stores/authStore')
      return authStore.then(store => store.useAuthStore.getState().user?.id)
    } catch {
      return undefined
    }
  }

  /**
   * Get session ID
   */
  private getSessionId(): string {
    let sessionId = sessionStorage.getItem('session_id')
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      sessionStorage.setItem('session_id', sessionId)
    }
    return sessionId
  }

  /**
   * Get error statistics
   */
  getStats(): {
    queuedErrors: number
    isOnline: boolean
    totalHandled: number
  } {
    return {
      queuedErrors: this.errorQueue.length,
      isOnline: this.isOnline,
      totalHandled: 0 // Would track this with a counter in production
    }
  }

  /**
   * Clear error queue
   */
  clearQueue(): void {
    this.errorQueue = []
  }
}

// Create singleton instance
export const errorHandler = new ErrorHandler()

// Convenience functions
export const handleError = (error: any, options?: ErrorHandlerOptions) => 
  errorHandler.handle(error, options)

export const handleApiError = (error: any, options?: ErrorHandlerOptions) => 
  errorHandler.handleApiError(error, options)

export const handleValidationError = (errors: Record<string, string[]> | string, options?: ErrorHandlerOptions) => 
  errorHandler.handleValidationError(errors, options)

export const handleBusinessError = (code: string, message: string, context?: Record<string, any>, options?: ErrorHandlerOptions) => 
  errorHandler.handleBusinessError(code, message, context, options)

export const handleUserError = (message: string, options?: ErrorHandlerOptions) => 
  errorHandler.handleUserError(message, options)

export const handleSystemError = (error: any, context?: Record<string, any>, options?: ErrorHandlerOptions) => 
  errorHandler.handleSystemError(error, context, options)

// React error boundary helper
export class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ComponentType<{ error: Error; errorDetails: ErrorDetails }> },
  { hasError: boolean; error?: Error; errorDetails?: ErrorDetails }
> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const errorDetails = errorHandler.handleSystemError(error, {
      componentStack: errorInfo.componentStack,
      errorBoundary: true
    }, {
      showNotification: false // Don't show notification in error boundary
    })

    this.setState({ errorDetails })
  }

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback
      return <FallbackComponent error={this.state.error!} errorDetails={this.state.errorDetails!} />
    }

    return this.props.children
  }
}

// Default error fallback component
const DefaultErrorFallback: React.FC<{ error: Error; errorDetails: ErrorDetails }> = ({ error, errorDetails }) => (
  <div style={{ padding: '20px', textAlign: 'center' }}>
    <h2>Something went wrong</h2>
    <details style={{ whiteSpace: 'pre-wrap', textAlign: 'left', maxWidth: '500px', margin: '0 auto' }}>
      <summary>Error details</summary>
      <p><strong>Code:</strong> {errorDetails.code}</p>
      <p><strong>Message:</strong> {errorDetails.message}</p>
      <p><strong>Timestamp:</strong> {errorDetails.timestamp}</p>
      {env.IS_DEVELOPMENT && (
        <pre>{error.stack}</pre>
      )}
    </details>
    <button onClick={() => window.location.reload()} style={{ marginTop: '20px' }}>
      Reload Page
    </button>
  </div>
)

export default errorHandler