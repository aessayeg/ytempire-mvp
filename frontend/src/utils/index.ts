/**
 * Utilities Index - Centralized utility exports
 * Owner: Frontend Team Lead
 */

// Error handling
export { 
  default as errorHandler, 
  handleError, 
  handleApiError, 
  handleValidationError, 
  handleBusinessError, 
  handleUserError, 
  handleSystemError,
  ErrorSeverity,
  ErrorCategory,
  type ErrorDetails,
  type ErrorHandlerOptions 
} from './errorHandler'

// Error boundaries
export { 
  ErrorBoundary,
  PageErrorFallback,
  ComponentErrorFallback,
  FeatureErrorFallback,
  useErrorHandler,
  withErrorBoundary,
  AsyncErrorBoundary
} from './errorBoundary'

// Validation
export {
  default as validationService,
  validators,
  schemas,
  useValidation,
  formatValidationErrors,
  getFirstError,
  type ValidationRule,
  type ValidationSchema,
  type ValidationResult,
  type FieldValidationResult
} from './validation'

// API utilities  
export { apiClient } from './api'

// Common utilities
export * from './common'
export * from './formatters'
export * from './constants'