/**
 * Validation Utilities - Form validation and data validation helpers
 * Owner: Frontend Team Lead
 */

import React from 'react'
import { handleValidationError } from './errorHandler'

export interface ValidationRule<T = any> {
  validator: (value: T, context?: any) => boolean | string
  message?: string
  trigger?: 'change' | 'blur' | 'submit'
}

export interface ValidationSchema {
  [field: string]: ValidationRule[]
}

export interface ValidationResult {
  isValid: boolean
  errors: Record<string, string[]>
  firstError?: string
}

export interface FieldValidationResult {
  isValid: boolean
  errors: string[]
  firstError?: string
}

class ValidationService {
  /**
   * Validate a single field
   */
  validateField<T>(
    value: T, 
    rules: ValidationRule[], 
    context?: any
  ): FieldValidationResult {
    const errors: string[] = []

    for (const rule of rules) {
      const result = rule.validator(value, context)
      
      if (result !== true) {
        const errorMessage = typeof result === 'string' ? result : (rule.message || 'Validation failed')
        errors.push(errorMessage)
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      firstError: errors[0]
    }
  }

  /**
   * Validate an object against a schema
   */
  validate<T extends Record<string, any>>(
    data: T,
    schema: ValidationSchema,
    context?: any
  ): ValidationResult {
    const errors: Record<string, string[]> = {}
    let firstError: string | undefined

    for (const [field, rules] of Object.entries(schema)) {
      const fieldResult = this.validateField(data[field], rules, { ...context, data })
      
      if (!fieldResult.isValid) {
        errors[field] = fieldResult.errors
        
        if (!firstError && fieldResult.firstError) {
          firstError = fieldResult.firstError
        }
      }
    }

    return {
      isValid: Object.keys(errors).length === 0,
      errors,
      firstError
    }
  }

  /**
   * Async validation
   */
  async validateAsync<T extends Record<string, any>>(
    data: T,
    schema: ValidationSchema,
    context?: any
  ): Promise<ValidationResult> {
    // For now, just call synchronous validation
    // In the future, this could support async validators
    return this.validate(data, schema, context)
  }
}

// Create singleton instance
export const validationService = new ValidationService()

// Common validation rules
export const validators = {
  required: (message = 'This field is required'): ValidationRule => ({
    validator: (value) => {
      if (value === null || value === undefined || value === '') {
        return message
      }
      if (Array.isArray(value) && value.length === 0) {
        return message
      }
      return true
    },
    message
  }),

  email: (message = 'Please enter a valid email address'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true // Allow empty for optional fields
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      return emailRegex.test(value) || message
    },
    message
  }),

  minLength: (min: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      const msg = message || `Must be at least ${min} characters long`
      return value.length >= min || msg
    },
    message: message || `Must be at least ${min} characters long`
  }),

  maxLength: (max: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      const msg = message || `Must be at most ${max} characters long`
      return value.length <= max || msg
    },
    message: message || `Must be at most ${max} characters long`
  }),

  password: (message = 'Password must contain at least 8 characters, including uppercase, lowercase, number, and special character'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/
      return passwordRegex.test(value) || message
    },
    message
  }),

  confirmPassword: (passwordField = 'password', message = 'Passwords do not match'): ValidationRule => ({
    validator: (value, context) => {
      if (!value) return true
      const password = context?.data?.[passwordField]
      return value === password || message
    },
    message
  }),

  url: (message = 'Please enter a valid URL'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      try {
        new URL(value)
        return true
      } catch {
        return message
      }
    },
    message
  }),

  number: (message = 'Please enter a valid number'): ValidationRule => ({
    validator: (value) => {
      if (value === '' || value === null || value === undefined) return true
      return !isNaN(Number(value)) || message
    },
    message
  }),

  min: (minValue: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (value === '' || value === null || value === undefined) return true
      const num = Number(value)
      const msg = message || `Must be at least ${minValue}`
      return num >= minValue || msg
    },
    message: message || `Must be at least ${minValue}`
  }),

  max: (maxValue: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (value === '' || value === null || value === undefined) return true
      const num = Number(value)
      const msg = message || `Must be at most ${maxValue}`
      return num <= maxValue || msg
    },
    message: message || `Must be at most ${maxValue}`
  }),

  pattern: (regex: RegExp, message = 'Invalid format'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      return regex.test(value) || message
    },
    message
  }),

  phone: (message = 'Please enter a valid phone number'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/
      return phoneRegex.test(value.replace(/\s/g, '')) || message
    },
    message
  }),

  custom: (fn: (value: any, context?: any) => boolean | string, message = 'Validation failed'): ValidationRule => ({
    validator: fn,
    message
  }),

  // Array validators
  minItems: (min: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (!Array.isArray(value)) return true
      const msg = message || `Must have at least ${min} items`
      return value.length >= min || msg
    },
    message: message || `Must have at least ${min} items`
  }),

  maxItems: (max: number, message?: string): ValidationRule => ({
    validator: (value) => {
      if (!Array.isArray(value)) return true
      const msg = message || `Must have at most ${max} items`
      return value.length <= max || msg
    },
    message: message || `Must have at most ${max} items`
  }),

  // File validators
  fileSize: (maxSizeBytes: number, message?: string): ValidationRule => ({
    validator: (file: File) => {
      if (!file) return true
      const maxSizeMB = Math.round(maxSizeBytes / (1024 * 1024))
      const msg = message || `File size must be less than ${maxSizeMB}MB`
      return file.size <= maxSizeBytes || msg
    },
    message: message || 'File is too large'
  }),

  fileType: (allowedTypes: string[], message?: string): ValidationRule => ({
    validator: (file: File) => {
      if (!file) return true
      const msg = message || `File type must be one of: ${allowedTypes.join(', ')}`
      return allowedTypes.includes(file.type) || msg
    },
    message: message || 'Invalid file type'
  }),

  // YouTube-specific validators
  youtubeUrl: (message = 'Please enter a valid YouTube URL'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/
      return youtubeRegex.test(value) || message
    },
    message
  }),

  channelName: (message = 'Channel name must be 3-50 characters and contain only letters, numbers, spaces, and hyphens'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      if (value.length < 3 || value.length > 50) return message
      const nameRegex = /^[a-zA-Z0-9\s\-]+$/
      return nameRegex.test(value) || message
    },
    message
  }),

  videoTitle: (message = 'Video title must be 10-100 characters long'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      if (value.length < 10 || value.length > 100) return message
      return true
    },
    message
  }),

  videoDescription: (message = 'Video description must be at least 50 characters long'): ValidationRule => ({
    validator: (value) => {
      if (!value) return true
      return value.length >= 50 || message
    },
    message
  }),

  tags: (message = 'Please provide 3-10 tags, each 2-30 characters long'): ValidationRule => ({
    validator: (value) => {
      if (!Array.isArray(value)) return true
      if (value.length < 3 || value.length > 10) {
        return 'Must have 3-10 tags'
      }
      for (const tag of value) {
        if (typeof tag !== 'string' || tag.length < 2 || tag.length > 30) {
          return 'Each tag must be 2-30 characters long'
        }
      }
      return true
    },
    message
  })
}

// Pre-defined schemas for common forms
export const schemas = {
  login: {
    email: [validators.required(), validators.email()],
    password: [validators.required()]
  },

  register: {
    fullName: [validators.required(), validators.minLength(2), validators.maxLength(50)],
    email: [validators.required(), validators.email()],
    password: [validators.required(), validators.password()],
    confirmPassword: [validators.required(), validators.confirmPassword()]
  },

  channel: {
    name: [validators.required(), validators.channelName()],
    description: [validators.required(), validators.minLength(20), validators.maxLength(500)],
    category: [validators.required()],
    target_audience: [validators.required()],
    tone: [validators.required()],
    language: [validators.required()]
  },

  video: {
    topic: [validators.required(), validators.minLength(10), validators.maxLength(200)],
    channel_id: [validators.required()],
    target_duration: [validators.number(), validators.min(30), validators.max(3600)],
    custom_instructions: [validators.maxLength(1000)]
  },

  profile: {
    fullName: [validators.required(), validators.minLength(2), validators.maxLength(50)],
    email: [validators.required(), validators.email()],
    phone: [validators.phone()]
  },

  changePassword: {
    currentPassword: [validators.required()],
    newPassword: [validators.required(), validators.password()],
    confirmNewPassword: [validators.required(), validators.confirmPassword('newPassword')]
  }
}

// React hook for form validation
export const useValidation = <T extends Record<string, any>>(
  schema: ValidationSchema,
  options: {
    validateOnChange?: boolean
    validateOnBlur?: boolean
    showErrorsOnChange?: boolean
  } = {}
) => {
  const [errors, setErrors] = React.useState<Record<string, string[]>>({})
  const [touched, setTouched] = React.useState<Record<string, boolean>>({})

  const validateField = React.useCallback((field: string, value: any, data: T) => {
    const rules = schema[field]
    if (!rules) return { isValid: true, errors: [] }

    const result = validationService.validateField(value, rules, { data })
    
    setErrors(prev => ({
      ...prev,
      [field]: result.errors
    }))

    return result
  }, [schema])

  const validateAll = React.useCallback((data: T) => {
    const result = validationService.validate(data, schema)
    
    setErrors(result.errors)
    
    if (!result.isValid) {
      handleValidationError(result.errors)
    }
    
    return result
  }, [schema])

  const clearFieldError = React.useCallback((field: string) => {
    setErrors(prev => {
      const newErrors = { ...prev }
      delete newErrors[field]
      return newErrors
    })
  }, [])

  const clearAllErrors = React.useCallback(() => {
    setErrors({})
    setTouched({})
  }, [])

  const setFieldTouched = React.useCallback((field: string, isTouched = true) => {
    setTouched(prev => ({
      ...prev,
      [field]: isTouched
    }))
  }, [])

  const getFieldProps = React.useCallback((field: string) => ({
    error: touched[field] && errors[field]?.length > 0,
    helperText: touched[field] ? errors[field]?.[0] : undefined,
    onBlur: () => {
      setFieldTouched(field, true)
    }
  }), [errors, touched, setFieldTouched])

  const isValid = Object.keys(errors).length === 0
  const hasErrors = Object.keys(errors).some(field => errors[field].length > 0)

  return {
    errors,
    touched,
    isValid,
    hasErrors,
    validateField,
    validateAll,
    clearFieldError,
    clearAllErrors,
    setFieldTouched,
    getFieldProps
  }
}

// Utility functions
export const formatValidationErrors = (errors: Record<string, string[]>): string => {
  return Object.entries(errors)
    .map(([field, fieldErrors]) => `${field}: ${fieldErrors.join(', ')}`)
    .join('; ')
}

export const getFirstError = (errors: Record<string, string[]>): string | undefined => {
  for (const fieldErrors of Object.values(errors)) {
    if (fieldErrors.length > 0) {
      return fieldErrors[0]
    }
  }
  return undefined
}

export default validationService