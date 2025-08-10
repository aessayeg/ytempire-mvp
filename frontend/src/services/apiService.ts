/**
 * API Service - Core HTTP client and utilities
 * Owner: Frontend Team Lead
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { env } from '@/config/env'
import { useAuthStore } from '@/stores/authStore'
import { useUI } from '@/hooks/useStores'

export interface ApiResponse<T = any> {
  data: T
  message?: string
  success: boolean
  errors?: Record<string, string[]>
  meta?: {
    pagination?: {
      page: number
      limit: number
      total: number
      total_pages: number
    }
    request_id?: string
    timestamp?: string
  }
}

export interface ApiError {
  message: string
  status: number
  code?: string
  details?: Record<string, any>
  field_errors?: Record<string, string[]>
}

export interface RequestOptions {
  skipAuth?: boolean
  skipLoading?: boolean
  skipErrorHandling?: boolean
  retries?: number
  timeout?: number
  headers?: Record<string, string>
  onProgress?: (progressEvent: any) => void
}

class ApiService {
  private client: AxiosInstance
  private requestQueue: Map<string, AbortController> = new Map()
  
  constructor() {
    this.client = this.createClient()
    this.setupInterceptors()
  }

  private createClient(): AxiosInstance {
    return axios.create({
      baseURL: `${env.API_URL}/api/${env.API_VERSION}`,
      timeout: env.API_TIMEOUT || 30000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Client-Version': env.APP_VERSION || '1.0.0',
        'X-Client-Platform': 'web'
      },
    })
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token
        const token = useAuthStore.getState().accessToken
        if (token && !config.headers?.skipAuth) {
          config.headers.Authorization = `Bearer ${token}`
        }

        // Add request ID for tracking
        const requestId = this.generateRequestId()
        config.headers['X-Request-ID'] = requestId
        config.metadata = { requestId }

        // Add AbortController for cancellation
        if (config.metadata?.cancelable) {
          const controller = new AbortController()
          this.requestQueue.set(requestId, controller)
          config.signal = controller.signal
        }

        // Development logging
        if (env.IS_DEVELOPMENT && env.ENABLE_DEBUG) {
          console.log('🚀 API Request:', {
            method: config.method?.toUpperCase(),
            url: config.url,
            data: config.data,
            headers: config.headers,
            requestId
          })
        }

        return config
      },
      (error) => {
        console.error('❌ Request Error:', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        const requestId = response.config.headers?.['X-Request-ID']
        if (requestId) {
          this.requestQueue.delete(requestId)
        }

        // Development logging
        if (env.IS_DEVELOPMENT && env.ENABLE_DEBUG) {
          console.log('✅ API Response:', {
            status: response.status,
            statusText: response.statusText,
            data: response.data,
            requestId
          })
        }

        return response
      },
      async (error) => {
        const originalRequest = error.config as AxiosRequestConfig & {
          _retry?: boolean
          metadata?: any
        }

        // Clean up request from queue
        const requestId = originalRequest.headers?.['X-Request-ID']
        if (requestId) {
          this.requestQueue.delete(requestId)
        }

        // Handle 401 Unauthorized with token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true

          try {
            await useAuthStore.getState().refreshAccessToken()
            
            // Update authorization header
            const newToken = useAuthStore.getState().accessToken
            if (newToken) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`
            }
            
            return this.client(originalRequest)
          } catch (refreshError) {
            useAuthStore.getState().logout()
            this.handleAuthFailure()
            return Promise.reject(refreshError)
          }
        }

        // Skip error handling if requested
        if (originalRequest.metadata?.skipErrorHandling) {
          return Promise.reject(error)
        }

        return this.handleError(error)
      }
    )
  }

  private generateRequestId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  private handleAuthFailure(): void {
    // Redirect to login or show auth modal
    if (typeof window !== 'undefined') {
      window.location.href = '/login'
    }
  }

  private async handleError(error: any): Promise<never> {
    const apiError: ApiError = {
      message: 'An unexpected error occurred',
      status: 500
    }

    if (error.response) {
      // Server responded with error
      const { status, data } = error.response
      apiError.status = status
      apiError.message = data?.message || data?.detail || this.getDefaultErrorMessage(status)
      apiError.code = data?.code
      apiError.details = data?.details
      apiError.field_errors = data?.errors

      // Show user notifications for certain errors
      this.showErrorNotification(apiError)
    } else if (error.request) {
      // Network error
      apiError.message = 'Network error. Please check your connection.'
      apiError.status = 0
      this.showErrorNotification(apiError)
    } else if (error.code === 'ECONNABORTED') {
      // Timeout error
      apiError.message = 'Request timeout. Please try again.'
      apiError.status = 408
      this.showErrorNotification(apiError)
    } else {
      // Other error
      apiError.message = error.message || 'An unexpected error occurred'
      this.showErrorNotification(apiError)
    }

    // Development logging
    if (env.IS_DEVELOPMENT) {
      console.error('❌ API Error:', {
        status: apiError.status,
        message: apiError.message,
        code: apiError.code,
        details: apiError.details,
        originalError: error
      })
    }

    return Promise.reject(apiError)
  }

  private getDefaultErrorMessage(status: number): string {
    switch (status) {
      case 400: return 'Invalid request data'
      case 401: return 'Authentication required'
      case 403: return 'Access denied'
      case 404: return 'Resource not found'
      case 409: return 'Resource conflict'
      case 429: return 'Too many requests. Please try again later'
      case 500: return 'Internal server error'
      case 502: return 'Service temporarily unavailable'
      case 503: return 'Service unavailable'
      default: return 'An error occurred'
    }
  }

  private showErrorNotification(error: ApiError): void {
    // This will integrate with the UI notification system
    if (typeof useUI !== 'undefined') {
      const { actions } = useUI()
      actions.showError(error.message)
    } else {
      console.error('Error notification:', error.message)
    }
  }

  // Core HTTP methods
  async get<T = any>(
    url: string, 
    config?: AxiosRequestConfig,
    options?: RequestOptions
  ): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(
      url, 
      this.mergeConfig(config, options)
    )
    return response.data.data
  }

  async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
    options?: RequestOptions
  ): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(
      url,
      data,
      this.mergeConfig(config, options)
    )
    return response.data.data
  }

  async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
    options?: RequestOptions
  ): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(
      url,
      data,
      this.mergeConfig(config, options)
    )
    return response.data.data
  }

  async patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
    options?: RequestOptions
  ): Promise<T> {
    const response = await this.client.patch<ApiResponse<T>>(
      url,
      data,
      this.mergeConfig(config, options)
    )
    return response.data.data
  }

  async delete<T = any>(
    url: string,
    config?: AxiosRequestConfig,
    options?: RequestOptions
  ): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(
      url,
      this.mergeConfig(config, options)
    )
    return response.data.data
  }

  // Advanced methods
  async upload<T = any>(
    url: string,
    file: File,
    options?: RequestOptions & {
      fieldName?: string
      additionalFields?: Record<string, any>
    }
  ): Promise<T> {
    const formData = new FormData()
    formData.append(options?.fieldName || 'file', file)
    
    if (options?.additionalFields) {
      Object.entries(options.additionalFields).forEach(([key, value]) => {
        formData.append(key, typeof value === 'string' ? value : JSON.stringify(value))
      })
    }

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data',
        ...options?.headers
      },
      onUploadProgress: options?.onProgress,
      timeout: options?.timeout || 60000 // Longer timeout for uploads
    }

    return this.post<T>(url, formData, config, options)
  }

  async download(
    url: string,
    filename?: string,
    options?: RequestOptions
  ): Promise<void> {
    const response = await this.client.get(url, {
      responseType: 'blob',
      ...this.mergeConfig({}, options)
    })

    // Create download link
    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || this.extractFilenameFromResponse(response) || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }

  async paginate<T = any>(
    url: string,
    params?: Record<string, any>,
    options?: RequestOptions
  ): Promise<{
    data: T[]
    pagination: {
      page: number
      limit: number
      total: number
      total_pages: number
      has_next: boolean
      has_prev: boolean
    }
  }> {
    const response = await this.client.get<ApiResponse<T[]>>(
      url,
      this.mergeConfig({ params }, options)
    )

    return {
      data: response.data.data,
      pagination: {
        ...response.data.meta?.pagination,
        has_next: response.data.meta?.pagination?.page < response.data.meta?.pagination?.total_pages,
        has_prev: response.data.meta?.pagination?.page > 1
      }
    }
  }

  // Request management
  cancelRequest(requestId: string): void {
    const controller = this.requestQueue.get(requestId)
    if (controller) {
      controller.abort()
      this.requestQueue.delete(requestId)
    }
  }

  cancelAllRequests(): void {
    this.requestQueue.forEach((controller) => {
      controller.abort()
    })
    this.requestQueue.clear()
  }

  // Health check
  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy'
    timestamp: string
    version: string
    uptime: number
  }> {
    try {
      return await this.get('/health', {}, { skipAuth: true, skipErrorHandling: true })
    } catch {
      return {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        version: 'unknown',
        uptime: 0
      }
    }
  }

  // Utility methods
  private mergeConfig(
    config: AxiosRequestConfig = {},
    options: RequestOptions = {}
  ): AxiosRequestConfig {
    return {
      ...config,
      headers: {
        ...config.headers,
        ...options.headers,
        ...(options.skipAuth ? { skipAuth: true } : {})
      },
      timeout: options.timeout || config.timeout,
      metadata: {
        ...config.metadata,
        skipErrorHandling: options.skipErrorHandling,
        cancelable: true
      }
    }
  }

  private extractFilenameFromResponse(response: AxiosResponse): string | null {
    const contentDisposition = response.headers['content-disposition']
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="(.+)"/)
      return filenameMatch ? filenameMatch[1] : null
    }
    return null
  }

  // Retry mechanism
  async withRetry<T>(
    operation: () => Promise<T>,
    maxRetries = 3,
    delay = 1000
  ): Promise<T> {
    let lastError: any

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation()
      } catch (error) {
        lastError = error
        
        if (attempt === maxRetries) {
          break
        }

        // Don't retry certain errors
        if (error.status && [400, 401, 403, 404, 422].includes(error.status)) {
          break
        }

        // Exponential backoff
        const backoffDelay = delay * Math.pow(2, attempt - 1)
        await new Promise(resolve => setTimeout(resolve, backoffDelay))
      }
    }

    throw lastError
  }

  // Batch requests
  async batch<T extends Record<string, any>>(
    requests: T
  ): Promise<{ [K in keyof T]: Awaited<T[K]> }> {
    const results = await Promise.allSettled(
      Object.entries(requests).map(async ([key, promise]) => ({
        key,
        result: await promise
      }))
    )

    const batchResults = {} as any

    results.forEach((result) => {
      if (result.status === 'fulfilled') {
        batchResults[result.value.key] = result.value.result
      } else {
        batchResults[result.value?.key || 'unknown'] = {
          error: result.reason
        }
      }
    })

    return batchResults
  }
}

// Create singleton instance
export const apiService = new ApiService()

// Export convenience methods
export const api = {
  get: apiService.get.bind(apiService),
  post: apiService.post.bind(apiService),
  put: apiService.put.bind(apiService),
  patch: apiService.patch.bind(apiService),
  delete: apiService.delete.bind(apiService),
  upload: apiService.upload.bind(apiService),
  download: apiService.download.bind(apiService),
  paginate: apiService.paginate.bind(apiService),
  withRetry: apiService.withRetry.bind(apiService),
  batch: apiService.batch.bind(apiService),
  healthCheck: apiService.healthCheck.bind(apiService),
  cancelRequest: apiService.cancelRequest.bind(apiService),
  cancelAllRequests: apiService.cancelAllRequests.bind(apiService)
}

export default apiService