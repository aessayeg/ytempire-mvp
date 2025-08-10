/**
 * API Utility Functions
 * Owner: Frontend Team Lead
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios'
import { env } from '@/config/env'
import { useAuthStore } from '@/stores/authStore'

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: `${env.API_URL}/api/${env.API_VERSION}`,
  timeout: env.API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().accessToken
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // Add request ID for tracking
    config.headers['X-Request-ID'] = generateRequestId()
    
    // Log request in development
    if (env.IS_DEVELOPMENT && env.ENABLE_DEBUG) {
      console.log('API Request:', {
        method: config.method,
        url: config.url,
        data: config.data,
      })
    }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Log response in development
    if (env.IS_DEVELOPMENT && env.ENABLE_DEBUG) {
      console.log('API Response:', {
        status: response.status,
        data: response.data,
      })
    }
    
    return response
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean }
    
    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      
      try {
        await useAuthStore.getState().refreshAccessToken()
        return api(originalRequest)
      } catch (refreshError) {
        useAuthStore.getState().logout()
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }
    
    // Handle other errors
    if (error.response) {
      // Server responded with error
      const errorMessage = (error.response.data as any)?.detail || 
                          (error.response.data as any)?.message || 
                          'An error occurred'
      
      console.error('API Error:', {
        status: error.response.status,
        message: errorMessage,
        url: originalRequest.url,
      })
      
      // Show user-friendly error messages
      handleApiError(error.response.status, errorMessage)
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message)
      handleNetworkError()
    } else {
      // Something else happened
      console.error('Error:', error.message)
    }
    
    return Promise.reject(error)
  }
)

// Helper functions
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

function handleApiError(status: number, message: string): void {
  switch (status) {
    case 400:
      showNotification('error', 'Invalid request: ' + message)
      break
    case 403:
      showNotification('error', 'You do not have permission to perform this action')
      break
    case 404:
      showNotification('error', 'Resource not found')
      break
    case 429:
      showNotification('warning', 'Too many requests. Please try again later')
      break
    case 500:
      showNotification('error', 'Server error. Please try again later')
      break
    default:
      showNotification('error', message)
  }
}

function handleNetworkError(): void {
  showNotification('error', 'Network error. Please check your connection')
}

function showNotification(type: 'error' | 'warning' | 'info' | 'success', message: string): void {
  // Integrate with UI store notifications
  try {
    import('@/hooks/useStores').then(({ useUI }) => {
      const { actions } = useUI()
      switch (type) {
        case 'error':
          actions.showError(message)
          break
        case 'warning':
          actions.showWarning(message)
          break
        case 'success':
          actions.showSuccess(message)
          break
        default:
          actions.showInfo(message)
      }
    })
  } catch {
    console.log(`[${type.toUpperCase()}]: ${message}`)
  }
}

// API methods
export const apiClient = {
  // Auth
  login: (email: string, password: string) => 
    api.post('/auth/login', { username: email, password }),
  
  register: (data: any) => 
    api.post('/auth/register', data),
  
  logout: () => 
    api.post('/auth/logout'),
  
  refreshToken: (refreshToken: string) => 
    api.post('/auth/refresh', { refresh_token: refreshToken }),
  
  // Users
  getCurrentUser: () => 
    api.get('/users/me'),
  
  updateCurrentUser: (data: any) => 
    api.patch('/users/me', data),
  
  getUserUsage: () => 
    api.get('/users/me/usage'),
  
  // Channels
  getChannels: () => 
    api.get('/channels/'),
  
  createChannel: (data: any) => 
    api.post('/channels/', data),
  
  getChannel: (id: number) => 
    api.get(`/channels/${id}`),
  
  updateChannel: (id: number, data: any) => 
    api.patch(`/channels/${id}`, data),
  
  deleteChannel: (id: number) => 
    api.delete(`/channels/${id}`),
  
  connectYouTube: (channelId: number, authCode: string) => 
    api.post(`/channels/${channelId}/connect-youtube`, { auth_code: authCode }),
  
  getChannelStats: (id: number) => 
    api.get(`/channels/${id}/stats`),
  
  // Videos
  generateVideo: (data: any) => 
    api.post('/videos/generate', data),
  
  getVideos: (params?: any) => 
    api.get('/videos/', { params }),
  
  getVideo: (id: number) => 
    api.get(`/videos/${id}`),
  
  publishVideo: (id: number, scheduleTime?: string) => 
    api.post(`/videos/${id}/publish`, { schedule_time: scheduleTime }),
  
  retryVideo: (id: number) => 
    api.post(`/videos/${id}/retry`),
  
  deleteVideo: (id: number) => 
    api.delete(`/videos/${id}`),
  
  getVideoCost: (id: number) => 
    api.get(`/videos/${id}/cost-breakdown`),
  
  // Analytics
  getDashboard: () => 
    api.get('/analytics/dashboard'),
  
  getChannelAnalytics: (id: number, params?: any) => 
    api.get(`/analytics/channels/${id}/analytics`, { params }),
  
  getVideoAnalytics: (id: number) => 
    api.get(`/analytics/videos/${id}/analytics`),
  
  getCostAnalytics: (params?: any) => 
    api.get('/analytics/costs', { params }),
  
  getWeeklyReport: () => 
    api.get('/analytics/reports/weekly'),
}

export default api