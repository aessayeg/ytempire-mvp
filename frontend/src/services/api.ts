import axios, { AxiosInstance } from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (_error) => {
    return Promise.reject(_error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (_error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(_error)
  }
)

// Auth API
export const authApi = {
  login: async (email: string, password: string) => {
    const response = await apiClient.post('/auth/login', { email, password })
    return response.data
  },
  
  register: async (userData: {
    email: string
    username: string
    password: string
    full_name?: string
  }) => {
    const response = await apiClient.post('/auth/register', userData)
    return response.data
  },
  
  getCurrentUser: async (token?: string) => {
    const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {}
    const response = await apiClient.get('/auth/me', config)
    return response.data
  },
  
  updateProfile: async (userData: unknown) => {
    const response = await apiClient.put('/auth/profile', userData)
    return response.data
  },
}

// Channels API
export const channelsApi = {
  getAll: async () => {
    const response = await apiClient.get('/channels')
    return response.data
  },
  
  getById: async (id: string) => {
    const response = await apiClient.get(`/channels/${id}`)
    return response.data
  },
  
  create: async (channelData: unknown) => {
    const response = await apiClient.post('/channels', channelData)
    return response.data
  },
  
  update: async (id: string, channelData: unknown) => {
    const response = await apiClient.put(`/channels/${id}`, channelData)
    return response.data
  },
  
  delete: async (id: string) => {
    const response = await apiClient.delete(`/channels/${id}`)
    return response.data
  },
}

// Videos API
export const videosApi = {
  getAll: async (channelId?: string) => {
    const params = channelId ? { channel_id: channelId } : {}
    const response = await apiClient.get('/videos', { params })
    return response.data
  },
  
  getById: async (id: string) => {
    const response = await apiClient.get(`/videos/${id}`)
    return response.data
  },
  
  generate: async (videoData: unknown) => {
    const response = await apiClient.post('/videos/generate', videoData)
    return response.data
  },
  
  getQueue: async () => {
    const response = await apiClient.get('/videos/queue')
    return response.data
  },
  
  updateStatus: async (id: string, status: string) => {
    const response = await apiClient.patch(`/videos/${id}/status`, { status })
    return response.data
  },
}

// Analytics API
export const analyticsApi = {
  getDashboard: async (dateRange?: { start: string; end: string }) => {
    const params = dateRange || {}
    const response = await apiClient.get('/analytics/dashboard', { params })
    return response.data
  },
  
  getChannelAnalytics: async (channelId: string, dateRange?: unknown) => {
    const params = dateRange || {}
    const response = await apiClient.get(`/analytics/channels/${channelId}`, { params })
    return response.data
  },
  
  getVideoAnalytics: async (videoId: string) => {
    const response = await apiClient.get(`/analytics/videos/${videoId}`)
    return response.data
  },
}

// Costs API
export const costsApi = {
  getAll: async (filters?: unknown) => {
    const response = await apiClient.get('/costs', { params: filters })
    return response.data
  },
  
  getSummary: async (period: string = 'month') => {
    const response = await apiClient.get('/costs/summary', { params: { period } })
    return response.data
  },
  
  getByVideo: async (videoId: string) => {
    const response = await apiClient.get(`/costs/videos/${videoId}`)
    return response.data
  },
}

// AI Tools API
export const aiToolsApi = {
  generateScript: async (params: {
    topic: string
    style: string
    length: string
    keywords?: string[]
  }) => {
    const response = await apiClient.post('/ai/generate-script', params)
    return response.data
  },
  
  generateThumbnail: async (params: {
    title: string
    style: string
  }) => {
    const response = await apiClient.post('/ai/generate-thumbnail', params)
    return response.data
  },
  
  analyzeTrends: async (niche: string) => {
    const response = await apiClient.get('/ai/trends', { params: { niche } })
    return response.data
  },
  
  optimizeTitle: async (title: string) => {
    const response = await apiClient.post('/ai/optimize-title', { title })
    return response.data
  },
}

export default apiClient