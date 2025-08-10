/**
 * Authentication Store
 * Owner: Frontend Team Lead
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import axios from 'axios'

interface User {
  id: number
  email: string
  username: string
  fullName?: string
  subscriptionTier: string
  channelsLimit: number
  dailyVideoLimit: number
  totalSpent: number
  monthlyBudget: number
  isBetaUser: boolean
}

interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  // Actions
  login: (email: string, password: string) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => void
  refreshAccessToken: () => Promise<void>
  setUser: (user: User) => void
  clearError: () => void
}

interface RegisterData {
  email: string
  username: string
  password: string
  fullName?: string
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      
      login: async (email, password) => {
        set({ isLoading: true, error: null })
        try {
          const response = await axios.post('/api/v1/auth/login', {
            username: email,
            password,
          })
          
          const { access_token, refresh_token } = response.data
          
          // Set tokens
          set({
            accessToken: access_token,
            refreshToken: refresh_token,
            isAuthenticated: true,
          })
          
          // Fetch user data
          const userResponse = await axios.get('/api/v1/users/me', {
            headers: { Authorization: `Bearer ${access_token}` },
          })
          
          set({
            user: userResponse.data,
            isLoading: false,
          })
        } catch (error: any) {
          set({
            error: error.response?.data?.detail || 'Login failed',
            isLoading: false,
          })
        }
      },
      
      register: async (data) => {
        set({ isLoading: true, error: null })
        try {
          const response = await axios.post('/api/v1/auth/register', data)
          set({
            user: response.data,
            isLoading: false,
          })
          // Auto-login after registration
          await get().login(data.email, data.password)
        } catch (error: any) {
          set({
            error: error.response?.data?.detail || 'Registration failed',
            isLoading: false,
          })
        }
      },
      
      logout: () => {
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        })
        // Clear axios default header
        delete axios.defaults.headers.common['Authorization']
      },
      
      refreshAccessToken: async () => {
        const refreshToken = get().refreshToken
        if (!refreshToken) {
          get().logout()
          return
        }
        
        try {
          const response = await axios.post('/api/v1/auth/refresh', {
            refresh_token: refreshToken,
          })
          
          set({
            accessToken: response.data.access_token,
            refreshToken: response.data.refresh_token,
          })
          
          // Update axios header
          axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`
        } catch (error) {
          get().logout()
        }
      },
      
      setUser: (user) => set({ user }),
      
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)

// Setup axios interceptor for token refresh
axios.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().accessToken
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      
      try {
        await useAuthStore.getState().refreshAccessToken()
        return axios(originalRequest)
      } catch (refreshError) {
        useAuthStore.getState().logout()
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }
    
    return Promise.reject(error)
  }
)