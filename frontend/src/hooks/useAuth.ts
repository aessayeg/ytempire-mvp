/**
 * Authentication Hook
 * Owner: React Engineer
 */

import { useCallback } from 'react'
import { useAuthStore } from '@/stores/authStore'
import { apiClient } from '@/utils/api'
import { User, LoginCredentials, RegisterCredentials } from '@/types/auth'

export interface UseAuthReturn {
  // State
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  // Actions
  login: (email: string, password: string) => Promise<void>
  register: (credentials: RegisterCredentials) => Promise<void>
  logout: () => Promise<void>
  refreshAccessToken: () => Promise<void>
  forgotPassword: (email: string) => Promise<void>
  resetPassword: (token: string, password: string) => Promise<void>
  updateProfile: (updates: Partial<User>) => Promise<void>
  
  // OAuth
  loginWithGoogle: () => Promise<void>
  registerWithGoogle: () => Promise<void>
  
  // Utilities
  clearError: () => void
  checkAuthStatus: () => Promise<void>
}

export const useAuth = (): UseAuthReturn => {
  const {
    user,
    accessToken,
    refreshToken,
    isLoading,
    error,
    setUser,
    setTokens,
    setLoading,
    setError,
    clearAuth,
    clearError: storeClearError,
  } = useAuthStore()

  const isAuthenticated = Boolean(user && accessToken)

  const setAuthData = useCallback((authData: {
    user: User
    access_token: string
    refresh_token: string
  }) => {
    setUser(authData.user)
    setTokens(authData.access_token, authData.refresh_token)
    setError(null)
  }, [setUser, setTokens, setError])

  const login = useCallback(async (email: string, password: string) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await apiClient.login(email, password)
      setAuthData(response.data)
      
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          'Login failed'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setLoading, setError, setAuthData])

  const register = useCallback(async (credentials: RegisterCredentials) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await apiClient.register(credentials)
      
      // Registration might return auth data immediately or require email verification
      if (response.data.access_token) {
        setAuthData(response.data)
      }
      
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          'Registration failed'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setLoading, setError, setAuthData])

  const logout = useCallback(async () => {
    try {
      setLoading(true)
      
      // Call logout endpoint if we have a token
      if (accessToken) {
        await apiClient.logout()
      }
      
    } catch (err) {
      // Ignore logout errors - clear local state anyway
      console.warn('Logout API call failed:', err)
    } finally {
      clearAuth()
      setLoading(false)
    }
  }, [accessToken, clearAuth, setLoading])

  const refreshAccessToken = useCallback(async () => {
    if (!refreshToken) {
      throw new Error('No refresh token available')
    }

    try {
      const response = await apiClient.refreshToken(refreshToken)
      setTokens(response.data.access_token, response.data.refresh_token || refreshToken)
      return response.data.access_token
    } catch (err: any) {
      // Refresh failed - clear auth state
      clearAuth()
      throw err
    }
  }, [refreshToken, setTokens, clearAuth])

  const forgotPassword = useCallback(async (email: string) => {
    try {
      setLoading(true)
      setError(null)
      
      await apiClient.post('/auth/forgot-password', { email })
      
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          'Failed to send reset email'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setLoading, setError])

  const resetPassword = useCallback(async (token: string, password: string) => {
    try {
      setLoading(true)
      setError(null)
      
      await apiClient.post('/auth/reset-password', { token, password })
      
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          'Failed to reset password'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setLoading, setError])

  const updateProfile = useCallback(async (updates: Partial<User>) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await apiClient.updateCurrentUser(updates)
      setUser(response.data)
      
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          'Failed to update profile'
      setError(errorMessage)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setLoading, setError, setUser])

  const loginWithGoogle = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Redirect to Google OAuth endpoint
      const redirectUri = encodeURIComponent(`${window.location.origin}/auth/callback/google`)
      const googleAuthUrl = `/api/v1/auth/google?redirect_uri=${redirectUri}`
      
      window.location.href = googleAuthUrl
      
    } catch (err: any) {
      setError('Google login failed')
      setLoading(false)
      throw err
    }
  }, [setLoading, setError])

  const registerWithGoogle = useCallback(async () => {
    // Same as login for OAuth - the backend handles registration vs login
    return loginWithGoogle()
  }, [loginWithGoogle])

  const checkAuthStatus = useCallback(async () => {
    if (!accessToken) {
      return
    }

    try {
      setLoading(true)
      const response = await apiClient.getCurrentUser()
      setUser(response.data)
    } catch (err) {
      // Token might be expired, try to refresh
      if (refreshToken) {
        try {
          await refreshAccessToken()
          const response = await apiClient.getCurrentUser()
          setUser(response.data)
        } catch (refreshErr) {
          clearAuth()
        }
      } else {
        clearAuth()
      }
    } finally {
      setLoading(false)
    }
  }, [accessToken, refreshToken, setLoading, setUser, clearAuth, refreshAccessToken])

  const clearError = useCallback(() => {
    storeClearError()
  }, [storeClearError])

  return {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    
    // Actions
    login,
    register,
    logout,
    refreshAccessToken,
    forgotPassword,
    resetPassword,
    updateProfile,
    
    // OAuth
    loginWithGoogle,
    registerWithGoogle,
    
    // Utilities
    clearError,
    checkAuthStatus,
  }
}

export default useAuth