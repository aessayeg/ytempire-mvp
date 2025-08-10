/**
 * Store Provider
 * Owner: Frontend Team Lead
 * 
 * Provider component to initialize stores and provide context
 */

import React, { useEffect } from 'react'
import { useUIStore, useAuthStore } from '@/stores'
import { useRealTimeUpdates } from '@/hooks/useStores'

interface StoreProviderProps {
  children: React.ReactNode
}

export const StoreProvider: React.FC<StoreProviderProps> = ({ children }) => {
  const { updateScreenSize, setGlobalLoading } = useUIStore()
  const { refreshAccessToken } = useAuthStore()
  
  // Initialize real-time updates
  useRealTimeUpdates()
  
  useEffect(() => {
    // Initialize screen size
    const handleResize = () => {
      updateScreenSize(window.innerWidth, window.innerHeight)
    }
    
    handleResize()
    window.addEventListener('resize', handleResize)
    
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [updateScreenSize])
  
  useEffect(() => {
    // Initialize auth token refresh
    const initializeAuth = async () => {
      const token = useAuthStore.getState().accessToken
      if (token) {
        try {
          await refreshAccessToken()
        } catch (error) {
          console.warn('Failed to refresh token on app initialization')
        }
      }
    }
    
    initializeAuth()
  }, [refreshAccessToken])
  
  useEffect(() => {
    // Set up global error handler
    const handleUnhandledError = (event: ErrorEvent) => {
      console.error('Unhandled error:', event.error)
      useUIStore.getState().showErrorMessage(
        'An unexpected error occurred. Please refresh the page.',
        'System Error'
      )
    }
    
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason)
      useUIStore.getState().showErrorMessage(
        'A network or processing error occurred. Please try again.',
        'Request Failed'
      )
    }
    
    window.addEventListener('error', handleUnhandledError)
    window.addEventListener('unhandledrejection', handleUnhandledRejection)
    
    return () => {
      window.removeEventListener('error', handleUnhandledError)
      window.removeEventListener('unhandledrejection', handleUnhandledRejection)
    }
  }, [])
  
  return <>{children}</>
}

export default StoreProvider