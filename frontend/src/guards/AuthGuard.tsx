/**
 * Authentication Guard Component
 * Owner: React Engineer
 */

import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { Box } from '@mui/material'

import { useAuth } from '@/hooks/useAuth'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

interface AuthGuardProps {
  children: React.ReactNode
  requireAuth?: boolean
  redirectTo?: string
  roles?: string[]
}

export const AuthGuard: React.FC<AuthGuardProps> = ({
  children,
  requireAuth = true,
  redirectTo,
  roles = [],
}) => {
  const { user, isLoading, isAuthenticated } = useAuth()
  const location = useLocation()

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <LoadingSpinner message="Checking authentication..." />
      </Box>
    )
  }

  // If authentication is required but user is not authenticated
  if (requireAuth && !isAuthenticated) {
    return (
      <Navigate
        to={redirectTo || '/auth/login'}
        state={{ from: location.pathname }}
        replace
      />
    )
  }

  // If authentication is not required but user is authenticated (e.g., login page)
  if (!requireAuth && isAuthenticated) {
    const from = location.state?.from || '/dashboard'
    return <Navigate to={from} replace />
  }

  // Check role-based access
  if (requireAuth && roles.length > 0 && user) {
    const userRoles = user.roles || []
    const hasRequiredRole = roles.some(role => userRoles.includes(role))
    
    if (!hasRequiredRole) {
      return <Navigate to="/unauthorized" replace />
    }
  }

  return <>{children}</>
}

// Convenience wrapper for protected routes
export const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <AuthGuard requireAuth={true}>
      {children}
    </AuthGuard>
  )
}

// Convenience wrapper for public routes (redirects authenticated users)
export const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <AuthGuard requireAuth={false}>
      {children}
    </AuthGuard>
  )
}

// Role-based guard
export const RoleGuard: React.FC<{ 
  children: React.ReactNode
  roles: string[]
  fallback?: React.ReactNode
}> = ({ children, roles, fallback }) => {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return <LoadingSpinner />
  }

  if (!user) {
    return <Navigate to="/auth/login" replace />
  }

  const userRoles = user.roles || []
  const hasRequiredRole = roles.some(role => userRoles.includes(role))

  if (!hasRequiredRole) {
    return fallback ? <>{fallback}</> : <Navigate to="/unauthorized" replace />
  }

  return <>{children}</>
}

export default AuthGuard