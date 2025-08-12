import React, { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { CircularProgress, Box } from '@mui/material'
import { useAuth } from '../contexts/AuthContext'

// Lazy load pages
const Login = lazy(() => import('../pages/Auth/Login'))
const Register = lazy(() => import('../pages/Auth/Register'))
const DashboardLayout = lazy(() => import('../layouts/DashboardLayout'))
const Dashboard = lazy(() => import('../pages/Dashboard/Dashboard'))
const ChannelManagement = lazy(() => import('../pages/Channels/ChannelManagement'))
const VideoQueue = lazy(() => import('../pages/Videos/VideoQueue'))
const Analytics = lazy(() => import('../pages/Analytics/Analytics'))
const AnalyticsDashboard = lazy(() => import('../pages/Analytics/AnalyticsDashboard'))
const CostTracking = lazy(() => import('../pages/Costs/CostTracking'))
const AITools = lazy(() => import('../pages/AI/AITools'))
const Profile = lazy(() => import('../pages/Profile/Profile'))
const Settings = lazy(() => import('../pages/Settings/Settings'))

// Loading component
const LoadingFallback = () => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
    }}
  >
    <CircularProgress />
  </Box>
)

// Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return <LoadingFallback />
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

// Public Route Component
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return <LoadingFallback />
  }

  if (user) {
    return <Navigate to="/dashboard" replace />
  }

  return <>{children}</>
}

const Router: React.FC = () => {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        <Route
          path="/register"
          element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          }
        />

        {/* Protected Routes */}
        <Route
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/channels" element={<ChannelManagement />} />
          <Route path="/videos" element={<VideoQueue />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/analytics/dashboard" element={<AnalyticsDashboard />} />
          <Route path="/costs" element={<CostTracking />} />
          <Route path="/ai-tools" element={<AITools />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
        </Route>

        {/* 404 */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  )
}

export default Router