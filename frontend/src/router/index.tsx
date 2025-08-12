import React, { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { CircularProgress, Box } from '@mui/material'
import { useAuth } from '../contexts/AuthContext'
import { RouteErrorBoundary } from '../components/ErrorBoundary/RouteErrorBoundary'

// Lazy load pages
const Login = lazy(() => import('../pages/Auth/Login'))
const Register = lazy(() => import('../pages/Auth/Register'))
const DashboardLayout = lazy(() => import('../layouts/DashboardLayout'))
const Dashboard = lazy(() => import('../pages/Dashboard/Dashboard'))
const ChannelManagement = lazy(() => import('../pages/Channels/ChannelManagement'))
const VideoQueue = lazy(() => import('../pages/Videos/VideoQueue'))
const VideoEditor = lazy(() => import('../pages/Videos/VideoEditor'))
const VideoGeneration = lazy(() => import('../pages/Videos/VideoGeneration'))
const BulkOperations = lazy(() => import('../pages/BulkOperations/BulkOperationsPage'))
const Analytics = lazy(() => import('../pages/Analytics/Analytics'))
const AnalyticsDashboard = lazy(() => import('../pages/Analytics/AnalyticsDashboard'))
const BusinessIntelligence = lazy(() => import('../pages/Analytics/BusinessIntelligence'))
const CostTracking = lazy(() => import('../pages/Costs/CostTracking'))
const AITools = lazy(() => import('../pages/AI/AITools'))
const Profile = lazy(() => import('../pages/Profile/Profile'))
const Settings = lazy(() => import('../pages/Settings/Settings'))
const AdvancedAnalytics = lazy(() => import('../pages/Analytics/AdvancedAnalytics'))
const SystemMonitoring = lazy(() => import('../pages/Monitoring/SystemMonitoring'))
const MobileDashboard = lazy(() => import('../pages/Dashboard/MobileDashboard'))
const ChannelDashboard = lazy(() => import('../pages/Channels/ChannelDashboard'))

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
              <RouteErrorBoundary>
                <DashboardLayout />
              </RouteErrorBoundary>
            </ProtectedRoute>
          }
        >
          <Route path="/dashboard" element={<RouteErrorBoundary><Dashboard /></RouteErrorBoundary>} />
          <Route path="/channels" element={<RouteErrorBoundary><ChannelManagement /></RouteErrorBoundary>} />
          <Route path="/videos" element={<RouteErrorBoundary><VideoQueue /></RouteErrorBoundary>} />
          <Route path="/videos/create" element={<RouteErrorBoundary><VideoGeneration /></RouteErrorBoundary>} />
          <Route path="/videos/editor/:id" element={<RouteErrorBoundary><VideoEditor /></RouteErrorBoundary>} />
          <Route path="/bulk-operations" element={<RouteErrorBoundary><BulkOperations /></RouteErrorBoundary>} />
          <Route path="/analytics" element={<RouteErrorBoundary><Analytics /></RouteErrorBoundary>} />
          <Route path="/analytics/dashboard" element={<RouteErrorBoundary><AnalyticsDashboard /></RouteErrorBoundary>} />
          <Route path="/analytics/business-intelligence" element={<RouteErrorBoundary><BusinessIntelligence /></RouteErrorBoundary>} />
          <Route path="/costs" element={<RouteErrorBoundary><CostTracking /></RouteErrorBoundary>} />
          <Route path="/ai-tools" element={<RouteErrorBoundary><AITools /></RouteErrorBoundary>} />
          <Route path="/profile" element={<RouteErrorBoundary><Profile /></RouteErrorBoundary>} />
          <Route path="/settings" element={<RouteErrorBoundary><Settings /></RouteErrorBoundary>} />
          <Route path="/analytics/advanced" element={<RouteErrorBoundary><AdvancedAnalytics /></RouteErrorBoundary>} />
          <Route path="/monitoring" element={<RouteErrorBoundary><SystemMonitoring /></RouteErrorBoundary>} />
          <Route path="/dashboard/mobile" element={<RouteErrorBoundary><MobileDashboard /></RouteErrorBoundary>} />
          <Route path="/channels/dashboard" element={<RouteErrorBoundary><ChannelDashboard /></RouteErrorBoundary>} />
        </Route>

        {/* 404 */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  )
}

export default Router