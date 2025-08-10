import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import { useAuthStore } from '../stores/authStore';

// Lazy load components for better performance
const LoginForm = lazy(() => import('../components/Auth/LoginForm').then(m => ({ default: m.LoginForm })));
const RegisterForm = lazy(() => import('../components/Auth/RegisterForm').then(m => ({ default: m.RegisterForm })));
const ForgotPasswordForm = lazy(() => import('../components/Auth/ForgotPasswordForm').then(m => ({ default: m.ForgotPasswordForm })));
const DashboardLayout = lazy(() => import('../layouts/DashboardLayout').then(m => ({ default: m.DashboardLayout })));
const Dashboard = lazy(() => import('../pages/Dashboard').then(m => ({ default: m.Dashboard })));
const VideoCreate = lazy(() => import('../pages/Videos/Create').then(m => ({ default: m.VideoCreate })));
const VideoLibrary = lazy(() => import('../pages/Videos/Library').then(m => ({ default: m.VideoLibrary })));
const VideoScheduled = lazy(() => import('../pages/Videos/Scheduled').then(m => ({ default: m.VideoScheduled })));
const VideoPublishing = lazy(() => import('../pages/Videos/Publishing').then(m => ({ default: m.VideoPublishing })));
const Channels = lazy(() => import('../pages/Channels').then(m => ({ default: m.Channels })));
const AnalyticsOverview = lazy(() => import('../pages/Analytics/Overview').then(m => ({ default: m.AnalyticsOverview })));
const AnalyticsPerformance = lazy(() => import('../pages/Analytics/Performance').then(m => ({ default: m.AnalyticsPerformance })));
const AnalyticsTrends = lazy(() => import('../pages/Analytics/Trends').then(m => ({ default: m.AnalyticsTrends })));
const Revenue = lazy(() => import('../pages/Monetization/Revenue').then(m => ({ default: m.Revenue })));
const Costs = lazy(() => import('../pages/Monetization/Costs').then(m => ({ default: m.Costs })));
const Billing = lazy(() => import('../pages/Monetization/Billing').then(m => ({ default: m.Billing })));
const Profile = lazy(() => import('../pages/Settings/Profile').then(m => ({ default: m.Profile })));
const Security = lazy(() => import('../pages/Settings/Security').then(m => ({ default: m.Security })));
const Notifications = lazy(() => import('../pages/Settings/Notifications').then(m => ({ default: m.Notifications })));
const Appearance = lazy(() => import('../pages/Settings/Appearance').then(m => ({ default: m.Appearance })));
const ApiKeys = lazy(() => import('../pages/Settings/ApiKeys').then(m => ({ default: m.ApiKeys })));
const Advanced = lazy(() => import('../pages/Settings/Advanced').then(m => ({ default: m.Advanced })));
const Help = lazy(() => import('../pages/Help').then(m => ({ default: m.Help })));
const NotFound = lazy(() => import('../pages/NotFound').then(m => ({ default: m.NotFound })));

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
);

// Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredTier?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, requiredTier }) => {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/auth/login" replace />;
  }

  if (requiredTier && user?.subscription_tier !== requiredTier && user?.subscription_tier !== 'enterprise') {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Public Route Component (redirects to dashboard if already authenticated)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

export const AppRouter: React.FC = () => {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        
        <Route path="/auth">
          <Route
            path="login"
            element={
              <PublicRoute>
                <LoginForm />
              </PublicRoute>
            }
          />
          <Route
            path="register"
            element={
              <PublicRoute>
                <RegisterForm />
              </PublicRoute>
            }
          />
          <Route
            path="forgot-password"
            element={
              <PublicRoute>
                <ForgotPasswordForm />
              </PublicRoute>
            }
          />
        </Route>

        {/* Protected Routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route path="dashboard" element={<Dashboard />} />
          
          {/* Videos Routes */}
          <Route path="videos">
            <Route path="create" element={<VideoCreate />} />
            <Route path="library" element={<VideoLibrary />} />
            <Route path="scheduled" element={<VideoScheduled />} />
            <Route path="publishing" element={<VideoPublishing />} />
          </Route>

          {/* Channels */}
          <Route path="channels" element={<Channels />} />

          {/* Analytics Routes */}
          <Route path="analytics">
            <Route path="overview" element={<AnalyticsOverview />} />
            <Route path="performance" element={<AnalyticsPerformance />} />
            <Route path="trends" element={<AnalyticsTrends />} />
          </Route>

          {/* Monetization Routes */}
          <Route path="monetization">
            <Route path="revenue" element={<Revenue />} />
            <Route path="costs" element={<Costs />} />
            <Route path="billing" element={<Billing />} />
          </Route>

          {/* Settings Routes */}
          <Route path="settings">
            <Route index element={<Profile />} />
            <Route path="profile" element={<Profile />} />
            <Route path="security" element={<Security />} />
            <Route path="notifications" element={<Notifications />} />
            <Route path="appearance" element={<Appearance />} />
            <Route
              path="api"
              element={
                <ProtectedRoute requiredTier="pro">
                  <ApiKeys />
                </ProtectedRoute>
              }
            />
            <Route
              path="advanced"
              element={
                <ProtectedRoute requiredTier="enterprise">
                  <Advanced />
                </ProtectedRoute>
              }
            />
          </Route>

          {/* Help */}
          <Route path="help" element={<Help />} />
        </Route>

        {/* 404 Not Found */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
};