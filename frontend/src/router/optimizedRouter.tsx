/**
 * Optimized Router with Enhanced Code Splitting
 * Implements lazy loading with retry logic and predictive preloading
 */

import React, { Suspense, useEffect } from 'react';
import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import { lazyWithRetry, setupPredictivePreloading } from '../utils/lazyWithRetry';
import { LoadingSkeleton } from '../components/Loading';
import { ErrorBoundary } from '../components/ErrorBoundary';

// Lazy load all route components with retry logic
const Login = lazyWithRetry(() => import('../pages/Auth/Login'), 'Login');
const Register = lazyWithRetry(() => import('../pages/Auth/Register'), 'Register');
const DashboardLayout = lazyWithRetry(() => import('../layouts/DashboardLayout'), 'DashboardLayout');
const Dashboard = lazyWithRetry(() => import('../pages/Dashboard/Dashboard'), 'Dashboard');
const ChannelManagement = lazyWithRetry(() => import('../pages/Channels/ChannelManagement'), 'ChannelManagement');
const VideoQueue = lazyWithRetry(() => import('../pages/Videos/VideoQueue'), 'VideoQueue');
const VideoDetail = lazyWithRetry(() => import('../pages/Videos/VideoDetail'), 'VideoDetail');
const VideoGenerator = lazyWithRetry(() => import('../pages/Videos/VideoGenerator'), 'VideoGenerator');
const Analytics = lazyWithRetry(() => import('../pages/Analytics/Analytics'), 'Analytics');
const AnalyticsDashboard = lazyWithRetry(() => import('../pages/Analytics/AnalyticsDashboard'), 'AnalyticsDashboard');
const CostTracking = lazyWithRetry(() => import('../pages/Costs/CostTracking'), 'CostTracking');
const AITools = lazyWithRetry(() => import('../pages/AI/AITools'), 'AITools');
const Profile = lazyWithRetry(() => import('../pages/Profile/Profile'), 'Profile');
const Settings = lazyWithRetry(() => import('../pages/Settings/Settings'), 'Settings');
const BetaSignup = lazyWithRetry(() => import('../pages/BetaSignup'), 'BetaSignup');

// Create preload map for predictive loading
const preloadMap = new Map([
  ['/dashboard', () => import('../pages/Dashboard/Dashboard')],
  ['/channels', () => import('../pages/Channels/ChannelManagement')],
  ['/videos', () => import('../pages/Videos/VideoQueue')],
  ['/analytics', () => import('../pages/Analytics/Analytics')],
  ['/costs', () => import('../pages/Costs/CostTracking')],
  ['/ai-tools', () => import('../pages/AI/AITools')],
  ['/settings', () => import('../pages/Settings/Settings')],
  ['/profile', () => import('../pages/Profile/Profile')]
]);

// Route wrapper with error boundary and suspense
const RouteWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  useEffect(() => {
    // Setup predictive preloading for visible links
    setupPredictivePreloading('a[href^="/"]', preloadMap);
  }, []);

  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingSkeleton />}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
};

// Protected route wrapper
const ProtectedRoute: React.FC = () => {
  return (
    <RouteWrapper>
      <DashboardLayout>
        <Outlet />
      </DashboardLayout>
    </RouteWrapper>
  );
};

// Create router with optimized lazy loading
export const optimizedRouter = createBrowserRouter([
  {
    path: '/login',
    element: (
      <RouteWrapper>
        <Login />
      </RouteWrapper>
    )
  },
  {
    path: '/register',
    element: (
      <RouteWrapper>
        <Register />
      </RouteWrapper>
    )
  },
  {
    path: '/beta-signup',
    element: (
      <RouteWrapper>
        <BetaSignup />
      </RouteWrapper>
    )
  },
  {
    path: '/',
    element: <ProtectedRoute />,
    children: [
      {
        index: true,
        element: (
          <RouteWrapper>
            <Dashboard />
          </RouteWrapper>
        )
      },
      {
        path: 'dashboard',
        element: (
          <RouteWrapper>
            <Dashboard />
          </RouteWrapper>
        )
      },
      {
        path: 'channels',
        element: (
          <RouteWrapper>
            <ChannelManagement />
          </RouteWrapper>
        )
      },
      {
        path: 'videos',
        children: [
          {
            index: true,
            element: (
              <RouteWrapper>
                <VideoQueue />
              </RouteWrapper>
            )
          },
          {
            path: 'create',
            element: (
              <RouteWrapper>
                <VideoGenerator />
              </RouteWrapper>
            )
          },
          {
            path: ':id',
            element: (
              <RouteWrapper>
                <VideoDetail />
              </RouteWrapper>
            )
          }
        ]
      },
      {
        path: 'analytics',
        children: [
          {
            index: true,
            element: (
              <RouteWrapper>
                <Analytics />
              </RouteWrapper>
            )
          },
          {
            path: 'dashboard',
            element: (
              <RouteWrapper>
                <AnalyticsDashboard />
              </RouteWrapper>
            )
          }
        ]
      },
      {
        path: 'costs',
        element: (
          <RouteWrapper>
            <CostTracking />
          </RouteWrapper>
        )
      },
      {
        path: 'ai-tools',
        element: (
          <RouteWrapper>
            <AITools />
          </RouteWrapper>
        )
      },
      {
        path: 'profile',
        element: (
          <RouteWrapper>
            <Profile />
          </RouteWrapper>
        )
      },
      {
        path: 'settings',
        element: (
          <RouteWrapper>
            <Settings />
          </RouteWrapper>
        )
      }
    ]
  }
]);

export const OptimizedRouter: React.FC = () => {
  return <RouterProvider router={optimizedRouter} />;
};