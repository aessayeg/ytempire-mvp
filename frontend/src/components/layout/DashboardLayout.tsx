/**
 * Dashboard Layout Component
 * Owner: Dashboard Specialist
 */

import React, { useState, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import {
  Box,
  useTheme,
  useMediaQuery,
  Toolbar,
  Container,
  Breadcrumbs,
  Link,
  Typography,
  Fade,
} from '@mui/material'
import { Home as HomeIcon } from '@mui/icons-material'

import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { useAuth } from '@/hooks/useAuth'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

const SIDEBAR_WIDTH = 280

interface DashboardLayoutProps {
  children?: React.ReactNode
}

// Route title mapping
const routeTitles: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/videos': 'Videos',
  '/videos/create': 'Create Video',
  '/videos/queue': 'Generation Queue',
  '/channels': 'Channels',
  '/channels/connect': 'Connect Channel',
  '/analytics': 'Analytics',
  '/analytics/performance': 'Performance Analytics',
  '/analytics/costs': 'Cost Analysis',
  '/trends': 'Trends',
  '/schedule': 'Schedule',
  '/settings': 'Settings',
  '/settings/profile': 'Profile Settings',
  '/settings/billing': 'Billing & Usage',
  '/help': 'Help & Support',
}

// Breadcrumb mapping
const getBreadcrumbs = (pathname: string) => {
  const paths = pathname.split('/').filter(Boolean)
  const breadcrumbs = [{ name: 'Home', path: '/dashboard' }]
  
  let currentPath = ''
  paths.forEach((path, index) => {
    currentPath += `/${path}`
    const title = routeTitles[currentPath] || 
                  path.charAt(0).toUpperCase() + path.slice(1)
    
    if (currentPath !== '/dashboard') {
      breadcrumbs.push({
        name: title,
        path: currentPath,
        isLast: index === paths.length - 1
      })
    }
  })
  
  return breadcrumbs
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const location = useLocation()
  const { user, isLoading, checkAuthStatus } = useAuth()
  
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)

  const currentTitle = routeTitles[location.pathname] || 'YTEmpire'
  const breadcrumbs = getBreadcrumbs(location.pathname)

  // Check auth status on mount
  useEffect(() => {
    checkAuthStatus()
  }, [checkAuthStatus])

  // Close sidebar on mobile route changes
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false)
    }
  }, [location.pathname, isMobile])

  // Handle sidebar toggle
  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  // Show loading spinner during authentication check
  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        bgcolor="background.default"
      >
        <LoadingSpinner message="Loading dashboard..." />
      </Box>
    )
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Header */}
      <Header onMenuToggle={handleSidebarToggle} title={currentTitle} />

      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onToggle={handleSidebarToggle}
        permanent={!isMobile}
      />

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: isMobile ? '100%' : `calc(100% - ${sidebarOpen ? SIDEBAR_WIDTH : 0}px)`,
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          bgcolor: 'background.default',
          minHeight: '100vh',
        }}
      >
        {/* Toolbar Spacer */}
        <Toolbar />

        {/* Breadcrumbs */}
        {breadcrumbs.length > 1 && (
          <Box sx={{ px: 3, py: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Breadcrumbs aria-label="breadcrumb">
              {breadcrumbs.map((crumb, index) => (
                index === breadcrumbs.length - 1 ? (
                  <Typography key={crumb.path} color="text.primary" fontWeight="medium">
                    {crumb.name}
                  </Typography>
                ) : (
                  <Link
                    key={crumb.path}
                    underline="hover"
                    color="inherit"
                    href={crumb.path}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      '&:hover': {
                        color: 'primary.main',
                      },
                    }}
                  >
                    {index === 0 && <HomeIcon sx={{ mr: 0.5, fontSize: 20 }} />}
                    {crumb.name}
                  </Link>
                )
              ))}
            </Breadcrumbs>
          </Box>
        )}

        {/* Page Content */}
        <Container 
          maxWidth="xl" 
          sx={{ 
            py: 3,
            px: { xs: 2, sm: 3 },
            minHeight: 'calc(100vh - 120px)',
          }}
        >
          <Fade in timeout={300}>
            <Box>
              {children || <Outlet />}
            </Box>
          </Fade>
        </Container>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            py: 2,
            px: 3,
            mt: 'auto',
            borderTop: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper',
          }}
        >
          <Typography variant="body2" color="text.secondary" textAlign="center">
            © 2025 YTEmpire. All rights reserved.
          </Typography>
        </Box>
      </Box>

      {/* Mobile Backdrop */}
      {isMobile && sidebarOpen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            bgcolor: 'rgba(0, 0, 0, 0.5)',
            zIndex: theme.zIndex.drawer - 1,
          }}
          onClick={handleSidebarToggle}
        />
      )}
    </Box>
  )
}

export default DashboardLayout