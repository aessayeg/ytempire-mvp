/**
 * Sidebar Navigation Component
 * Owner: Dashboard Specialist
 */

import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Box,
  Divider,
  Collapse,
  Badge,
  IconButton,
  useTheme,
  useMediaQuery,
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  VideoLibrary as VideoLibraryIcon,
  Channel as ChannelIcon,
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  AccountCircle as AccountCircleIcon,
  Notifications as NotificationsIcon,
  ExpandLess,
  ExpandMore,
  Add as AddIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Queue as QueueIcon,
  AttachMoney as MoneyIcon,
  Assessment as AssessmentIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material'

import { useAuth } from '@/hooks/useAuth'

const SIDEBAR_WIDTH = 280
const SIDEBAR_COLLAPSED_WIDTH = 64

interface NavigationItem {
  id: string
  label: string
  icon: React.ReactNode
  path: string
  badge?: number
  children?: NavigationItem[]
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: <DashboardIcon />,
    path: '/dashboard',
  },
  {
    id: 'videos',
    label: 'Videos',
    icon: <VideoLibraryIcon />,
    path: '/videos',
    children: [
      {
        id: 'videos-all',
        label: 'All Videos',
        icon: <VideoLibraryIcon />,
        path: '/videos',
      },
      {
        id: 'videos-queue',
        label: 'Generation Queue',
        icon: <QueueIcon />,
        path: '/videos/queue',
        badge: 3, // Example: 3 videos in queue
      },
      {
        id: 'videos-create',
        label: 'Create Video',
        icon: <AddIcon />,
        path: '/videos/create',
      },
    ],
  },
  {
    id: 'channels',
    label: 'Channels',
    icon: <ChannelIcon />,
    path: '/channels',
    children: [
      {
        id: 'channels-all',
        label: 'My Channels',
        icon: <ChannelIcon />,
        path: '/channels',
      },
      {
        id: 'channels-add',
        label: 'Connect Channel',
        icon: <AddIcon />,
        path: '/channels/connect',
      },
    ],
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: <AnalyticsIcon />,
    path: '/analytics',
    children: [
      {
        id: 'analytics-overview',
        label: 'Overview',
        icon: <AssessmentIcon />,
        path: '/analytics',
      },
      {
        id: 'analytics-performance',
        label: 'Performance',
        icon: <TrendingUpIcon />,
        path: '/analytics/performance',
      },
      {
        id: 'analytics-costs',
        label: 'Cost Analysis',
        icon: <MoneyIcon />,
        path: '/analytics/costs',
      },
    ],
  },
  {
    id: 'trends',
    label: 'Trends',
    icon: <TrendingUpIcon />,
    path: '/trends',
  },
  {
    id: 'schedule',
    label: 'Schedule',
    icon: <ScheduleIcon />,
    path: '/schedule',
  },
]

const bottomNavigationItems: NavigationItem[] = [
  {
    id: 'settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    path: '/settings',
  },
  {
    id: 'help',
    label: 'Help & Support',
    icon: <HelpIcon />,
    path: '/help',
  },
]

interface SidebarProps {
  open: boolean
  onToggle: () => void
  permanent?: boolean
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  open, 
  onToggle, 
  permanent = false 
}) => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuth()
  
  const [expandedItems, setExpandedItems] = useState<string[]>(['videos', 'channels', 'analytics'])

  const toggleExpanded = (itemId: string) => {
    setExpandedItems(prev => 
      prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    )
  }

  const handleNavigation = (path: string) => {
    navigate(path)
    if (isMobile) {
      onToggle()
    }
  }

  const isItemActive = (path: string): boolean => {
    return location.pathname === path || location.pathname.startsWith(path + '/')
  }

  const renderNavigationItem = (item: NavigationItem, level: number = 0) => {
    const hasChildren = item.children && item.children.length > 0
    const isExpanded = expandedItems.includes(item.id)
    const isActive = isItemActive(item.path)

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding>
          <ListItemButton
            selected={isActive && !hasChildren}
            onClick={() => {
              if (hasChildren) {
                toggleExpanded(item.id)
              } else {
                handleNavigation(item.path)
              }
            }}
            sx={{
              pl: 2 + (level * 2),
              py: 1,
              borderRadius: 1,
              mx: 1,
              '&.Mui-selected': {
                backgroundColor: theme.palette.primary.main + '20',
                color: theme.palette.primary.main,
                '&:hover': {
                  backgroundColor: theme.palette.primary.main + '30',
                },
              },
            }}
          >
            <ListItemIcon 
              sx={{ 
                color: isActive ? theme.palette.primary.main : 'inherit',
                minWidth: 40,
              }}
            >
              {item.badge ? (
                <Badge badgeContent={item.badge} color="error">
                  {item.icon}
                </Badge>
              ) : (
                item.icon
              )}
            </ListItemIcon>
            
            <ListItemText 
              primary={item.label}
              sx={{
                '& .MuiTypography-root': {
                  fontSize: '0.875rem',
                  fontWeight: isActive ? 600 : 400,
                }
              }}
            />
            
            {hasChildren && (
              isExpanded ? <ExpandLess /> : <ExpandMore />
            )}
          </ListItemButton>
        </ListItem>
        
        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children?.map(child => 
                renderNavigationItem(child, level + 1)
              )}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    )
  }

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar sx={{ px: 2, py: 2 }}>
        <Box display="flex" alignItems="center" width="100%">
          <PlayIcon 
            sx={{ 
              color: theme.palette.primary.main, 
              fontSize: 32,
              mr: 1 
            }} 
          />
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              fontWeight: 'bold',
              color: theme.palette.primary.main,
              flexGrow: 1
            }}
          >
            YTEmpire
          </Typography>
          <IconButton size="small" color="inherit">
            <NotificationsIcon />
          </IconButton>
        </Box>
      </Toolbar>

      <Divider />

      {/* User Info */}
      <Box sx={{ p: 2 }}>
        <Box display="flex" alignItems="center">
          <AccountCircleIcon 
            sx={{ 
              fontSize: 40, 
              color: theme.palette.text.secondary,
              mr: 1.5 
            }} 
          />
          <Box>
            <Typography variant="subtitle2" fontWeight="medium">
              {user?.firstName} {user?.lastName}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {user?.subscriptionPlan?.toUpperCase() || 'FREE'} Plan
            </Typography>
          </Box>
        </Box>
      </Box>

      <Divider />

      {/* Main Navigation */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', py: 1 }}>
        <List>
          {navigationItems.map(item => renderNavigationItem(item))}
        </List>
      </Box>

      {/* Bottom Navigation */}
      <Box>
        <Divider />
        <List>
          {bottomNavigationItems.map(item => renderNavigationItem(item))}
        </List>
      </Box>

      {/* Quick Stats */}
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Box 
          sx={{ 
            bgcolor: theme.palette.grey[100],
            borderRadius: 1,
            p: 1.5,
          }}
        >
          <Typography variant="caption" color="text.secondary" gutterBottom>
            This Month
          </Typography>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box textAlign="center">
              <Typography variant="h6" color="primary">
                12
              </Typography>
              <Typography variant="caption">
                Videos
              </Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="h6" color="success.main">
                $24.50
              </Typography>
              <Typography variant="caption">
                Costs
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>
    </Box>
  )

  if (permanent) {
    return (
      <Drawer
        variant="permanent"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: SIDEBAR_WIDTH,
            boxSizing: 'border-box',
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
      >
        {drawerContent}
      </Drawer>
    )
  }

  return (
    <Drawer
      anchor="left"
      open={open}
      onClose={onToggle}
      variant={isMobile ? 'temporary' : 'persistent'}
      sx={{
        width: SIDEBAR_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: SIDEBAR_WIDTH,
          boxSizing: 'border-box',
          borderRight: `1px solid ${theme.palette.divider}`,
        },
      }}
      ModalProps={{
        keepMounted: true, // Better mobile performance
      }}
    >
      {drawerContent}
    </Drawer>
  )
}

export default Sidebar