/**
 * Header Component
 * Owner: Dashboard Specialist
 */

import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Box,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  AccountCircle as AccountCircleIcon,
  Logout as LogoutIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Help as HelpIcon,
  Person as PersonIcon,
  CreditCard as CreditCardIcon,
} from '@mui/icons-material'

import { useAuth } from '@/hooks/useAuth'
import { useTheme } from '@/hooks/useTheme'

interface HeaderProps {
  onMenuToggle: () => void
  title?: string
}

export const Header: React.FC<HeaderProps> = ({ onMenuToggle, title = 'Dashboard' }) => {
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const { isDarkMode, toggleTheme } = useTheme()
  
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null)
  const [notificationMenuAnchor, setNotificationMenuAnchor] = useState<null | HTMLElement>(null)

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget)
  }

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null)
  }

  const handleNotificationMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationMenuAnchor(event.currentTarget)
  }

  const handleNotificationMenuClose = () => {
    setNotificationMenuAnchor(null)
  }

  const handleLogout = async () => {
    handleUserMenuClose()
    await logout()
    navigate('/auth/login')
  }

  const handleProfileClick = () => {
    handleUserMenuClose()
    navigate('/settings/profile')
  }

  const handleSettingsClick = () => {
    handleUserMenuClose()
    navigate('/settings')
  }

  const handleBillingClick = () => {
    handleUserMenuClose()
    navigate('/settings/billing')
  }

  const mockNotifications = [
    { id: 1, title: 'Video Generation Complete', message: '"AI Trends 2025" is ready for review', time: '2 min ago', unread: true },
    { id: 2, title: 'Channel Connected', message: 'Tech Tutorials channel linked successfully', time: '1 hour ago', unread: true },
    { id: 3, title: 'Cost Alert', message: 'Monthly spend: $127 of $300 budget', time: '3 hours ago', unread: false },
  ]

  const unreadNotifications = mockNotifications.filter(n => n.unread).length

  return (
    <AppBar 
      position="fixed" 
      sx={{ 
        zIndex: (theme) => theme.zIndex.drawer + 1,
        bgcolor: 'background.paper',
        color: 'text.primary',
        boxShadow: 1,
        borderBottom: 1,
        borderColor: 'divider'
      }}
      elevation={0}
    >
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="open drawer"
          onClick={onMenuToggle}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography 
          variant="h6" 
          component="div" 
          sx={{ 
            flexGrow: 1,
            fontWeight: 600,
            color: 'primary.main'
          }}
        >
          {title}
        </Typography>

        {/* Theme Toggle */}
        <Tooltip title={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}>
          <IconButton
            color="inherit"
            onClick={toggleTheme}
            sx={{ mr: 1 }}
          >
            {isDarkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Tooltip>

        {/* Notifications */}
        <Tooltip title="Notifications">
          <IconButton
            color="inherit"
            onClick={handleNotificationMenuOpen}
            sx={{ mr: 1 }}
          >
            <Badge badgeContent={unreadNotifications} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* User Menu */}
        <Tooltip title="Account settings">
          <IconButton
            onClick={handleUserMenuOpen}
            size="small"
            sx={{ ml: 1 }}
            aria-controls={Boolean(userMenuAnchor) ? 'account-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={Boolean(userMenuAnchor) ? 'true' : undefined}
          >
            <Avatar sx={{ width: 32, height: 32 }}>
              {user?.firstName?.[0]}{user?.lastName?.[0]}
            </Avatar>
          </IconButton>
        </Tooltip>

        {/* User Menu Dropdown */}
        <Menu
          anchorEl={userMenuAnchor}
          id="account-menu"
          open={Boolean(userMenuAnchor)}
          onClose={handleUserMenuClose}
          onClick={handleUserMenuClose}
          PaperProps={{
            elevation: 0,
            sx: {
              overflow: 'visible',
              filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
              mt: 1.5,
              '& .MuiAvatar-root': {
                width: 32,
                height: 32,
                ml: -0.5,
                mr: 1,
              },
              '&:before': {
                content: '""',
                display: 'block',
                position: 'absolute',
                top: 0,
                right: 14,
                width: 10,
                height: 10,
                bgcolor: 'background.paper',
                transform: 'translateY(-50%) rotate(45deg)',
                zIndex: 0,
              },
            },
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          {/* User Info */}
          <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="subtitle1" fontWeight="medium">
              {user?.firstName} {user?.lastName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {user?.email}
            </Typography>
            <Typography variant="caption" color="primary">
              {user?.subscriptionPlan?.toUpperCase() || 'FREE'} Plan
            </Typography>
          </Box>

          <MenuItem onClick={handleProfileClick}>
            <ListItemIcon>
              <PersonIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>My Profile</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleBillingClick}>
            <ListItemIcon>
              <CreditCardIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Billing & Usage</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleSettingsClick}>
            <ListItemIcon>
              <SettingsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Settings</ListItemText>
          </MenuItem>

          <MenuItem onClick={() => navigate('/help')}>
            <ListItemIcon>
              <HelpIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Help & Support</ListItemText>
          </MenuItem>

          <Divider />

          <MenuItem onClick={handleLogout}>
            <ListItemIcon>
              <LogoutIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Logout</ListItemText>
          </MenuItem>
        </Menu>

        {/* Notifications Menu */}
        <Menu
          anchorEl={notificationMenuAnchor}
          open={Boolean(notificationMenuAnchor)}
          onClose={handleNotificationMenuClose}
          PaperProps={{
            sx: { width: 360, maxHeight: 400 }
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h6">Notifications</Typography>
          </Box>

          {mockNotifications.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography color="text.secondary">
                No new notifications
              </Typography>
            </Box>
          ) : (
            mockNotifications.map((notification) => (
              <MenuItem
                key={notification.id}
                onClick={handleNotificationMenuClose}
                sx={{
                  alignItems: 'flex-start',
                  py: 1.5,
                  borderLeft: notification.unread ? 3 : 0,
                  borderColor: 'primary.main',
                  bgcolor: notification.unread ? 'primary.light' + '08' : 'transparent',
                }}
              >
                <Box>
                  <Typography variant="subtitle2" fontWeight="medium">
                    {notification.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                    {notification.message}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {notification.time}
                  </Typography>
                </Box>
              </MenuItem>
            ))
          )}

          <Divider />
          <MenuItem onClick={() => navigate('/notifications')}>
            <Typography variant="body2" color="primary" textAlign="center" width="100%">
              View All Notifications
            </Typography>
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  )
}

export default Header