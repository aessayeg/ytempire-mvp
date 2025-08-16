/**
 * Mobile Layout Component
 * Provides responsive mobile-first layout for the application
 */

import React, { useState, useEffect } from 'react';
import { 
  Box,
  Drawer,
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  useTheme,
  useMediaQuery,
  BottomNavigation,
  BottomNavigationAction,
  SwipeableDrawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Paper
 } from '@mui/material';
import { 
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Home as HomeIcon
,
  Add as AddIcon
 } from '@mui/icons-material';
import {  useNavigate, useLocation, Outlet  } from 'react-router-dom';

interface MobileLayoutProps {
  
children?: React.ReactNode;


}

const MobileLayout: React.FC<MobileLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [bottomNavValue, setBottomNavValue] = useState(0);
  const [notificationCount, setNotificationCount] = useState(3);

  // Navigation items for drawer
  const navigationItems = [ { label: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { label: 'Videos', icon: <VideoLibraryIcon />, path: '/videos' },
    { label: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
    { label: 'Settings', icon: <SettingsIcon />, path: '/settings' } ];

  // Bottom navigation items for mobile
  const bottomNavItems = [ { label: 'Home', icon: <HomeIcon />, path: '/dashboard' },
    { label: 'Videos', icon: <VideoLibraryIcon />, path: '/videos' },
    { label: 'Create', icon: <AddIcon />, path: '/videos/new' },
    { label: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
    { label: 'Profile', icon: <PersonIcon />, path: '/profile' } ];

  useEffect(() => {
    // Update bottom navigation based on current path
    const currentIndex = bottomNavItems.findIndex(item =>
      location.pathname.startsWith(item.path)
    );
    if (currentIndex !== -1) {
      setBottomNavValue(currentIndex)}
  }, [location.pathname]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen)
};
  const handleNavigation = (path: string) => {
    navigate(path);
    setDrawerOpen(false)
};
  const handleBottomNavChange = (_: React.SyntheticEvent, newValue: number) => {
    setBottomNavValue(newValue);
    navigate(bottomNavItems[newValue].path)
};
  const drawerContent = (
    <Box sx={{ width: 280, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Drawer Header */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" fontWeight="bold">
          YTEmpire
        </Typography>
        <IconButton onClick={handleDrawerToggle} size="small">
          <CloseIcon />
        </IconButton>
      </Box>
      
      <Divider />
      
      {/* User Info */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Avatar sx={{ width: 48, height: 48 }}>U</Avatar>
        <Box>
          <Typography variant="subtitle1" fontWeight="medium">
            User Name
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Premium Plan
          </Typography>
        </Box>
      </Box>
      
      <Divider />
      
      {/* Navigation Items */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem
            button
            key={item.label}
            onClick={() => handleNavigation(item.path}
            selected={location.pathname.startsWith(item.path)}
            sx={ {
              mx: 1,
              borderRadius: 1,
              '&.Mui-selected': {
                backgroundColor: theme.palette.primary.main + '20',
                '&:hover': {
                  backgroundColor: theme.palette.primary.main + '30' }
          >
            <ListItemIcon sx={{ minWidth: 40 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>
      
      <Divider />
      
      {/* Quick Stats */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          QUICK STATS
        </Typography>
        <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <Typography variant="body2">
            Active Videos: <strong>24</strong>
          </Typography>
          <Typography variant="body2">
            Today's Views: <strong>1,234</strong>
          </Typography>
          <Typography variant="body2">
            Revenue: <strong>$456</strong>
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <>
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100 vh' }}>
      {/* App Bar for Mobile */}
      {isMobile && (
        <AppBar 
          position="fixed" 
          elevation={0}
          sx={{ 
            backgroundColor: theme.palette.background.paper,
            color: theme.palette.text.primary,
            borderBottom: `1px solid ${theme.palette.divider}
          }}
        >
          <Toolbar sx={{ px: 2 }}>
            <IconButton
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
      <Typography variant="h6" sx={{ flexGrow: 1 }}>
              YTEmpire
            </Typography>
            
            <IconButton>
              <Badge badgeContent={notificationCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Toolbar>
        </AppBar>
      )}
      {/* Drawer for Mobile/Tablet */}
      <SwipeableDrawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false}
        onOpen={() => setDrawerOpen(true}
        sx={ {
          '& .MuiDrawer-paper': {
            width: 280,
            boxSizing: 'border-box' }
        }}
      >
        {drawerContent}
      </SwipeableDrawer>

      {/* Main Content Area */}
      <Box
        component="main"
        sx={ {
          flexGrow: 1,
          overflow: 'auto',
          pt: isMobile ? 7 : 0,
          pb: isMobile ? 7 : 0,
          px: isMobile ? 2 : 3,
          py: 2,
          backgroundColor: theme.palette.background.default }}
      >
        {children || <Outlet />}
      </Box>

      {/* Bottom Navigation for Mobile */}
      {isMobile && (
        <Paper
          elevation={8}
          sx={ {
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            zIndex: theme.zIndex.appBar }}
        >
          <BottomNavigation
            value={bottomNavValue}
            onChange={handleBottomNavChange}
            showLabels={false}
            sx={ {
              height: 56,
              '& .MuiBottomNavigationAction-root': {
                minWidth: 'auto',
                padding: '6px 0' }
            }}
          >
            {bottomNavItems.map((item) => (
              <BottomNavigationAction
                key={item.label}
                label={item.label}
                icon={ item.label === 'Create' ? (
                  <Box
                    sx={{
                      width: 48,
                      height: 48,
                      borderRadius: '50%',
                      backgroundColor: theme.palette.primary.main,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      transform: 'translateY(-8px)',
                      boxShadow: theme.shadows[4] }}
                  >
                    {item.icon}
                  </Box>
                ) : (
                  item.icon
                )}
              />
            ))}
          </BottomNavigation>
        </Paper>
      )}
    </Box>
  </>
  )};

export default MobileLayout;

// Export additional mobile-specific components
export const MobileHeader: React.FC<{ title: string; onMenuClick?: () => void }> = ({ 
  title, 
  onMenuClick }) => (
  <AppBar position="static" elevation={0} color="transparent">
    <Toolbar>
      {onMenuClick && (
        <IconButton edge="start" onClick={onMenuClick} sx={{ mr: 2 }}>
          <MenuIcon />
        </IconButton>
      )}
      <Typography variant="h6" sx={{ flexGrow: 1 }}>
        {title}
      </Typography>
    </Toolbar>
  </AppBar>
);

export const MobileCard: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Paper
    elevation={0}
    sx={{
      p: 2,
      mb: 2,
      borderRadius: 2,
      border: (theme) => `1px solid ${theme.palette.divider}}}
  >
    {children}
  </Paper>
);
