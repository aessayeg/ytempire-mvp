import React, { useState } from 'react';
import {  useNavigate, useLocation  } from 'react-router-dom';
import { 
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
  IconButton,
  Collapse,
  Avatar,
  Chip
 } from '@mui/material';
import { 
  Dashboard,
  VideoLibrary,
  Analytics,
  TrendingUp,
  AttachMoney,
  Settings,
  Help,
  ChevronLeft,
  ChevronRight,
  ExpandLess,
  ExpandMore,
  PlayCircleOutline,
  Schedule,
  CloudUpload,
  YouTube,
  AddCircleOutline,
  ViewList,
  BarChart,
  Timeline,
  MonetizationOn,
  Receipt,
  AccountBalance,
  BusinessCenter,
  Person,
  Security,
  Notifications,
  Palette,
  Storage,
  Api,
  BugReport,
  ExitToApp,
  Monitor,
  MobileFriendly,
  Assessment,
  SpaceDashboard
 } from '@mui/icons-material';
import {  useAuthStore  } from '../../stores/authStore';

const drawerWidth = 280;
const miniDrawerWidth = 80;

interface MenuItem {
  id: string,
  label: string,

  icon: React.ReactNode;
  path?: string;
  children?: MenuItem[];
  badge?: string | number;
  requiredTier?: string;
}

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  
  const [open, setOpen] = useState(true);
  const [expandedItems, setExpandedItems] = useState<string[]>(['videos']);

  const menuItems: MenuItem[] = [ { id: 'dashboard',
      label: 'Dashboard',
      icon: <Dashboard />,
      children: [
        {
          id: 'main-dashboard',
          label: 'Main Dashboard',
          icon: <Dashboard />,
          path: '/dashboard' },
        { id: 'mobile-dashboard',
          label: 'Mobile View',
          icon: <MobileFriendly />,
          path: '/dashboard/mobile',
          badge: 'NEW' } ]
    },
    { id: 'videos',
      label: 'Videos',
      icon: <VideoLibrary />,
      children: [ {,
  id: 'create-video',
          label: 'Create New',
          icon: <AddCircleOutline />,
          path: '/videos/create',
          badge: 'NEW' },
        { id: 'video-library',
          label: 'Library',
          icon: <ViewList />,
          path: '/videos/library' },
        { id: 'scheduled',
          label: 'Scheduled',
          icon: <Schedule />,
          path: '/videos/scheduled',
          badge: user?.videos_per_day_limit },
        { id: 'publishing',
          label: 'Publishing Queue',
          icon: <CloudUpload />,
          path: '/videos/publishing' },
        { id: 'video-editor',
          label: 'Video Editor',
          icon: <PlayCircleOutline />,
          path: '/videos/editor/demo',
          badge: 'NEW' } ]
    },
    {
      id: 'channels',
      label: 'Channels',
      icon: <YouTube />,
      children: [ {,
  id: 'channels-manage',
          label: 'Manage',
          icon: <YouTube />,
          path: '/channels',
          badge: `${user?.channels_limit || 0} max`
        },
        { id: 'channels-dashboard',
          label: 'Dashboard',
          icon: <SpaceDashboard />,
          path: '/channels/dashboard',
          badge: 'NEW' } ]
    },
    { id: 'bulk-operations',
      label: 'Bulk Operations',
      icon: <ViewList />,
      path: '/bulk-operations',
      badge: 'NEW' },
    { id: 'analytics',
      label: 'Analytics',
      icon: <Analytics />,
      children: [ {,
  id: 'overview',
          label: 'Overview',
          icon: <BarChart />,
          path: '/analytics/overview' },
        { id: 'performance',
          label: 'Performance',
          icon: <Timeline />,
          path: '/analytics/performance' },
        { id: 'trends',
          label: 'Trends',
          icon: <TrendingUp />,
          path: '/analytics/trends' },
        { id: 'metrics-dashboard',
          label: 'Metrics Dashboard',
          icon: <Dashboard />,
          path: '/analytics/dashboard',
          badge: 'NEW' },
        { id: 'business-intelligence',
          label: 'Business Intelligence',
          icon: <BusinessCenter />,
          path: '/analytics/business-intelligence',
          badge: 'EXEC',
          requiredTier: 'pro' },
        { id: 'advanced-analytics',
          label: 'Advanced Analytics',
          icon: <Assessment />,
          path: '/analytics/advanced',
          badge: 'NEW' } ]
    },
    { id: 'monetization',
      label: 'Monetization',
      icon: <AttachMoney />,
      children: [ {,
  id: 'revenue',
          label: 'Revenue',
          icon: <MonetizationOn />,
          path: '/monetization/revenue' },
        { id: 'costs',
          label: 'Costs',
          icon: <Receipt />,
          path: '/monetization/costs' },
        { id: 'billing',
          label: 'Billing',
          icon: <AccountBalance />,
          path: '/monetization/billing' } ]
    },
    { id: 'settings',
      label: 'Settings',
      icon: <Settings />,
      children: [ {,
  id: 'profile',
          label: 'Profile',
          icon: <Person />,
          path: '/settings/profile' },
        { id: 'security',
          label: 'Security',
          icon: <Security />,
          path: '/settings/security' },
        { id: 'notifications',
          label: 'Notifications',
          icon: <Notifications />,
          path: '/settings/notifications' },
        { id: 'appearance',
          label: 'Appearance',
          icon: <Palette />,
          path: '/settings/appearance' },
        { id: 'api',
          label: 'API Keys',
          icon: <Api />,
          path: '/settings/api',
          requiredTier: 'pro' },
        { id: 'advanced',
          label: 'Advanced',
          icon: <Storage />,
          path: '/settings/advanced',
          requiredTier: 'enterprise' } ]
    },
    { id: 'monitoring',
      label: 'System Monitoring',
      icon: <Monitor />,
      path: '/monitoring',
      badge: 'NEW' }];

  const bottomMenuItems: MenuItem[] = [ { id: 'help',
      label: 'Help & Support',
      icon: <Help />,
      path: '/help' },
    { id: 'debug',
      label: 'Debug',
      icon: <BugReport />,
      path: '/debug',
      requiredTier: 'enterprise' } ];

  const handleDrawerToggle = () => {
    setOpen(!open)};

  const handleItemClick = (item: MenuItem) => {
    if (item.path) {
      navigate(item.path)} else if (item.children) {
      setExpandedItems(prev => {}
        prev.includes(item.id)
          ? prev.filter(id => id !== item.id)
          : [...prev, item.id]
      )}
  };

  const handleLogout = () => {
    logout();
    navigate('/auth/login')};

  const isItemActive = (item: MenuItem): boolean => {
    if (item.path) {
      return location.pathname === item.path;
    }
    if (item.children) {
      return item.children.some(child => child.path === location.pathname)}
    return false;
  };

  const renderMenuItem = (item: MenuItem, _depth = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.id);
    const isActive = isItemActive(item);
    const isLocked = item.requiredTier && user?.subscription_tier !== item.requiredTier;

    return (
    <>
      <React.Fragment key={item.id}>
        <ListItem disablePadding sx={{ display: 'block' }}>
          <ListItemButton
            onClick={() => handleItemClick(item}
            disabled={isLocked}
            sx={ {
              minHeight: 48,
              justifyContent: open ? 'initial' : 'center',
              px: 2.5,
              pl: depth > 0 ? 4 : 2.5,
              backgroundColor: isActive ? 'action.selected' : 'transparent',
              '&:hover': {
                backgroundColor: 'action.hover' }
            }}
          >
            <ListItemIcon
              sx={ {
                minWidth: 0,
                mr: open ? 3 : 'auto',
                justifyContent: 'center',
                color: isActive ? 'primary.main' : 'inherit' }}
            >
              {item.icon}
            </ListItemIcon>
      <ListItemText
              primary={item.label}
              sx={ {
                opacity: open ? 1 : 0,
                color: isActive ? 'primary.main' : 'inherit' }}
            />
            {open && (
              <>
                {item.badge && !isLocked && (
                  <Chip
                    label={item.badge}
                    size="small"
                    color={typeof item.badge === 'string' && item.badge === 'NEW' ? 'secondary' : 'default'}
                    sx={{ ml: 1 }}
                  />
                )}
                {isLocked && (
                  <Chip
                    label="PRO"
                    size="small"
                    color="warning"
                    sx={{ ml: 1 }}
                  />
                )}
                {hasChildren && (
                  isExpanded ? <ExpandLess /> </>: <ExpandMore />
                )}
              </>
            )}
          </ListItemButton>
        </ListItem>
        {hasChildren && (
          <Collapse in={isExpanded && open} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => renderMenuItem(child, depth + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    </>
  )};

  return (
    <>
      <Drawer
      variant="permanent"
      sx={ {
        width: open ? drawerWidth : miniDrawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? drawerWidth : miniDrawerWidth,
          boxSizing: 'border-box',
          transition: theme => theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen }),
          overflowX: 'hidden',

        }
      }}
    >
      <Box
        sx={ {
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          p: 2,
          minHeight: 64 }}
      >
        {open && (
          <Typography variant="h6" fontWeight="bold" color="primary">
            YTEmpire
          </Typography>
        )}
        <IconButton onClick={handleDrawerToggle}>
          {open ? <ChevronLeft /> </>: <ChevronRight />}
        </IconButton>
      </Box>
      <Divider />
      
      {open && user && (
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Avatar sx={{ width: 40, height: 40, mr: 2 }}>
              {user.username[0].toUpperCase()}
            </Avatar>
            <Box>
              <Typography variant="subtitle2" fontWeight="bold">
                {user.username}
              </Typography>
              <Chip
                label={user.subscription_tier}
                size="small"
                color={user.subscription_tier === 'enterprise' ? 'success' : 'primary'}
              />
            </Box>
          </Box>
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Videos Today: {user.total_videos_generated}/{user.videos_per_day_limit}
            </Typography>
          </Box>
        </Box>
      )}
      <Divider />
      
      <List sx={{ flex: 1, overflow: 'auto' }}>
        {menuItems.map(item => renderMenuItem(item))}
      </List>
      
      <Divider />
      
      <List>
        {bottomMenuItems.map(item => renderMenuItem(item))}
        <ListItem disablePadding>
          <ListItemButton
            onClick={handleLogout}
            sx={ {
              minHeight: 48,
              justifyContent: open ? 'initial' : 'center',
              px: 2.5 }}
          >
            <ListItemIcon
              sx={ {
                minWidth: 0,
                mr: open ? 3 : 'auto',
                justifyContent: 'center' }}
            >
              <ExitToApp />
            </ListItemIcon>
            <ListItemText
              primary="Logout"
              sx={{ opacity: open ? 1 : 0 }}
            />
          </ListItemButton>
        </ListItem>
      </List>
    </Drawer>
  </>
  )};`