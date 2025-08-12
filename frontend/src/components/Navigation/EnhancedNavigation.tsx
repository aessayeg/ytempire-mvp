import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Badge,
  Avatar,
  Collapse,
  Chip,
  InputBase,
  Paper,
  useMediaQuery,
  useTheme,
  SwipeableDrawer,
  BottomNavigation,
  BottomNavigationAction,
  Fab,
  Zoom,
  Tooltip,
  Menu,
  MenuItem,
  alpha,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search as SearchIcon,
  Close as CloseIcon,
  Dashboard,
  VideoLibrary,
  Analytics,
  YouTube,
  AttachMoney,
  Settings,
  Notifications,
  ExpandLess,
  ExpandMore,
  Add,
  Person,
  ChevronLeft,
  KeyboardArrowDown,
  Home,
  TrendingUp,
  CloudQueue,
  Schedule,
  PlayCircle,
  BarChart,
  Help,
  Feedback,
  Logout,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { HelpTooltip } from '../Common/HelpTooltip';

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  path?: string;
  children?: NavigationItem[];
  badge?: string | number;
  helpText?: string;
  quickAction?: () => void;
}

export const EnhancedNavigation: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  const navigate = useNavigate();
  const location = useLocation();
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedItems, setExpandedItems] = useState<string[]>([]);
  const [bottomNavValue, setBottomNavValue] = useState(0);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showFab, setShowFab] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  // Navigation items with help text
  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <Dashboard />,
      path: '/dashboard',
      helpText: 'View your overall performance metrics and activity',
    },
    {
      id: 'videos',
      label: 'Videos',
      icon: <VideoLibrary />,
      badge: '3',
      helpText: 'Manage your video creation and publishing',
      children: [
        {
          id: 'create',
          label: 'Create New',
          icon: <Add />,
          path: '/videos/create',
          helpText: 'Start creating a new video with AI assistance',
        },
        {
          id: 'library',
          label: 'Video Library',
          icon: <PlayCircle />,
          path: '/videos/library',
          helpText: 'Browse all your created videos',
        },
        {
          id: 'scheduled',
          label: 'Scheduled',
          icon: <Schedule />,
          path: '/videos/scheduled',
          badge: '5',
          helpText: 'View and manage scheduled uploads',
        },
        {
          id: 'processing',
          label: 'Processing',
          icon: <CloudQueue />,
          path: '/videos/processing',
          badge: '2',
          helpText: 'Track videos currently being generated',
        },
      ],
    },
    {
      id: 'channels',
      label: 'Channels',
      icon: <YouTube />,
      path: '/channels',
      helpText: 'Manage your YouTube channels',
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <Analytics />,
      path: '/analytics',
      helpText: 'Deep dive into your performance data',
      children: [
        {
          id: 'overview',
          label: 'Overview',
          icon: <BarChart />,
          path: '/analytics/overview',
        },
        {
          id: 'trends',
          label: 'Trends',
          icon: <TrendingUp />,
          path: '/analytics/trends',
        },
      ],
    },
    {
      id: 'monetization',
      label: 'Monetization',
      icon: <AttachMoney />,
      path: '/monetization',
      helpText: 'Track revenue and manage billing',
    },
  ];

  // Handle scroll for FAB visibility
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      setShowFab(currentScrollY < lastScrollY || currentScrollY < 100);
      setLastScrollY(currentScrollY);
    };

    if (isMobile) {
      window.addEventListener('scroll', handleScroll, { passive: true });
      return () => window.removeEventListener('scroll', handleScroll);
    }
  }, [lastScrollY, isMobile]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleSearchToggle = () => {
    setSearchOpen(!searchOpen);
    if (!searchOpen) {
      setTimeout(() => {
        document.getElementById('search-input')?.focus();
      }, 100);
    }
  };

  const handleExpandClick = (itemId: string) => {
    setExpandedItems(prev =>
      prev.includes(itemId)
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  };

  const handleQuickAction = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleQuickActionClose = () => {
    setAnchorEl(null);
  };

  const isActive = (path?: string) => {
    return path === location.pathname;
  };

  const renderNavigationItem = (item: NavigationItem, depth = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.id);
    const active = isActive(item.path);

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding sx={{ display: 'block' }}>
          <ListItemButton
            onClick={() => {
              if (item.path) {
                navigate(item.path);
                if (isMobile) setMobileOpen(false);
              } else if (hasChildren) {
                handleExpandClick(item.id);
              }
            }}
            sx={{
              minHeight: 48,
              px: depth > 0 ? 4 : 2.5,
              backgroundColor: active ? alpha(theme.palette.primary.main, 0.08) : 'transparent',
              borderLeft: active ? `3px solid ${theme.palette.primary.main}` : '3px solid transparent',
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.04),
              },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 40,
                color: active ? 'primary.main' : 'text.secondary',
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" fontWeight={active ? 600 : 400}>
                    {item.label}
                  </Typography>
                  {item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      color={active ? 'primary' : 'default'}
                      sx={{ height: 20, fontSize: 11 }}
                    />
                  )}
                  {item.helpText && !isMobile && (
                    <HelpTooltip
                      title={item.helpText}
                      size="small"
                    />
                  )}
                </Box>
              }
            />
            {hasChildren && (
              <IconButton size="small" edge="end">
                {isExpanded ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            )}
          </ListItemButton>
        </ListItem>
        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => renderNavigationItem(child, depth + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Drawer Header */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" fontWeight="bold" color="primary">
          YTEmpire
        </Typography>
        {isMobile && (
          <IconButton onClick={handleDrawerToggle}>
            <ChevronLeft />
          </IconButton>
        )}
      </Box>
      
      <Divider />
      
      {/* User Section */}
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Avatar sx={{ width: 40, height: 40 }}>U</Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" fontWeight="medium">
              User Name
            </Typography>
            <Chip label="Pro" size="small" color="primary" sx={{ height: 18 }} />
          </Box>
          <IconButton size="small">
            <KeyboardArrowDown />
          </IconButton>
        </Box>
      </Box>
      
      <Divider />
      
      {/* Navigation List */}
      <List sx={{ flex: 1, overflow: 'auto', py: 0 }}>
        {navigationItems.map(item => renderNavigationItem(item))}
      </List>
      
      <Divider />
      
      {/* Bottom Actions */}
      <List>
        <ListItemButton onClick={() => navigate('/help')}>
          <ListItemIcon>
            <Help />
          </ListItemIcon>
          <ListItemText primary="Help & Support" />
        </ListItemButton>
        <ListItemButton onClick={() => navigate('/settings')}>
          <ListItemIcon>
            <Settings />
          </ListItemIcon>
          <ListItemText primary="Settings" />
        </ListItemButton>
      </List>
    </Box>
  );

  return (
    <>
      {/* Top App Bar */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          backgroundColor: 'background.paper',
          borderBottom: 1,
          borderColor: 'divider',
          backdropFilter: 'blur(10px)',
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          {!searchOpen ? (
            <>
              <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
                {isMobile ? 'YTE' : 'YTEmpire'}
              </Typography>
              
              {/* Search */}
              <IconButton onClick={handleSearchToggle}>
                <SearchIcon />
              </IconButton>
              
              {/* Notifications */}
              <IconButton>
                <Badge badgeContent={4} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
              
              {/* Profile */}
              {!isMobile && (
                <IconButton onClick={handleQuickAction}>
                  <Avatar sx={{ width: 32, height: 32 }}>U</Avatar>
                </IconButton>
              )}
            </>
          ) : (
            <Paper
              sx={{
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                px: 2,
                py: 0.5,
                backgroundColor: 'background.default',
              }}
              elevation={0}
            >
              <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
              <InputBase
                id="search-input"
                placeholder="Search videos, channels, analytics..."
                fullWidth
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    // Handle search
                    setSearchOpen(false);
                  }
                }}
              />
              <IconButton onClick={handleSearchToggle}>
                <CloseIcon />
              </IconButton>
            </Paper>
          )}
        </Toolbar>
      </AppBar>
      
      {/* Side Drawer */}
      {isMobile ? (
        <SwipeableDrawer
          anchor="left"
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          onOpen={() => setMobileOpen(true)}
          sx={{
            '& .MuiDrawer-paper': {
              width: 280,
              boxSizing: 'border-box',
            },
          }}
        >
          {drawer}
        </SwipeableDrawer>
      ) : (
        <Drawer
          variant="permanent"
          sx={{
            width: 280,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 280,
              boxSizing: 'border-box',
              top: 64,
              height: 'calc(100% - 64px)',
            },
          }}
        >
          {drawer}
        </Drawer>
      )}
      
      {/* Mobile Bottom Navigation */}
      {isMobile && (
        <Paper
          sx={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            zIndex: theme.zIndex.appBar,
          }}
          elevation={3}
        >
          <BottomNavigation
            value={bottomNavValue}
            onChange={(event, newValue) => {
              setBottomNavValue(newValue);
            }}
            showLabels
          >
            <BottomNavigationAction
              label="Home"
              icon={<Home />}
              onClick={() => navigate('/dashboard')}
            />
            <BottomNavigationAction
              label="Videos"
              icon={<VideoLibrary />}
              onClick={() => navigate('/videos')}
            />
            <BottomNavigationAction
              label="Analytics"
              icon={<Analytics />}
              onClick={() => navigate('/analytics')}
            />
            <BottomNavigationAction
              label="More"
              icon={<Person />}
              onClick={handleQuickAction}
            />
          </BottomNavigation>
        </Paper>
      )}
      
      {/* Floating Action Button */}
      {isMobile && (
        <Zoom in={showFab}>
          <Fab
            color="primary"
            sx={{
              position: 'fixed',
              bottom: 72,
              right: 16,
            }}
            onClick={() => navigate('/videos/create')}
          >
            <Add />
          </Fab>
        </Zoom>
      )}
      
      {/* Quick Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleQuickActionClose}
      >
        <MenuItem onClick={() => { navigate('/settings'); handleQuickActionClose(); }}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { navigate('/feedback'); handleQuickActionClose(); }}>
          <ListItemIcon>
            <Feedback fontSize="small" />
          </ListItemIcon>
          <ListItemText>Feedback</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => { /* logout */ handleQuickActionClose(); }}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          <ListItemText>Logout</ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
};