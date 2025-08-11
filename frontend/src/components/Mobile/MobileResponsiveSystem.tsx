/**
 * Mobile Responsive System
 * Comprehensive mobile-first design system with adaptive layouts
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Drawer,
  AppBar,
  Toolbar,
  Badge,
  Fab,
  Skeleton,
  Button,
  SwipeableDrawer,
  BottomNavigation,
  BottomNavigationAction,
  Tabs,
  Tab,
  Chip,
  LinearProgress,
  CircularProgress,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Snackbar,
  Alert,
  Paper,
  Grid,
  useMediaQuery,
  useTheme,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Stack,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search as SearchIcon,
  Close as CloseIcon,
  Add as AddIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  Home as HomeIcon,
  Analytics as AnalyticsIcon,
  VideoLibrary as VideoLibraryIcon,
  Settings as SettingsIcon,
  AccountCircle as AccountIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Share as ShareIcon,
  Download as DownloadIcon,
  MoreVert as MoreVertIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  AttachMoney as MoneyIcon,
  Visibility as ViewsIcon,
  ThumbUp as LikesIcon,
  Schedule as ScheduleIcon,
  CloudUpload as UploadIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useOptimizedStore } from '../../stores/optimizedStore';

// Types
interface MobileMetric {
  id: string;
  title: string;
  value: string | number;
  change: number;
  changeType: 'positive' | 'negative' | 'neutral';
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}

interface MobileCard {
  id: string;
  title: string;
  subtitle: string;
  avatar?: string;
  status: 'active' | 'pending' | 'completed' | 'failed';
  progress?: number;
  actions?: Array<{
    icon: React.ReactNode;
    label: string;
    action: () => void;
  }>;
  metadata?: Record<string, any>;
}

interface NavigationTab {
  label: string;
  icon: React.ReactNode;
  badge?: number;
  disabled?: boolean;
}

// Custom hooks for mobile features
const useSwipeGestures = (onSwipeLeft?: () => void, onSwipeRight?: () => void) => {
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  const minSwipeDistance = 50;

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && onSwipeLeft) onSwipeLeft();
    if (isRightSwipe && onSwipeRight) onSwipeRight();
  };

  return {
    onTouchStart,
    onTouchMove,
    onTouchEnd,
  };
};

const usePullToRefresh = (onRefresh: () => Promise<void>) => {
  const [isPulling, setIsPulling] = useState(false);
  const [pullDistance, setPullDistance] = useState(0);
  const startY = useRef<number>(0);
  const currentY = useRef<number>(0);

  const handleTouchStart = (e: React.TouchEvent) => {
    startY.current = e.touches[0].clientY;
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    currentY.current = e.touches[0].clientY;
    const distance = currentY.current - startY.current;
    
    if (distance > 0 && window.scrollY === 0) {
      e.preventDefault();
      setPullDistance(Math.min(distance, 100));
      setIsPulling(distance > 60);
    }
  };

  const handleTouchEnd = async () => {
    if (isPulling && pullDistance > 60) {
      await onRefresh();
    }
    setIsPulling(false);
    setPullDistance(0);
  };

  return {
    isPulling,
    pullDistance,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
  };
};

export const MobileResponsiveSystem: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'lg'));
  const isSmallMobile = useMediaQuery('(max-width:400px)');

  // State management
  const [bottomNavValue, setBottomNavValue] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [speedDialOpen, setSpeedDialOpen] = useState(false);
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [showNotifications, setShowNotifications] = useState(false);

  const { addNotification } = useOptimizedStore();

  // Sample data
  const metrics: MobileMetric[] = [
    {
      id: 'revenue',
      title: 'Revenue',
      value: '$12,450',
      change: 15.3,
      changeType: 'positive',
      icon: <MoneyIcon />,
      color: '#4caf50',
      subtitle: 'vs last month',
    },
    {
      id: 'views',
      title: 'Views',
      value: '2.4M',
      change: -5.2,
      changeType: 'negative',
      icon: <ViewsIcon />,
      color: '#2196f3',
      subtitle: 'total views',
    },
    {
      id: 'videos',
      title: 'Videos',
      value: 156,
      change: 12.0,
      changeType: 'positive',
      icon: <VideoLibraryIcon />,
      color: '#ff9800',
      subtitle: 'generated',
    },
    {
      id: 'engagement',
      title: 'Engagement',
      value: '4.2%',
      change: 0.8,
      changeType: 'positive',
      icon: <LikesIcon />,
      color: '#e91e63',
      subtitle: 'avg rate',
    },
  ];

  const cards: MobileCard[] = [
    {
      id: '1',
      title: 'Tech Review Video',
      subtitle: 'Processing • 78% complete',
      status: 'active',
      progress: 78,
      avatar: '/tech-avatar.jpg',
      actions: [
        { icon: <PauseIcon />, label: 'Pause', action: () => {} },
        { icon: <MoreVertIcon />, label: 'More', action: () => {} },
      ],
    },
    {
      id: '2',
      title: 'Gaming Highlights',
      subtitle: 'Scheduled for 2:00 PM',
      status: 'pending',
      avatar: '/gaming-avatar.jpg',
      actions: [
        { icon: <EditIcon />, label: 'Edit', action: () => {} },
        { icon: <DeleteIcon />, label: 'Delete', action: () => {} },
      ],
    },
    {
      id: '3',
      title: 'Product Review',
      subtitle: 'Published • 1.2K views',
      status: 'completed',
      avatar: '/product-avatar.jpg',
      actions: [
        { icon: <ShareIcon />, label: 'Share', action: () => {} },
        { icon: <AnalyticsIcon />, label: 'Analytics', action: () => {} },
      ],
    },
  ];

  const navigationTabs: NavigationTab[] = [
    { label: 'Home', icon: <HomeIcon />, badge: 0 },
    { label: 'Videos', icon: <VideoLibraryIcon />, badge: 3 },
    { label: 'Analytics', icon: <AnalyticsIcon /> },
    { label: 'Profile', icon: <AccountIcon /> },
  ];

  // Hooks
  const swipeGestures = useSwipeGestures(
    () => setSelectedTab(prev => Math.min(prev + 1, 3)),
    () => setSelectedTab(prev => Math.max(prev - 1, 0))
  );

  const pullToRefresh = usePullToRefresh(async () => {
    setRefreshing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    setRefreshing(false);
    addNotification({
      type: 'success',
      message: 'Dashboard refreshed',
    });
  });

  // Mobile-specific components
  const MobileHeader = () => (
    <AppBar position="sticky" elevation={0}>
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          onClick={() => setDrawerOpen(true)}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          YTEmpire
        </Typography>
        
        <IconButton color="inherit" onClick={() => setShowNotifications(true)}>
          <Badge badgeContent={notifications.length} color="error">
            <NotificationsIcon />
          </Badge>
        </IconButton>
      </Toolbar>
      
      {/* Pull to refresh indicator */}
      {pullToRefresh.isPulling && (
        <Box
          sx={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
          }}
        >
          <CircularProgress size={24} />
        </Box>
      )}
    </AppBar>
  );

  const MobileMetricCard = ({ metric }: { metric: MobileMetric }) => {
    const getTrendColor = () => {
      switch (metric.changeType) {
        case 'positive': return '#4caf50';
        case 'negative': return '#f44336';
        default: return '#757575';
      }
    };

    const getTrendIcon = () => {
      switch (metric.changeType) {
        case 'positive': return <TrendingUpIcon fontSize="small" />;
        case 'negative': return <TrendingDownIcon fontSize="small" />;
        default: return null;
      }
    };

    return (
      <Card 
        sx={{ 
          height: '100%',
          background: `linear-gradient(135deg, ${metric.color}10, ${metric.color}05)`,
          border: `1px solid ${metric.color}20`,
        }}
      >
        <CardContent sx={{ pb: 2, '&:last-child': { pb: 2 } }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box sx={{ color: metric.color, opacity: 0.8 }}>
              {metric.icon}
            </Box>
            <Box textAlign="right">
              <Typography variant="h5" component="div" fontWeight="bold">
                {metric.value}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {metric.title}
              </Typography>
            </Box>
          </Box>
          
          <Box display="flex" alignItems="center" justifyContent="space-between" mt={1}>
            <Typography variant="body2" color="text.secondary">
              {metric.subtitle}
            </Typography>
            <Box display="flex" alignItems="center" sx={{ color: getTrendColor() }}>
              {getTrendIcon()}
              <Typography variant="body2" sx={{ ml: 0.5 }}>
                {metric.change > 0 ? '+' : ''}{metric.change}%
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const MobileVideoCard = ({ card }: { card: MobileCard }) => {
    const isExpanded = expandedCard === card.id;
    
    const getStatusColor = () => {
      switch (card.status) {
        case 'active': return '#2196f3';
        case 'pending': return '#ff9800';
        case 'completed': return '#4caf50';
        case 'failed': return '#f44336';
        default: return '#757575';
      }
    };

    return (
      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ pb: 1, '&:last-child': { pb: 1 } }}>
          <Box display="flex" alignItems="center">
            <Avatar
              src={card.avatar}
              sx={{ 
                mr: 2, 
                bgcolor: getStatusColor(),
                width: 48,
                height: 48,
              }}
            >
              {card.title[0]}
            </Avatar>
            
            <Box flexGrow={1} minWidth={0}>
              <Typography variant="subtitle1" noWrap>
                {card.title}
              </Typography>
              <Typography variant="body2" color="text.secondary" noWrap>
                {card.subtitle}
              </Typography>
              
              {card.progress !== undefined && (
                <Box mt={1}>
                  <LinearProgress 
                    variant="determinate" 
                    value={card.progress} 
                    sx={{ 
                      height: 4, 
                      borderRadius: 2,
                      backgroundColor: `${getStatusColor()}20`,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getStatusColor(),
                      },
                    }} 
                  />
                  <Typography variant="caption" color="text.secondary">
                    {card.progress}% complete
                  </Typography>
                </Box>
              )}
            </Box>
            
            <IconButton 
              size="small" 
              onClick={() => setExpandedCard(isExpanded ? null : card.id)}
            >
              {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
        </CardContent>
        
        <Collapse in={isExpanded}>
          <CardContent sx={{ pt: 0 }}>
            <Divider sx={{ mb: 2 }} />
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {card.actions?.map((action, index) => (
                <Button
                  key={index}
                  size="small"
                  startIcon={action.icon}
                  onClick={action.action}
                  variant="outlined"
                  sx={{ mb: 1 }}
                >
                  {action.label}
                </Button>
              ))}
            </Stack>
          </CardContent>
        </Collapse>
      </Card>
    );
  };

  const MobileBottomNav = () => (
    <BottomNavigation
      value={bottomNavValue}
      onChange={(event, newValue) => {
        setBottomNavValue(newValue);
        setSelectedTab(newValue);
      }}
      showLabels
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        borderTop: 1,
        borderColor: 'divider',
      }}
    >
      {navigationTabs.map((tab, index) => (
        <BottomNavigationAction
          key={index}
          label={tab.label}
          icon={
            <Badge badgeContent={tab.badge} color="error">
              {tab.icon}
            </Badge>
          }
          disabled={tab.disabled}
        />
      ))}
    </BottomNavigation>
  );

  const MobileSpeedDial = () => (
    <SpeedDial
      ariaLabel="Quick Actions"
      sx={{ 
        position: 'fixed', 
        bottom: isMobile ? 80 : 16, 
        right: 16,
        zIndex: 999,
      }}
      icon={<SpeedDialIcon />}
      open={speedDialOpen}
      onOpen={() => setSpeedDialOpen(true)}
      onClose={() => setSpeedDialOpen(false)}
    >
      <SpeedDialAction
        icon={<AddIcon />}
        tooltipTitle="New Video"
        onClick={() => setSpeedDialOpen(false)}
      />
      <SpeedDialAction
        icon={<UploadIcon />}
        tooltipTitle="Upload"
        onClick={() => setSpeedDialOpen(false)}
      />
      <SpeedDialAction
        icon={<AnalyticsIcon />}
        tooltipTitle="Analytics"
        onClick={() => setSpeedDialOpen(false)}
      />
    </SpeedDial>
  );

  const TabPanel = ({ children, value, index }: any) => (
    <Box
      role="tabpanel"
      hidden={value !== index}
      sx={{ 
        minHeight: 'calc(100vh - 128px)', // Account for app bar and bottom nav
        pb: isMobile ? 10 : 2, // Extra padding for bottom nav
      }}
      {...swipeGestures}
      {...pullToRefresh}
    >
      {value === index && (
        <Box p={2}>
          {children}
        </Box>
      )}
    </Box>
  );

  if (!isMobile) {
    // Desktop/tablet layout
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Desktop Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          This is the desktop version. Mobile responsive features are optimized for mobile devices.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <MobileHeader />
      
      {/* Main Content */}
      <Box>
        <TabPanel value={selectedTab} index={0}>
          {/* Dashboard Tab */}
          <Typography variant="h5" gutterBottom>
            Dashboard
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {metrics.map((metric) => (
              <Grid item xs={6} key={metric.id}>
                <MobileMetricCard metric={metric} />
              </Grid>
            ))}
          </Grid>
          
          <Typography variant="h6" gutterBottom>
            Recent Activity
          </Typography>
          {cards.map((card) => (
            <MobileVideoCard key={card.id} card={card} />
          ))}
        </TabPanel>
        
        <TabPanel value={selectedTab} index={1}>
          {/* Videos Tab */}
          <Typography variant="h5" gutterBottom>
            Videos
          </Typography>
          {cards.map((card) => (
            <MobileVideoCard key={card.id} card={card} />
          ))}
        </TabPanel>
        
        <TabPanel value={selectedTab} index={2}>
          {/* Analytics Tab */}
          <Typography variant="h5" gutterBottom>
            Analytics
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            Swipe left or right to navigate between tabs
          </Alert>
          <Grid container spacing={2}>
            {metrics.map((metric) => (
              <Grid item xs={12} sm={6} key={metric.id}>
                <MobileMetricCard metric={metric} />
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={selectedTab} index={3}>
          {/* Profile Tab */}
          <Typography variant="h5" gutterBottom>
            Profile
          </Typography>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Avatar sx={{ width: 56, height: 56, mr: 2 }}>
                  U
                </Avatar>
                <Box>
                  <Typography variant="h6">User Name</Typography>
                  <Typography variant="body2" color="text.secondary">
                    user@example.com
                  </Typography>
                </Box>
              </Box>
              
              <List>
                <ListItem>
                  <ListItemText primary="Settings" />
                  <SettingsIcon />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Help & Support" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Sign Out" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </TabPanel>
      </Box>
      
      {/* Bottom Navigation */}
      <MobileBottomNav />
      
      {/* Speed Dial */}
      <MobileSpeedDial />
      
      {/* Side Drawer */}
      <SwipeableDrawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onOpen={() => setDrawerOpen(true)}
        disableSwipeToOpen={false}
      >
        <Box sx={{ width: 250, pt: 2 }}>
          <Typography variant="h6" sx={{ px: 2, mb: 2 }}>
            Menu
          </Typography>
          <List>
            {navigationTabs.map((tab, index) => (
              <ListItem 
                key={index}
                onClick={() => {
                  setSelectedTab(index);
                  setBottomNavValue(index);
                  setDrawerOpen(false);
                }}
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'primary.main' }}>
                    {tab.icon}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText primary={tab.label} />
                {tab.badge && tab.badge > 0 && (
                  <Badge badgeContent={tab.badge} color="error" />
                )}
              </ListItem>
            ))}
          </List>
        </Box>
      </SwipeableDrawer>
      
      {/* Notifications Drawer */}
      <Drawer
        anchor="right"
        open={showNotifications}
        onClose={() => setShowNotifications(false)}
      >
        <Box sx={{ width: 300, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Notifications
          </Typography>
          {notifications.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No notifications
            </Typography>
          ) : (
            <List>
              {notifications.map((notification, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={notification.title}
                    secondary={notification.message}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </Drawer>
      
      {/* Loading Indicator */}
      {refreshing && (
        <Box
          sx={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 2000,
          }}
        >
          <CircularProgress />
        </Box>
      )}
    </Box>
  );
};