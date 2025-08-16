import React, { useState } from 'react';
import {  ThemeToggle  } from '../ThemeToggle/ThemeToggle';
import { 
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Menu,
  MenuItem,
  Box,
  Avatar,
  Divider,
  ListItemIcon,
  ListItemText,
  TextField,
  Chip,
  LinearProgress
 } from '@mui/material';
import { 
  Notifications,
  Search,
  Settings,
  Person,
  ExitToApp,
  DarkMode,
  LightMode,
  Help,
  Feedback,
  CloudQueue,
  CheckCircle,
  Error,
  Warning,
  Info,
  TrendingUp,
  AttachMoney
 } from '@mui/icons-material';
import {  useNavigate  } from 'react-router-dom';
import {  useAuthStore  } from '../../stores/authStore';

interface HeaderProps {
  
darkMode?: boolean;
onToggleDarkMode?: () => void;


}

interface Notification {
  
id: string;
type: 'success' | 'error' | 'warning' | 'info';

title: string;
message: string;

timestamp: Date;
read: boolean;

}

export const Header: React.FC<HeaderProps> = ({ darkMode = false, onToggleDarkMode }) => {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  
  const [anchorElUser, setAnchorElUser] = useState<null | HTMLElement>(null);
  const [anchorElNotif, setAnchorElNotif] = useState<null | HTMLElement>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Mock notifications - in production, these would come from the backend
  const [notifications, setNotifications] = useState<Notification[]>([ { id: '1',
      type: 'success',
      title: 'Video Published',
      message: 'Your video "10 AI Tools" was successfully published',
      timestamp: new Date(Date.now() - 1000 * 60 * 5),
      read: false },
    { id: '2',
      type: 'info',
      title: 'Processing Complete',
      message: 'Video generation completed for 3 videos',
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      read: false },
    { id: '3',
      type: 'warning',
      title: 'Daily Limit Warning',
      message: 'You have 2 videos remaining for today',
      timestamp: new Date(Date.now() - 1000 * 60 * 60),
      read: true } ]);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleOpenUserMenu = (_: React.MouseEvent<HTMLElement>) => {
    setAnchorElUser(event.currentTarget)
};
  const handleCloseUserMenu = () => {
    setAnchorElUser(null)
};
  const handleOpenNotifications = (_: React.MouseEvent<HTMLElement>) => {
    setAnchorElNotif(event.currentTarget)
};
  const handleCloseNotifications = () => {
    setAnchorElNotif(null)
};
  const handleMarkAllRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })))
};
  const handleLogout = () => {
    handleCloseUserMenu();
    logout();
    navigate('/auth/login')
};
  const handleSearch = (_: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`)}
  };
  const getNotificationIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle color="success" fontSize="small" />;
      case 'error':
        return <Error color="error" fontSize="small" />;
      case 'warning':
        return <Warning color="warning" fontSize="small" />;
      case 'info':
        return <Info color="info" fontSize="small" />}};
  const formatTimestamp = (date: Date) => {
    const diff = Date.now() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago
  };
  return (
    <>
      <AppBar
        position="fixed"
        sx={ {
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backdropFilter: 'blur(8px)',
          backgroundColor: (theme) =>
            theme.palette.mode === 'dark' 
              ? 'rgba(18, 18, 18, 0.9)' 
              : 'rgba(255, 255, 255, 0.9)' }}
        elevation={0}
      >
        <Toolbar>
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
            <Box
              component="form"
              onSubmit={handleSearch}
              sx={ {
                display: 'flex',
                alignItems: 'center',
                backgroundColor: 'action.hover',
                borderRadius: 2,
                px: 2,
                py: 0.5,
                minWidth: 300 }}
            >
              <Search sx={{ color: 'text.secondary', mr: 1 }} />
              <TextField
                placeholder="Search videos, channels, analytics..."
                variant="standard"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={ {
                  disableUnderline: true }}
                sx={{ flex: 1 }}
              />
            </Box>
            
            {/* Quick Stats */}
            <Box sx={{ display: 'flex', gap: 2, ml: 4 }}>
              <Tooltip title="Today's Performance">
                <Chip
                  icon={<TrendingUp />}
                  label="+12.5%"
                  color="success"
                  size="small"
                />
              </Tooltip>
              <Tooltip title="Revenue Today">
                <Chip
                  icon={<AttachMoney />}
                  label="$127.50"
                  color="primary"
                  size="small"
                />
              </Tooltip>
              <Tooltip title="Processing Queue">
                <Chip
                  icon={<CloudQueue />}
                  label="3 Active"
                  color="warning"
                  size="small"
                />
              </Tooltip>
            </Box>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Dark Mode Toggle */}
            {onToggleDarkMode && (
              <Tooltip title="Toggle Dark Mode">
                <IconButton onClick={onToggleDarkMode} color="inherit">
                  {darkMode ? <LightMode /> </>: <DarkMode />}
                </IconButton>
              </Tooltip>
            )}
            {/* Help */}
            <Tooltip title="Help & Documentation">
              <IconButton
                onClick={() => navigate('/help'}
                color="inherit"
              >
                <Help />
              </IconButton>
            </Tooltip>

            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton
                onClick={handleOpenNotifications}
                color="inherit"
              >
                <Badge badgeContent={unreadCount} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* User Menu */}
            <Tooltip title="Account">
              <IconButton onClick={handleOpenUserMenu} sx={{ p: 0, ml: 2 }}>
                <Avatar sx={{ width: 36, height: 36 }}>
                  {user?.username[0].toUpperCase()}
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
        
        { isProcessing && (
          <LinearProgress
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0 }}
          />
        )}
      </AppBar>

      {/* User Menu */}
      <Menu
        anchorEl={anchorElUser}
        open={Boolean(anchorElUser)}
        onClose={handleCloseUserMenu}
        PaperProps={{
          sx: { width: 280, mt: 1.5 }
        }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle1" fontWeight="bold">
            {user?.full_name || user?.username}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {user?.email}
          </Typography>
          <Chip
            label={user?.subscription_tier}
            size="small"
            color={user?.subscription_tier === 'enterprise' ? 'success' : 'primary'}
            sx={{ mt: 1 }}
          />
        </Box>
        
        <Divider />
        
        <MenuItem onClick={() => { handleCloseUserMenu(), navigate('/settings/profile')}}>
          <ListItemIcon>
            <Person fontSize="small" />
          </ListItemIcon>
          <ListItemText>My Profile</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => { handleCloseUserMenu(), navigate('/settings')}}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => { handleCloseUserMenu(), navigate('/feedback')}}>
          <ListItemIcon>
            <Feedback fontSize="small" />
          </ListItemIcon>
          <ListItemText>Send Feedback</ListItemText>
        </MenuItem>
        
        <Divider />
        
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <ExitToApp fontSize="small" />
          </ListItemIcon>
          <ListItemText>Logout</ListItemText>
        </MenuItem>
      </Menu>

      {/* Notifications Menu */}
      <Menu
        anchorEl={anchorElNotif}
        open={Boolean(anchorElNotif)}
        onClose={handleCloseNotifications}
        PaperProps={{
          sx: { width: 360, maxHeight: 480, mt: 1.5 }}}
      >
        <Box sx={{ px: 2, py: 1.5, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Notifications</Typography>
          {unreadCount > 0 && (
            <Typography
              variant="body2"
              color="primary"
              sx={{ cursor: 'pointer' }}
              onClick={handleMarkAllRead}
            >
              Mark all read
            </Typography>
          )}
        </Box>
        
        <Divider />
        
        {notifications.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No notifications
            </Typography>
          </Box>
        ) : (_notifications.map((notification => (
            <MenuItem
              key={notification.id}
              onClick={() => {
                setNotifications(prev =>
                  prev.map(n => n.id === notification.id ? { ...n, read: true } : n)
                )}}
              sx={ {
                backgroundColor: notification.read ? 'transparent' : 'action.hover',
                '&:hover': {
                  backgroundColor: 'action.selected' }}}
            >
              <ListItemIcon>
                {getNotificationIcon(notification.type)}
              </ListItemIcon>
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2" fontWeight={notification.read ? 'normal' : 'bold'}>
                  {notification.title}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {notification.message}
                </Typography>
                <Typography variant="caption" display="block" color="text.disabled" sx={{ mt: 0.5 }}>
                  {formatTimestamp(notification.timestamp)}
                </Typography>
              </Box>
            </MenuItem>
          ))
        )}
        <Divider />
        
        <MenuItem
          onClick={() => {
            handleCloseNotifications(),
            navigate('/notifications')}}
          sx={{ justifyContent: 'center' }}
        >
          <Typography variant="body2" color="primary">
            View All Notifications
          </Typography>
        </MenuItem>
      </Menu>
    </>
  )};
