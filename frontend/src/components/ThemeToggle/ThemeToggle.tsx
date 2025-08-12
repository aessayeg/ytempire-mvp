/**
 * Theme Toggle Component
 * Provides UI controls for switching between light, dark, and system themes
 */

import React, { useState } from 'react';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Switch,
  FormControlLabel,
  Box,
  Typography,
  Divider,
  alpha
} from '@mui/material';
import {
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  BrightnessAuto as AutoModeIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useTheme } from '../../contexts/ThemeContext';

export const ThemeToggle: React.FC = () => {
  const { isDarkMode, toggleTheme, setTheme, themeMode } = useTheme();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleThemeChange = (mode: 'light' | 'dark' | 'system') => {
    setTheme(mode);
    handleClose();
  };

  const getIcon = () => {
    if (themeMode === 'system') {
      return <AutoModeIcon />;
    }
    return isDarkMode ? <DarkModeIcon /> : <LightModeIcon />;
  };

  return (
    <>
      <Tooltip title="Theme settings">
        <IconButton
          onClick={handleClick}
          size="small"
          sx={{
            ml: 1,
            backgroundColor: (theme) => 
              alpha(theme.palette.primary.main, 0.1),
            '&:hover': {
              backgroundColor: (theme) => 
                alpha(theme.palette.primary.main, 0.2),
            }
          }}
        >
          {getIcon()}
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        PaperProps={{
          elevation: 3,
          sx: {
            minWidth: 220,
            mt: 1.5,
            borderRadius: 2,
            '& .MuiMenuItem-root': {
              borderRadius: 1,
              mx: 0.5,
              my: 0.25
            }
          }
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Theme Settings
          </Typography>
        </Box>
        <Divider />
        <MenuItem 
          selected={themeMode === 'light'}
          onClick={() => handleThemeChange('light')}
        >
          <ListItemIcon>
            <LightModeIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Light</ListItemText>
        </MenuItem>
        <MenuItem 
          selected={themeMode === 'dark'}
          onClick={() => handleThemeChange('dark')}
        >
          <ListItemIcon>
            <DarkModeIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Dark</ListItemText>
        </MenuItem>
        <MenuItem 
          selected={themeMode === 'system'}
          onClick={() => handleThemeChange('system')}
        >
          <ListItemIcon>
            <AutoModeIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>System</ListItemText>
        </MenuItem>
        <Divider sx={{ my: 1 }} />
        <Box sx={{ px: 2, py: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={isDarkMode}
                onChange={toggleTheme}
                size="small"
              />
            }
            label={
              <Typography variant="body2">
                Quick toggle
              </Typography>
            }
          />
        </Box>
      </Menu>
    </>
  );
};

// Compact version for mobile
export const ThemeToggleCompact: React.FC = () => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <Tooltip title={isDarkMode ? 'Light mode' : 'Dark mode'}>
      <IconButton onClick={toggleTheme} size="small">
        {isDarkMode ? <LightModeIcon /> : <DarkModeIcon />}
      </IconButton>
    </Tooltip>
  );
};

// Floating theme toggle for demo/beta
export const FloatingThemeToggle: React.FC = () => {
  const { isDarkMode, toggleTheme } = useTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 1300
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1
        }}
      >
        {isExpanded && (
          <Box
            sx={{
              backgroundColor: 'background.paper',
              borderRadius: 2,
              p: 1.5,
              boxShadow: 3,
              mb: 1
            }}
          >
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Theme
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={isDarkMode}
                  onChange={toggleTheme}
                  size="small"
                />
              }
              label={
                <Typography variant="body2">
                  {isDarkMode ? 'Dark' : 'Light'}
                </Typography>
              }
            />
          </Box>
        )}
        <IconButton
          onClick={() => setIsExpanded(!isExpanded)}
          sx={{
            backgroundColor: 'primary.main',
            color: 'primary.contrastText',
            '&:hover': {
              backgroundColor: 'primary.dark'
            },
            width: 56,
            height: 56
          }}
        >
          {isExpanded ? <SettingsIcon /> : (isDarkMode ? <DarkModeIcon /> : <LightModeIcon />)}
        </IconButton>
      </Box>
    </Box>
  );
};