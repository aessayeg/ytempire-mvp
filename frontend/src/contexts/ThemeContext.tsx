/**
 * Theme Context Provider
 * Manages theme state and provides theme switching functionality
 */

import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  lightTheme, 
  darkTheme, 
  getStoredTheme, 
  setStoredTheme, 
  getSystemTheme,
  enableSmoothTransition 
} from '../theme/darkMode';

interface ThemeContextType {
  isDarkMode: boolean;
  toggleTheme: () => void;
  setTheme: (mode: 'light' | 'dark' | 'system') => void;
  themeMode: 'light' | 'dark' | 'system';
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeMode, setThemeMode] = useState<'light' | 'dark' | 'system'>(() => {
    const stored = getStoredTheme();
    return stored || 'system';
  });

  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(getSystemTheme());

  const isDarkMode = useMemo(() => {
    if (themeMode === 'system') {
      return systemTheme === 'dark';
    }
    return themeMode === 'dark';
  }, [themeMode, systemTheme]);

  const theme = useMemo(() => {
    return isDarkMode ? darkTheme : lightTheme;
  }, [isDarkMode]);

  const toggleTheme = useCallback(() => {
    enableSmoothTransition();
    const newMode = isDarkMode ? 'light' : 'dark';
    setThemeMode(newMode);
    setStoredTheme(newMode);
  }, [isDarkMode]);

  const setTheme = useCallback((mode: 'light' | 'dark' | 'system') => {
    enableSmoothTransition();
    setThemeMode(mode);
    if (mode !== 'system') {
      setStoredTheme(mode);
    } else {
      localStorage.removeItem('ytempire_theme_mode');
    }
  }, []);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };

    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    } 
    // Legacy browsers
    else if (mediaQuery.addListener) {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, []);

  // Apply theme class to body for CSS variables
  useEffect(() => {
    document.body.classList.toggle('dark-theme', isDarkMode);
    document.body.classList.toggle('light-theme', !isDarkMode);
    
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', isDarkMode ? '#1a1d21' : '#ffffff');
    }
  }, [isDarkMode]);

  const contextValue = useMemo(() => ({
    isDarkMode,
    toggleTheme,
    setTheme,
    themeMode
  }), [isDarkMode, toggleTheme, setTheme, themeMode]);

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};