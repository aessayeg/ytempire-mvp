/**
 * Enhanced Theme Context
 * Provides comprehensive theming support with dark mode, custom colors, and animations
 */

import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { 
  createTheme,
  Theme,
  PaletteMode,
  alpha
 } from '@mui/material';
import { 
  CssBaseline
 } from '@mui/material';

// Theme configuration types
interface ThemeConfig {
  mode: PaletteMode,
  primaryColor: string,

  secondaryColor: string,
  fontFamily: string,

  borderRadius: number,
  animationsEnabled: boolean,

  reducedMotion: boolean,
  highContrast: boolean}

interface ThemeContextType {
  theme: Theme,
  themeConfig: ThemeConfig,

  isDarkMode: boolean,
  toggleTheme: () => void,

  setThemeMode: (mode: PaletteMode | 'system') => void,
  setPrimaryColor: (color: string) => void,
  setSecondaryColor: (color: string) => void,
  setAnimationsEnabled: (enabled: boolean) => void,
  setReducedMotion: (reduced: boolean) => void,
  setHighContrast: (high: boolean) => void,
  resetTheme: () => void}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Default theme configuration
const defaultThemeConfig: ThemeConfig = {,
  mode: 'light',
  primaryColor: '#1976 d2',
  secondaryColor: '#dc004 e',
  fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  borderRadius: 8,
  animationsEnabled: true,
  reducedMotion: false,
  highContrast: false
};

// Color palette for dark mode
const darkPalette = {
  primary: {,
  main: '#90 caf9',
    light: '#e3 f2 fd',
    dark: '#42 a5 f5',
    contrastText: '#000',

  },
  secondary: {,
  main: '#f48 fb1',
    light: '#ffc1 e3',
    dark: '#bf5 f82',
    contrastText: '#000',

  },
  background: {,
  default: '#121212',
    paper: '#1 e1 e1 e',

  },
  text: {,
  primary: '#ffffff',
    secondary: 'rgba(255, 255, 255, 0.7)',
    disabled: 'rgba(255, 255, 255, 0.5)'
  },
  divider: 'rgba(255, 255, 255, 0.12)',
  action: {,
  active: '#fff',
    hover: 'rgba(255, 255, 255, 0.08)',
    selected: 'rgba(255, 255, 255, 0.16)',
    disabled: 'rgba(255, 255, 255, 0.3)',
    disabledBackground: 'rgba(255, 255, 255, 0.12)'
  }
};

// Color palette for light mode
const lightPalette = {
  primary: {,
  main: '#1976 d2',
    light: '#42 a5 f5',
    dark: '#1565 c0',
    contrastText: '#fff',

  },
  secondary: {,
  main: '#dc004 e',
    light: '#ff5983',
    dark: '#9 a0036',
    contrastText: '#fff',

  },
  background: {,
  default: '#fafafa',
    paper: '#ffffff',

  },
  text: {,
  primary: 'rgba(0, 0, 0, 0.87)',
    secondary: 'rgba(0, 0, 0, 0.6)',
    disabled: 'rgba(0, 0, 0, 0.38)'
  },
  divider: 'rgba(0, 0, 0, 0.12)',
  action: {,
  active: 'rgba(0, 0, 0, 0.54)',
    hover: 'rgba(0, 0, 0, 0.04)',
    selected: 'rgba(0, 0, 0, 0.08)',
    disabled: 'rgba(0, 0, 0, 0.26)',
    disabledBackground: 'rgba(0, 0, 0, 0.12)'
  }
};

export const EnhancedThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    
  const [themeConfig, setThemeConfig] = useState<ThemeConfig>(() => {
    // Load saved theme from localStorage
    const savedTheme = localStorage.getItem('ytempire-theme');
    if (savedTheme) {
      try {
        return { ...defaultThemeConfig, ...JSON.parse(savedTheme) 
  } catch {
        return defaultThemeConfig;
      }
    }
    return defaultThemeConfig});

  const [systemPrefersDark, setSystemPrefersDark] = useState(
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (_: MediaQueryListEvent) => {
      setSystemPrefersDark(e.matches)};
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange)}, []);

  // Listen for reduced motion preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handleChange = (_: MediaQueryListEvent) => {
      if (e.matches) {
        setThemeConfig(prev => ({ ...prev, reducedMotion: true }))}
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange)}, []);

  // Save theme to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('ytempire-theme', JSON.stringify(themeConfig))}, [themeConfig]);

  // Create MUI theme
  const theme = useMemo(() => {
    const isDark = themeConfig.mode === 'dark' || 
                   (themeConfig.mode === 'system' && systemPrefersDark);
    
    return createTheme({
      palette: {,
  mode: isDark ? 'dark' : 'light',
        ...(isDark ? darkPalette : lightPalette),
        primary: {,
  main: themeConfig.primaryColor,
          ...(isDark ? darkPalette.primary : lightPalette.primary)
        },
        secondary: {,
  main: themeConfig.secondaryColor,
          ...(isDark ? darkPalette.secondary : lightPalette.secondary)
        },
        // Enhanced colors for dark mode
        error: {,
  main: isDark ? '#f44336' : '#d32 f2 f',
          light: isDark ? '#ef5350' : '#ef5350',
          dark: isDark ? '#c62828' : '#c62828',

        },
        warning: {,
  main: isDark ? '#ffa726' : '#ed6 c02',
          light: isDark ? '#ffb74 d' : '#ff9800',
          dark: isDark ? '#f57 c00' : '#e65100',

        },
        info: {,
  main: isDark ? '#29 b6 f6' : '#0288 d1',
          light: isDark ? '#4 fc3 f7' : '#03 a9 f4',
          dark: isDark ? '#0288 d1' : '#01579 b',

        },
        success: {,
  main: isDark ? '#66 bb6 a' : '#2 e7 d32',
          light: isDark ? '#81 c784' : '#4 caf50',
          dark: isDark ? '#388 e3 c' : '#1 b5 e20',

        }
      },
      typography: {,
  fontFamily: themeConfig.fontFamily,
        h1: {,
  fontWeight: 700,
          fontSize: '2.5 rem',
          lineHeight: 1.2,

        },
        h2: {,
  fontWeight: 600,
          fontSize: '2 rem',
          lineHeight: 1.3,

        },
        h3: {,
  fontWeight: 600,
          fontSize: '1.75 rem',
          lineHeight: 1.4,

        },
        h4: {,
  fontWeight: 500,
          fontSize: '1.5 rem',
          lineHeight: 1.4,

        },
        h5: {,
  fontWeight: 500,
          fontSize: '1.25 rem',
          lineHeight: 1.5,

        },
        h6: {,
  fontWeight: 500,
          fontSize: '1 rem',
          lineHeight: 1.6,

        }
      },
      shape: {,
  borderRadius: themeConfig.borderRadius,

      },
      transitions: {,
  duration: {,

          shortest: themeConfig.reducedMotion ? 0 : 150,
          shorter: themeConfig.reducedMotion ? 0 : 200,
          short: themeConfig.reducedMotion ? 0 : 250,
          standard: themeConfig.reducedMotion ? 0 : 300,
          complex: themeConfig.reducedMotion ? 0 : 375,
          enteringScreen: themeConfig.reducedMotion ? 0 : 225,
          leavingScreen: themeConfig.reducedMotion ? 0 : 195
}
      },
      components: {
        // Global component overrides for dark mode
        MuiCssBaseline: {,
  styleOverrides: {,

            body: {,
  scrollbarColor: isDark ? '#6 b6 b6 b #2 b2 b2 b' : '#959595 #f1 f1 f1',
              '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
                width: 8,
                height: 8
},
              '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
                borderRadius: 8,
                backgroundColor: isDark ? '#6 b6 b6 b' : '#959595',
                minHeight: 24,
                border: `2px solid ${isDark ? '#2 b2 b2 b' : '#f1 f1 f1'}
              },
              '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
                borderRadius: 8,
                backgroundColor: isDark ? '#2 b2 b2 b' : '#f1 f1 f1',

              }
            }
          }
        },
        MuiPaper: {,
  styleOverrides: {,

            root: {,
  backgroundImage: isDark 
                ? 'linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.05))'
                : 'none',
              transition: themeConfig.animationsEnabled 
                ? 'background-color 0.3s ease, box-shadow 0.3s ease'
                : 'none'
            }
          }
        },
        MuiCard: {,
  styleOverrides: {,

            root: {,
  boxShadow: isDark 
                ? '0 4px 6px rgba(0, 0, 0, 0.3)'
                : '0 2px 4px rgba(0, 0, 0, 0.1)',
              '&:hover': themeConfig.animationsEnabled ? {
                boxShadow: isDark 
                  ? '0 8px 12px rgba(0, 0, 0, 0.4)'
                  : '0 4px 8px rgba(0, 0, 0, 0.15)',
                transform: 'translateY(-2px)',

              } : {}
            }
          }
        },
        MuiButton: {,
  styleOverrides: {,

            root: {,
  textTransform: 'none',
              fontWeight: 500,
              transition: themeConfig.animationsEnabled 
                ? 'all 0.3s ease'
                : 'none',
              '&:hover': {
                transform: themeConfig.animationsEnabled ? 'translateY(-1px)' : 'none',

              }
            },
            contained: {,
  boxShadow: isDark 
                ? '0 2px 4px rgba(0, 0, 0, 0.4)'
                : '0 2px 4px rgba(0, 0, 0, 0.2)',
              '&:hover': {
                boxShadow: isDark 
                  ? '0 4px 8px rgba(0, 0, 0, 0.5)'
                  : '0 4px 8px rgba(0, 0, 0, 0.25)'
              }
            }
          }
        },
        MuiIconButton: {,
  styleOverrides: {,

            root: {,
  transition: themeConfig.animationsEnabled 
                ? 'all 0.3s ease'
                : 'none',
              '&:hover': {
                transform: themeConfig.animationsEnabled ? 'scale(1.1)' : 'none',
                backgroundColor: isDark 
                  ? alpha('#fff', 0.08)
                  : alpha('#000', 0.04)}
            }
          }
        },
        MuiTextField: {,
  styleOverrides: {,

            root: {
              '& .MuiOutlinedInput-root': {
                transition: themeConfig.animationsEnabled 
                  ? 'all 0.3s ease'
                  : 'none',
                '&:hover': {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: isDark 
                      ? alpha('#fff', 0.5)
                      : alpha('#000', 0.3)}
                }
              }
            }
          }
        },
        MuiChip: {,
  styleOverrides: {,

            root: {,
  transition: themeConfig.animationsEnabled 
                ? 'all 0.3s ease'
                : 'none',
              '&:hover': {
                transform: themeConfig.animationsEnabled ? 'scale(1.05)' : 'none',

              }
            }
          }
        },
        MuiTooltip: {,
  styleOverrides: {,

            tooltip: {,
  backgroundColor: isDark 
                ? alpha('#fff', 0.9)
                : alpha('#000', 0.87),
              color: isDark ? '#000' : '#fff',
              fontSize: '0.875 rem',
              fontWeight: 400
}
          }
        },
        MuiAlert: {,
  styleOverrides: {,

            root: {,
  borderRadius: themeConfig.borderRadius,
              boxShadow: isDark 
                ? '0 2px 8px rgba(0, 0, 0, 0.3)'
                : '0 2px 8px rgba(0, 0, 0, 0.15)'
            }
          }
        },
        MuiDrawer: {,
  styleOverrides: {,

            paper: {,
  backgroundColor: isDark ? '#1 e1 e1 e' : '#fff',
              backgroundImage: isDark 
                ? 'linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.05))'
                : 'none'
            }
          }
        },
        MuiAppBar: {,
  styleOverrides: {,

            root: {,
  backgroundColor: isDark ? '#1 e1 e1 e' : '#fff',
              backgroundImage: isDark 
                ? 'linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.05))'
                : 'none',
              color: isDark ? '#fff' : 'rgba(0, 0, 0, 0.87)'
            }
          }
        },
        MuiTable: {,
  styleOverrides: {,

            root: {,
  backgroundColor: isDark ? '#1 e1 e1 e' : '#fff',

            }
          }
        },
        MuiTableCell: {,
  styleOverrides: {,

            root: {`,
  borderBottom: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)'}
            }
          }
        },
        MuiDivider: {,
  styleOverrides: {,

            root: {,
  backgroundColor: isDark 
                ? 'rgba(255, 255, 255, 0.12)'
                : 'rgba(0, 0, 0, 0.12)'
            }
          }
        }
      }
    })}, [themeConfig, systemPrefersDark]);

  // Context value methods
  const toggleTheme = useCallback(() => {
    setThemeConfig(prev => ({
      ...prev,
      mode: prev.mode === 'light' ? 'dark' : 'light',

    }))}, []);

  const setThemeMode = useCallback(_(mode: PaletteMode | 'system') => {
    setThemeConfig(prev => ({ ...prev, mode: mode as PaletteMode }))}, []);

  const setPrimaryColor = useCallback(_(color: string) => {
    setThemeConfig(prev => ({ ...prev, primaryColor: color }))}, []);

  const setSecondaryColor = useCallback(_(color: string) => {
    setThemeConfig(prev => ({ ...prev, secondaryColor: color }))}, []);

  const setAnimationsEnabled = useCallback(_(enabled: boolean) => {
    setThemeConfig(prev => ({ ...prev, animationsEnabled: enabled }))}, []);

  const setReducedMotion = useCallback(_(reduced: boolean) => {
    setThemeConfig(prev => ({ ...prev, reducedMotion: reduced }))}, []);

  const setHighContrast = useCallback(_(high: boolean) => {
    setThemeConfig(prev => ({ ...prev, highContrast: high }))}, []);

  const resetTheme = useCallback(() => {
    setThemeConfig(defaultThemeConfig);
    localStorage.removeItem('ytempire-theme')}, []); // eslint-disable-line react-hooks/exhaustive-deps

  const isDarkMode = themeConfig.mode === 'dark' || 
                     (themeConfig.mode === 'system' && systemPrefersDark);

  const contextValue: ThemeContextType = {
    theme,
    themeConfig,
    isDarkMode,
    toggleTheme,
    setThemeMode,
    setPrimaryColor,
    setSecondaryColor,
    setAnimationsEnabled,
    setReducedMotion,
    setHighContrast,
    resetTheme
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  )};

export const useEnhancedTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useEnhancedTheme must be used within EnhancedThemeProvider')}
  return context;
};

export default EnhancedThemeProvider;