/**
 * Theme Context
 * Owner: Dashboard Specialist
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { ThemeProvider as MuiThemeProvider, createTheme, Theme } from '@mui/material/styles'
import { CssBaseline } from '@mui/material'

// Theme type
type ThemeMode = 'light' | 'dark' | 'system'

interface ThemeContextType {
  themeMode: ThemeMode
  isDarkMode: boolean
  toggleTheme: () => void
  setThemeMode: (mode: ThemeMode) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

// Create custom themes
const createCustomTheme = (mode: 'light' | 'dark'): Theme => {
  const isDark = mode === 'dark'
  
  return createTheme({
    palette: {
      mode,
      primary: {
        main: '#FF4444', // YTEmpire red
        light: '#FF7777',
        dark: '#CC3333',
        contrastText: '#FFFFFF',
      },
      secondary: {
        main: '#6C5CE7', // Purple accent
        light: '#A29BFE',
        dark: '#5A4FCF',
        contrastText: '#FFFFFF',
      },
      background: {
        default: isDark ? '#0D1117' : '#F8F9FA',
        paper: isDark ? '#161B22' : '#FFFFFF',
      },
      text: {
        primary: isDark ? '#F0F6FC' : '#1F2937',
        secondary: isDark ? '#9CA3AF' : '#6B7280',
      },
      divider: isDark ? '#30363D' : '#E5E7EB',
      success: {
        main: '#10B981',
        light: '#34D399',
        dark: '#059669',
      },
      error: {
        main: '#EF4444',
        light: '#F87171',
        dark: '#DC2626',
      },
      warning: {
        main: '#F59E0B',
        light: '#FBBF24',
        dark: '#D97706',
      },
      info: {
        main: '#3B82F6',
        light: '#60A5FA',
        dark: '#2563EB',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
        fontSize: '2.5rem',
        lineHeight: 1.2,
      },
      h2: {
        fontWeight: 600,
        fontSize: '2rem',
        lineHeight: 1.3,
      },
      h3: {
        fontWeight: 600,
        fontSize: '1.75rem',
        lineHeight: 1.3,
      },
      h4: {
        fontWeight: 600,
        fontSize: '1.5rem',
        lineHeight: 1.4,
      },
      h5: {
        fontWeight: 600,
        fontSize: '1.25rem',
        lineHeight: 1.4,
      },
      h6: {
        fontWeight: 600,
        fontSize: '1.125rem',
        lineHeight: 1.4,
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.5,
      },
      button: {
        textTransform: 'none',
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            fontWeight: 500,
            fontSize: '0.875rem',
            padding: '8px 16px',
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            },
          },
          contained: {
            '&:hover': {
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: isDark 
              ? '0 1px 3px rgba(0,0,0,0.3)' 
              : '0 1px 3px rgba(0,0,0,0.1)',
            border: isDark ? '1px solid #30363D' : '1px solid #E5E7EB',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            border: isDark ? '1px solid #30363D' : '1px solid #E5E7EB',
          },
          outlined: {
            borderColor: isDark ? '#30363D' : '#E5E7EB',
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
              '& fieldset': {
                borderColor: isDark ? '#30363D' : '#E5E7EB',
              },
              '&:hover fieldset': {
                borderColor: isDark ? '#6C5CE7' : '#FF4444',
              },
            },
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRight: isDark ? '1px solid #30363D' : '1px solid #E5E7EB',
            backgroundColor: isDark ? '#161B22' : '#FFFFFF',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDark ? '#161B22' : '#FFFFFF',
            borderBottom: isDark ? '1px solid #30363D' : '1px solid #E5E7EB',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
          },
        },
      },
      MuiAlert: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
    },
  })
}

interface ThemeProviderProps {
  children: ReactNode
}

export const ThemeContextProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeMode, setThemeModeState] = useState<ThemeMode>('system')
  const [systemPrefersDark, setSystemPrefersDark] = useState(false)

  // Check system preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setSystemPrefersDark(mediaQuery.matches)

    const handler = (e: MediaQueryListEvent) => {
      setSystemPrefersDark(e.matches)
    }

    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [])

  // Load theme preference from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('ytempire-theme-mode') as ThemeMode
    if (savedTheme && ['light', 'dark', 'system'].includes(savedTheme)) {
      setThemeModeState(savedTheme)
    }
  }, [])

  // Save theme preference to localStorage
  const setThemeMode = (mode: ThemeMode) => {
    setThemeModeState(mode)
    localStorage.setItem('ytempire-theme-mode', mode)
  }

  // Determine actual theme
  const getActualTheme = (): 'light' | 'dark' => {
    if (themeMode === 'system') {
      return systemPrefersDark ? 'dark' : 'light'
    }
    return themeMode as 'light' | 'dark'
  }

  const actualTheme = getActualTheme()
  const isDarkMode = actualTheme === 'dark'

  // Toggle between light and dark (ignores system)
  const toggleTheme = () => {
    if (themeMode === 'system') {
      setThemeMode(systemPrefersDark ? 'light' : 'dark')
    } else {
      setThemeMode(isDarkMode ? 'light' : 'dark')
    }
  }

  const theme = createCustomTheme(actualTheme)

  const value: ThemeContextType = {
    themeMode,
    isDarkMode,
    toggleTheme,
    setThemeMode,
  }

  return (
    <ThemeContext.Provider value={value}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  )
}

export const useThemeContext = (): ThemeContextType => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useThemeContext must be used within a ThemeContextProvider')
  }
  return context
}

export default ThemeContextProvider