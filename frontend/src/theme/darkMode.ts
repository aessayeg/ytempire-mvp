/**
 * Dark Mode Theme Configuration
 * Comprehensive theming system with automatic switching and persistence
 */

import { createTheme } from '@mui/material/styles';
import type { ThemeOptions } from '@mui/material/styles';

// Color palette for dark mode
const darkPalette = {
  primary: {
    main: '#667eea',
    light: '#8b9cff',
    dark: '#4c63b6',
    contrastText: '#ffffff'
  },
  secondary: {
    main: '#ed64a6',
    light: '#ff8fc7',
    dark: '#b83280',
    contrastText: '#ffffff'
  },
  error: {
    main: '#f56565',
    light: '#ff8787',
    dark: '#c53030'
  },
  warning: {
    main: '#ed8936',
    light: '#ffb074',
    dark: '#c05621'
  },
  info: {
    main: '#4299e1',
    light: '#63b3ed',
    dark: '#2c5282'
  },
  success: {
    main: '#48bb78',
    light: '#68d391',
    dark: '#2f855a'
  },
  background: {
    default: '#0f1114',
    paper: '#1a1d21',
    elevated: '#22262b'
  },
  text: {
    primary: '#e2e8f0',
    secondary: '#a0aec0',
    disabled: '#718096'
  },
  divider: 'rgba(255, 255, 255, 0.08)',
  action: {
    active: '#e2e8f0',
    hover: 'rgba(255, 255, 255, 0.08)',
    selected: 'rgba(102, 126, 234, 0.16)',
    disabled: 'rgba(255, 255, 255, 0.3)',
    disabledBackground: 'rgba(255, 255, 255, 0.12)'
  }
};

// Light mode palette
const lightPalette = {
  primary: {
    main: '#667eea',
    light: '#8b9cff',
    dark: '#4c63b6',
    contrastText: '#ffffff'
  },
  secondary: {
    main: '#ed64a6',
    light: '#ff8fc7',
    dark: '#b83280',
    contrastText: '#ffffff'
  },
  error: {
    main: '#f56565',
    light: '#fc8181',
    dark: '#e53e3e'
  },
  warning: {
    main: '#ed8936',
    light: '#f6ad55',
    dark: '#dd6b20'
  },
  info: {
    main: '#4299e1',
    light: '#63b3ed',
    dark: '#3182ce'
  },
  success: {
    main: '#48bb78',
    light: '#68d391',
    dark: '#38a169'
  },
  background: {
    default: '#f7fafc',
    paper: '#ffffff',
    elevated: '#ffffff'
  },
  text: {
    primary: '#2d3748',
    secondary: '#4a5568',
    disabled: '#a0aec0'
  },
  divider: 'rgba(0, 0, 0, 0.12)',
  action: {
    active: '#2d3748',
    hover: 'rgba(0, 0, 0, 0.04)',
    selected: 'rgba(102, 126, 234, 0.08)',
    disabled: 'rgba(0, 0, 0, 0.26)',
    disabledBackground: 'rgba(0, 0, 0, 0.12)'
  }
};

// Shared theme configuration
const getSharedThemeOptions = (isDarkMode: boolean): ThemeOptions => ({
  palette: {
    mode: isDarkMode ? 'dark' : 'light',
    ...(isDarkMode ? darkPalette : lightPalette)
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      lineHeight: 1.2
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.5
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.5
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6
    },
    button: {
      textTransform: 'none',
      fontWeight: 500
    }
  },
  shape: {
    borderRadius: 12
  },
  shadows: isDarkMode ? [
    'none',
    '0px 2px 4px rgba(0,0,0,0.4)',
    '0px 3px 6px rgba(0,0,0,0.4)',
    '0px 3px 8px rgba(0,0,0,0.4)',
    '0px 4px 10px rgba(0,0,0,0.4)',
    '0px 5px 12px rgba(0,0,0,0.4)',
    '0px 6px 14px rgba(0,0,0,0.4)',
    '0px 7px 16px rgba(0,0,0,0.4)',
    '0px 8px 18px rgba(0,0,0,0.4)',
    '0px 9px 20px rgba(0,0,0,0.4)',
    '0px 10px 22px rgba(0,0,0,0.4)',
    '0px 11px 24px rgba(0,0,0,0.4)',
    '0px 12px 26px rgba(0,0,0,0.4)',
    '0px 13px 28px rgba(0,0,0,0.5)',
    '0px 14px 30px rgba(0,0,0,0.5)',
    '0px 15px 32px rgba(0,0,0,0.5)',
    '0px 16px 34px rgba(0,0,0,0.5)',
    '0px 17px 36px rgba(0,0,0,0.5)',
    '0px 18px 38px rgba(0,0,0,0.5)',
    '0px 19px 40px rgba(0,0,0,0.5)',
    '0px 20px 42px rgba(0,0,0,0.5)',
    '0px 21px 44px rgba(0,0,0,0.6)',
    '0px 22px 46px rgba(0,0,0,0.6)',
    '0px 23px 48px rgba(0,0,0,0.6)',
    '0px 24px 50px rgba(0,0,0,0.6)'
  ] : undefined,
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: isDarkMode ? '#4a5568 #1a1d21' : '#cbd5e0 #f7fafc',
          '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
            width: '8px',
            height: '8px'
          },
          '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
            borderRadius: '4px',
            backgroundColor: isDarkMode ? '#4a5568' : '#cbd5e0'
          },
          '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
            backgroundColor: isDarkMode ? '#1a1d21' : '#f7fafc'
          }
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '8px 16px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: isDarkMode 
              ? '0 4px 12px rgba(102, 126, 234, 0.3)'
              : '0 4px 12px rgba(102, 126, 234, 0.2)'
          }
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: isDarkMode
              ? '0 4px 12px rgba(102, 126, 234, 0.3)'
              : '0 4px 12px rgba(102, 126, 234, 0.2)'
          }
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          boxShadow: isDarkMode
            ? '0 4px 6px rgba(0, 0, 0, 0.3)'
            : '0 4px 6px rgba(0, 0, 0, 0.1)',
          backgroundImage: isDarkMode
            ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(237, 100, 166, 0.03) 100%)'
            : 'none',
          border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.05)' : 'none',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: isDarkMode
              ? '0 8px 16px rgba(0, 0, 0, 0.4)'
              : '0 8px 16px rgba(0, 0, 0, 0.15)'
          }
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          backgroundImage: isDarkMode
            ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.02) 0%, rgba(237, 100, 166, 0.02) 100%)'
            : 'none',
          border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.05)' : 'none'
        }
      }
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: isDarkMode ? '#1a1d21' : '#ffffff',
          color: isDarkMode ? '#e2e8f0' : '#2d3748',
          boxShadow: isDarkMode
            ? '0 2px 4px rgba(0, 0, 0, 0.3)'
            : '0 2px 4px rgba(0, 0, 0, 0.1)',
          borderBottom: isDarkMode
            ? '1px solid rgba(255, 255, 255, 0.05)'
            : '1px solid rgba(0, 0, 0, 0.1)'
        }
      }
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: isDarkMode ? '#1a1d21' : '#ffffff',
          borderRight: isDarkMode
            ? '1px solid rgba(255, 255, 255, 0.05)'
            : '1px solid rgba(0, 0, 0, 0.1)'
        }
      }
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '8px',
            '& fieldset': {
              borderColor: isDarkMode
                ? 'rgba(255, 255, 255, 0.1)'
                : 'rgba(0, 0, 0, 0.23)'
            },
            '&:hover fieldset': {
              borderColor: isDarkMode
                ? 'rgba(255, 255, 255, 0.2)'
                : 'rgba(0, 0, 0, 0.4)'
            }
          }
        }
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '6px',
          fontWeight: 500
        }
      }
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '8px'
        }
      }
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: isDarkMode ? '#2d3748' : '#4a5568',
          color: '#ffffff',
          fontSize: '0.875rem',
          borderRadius: '6px'
        }
      }
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          width: 42,
          height: 26,
          padding: 0,
          '& .MuiSwitch-switchBase': {
            padding: 0,
            margin: 2,
            transitionDuration: '300ms',
            '&.Mui-checked': {
              transform: 'translateX(16px)',
              color: '#fff',
              '& + .MuiSwitch-track': {
                backgroundColor: '#667eea',
                opacity: 1,
                border: 0
              }
            }
          },
          '& .MuiSwitch-thumb': {
            boxSizing: 'border-box',
            width: 22,
            height: 22
          },
          '& .MuiSwitch-track': {
            borderRadius: 26 / 2,
            backgroundColor: isDarkMode ? '#39393D' : '#E9E9EA',
            opacity: 1,
            transition: 'background-color 300ms'
          }
        }
      }
    }
  }
});

// Create themes
export const lightTheme = createTheme(getSharedThemeOptions(false));
export const darkTheme = createTheme(getSharedThemeOptions(true));

// Theme persistence utilities
export const THEME_KEY = 'ytempire_theme_mode';

export const getStoredTheme = (): 'light' | 'dark' | null => {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === 'light' || stored === 'dark' ? stored : null;
  } catch {
    return null;
  }
};

export const setStoredTheme = (mode: 'light' | 'dark'): void => {
  try {
    localStorage.setItem(THEME_KEY, mode);
  } catch {
    console.warn('Failed to save theme preference');
  }
};

// System preference detection
export const getSystemTheme = (): 'light' | 'dark' => {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

// Theme transition helper
export const enableSmoothTransition = (): void => {
  const style = document.createElement('style');
  style.innerHTML = `
    * {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
    }
  `;
  document.head.appendChild(style);
  setTimeout(() => {
    document.head.removeChild(style);
  }, 300);
};