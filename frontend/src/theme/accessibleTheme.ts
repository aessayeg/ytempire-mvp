import { createTheme } from '@mui/material/styles';
import type { ThemeOptions } from '@mui/material/styles';
import { meetsWCAGAA } from '../utils/accessibility';

// WCAG AA Compliant Color Palette
const accessibleColors = {
  primary: {
    main: '#5E35B1', // Deeper purple for better contrast
    light: '#7E57C2',
    dark: '#4527A0',
    contrastText: '#FFFFFF',
  },
  secondary: {
    main: '#00897B', // Teal with good contrast
    light: '#4DB6AC',
    dark: '#00695C',
    contrastText: '#FFFFFF',
  },
  error: {
    main: '#C62828', // Darker red for better contrast
    light: '#EF5350',
    dark: '#B71C1C',
    contrastText: '#FFFFFF',
  },
  warning: {
    main: '#F57C00', // Darker orange
    light: '#FFB74D',
    dark: '#E65100',
    contrastText: '#000000', // Black text on orange
  },
  info: {
    main: '#0277BD', // Darker blue
    light: '#4FC3F7',
    dark: '#01579B',
    contrastText: '#FFFFFF',
  },
  success: {
    main: '#2E7D32', // Darker green
    light: '#66BB6A',
    dark: '#1B5E20',
    contrastText: '#FFFFFF',
  },
  grey: {
    50: '#FAFAFA',
    100: '#F5F5F5',
    200: '#EEEEEE',
    300: '#E0E0E0',
    400: '#BDBDBD',
    500: '#9E9E9E',
    600: '#757575',
    700: '#616161',
    800: '#424242',
    900: '#212121',
  },
};

// Verify color contrast ratios
const verifyContrast = () => {
  const checks = [
    { fg: accessibleColors.primary.contrastText, bg: accessibleColors.primary.main, name: 'Primary' },
    { fg: accessibleColors.secondary.contrastText, bg: accessibleColors.secondary.main, name: 'Secondary' },
    { fg: accessibleColors.error.contrastText, bg: accessibleColors.error.main, name: 'Error' },
    { fg: accessibleColors.warning.contrastText, bg: accessibleColors.warning.main, name: 'Warning' },
    { fg: accessibleColors.info.contrastText, bg: accessibleColors.info.main, name: 'Info' },
    { fg: accessibleColors.success.contrastText, bg: accessibleColors.success.main, name: 'Success' },
  ];

  checks.forEach(({ fg, bg, name }) => {
    const passes = meetsWCAGAA(fg, bg);
    if (!passes) {
      console.warn(`${name} color combination does not meet WCAG AA standards`);
    }
  });
};

// Create accessible theme
export const createAccessibleTheme = (mode: 'light' | 'dark' = 'light'): ReturnType<typeof createTheme> => {
  const themeOptions: ThemeOptions = {
    palette: {
      mode,
      primary: accessibleColors.primary,
      secondary: accessibleColors.secondary,
      error: accessibleColors.error,
      warning: accessibleColors.warning,
      info: accessibleColors.info,
      success: accessibleColors.success,
      grey: accessibleColors.grey,
      background: {
        default: mode === 'light' ? '#FFFFFF' : '#121212',
        paper: mode === 'light' ? '#FFFFFF' : '#1E1E1E',
      },
      text: {
        primary: mode === 'light' ? 'rgba(0, 0, 0, 0.87)' : 'rgba(255, 255, 255, 0.87)',
        secondary: mode === 'light' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(255, 255, 255, 0.6)',
        disabled: mode === 'light' ? 'rgba(0, 0, 0, 0.38)' : 'rgba(255, 255, 255, 0.38)',
      },
    },
    typography: {
      // Ensure readable font sizes
      fontSize: 14,
      h1: {
        fontSize: '2.5rem',
        fontWeight: 500,
        lineHeight: 1.2,
        letterSpacing: '-0.01562em',
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 500,
        lineHeight: 1.2,
        letterSpacing: '-0.00833em',
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 500,
        lineHeight: 1.2,
        letterSpacing: '0em',
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 500,
        lineHeight: 1.3,
        letterSpacing: '0.00735em',
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 500,
        lineHeight: 1.4,
        letterSpacing: '0em',
      },
      h6: {
        fontSize: '1.125rem',
        fontWeight: 500,
        lineHeight: 1.4,
        letterSpacing: '0.0075em',
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.5,
        letterSpacing: '0.00938em',
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.43,
        letterSpacing: '0.01071em',
      },
      button: {
        fontSize: '0.875rem',
        fontWeight: 500,
        lineHeight: 1.75,
        letterSpacing: '0.02857em',
        textTransform: 'uppercase',
      },
    },
    components: {
      // Ensure all interactive elements have proper focus indicators
      MuiButton: {
        styleOverrides: {
          root: {
            minHeight: 36, // Minimum touch target size
            '&:focus-visible': {
              outline: '3px solid',
              outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
              outlineOffset: 2,
            },
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            minWidth: 44, // Minimum touch target size
            minHeight: 44,
            '&:focus-visible': {
              outline: '3px solid',
              outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
              outlineOffset: 2,
            },
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              '&:focus-within': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderWidth: 2,
                  borderColor: accessibleColors.primary.main,
                },
              },
            },
          },
        },
      },
      MuiLink: {
        styleOverrides: {
          root: {
            textDecorationLine: 'underline',
            '&:focus-visible': {
              outline: '3px solid',
              outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
              outlineOffset: 2,
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            '&:focus-visible': {
              outline: '3px solid',
              outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
              outlineOffset: 2,
            },
          },
        },
      },
      // Add focus indicators to all focusable elements
      MuiCheckbox: {
        styleOverrides: {
          root: {
            '&:focus-visible': {
              '& .MuiSvgIcon-root': {
                outline: '3px solid',
                outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
                outlineOffset: 2,
              },
            },
          },
        },
      },
      MuiRadio: {
        styleOverrides: {
          root: {
            '&:focus-visible': {
              '& .MuiSvgIcon-root': {
                outline: '3px solid',
                outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
                outlineOffset: 2,
              },
            },
          },
        },
      },
      MuiSwitch: {
        styleOverrides: {
          root: {
            '&:focus-within': {
              '& .MuiSwitch-thumb': {
                outline: '3px solid',
                outlineColor: mode === 'light' ? accessibleColors.primary.main : accessibleColors.primary.light,
                outlineOffset: 2,
              },
            },
          },
        },
      },
      // Ensure sufficient padding for touch targets
      MuiListItem: {
        styleOverrides: {
          root: {
            minHeight: 48,
          },
        },
      },
      MuiMenuItem: {
        styleOverrides: {
          root: {
            minHeight: 48,
          },
        },
      },
      // High contrast borders for inputs
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: mode === 'light' ? 'rgba(0, 0, 0, 0.23)' : 'rgba(255, 255, 255, 0.23)',
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: mode === 'light' ? 'rgba(0, 0, 0, 0.87)' : 'rgba(255, 255, 255, 0.87)',
            },
          },
        },
      },
    },
    shape: {
      borderRadius: 4,
    },
    spacing: 8,
  };

  const theme = createTheme(themeOptions);
  
  // Verify contrast in development
  if (import.meta.env.DEV) {
    verifyContrast();
  }

  return theme;
};

// Export default accessible themes
export const lightAccessibleTheme = createAccessibleTheme('light');
export const darkAccessibleTheme = createAccessibleTheme('dark');

// Utility to check if current theme meets WCAG standards
export const auditThemeContrast = (theme: ReturnType<typeof createTheme>) => {
  const results: { component: string; passes: boolean; ratio?: number }[] = [];
  
  // Check primary colors
  const primaryCheck = meetsWCAGAA(
    theme.palette.primary.contrastText,
    theme.palette.primary.main
  );
  results.push({ component: 'Primary Button', passes: primaryCheck });

  // Check error colors
  const errorCheck = meetsWCAGAA(
    theme.palette.error.contrastText,
    theme.palette.error.main
  );
  results.push({ component: 'Error Message', passes: errorCheck });

  // Check text on background
  const textCheck = meetsWCAGAA(
    theme.palette.text.primary,
    theme.palette.background.default
  );
  results.push({ component: 'Body Text', passes: textCheck });

  return results;
};