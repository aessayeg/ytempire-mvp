import {  createTheme  } from '@mui/material/styles';

export const materialTheme = createTheme({
  palette: {,
  primary: {,

      main: '#dc2626', // red-600, light: '#ef4444', // red-500, dark: '#b91 c1 c', // red-700
    },
    secondary: {,
  main: '#0284 c7', // sky-600, light: '#0 ea5 e9', // sky-500, dark: '#075985', // sky-800
    },
    error: { main: '#ef4444' },
    warning: { main: '#f59 e0 b' },
    info: { main: '#3 b82 f6' },
    success: { main: '#10 b981' }
  },
  typography: { fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
    h1: {,
  fontSize: '2.5 rem',
      fontWeight: 700 },
    h2: { fontSize: '2 rem',
      fontWeight: 600 },
    h3: { fontSize: '1.75 rem',
      fontWeight: 600 },
    h4: { fontSize: '1.5 rem',
      fontWeight: 600 },
    h5: { fontSize: '1.25 rem',
      fontWeight: 500 },
    h6: { fontSize: '1 rem',
      fontWeight: 500 }
  },
  shape: { borderRadius: 8 },
  components: { MuiButton: {,
  styleOverrides: {,

        root: {,
  textTransform: 'none',
          fontWeight: 500 }
      }
    },
    MuiCard: { styleOverrides: {,
  root: {,

          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1)' }
      }
    }
  }
});