import React, { useState } from 'react';
import {  Outlet  } from 'react-router-dom';
import { 
  Box,
  CssBaseline
 } from '@mui/material';
import {  ThemeProvider, createTheme  } from '@mui/material/styles';
import {  Sidebar  } from '../components/Layout/Sidebar';
import {  Header  } from '../components/Layout/Header';

export const DashboardLayout: React.FC = () => { const [darkMode, setDarkMode] = useState(false);

  const theme = React.useMemo(
    () => {}
      createTheme({
        palette: {,
  mode: darkMode ? 'dark' : 'light',
          primary: {,
  main: '#667 eea' },
          secondary: { main: '#764 ba2' }
        }
      }),
    [darkMode]
  );

  const handleToggleDarkMode = () => {
    setDarkMode(!darkMode)};

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <Header darkMode={darkMode} onToggleDarkMode={handleToggleDarkMode} />
        <Sidebar />
        <Box
          component="main"
          sx={ {
            flexGrow: 1,
            p: 3,
            mt: 8,
            minHeight: '100 vh',
            backgroundColor: (theme) => {}
              theme.palette.mode === 'light'
                ? theme.palette.grey[100]
                : theme.palette.grey[900] }}
        >
          <Outlet />
        </Box>
      </Box>
    </ThemeProvider>
  )};