/**
 * Sample Integration of P2 Frontend Features
 * This shows how to integrate all P2 components into the main application
 */

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Assessment as ReportsIcon,
  GetApp as ExportIcon
} from '@mui/icons-material';

// Import P2 Components
import { EnhancedThemeProvider, useEnhancedTheme } from './contexts/EnhancedThemeContext';
import { CustomReports } from './components/Reports/CustomReports';
import { CompetitiveAnalysisDashboard } from './components/Analytics/CompetitiveAnalysisDashboard';
import { 
  AnimatedCard, 
  PageTransition,
  FloatingActionButton 
} from './components/Animations/AdvancedAnimations';
import { UniversalExportManager, useExport } from './components/Export/UniversalExportManager';

// Main App Component with P2 Features
const AppWithP2Features: React.FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { isDarkMode, toggleTheme } = useEnhancedTheme();
  
  // Sample data for export
  const exportData = {
    title: 'Application Data',
    data: [],
    columns: []
  };
  
  const { openExportDialog, ExportComponent } = useExport(exportData);

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Custom Reports', icon: <ReportsIcon />, path: '/reports' },
    { text: 'Competitive Analysis', icon: <AnalyticsIcon />, path: '/competitive' },
    { text: 'Export Data', icon: <ExportIcon />, action: openExportDialog }
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar position="fixed">
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
            YTEmpire MVP - P2 Features
          </Typography>
          <IconButton color="inherit" onClick={toggleTheme}>
            {isDarkMode ? '🌞' : '🌙'}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <List sx={{ width: 250 }}>
          {menuItems.map((item) => (
            <ListItem
              button
              key={item.text}
              component={item.path ? Link : 'div'}
              to={item.path}
              onClick={() => {
                if (item.action) item.action();
                setDrawerOpen(false);
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItem>
          ))}
        </List>
      </Drawer>

      {/* Main Content */}
      <Container sx={{ mt: 10, mb: 4 }}>
        <Routes>
          <Route path="/" element={
            <PageTransition>
              <AnimatedCard>
                <Typography variant="h4">Welcome to YTEmpire MVP</Typography>
                <Typography>All P2 Frontend Features are integrated!</Typography>
              </AnimatedCard>
            </PageTransition>
          } />
          <Route path="/reports" element={
            <PageTransition>
              <CustomReports />
            </PageTransition>
          } />
          <Route path="/competitive" element={
            <PageTransition>
              <CompetitiveAnalysisDashboard />
            </PageTransition>
          } />
        </Routes>
      </Container>

      {/* Export Dialog */}
      <ExportComponent />

      {/* Floating Action Button */}
      <FloatingActionButton onClick={openExportDialog}>
        <ExportIcon />
      </FloatingActionButton>
    </Box>
  );
};

// Root App with Theme Provider
const App: React.FC = () => {
  return (
    <EnhancedThemeProvider>
      <Router>
        <AppWithP2Features />
      </Router>
    </EnhancedThemeProvider>
  );
};

export default App;
