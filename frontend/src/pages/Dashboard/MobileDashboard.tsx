import React from 'react';
import { 
  Box,
  useMediaQuery,
  useTheme
 } from '@mui/material';
import {  MobileResponsiveSystem  } from '../../components/Mobile/MobileResponsiveSystem';
import {  MobileOptimizedDashboard  } from '../../components/Mobile/MobileOptimizedDashboard';
import {  MainDashboard  } from '../../components/Dashboard/MainDashboard';
import {  CustomizableWidgets  } from '../../components/Dashboard/CustomizableWidgets';
import {  RealTimeMetrics  } from '../../components/Dashboard/RealTimeMetrics';

const MobileDashboardPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));

  if (isMobile) {
    return (
    <>
      <Box>
        <MobileOptimizedDashboard />
      </Box>
    )}

  if (isTablet) {
    return (
    <Box>
        <MobileResponsiveSystem />
      </Box>
    )}

  // Desktop view with all features
  return (
    <Box>
      <MainDashboard />
      <Box sx={{ mt: 3 }}>
        <RealTimeMetrics />
      </Box>
      <Box sx={{ mt: 3 }}>
        <CustomizableWidgets />
      </Box>
    </Box>
  </>
  )
};
export default MobileDashboardPage;
