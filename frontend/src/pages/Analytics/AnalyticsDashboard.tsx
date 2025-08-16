import React, { useState } from 'react';
import { 
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Tooltip as RechartsTooltip,
  Box,
  Tabs,
  Tab,
  Paper,
  Typography,
  Container
 } from '@mui/material';
import {  RevenueDashboard  } from '../../components/Dashboard/RevenueDashboard';
import {  UserBehaviorDashboard  } from '../../components/Analytics/UserBehaviorDashboard';
import {  PerformanceDashboard  } from '../../components/Performance/PerformanceDashboard';
import {  ABTestDashboard  } from '../../components/Experiments/ABTestDashboard';

interface TabPanelProps {
  
children?: React.ReactNode;
index: number;
value: number;

}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <>
      <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )}

export const AnalyticsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0</>
  );

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
};
  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Analytics & Metrics Dashboard
        </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
          Comprehensive analytics for, revenue, user, behavior, performance, and experiments
        </Typography>

        <Paper sx={{ width: '100%', mt: 3 }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
          >
            <Tab label="Revenue Tracking" />
            <Tab label="User Behavior" />
            <Tab label="Performance" />
            <Tab label="A/B Testing" />
          </Tabs>

          <TabPanel value={activeTab} index={0}>
            <RevenueDashboard />
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <UserBehaviorDashboard />
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <PerformanceDashboard />
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <ABTestDashboard />
          </TabPanel>
        </Paper>
      </Box>
    </Container>
  )};
