import React, { useState } from 'react';
import { Container, Box, Typography, Grid2 as Grid, Paper, Tabs, Tab } from '@mui/material';
import { LiveVideoGenerationMonitor } from '../../components/Monitoring/LiveVideoGenerationMonitor';
import { SystemHealthMonitors } from '../../components/Monitoring/SystemHealthMonitors';
import { CostTrackingDashboard } from '../../components/Monitoring/CostTrackingDashboard';
import { PerformanceDashboard } from '../../components/Performance/PerformanceDashboard';

const SystemMonitoringPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <Container maxWidth={false}>
      <Box sx={{ py: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Monitoring
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Real-time monitoring of system health, performance, and costs
        </Typography>

        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={activeTab}
            onChange={(e, v) => setActiveTab(v)}
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab label="Overview" />
            <Tab label="Video Generation" />
            <Tab label="System Health" />
            <Tab label="Cost Tracking" />
            <Tab label="Performance" />
          </Tabs>
        </Paper>

        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 6 }}>
              <LiveVideoGenerationMonitor />
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <SystemHealthMonitors />
            </Grid>
            <Grid size={12}>
              <CostTrackingDashboard />
            </Grid>
          </Grid>
        )}

        {activeTab === 1 && <LiveVideoGenerationMonitor />}
        {activeTab === 2 && <SystemHealthMonitors />}
        {activeTab === 3 && <CostTrackingDashboard />}
        {activeTab === 4 && <PerformanceDashboard />}
      </Box>
    </Container>
  );
};

export default SystemMonitoringPage;