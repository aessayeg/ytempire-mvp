import React, { useState } from 'react';
import { 
  Container,
  Box,
  Typography,
  Paper,
  Tabs,
  Tab
 } from '@mui/material';
import {  AdvancedCharts  } from '../../components/Charts/AdvancedCharts';
import {  ChartComponents  } from '../../components/Charts/ChartComponents';
import {  ChannelPerformanceCharts  } from '../../components/Charts/ChannelPerformanceCharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number,
  value: number}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <>
      <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  )}

const AdvancedAnalyticsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0</>
  );

  return (
    <Container maxWidth={false}>
      <Box sx={{ py: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Advanced Analytics
        </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Comprehensive analytics with advanced visualizations
        </Typography>

        <Paper sx={{ width: '100%' }}>
          <Tabs
            value={activeTab}
            onChange={(_, v) => setActiveTab(v}
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab label="Advanced Charts" />
            <Tab label="Standard Charts" />
            <Tab label="Channel Performance" />
          </Tabs>

          <TabPanel value={activeTab} index={0}>
            <AdvancedCharts />
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <ChartComponents />
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <ChannelPerformanceCharts />
          </TabPanel>
        </Paper>
      </Box>
    </Container>
  )};

export default AdvancedAnalyticsPage;