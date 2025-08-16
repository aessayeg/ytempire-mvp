import React, { useState } from 'react';
import { 
  Container,
  Box,
  Typography,
  Tabs,
  Tab,
  Paper
} from '@mui/material';
import Grid from '@mui/material/Grid2';
import ChannelDashboard from '../../components/Channels/ChannelDashboard';
import ChannelList from '../../components/Channels/ChannelList';
import ChannelHealthDashboard from '../../components/Channels/ChannelHealthDashboard';
import ChannelTemplates from '../../components/Channels/ChannelTemplates';
import BulkOperations from '../../components/Channels/BulkOperations';

const ChannelDashboardPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <>
      <Container maxWidth={false}>
      <Box sx={{ py: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Channel Management Dashboard
        </Typography>
      <Paper sx={{ mb: 3 }}>
          <Tabs
            value={activeTab}
            onChange={(_, v) => setActiveTab(v)}
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Overview" />
            <Tab label="Channel List" />
            <Tab label="Health Dashboard" />
            <Tab label="Templates" />
            <Tab label="Bulk Operations" />
          </Tabs>
        </Paper>

        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid size={{ xs: 12, lg: 8 }}>
              <ChannelDashboard />
            </Grid>
            <Grid size={{ xs: 12, lg: 4 }}>
              <ChannelHealthDashboard />
            </Grid>
          </Grid>
        )}
        {activeTab === 1 && <ChannelList />}
        {activeTab === 2 && <ChannelHealthDashboard />}
        {activeTab === 3 && <ChannelTemplates />}
        {activeTab === 4 && <BulkOperations />}
      </Box>
    </Container>
    </>
  );
};
export default ChannelDashboardPage;
