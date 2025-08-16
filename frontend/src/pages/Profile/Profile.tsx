/**
 * Profile Screen Component
 * Comprehensive user profile management and account overview
 */
import React, { useState, useEffect } from 'react';
import { 
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Avatar,
  Button,
  TextField,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Badge,
  Tab,
  Tabs,
  Switch,
  useTheme
 } from '@mui/material';
import { 
  Edit,
  Save,
  Cancel,
  Upload,
  Download,
  TrendingUp,
  VideoLibrary,
  AttachMoney,
  Notifications,
  Security,
  CreditCard,
  Receipt,
  YouTube,
  Twitter,
  Instagram,
  LinkedIn,
  Language,
  Email,
  Phone,
  LocationOn,
  CalendarToday,
  Verified,
  Group,
  Share,
  Settings,
  History,
  WorkspacePremium
 } from '@mui/icons-material';
import { format } from 'date-fns';
import { useAuthStore } from '../../stores/authStore';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const mockUserStats = {
  totalVideos: 156,
  totalViews: 2450000,
  totalSubscribers: 471000,
  totalRevenue: 31300,
  averageEngagement: 8.7,
  channelsManaged: 5,
  joinDate: '2023-03-15',
  lastActive: '2024-01-14T10:30:00Z'
};
const mockAchievements = [
  {
    id: 1,
    title: 'First Million Views',
    description: 'Reached 1 million total views',
    icon: <TrendingUp />,
    earned: true,
    dateEarned: '2023-08-15',
    rarity: 'rare'
  },
  {
    id: 2,
    title: 'Content Creator',
    description: 'Published 100+ videos',
    icon: <VideoLibrary />,
    earned: true,
    dateEarned: '2023-12-01',
    rarity: 'common'
  },
  {
    id: 3,
    title: 'Revenue Milestone',
    description: 'Earned $10,000+ in revenue',
    icon: <AttachMoney />,
    earned: true,
    dateEarned: '2023-10-20',
    rarity: 'uncommon'
  },
  {
    id: 4,
    title: 'AI Pioneer',
    description: 'Used AI tools for 6+ months',
    icon: <WorkspacePremium />,
    earned: true,
    dateEarned: '2023-09-15',
    rarity: 'epic'
  },
  {
    id: 5,
    title: 'Multi-Channel Master',
    description: 'Manage 5+ channels simultaneously',
    icon: <Group />,
    earned: false,
    dateEarned: null,
    rarity: 'legendary'
  }
];

const mockRecentActivity = [
  {
    id: 1,
    action: 'Published video',
    details: '"10 AI Tools Every Creator Needs"',
    timestamp: '2024-01-14T08:30:00Z',
    channel: 'Tech Reviews'
  },
  {
    id: 2,
    action: 'Updated channel settings',
    details: 'Gaming Central - Changed upload schedule',
    timestamp: '2024-01-13T15:45:00Z',
    channel: 'Gaming Central'
  },
  {
    id: 3,
    action: 'Generated AI script',
    details: 'Tutorial on video editing',
    timestamp: '2024-01-13T12:20:00Z',
    channel: 'Educational Hub'
  },
  {
    id: 4,
    action: 'Revenue milestone',
    details: 'Reached $30,000 total earnings',
    timestamp: '2024-01-12T09:15:00Z',
    channel: 'All Channels'
  }
];

const mockSocialConnections = [
  { platform: 'YouTube', handle: '@techcreator', followers: '471K', connected: true, verified: true },
  { platform: 'Twitter', handle: '@techcreator', followers: '89K', connected: true, verified: false },
  { platform: 'Instagram', handle: '@techcreator', followers: '145K', connected: true, verified: true },
  { platform: 'LinkedIn', handle: 'Tech Creator', followers: '23K', connected: false, verified: false }
];

export const Profile: React.FC = () => {
  const theme = useTheme();
  const { user, logout } = useAuthStore();
  const [activeTab, setActiveTab] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const [profileData, setProfileData] = useState({
    fullName: user?.full_name || 'Tech Creator',
    email: user?.email || 'tech@creator.com',
    bio: 'YouTube automation enthusiast helping creators scale with AI. Building the future of content, creation, one video at a time.',
    location: 'San Francisco, CA',
    website: 'https://techcreator.com',
    phone: '+1 (555) 123-4567',
    timezone: 'America/Los_Angeles',
    language: 'English',
    notifications: {
      email: true,
      push: true,
      marketing: false
    }
  });

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  const handleSaveProfile = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setEditMode(false);
    }, 2000);
  };
  const handleInputChange = (field: string, value: string) => {
    setProfileData(prev => ({
      ...prev,
      [field]: value
    }));
  };
  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };
  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case 'legendary': return '#FFD700';
      case 'epic': return '#9C27B0';
      case 'rare': return '#2196F3';
      case 'uncommon': return '#4CAF50';
      default: return '#9E9E9E';
    }
  };
  const getSocialIcon = (platform: string) => {
    switch (platform) {
      case 'YouTube': return <YouTube />;
      case 'Twitter': return <Twitter />;
      case 'Instagram': return <Instagram />;
      case 'LinkedIn': return <LinkedIn />;
      default: return <Share />;
    }
  };
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ p: 4 }}>
              <Grid container spacing={3} alignItems="center">
                <Grid item>
                  <Badge
                    overlap="circular"
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    badgeContent={
                      <Avatar sx={{ width: 24, height: 24, bgcolor: 'success.main' }}>
                        <Verified sx={{ fontSize: 16 }} />
                      </Avatar>}
                  >
                    <Avatar 
                      sx={{ 
                        width: 120, 
                        height: 120, 
                        fontSize: '3rem',
                        border: `4px solid ${theme.palette.primary.main}`
                      }}
                    >
                      {profileData.fullName.charAt(0)}
                    </Avatar>
                  </Badge>
                </Grid>
                <Grid item xs>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h4" fontWeight="bold" sx={{ mr: 2 }}>
                      {profileData.fullName}
                    </Typography>
                    <Chip label="Pro Member" color="primary" variant="outlined" />
                    <Chip label="Verified" color="success" variant="outlined" sx={{ ml: 1 }} />
                  </Box>
                  
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 2, maxWidth: 600 }}>
                    {profileData.bio}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <LocationOn fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">{profileData.location}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <CalendarToday fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">
                        Joined {format(new Date(mockUserStats.joinDate), 'MMMM yyyy')}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Language fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">{profileData.website}</Typography>
                    </Box>
                  </Box>

                  {/* Quick Stats */}
                  <Grid container spacing={3} sx={{ mt: 2 }}>
                    {[ { label: 'Videos', value: mockUserStats.totalVideos, icon: <VideoLibrary /> },
                      { label: 'Views', value: formatNumber(mockUserStats.totalViews), icon: <TrendingUp /> },
                      { label: 'Subscribers', value: formatNumber(mockUserStats.totalSubscribers), icon: <YouTube /> },
                      { label: 'Revenue', value: formatCurrency(mockUserStats.totalRevenue), icon: <AttachMoney /> } ].map((stat, index) => (
                      <Grid item key={index}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                            {stat.icon}
                            <Typography variant="h6" sx={{ ml: 1, fontWeight: 'bold' }}>
                              {stat.value}
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {stat.label}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </Grid>

                <Grid item>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button 
                      variant={editMode ? 'contained' : 'outlined'} 
                      startIcon={editMode ? <Save /> : <Edit />}
                      onClick={editMode ? handleSaveProfile : () => setEditMode(true)}
                      disabled={loading}
                    >
                      {loading ? 'Saving...' : editMode ? 'Save' : 'Edit Profile'}
                    </Button>
                    {editMode && (
                      <Button 
                        variant="outlined" 
                        startIcon={<Cancel />}
                        onClick={() => setEditMode(false)}
                      >
                        Cancel
                      </Button>
                    )}
                    <Button variant="outlined" startIcon={<Settings />}>
                      Settings
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Profile Tabs */}
      <Paper sx={{ width: '100%' }}>
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }>
          <Tab label="Overview" />
          <Tab label="Activity" />
          <Tab label="Achievements" />
          <Tab label="Social" />
          <Tab label="Settings" />
        </Tabs>

        {/* Overview Tab */}
        <TabPanel value={activeTab} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card sx={{ mb: 3 }}>
                <CardHeader title="Profile Information" />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Full Name"
                        value={profileData.fullName}
                        onChange={(e) => handleInputChange('fullName', e.target.value)}
                        disabled={!editMode}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Email"
                        value={profileData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        disabled={!editMode}
                        type="email"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Bio"
                        value={profileData.bio}
                        onChange={(e) => handleInputChange('bio', e.target.value)}
                        disabled={!editMode}
                        multiline
                        rows={3}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Location"
                        value={profileData.location}
                        onChange={(e) => handleInputChange('location', e.target.value)}
                        disabled={!editMode}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Website"
                        value={profileData.website}
                        onChange={(e) => handleInputChange('website', e.target.value)}
                        disabled={!editMode}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Phone"
                        value={profileData.phone}
                        onChange={(e) => handleInputChange('phone', e.target.value)}
                        disabled={!editMode}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Timezone"
                        value={profileData.timezone}
                        onChange={(e) => handleInputChange('timezone', e.target.value)}
                        disabled={!editMode}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ mb: 3 }}>
                <CardHeader title="Account Status" />
                <CardContent>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Verified color="success" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Verified Account" 
                        secondary="Email and phone verified"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <WorkspacePremium color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Pro Subscription" 
                        secondary="Active until March 2024"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Security color="success" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Two-Factor Auth" 
                        secondary="Enabled"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>

              <Card>
                <CardHeader title="Quick Actions" />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button startIcon={<Download />} variant="outlined" fullWidth>
                      Export Data
                    </Button>
                    <Button startIcon={<Upload />} variant="outlined" fullWidth>
                      Import Settings
                    </Button>
                    <Button startIcon={<Receipt />} variant="outlined" fullWidth>
                      Billing History
                    </Button>
                    <Button startIcon={<CreditCard />} variant="outlined" fullWidth>
                      Payment Methods
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Activity Tab */}
        <TabPanel value={activeTab} index={1}>
          <Card>
            <CardHeader title="Recent Activity" />
            <CardContent>
              <List>
                {mockRecentActivity.map((activity) => (
                  <ListItem key={activity.id} divider>
                    <ListItemIcon>
                      <History />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box>
                          <Typography variant="body1" component="span" fontWeight="bold">
                            {activity.action}
                          </Typography>
                          <Typography variant="body1" component="span" sx={{ ml: 1 }}>
                            {activity.details}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            {format(new Date(activity.timestamp), 'PPp')}
                          </Typography>
                          <Chip label={activity.channel} size="small" variant="outlined" />
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </TabPanel>

        {/* Achievements Tab */}
        <TabPanel value={activeTab} index={2}>
          <Card>
            <CardHeader title="Achievements & Milestones" />
            <CardContent>
              <Grid container spacing={2}>
                {mockAchievements.map((achievement) => (
                  <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        opacity: achievement.earned ? 1 : 0.5,
                        border: achievement.earned ? `2px solid ${getRarityColor(achievement.rarity)}` : undefined
                      }}
                    >
                      <CardContent sx={{ textAlign: 'center' }}>
                        <Avatar 
                          sx={{ 
                            bgcolor: getRarityColor(achievement.rarity),
                            width: 64,
                            height: 64,
                            mx: 'auto',
                            mb: 2
                          }}
                        >
                          {achievement.icon}
                        </Avatar>
                        <Typography variant="h6" gutterBottom>
                          {achievement.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {achievement.description}
                        </Typography>
                        <Chip 
                          label={achievement.rarity} 
                          size="small" 
                          sx={{ 
                            bgcolor: getRarityColor(achievement.rarity),
                            color: 'white',
                            textTransform: 'capitalize'
                          }}
                        />
                        {achievement.earned && (
                          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                            Earned: {format(new Date(achievement.dateEarned!), 'PP')}
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>

        {/* Social Tab */}
        <TabPanel value={activeTab} index={3}>
          <Card>
            <CardHeader title="Social Media Connections" />
            <CardContent>
              <List>
                {mockSocialConnections.map((social, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {getSocialIcon(social.platform)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body1">{social.platform}</Typography>
                          {social.verified && <Verified color="primary" fontSize="small" />}
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2">{social.handle}</Typography>
                          <Typography variant="caption">{social.followers} followers</Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Switch checked={social.connected} />
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </TabPanel>

        {/* Settings Tab */}
        <TabPanel value={activeTab} index={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Notification Preferences" />
                <CardContent>
                  <List>
                    <ListItem>
                      <ListItemText primary="Email Notifications" secondary="Receive updates via email" />
                      <ListItemSecondaryAction>
                        <Switch checked={profileData.notifications.email} />
                      </ListItemSecondaryAction>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Push Notifications" secondary="Browser and mobile notifications" />
                      <ListItemSecondaryAction>
                        <Switch checked={profileData.notifications.push} />
                      </ListItemSecondaryAction>
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Marketing Emails" secondary="Product updates and promotions" />
                      <ListItemSecondaryAction>
                        <Switch checked={profileData.notifications.marketing} />
                      </ListItemSecondaryAction>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Account Management" />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Button startIcon={<Security />} variant="outlined" fullWidth>
                      Change Password
                    </Button>
                    <Button startIcon={<CreditCard />} variant="outlined" fullWidth>
                      Manage Billing
                    </Button>
                    <Button startIcon={<Download />} variant="outlined" fullWidth>
                      Download Data
                    </Button>
                    <Divider />
                    <Button 
                      startIcon={<Cancel />} 
                      variant="outlined" 
                      color="error" 
                      fullWidth
                      onClick={() => setShowDeleteDialog(true)}
                    >
                      Delete Account
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* Delete Account Dialog */}
      <Dialog open={showDeleteDialog} onClose={() => setShowDeleteDialog(false)}>
        <DialogTitle>Delete Account</DialogTitle>
        <DialogContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            This action cannot be undone. All your data will be permanently deleted.
          </Alert>
          <Typography>
            Are you sure you want to delete your account? This will remove all your videos, 
            channels, and associated data.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDeleteDialog(false)}>Cancel</Button>
          <Button color="error" variant="contained">
            Delete Account
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Profile;
