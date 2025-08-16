/**
 * User Settings Page
 * P2 Task: [FRONTEND] User Settings Pages
 * Comprehensive user preferences and account management
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  Switch,
  FormGroup,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  InputAdornment
} from '@mui/material';
import {
  Save,
  Cancel,
  Delete,
  Add,
  Visibility,
  VisibilityOff,
  Phone
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
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

interface ApiKey {
  id: string;
  name: string;
  key_preview: string;
  created_at: string;
  last_used?: string;
}

export const UserSettings: React.FC = () => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [deleteAccountDialog, setDeleteAccountDialog] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  
  const { control, handleSubmit, reset } = useForm({
    defaultValues: {
      fullName: user?.full_name || '',
      email: user?.email || '',
      phone: '',
      company: '',
      bio: '',
      timezone: 'UTC',
      language: 'en',
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
      notifications: {
        email_enabled: true,
        sms_enabled: false,
        push_enabled: true,
        video_complete: true,
        quota_warning: true,
        cost_alert: true,
        system_updates: true,
        marketing: false,
        email_frequency: 'instant'
      }
    }
  });

  useEffect(() => {
    // Load user settings
    loadSettings();
    loadApiKeys();
  }, []);

  const loadSettings = async () => {
    // Load user settings from API
    console.log('Loading settings...');
  };

  const loadApiKeys = async () => {
    // Mock API keys
    setApiKeys([
      {
        id: '1',
        name: 'Production API Key',
        key_preview: 'sk_live_abc123',
        created_at: '2024-01-01',
        last_used: '2024-01-14'
      }
    ]);
  };

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const onSubmitProfile = async (data: any) => {
    setLoading(true);
    try {
      // Save profile settings
      console.log('Saving profile:', data);
    } finally {
      setLoading(false);
    }
  };

  const onSubmitPassword = async (data: any) => {
    setLoading(true);
    try {
      // Change password
      console.log('Changing password:', data);
    } finally {
      setLoading(false);
    }
  };

  const onSubmitNotifications = async (data: any) => {
    setLoading(true);
    try {
      // Save notification preferences
      console.log('Saving notifications:', data);
    } finally {
      setLoading(false);
    }
  };

  const handleEnable2FA = () => {
    setTwoFactorEnabled(true);
  };

  const handleDisable2FA = () => {
    setTwoFactorEnabled(false);
  };

  const handleDeleteAccount = () => {
    console.log('Deleting account...');
    setDeleteAccountDialog(false);
  };

  const handleDeleteApiKey = (id: string) => {
    setApiKeys(apiKeys.filter(key => key.id !== id));
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your account settings and preferences
        </Typography>
      </Box>

      <Box sx={{ width: '100%' }}>
        <Paper>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Profile" />
            <Tab label="Security" />
            <Tab label="Notifications" />
            <Tab label="Billing" />
            <Tab label="API Keys" />
          </Tabs>

          <TabPanel value={activeTab} index={0}>
            {/* Profile Settings */}
            <form onSubmit={handleSubmit(onSubmitProfile)}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Personal Information
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="fullName"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Full Name"
                            required
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="email"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Email"
                            type="email"
                            disabled
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="phone"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Phone"
                            InputProps={{
                              startAdornment: <Phone sx={{ mr: 1, color: 'text.secondary' }} />
                            }}
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="company"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Company"
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Controller
                        name="bio"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Bio"
                            multiline
                            rows={4}
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="timezone"
                        control={control}
                        render={({ field }) => (
                          <FormControl fullWidth>
                            <InputLabel>Timezone</InputLabel>
                            <Select {...field} label="Timezone">
                              <MenuItem value="UTC">UTC</MenuItem>
                              <MenuItem value="America/New_York">Eastern Time</MenuItem>
                              <MenuItem value="America/Chicago">Central Time</MenuItem>
                              <MenuItem value="America/Denver">Mountain Time</MenuItem>
                              <MenuItem value="America/Los_Angeles">Pacific Time</MenuItem>
                            </Select>
                          </FormControl>
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="language"
                        control={control}
                        render={({ field }) => (
                          <FormControl fullWidth>
                            <InputLabel>Language</InputLabel>
                            <Select {...field} label="Language">
                              <MenuItem value="en">English</MenuItem>
                              <MenuItem value="es">Spanish</MenuItem>
                              <MenuItem value="fr">French</MenuItem>
                              <MenuItem value="de">German</MenuItem>
                              <MenuItem value="zh">Chinese</MenuItem>
                            </Select>
                          </FormControl>
                        )}
                      />
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<Save />}
                      disabled={loading}
                    >
                      Save Changes
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<Cancel />}
                      onClick={() => reset()}
                    >
                      Cancel
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </form>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            {/* Security Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Password
                </Typography>
                <form onSubmit={handleSubmit(onSubmitPassword)}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Controller
                        name="currentPassword"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Current Password"
                            type={showPassword ? 'text' : 'password'}
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    onClick={() => setShowPassword(!showPassword)}
                                    edge="end"
                                  >
                                    {showPassword ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              )
                            }}
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="newPassword"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="New Password"
                            type="password"
                          />
                        )}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="confirmPassword"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="Confirm Password"
                            type="password"
                          />
                        )}
                      />
                    </Grid>
                  </Grid>
                  
                  <Button
                    type="submit"
                    variant="contained"
                    sx={{ mt: 2 }}
                    disabled={loading}
                  >
                    Change Password
                  </Button>
                </form>
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Two-Factor Authentication
                </Typography>
                <Card variant="outlined">
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography variant="subtitle1">
                          {twoFactorEnabled ? 'Enabled' : 'Disabled'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Add an extra layer of security to your account
                        </Typography>
                      </Box>
                      <Button
                        variant={twoFactorEnabled ? 'outlined' : 'contained'}
                        onClick={twoFactorEnabled ? handleDisable2FA : handleEnable2FA}
                      >
                        {twoFactorEnabled ? 'Disable' : 'Enable'}
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom color="error">
                  Danger Zone
                </Typography>
                <Card variant="outlined" sx={{ borderColor: 'error.main' }}>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Delete Account
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Once you delete your account, there is no going back. Please be certain.
                    </Typography>
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<Delete />}
                      onClick={() => setDeleteAccountDialog(true)}
                    >
                      Delete Account
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            {/* Notification Settings */}
            <form onSubmit={handleSubmit(onSubmitNotifications)}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Notification Channels
                  </Typography>
                  <FormGroup>
                    <Controller
                      name="notifications.email_enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Email Notifications"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.sms_enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="SMS Notifications"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.push_enabled"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Push Notifications"
                        />
                      )}
                    />
                  </FormGroup>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Notification Types
                  </Typography>
                  <FormGroup>
                    <Controller
                      name="notifications.video_complete"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Video Generation Complete"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.quota_warning"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="YouTube Quota Warnings"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.cost_alert"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Cost Alerts"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.system_updates"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="System Updates"
                        />
                      )}
                    />
                    <Controller
                      name="notifications.marketing"
                      control={control}
                      render={({ field }) => (
                        <FormControlLabel
                          control={<Switch {...field} checked={field.value} />}
                          label="Marketing & Promotions"
                        />
                      )}
                    />
                  </FormGroup>
                </Grid>
                
                <Grid item xs={12}>
                  <Controller
                    name="notifications.email_frequency"
                    control={control}
                    render={({ field }) => (
                      <FormControl fullWidth>
                        <InputLabel>Email Frequency</InputLabel>
                        <Select {...field} label="Email Frequency">
                          <MenuItem value="instant">Instant</MenuItem>
                          <MenuItem value="daily">Daily Digest</MenuItem>
                          <MenuItem value="weekly">Weekly Summary</MenuItem>
                        </Select>
                      </FormControl>
                    )}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    disabled={loading}
                  >
                    Save Preferences
                  </Button>
                </Grid>
              </Grid>
            </form>
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            {/* Billing Settings */}
            <Typography variant="h6" gutterBottom>
              Billing & Subscription
            </Typography>
            <Alert severity="info" sx={{ mb: 2 }}>
              Billing features will be available in the next release.
            </Alert>
          </TabPanel>

          <TabPanel value={activeTab} index={4}>
            {/* API Keys */}
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h6">API Keys</Typography>
                <Button
                  variant="contained"
                  startIcon={<Add />}
                  onClick={() => setShowApiKeyDialog(true)}
                >
                  Create New Key
                </Button>
              </Box>
              
              <List>
                {apiKeys.map((key) => (
                  <ListItem key={key.id} divider>
                    <ListItemText
                      primary={key.name}
                      secondary={
                        <Box>
                          <Typography variant="body2" component="span">
                            {key.key_preview}...
                          </Typography>
                          <Box mt={1}>
                            <Chip
                              size="small"
                              label={`Created: ${new Date(key.created_at).toLocaleDateString()}`}
                              sx={{ mr: 1 }}
                            />
                            {key.last_used && (
                              <Chip
                                size="small"
                                label={`Last used: ${new Date(key.last_used).toLocaleDateString()}`}
                              />
                            )}
                          </Box>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => handleDeleteApiKey(key.id)}
                      >
                        <Delete />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
              
              {apiKeys.length === 0 && (
                <Alert severity="info">
                  No API keys created yet. Create one to start using the API.
                </Alert>
              )}
            </Box>
          </TabPanel>
        </Paper>
      </Box>

      {/* Delete Account Dialog */}
      <Dialog open={deleteAccountDialog} onClose={() => setDeleteAccountDialog(false)}>
        <DialogTitle>Delete Account</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action cannot be undone. All your data will be permanently deleted.
          </Alert>
          <Typography>
            Type "DELETE" to confirm:
          </Typography>
          <TextField
            fullWidth
            margin="normal"
            placeholder="DELETE"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteAccountDialog(false)}>Cancel</Button>
          <Button
            onClick={handleDeleteAccount}
            color="error"
            variant="contained"
          >
            Delete Account
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default UserSettings;