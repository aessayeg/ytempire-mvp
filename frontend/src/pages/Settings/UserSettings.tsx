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
  Avatar,
  IconButton,
  Divider,
  Alert,
  Card,
  CardContent,
  CardActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  InputAdornment,
  Tooltip,
  LinearProgress,
  Stack,
} from '@mui/material';
import {
  Person,
  Security,
  Notifications,
  Payment,
  Api,
  Palette,
  Language,
  CloudUpload,
  Edit,
  Save,
  Cancel,
  Delete,
  Add,
  Visibility,
  VisibilityOff,
  ContentCopy,
  Refresh,
  Warning,
  CheckCircle,
  Info,
  Key,
  CreditCard,
  Email,
  Phone,
  Lock,
  TwoFactorAuth,
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import { useAuthStore } from '../../stores/authStore';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface UserProfile {
  id: string;
  email: string;
  name: string;
  phone?: string;
  avatar?: string;
  timezone: string;
  language: string;
  created_at: string;
  subscription_tier: string;
  company?: string;
  bio?: string;
}

interface NotificationSettings {
  email_enabled: boolean;
  sms_enabled: boolean;
  push_enabled: boolean;
  email_frequency: 'instant' | 'daily' | 'weekly';
  notification_types: {
    video_complete: boolean;
    quota_warning: boolean;
    cost_alert: boolean;
    system_updates: boolean;
    marketing: boolean;
  };
}

interface APIKey {
  id: string;
  name: string;
  key_preview: string;
  created_at: string;
  last_used?: string;
  permissions: string[];
}

const UserSettings: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [deleteAccountDialog, setDeleteAccountDialog] = useState(false);
  
  const { user, updateUser } = useAuthStore();
  const navigate = useNavigate();
  
  const { control, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      name: user?.name || '',
      email: user?.email || '',
      phone: user?.phone || '',
      company: user?.company || '',
      bio: user?.bio || '',
      timezone: user?.timezone || 'UTC',
      language: user?.language || 'en',
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
      notifications: {
        email_enabled: true,
        sms_enabled: false,
        push_enabled: true,
        email_frequency: 'instant',
        video_complete: true,
        quota_warning: true,
        cost_alert: true,
        system_updates: true,
        marketing: false,
      },
      theme: 'light',
      autoplay: true,
      quality: 'high',
    }
  });

  useEffect(() => {
    fetchUserSettings();
    fetchApiKeys();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  const fetchUserSettings = async () => {
    try {
      setLoading(true);
      const response = await api.get('/user/settings');
      reset(response.data);
    } catch (_err) {
      setError('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const fetchApiKeys = async () => {
    try {
      const response = await api.get('/user/api-keys');
      setApiKeys(response.data);
    } catch (_err) {
      console.error('Failed to fetch API keys:', err);
    }
  };

  const onSubmitProfile = async (data: unknown) => {
    try {
      setLoading(true);
      await api.patch('/user/profile', {
        name: data.name,
        phone: data.phone,
        company: data.company,
        bio: data.bio,
        timezone: data.timezone,
        language: data.language,
      });
      setSuccess('Profile updated successfully');
      updateUser(data);
    } catch (_err) {
      setError('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const onSubmitPassword = async (data: unknown) => {
    if (data.newPassword !== data.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    try {
      setLoading(true);
      await api.post('/user/change-password', {
        current_password: data.currentPassword,
        new_password: data.newPassword,
      });
      setSuccess('Password changed successfully');
      reset({ currentPassword: '', newPassword: '', confirmPassword: '' });
    } catch (_err) {
      setError('Failed to change password');
    } finally {
      setLoading(false);
    }
  };

  const onSubmitNotifications = async (data: unknown) => {
    try {
      setLoading(true);
      await api.patch('/user/notifications', data.notifications);
      setSuccess('Notification preferences updated');
    } catch (_err) {
      setError('Failed to update notifications');
    } finally {
      setLoading(false);
    }
  };

  const handleEnable2FA = async () => {
    try {
      const response = await api.post('/user/2fa/enable');
      // Show QR code dialog
      setTwoFactorEnabled(true);
      setSuccess('Two-factor authentication enabled');
    } catch (_err) {
      setError('Failed to enable 2FA');
    }
  };

  const handleDisable2FA = async () => {
    try {
      await api.post('/user/2fa/disable');
      setTwoFactorEnabled(false);
      setSuccess('Two-factor authentication disabled');
    } catch (_err) {
      setError('Failed to disable 2FA');
    }
  };

  const handleCreateApiKey = async (name: string, permissions: string[]) => {
    try {
      const response = await api.post('/user/api-keys', { name, permissions });
      setNewApiKey(response.data.key);
      fetchApiKeys();
    } catch (_err) {
      setError('Failed to create API key');
    }
  };

  const handleDeleteApiKey = async (keyId: string) => {
    try {
      await api.delete(`/user/api-keys/${keyId}`);
      fetchApiKeys();
      setSuccess('API key deleted');
    } catch (_err) {
      setError('Failed to delete API key');
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await api.delete('/user/account');
      // Logout and redirect
      navigate('/');
    } catch (_err) {
      setError('Failed to delete account');
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
        
        {success && (
          <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}
        
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper sx={{ width: '100%' }}>
          <Tabs
            value={activeTab}
            onChange={(_, value) => setActiveTab(value)}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab icon={<Person />} label="Profile" />
            <Tab icon={<Security />} label="Security" />
            <Tab icon={<Notifications />} label="Notifications" />
            <Tab icon={<Payment />} label="Billing" />
            <Tab icon={<Api />} label="API Keys" />
            <Tab icon={<Palette />} label="Appearance" />
          </Tabs>

          <TabPanel value={activeTab} index={0}>
            {/* Profile Settings */}
            <form onSubmit={handleSubmit(onSubmitProfile)}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Box textAlign="center">
                    <Avatar
                      sx={{ width: 120, height: 120, mx: 'auto', mb: 2 }}
                      src={user?.avatar}
                    >
                      {user?.name?.charAt(0)}
                    </Avatar>
                    <Button
                      variant="outlined"
                      startIcon={<CloudUpload />}
                      component="label"
                    >
                      Upload Avatar
                      <input type="file" hidden accept="image/*" />
                    </Button>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={8}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Controller
                        name="name"
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
                              ),
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