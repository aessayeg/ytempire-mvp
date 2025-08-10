/**
 * Login Screen Component
 * Production-ready login form with Material-UI styling
 */
import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink, useLocation } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  IconButton,
  InputAdornment,
  Divider,
  Link,
  CircularProgress,
  Container,
  Paper,
  Grid,
  useTheme,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Email,
  Lock,
  Login as LoginIcon,
  Google,
  GitHub,
  YouTube,
  TrendingUp,
  VideoLibrary,
  AutoAwesome,
} from '@mui/icons-material';
import { useAuthStore } from '../../stores/authStore';

interface LocationState {
  from?: {
    pathname: string;
  };
  message?: string;
}

export const Login: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as LocationState;

  const { login, isLoading, error, clearError, isAuthenticated } = useAuthStore();

  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [formErrors, setFormErrors] = useState<{[key: string]: string}>({});

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      const from = state?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, state]);

  // Clear errors when component mounts
  useEffect(() => {
    clearError();
  }, [clearError]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    
    // Clear field error when user starts typing
    if (formErrors[name]) {
      setFormErrors(prev => ({
        ...prev,
        [name]: '',
      }));
    }
  };

  const validateForm = () => {
    const errors: {[key: string]: string} = {};

    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      errors.password = 'Password must be at least 6 characters';
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await login(formData.email, formData.password);
      // Navigation will be handled by useEffect when isAuthenticated changes
    } catch (error) {
      // Error is handled by the store
    }
  };

  const handleTogglePassword = () => {
    setShowPassword(prev => !prev);
  };

  // Mock social login handlers
  const handleGoogleLogin = () => {
    // TODO: Implement Google OAuth
    console.log('Google login not implemented yet');
  };

  const handleGitHubLogin = () => {
    // TODO: Implement GitHub OAuth
    console.log('GitHub login not implemented yet');
  };

  const features = [
    {
      icon: <VideoLibrary color="primary" />,
      title: "Automated Video Creation",
      description: "AI-powered content generation for YouTube channels"
    },
    {
      icon: <TrendingUp color="success" />,
      title: "Performance Analytics",
      description: "Track views, engagement, and revenue in real-time"
    },
    {
      icon: <AutoAwesome color="warning" />,
      title: "Smart Optimization",
      description: "AI-driven thumbnail and title optimization"
    }
  ];

  return (
    <Container maxWidth="lg" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center' }}>
      <Grid container spacing={4} sx={{ width: '100%' }}>
        {/* Left side - Branding and Features */}
        <Grid item xs={12} md={6}>
          <Box sx={{ pr: { md: 4 } }}>
            <Box sx={{ mb: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <YouTube sx={{ fontSize: 40, color: '#FF0000', mr: 2 }} />
                <Typography variant="h3" fontWeight="bold" color="primary">
                  YTEmpire
                </Typography>
              </Box>
              <Typography variant="h5" color="text.secondary" gutterBottom>
                Scale Your YouTube Success with AI
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                The all-in-one platform for automated YouTube content creation, 
                analytics, and channel management. Join thousands of creators 
                who've scaled their channels with our AI-powered tools.
              </Typography>
            </Box>

            {/* Feature highlights */}
            <Box sx={{ display: { xs: 'none', md: 'block' } }}>
              {features.map((feature, index) => (
                <Box key={index} sx={{ display: 'flex', mb: 3, alignItems: 'flex-start' }}>
                  <Box sx={{ mr: 2, mt: 0.5 }}>
                    {feature.icon}
                  </Box>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {feature.description}
                    </Typography>
                  </Box>
                </Box>
              ))}
            </Box>
          </Box>
        </Grid>

        {/* Right side - Login Form */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={8}
            sx={{ 
              p: 4, 
              borderRadius: 3,
              background: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(255, 255, 255, 0.9)',
              backdropFilter: 'blur(10px)',
            }}
          >
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <LoginIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                Welcome Back
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Sign in to your YTEmpire account
              </Typography>
            </Box>

            {/* Redirect message */}
            {state?.message && (
              <Alert severity="info" sx={{ mb: 3 }}>
                {state.message}
              </Alert>
            )}

            {/* Error Alert */}
            {error && (
              <Alert 
                severity="error" 
                sx={{ mb: 3 }}
                onClose={clearError}
              >
                {error}
              </Alert>
            )}

            {/* Login Form */}
            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Email Address"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleInputChange}
                error={!!formErrors.email}
                helperText={formErrors.email}
                margin="normal"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Email color="action" />
                    </InputAdornment>
                  ),
                }}
                disabled={isLoading}
                autoComplete="email"
                autoFocus
              />

              <TextField
                fullWidth
                label="Password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleInputChange}
                error={!!formErrors.password}
                helperText={formErrors.password}
                margin="normal"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Lock color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={handleTogglePassword}
                        edge="end"
                        disabled={isLoading}
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                disabled={isLoading}
                autoComplete="current-password"
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                disabled={isLoading}
                sx={{ 
                  mt: 3, 
                  mb: 2,
                  py: 1.5,
                  borderRadius: 2,
                  textTransform: 'none',
                  fontSize: '1.1rem',
                  fontWeight: 600,
                }}
                startIcon={isLoading ? <CircularProgress size={20} /> : <LoginIcon />}
              >
                {isLoading ? 'Signing In...' : 'Sign In'}
              </Button>
            </form>

            <Box sx={{ textAlign: 'center', mt: 2 }}>
              <Link
                component={RouterLink}
                to="/forgot-password"
                variant="body2"
                sx={{ textDecoration: 'none' }}
              >
                Forgot your password?
              </Link>
            </Box>

            <Divider sx={{ my: 3 }}>
              <Typography variant="body2" color="text.secondary">
                or continue with
              </Typography>
            </Divider>

            {/* Social Login Buttons */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Google />}
                onClick={handleGoogleLogin}
                disabled={isLoading}
                sx={{ textTransform: 'none' }}
              >
                Google
              </Button>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<GitHub />}
                onClick={handleGitHubLogin}
                disabled={isLoading}
                sx={{ textTransform: 'none' }}
              >
                GitHub
              </Button>
            </Box>

            <Divider />

            <Box sx={{ textAlign: 'center', mt: 3 }}>
              <Typography variant="body2" color="text.secondary">
                Don't have an account?{' '}
                <Link
                  component={RouterLink}
                  to="/register"
                  variant="body2"
                  sx={{ fontWeight: 600, textDecoration: 'none' }}
                >
                  Sign up for free
                </Link>
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Login;