/**
 * Register Screen Component
 * Production-ready registration form with Material-UI styling
 */
import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
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
  Checkbox,
  FormControlLabel,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Email,
  Lock,
  Person,
  PersonAdd,
  Google,
  GitHub,
  YouTube,
  Check,
  Close,
  Security,
  Stars,
  Rocket,
} from '@mui/icons-material';
import { useAuthStore } from '../../stores/authStore';

interface PasswordStrength {
  score: number;
  feedback: string[];
  color: 'error' | 'warning' | 'info' | 'success';
}

export const Register: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { register, isLoading, error, clearError, isAuthenticated } = useAuthStore();

  const [formData, setFormData] = useState({
    email: '',
    username: '',
    full_name: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formErrors, setFormErrors] = useState<{[key: string]: string}>({});
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [acceptedPrivacy, setAcceptedPrivacy] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState<PasswordStrength>({
    score: 0,
    feedback: [],
    color: 'error',
  });

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Clear errors when component mounts
  useEffect(() => {
    clearError();
  }, [clearError]);

  // Password strength checker
  const checkPasswordStrength = (password: string): PasswordStrength => {
    let score = 0;
    const feedback: string[] = [];

    if (password.length >= 8) score += 1;
    else feedback.push('At least 8 characters');

    if (/[a-z]/.test(password)) score += 1;
    else feedback.push('Lowercase letter');

    if (/[A-Z]/.test(password)) score += 1;
    else feedback.push('Uppercase letter');

    if (/\d/.test(password)) score += 1;
    else feedback.push('Number');

    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) score += 1;
    else feedback.push('Special character');

    let color: 'error' | 'warning' | 'info' | 'success' = 'error';
    if (score >= 4) color = 'success';
    else if (score >= 3) color = 'info';
    else if (score >= 2) color = 'warning';

    return { score, feedback, color };
  };

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

    // Check password strength
    if (name === 'password') {
      setPasswordStrength(checkPasswordStrength(value));
    }
  };

  const validateForm = () => {
    const errors: {[key: string]: string} = {};

    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }

    if (!formData.username) {
      errors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      errors.username = 'Username must be at least 3 characters';
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
      errors.username = 'Username can only contain letters, numbers, and underscores';
    }

    if (!formData.full_name) {
      errors.full_name = 'Full name is required';
    } else if (formData.full_name.length < 2) {
      errors.full_name = 'Full name must be at least 2 characters';
    }

    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (passwordStrength.score < 3) {
      errors.password = 'Password is too weak. Please include more complexity.';
    }

    if (!formData.confirmPassword) {
      errors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }

    if (!acceptedTerms) {
      errors.terms = 'You must accept the terms of service';
    }

    if (!acceptedPrivacy) {
      errors.privacy = 'You must accept the privacy policy';
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
      await register(formData.email, formData.username, formData.password, formData.full_name);
      // Navigation will be handled by useEffect when isAuthenticated changes
    } catch (error) {
      // Error is handled by the store
    }
  };

  const handleTogglePassword = () => {
    setShowPassword(prev => !prev);
  };

  const handleToggleConfirmPassword = () => {
    setShowConfirmPassword(prev => !prev);
  };

  // Mock social login handlers
  const handleGoogleLogin = () => {
    console.log('Google login not implemented yet');
  };

  const handleGitHubLogin = () => {
    console.log('GitHub login not implemented yet');
  };

  const subscriptionTiers = [
    {
      name: 'Starter',
      price: 'Free',
      features: ['5 videos/month', '1 YouTube channel', 'Basic analytics', 'Community support'],
      icon: <Rocket color="primary" />,
      popular: false,
    },
    {
      name: 'Creator',
      price: '$29/month',
      features: ['100 videos/month', '5 YouTube channels', 'Advanced analytics', 'Priority support'],
      icon: <Stars color="warning" />,
      popular: true,
    },
    {
      name: 'Enterprise',
      price: '$99/month',
      features: ['Unlimited videos', 'Unlimited channels', 'White-label solution', '24/7 support'],
      icon: <Security color="success" />,
      popular: false,
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', py: 4 }}>
      <Grid container spacing={4} sx={{ width: '100%' }}>
        {/* Left side - Branding and Pricing */}
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
                Join the Creator Revolution
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Start with our free plan and scale as you grow. 
                No credit card required to get started.
              </Typography>
            </Box>

            {/* Subscription tiers preview */}
            <Box sx={{ display: { xs: 'none', md: 'block' } }}>
              <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                Choose Your Plan
              </Typography>
              {subscriptionTiers.map((tier, index) => (
                <Card 
                  key={index} 
                  variant="outlined" 
                  sx={{ 
                    mb: 2, 
                    position: 'relative',
                    border: tier.popular ? '2px solid' : '1px solid',
                    borderColor: tier.popular ? 'primary.main' : 'divider',
                  }}
                >
                  {tier.popular && (
                    <Chip
                      label="Most Popular"
                      color="primary"
                      size="small"
                      sx={{
                        position: 'absolute',
                        top: -10,
                        right: 16,
                        zIndex: 1,
                      }}
                    />
                  )}
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      {tier.icon}
                      <Typography variant="h6" sx={{ ml: 1, mr: 'auto' }}>
                        {tier.name}
                      </Typography>
                      <Typography variant="h6" color="primary" fontWeight="bold">
                        {tier.price}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {tier.features.slice(0, 2).map((feature, idx) => (
                        <Chip key={idx} label={feature} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Box>
        </Grid>

        {/* Right side - Registration Form */}
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
              <PersonAdd sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                Create Account
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Start your YouTube automation journey
              </Typography>
            </Box>

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

            {/* Registration Form */}
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Full Name"
                    name="full_name"
                    value={formData.full_name}
                    onChange={handleInputChange}
                    error={!!formErrors.full_name}
                    helperText={formErrors.full_name}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Person color="action" />
                        </InputAdornment>
                      ),
                    }}
                    disabled={isLoading}
                    autoComplete="name"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Username"
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    error={!!formErrors.username}
                    helperText={formErrors.username || 'This will be your unique identifier'}
                    disabled={isLoading}
                    autoComplete="username"
                  />
                </Grid>
              </Grid>

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
                autoComplete="new-password"
              />

              {/* Password Strength Indicator */}
              {formData.password && (
                <Box sx={{ mt: 1, mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ mr: 1 }}>
                      Password Strength:
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(passwordStrength.score / 5) * 100}
                      color={passwordStrength.color}
                      sx={{ flexGrow: 1, mr: 1 }}
                    />
                    <Typography variant="body2" color={`${passwordStrength.color}.main`}>
                      {passwordStrength.score < 2 ? 'Weak' :
                       passwordStrength.score < 4 ? 'Medium' : 'Strong'}
                    </Typography>
                  </Box>
                  {passwordStrength.feedback.length > 0 && (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {passwordStrength.feedback.map((item, idx) => (
                        <Chip
                          key={idx}
                          label={item}
                          size="small"
                          icon={<Close />}
                          variant="outlined"
                          color="error"
                        />
                      ))}
                    </Box>
                  )}
                </Box>
              )}

              <TextField
                fullWidth
                label="Confirm Password"
                name="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={handleInputChange}
                error={!!formErrors.confirmPassword}
                helperText={formErrors.confirmPassword}
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
                        onClick={handleToggleConfirmPassword}
                        edge="end"
                        disabled={isLoading}
                      >
                        {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                disabled={isLoading}
                autoComplete="new-password"
              />

              {/* Terms and Privacy checkboxes */}
              <Box sx={{ mt: 2 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={acceptedTerms}
                      onChange={(e) => setAcceptedTerms(e.target.checked)}
                      disabled={isLoading}
                    />
                  }
                  label={
                    <Typography variant="body2">
                      I agree to the{' '}
                      <Link href="/terms" target="_blank" rel="noopener">
                        Terms of Service
                      </Link>
                    </Typography>
                  }
                />
                {formErrors.terms && (
                  <Typography variant="body2" color="error" sx={{ ml: 4 }}>
                    {formErrors.terms}
                  </Typography>
                )}
              </Box>

              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={acceptedPrivacy}
                      onChange={(e) => setAcceptedPrivacy(e.target.checked)}
                      disabled={isLoading}
                    />
                  }
                  label={
                    <Typography variant="body2">
                      I agree to the{' '}
                      <Link href="/privacy" target="_blank" rel="noopener">
                        Privacy Policy
                      </Link>
                    </Typography>
                  }
                />
                {formErrors.privacy && (
                  <Typography variant="body2" color="error" sx={{ ml: 4 }}>
                    {formErrors.privacy}
                  </Typography>
                )}
              </Box>

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
                startIcon={isLoading ? <CircularProgress size={20} /> : <PersonAdd />}
              >
                {isLoading ? 'Creating Account...' : 'Create Account'}
              </Button>
            </form>

            <Divider sx={{ my: 3 }}>
              <Typography variant="body2" color="text.secondary">
                or sign up with
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
                Already have an account?{' '}
                <Link
                  component={RouterLink}
                  to="/login"
                  variant="body2"
                  sx={{ fontWeight: 600, textDecoration: 'none' }}
                >
                  Sign in instead
                </Link>
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Register;