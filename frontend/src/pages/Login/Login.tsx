import React, { useState } from 'react';
import {  useNavigate, Link  } from 'react-router-dom';
import { 
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  IconButton,
  InputAdornment,
  Divider,
  Checkbox,
  FormControlLabel,
  Grid,
  FormControl
 } from '@mui/material';
import { 
  Visibility,
  VisibilityOff,
  Google as GoogleIcon,
  YouTube as YouTubeIcon,
  Email as EmailIcon
 } from '@mui/icons-material';
import {  useAuthStore  } from '../../stores/authStore';
import {  authService  } from '../../services/authService';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuthStore();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false
});
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (_: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'rememberMe' ? checked : value
    }));
    setError('')};

  const handleSubmit = async (_: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await authService.login(formData.email, formData.password);
      login(response.user, response.token, formData.rememberMe);
      navigate('/dashboard')} catch (_err: unknown) {
      setError(err.response?.data?.message || 'Invalid email or password')} finally {
      setIsLoading(false)}
  };

  const handleGoogleLogin = async () => {
    try {
      window.location.href = '/api/v1/auth/google';
    } catch (error) {
      setError('Google login failed. Please try again.')}
  };

  const handleYouTubeLogin = async () => {
    try {
      window.location.href = '/api/v1/auth/youtube';
    } catch (error) {
      setError('YouTube login failed. Please try again.')}
  };

  return (
    <>
      <Container component="main" maxWidth="xs">
      <Box
        sx={ {
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center' }}
      >
        <Paper elevation={3} sx={{ padding: 4, width: '100%' }}>
          {/* Logo and Title */}
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <YouTubeIcon sx={{ fontSize: 48, color: 'error.main', mb: 1 }} />
            <Typography component="h1" variant="h4" fontWeight="bold">
              YTEmpire
            </Typography>
      <Typography variant="body2" color="text.secondary" mt={1}>
              Sign in to your account
            </Typography>
          </Box>

          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(''}>
              {error}
            </Alert>
          )}
          {/* Login Form */}
          <Box component="form" onSubmit={handleSubmit} noValidate>
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              autoFocus
              value={formData.email}
              onChange={handleChange}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <EmailIcon color="action" />
                  </InputAdornment>
                ) }}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              id="password"
              autoComplete="current-password"
              value={formData.password}
              onChange={handleChange}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <LockIcon color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={() => setShowPassword(!showPassword}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> </>: <Visibility />}
                    </IconButton>
                  </InputAdornment>
                )
              }}
            />

            {/* Remember Me and Forgot Password */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1, mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    name="rememberMe"
                    color="primary"
                    checked={formData.rememberMe}
                    onChange={handleChange}
                  />
                }
                label="Remember me"
              />
              <Link to="/forgot-password" style={{ textDecoration: 'none' }}>
                <Typography variant="body2" color="primary" sx={{ cursor: 'pointer' }}>
                  Forgot password?
                </Typography>
              </Link>
            </Box>

            {/* Submit Button */}
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 2, mb: 2, py: 1.5 }}
              disabled={isLoading || !formData.email || !formData.password}
            >
              {isLoading ? <CircularProgress size={24} /> : 'Sign In'}
            </Button>

            {/* Social Login Divider */}
            <Divider sx={{ my: 3 }}>OR</Divider>

            {/* Social Login Buttons */}
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<GoogleIcon />}
                  onClick={handleGoogleLogin}
                  sx={{ py: 1.5 }}
                >
                  Continue with Google
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<YouTubeIcon />}
                  onClick={handleYouTubeLogin}
                  sx={{ 
                    py: 1.5,
                    borderColor: 'error.main',
                    color: 'error.main',
                    '&:hover': {
                      borderColor: 'error.dark',
                      backgroundColor: 'error.light',
                      opacity: 0.1,

                    }
                  }}
                >
                  Continue with YouTube
                </Button>
              </Grid>
            </Grid>

            {/* Sign Up Link */}
            <Box sx={{ textAlign: 'center', mt: 3 }}>
              <Typography variant="body2" color="text.secondary">
                Don't have an account?{' '}
                <Link to="/register" style={{ textDecoration: 'none' }}>
                  <Typography component="span" variant="body2" color="primary" sx={{ fontWeight: 'bold' }}>
                    Sign up
                  </Typography>
                </Link>
              </Typography>
            </Box>
          </Box>
        </Paper>

        {/* Footer */}
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            By signing in, you agree to our{' '}
            <Link to="/terms" style={{ textDecoration: 'none', color: 'inherit' }}>
              <Typography component="span" variant="body2" sx={{ textDecoration: 'underline' }}>
                Terms of Service
              </Typography>
            </Link>
            {' and '}
            <Link to="/privacy" style={{ textDecoration: 'none', color: 'inherit' }}>
              <Typography component="span" variant="body2" sx={{ textDecoration: 'underline' }}>
                Privacy Policy
              </Typography>
            </Link>
          </Typography>
        </Box>
      </Box>
    </Container>
  </>
  )};

export default Login;