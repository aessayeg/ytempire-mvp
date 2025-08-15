import React, { useState } from 'react';
import {  useNavigate, Link  } from 'react-router-dom';
import {  useAuthStore  } from '../../stores/authStore';
import { 
  Box,
  Button,
  TextField,
  Typography,
  Alert,
  Paper,
  InputAdornment,
  IconButton,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Checkbox,
  FormControlLabel,
  FormControl
 } from '@mui/material';
import {  Visibility,
  VisibilityOff,
  Email,
  Lock,
  Person  } from '@mui/icons-material';

const steps = ['Account Details', 'Personal Information', 'Confirmation'];

export const RegisterForm: React.FC = () => {
  const navigate = useNavigate();
  const { register, isLoading, error, clearError } = useAuthStore();
  
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({ email: '',
    username: '',
    password: '',
    confirmPassword: '',
    fullName: '',
    agreeToTerms: false });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validateStep = (step: number): boolean => {
    const errors: Record<string, string> = {};
    
    if (step === 0) {
      // Validate email and username
      if (!formData.email) {
        errors.email = 'Email is required';
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        errors.email = 'Email is invalid';
      }
      
      if (!formData.username) {
        errors.username = 'Username is required';
      } else if (formData.username.length < 3) {
        errors.username = 'Username must be at least 3 characters';
      } else if (!/^[a-zA-Z0-9 _]+$/.test(formData.username)) {
        errors.username = 'Username can only contain letters, numbers, and underscores';
      }
    } else if (step === 1) {
      // Validate password and full name
      if (!formData.password) {
        errors.password = 'Password is required';
      } else if (formData.password.length < 8) {
        errors.password = 'Password must be at least 8 characters';
      } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
        errors.password = 'Password must contain uppercase, lowercase, and number';
      }
      
      if (!formData.confirmPassword) {
        errors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        errors.confirmPassword = 'Passwords do not match';
      }
      
      if (formData.fullName && formData.fullName.length < 2) {
        errors.fullName = 'Full name must be at least 2 characters';
      }
    } else if (step === 2) {
      // Validate terms agreement
      if (!formData.agreeToTerms) {
        errors.agreeToTerms = 'You must agree to the terms and conditions';
      }
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(activeStep)) {
      setActiveStep((prevStep) => prevStep + 1)}
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1)};

  const handleSubmit = async () => {
    if (!validateStep(2)) return;
    
    try {
      await register(
        formData.email,
        formData.username,
        formData.password,
        formData.fullName || undefined
      );
      navigate('/dashboard')} catch (error) {
      console.error('Registration, failed:', error);
      setActiveStep(0); // Go back to first step on error
    }
  };

  const handleChange = (_: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'agreeToTerms' ? checked : value
    }));
    
    // Clear validation error when user starts typing
    if (validationErrors[name]) {
      setValidationErrors(prev => ({ ...prev, [name]: '' }))}
    
    // Clear auth error
    if (error) {
      clearError()}
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <>
            <TextField
              fullWidth
              margin="normal"
              name="email"
              type="email"
              label="Email Address"
              value={formData.email}
              onChange={handleChange}
              error={!!validationErrors.email}
              helperText={validationErrors.email}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <Email color="action" />
                  </InputAdornment>
                ) }}
              autoComplete="email"
              autoFocus
            />
            
            <TextField
              fullWidth
              margin="normal"
              name="username"
              label="Username"
              value={formData.username}
              onChange={handleChange}
              error={!!validationErrors.username}
              helperText={validationErrors.username}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <Person color="action" />
                  </InputAdornment>
                ) }}
              autoComplete="username"
            />
          </>
        );
        
      case 1:
        return (
          <>
            <TextField
              fullWidth
              margin="normal"
              name="fullName"
              label="Full Name (Optional)"
              value={formData.fullName}
              onChange={handleChange}
              error={!!validationErrors.fullName}
              helperText={validationErrors.fullName}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <Badge color="action" />
                  </InputAdornment>
                ) }}
              autoComplete="name"
            />
            
            <TextField
              fullWidth
              margin="normal"
              name="password"
              type={showPassword ? 'text' : 'password'}
              label="Password"
              value={formData.password}
              onChange={handleChange}
              error={!!validationErrors.password}
              helperText={validationErrors.password || 'Min 8 chars, include uppercase, lowercase, and number'}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
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
              autoComplete="new-password"
            />
            
            <TextField
              fullWidth
              margin="normal"
              name="confirmPassword"
              type={showConfirmPassword ? 'text' : 'password'}
              label="Confirm Password"
              value={formData.confirmPassword}
              onChange={handleChange}
              error={!!validationErrors.confirmPassword}
              helperText={validationErrors.confirmPassword}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle confirm password visibility"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword}
                      edge="end"
                    >
                      {showConfirmPassword ? <VisibilityOff /> </>: <Visibility />}
                    </IconButton>
                  </InputAdornment>
                )
              }}
              autoComplete="new-password"
            />
          </>
        );
        
      case 2:
        return (
    <>
      <Box sx={{ mt: 2 }}>
            <Paper sx={{ p: 3, mb: 2, bgcolor: 'grey.50' }}>
              <Typography variant="h6" gutterBottom>
                Account Summary
              </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>Email:</strong> {formData.email}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>Username:</strong> {formData.username}
              </Typography>
              {formData.fullName && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  <strong>Full, Name:</strong> {formData.fullName}
                </Typography>
              )}
            </Paper>
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={formData.agreeToTerms}
                  onChange={handleChange}
                  name="agreeToTerms"
                  color="primary"
                />
              }
              label={
                <Typography variant="body2">
                  I agree to the{' '}
                  <Link to="/terms" style={{ color: '#667 eea' }}>
                    Terms and Conditions
                  </Link>
                  {' '}and{' '}
                  <Link to="/privacy" style={{ color: '#667 eea' }}>
                    Privacy Policy
                  </Link>
                </Typography>
              }
            />
            {validationErrors.agreeToTerms && (
              <Typography color="error" variant="caption" display="block" sx={{ mt: 1 }}>
                {validationErrors.agreeToTerms}
              </Typography>
            )}
          </Box>
        </>
  );
        
        return null}
  };

  return (
    <>
      <Box
      sx={ {
        minHeight: '100 vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135 deg, #667 eea 0%, #764 ba2 100%)',
        padding: 2 }}
    >
      <Paper
        elevation={10}
        sx={ {
          padding: 4,
          maxWidth: 500,
          width: '100%',
          borderRadius: 2 }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
            YTEmpire
          </Typography>
      <Typography variant="subtitle1" color="text.secondary">
            Create your account
          </Typography>
        </Box>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={clearError}>
            {error}
          </Alert>
        )}
        <Box>
          {renderStepContent(activeStep)}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ mr: 1 }}
            >
              Back
            </Button>
            
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={isLoading}
                sx={ {
                  background: 'linear-gradient(135 deg, #667 eea 0%, #764 ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135 deg, #5 a6 fd8 0%, #6 a4290 100%)' }
                }}
              >
                {isLoading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Create Account'
                )}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
                sx={ {
                  background: 'linear-gradient(135 deg, #667 eea 0%, #764 ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135 deg, #5 a6 fd8 0%, #6 a4290 100%)' }
                }}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>

        <Box sx={{ textAlign: 'center', mt: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Already have an account?{' '}
            <Link
              to="/auth/login"
              style={ {
                color: '#667 eea',
                textDecoration: 'none',
                fontWeight: 'bold' }}
            >
              Sign In
            </Link>
          </Typography>
        </Box>
      </Paper>
    </Box>
  </>
  )};