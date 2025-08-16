import React, { useState } from 'react';
import {  useNavigate, Link  } from 'react-router-dom';
import { 
  Box,
  Button,
  TextField,
  Typography,
  Alert,
  Paper,
  InputAdornment,
  CircularProgress,
  Stepper,
  Step,
  StepLabel
 } from '@mui/material';
import { 
  Email,
  Lock,
  VpnKey,
  ArrowBack,
  CheckCircle
 } from '@mui/icons-material';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const steps = ['Enter Email', 'Verify Code', 'Reset Password'];

export const ForgotPasswordForm: React.FC = () => { const navigate = useNavigate();
  
  const [activeStep, setActiveStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [formData, setFormData] = useState({
    email: '',
    resetCode: '',
    newPassword: '',
    confirmPassword: '' });
  
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const validateStep = (step: number): boolean => {
    const errors: Record<string, string>  = {};
    if (step === 0) {
      // Validate email
      if (!formData.email) {
        errors.email = 'Email is required'
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        errors.email = 'Email is invalid'
      }
    } else if (step === 1) {
      // Validate reset code
      if (!formData.resetCode) {
        errors.resetCode = 'Reset code is required'
      } else if (formData.resetCode.length !== 6) {
        errors.resetCode = 'Reset code must be 6 digits'
      }
    } else if (step === 2) {
      // Validate new password
      if (!formData.newPassword) {
        errors.newPassword = 'Password is required'
      } else if (formData.newPassword.length < 8) {
        errors.newPassword = 'Password must be at least 8 characters'
      } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.newPassword)) {
        errors.newPassword = 'Password must contain uppercase, lowercase, and number'
      }
      
      if (!formData.confirmPassword) {
        errors.confirmPassword = 'Please confirm your password'
      } else if (formData.newPassword !== formData.confirmPassword) {
        errors.confirmPassword = 'Passwords do not match'
      }
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0
  };
  const handleSendResetCode = async () => {
    if (!validateStep(0)) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      await axios.post(`${API_URL}/api/v1/auth/forgot-password`, { email: formData.email });
      
      setSuccess('Reset code sent to your email!');
      setActiveStep(1)} catch (_: unknown) {
      const axiosError = error && typeof error === 'object' && 'response' in error ? error as { response?: { data?: { detail?: string } } } : null;
      setError(axiosError?.response?.data?.detail || 'Failed to send reset code')} finally {
      setIsLoading(false)}};
  const handleVerifyCode = async () => {
    if (!validateStep(1)) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      await axios.post(`${API_URL}/api/v1/auth/verify-reset-code`, { email: formData.email;
        code: formData.resetCode });
      
      setSuccess('Code verified successfully!');
      setActiveStep(2)} catch (_: unknown) {
      const axiosError = error && typeof error === 'object' && 'response' in error ? error as { response?: { data?: { detail?: string } } } : null;
      setError(axiosError?.response?.data?.detail || 'Invalid or expired reset code')} finally {
      setIsLoading(false)}};
  const handleResetPassword = async () => {
    if (!validateStep(2)) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      await axios.post(`${API_URL}/api/v1/auth/reset-password`, { email: formData.email;
        code: formData.resetCode;
        new_password: formData.newPassword });
      
      setSuccess('Password reset successfully! Redirecting to login...');
      setTimeout(() => {
        navigate('/auth/login')}, 2000)} catch (_: unknown) {
      const axiosError = error && typeof error === 'object' && 'response' in error ? error as { response?: { data?: { detail?: string } } } : null;
      setError(axiosError?.response?.data?.detail || 'Failed to reset password')} finally {
      setIsLoading(false)}
  };
  const handleChange = (_: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear validation error when user starts typing
    if (validationErrors[name]) {
      setValidationErrors(prev => ({ ...prev, [name]: '' }))}
    
    // Clear error/success messages
    if (error) setError(null);
    if (success) setSuccess(null)
};
  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
    <>
      <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Enter your email address and we'll send you a code to reset your password.
            </Typography>
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
            
            <Button
              fullWidth
              variant="contained"
              onClick={handleSendResetCode}
              disabled={isLoading}
              sx={ {
                mt: 3,
                background: 'linear-gradient(135 deg, #667eea0%, #764ba2100%)',
                '&:hover': {
                  background: 'linear-gradient(135 deg, #5a6fd80%, #6 a4290 100%)' }
              }}
            >
              {isLoading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Send Reset Code'
              )}
            </Button>
          </Box>
        </>
  );
        
      case 1:
        return (
    <>
      <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Enter the 6-digit code sent to {formData.email}
            </Typography>
      <TextField
              fullWidth
              margin="normal"
              name="resetCode"
              label="Reset Code"
              value={formData.resetCode}
              onChange={handleChange}
              error={!!validationErrors.resetCode}
              helperText={validationErrors.resetCode}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <VpnKey color="action" />
                  </InputAdornment>
                ) }}
              inputProps={ {
                maxLength: 6,
                pattern: '[0-9]*' }}
              autoFocus
            />
            
            <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
              <Button
                variant="outlined"
                onClick={() => setActiveStep(0}
                startIcon={<ArrowBack />}
              >
                Back
              </Button>
              
              <Button
                fullWidth
                variant="contained"
                onClick={handleVerifyCode}
                disabled={isLoading}
                sx={ {
                  background: 'linear-gradient(135 deg, #667eea0%, #764ba2100%)',
                  '&:hover': {
                    background: 'linear-gradient(135 deg, #5a6fd80%, #6 a4290 100%)' }
                }}
              >
                {isLoading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Verify Code'
                )}
              </Button>
            </Box>
            
            <Button
              fullWidth
              variant="text"
              onClick={handleSendResetCode}
              disabled={isLoading}
              sx={{ mt: 2 }}
            >
              Resend Code
            </Button>
          </Box>
        </>
  );
        
      case 2:
        return (
    <>
      <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Create a new password for your account
            </Typography>
      <TextField
              fullWidth
              margin="normal"
              name="newPassword"
              type="password"
              label="New Password"
              value={formData.newPassword}
              onChange={handleChange}
              error={!!validationErrors.newPassword}
              helperText={validationErrors.newPassword || 'Min 8, chars, include, uppercase, lowercase, and number'}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ) }}
              autoComplete="new-password"
              autoFocus
            />
            
            <TextField
              fullWidth
              margin="normal"
              name="confirmPassword"
              type="password"
              label="Confirm New Password"
              value={formData.confirmPassword}
              onChange={handleChange}
              error={!!validationErrors.confirmPassword}
              helperText={validationErrors.confirmPassword}
              InputProps={ {
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ) }}
              autoComplete="new-password"
            />
            
            <Button
              fullWidth
              variant="contained"
              onClick={handleResetPassword}
              disabled={isLoading}
              sx={ {
                mt: 3,
                background: 'linear-gradient(135 deg, #667eea0%, #764ba2100%)',
                '&:hover': {
                  background: 'linear-gradient(135 deg, #5a6fd80%, #6 a4290 100%)' }}}
            >
              {isLoading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Reset Password'
              )}
            </Button>
          </Box>
        </>
  );
        
        return null}};
  return (
    <>
      <Box
      sx={ {
        minHeight: '100 vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135 deg, #667eea0%, #764ba2100%)',
        padding: 2 }}
    >
      <Paper
        elevation={10}
        sx={ {
          padding: 4,
          maxWidth: 450,
          width: '100%',
          borderRadius: 2 }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
            YTEmpire
          </Typography>
      <Typography variant="h6" color="text.secondary">
            Reset Password
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
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert 
            severity="success" 
            sx={{ mb: 2 }} 
            icon={<CheckCircle />}
            onClose={() => setSuccess(null}
          >
            {success}
          </Alert>
        )}
        {renderStepContent(activeStep)}
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Typography variant="body2" color="text.secondary">
            Remember your password?{' '}
            <Link
              to="/auth/login"
              style={ {
                color: '#667eea',
                textDecoration: 'none',
                fontWeight: 'bold' }}
            >
              Back to Login
            </Link>
          </Typography>
        </Box>
      </Paper>
    </Box>
  </>
  )};
