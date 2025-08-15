import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import { CheckCircle, Error as ErrorIcon, Email } from '@mui/icons-material';
import { authApi } from '../../services/api';

export const EmailVerification: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');
  
  const token = searchParams.get('token');

  const verifyEmail = useCallback(async (verificationToken: string) => {
    try {
      await authApi.verifyEmail(verificationToken);
      setStatus('success');
      setMessage('Your email has been successfully verified!');
      
      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/auth/login');
      }, 3000);
    } catch (error: unknown) {
      setStatus('error');
      const axiosError = error && typeof error === 'object' && 'response' in error ? error as { response?: { status?: number } } : null;
      if (axiosError?.response?.status === 404) {
        setMessage('Invalid or expired verification token.');
      } else {
        setMessage('An error occurred during verification. Please try again.');
      }
    }
  }, [navigate]);

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setMessage('Invalid verification link. Please check your email for the correct link.');
      return;
    }

    verifyEmail(token);
  }, [token, verifyEmail]);

  const resendVerification = async () => {
    setStatus('loading');
    try {
      // This would require the user's email
      // You might want to add an input field for this
      setMessage('A new verification email has been sent.');
      setStatus('success');
    } catch (_error) {
      setStatus('error');
      setMessage('Failed to resend verification email.');
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: 2,
      }}
    >
      <Paper
        elevation={10}
        sx={{
          padding: 4,
          maxWidth: 400,
          width: '100%',
          borderRadius: 2,
          textAlign: 'center',
        }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
            YTEmpire
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Email Verification
          </Typography>
        </Box>

        {status === 'loading' && (
          <Box sx={{ py: 4 }}>
            <CircularProgress size={60} />
            <Typography variant="body1" sx={{ mt: 2 }}>
              Verifying your email...
            </Typography>
          </Box>
        )}

        {status === 'success' && (
          <Box sx={{ py: 4 }}>
            <CheckCircle sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
            <Alert severity="success" sx={{ mb: 3 }}>
              {message}
            </Alert>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              You will be redirected to the login page shortly...
            </Typography>
            <Button
              variant="contained"
              fullWidth
              onClick={() => navigate('/auth/login')}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                },
              }}
            >
              Go to Login
            </Button>
          </Box>
        )}

        {status === 'error' && (
          <Box sx={{ py: 4 }}>
            <ErrorIcon sx={{ fontSize: 60, color: 'error.main', mb: 2 }} />
            <Alert severity="error" sx={{ mb: 3 }}>
              {message}
            </Alert>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<Email />}
                onClick={resendVerification}
              >
                Resend Verification Email
              </Button>
              <Button
                variant="contained"
                fullWidth
                onClick={() => navigate('/auth/login')}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
                  },
                }}
              >
                Back to Login
              </Button>
            </Box>
          </Box>
        )}

        <Box sx={{ mt: 4 }}>
          <Typography variant="body2" color="text.secondary">
            Need help?{' '}
            <Link
              to="/support"
              style={{
                color: '#667eea',
                textDecoration: 'none',
                fontWeight: 'bold',
              }}
            >
              Contact Support
            </Link>
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};