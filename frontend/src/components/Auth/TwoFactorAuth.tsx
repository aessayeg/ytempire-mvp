import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Link as MuiLink,
} from '@mui/material';
import { Shield, ArrowBack } from '@mui/icons-material';
import { useAuthStore } from '../../stores/authStore';

interface TwoFactorAuthProps {
  onSuccess?: () => void;
  onCancel?: () => void;
  email?: string;
}

export const TwoFactorAuth: React.FC<TwoFactorAuthProps> = ({
  onSuccess,
  onCancel,
  email,
}) => {
  const { verifyTwoFactorCode, isLoading, error, clearError } = useAuthStore();
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [resendTimer, setResendTimer] = useState(30);
  const [canResend, setCanResend] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => {
    // Focus first input on mount
    inputRefs.current[0]?.focus();
  }, []) // eslint-disable-line react-hooks/exhaustive-deps;

  useEffect(() => {
    // Countdown timer for resend
    if (resendTimer > 0) {
      const timer = setTimeout(() => setResendTimer(resendTimer - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      setCanResend(true);
    }
  }, [resendTimer]);

  const handleCodeChange = (index: number, value: string) => {
    // Only allow numbers
    if (value && !/^\d+$/.test(value)) return;

    const newCode = [...code];
    newCode[index] = value;
    setCode(newCode);

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // Clear error when user types
    if (error) {
      clearError();
    }

    // Auto-submit when all digits are entered
    if (newCode.every(digit => digit !== '') && index === 5) {
      handleSubmit(newCode.join(''));
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    // Handle backspace
    if (e.key === 'Backspace' && !code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }

    // Handle paste
    if (e.key === 'v' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      navigator.clipboard.readText().then(text => {
        const pastedCode = text.replace(/\D/g, '').slice(0, 6);
        const newCode = [...code];
        for (let i = 0; i < pastedCode.length; i++) {
          newCode[i] = pastedCode[i];
        }
        setCode(newCode);
        
        // Focus last filled input or last input if all filled
        const lastFilledIndex = Math.min(pastedCode.length - 1, 5);
        inputRefs.current[lastFilledIndex]?.focus();
        
        // Auto-submit if complete
        if (pastedCode.length === 6) {
          handleSubmit(pastedCode);
        }
      });
    }
  };

  const handleSubmit = async (codeString?: string) => {
    const verificationCode = codeString || code.join('');
    
    if (verificationCode.length !== 6) {
      return;
    }

    try {
      const success = await verifyTwoFactorCode(verificationCode);
      if (success && onSuccess) {
        onSuccess();
      }
    } catch (_error) {
      console.error('2FA verification failed:', error);
    }
  };

  const handleResendCode = async () => {
    if (!canResend) return;
    
    // Call resend API here
    setResendTimer(30);
    setCanResend(false);
    
    // Show success message
    // You can implement a toast notification here
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
        }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Box
            sx={{
              width: 64,
              height: 64,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
            }}
          >
            <Shield sx={{ color: 'white', fontSize: 32 }} />
          </Box>
          
          <Typography variant="h5" component="h1" gutterBottom fontWeight="bold">
            Two-Factor Authentication
          </Typography>
          
          <Typography variant="body2" color="text.secondary">
            Enter the 6-digit code sent to
          </Typography>
          
          <Typography variant="body2" fontWeight="bold" color="primary">
            {email || 'your email'}
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={clearError}>
            {error}
          </Alert>
        )}

        <Box
          sx={{
            display: 'flex',
            gap: 1,
            justifyContent: 'center',
            mb: 3,
          }}
        >
          {code.map((digit, index) => (
            <TextField
              key={index}
              inputRef={el => inputRefs.current[index] = el}
              value={digit}
              onChange={(e) => handleCodeChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
              inputProps={{
                maxLength: 1,
                style: {
                  textAlign: 'center',
                  fontSize: '24px',
                  fontWeight: 'bold',
                },
              }}
              sx={{
                width: 50,
                '& .MuiOutlinedInput-root': {
                  '&.Mui-focused fieldset': {
                    borderColor: '#667eea',
                  },
                },
              }}
              disabled={isLoading}
            />
          ))}
        </Box>

        <Button
          fullWidth
          variant="contained"
          size="large"
          onClick={() => handleSubmit()}
          disabled={isLoading || code.some(digit => !digit)}
          sx={{
            mb: 2,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4290 100%)',
            },
          }}
        >
          {isLoading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Verify Code'
          )}
        </Button>

        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Didn't receive the code?{' '}
            {canResend ? (
              <MuiLink
                component="button"
                variant="body2"
                onClick={handleResendCode}
                sx={{
                  color: '#667eea',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                }}
              >
                Resend Code
              </MuiLink>
            ) : (
              <span>Resend in {resendTimer}s</span>
            )}
          </Typography>
        </Box>

        {onCancel && (
          <Button
            fullWidth
            variant="text"
            startIcon={<ArrowBack />}
            onClick={onCancel}
            sx={{ color: 'text.secondary' }}
          >
            Back to Login
          </Button>
        )}
      </Paper>
    </Box>
  );
};