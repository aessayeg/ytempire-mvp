/**
 * Password Reset Component
 * Owner: React Engineer
 */

import React, { useState } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  InputAdornment,
  IconButton,
  CircularProgress,
} from '@mui/material'
import {
  Email as EmailIcon,
  Lock as LockIcon,
  Visibility,
  VisibilityOff,
  ArrowBack as ArrowBackIcon,
} from '@mui/icons-material'
import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

import { useAuth } from '@/hooks/useAuth'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

// Schema for forgot password form
const forgotPasswordSchema = z.object({
  email: z
    .string()
    .min(1, 'Email is required')
    .email('Please enter a valid email address'),
})

// Schema for reset password form
const resetPasswordSchema = z.object({
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Password must contain uppercase, lowercase, number and special character'
    ),
  confirmPassword: z.string().min(1, 'Please confirm your password'),
}).refine((data) => data.password === data.confirmPassword, {
  message: 'Passwords do not match',
  path: ['confirmPassword'],
})

type ForgotPasswordFormData = z.infer<typeof forgotPasswordSchema>
type ResetPasswordFormData = z.infer<typeof resetPasswordSchema>

export const PasswordReset: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { forgotPassword, resetPassword, isLoading, error } = useAuth()
  
  const token = searchParams.get('token')
  const email = searchParams.get('email')
  const isResetMode = Boolean(token)
  
  const [emailSent, setEmailSent] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)

  // Forgot password form
  const forgotPasswordForm = useForm<ForgotPasswordFormData>({
    resolver: zodResolver(forgotPasswordSchema),
    defaultValues: { email: email || '' },
  })

  // Reset password form
  const resetPasswordForm = useForm<ResetPasswordFormData>({
    resolver: zodResolver(resetPasswordSchema),
    defaultValues: { password: '', confirmPassword: '' },
  })

  const onForgotPasswordSubmit = async (data: ForgotPasswordFormData) => {
    try {
      await forgotPassword(data.email)
      setEmailSent(true)
    } catch (err: any) {
      forgotPasswordForm.setError('root', {
        message: err.message || 'Failed to send reset email. Please try again.',
      })
    }
  }

  const onResetPasswordSubmit = async (data: ResetPasswordFormData) => {
    if (!token) {
      resetPasswordForm.setError('root', {
        message: 'Invalid reset token. Please request a new password reset.',
      })
      return
    }

    try {
      await resetPassword(token, data.password)
      navigate('/auth/login', {
        state: { 
          message: 'Password reset successful. Please log in with your new password.' 
        },
      })
    } catch (err: any) {
      if (err.response?.status === 400) {
        resetPasswordForm.setError('root', {
          message: 'Invalid or expired reset token. Please request a new password reset.',
        })
      } else {
        resetPasswordForm.setError('root', {
          message: err.message || 'Failed to reset password. Please try again.',
        })
      }
    }
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <LoadingSpinner message={isResetMode ? "Resetting password..." : "Sending reset email..."} />
      </Box>
    )
  }

  if (emailSent) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        sx={{ bgcolor: 'background.default', p: 2 }}
      >
        <Paper
          elevation={8}
          sx={{
            p: 4,
            width: '100%',
            maxWidth: 400,
            borderRadius: 2,
            textAlign: 'center',
          }}
        >
          <Typography variant="h5" component="h1" gutterBottom color="primary">
            Check Your Email
          </Typography>
          
          <Typography variant="body1" color="text.secondary" paragraph>
            We've sent a password reset link to{' '}
            <strong>{forgotPasswordForm.getValues('email')}</strong>
          </Typography>
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Please check your email and click the link to reset your password.
            The link will expire in 1 hour.
          </Typography>

          <Button
            variant="outlined"
            onClick={() => setEmailSent(false)}
            sx={{ mt: 2, mr: 1 }}
          >
            Resend Email
          </Button>

          <Button
            variant="text"
            component={Link}
            to="/auth/login"
            startIcon={<ArrowBackIcon />}
            sx={{ mt: 2 }}
          >
            Back to Login
          </Button>
        </Paper>
      </Box>
    )
  }

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      sx={{ bgcolor: 'background.default', p: 2 }}
    >
      <Paper
        elevation={8}
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 400,
          borderRadius: 2,
        }}
      >
        <Box textAlign="center" mb={3}>
          <Typography variant="h4" component="h1" gutterBottom>
            {isResetMode ? 'Reset Password' : 'Forgot Password'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {isResetMode 
              ? 'Enter your new password below'
              : 'Enter your email address and we\'ll send you a reset link'
            }
          </Typography>
        </Box>

        {(error || forgotPasswordForm.formState.errors.root || resetPasswordForm.formState.errors.root) && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error || forgotPasswordForm.formState.errors.root?.message || resetPasswordForm.formState.errors.root?.message}
          </Alert>
        )}

        {isResetMode ? (
          <Box component="form" onSubmit={resetPasswordForm.handleSubmit(onResetPasswordSubmit)} noValidate>
            <Controller
              name="password"
              control={resetPasswordForm.control}
              render={({ field }) => (
                <TextField
                  {...field}
                  fullWidth
                  label="New Password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  autoFocus
                  error={!!resetPasswordForm.formState.errors.password}
                  helperText={resetPasswordForm.formState.errors.password?.message}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <LockIcon color="action" />
                      </InputAdornment>
                    ),
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
                  sx={{ mb: 2 }}
                />
              )}
            />

            <Controller
              name="confirmPassword"
              control={resetPasswordForm.control}
              render={({ field }) => (
                <TextField
                  {...field}
                  fullWidth
                  label="Confirm New Password"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  error={!!resetPasswordForm.formState.errors.confirmPassword}
                  helperText={resetPasswordForm.formState.errors.confirmPassword?.message}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <LockIcon color="action" />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          edge="end"
                        >
                          {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 3 }}
                />
              )}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={resetPasswordForm.formState.isSubmitting}
              sx={{ mb: 2, py: 1.5 }}
            >
              {resetPasswordForm.formState.isSubmitting ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Reset Password'
              )}
            </Button>
          </Box>
        ) : (
          <Box component="form" onSubmit={forgotPasswordForm.handleSubmit(onForgotPasswordSubmit)} noValidate>
            <Controller
              name="email"
              control={forgotPasswordForm.control}
              render={({ field }) => (
                <TextField
                  {...field}
                  fullWidth
                  label="Email Address"
                  type="email"
                  autoComplete="email"
                  autoFocus
                  error={!!forgotPasswordForm.formState.errors.email}
                  helperText={forgotPasswordForm.formState.errors.email?.message}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <EmailIcon color="action" />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ mb: 3 }}
                />
              )}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={forgotPasswordForm.formState.isSubmitting}
              sx={{ mb: 2, py: 1.5 }}
            >
              {forgotPasswordForm.formState.isSubmitting ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Send Reset Email'
              )}
            </Button>
          </Box>
        )}

        <Box textAlign="center">
          <Button
            component={Link}
            to="/auth/login"
            startIcon={<ArrowBackIcon />}
            variant="text"
            size="small"
          >
            Back to Login
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}

export default PasswordReset