/**
 * Button Component
 * Owner: React Engineer
 */

import React from 'react'
import { Button as MuiButton, ButtonProps as MuiButtonProps, CircularProgress } from '@mui/material'
import { styled } from '@mui/material/styles'

interface ButtonProps extends Omit<MuiButtonProps, 'variant'> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger'
  loading?: boolean
  icon?: React.ReactNode
}

const StyledButton = styled(MuiButton)<{ customVariant?: string }>(({ theme, customVariant }) => ({
  borderRadius: theme.shape.borderRadius,
  textTransform: 'none',
  fontWeight: 500,
  transition: 'all 0.2s ease',
  
  ...(customVariant === 'primary' && {
    background: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
    '&:hover': {
      background: theme.palette.primary.dark,
      transform: 'translateY(-1px)',
      boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
    },
  }),
  
  ...(customVariant === 'outline' && {
    background: 'transparent',
    border: `2px solid ${theme.palette.primary.main}`,
    color: theme.palette.primary.main,
    '&:hover': {
      background: theme.palette.primary.main,
      color: theme.palette.primary.contrastText,
    },
  }),
  
  ...(customVariant === 'ghost' && {
    background: 'transparent',
    color: theme.palette.text.primary,
    '&:hover': {
      background: theme.palette.action.hover,
    },
  }),
  
  ...(customVariant === 'danger' && {
    background: theme.palette.error.main,
    color: theme.palette.error.contrastText,
    '&:hover': {
      background: theme.palette.error.dark,
    },
  }),
}))

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  loading = false,
  disabled = false,
  icon,
  children,
  ...props
}) => {
  const muiVariant = variant === 'outline' ? 'outlined' : 'contained'
  
  return (
    <StyledButton
      customVariant={variant}
      variant={muiVariant}
      disabled={disabled || loading}
      startIcon={!loading && icon}
      {...props}
    >
      {loading ? (
        <CircularProgress size={20} color="inherit" />
      ) : (
        children
      )}
    </StyledButton>
  )
}

export default Button