/**
 * Input Component
 * Owner: React Engineer
 */

import React from 'react'
import { TextField, TextFieldProps, InputAdornment } from '@mui/material'
import { styled } from '@mui/material/styles'

interface InputProps extends Omit<TextFieldProps, 'variant'> {
  icon?: React.ReactNode
  endIcon?: React.ReactNode
}

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: theme.shape.borderRadius,
    transition: 'all 0.2s ease',
    
    '&:hover': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.primary.main,
      },
    },
    
    '&.Mui-focused': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.primary.main,
        borderWidth: 2,
      },
    },
    
    '&.Mui-error': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.error.main,
      },
    },
  },
  
  '& .MuiInputLabel-root': {
    '&.Mui-focused': {
      color: theme.palette.primary.main,
    },
  },
}))

export const Input: React.FC<InputProps> = ({
  icon,
  endIcon,
  ...props
}) => {
  return (
    <StyledTextField
      variant="outlined"
      fullWidth
      InputProps={{
        startAdornment: icon && (
          <InputAdornment position="start">{icon}</InputAdornment>
        ),
        endAdornment: endIcon && (
          <InputAdornment position="end">{endIcon}</InputAdornment>
        ),
        ...props.InputProps,
      }}
      {...props}
    />
  )
}

export default Input