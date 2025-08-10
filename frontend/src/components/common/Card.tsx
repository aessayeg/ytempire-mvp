/**
 * Card Component
 * Owner: React Engineer
 */

import React from 'react'
import { Card as MuiCard, CardContent, CardActions, CardProps as MuiCardProps } from '@mui/material'
import { styled } from '@mui/material/styles'

interface CardProps extends MuiCardProps {
  variant?: 'default' | 'outlined' | 'elevated'
  interactive?: boolean
  children: React.ReactNode
  actions?: React.ReactNode
}

const StyledCard = styled(MuiCard)<{ interactive?: boolean; customVariant?: string }>(
  ({ theme, interactive, customVariant }) => ({
    borderRadius: theme.shape.borderRadius * 1.5,
    transition: 'all 0.3s ease',
    position: 'relative',
    overflow: 'visible',
    
    ...(customVariant === 'default' && {
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      border: 'none',
    }),
    
    ...(customVariant === 'outlined' && {
      boxShadow: 'none',
      border: `1px solid ${theme.palette.divider}`,
    }),
    
    ...(customVariant === 'elevated' && {
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
      border: 'none',
    }),
    
    ...(interactive && {
      cursor: 'pointer',
      '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
      },
      '&:active': {
        transform: 'translateY(0)',
      },
    }),
  })
)

export const Card: React.FC<CardProps> = ({
  variant = 'default',
  interactive = false,
  children,
  actions,
  ...props
}) => {
  return (
    <StyledCard
      customVariant={variant}
      interactive={interactive}
      {...props}
    >
      <CardContent>{children}</CardContent>
      {actions && <CardActions>{actions}</CardActions>}
    </StyledCard>
  )
}

export default Card