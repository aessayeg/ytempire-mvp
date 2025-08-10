/**
 * Loading Spinner Component
 * Owner: React Engineer
 */

import React from 'react'
import { Box, CircularProgress, Typography } from '@mui/material'
import { styled } from '@mui/material/styles'

interface LoadingSpinnerProps {
  size?: number
  message?: string
  fullScreen?: boolean
  overlay?: boolean
}

const FullScreenContainer = styled(Box)(({ theme }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: theme.palette.background.default,
  zIndex: theme.zIndex.modal,
}))

const OverlayContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  zIndex: theme.zIndex.modal - 1,
}))

const InlineContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '20px',
})

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 40,
  message,
  fullScreen = false,
  overlay = false,
}) => {
  const content = (
    <>
      <CircularProgress size={size} />
      {message && (
        <Typography
          variant="body2"
          sx={{ mt: 2, color: 'text.secondary' }}
        >
          {message}
        </Typography>
      )}
    </>
  )

  if (fullScreen) {
    return <FullScreenContainer>{content}</FullScreenContainer>
  }

  if (overlay) {
    return <OverlayContainer>{content}</OverlayContainer>
  }

  return <InlineContainer>{content}</InlineContainer>
}

export default LoadingSpinner