/**
 * Global Loading Component
 * Owner: Frontend Team Lead
 * 
 * Global loading overlay using Zustand state
 */

import React from 'react'
import {
  Backdrop,
  CircularProgress,
  Typography,
  Box,
  Fade
} from '@mui/material'
import { useUI } from '@/hooks/useStores'

export const GlobalLoading: React.FC = () => {
  const { globalLoading, loadingMessage } = useUI()
  
  return (
    <Backdrop
      open={globalLoading}
      sx={{
        zIndex: 9998,
        backdropFilter: 'blur(4px)',
        backgroundColor: 'rgba(0, 0, 0, 0.3)'
      }}
    >
      <Fade in={globalLoading} timeout={300}>
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          gap={2}
          p={4}
          bgcolor="background.paper"
          borderRadius={2}
          boxShadow={3}
          minWidth={200}
        >
          <CircularProgress size={60} thickness={4} />
          <Typography variant="h6" textAlign="center">
            {loadingMessage || 'Loading...'}
          </Typography>
        </Box>
      </Fade>
    </Backdrop>
  )
}

export default GlobalLoading