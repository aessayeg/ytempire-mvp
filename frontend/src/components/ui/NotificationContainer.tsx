/**
 * Notification Container Component
 * Owner: Frontend Team Lead
 * 
 * Container for displaying global notifications using Zustand state
 */

import React from 'react'
import {
  Snackbar,
  Alert,
  AlertTitle,
  Button,
  Box,
  Fade,
  Slide
} from '@mui/material'
import { Close } from '@mui/icons-material'
import { useNotifications } from '@/hooks/useStores'

const getPositionStyles = (position: string) => {
  switch (position) {
    case 'top-left':
      return { top: 24, left: 24 }
    case 'top-right':
      return { top: 24, right: 24 }
    case 'bottom-left':
      return { bottom: 24, left: 24 }
    case 'bottom-right':
      return { bottom: 24, right: 24 }
    default:
      return { top: 24, right: 24 }
  }
}

export const NotificationContainer: React.FC = () => {
  const { notifications, position, actions } = useNotifications()
  
  const positionStyles = getPositionStyles(position)
  
  return (
    <Box
      sx={{
        position: 'fixed',
        zIndex: 9999,
        maxWidth: 400,
        width: '100%',
        ...positionStyles
      }}
    >
      {notifications.map((notification, index) => (
        <Slide
          key={notification.id}
          direction={position.includes('right') ? 'left' : 'right'}
          in={true}
          timeout={300}
        >
          <Box
            sx={{
              mb: 1,
              transform: `translateY(-${index * 8}px)`,
              transition: 'transform 0.2s ease-in-out'
            }}
          >
            <Alert
              severity={notification.type}
              onClose={() => actions.dismiss(notification.id)}
              action={
                notification.action ? (
                  <Box display="flex" alignItems="center" gap={1}>
                    <Button
                      color="inherit"
                      size="small"
                      onClick={notification.action.handler}
                    >
                      {notification.action.label}
                    </Button>
                    <Button
                      color="inherit"
                      size="small"
                      onClick={() => actions.dismiss(notification.id)}
                    >
                      <Close fontSize="small" />
                    </Button>
                  </Box>
                ) : undefined
              }
              sx={{
                boxShadow: 3,
                backdropFilter: 'blur(8px)',
                backgroundColor: theme => `${theme.palette.background.paper}f0`
              }}
            >
              <AlertTitle>{notification.title}</AlertTitle>
              {notification.message}
            </Alert>
          </Box>
        </Slide>
      ))}
      
      {/* Clear All Button */}
      {notifications.length > 2 && (
        <Fade in={true} timeout={500}>
          <Box textAlign="center" mt={2}>
            <Button
              variant="outlined"
              size="small"
              onClick={actions.dismissAll}
              sx={{
                backdropFilter: 'blur(8px)',
                backgroundColor: theme => `${theme.palette.background.paper}cc`
              }}
            >
              Clear All ({notifications.length})
            </Button>
          </Box>
        </Fade>
      )}
    </Box>
  )
}

export default NotificationContainer