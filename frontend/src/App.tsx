import { BrowserRouter } from 'react-router-dom'
import CssBaseline from '@mui/material/CssBaseline'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { Toaster } from 'react-hot-toast'

import Router from './router'
import { ThemeProvider } from './contexts/ThemeContext'
import { AuthProvider } from './contexts/AuthContext'
import { PWAProvider } from './contexts/PWAContext'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ScreenReaderAnnouncer } from './components/Accessibility/ScreenReaderAnnouncer'
import { SkipNavigation } from './components/Accessibility/SkipNavigation'
import { InstallPrompt, OfflineIndicator } from './components/PWA/InstallPrompt'
import './index.css'

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
})

function App() {
  return (
    <ErrorBoundary level="page">
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <CssBaseline />
          <BrowserRouter>
            <PWAProvider>
              <AuthProvider>
                <SkipNavigation />
                <ScreenReaderAnnouncer />
                <div id="main-content" tabIndex={-1}>
                  <Router />
                </div>
                <InstallPrompt />
                <OfflineIndicator />
                <Toaster
                  position="top-right"
                  reverseOrder={false}
                  gutter={8}
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: '#363636',
                      color: '#fff',
                    },
                    success: {
                      style: {
                        background: '#2E7D32',
                      },
                    },
                    error: {
                      style: {
                        background: '#C62828',
                      },
                    },
                  }}
                />
              </AuthProvider>
            </PWAProvider>
          </BrowserRouter>
        </ThemeProvider>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App