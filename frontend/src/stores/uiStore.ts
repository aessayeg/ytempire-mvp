/**
 * UI Store
 * Owner: Frontend Team Lead
 * 
 * Zustand store for managing global UI state including theme,
 * layout, modals, notifications, and user preferences.
 */

import { create } from 'zustand'
import { devtools, persist, createJSONStorage } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

// Types
type Theme = 'light' | 'dark' | 'system'
type Density = 'comfortable' | 'compact' | 'spacious'
type SidebarState = 'expanded' | 'collapsed' | 'hidden'

interface Modal {
  id: string
  component: string
  props?: Record<string, any>
  closable?: boolean
  backdrop?: boolean
}

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number
  action?: {
    label: string
    handler: () => void
  }
  timestamp: number
}

interface Breadcrumb {
  label: string
  href?: string
  current?: boolean
}

interface UIState {
  // Theme and appearance
  theme: Theme
  primaryColor: string
  density: Density
  reducedMotion: boolean
  
  // Layout
  sidebarState: SidebarState
  headerHeight: number
  
  // Modals
  modals: Modal[]
  
  // Notifications
  notifications: Notification[]
  notificationPosition: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'
  
  // Loading states
  globalLoading: boolean
  loadingMessage: string | null
  
  // Navigation
  breadcrumbs: Breadcrumb[]
  currentPage: string
  previousPage: string | null
  
  // User preferences
  preferences: {
    autoSave: boolean
    confirmDeletions: boolean
    showTours: boolean
    animateTransitions: boolean
    compactTables: boolean
    defaultVideoView: 'grid' | 'list'
    itemsPerPage: 10 | 20 | 50 | 100
    language: string
    timezone: string
  }
  
  // Mobile and responsive
  isMobile: boolean
  isTablet: boolean
  screenWidth: number
  screenHeight: number
  
  // Focus management
  focusedElement: string | null
  keyboardNavigation: boolean
  
  // Search and filters
  globalSearch: string
  quickFilters: Record<string, any>
  
  // Actions
  setTheme: (theme: Theme) => void
  setPrimaryColor: (color: string) => void
  setDensity: (density: Density) => void
  toggleReducedMotion: () => void
  
  // Layout actions
  setSidebarState: (state: SidebarState) => void
  toggleSidebar: () => void
  setHeaderHeight: (height: number) => void
  
  // Modal actions
  openModal: (modal: Omit<Modal, 'id'>) => string
  closeModal: (id: string) => void
  closeAllModals: () => void
  isModalOpen: (component: string) => boolean
  
  // Notification actions
  showNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string
  dismissNotification: (id: string) => void
  dismissAllNotifications: () => void
  setNotificationPosition: (position: UIState['notificationPosition']) => void
  
  // Loading actions
  setGlobalLoading: (loading: boolean, message?: string) => void
  
  // Navigation actions
  setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => void
  setCurrentPage: (page: string) => void
  goBack: () => void
  
  // Preferences actions
  updatePreferences: (updates: Partial<UIState['preferences']>) => void
  resetPreferences: () => void
  
  // Responsive actions
  updateScreenSize: (width: number, height: number) => void
  
  // Focus actions
  setFocusedElement: (element: string | null) => void
  enableKeyboardNavigation: () => void
  disableKeyboardNavigation: () => void
  
  // Search and filter actions
  setGlobalSearch: (query: string) => void
  setQuickFilter: (key: string, value: any) => void
  clearQuickFilters: () => void
  
  // Utility actions
  showSuccessMessage: (message: string, title?: string) => string
  showErrorMessage: (message: string, title?: string) => string
  showWarningMessage: (message: string, title?: string) => string
  showInfoMessage: (message: string, title?: string) => string
  
  // Confirmation modal helpers
  showConfirmDialog: (options: {
    title: string
    message: string
    confirmText?: string
    cancelText?: string
    onConfirm: () => void
    onCancel?: () => void
    variant?: 'danger' | 'warning' | 'info'
  }) => void
  
  // Tour and onboarding
  startTour: (tourId: string) => void
  completeTour: (tourId: string) => void
  skipTour: (tourId: string) => void
  
  // Keyboard shortcuts
  registerShortcut: (key: string, handler: () => void) => void
  unregisterShortcut: (key: string) => void
}

const defaultPreferences: UIState['preferences'] = {
  autoSave: true,
  confirmDeletions: true,
  showTours: true,
  animateTransitions: true,
  compactTables: false,
  defaultVideoView: 'grid',
  itemsPerPage: 20,
  language: 'en',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        theme: 'system',
        primaryColor: '#3B82F6',
        density: 'comfortable',
        reducedMotion: false,
        
        sidebarState: 'expanded',
        headerHeight: 64,
        
        modals: [],
        
        notifications: [],
        notificationPosition: 'top-right',
        
        globalLoading: false,
        loadingMessage: null,
        
        breadcrumbs: [],
        currentPage: '/',
        previousPage: null,
        
        preferences: defaultPreferences,
        
        isMobile: false,
        isTablet: false,
        screenWidth: typeof window !== 'undefined' ? window.innerWidth : 1920,
        screenHeight: typeof window !== 'undefined' ? window.innerHeight : 1080,
        
        focusedElement: null,
        keyboardNavigation: false,
        
        globalSearch: '',
        quickFilters: {},
        
        // Actions
        setTheme: (theme: Theme) => {
          set((state) => {
            state.theme = theme
          })
        },
        
        setPrimaryColor: (color: string) => {
          set((state) => {
            state.primaryColor = color
          })
        },
        
        setDensity: (density: Density) => {
          set((state) => {
            state.density = density
          })
        },
        
        toggleReducedMotion: () => {
          set((state) => {
            state.reducedMotion = !state.reducedMotion
          })
        },
        
        // Layout actions
        setSidebarState: (sidebarState: SidebarState) => {
          set((state) => {
            state.sidebarState = sidebarState
          })
        },
        
        toggleSidebar: () => {
          set((state) => {
            state.sidebarState = state.sidebarState === 'expanded' ? 'collapsed' : 'expanded'
          })
        },
        
        setHeaderHeight: (height: number) => {
          set((state) => {
            state.headerHeight = height
          })
        },
        
        // Modal actions
        openModal: (modal: Omit<Modal, 'id'>) => {
          const id = `modal-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
          
          set((state) => {
            state.modals.push({
              ...modal,
              id
            })
          })
          
          return id
        },
        
        closeModal: (id: string) => {
          set((state) => {
            state.modals = state.modals.filter(modal => modal.id !== id)
          })
        },
        
        closeAllModals: () => {
          set((state) => {
            state.modals = []
          })
        },
        
        isModalOpen: (component: string) => {
          return get().modals.some(modal => modal.component === component)
        },
        
        // Notification actions
        showNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => {
          const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
          
          set((state) => {
            state.notifications.push({
              ...notification,
              id,
              timestamp: Date.now()
            })
          })
          
          // Auto-dismiss after duration
          if (notification.duration !== 0) {
            setTimeout(() => {
              get().dismissNotification(id)
            }, notification.duration || 5000)
          }
          
          return id
        },
        
        dismissNotification: (id: string) => {
          set((state) => {
            state.notifications = state.notifications.filter(notification => notification.id !== id)
          })
        },
        
        dismissAllNotifications: () => {
          set((state) => {
            state.notifications = []
          })
        },
        
        setNotificationPosition: (position: UIState['notificationPosition']) => {
          set((state) => {
            state.notificationPosition = position
          })
        },
        
        // Loading actions
        setGlobalLoading: (loading: boolean, message?: string) => {
          set((state) => {
            state.globalLoading = loading
            state.loadingMessage = loading ? (message || null) : null
          })
        },
        
        // Navigation actions
        setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => {
          set((state) => {
            state.breadcrumbs = breadcrumbs
          })
        },
        
        setCurrentPage: (page: string) => {
          set((state) => {
            state.previousPage = state.currentPage
            state.currentPage = page
          })
        },
        
        goBack: () => {
          const { previousPage } = get()
          if (previousPage && typeof window !== 'undefined') {
            window.history.back()
          }
        },
        
        // Preferences actions
        updatePreferences: (updates: Partial<UIState['preferences']>) => {
          set((state) => {
            state.preferences = { ...state.preferences, ...updates }
          })
        },
        
        resetPreferences: () => {
          set((state) => {
            state.preferences = { ...defaultPreferences }
          })
        },
        
        // Responsive actions
        updateScreenSize: (width: number, height: number) => {
          set((state) => {
            state.screenWidth = width
            state.screenHeight = height
            state.isMobile = width < 768
            state.isTablet = width >= 768 && width < 1024
            
            // Auto-collapse sidebar on mobile
            if (width < 768 && state.sidebarState === 'expanded') {
              state.sidebarState = 'collapsed'
            }
          })
        },
        
        // Focus actions
        setFocusedElement: (element: string | null) => {
          set((state) => {
            state.focusedElement = element
          })
        },
        
        enableKeyboardNavigation: () => {
          set((state) => {
            state.keyboardNavigation = true
          })
        },
        
        disableKeyboardNavigation: () => {
          set((state) => {
            state.keyboardNavigation = false
          })
        },
        
        // Search and filter actions
        setGlobalSearch: (query: string) => {
          set((state) => {
            state.globalSearch = query
          })
        },
        
        setQuickFilter: (key: string, value: any) => {
          set((state) => {
            state.quickFilters[key] = value
          })
        },
        
        clearQuickFilters: () => {
          set((state) => {
            state.quickFilters = {}
          })
        },
        
        // Utility actions
        showSuccessMessage: (message: string, title = 'Success') => {
          return get().showNotification({
            type: 'success',
            title,
            message,
            duration: 4000
          })
        },
        
        showErrorMessage: (message: string, title = 'Error') => {
          return get().showNotification({
            type: 'error',
            title,
            message,
            duration: 6000
          })
        },
        
        showWarningMessage: (message: string, title = 'Warning') => {
          return get().showNotification({
            type: 'warning',
            title,
            message,
            duration: 5000
          })
        },
        
        showInfoMessage: (message: string, title = 'Info') => {
          return get().showNotification({
            type: 'info',
            title,
            message,
            duration: 4000
          })
        },
        
        // Confirmation modal helpers
        showConfirmDialog: (options) => {
          get().openModal({
            component: 'ConfirmDialog',
            props: {
              title: options.title,
              message: options.message,
              confirmText: options.confirmText || 'Confirm',
              cancelText: options.cancelText || 'Cancel',
              variant: options.variant || 'info',
              onConfirm: () => {
                options.onConfirm()
                get().closeModal('confirm-dialog')
              },
              onCancel: () => {
                options.onCancel?.()
                get().closeModal('confirm-dialog')
              }
            },
            closable: true
          })
        },
        
        // Tour and onboarding
        startTour: (tourId: string) => {
          // Implementation would depend on tour library
          console.log(`Starting tour: ${tourId}`)
        },
        
        completeTour: (tourId: string) => {
          set((state) => {
            state.preferences.showTours = false
          })
          console.log(`Completed tour: ${tourId}`)
        },
        
        skipTour: (tourId: string) => {
          console.log(`Skipped tour: ${tourId}`)
        },
        
        // Keyboard shortcuts
        registerShortcut: (key: string, handler: () => void) => {
          // Implementation would depend on keyboard shortcut library
          console.log(`Registered shortcut: ${key}`)
        },
        
        unregisterShortcut: (key: string) => {
          console.log(`Unregistered shortcut: ${key}`)
        }
      })),
      {
        name: 'ui-storage',
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => ({
          theme: state.theme,
          primaryColor: state.primaryColor,
          density: state.density,
          reducedMotion: state.reducedMotion,
          sidebarState: state.sidebarState,
          notificationPosition: state.notificationPosition,
          preferences: state.preferences
        })
      }
    ),
    {
      name: 'ui-store'
    }
  )
)

// Initialize responsive listener
if (typeof window !== 'undefined') {
  const handleResize = () => {
    useUIStore.getState().updateScreenSize(window.innerWidth, window.innerHeight)
  }
  
  window.addEventListener('resize', handleResize)
  handleResize() // Initial call
}