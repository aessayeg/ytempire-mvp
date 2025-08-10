/**
 * Store Index
 * Owner: Frontend Team Lead
 * 
 * Centralized export for all Zustand stores
 */

// Export all stores
export { useAuthStore } from './authStore'
export { useChannelStore } from './channelStore'
export { useVideoStore } from './videoStore'
export { useAnalyticsStore } from './analyticsStore'
export { useUIStore } from './uiStore'

// Export store types for TypeScript
export type { User } from './authStore'
export type { Channel, ChannelAnalytics } from '../services/channelService'
export type { Video, VideoGenerationRequest, VideoQueue } from '../services/videoService'

// Store selectors - common patterns for accessing store data
export const storeSelectors = {
  // Auth selectors
  auth: {
    isAuthenticated: () => useAuthStore((state) => state.isAuthenticated),
    user: () => useAuthStore((state) => state.user),
    isLoading: () => useAuthStore((state) => state.isLoading),
    error: () => useAuthStore((state) => state.error)
  },
  
  // Channel selectors
  channels: {
    all: () => useChannelStore((state) => state.channels),
    filtered: () => useChannelStore((state) => state.getFilteredChannels()),
    selected: () => useChannelStore((state) => state.selectedChannel),
    isLoading: () => useChannelStore((state) => state.isLoading),
    byId: (id: string) => useChannelStore((state) => state.getChannelById(id))
  },
  
  // Video selectors
  videos: {
    all: () => useVideoStore((state) => state.videos),
    filtered: () => useVideoStore((state) => state.getFilteredVideos()),
    selected: () => useVideoStore((state) => state.selectedVideo),
    queue: () => useVideoStore((state) => state.generationQueue),
    isLoading: () => useVideoStore((state) => state.isLoading),
    byId: (id: string) => useVideoStore((state) => state.getVideoById(id))
  },
  
  // Analytics selectors
  analytics: {
    dashboard: () => useAnalyticsStore((state) => state.dashboardMetrics),
    channelAnalytics: (channelId: string) => useAnalyticsStore((state) => 
      state.getChannelAnalyticsById(channelId)
    ),
    videoAnalytics: (videoId: string) => useAnalyticsStore((state) => 
      state.getVideoAnalyticsById(videoId)
    ),
    isLoading: () => useAnalyticsStore((state) => state.isLoadingDashboard),
    calculatedMetrics: () => useAnalyticsStore((state) => state.calculateMetrics())
  },
  
  // UI selectors
  ui: {
    theme: () => useUIStore((state) => state.theme),
    sidebar: () => useUIStore((state) => state.sidebarState),
    modals: () => useUIStore((state) => state.modals),
    notifications: () => useUIStore((state) => state.notifications),
    isLoading: () => useUIStore((state) => state.globalLoading),
    isMobile: () => useUIStore((state) => state.isMobile),
    preferences: () => useUIStore((state) => state.preferences)
  }
}

// Store actions - common patterns for updating store data
export const storeActions = {
  // Auth actions
  auth: {
    login: (email: string, password: string) => useAuthStore.getState().login(email, password),
    logout: () => useAuthStore.getState().logout(),
    clearError: () => useAuthStore.getState().clearError()
  },
  
  // Channel actions
  channels: {
    load: () => useChannelStore.getState().loadChannels(),
    create: (data: any) => useChannelStore.getState().createChannel(data),
    update: (id: string, data: any) => useChannelStore.getState().updateChannel(id, data),
    delete: (id: string) => useChannelStore.getState().deleteChannel(id),
    select: (channel: any) => useChannelStore.getState().setSelectedChannel(channel),
    setFilters: (filters: any) => useChannelStore.getState().setFilters(filters)
  },
  
  // Video actions
  videos: {
    load: (params?: any) => useVideoStore.getState().loadVideos(params),
    generate: (request: any) => useVideoStore.getState().generateVideo(request),
    update: (id: string, data: any) => useVideoStore.getState().updateVideo(id, data),
    delete: (id: string) => useVideoStore.getState().deleteVideo(id),
    publish: (id: string, schedule?: string) => useVideoStore.getState().publishVideo(id, schedule),
    select: (video: any) => useVideoStore.getState().setSelectedVideo(video),
    setFilters: (filters: any) => useVideoStore.getState().setFilters(filters)
  },
  
  // Analytics actions
  analytics: {
    loadDashboard: (timeRange?: string) => useAnalyticsStore.getState().loadDashboardMetrics(timeRange),
    loadChannelAnalytics: (id: string, timeRange?: string) => 
      useAnalyticsStore.getState().loadChannelAnalytics(id, timeRange),
    loadVideoAnalytics: (id: string) => useAnalyticsStore.getState().loadVideoAnalytics(id),
    setTimeRange: (range: any) => useAnalyticsStore.getState().setTimeRange(range)
  },
  
  // UI actions
  ui: {
    setTheme: (theme: any) => useUIStore.getState().setTheme(theme),
    toggleSidebar: () => useUIStore.getState().toggleSidebar(),
    showNotification: (notification: any) => useUIStore.getState().showNotification(notification),
    showSuccess: (message: string) => useUIStore.getState().showSuccessMessage(message),
    showError: (message: string) => useUIStore.getState().showErrorMessage(message),
    openModal: (modal: any) => useUIStore.getState().openModal(modal),
    closeModal: (id: string) => useUIStore.getState().closeModal(id),
    setLoading: (loading: boolean, message?: string) => 
      useUIStore.getState().setGlobalLoading(loading, message)
  }
}

// Store persistence utilities
export const storePersistence = {
  // Clear all persisted data
  clearAllData: () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth-storage')
      localStorage.removeItem('ui-storage')
    }
  },
  
  // Export store data for backup
  exportData: () => {
    if (typeof window !== 'undefined') {
      return {
        auth: localStorage.getItem('auth-storage'),
        ui: localStorage.getItem('ui-storage')
      }
    }
    return {}
  },
  
  // Import store data from backup
  importData: (data: { auth?: string; ui?: string }) => {
    if (typeof window !== 'undefined') {
      if (data.auth) localStorage.setItem('auth-storage', data.auth)
      if (data.ui) localStorage.setItem('ui-storage', data.ui)
    }
  }
}

// Store debugging utilities (development only)
export const storeDebug = {
  // Log current state of all stores
  logAllStates: () => {
    if (process.env.NODE_ENV === 'development') {
      console.group('🏪 Store States')
      console.log('Auth:', useAuthStore.getState())
      console.log('Channels:', useChannelStore.getState())
      console.log('Videos:', useVideoStore.getState())
      console.log('Analytics:', useAnalyticsStore.getState())
      console.log('UI:', useUIStore.getState())
      console.groupEnd()
    }
  },
  
  // Subscribe to all store changes
  subscribeToAll: () => {
    if (process.env.NODE_ENV === 'development') {
      useAuthStore.subscribe(() => console.log('Auth store updated'))
      useChannelStore.subscribe(() => console.log('Channel store updated'))
      useVideoStore.subscribe(() => console.log('Video store updated'))
      useAnalyticsStore.subscribe(() => console.log('Analytics store updated'))
      useUIStore.subscribe(() => console.log('UI store updated'))
    }
  }
}