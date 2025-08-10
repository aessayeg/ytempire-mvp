/**
 * Store Hooks
 * Owner: Frontend Team Lead
 * 
 * Custom hooks for easier store access and common patterns
 */

import { useEffect, useCallback } from 'react'
import {
  useAuthStore,
  useChannelStore,
  useVideoStore,
  useAnalyticsStore,
  useUIStore,
  storeActions
} from '@/stores'

// Auth hooks
export const useAuth = () => {
  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    register,
    clearError
  } = useAuthStore()
  
  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    register,
    clearError
  }
}

export const useAuthGuard = (redirectTo = '/login') => {
  const { isAuthenticated, isLoading } = useAuthStore()
  
  useEffect(() => {
    if (!isLoading && !isAuthenticated && typeof window !== 'undefined') {
      window.location.href = redirectTo
    }
  }, [isAuthenticated, isLoading, redirectTo])
  
  return { isAuthenticated, isLoading }
}

// Channel hooks
export const useChannels = () => {
  const store = useChannelStore()
  
  const loadChannels = useCallback(async () => {
    if (store.channels.length === 0) {
      await store.loadChannels()
    }
  }, [store])
  
  useEffect(() => {
    loadChannels()
  }, [loadChannels])
  
  return {
    channels: store.getFilteredChannels(),
    selectedChannel: store.selectedChannel,
    isLoading: store.isLoading,
    error: store.error,
    filters: store.filters,
    actions: {
      create: store.createChannel,
      update: store.updateChannel,
      delete: store.deleteChannel,
      select: store.setSelectedChannel,
      setFilters: store.setFilters,
      clearError: store.clearError
    }
  }
}

export const useChannel = (channelId?: string) => {
  const store = useChannelStore()
  
  const channel = channelId ? store.getChannelById(channelId) : store.selectedChannel
  
  const loadChannel = useCallback(async () => {
    if (channelId && !channel) {
      await store.loadChannel(channelId)
    }
  }, [channelId, channel, store])
  
  useEffect(() => {
    loadChannel()
  }, [loadChannel])
  
  return {
    channel,
    isLoading: store.isLoading,
    error: store.error,
    actions: {
      update: (updates: any) => channelId ? store.updateChannel(channelId, updates) : null,
      delete: () => channelId ? store.deleteChannel(channelId) : null,
      connectYouTube: () => channelId ? store.connectYouTube(channelId) : null,
      syncFromYouTube: () => channelId ? store.syncFromYouTube(channelId) : null
    }
  }
}

// Video hooks
export const useVideos = (autoLoad = true) => {
  const store = useVideoStore()
  
  const loadVideos = useCallback(async () => {
    if (autoLoad && store.videos.length === 0) {
      await store.loadVideos()
    }
  }, [autoLoad, store])
  
  useEffect(() => {
    loadVideos()
  }, [loadVideos])
  
  return {
    videos: store.getFilteredVideos(),
    selectedVideo: store.selectedVideo,
    generationQueue: store.generationQueue,
    isLoading: store.isLoading,
    isGenerating: store.isGenerating,
    error: store.error,
    pagination: {
      currentPage: store.currentPage,
      totalPages: store.totalPages,
      totalVideos: store.totalVideos,
      videosPerPage: store.videosPerPage
    },
    filters: store.filters,
    selectedIds: store.selectedVideoIds,
    actions: {
      generate: store.generateVideo,
      update: store.updateVideo,
      delete: store.deleteVideo,
      publish: store.publishVideo,
      retry: store.retryVideo,
      select: store.setSelectedVideo,
      setFilters: store.setFilters,
      setPage: store.setPage,
      toggleSelection: store.toggleVideoSelection,
      clearSelection: store.clearSelection,
      bulkDelete: store.bulkDelete,
      bulkPublish: store.bulkPublish
    }
  }
}

export const useVideo = (videoId?: string) => {
  const store = useVideoStore()
  
  const video = videoId ? store.getVideoById(videoId) : store.selectedVideo
  
  const loadVideo = useCallback(async () => {
    if (videoId && !video) {
      await store.loadVideo(videoId)
    }
  }, [videoId, video, store])
  
  useEffect(() => {
    loadVideo()
  }, [loadVideo])
  
  return {
    video,
    isLoading: store.isLoading,
    error: store.error,
    actions: {
      update: (updates: any) => videoId ? store.updateVideo(videoId, updates) : null,
      delete: () => videoId ? store.deleteVideo(videoId) : null,
      publish: (schedule?: string) => videoId ? store.publishVideo(videoId, schedule) : null,
      retry: () => videoId ? store.retryVideo(videoId) : null,
      loadCosts: () => videoId ? store.loadCostBreakdown(videoId) : null,
      loadAnalytics: () => videoId ? store.loadVideoAnalytics(videoId) : null
    }
  }
}

// Analytics hooks
export const useAnalytics = (autoLoad = true) => {
  const store = useAnalyticsStore()
  
  const loadDashboard = useCallback(async () => {
    if (autoLoad && !store.dashboardMetrics) {
      await store.loadDashboardMetrics()
    }
  }, [autoLoad, store])
  
  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])
  
  return {
    dashboard: store.dashboardMetrics,
    calculatedMetrics: store.calculateMetrics(),
    timeRange: store.selectedTimeRange,
    customRange: store.customDateRange,
    isLoading: store.isLoadingDashboard,
    error: store.error,
    actions: {
      loadDashboard: store.loadDashboardMetrics,
      setTimeRange: store.setTimeRange,
      setCustomRange: store.setCustomDateRange,
      clearError: store.clearError
    }
  }
}

export const useChannelAnalytics = (channelId: string, autoLoad = true) => {
  const store = useAnalyticsStore()
  
  const analytics = store.getChannelAnalyticsById(channelId)
  
  const loadAnalytics = useCallback(async () => {
    if (autoLoad && !analytics) {
      await store.loadChannelAnalytics(channelId)
    }
  }, [autoLoad, analytics, channelId, store])
  
  useEffect(() => {
    loadAnalytics()
  }, [loadAnalytics])
  
  return {
    analytics,
    isLoading: store.isLoadingChannelAnalytics,
    error: store.error,
    actions: {
      reload: () => store.loadChannelAnalytics(channelId),
      export: () => store.exportAnalyticsData('channel', channelId)
    }
  }
}

export const useVideoAnalytics = (videoId: string, autoLoad = true) => {
  const store = useAnalyticsStore()
  
  const analytics = store.getVideoAnalyticsById(videoId)
  
  const loadAnalytics = useCallback(async () => {
    if (autoLoad && !analytics) {
      await store.loadVideoAnalytics(videoId)
    }
  }, [autoLoad, analytics, videoId, store])
  
  useEffect(() => {
    loadAnalytics()
  }, [loadAnalytics])
  
  return {
    analytics,
    isLoading: store.isLoadingVideoAnalytics,
    error: store.error,
    actions: {
      reload: () => store.loadVideoAnalytics(videoId),
      export: () => store.exportAnalyticsData('video', videoId)
    }
  }
}

// UI hooks
export const useUI = () => {
  const store = useUIStore()
  
  return {
    theme: store.theme,
    primaryColor: store.primaryColor,
    density: store.density,
    sidebar: store.sidebarState,
    isMobile: store.isMobile,
    isTablet: store.isTablet,
    globalLoading: store.globalLoading,
    loadingMessage: store.loadingMessage,
    modals: store.modals,
    notifications: store.notifications,
    preferences: store.preferences,
    actions: {
      setTheme: store.setTheme,
      setPrimaryColor: store.setPrimaryColor,
      setDensity: store.setDensity,
      toggleSidebar: store.toggleSidebar,
      setSidebar: store.setSidebarState,
      setLoading: store.setGlobalLoading,
      showSuccess: store.showSuccessMessage,
      showError: store.showErrorMessage,
      showWarning: store.showWarningMessage,
      showInfo: store.showInfoMessage,
      openModal: store.openModal,
      closeModal: store.closeModal,
      showConfirm: store.showConfirmDialog,
      updatePreferences: store.updatePreferences
    }
  }
}

export const useModal = (component: string) => {
  const { modals, openModal, closeModal, isModalOpen } = useUIStore()
  
  const modal = modals.find(m => m.component === component)
  const isOpen = isModalOpen(component)
  
  return {
    isOpen,
    modal,
    open: (props?: Record<string, any>) => openModal({ component, props }),
    close: () => {
      if (modal) closeModal(modal.id)
    }
  }
}

export const useNotifications = () => {
  const store = useUIStore()
  
  return {
    notifications: store.notifications,
    position: store.notificationPosition,
    actions: {
      show: store.showNotification,
      showSuccess: store.showSuccessMessage,
      showError: store.showErrorMessage,
      showWarning: store.showWarningMessage,
      showInfo: store.showInfoMessage,
      dismiss: store.dismissNotification,
      dismissAll: store.dismissAllNotifications,
      setPosition: store.setNotificationPosition
    }
  }
}

// Combined hooks for common use cases
export const useDashboard = () => {
  const auth = useAuth()
  const { dashboard, calculatedMetrics, actions: analyticsActions } = useAnalytics()
  const { channels } = useChannels()
  const { videos, generationQueue } = useVideos()
  const ui = useUI()
  
  return {
    user: auth.user,
    dashboard,
    metrics: calculatedMetrics,
    channels,
    videos,
    queue: generationQueue,
    isLoading: ui.globalLoading,
    actions: {
      refreshAll: async () => {
        ui.actions.setLoading(true, 'Refreshing dashboard...')
        try {
          await Promise.all([
            analyticsActions.loadDashboard(),
            storeActions.channels.load(),
            storeActions.videos.load()
          ])
        } finally {
          ui.actions.setLoading(false)
        }
      }
    }
  }
}

export const useChannelManagement = (channelId?: string) => {
  const channel = useChannel(channelId)
  const channelAnalytics = useChannelAnalytics(channelId || '', !!channelId)
  const { videos, actions: videoActions } = useVideos(false)
  const ui = useUI()
  
  // Filter videos for current channel
  const channelVideos = videos.filter(v => v.channel_id === channelId)
  
  return {
    channel: channel.channel,
    analytics: channelAnalytics.analytics,
    videos: channelVideos,
    isLoading: channel.isLoading || channelAnalytics.isLoading,
    error: channel.error || channelAnalytics.error,
    actions: {
      ...channel.actions,
      loadVideos: () => videoActions.setFilters({ channelId }),
      generateVideo: videoActions.generate,
      refreshAll: async () => {
        if (!channelId) return
        
        ui.actions.setLoading(true, 'Refreshing channel data...')
        try {
          await Promise.all([
            channel.actions.update ? channel.actions.update({}) : Promise.resolve(), // Refresh channel
            channelAnalytics.actions.reload(),
            videoActions.setFilters({ channelId })
          ])
        } finally {
          ui.actions.setLoading(false)
        }
      }
    }
  }
}

// Real-time updates hook
export const useRealTimeUpdates = () => {
  const videoStore = useVideoStore()
  const analyticsStore = useAnalyticsStore()
  const uiStore = useUIStore()
  
  useEffect(() => {
    // WebSocket connection for real-time updates
    // This would be implemented with actual WebSocket connection
    const connectWebSocket = () => {
      // Example WebSocket implementation
      // const ws = new WebSocket('ws://localhost:8000/ws')
      // 
      // ws.onmessage = (event) => {
      //   const data = JSON.parse(event.data)
      //   
      //   switch (data.type) {
      //     case 'video_status_update':
      //       videoStore.updateVideoStatus(data.video_id, data.status, data.metadata)
      //       break
      //     case 'queue_update':
      //       videoStore.updateQueueItem(data.queue_id, data.updates)
      //       break
      //     case 'analytics_update':
      //       analyticsStore.updateMetrics(data.updates)
      //       break
      //     default:
      //       console.log('Unknown WebSocket message type:', data.type)
      //   }
      // }
      // 
      // return ws
    }
    
    // const ws = connectWebSocket()
    
    return () => {
      // ws?.close()
    }
  }, [videoStore, analyticsStore])
}