/**
 * Analytics Store
 * Owner: Frontend Team Lead
 * 
 * Zustand store for managing analytics data across channels,
 * videos, and overall platform performance.
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

// Types for analytics data
interface ChannelAnalytics {
  channel_id: string
  subscriber_growth: {
    current: number
    change: number
    change_percentage: number
  }
  view_analytics: {
    total_views: number
    views_last_30d: number
    average_view_duration: number
  }
  engagement: {
    likes: number
    comments: number
    shares: number
    engagement_rate: number
  }
  revenue: {
    estimated_monthly: number
    last_month: number
    ytd: number
  }
}

interface VideoAnalytics {
  video_id: string
  views: number
  likes: number
  comments: number
  shares: number
  watch_time: number
  engagement_rate: number
  revenue: number
  demographics: {
    age_groups: Record<string, number>
    locations: Record<string, number>
    devices: Record<string, number>
  }
  traffic_sources: Record<string, number>
}

interface DashboardMetrics {
  overview: {
    total_channels: number
    total_videos: number
    total_subscribers: number
    total_views: number
    monthly_revenue: number
    total_costs: number
    net_profit: number
  }
  performance: {
    top_performing_channels: Array<{
      id: string
      name: string
      views: number
      revenue: number
      growth_rate: number
    }>
    top_performing_videos: Array<{
      id: string
      title: string
      channel_name: string
      views: number
      engagement_rate: number
    }>
    trending_topics: Array<{
      topic: string
      frequency: number
      avg_views: number
    }>
  }
  costs: {
    breakdown_by_service: Record<string, number>
    breakdown_by_channel: Record<string, number>
    monthly_trend: Array<{
      month: string
      total_cost: number
      video_count: number
      cost_per_video: number
    }>
  }
  growth_metrics: {
    subscriber_growth: Array<{
      date: string
      count: number
      growth_rate: number
    }>
    revenue_growth: Array<{
      date: string
      revenue: number
      growth_rate: number
    }>
    video_performance_trend: Array<{
      date: string
      avg_views: number
      avg_engagement: number
    }>
  }
}

interface ComparisonData {
  channels: Record<string, ChannelAnalytics>
  videos: Record<string, VideoAnalytics>
  timeRange: {
    start: string
    end: string
  }
}

interface AnalyticsState {
  // Data state
  dashboardMetrics: DashboardMetrics | null
  channelAnalytics: Record<string, ChannelAnalytics>
  videoAnalytics: Record<string, VideoAnalytics>
  comparisonData: ComparisonData | null
  
  // Time range filters
  selectedTimeRange: '7d' | '30d' | '90d' | '1y' | 'custom'
  customDateRange: {
    start: string | null
    end: string | null
  }
  
  // UI state
  isLoadingDashboard: boolean
  isLoadingChannelAnalytics: boolean
  isLoadingVideoAnalytics: boolean
  isLoadingComparison: boolean
  
  // Error state
  error: string | null
  
  // Selected items for comparison
  selectedChannelsForComparison: string[]
  selectedVideosForComparison: string[]
  
  // Actions
  loadDashboardMetrics: (timeRange?: string) => Promise<void>
  
  loadChannelAnalytics: (channelId: string, timeRange?: string) => Promise<ChannelAnalytics | null>
  
  loadVideoAnalytics: (videoId: string) => Promise<VideoAnalytics | null>
  
  loadBulkChannelAnalytics: (channelIds: string[], timeRange?: string) => Promise<void>
  
  loadBulkVideoAnalytics: (videoIds: string[], timeRange?: string) => Promise<void>
  
  // Comparison functions
  compareChannels: (channelIds: string[], timeRange?: string) => Promise<void>
  
  compareVideos: (videoIds: string[], timeRange?: string) => Promise<void>
  
  // Time range management
  setTimeRange: (range: AnalyticsState['selectedTimeRange']) => void
  
  setCustomDateRange: (start: string, end: string) => void
  
  // Selection management
  toggleChannelForComparison: (channelId: string) => void
  
  toggleVideoForComparison: (videoId: string) => void
  
  clearComparisons: () => void
  
  // Data export
  exportAnalyticsData: (type: 'dashboard' | 'channel' | 'video', id?: string) => Promise<Blob | null>
  
  // Utility functions
  clearError: () => void
  
  getChannelAnalyticsById: (channelId: string) => ChannelAnalytics | null
  
  getVideoAnalyticsById: (videoId: string) => VideoAnalytics | null
  
  calculateMetrics: () => {
    totalROI: number
    averageEngagement: number
    costPerVideo: number
    revenuePerSubscriber: number
  }
  
  // Real-time updates
  updateMetrics: (updates: Partial<DashboardMetrics>) => void
  
  updateChannelAnalytics: (channelId: string, analytics: Partial<ChannelAnalytics>) => void
  
  updateVideoAnalytics: (videoId: string, analytics: Partial<VideoAnalytics>) => void
}

export const useAnalyticsStore = create<AnalyticsState>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      dashboardMetrics: null,
      channelAnalytics: {},
      videoAnalytics: {},
      comparisonData: null,
      
      selectedTimeRange: '30d',
      customDateRange: {
        start: null,
        end: null
      },
      
      isLoadingDashboard: false,
      isLoadingChannelAnalytics: false,
      isLoadingVideoAnalytics: false,
      isLoadingComparison: false,
      
      error: null,
      
      selectedChannelsForComparison: [],
      selectedVideosForComparison: [],
      
      // Actions
      loadDashboardMetrics: async (timeRange) => {
        set((state) => {
          state.isLoadingDashboard = true
          state.error = null
        })
        
        try {
          const range = timeRange || get().selectedTimeRange
          
          // Mock API call - replace with actual API
          const response = await fetch(`/api/v1/analytics/dashboard?time_range=${range}`)
          const metrics = await response.json()
          
          set((state) => {
            state.dashboardMetrics = metrics
            state.isLoadingDashboard = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load dashboard metrics'
            state.isLoadingDashboard = false
          })
        }
      },
      
      loadChannelAnalytics: async (channelId: string, timeRange) => {
        set((state) => {
          state.isLoadingChannelAnalytics = true
          state.error = null
        })
        
        try {
          const range = timeRange || get().selectedTimeRange
          
          // Mock API call - replace with actual API
          const response = await fetch(`/api/v1/analytics/channels/${channelId}?time_range=${range}`)
          const analytics = await response.json()
          
          set((state) => {
            state.channelAnalytics[channelId] = analytics
            state.isLoadingChannelAnalytics = false
          })
          
          return analytics
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load channel analytics'
            state.isLoadingChannelAnalytics = false
          })
          return null
        }
      },
      
      loadVideoAnalytics: async (videoId: string) => {
        set((state) => {
          state.isLoadingVideoAnalytics = true
          state.error = null
        })
        
        try {
          // Mock API call - replace with actual API
          const response = await fetch(`/api/v1/analytics/videos/${videoId}`)
          const analytics = await response.json()
          
          set((state) => {
            state.videoAnalytics[videoId] = analytics
            state.isLoadingVideoAnalytics = false
          })
          
          return analytics
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load video analytics'
            state.isLoadingVideoAnalytics = false
          })
          return null
        }
      },
      
      loadBulkChannelAnalytics: async (channelIds: string[], timeRange) => {
        set((state) => {
          state.isLoadingChannelAnalytics = true
          state.error = null
        })
        
        try {
          const range = timeRange || get().selectedTimeRange
          
          const promises = channelIds.map(id => 
            fetch(`/api/v1/analytics/channels/${id}?time_range=${range}`).then(r => r.json())
          )
          
          const results = await Promise.all(promises)
          
          set((state) => {
            channelIds.forEach((id, index) => {
              state.channelAnalytics[id] = results[index]
            })
            state.isLoadingChannelAnalytics = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load bulk channel analytics'
            state.isLoadingChannelAnalytics = false
          })
        }
      },
      
      loadBulkVideoAnalytics: async (videoIds: string[], timeRange) => {
        set((state) => {
          state.isLoadingVideoAnalytics = true
          state.error = null
        })
        
        try {
          const promises = videoIds.map(id => 
            fetch(`/api/v1/analytics/videos/${id}`).then(r => r.json())
          )
          
          const results = await Promise.all(promises)
          
          set((state) => {
            videoIds.forEach((id, index) => {
              state.videoAnalytics[id] = results[index]
            })
            state.isLoadingVideoAnalytics = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load bulk video analytics'
            state.isLoadingVideoAnalytics = false
          })
        }
      },
      
      // Comparison functions
      compareChannels: async (channelIds: string[], timeRange) => {
        set((state) => {
          state.isLoadingComparison = true
          state.error = null
        })
        
        try {
          const range = timeRange || get().selectedTimeRange
          
          // Load analytics for all channels
          await get().loadBulkChannelAnalytics(channelIds, range)
          
          const channelAnalytics = get().channelAnalytics
          const selectedChannelAnalytics: Record<string, ChannelAnalytics> = {}
          
          channelIds.forEach(id => {
            if (channelAnalytics[id]) {
              selectedChannelAnalytics[id] = channelAnalytics[id]
            }
          })
          
          set((state) => {
            state.comparisonData = {
              channels: selectedChannelAnalytics,
              videos: {},
              timeRange: {
                start: state.customDateRange.start || '',
                end: state.customDateRange.end || ''
              }
            }
            state.isLoadingComparison = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to compare channels'
            state.isLoadingComparison = false
          })
        }
      },
      
      compareVideos: async (videoIds: string[], timeRange) => {
        set((state) => {
          state.isLoadingComparison = true
          state.error = null
        })
        
        try {
          // Load analytics for all videos
          await get().loadBulkVideoAnalytics(videoIds, timeRange)
          
          const videoAnalytics = get().videoAnalytics
          const selectedVideoAnalytics: Record<string, VideoAnalytics> = {}
          
          videoIds.forEach(id => {
            if (videoAnalytics[id]) {
              selectedVideoAnalytics[id] = videoAnalytics[id]
            }
          })
          
          set((state) => {
            state.comparisonData = {
              channels: {},
              videos: selectedVideoAnalytics,
              timeRange: {
                start: state.customDateRange.start || '',
                end: state.customDateRange.end || ''
              }
            }
            state.isLoadingComparison = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to compare videos'
            state.isLoadingComparison = false
          })
        }
      },
      
      // Time range management
      setTimeRange: (range: AnalyticsState['selectedTimeRange']) => {
        set((state) => {
          state.selectedTimeRange = range
          if (range !== 'custom') {
            state.customDateRange = { start: null, end: null }
          }
        })
      },
      
      setCustomDateRange: (start: string, end: string) => {
        set((state) => {
          state.selectedTimeRange = 'custom'
          state.customDateRange = { start, end }
        })
      },
      
      // Selection management
      toggleChannelForComparison: (channelId: string) => {
        set((state) => {
          const index = state.selectedChannelsForComparison.indexOf(channelId)
          if (index === -1) {
            state.selectedChannelsForComparison.push(channelId)
          } else {
            state.selectedChannelsForComparison.splice(index, 1)
          }
        })
      },
      
      toggleVideoForComparison: (videoId: string) => {
        set((state) => {
          const index = state.selectedVideosForComparison.indexOf(videoId)
          if (index === -1) {
            state.selectedVideosForComparison.push(videoId)
          } else {
            state.selectedVideosForComparison.splice(index, 1)
          }
        })
      },
      
      clearComparisons: () => {
        set((state) => {
          state.selectedChannelsForComparison = []
          state.selectedVideosForComparison = []
          state.comparisonData = null
        })
      },
      
      // Data export
      exportAnalyticsData: async (type: 'dashboard' | 'channel' | 'video', id) => {
        try {
          let endpoint = `/api/v1/analytics/export/${type}`
          if (id) {
            endpoint += `/${id}`
          }
          
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              time_range: get().selectedTimeRange,
              custom_range: get().customDateRange
            })
          })
          
          if (!response.ok) {
            throw new Error('Export failed')
          }
          
          return await response.blob()
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to export analytics data'
          })
          return null
        }
      },
      
      // Utility functions
      clearError: () => {
        set((state) => {
          state.error = null
        })
      },
      
      getChannelAnalyticsById: (channelId: string) => {
        return get().channelAnalytics[channelId] || null
      },
      
      getVideoAnalyticsById: (videoId: string) => {
        return get().videoAnalytics[videoId] || null
      },
      
      calculateMetrics: () => {
        const { dashboardMetrics } = get()
        
        if (!dashboardMetrics) {
          return {
            totalROI: 0,
            averageEngagement: 0,
            costPerVideo: 0,
            revenuePerSubscriber: 0
          }
        }
        
        const totalROI = dashboardMetrics.overview.total_costs > 0 
          ? (dashboardMetrics.overview.monthly_revenue / dashboardMetrics.overview.total_costs) * 100 
          : 0
          
        const costPerVideo = dashboardMetrics.overview.total_videos > 0
          ? dashboardMetrics.overview.total_costs / dashboardMetrics.overview.total_videos
          : 0
          
        const revenuePerSubscriber = dashboardMetrics.overview.total_subscribers > 0
          ? dashboardMetrics.overview.monthly_revenue / dashboardMetrics.overview.total_subscribers
          : 0
        
        // Calculate average engagement from top performing videos
        const averageEngagement = dashboardMetrics.performance.top_performing_videos.length > 0
          ? dashboardMetrics.performance.top_performing_videos.reduce((sum, video) => sum + video.engagement_rate, 0) / dashboardMetrics.performance.top_performing_videos.length
          : 0
        
        return {
          totalROI: Number(totalROI.toFixed(2)),
          averageEngagement: Number(averageEngagement.toFixed(2)),
          costPerVideo: Number(costPerVideo.toFixed(2)),
          revenuePerSubscriber: Number(revenuePerSubscriber.toFixed(2))
        }
      },
      
      // Real-time updates
      updateMetrics: (updates: Partial<DashboardMetrics>) => {
        set((state) => {
          if (state.dashboardMetrics) {
            state.dashboardMetrics = {
              ...state.dashboardMetrics,
              ...updates
            }
          }
        })
      },
      
      updateChannelAnalytics: (channelId: string, analytics: Partial<ChannelAnalytics>) => {
        set((state) => {
          if (state.channelAnalytics[channelId]) {
            state.channelAnalytics[channelId] = {
              ...state.channelAnalytics[channelId],
              ...analytics
            }
          }
        })
      },
      
      updateVideoAnalytics: (videoId: string, analytics: Partial<VideoAnalytics>) => {
        set((state) => {
          if (state.videoAnalytics[videoId]) {
            state.videoAnalytics[videoId] = {
              ...state.videoAnalytics[videoId],
              ...analytics
            }
          }
        })
      }
    })),
    {
      name: 'analytics-store'
    }
  )
)