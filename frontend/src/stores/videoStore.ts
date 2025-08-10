/**
 * Video Store
 * Owner: Frontend Team Lead
 * 
 * Zustand store for managing video state, including generation,
 * queue management, analytics, and publishing.
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { VideoService, type Video, type VideoGenerationRequest, type VideoQueue, type CostBreakdown } from '@/services/videoService'
import { immer } from 'zustand/middleware/immer'

interface VideoState {
  // Data state
  videos: Video[]
  selectedVideo: Video | null
  generationQueue: VideoQueue[]
  costBreakdowns: Record<string, CostBreakdown>
  videoAnalytics: Record<string, any>
  
  // UI state
  isLoading: boolean
  isGenerating: boolean
  isPublishing: boolean
  isUpdating: boolean
  isDeleting: boolean
  isLoadingQueue: boolean
  
  // Error state
  error: string | null
  
  // Pagination
  currentPage: number
  totalPages: number
  totalVideos: number
  videosPerPage: number
  
  // Filters and sorting
  filters: {
    status: 'all' | 'pending' | 'processing' | 'completed' | 'failed' | 'published'
    channelId: string | null
    searchQuery: string
    dateRange: {
      start: string | null
      end: string | null
    }
  }
  
  sortBy: 'created_at' | 'title' | 'status' | 'total_cost' | 'updated_at'
  sortOrder: 'asc' | 'desc'
  
  // Selection for bulk operations
  selectedVideoIds: string[]
  
  // Actions
  loadVideos: (params?: {
    page?: number
    limit?: number
    status?: string
    channel_id?: string
    search?: string
    sort?: string
  }) => Promise<void>
  
  loadVideo: (id: string) => Promise<Video | null>
  
  generateVideo: (request: VideoGenerationRequest) => Promise<{ video_id: string; queue_id: string } | null>
  
  updateVideo: (id: string, updates: Partial<Video>) => Promise<Video | null>
  
  deleteVideo: (id: string) => Promise<boolean>
  
  retryVideo: (id: string) => Promise<string | null>
  
  publishVideo: (id: string, scheduleTime?: string) => Promise<string | null>
  
  setSelectedVideo: (video: Video | null) => void
  
  // Queue management
  loadQueue: () => Promise<void>
  getQueueItem: (id: string) => Promise<VideoQueue | null>
  cancelGeneration: (queueId: string) => Promise<boolean>
  
  // Analytics and costs
  loadCostBreakdown: (videoId: string) => Promise<CostBreakdown | null>
  loadVideoAnalytics: (videoId: string) => Promise<any>
  
  // Bulk operations
  bulkDelete: (videoIds: string[]) => Promise<{ deleted: number; failed: string[] } | null>
  bulkPublish: (videoIds: string[]) => Promise<{ published: number; failed: string[] } | null>
  
  // Selection management
  toggleVideoSelection: (videoId: string) => void
  selectAllVideos: (videoIds: string[]) => void
  clearSelection: () => void
  
  // Filters and sorting
  setFilters: (filters: Partial<VideoState['filters']>) => void
  setSorting: (sortBy: VideoState['sortBy'], sortOrder?: VideoState['sortOrder']) => void
  resetFilters: () => void
  
  // Pagination
  setPage: (page: number) => void
  setVideosPerPage: (perPage: number) => void
  
  // Utility
  clearError: () => void
  getVideoById: (id: string) => Video | null
  getFilteredVideos: () => Video[]
  getQueueItemById: (id: string) => VideoQueue | null
  
  // Real-time updates (for WebSocket integration)
  updateVideoStatus: (videoId: string, status: Video['status'], metadata?: any) => void
  updateQueueItem: (queueId: string, updates: Partial<VideoQueue>) => void
}

export const useVideoStore = create<VideoState>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      videos: [],
      selectedVideo: null,
      generationQueue: [],
      costBreakdowns: {},
      videoAnalytics: {},
      
      isLoading: false,
      isGenerating: false,
      isPublishing: false,
      isUpdating: false,
      isDeleting: false,
      isLoadingQueue: false,
      
      error: null,
      
      currentPage: 1,
      totalPages: 1,
      totalVideos: 0,
      videosPerPage: 20,
      
      filters: {
        status: 'all',
        channelId: null,
        searchQuery: '',
        dateRange: {
          start: null,
          end: null
        }
      },
      
      sortBy: 'created_at',
      sortOrder: 'desc',
      
      selectedVideoIds: [],
      
      // Actions
      loadVideos: async (params = {}) => {
        set((state) => {
          state.isLoading = true
          state.error = null
        })
        
        try {
          const { filters, sortBy, sortOrder, currentPage, videosPerPage } = get()
          
          const queryParams = {
            page: params.page || currentPage,
            limit: params.limit || videosPerPage,
            status: params.status || (filters.status !== 'all' ? filters.status : undefined),
            channel_id: params.channel_id || filters.channelId || undefined,
            search: params.search || filters.searchQuery || undefined,
            sort: params.sort || `${sortBy}:${sortOrder}`
          }
          
          // Remove undefined values
          Object.keys(queryParams).forEach(key => {
            if (queryParams[key as keyof typeof queryParams] === undefined) {
              delete queryParams[key as keyof typeof queryParams]
            }
          })
          
          const response = await VideoService.getVideos(queryParams)
          
          set((state) => {
            state.videos = response.videos
            state.currentPage = response.page
            state.totalPages = response.total_pages
            state.totalVideos = response.total
            state.isLoading = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load videos'
            state.isLoading = false
          })
        }
      },
      
      loadVideo: async (id: string) => {
        try {
          const video = await VideoService.getVideo(id)
          set((state) => {
            // Update video in list if it exists
            const index = state.videos.findIndex(v => v.id === id)
            if (index !== -1) {
              state.videos[index] = video
            } else {
              state.videos.unshift(video)
            }
          })
          return video
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load video'
          })
          return null
        }
      },
      
      generateVideo: async (request: VideoGenerationRequest) => {
        set((state) => {
          state.isGenerating = true
          state.error = null
        })
        
        try {
          const result = await VideoService.generateVideo(request)
          
          // Refresh videos and queue
          await get().loadVideos()
          await get().loadQueue()
          
          set((state) => {
            state.isGenerating = false
          })
          
          return result
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to generate video'
            state.isGenerating = false
          })
          return null
        }
      },
      
      updateVideo: async (id: string, updates: Partial<Video>) => {
        set((state) => {
          state.isUpdating = true
          state.error = null
        })
        
        try {
          const updatedVideo = await VideoService.updateVideo(id, updates)
          set((state) => {
            const index = state.videos.findIndex(v => v.id === id)
            if (index !== -1) {
              state.videos[index] = updatedVideo
            }
            if (state.selectedVideo?.id === id) {
              state.selectedVideo = updatedVideo
            }
            state.isUpdating = false
          })
          return updatedVideo
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to update video'
            state.isUpdating = false
          })
          return null
        }
      },
      
      deleteVideo: async (id: string) => {
        set((state) => {
          state.isDeleting = true
          state.error = null
        })
        
        try {
          await VideoService.deleteVideo(id)
          set((state) => {
            state.videos = state.videos.filter(v => v.id !== id)
            state.selectedVideoIds = state.selectedVideoIds.filter(vid => vid !== id)
            if (state.selectedVideo?.id === id) {
              state.selectedVideo = null
            }
            state.isDeleting = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to delete video'
            state.isDeleting = false
          })
          return false
        }
      },
      
      retryVideo: async (id: string) => {
        try {
          const result = await VideoService.retryVideo(id)
          
          // Refresh queue and update video status
          await get().loadQueue()
          await get().loadVideo(id)
          
          return result.queue_id
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to retry video generation'
          })
          return null
        }
      },
      
      publishVideo: async (id: string, scheduleTime?: string) => {
        set((state) => {
          state.isPublishing = true
          state.error = null
        })
        
        try {
          const result = await VideoService.publishVideo(id, scheduleTime)
          
          // Update video status
          set((state) => {
            const video = state.videos.find(v => v.id === id)
            if (video) {
              video.status = 'published'
              video.youtube_video_id = result.youtube_video_id
            }
            if (state.selectedVideo?.id === id) {
              state.selectedVideo.status = 'published'
              state.selectedVideo.youtube_video_id = result.youtube_video_id
            }
            state.isPublishing = false
          })
          
          return result.youtube_video_id
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to publish video'
            state.isPublishing = false
          })
          return null
        }
      },
      
      setSelectedVideo: (video: Video | null) => {
        set((state) => {
          state.selectedVideo = video
        })
      },
      
      // Queue management
      loadQueue: async () => {
        set((state) => {
          state.isLoadingQueue = true
          state.error = null
        })
        
        try {
          const queue = await VideoService.getQueue()
          set((state) => {
            state.generationQueue = queue
            state.isLoadingQueue = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load generation queue'
            state.isLoadingQueue = false
          })
        }
      },
      
      getQueueItem: async (id: string) => {
        try {
          const queueItem = await VideoService.getQueueItem(id)
          set((state) => {
            // Update queue item if it exists
            const index = state.generationQueue.findIndex(q => q.id === id)
            if (index !== -1) {
              state.generationQueue[index] = queueItem
            } else {
              state.generationQueue.push(queueItem)
            }
          })
          return queueItem
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load queue item'
          })
          return null
        }
      },
      
      cancelGeneration: async (queueId: string) => {
        try {
          await VideoService.cancelGeneration(queueId)
          set((state) => {
            state.generationQueue = state.generationQueue.filter(q => q.id !== queueId)
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to cancel generation'
          })
          return false
        }
      },
      
      // Analytics and costs
      loadCostBreakdown: async (videoId: string) => {
        try {
          const breakdown = await VideoService.getCostBreakdown(videoId)
          set((state) => {
            state.costBreakdowns[videoId] = breakdown
          })
          return breakdown
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load cost breakdown'
          })
          return null
        }
      },
      
      loadVideoAnalytics: async (videoId: string) => {
        try {
          const analytics = await VideoService.getVideoAnalytics(videoId)
          set((state) => {
            state.videoAnalytics[videoId] = analytics
          })
          return analytics
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load video analytics'
          })
          return null
        }
      },
      
      // Bulk operations
      bulkDelete: async (videoIds: string[]) => {
        set((state) => {
          state.isDeleting = true
          state.error = null
        })
        
        try {
          const result = await VideoService.bulkDelete(videoIds)
          
          // Remove successfully deleted videos
          set((state) => {
            const successfullyDeleted = videoIds.filter(id => !result.failed.includes(id))
            state.videos = state.videos.filter(v => !successfullyDeleted.includes(v.id))
            state.selectedVideoIds = state.selectedVideoIds.filter(id => !successfullyDeleted.includes(id))
            
            if (state.selectedVideo && successfullyDeleted.includes(state.selectedVideo.id)) {
              state.selectedVideo = null
            }
            
            state.isDeleting = false
          })
          
          return result
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to delete videos'
            state.isDeleting = false
          })
          return null
        }
      },
      
      bulkPublish: async (videoIds: string[]) => {
        set((state) => {
          state.isPublishing = true
          state.error = null
        })
        
        try {
          const result = await VideoService.bulkPublish(videoIds)
          
          // Update status of successfully published videos
          set((state) => {
            const successfullyPublished = videoIds.filter(id => !result.failed.includes(id))
            successfullyPublished.forEach(id => {
              const video = state.videos.find(v => v.id === id)
              if (video) {
                video.status = 'published'
              }
            })
            
            state.isPublishing = false
          })
          
          return result
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to publish videos'
            state.isPublishing = false
          })
          return null
        }
      },
      
      // Selection management
      toggleVideoSelection: (videoId: string) => {
        set((state) => {
          const index = state.selectedVideoIds.indexOf(videoId)
          if (index === -1) {
            state.selectedVideoIds.push(videoId)
          } else {
            state.selectedVideoIds.splice(index, 1)
          }
        })
      },
      
      selectAllVideos: (videoIds: string[]) => {
        set((state) => {
          state.selectedVideoIds = [...videoIds]
        })
      },
      
      clearSelection: () => {
        set((state) => {
          state.selectedVideoIds = []
        })
      },
      
      // Filters and sorting
      setFilters: (newFilters) => {
        set((state) => {
          state.filters = { ...state.filters, ...newFilters }
          state.currentPage = 1 // Reset to first page when filtering
        })
      },
      
      setSorting: (sortBy, sortOrder = 'desc') => {
        set((state) => {
          state.sortBy = sortBy
          state.sortOrder = sortOrder
          state.currentPage = 1 // Reset to first page when sorting
        })
      },
      
      resetFilters: () => {
        set((state) => {
          state.filters = {
            status: 'all',
            channelId: null,
            searchQuery: '',
            dateRange: {
              start: null,
              end: null
            }
          }
          state.currentPage = 1
        })
      },
      
      // Pagination
      setPage: (page: number) => {
        set((state) => {
          state.currentPage = page
        })
      },
      
      setVideosPerPage: (perPage: number) => {
        set((state) => {
          state.videosPerPage = perPage
          state.currentPage = 1 // Reset to first page
        })
      },
      
      // Utility
      clearError: () => {
        set((state) => {
          state.error = null
        })
      },
      
      getVideoById: (id: string) => {
        return get().videos.find(v => v.id === id) || null
      },
      
      getFilteredVideos: () => {
        const { videos, filters } = get()
        
        return videos.filter(video => {
          // Status filter
          if (filters.status !== 'all' && video.status !== filters.status) {
            return false
          }
          
          // Channel filter
          if (filters.channelId && video.channel_id !== filters.channelId) {
            return false
          }
          
          // Search query
          if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase()
            return (
              video.title.toLowerCase().includes(query) ||
              video.description.toLowerCase().includes(query) ||
              video.topic.toLowerCase().includes(query)
            )
          }
          
          // Date range filter
          if (filters.dateRange.start || filters.dateRange.end) {
            const videoDate = new Date(video.created_at)
            if (filters.dateRange.start && videoDate < new Date(filters.dateRange.start)) {
              return false
            }
            if (filters.dateRange.end && videoDate > new Date(filters.dateRange.end)) {
              return false
            }
          }
          
          return true
        })
      },
      
      getQueueItemById: (id: string) => {
        return get().generationQueue.find(q => q.id === id) || null
      },
      
      // Real-time updates
      updateVideoStatus: (videoId: string, status: Video['status'], metadata?: any) => {
        set((state) => {
          const video = state.videos.find(v => v.id === videoId)
          if (video) {
            video.status = status
            if (metadata) {
              video.metadata = { ...video.metadata, ...metadata }
            }
            video.updated_at = new Date().toISOString()
          }
          
          if (state.selectedVideo?.id === videoId) {
            state.selectedVideo.status = status
            if (metadata) {
              state.selectedVideo.metadata = { ...state.selectedVideo.metadata, ...metadata }
            }
            state.selectedVideo.updated_at = new Date().toISOString()
          }
        })
      },
      
      updateQueueItem: (queueId: string, updates: Partial<VideoQueue>) => {
        set((state) => {
          const queueItem = state.generationQueue.find(q => q.id === queueId)
          if (queueItem) {
            Object.assign(queueItem, updates)
          }
        })
      }
    })),
    {
      name: 'video-store'
    }
  )
)