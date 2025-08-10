/**
 * Channel Store
 * Owner: Frontend Team Lead
 * 
 * Zustand store for managing channel state, including CRUD operations,
 * YouTube connections, analytics, and settings management.
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { ChannelService, type Channel, type ChannelAnalytics } from '@/services/channelService'
import { immer } from 'zustand/middleware/immer'

interface ChannelState {
  // Data state
  channels: Channel[]
  selectedChannel: Channel | null
  channelAnalytics: Record<string, ChannelAnalytics>
  
  // UI state  
  isLoading: boolean
  isCreating: boolean
  isUpdating: boolean
  isDeleting: boolean
  isConnectingYouTube: boolean
  isSyncing: boolean
  
  // Error state
  error: string | null
  
  // Filters and sorting
  filters: {
    status: 'all' | 'active' | 'inactive' | 'connecting' | 'failed'
    category: string | null
    searchQuery: string
  }
  
  sortBy: 'name' | 'created_at' | 'subscriber_count' | 'video_count'
  sortOrder: 'asc' | 'desc'
  
  // Actions
  loadChannels: () => Promise<void>
  loadChannel: (id: string) => Promise<Channel | null>
  createChannel: (channelData: {
    name: string
    description: string
    category: string
    target_audience: string
    tone: string
    language: string
  }) => Promise<Channel | null>
  updateChannel: (id: string, updates: Partial<Channel>) => Promise<Channel | null>
  deleteChannel: (id: string) => Promise<boolean>
  setSelectedChannel: (channel: Channel | null) => void
  
  // YouTube integration
  connectYouTube: (channelId: string) => Promise<string | null> // Returns auth URL
  completeYouTubeConnection: (channelId: string, authCode: string) => Promise<boolean>
  disconnectYouTube: (channelId: string) => Promise<boolean>
  syncFromYouTube: (channelId: string) => Promise<boolean>
  
  // Analytics
  loadChannelAnalytics: (channelId: string, params?: {
    start_date?: string
    end_date?: string
    metrics?: string[]
  }) => Promise<ChannelAnalytics | null>
  
  // Settings
  updateChannelSettings: (channelId: string, settings: Partial<Channel['settings']>) => Promise<boolean>
  uploadAvatar: (channelId: string, file: File) => Promise<string | null>
  uploadBanner: (channelId: string, file: File) => Promise<string | null>
  
  // Filters and sorting
  setFilters: (filters: Partial<ChannelState['filters']>) => void
  setSorting: (sortBy: ChannelState['sortBy'], sortOrder?: ChannelState['sortOrder']) => void
  resetFilters: () => void
  
  // Utility
  clearError: () => void
  getChannelById: (id: string) => Channel | null
  getFilteredChannels: () => Channel[]
  
  // Bulk operations
  bulkUpdateChannels: (channelIds: string[], updates: Partial<Channel>) => Promise<boolean>
  bulkDeleteChannels: (channelIds: string[]) => Promise<boolean>
}

export const useChannelStore = create<ChannelState>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      channels: [],
      selectedChannel: null,
      channelAnalytics: {},
      isLoading: false,
      isCreating: false,
      isUpdating: false,
      isDeleting: false,
      isConnectingYouTube: false,
      isSyncing: false,
      error: null,
      
      filters: {
        status: 'all',
        category: null,
        searchQuery: ''
      },
      sortBy: 'created_at',
      sortOrder: 'desc',
      
      // Actions
      loadChannels: async () => {
        set((state) => {
          state.isLoading = true
          state.error = null
        })
        
        try {
          const channels = await ChannelService.getChannels()
          set((state) => {
            state.channels = channels
            state.isLoading = false
          })
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load channels'
            state.isLoading = false
          })
        }
      },
      
      loadChannel: async (id: string) => {
        set((state) => {
          state.isLoading = true
          state.error = null
        })
        
        try {
          const channel = await ChannelService.getChannel(id)
          set((state) => {
            // Update channel in list if it exists
            const index = state.channels.findIndex(c => c.id === id)
            if (index !== -1) {
              state.channels[index] = channel
            } else {
              state.channels.push(channel)
            }
            state.isLoading = false
          })
          return channel
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load channel'
            state.isLoading = false
          })
          return null
        }
      },
      
      createChannel: async (channelData) => {
        set((state) => {
          state.isCreating = true
          state.error = null
        })
        
        try {
          const channel = await ChannelService.createChannel(channelData)
          set((state) => {
            state.channels.unshift(channel)
            state.selectedChannel = channel
            state.isCreating = false
          })
          return channel
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to create channel'
            state.isCreating = false
          })
          return null
        }
      },
      
      updateChannel: async (id: string, updates: Partial<Channel>) => {
        set((state) => {
          state.isUpdating = true
          state.error = null
        })
        
        try {
          const updatedChannel = await ChannelService.updateChannel(id, updates)
          set((state) => {
            const index = state.channels.findIndex(c => c.id === id)
            if (index !== -1) {
              state.channels[index] = updatedChannel
            }
            if (state.selectedChannel?.id === id) {
              state.selectedChannel = updatedChannel
            }
            state.isUpdating = false
          })
          return updatedChannel
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to update channel'
            state.isUpdating = false
          })
          return null
        }
      },
      
      deleteChannel: async (id: string) => {
        set((state) => {
          state.isDeleting = true
          state.error = null
        })
        
        try {
          await ChannelService.deleteChannel(id)
          set((state) => {
            state.channels = state.channels.filter(c => c.id !== id)
            if (state.selectedChannel?.id === id) {
              state.selectedChannel = null
            }
            state.isDeleting = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to delete channel'
            state.isDeleting = false
          })
          return false
        }
      },
      
      setSelectedChannel: (channel: Channel | null) => {
        set((state) => {
          state.selectedChannel = channel
        })
      },
      
      // YouTube integration
      connectYouTube: async (channelId: string) => {
        set((state) => {
          state.isConnectingYouTube = true
          state.error = null
        })
        
        try {
          const flow = await ChannelService.startYouTubeConnection(channelId)
          set((state) => {
            state.isConnectingYouTube = false
          })
          return flow.authorization_url
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to start YouTube connection'
            state.isConnectingYouTube = false
          })
          return null
        }
      },
      
      completeYouTubeConnection: async (channelId: string, authCode: string) => {
        set((state) => {
          state.isConnectingYouTube = true
          state.error = null
        })
        
        try {
          const updatedChannel = await ChannelService.completeYouTubeConnection(channelId, authCode)
          set((state) => {
            const index = state.channels.findIndex(c => c.id === channelId)
            if (index !== -1) {
              state.channels[index] = updatedChannel
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel = updatedChannel
            }
            state.isConnectingYouTube = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to complete YouTube connection'
            state.isConnectingYouTube = false
          })
          return false
        }
      },
      
      disconnectYouTube: async (channelId: string) => {
        try {
          await ChannelService.disconnectYouTube(channelId)
          set((state) => {
            const channel = state.channels.find(c => c.id === channelId)
            if (channel) {
              channel.youtube_channel_id = undefined
              channel.youtube_channel_url = undefined
              channel.status = 'active'
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel.youtube_channel_id = undefined
              state.selectedChannel.youtube_channel_url = undefined
              state.selectedChannel.status = 'active'
            }
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to disconnect from YouTube'
          })
          return false
        }
      },
      
      syncFromYouTube: async (channelId: string) => {
        set((state) => {
          state.isSyncing = true
          state.error = null
        })
        
        try {
          const updatedChannel = await ChannelService.syncFromYouTube(channelId)
          set((state) => {
            const index = state.channels.findIndex(c => c.id === channelId)
            if (index !== -1) {
              state.channels[index] = updatedChannel
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel = updatedChannel
            }
            state.isSyncing = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to sync from YouTube'
            state.isSyncing = false
          })
          return false
        }
      },
      
      // Analytics
      loadChannelAnalytics: async (channelId: string, params) => {
        try {
          const analytics = await ChannelService.getChannelAnalytics(channelId, params)
          set((state) => {
            state.channelAnalytics[channelId] = analytics
          })
          return analytics
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to load analytics'
          })
          return null
        }
      },
      
      // Settings
      updateChannelSettings: async (channelId: string, settings) => {
        try {
          const updatedChannel = await ChannelService.updateSettings(channelId, settings)
          set((state) => {
            const index = state.channels.findIndex(c => c.id === channelId)
            if (index !== -1) {
              state.channels[index] = updatedChannel
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel = updatedChannel
            }
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to update settings'
          })
          return false
        }
      },
      
      uploadAvatar: async (channelId: string, file: File) => {
        try {
          const result = await ChannelService.uploadAvatar(channelId, file)
          set((state) => {
            const channel = state.channels.find(c => c.id === channelId)
            if (channel) {
              channel.avatar_url = result.avatar_url
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel.avatar_url = result.avatar_url
            }
          })
          return result.avatar_url
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to upload avatar'
          })
          return null
        }
      },
      
      uploadBanner: async (channelId: string, file: File) => {
        try {
          const result = await ChannelService.uploadBanner(channelId, file)
          set((state) => {
            const channel = state.channels.find(c => c.id === channelId)
            if (channel) {
              channel.banner_url = result.banner_url
            }
            if (state.selectedChannel?.id === channelId) {
              state.selectedChannel.banner_url = result.banner_url
            }
          })
          return result.banner_url
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to upload banner'
          })
          return null
        }
      },
      
      // Filters and sorting
      setFilters: (newFilters) => {
        set((state) => {
          state.filters = { ...state.filters, ...newFilters }
        })
      },
      
      setSorting: (sortBy, sortOrder = 'desc') => {
        set((state) => {
          state.sortBy = sortBy
          state.sortOrder = sortOrder
        })
      },
      
      resetFilters: () => {
        set((state) => {
          state.filters = {
            status: 'all',
            category: null,
            searchQuery: ''
          }
        })
      },
      
      // Utility
      clearError: () => {
        set((state) => {
          state.error = null
        })
      },
      
      getChannelById: (id: string) => {
        return get().channels.find(c => c.id === id) || null
      },
      
      getFilteredChannels: () => {
        const { channels, filters, sortBy, sortOrder } = get()
        
        let filtered = channels.filter(channel => {
          // Status filter
          if (filters.status !== 'all' && channel.status !== filters.status) {
            return false
          }
          
          // Category filter
          if (filters.category && channel.category !== filters.category) {
            return false
          }
          
          // Search query
          if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase()
            return (
              channel.name.toLowerCase().includes(query) ||
              channel.description.toLowerCase().includes(query) ||
              channel.category.toLowerCase().includes(query)
            )
          }
          
          return true
        })
        
        // Sort
        filtered.sort((a, b) => {
          let comparison = 0
          
          switch (sortBy) {
            case 'name':
              comparison = a.name.localeCompare(b.name)
              break
            case 'created_at':
              comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
              break
            case 'subscriber_count':
              comparison = a.subscriber_count - b.subscriber_count
              break
            case 'video_count':
              comparison = a.video_count - b.video_count
              break
          }
          
          return sortOrder === 'desc' ? -comparison : comparison
        })
        
        return filtered
      },
      
      // Bulk operations
      bulkUpdateChannels: async (channelIds: string[], updates: Partial<Channel>) => {
        set((state) => {
          state.isUpdating = true
          state.error = null
        })
        
        try {
          const promises = channelIds.map(id => ChannelService.updateChannel(id, updates))
          const updatedChannels = await Promise.all(promises)
          
          set((state) => {
            updatedChannels.forEach(updatedChannel => {
              const index = state.channels.findIndex(c => c.id === updatedChannel.id)
              if (index !== -1) {
                state.channels[index] = updatedChannel
              }
            })
            state.isUpdating = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to update channels'
            state.isUpdating = false
          })
          return false
        }
      },
      
      bulkDeleteChannels: async (channelIds: string[]) => {
        set((state) => {
          state.isDeleting = true
          state.error = null
        })
        
        try {
          const promises = channelIds.map(id => ChannelService.deleteChannel(id))
          await Promise.all(promises)
          
          set((state) => {
            state.channels = state.channels.filter(c => !channelIds.includes(c.id))
            if (state.selectedChannel && channelIds.includes(state.selectedChannel.id)) {
              state.selectedChannel = null
            }
            state.isDeleting = false
          })
          return true
        } catch (error: any) {
          set((state) => {
            state.error = error.message || 'Failed to delete channels'
            state.isDeleting = false
          })
          return false
        }
      }
    })),
    {
      name: 'channel-store'
    }
  )
)