# Frontend State Management with Zustand

## Overview

YTEmpire's frontend uses **Zustand** as the primary state management solution, providing a lightweight, TypeScript-friendly, and performant alternative to Redux. The state management architecture follows modern React patterns with emphasis on developer experience and type safety.

## Architecture

### Store Structure

The application state is organized into five main stores:

1. **Auth Store** (`authStore.ts`) - User authentication and session management
2. **Channel Store** (`channelStore.ts`) - Channel CRUD operations and management
3. **Video Store** (`videoStore.ts`) - Video generation, management, and queue handling
4. **Analytics Store** (`analyticsStore.ts`) - Analytics data and metrics calculation
5. **UI Store** (`uiStore.ts`) - Global UI state, theme, notifications, and modals

### Key Features

- **🎯 Type Safety**: Full TypeScript support with typed actions and selectors
- **💾 Persistence**: Automatic localStorage persistence for auth and UI preferences
- **🔄 Immutable Updates**: Uses Immer for safe state mutations
- **🎨 DevTools**: Integration with Redux DevTools for debugging
- **⚡ Performance**: Optimized renders with selective subscriptions
- **🚀 Real-time**: WebSocket integration for live updates
- **🧩 Modular**: Composable hooks for different use cases

## Store Details

### Auth Store

Manages user authentication, session tokens, and user profile data.

```typescript
interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  // Actions
  login: (email: string, password: string) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => void
  refreshAccessToken: () => Promise<void>
  setUser: (user: User) => void
  clearError: () => void
}
```

**Key Features:**
- Automatic token refresh with axios interceptors
- Persistent session with localStorage
- Form validation and error handling
- Auto-login after registration

### Channel Store

Handles channel management, YouTube integration, and channel analytics.

```typescript
interface ChannelState {
  channels: Channel[]
  selectedChannel: Channel | null
  channelAnalytics: Record<string, ChannelAnalytics>
  
  // UI states
  isLoading: boolean
  isCreating: boolean
  isUpdating: boolean
  isConnectingYouTube: boolean
  
  // Filtering and sorting
  filters: {
    status: 'all' | 'active' | 'inactive' | 'connecting' | 'failed'
    category: string | null
    searchQuery: string
  }
  
  // Actions
  loadChannels: () => Promise<void>
  createChannel: (data) => Promise<Channel | null>
  updateChannel: (id, updates) => Promise<Channel | null>
  deleteChannel: (id) => Promise<boolean>
  connectYouTube: (channelId) => Promise<string | null>
  // ... more actions
}
```

**Key Features:**
- Real-time filtering and sorting
- YouTube API integration
- Bulk operations support
- Analytics data caching
- Optimistic updates

### Video Store

Manages video generation, processing queue, and video lifecycle.

```typescript
interface VideoState {
  videos: Video[]
  selectedVideo: Video | null
  generationQueue: VideoQueue[]
  costBreakdowns: Record<string, CostBreakdown>
  
  // Pagination
  currentPage: number
  totalPages: number
  totalVideos: number
  
  // Selection for bulk operations
  selectedVideoIds: string[]
  
  // Actions
  generateVideo: (request) => Promise<{video_id: string; queue_id: string} | null>
  publishVideo: (id, schedule?) => Promise<string | null>
  bulkDelete: (videoIds) => Promise<{deleted: number; failed: string[]} | null>
  // ... more actions
}
```

**Key Features:**
- Generation queue management
- Real-time status updates
- Bulk operations (delete, publish)
- Cost tracking and analytics
- Advanced filtering and pagination

### Analytics Store

Handles dashboard metrics, channel analytics, and performance data.

```typescript
interface AnalyticsState {
  dashboardMetrics: DashboardMetrics | null
  channelAnalytics: Record<string, ChannelAnalytics>
  videoAnalytics: Record<string, VideoAnalytics>
  
  // Time range management
  selectedTimeRange: '7d' | '30d' | '90d' | '1y' | 'custom'
  customDateRange: {start: string | null; end: string | null}
  
  // Actions
  loadDashboardMetrics: (timeRange?) => Promise<void>
  compareChannels: (channelIds, timeRange?) => Promise<void>
  exportAnalyticsData: (type, id?) => Promise<Blob | null>
  calculateMetrics: () => {totalROI: number; averageEngagement: number; ...}
}
```

**Key Features:**
- Time range filtering
- Data comparison tools
- Export functionality
- Real-time metric calculation
- Performance optimization

### UI Store

Manages global UI state, theme, notifications, and user preferences.

```typescript
interface UIState {
  // Theme and appearance
  theme: 'light' | 'dark' | 'system'
  primaryColor: string
  density: 'comfortable' | 'compact' | 'spacious'
  
  // Layout
  sidebarState: 'expanded' | 'collapsed' | 'hidden'
  
  // Notifications
  notifications: Notification[]
  
  // Modals
  modals: Modal[]
  
  // Preferences
  preferences: {
    autoSave: boolean
    confirmDeletions: boolean
    defaultVideoView: 'grid' | 'list'
    // ... more preferences
  }
  
  // Actions
  showNotification: (notification) => string
  openModal: (modal) => string
  setTheme: (theme) => void
  showConfirmDialog: (options) => void
}
```

**Key Features:**
- Persistent theme and preferences
- Notification queue management
- Modal state management
- Responsive breakpoint tracking
- Keyboard navigation support

## Usage Patterns

### Basic Store Usage

```typescript
import { useChannelStore } from '@/stores'

const MyComponent = () => {
  const { channels, isLoading, loadChannels } = useChannelStore()
  
  useEffect(() => {
    loadChannels()
  }, [loadChannels])
  
  return <div>{/* component JSX */}</div>
}
```

### Custom Hooks

The application provides custom hooks for common patterns:

```typescript
import { useChannels, useUI } from '@/hooks/useStores'

const ChannelManager = () => {
  const {
    channels,
    selectedChannel,
    isLoading,
    actions
  } = useChannels()
  
  const { actions: uiActions } = useUI()
  
  const handleCreate = async (data) => {
    const channel = await actions.create(data)
    if (channel) {
      uiActions.showSuccess('Channel created successfully')
    }
  }
  
  return <div>{/* component JSX */}</div>
}
```

### Bulk Operations

```typescript
const VideoManager = () => {
  const { selectedIds, actions } = useVideos()
  
  const handleBulkDelete = async () => {
    const result = await actions.bulkDelete(selectedIds)
    if (result) {
      console.log(`Deleted ${result.deleted} videos`)
      if (result.failed.length > 0) {
        console.log(`Failed to delete: ${result.failed.join(', ')}`)
      }
    }
  }
  
  return <div>{/* component JSX */}</div>
}
```

### Real-time Updates

```typescript
const useRealTimeUpdates = () => {
  const videoStore = useVideoStore()
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'video_status_update':
          videoStore.updateVideoStatus(data.video_id, data.status)
          break
        case 'queue_update':
          videoStore.updateQueueItem(data.queue_id, data.updates)
          break
      }
    }
    
    return () => ws.close()
  }, [videoStore])
}
```

## Advanced Features

### Store Composition

```typescript
const useDashboard = () => {
  const auth = useAuth()
  const { dashboard } = useAnalytics()
  const { channels } = useChannels()
  const { videos, queue } = useVideos()
  
  return {
    user: auth.user,
    dashboard,
    channels,
    videos,
    queue,
    actions: {
      refreshAll: async () => {
        // Refresh all dashboard data
        await Promise.all([
          storeActions.analytics.loadDashboard(),
          storeActions.channels.load(),
          storeActions.videos.load()
        ])
      }
    }
  }
}
```

### Computed Values

```typescript
const calculateMetrics = () => {
  const { dashboardMetrics } = get()
  
  if (!dashboardMetrics) return defaultMetrics
  
  const totalROI = dashboardMetrics.overview.total_costs > 0 
    ? (dashboardMetrics.overview.monthly_revenue / dashboardMetrics.overview.total_costs) * 100 
    : 0
    
  const costPerVideo = dashboardMetrics.overview.total_videos > 0
    ? dashboardMetrics.overview.total_costs / dashboardMetrics.overview.total_videos
    : 0
  
  return { totalROI, costPerVideo, /* ... */ }
}
```

### Middleware Integration

```typescript
export const useVideoStore = create<VideoState>()(
  devtools(
    immer((set, get) => ({
      // Store implementation
    })),
    { name: 'video-store' }
  )
)
```

## Component Examples

### Dashboard Component

```typescript
export const DashboardOverview: React.FC = () => {
  const {
    user,
    dashboard,
    metrics,
    channels,
    videos,
    queue,
    isLoading,
    actions
  } = useDashboard()
  
  const { actions: uiActions } = useUI()
  
  const handleRefresh = async () => {
    try {
      await actions.refreshAll()
      uiActions.showSuccess('Dashboard refreshed')
    } catch (error) {
      uiActions.showError('Failed to refresh dashboard')
    }
  }
  
  return (
    <Box>
      {/* Dashboard content */}
    </Box>
  )
}
```

### Notification System

```typescript
export const NotificationContainer: React.FC = () => {
  const { notifications, actions } = useNotifications()
  
  return (
    <Box>
      {notifications.map(notification => (
        <Alert
          key={notification.id}
          severity={notification.type}
          onClose={() => actions.dismiss(notification.id)}
        >
          <AlertTitle>{notification.title}</AlertTitle>
          {notification.message}
        </Alert>
      ))}
    </Box>
  )
}
```

## Performance Optimizations

### Selective Subscriptions

```typescript
// Only subscribe to specific state slices
const channels = useChannelStore(state => state.channels)
const isLoading = useChannelStore(state => state.isLoading)

// Use selectors for computed values
const filteredChannels = useChannelStore(state => state.getFilteredChannels())
```

### Memoized Selectors

```typescript
const useChannelStats = () => {
  return useChannelStore(
    useCallback(
      state => ({
        total: state.channels.length,
        active: state.channels.filter(c => c.status === 'active').length,
        connected: state.channels.filter(c => c.youtube_channel_id).length
      }),
      []
    )
  )
}
```

### Batch Updates

```typescript
const updateMultipleChannels = async (updates: Array<{id: string, data: any}>) => {
  set(state => {
    // Batch multiple updates in single render
    updates.forEach(({id, data}) => {
      const channel = state.channels.find(c => c.id === id)
      if (channel) {
        Object.assign(channel, data)
      }
    })
  })
}
```

## Testing

### Store Testing

```typescript
import { renderHook, act } from '@testing-library/react'
import { useChannelStore } from '@/stores'

describe('Channel Store', () => {
  beforeEach(() => {
    useChannelStore.setState({
      channels: [],
      isLoading: false,
      error: null
    })
  })
  
  it('should load channels', async () => {
    const { result } = renderHook(() => useChannelStore())
    
    await act(async () => {
      await result.current.loadChannels()
    })
    
    expect(result.current.channels).toHaveLength(5)
    expect(result.current.isLoading).toBe(false)
  })
  
  it('should handle errors', async () => {
    // Mock API error
    jest.spyOn(ChannelService, 'getChannels').mockRejectedValue(new Error('API Error'))
    
    const { result } = renderHook(() => useChannelStore())
    
    await act(async () => {
      await result.current.loadChannels()
    })
    
    expect(result.current.error).toBe('API Error')
  })
})
```

### Hook Testing

```typescript
import { renderHook } from '@testing-library/react'
import { useChannels } from '@/hooks/useStores'

describe('useChannels Hook', () => {
  it('should return filtered channels', () => {
    const { result } = renderHook(() => useChannels())
    
    act(() => {
      result.current.actions.setFilters({ status: 'active' })
    })
    
    const activeChannels = result.current.channels.filter(c => c.status === 'active')
    expect(result.current.channels).toEqual(activeChannels)
  })
})
```

## Migration Guide

### From useState to Zustand

**Before:**
```typescript
const [channels, setChannels] = useState<Channel[]>([])
const [isLoading, setIsLoading] = useState(false)
const [error, setError] = useState<string | null>(null)

const loadChannels = async () => {
  setIsLoading(true)
  try {
    const data = await ChannelService.getChannels()
    setChannels(data)
  } catch (err) {
    setError(err.message)
  } finally {
    setIsLoading(false)
  }
}
```

**After:**
```typescript
const {
  channels,
  isLoading,
  error,
  actions
} = useChannels()

// Loading is handled automatically by the store
useEffect(() => {
  actions.load()
}, [])
```

### From Context to Zustand

**Before:**
```typescript
const ChannelContext = createContext<ChannelContextType | undefined>(undefined)

export const ChannelProvider: React.FC = ({ children }) => {
  const [state, setState] = useState(initialState)
  
  return (
    <ChannelContext.Provider value={{ state, setState }}>
      {children}
    </ChannelContext.Provider>
  )
}
```

**After:**
```typescript
// No provider needed - Zustand stores are available globally
import { useChannelStore } from '@/stores'

const MyComponent = () => {
  const { channels, actions } = useChannelStore()
  // Direct access to store state and actions
}
```

## Best Practices

### 1. Store Organization

- **Single Responsibility**: Each store handles one domain
- **Flat Structure**: Avoid deeply nested state
- **Computed Values**: Use selectors for derived data
- **Type Safety**: Full TypeScript coverage

### 2. Action Patterns

- **Async Actions**: Handle loading states and errors
- **Optimistic Updates**: Update UI immediately, revert on failure
- **Batch Operations**: Group related updates
- **Error Handling**: Consistent error patterns across actions

### 3. Component Integration

- **Custom Hooks**: Create domain-specific hooks
- **Selective Subscriptions**: Subscribe to minimal state slices
- **Memoization**: Use React.memo and useMemo appropriately
- **Side Effects**: Keep effects in stores, not components

### 4. Performance

- **Immer Usage**: Enable safe mutations without performance cost
- **DevTools**: Use only in development
- **Persistence**: Persist only necessary state
- **Cleanup**: Remove unused subscriptions

## Debugging

### DevTools Integration

The stores integrate with Redux DevTools for debugging:

```typescript
export const useChannelStore = create<ChannelState>()(
  devtools(
    immer((set, get) => ({
      // Store implementation
    })),
    {
      name: 'channel-store',
      serialize: true
    }
  )
)
```

### Debug Utilities

```typescript
import { storeDebug } from '@/stores'

// Development debugging
if (process.env.NODE_ENV === 'development') {
  // Log all store states
  storeDebug.logAllStates()
  
  // Subscribe to all changes
  storeDebug.subscribeToAll()
}
```

### Common Issues

1. **State Not Updating**: Check if using immer correctly
2. **Memory Leaks**: Ensure WebSocket connections are cleaned up
3. **Type Errors**: Verify store interfaces match implementation
4. **Persistence Issues**: Check localStorage quotas and serialization

## Future Enhancements

### Planned Features

1. **Offline Support**: Cache actions when offline, sync when online
2. **State Snapshots**: Export/import functionality for debugging
3. **Middleware System**: Plugin architecture for custom functionality
4. **Performance Monitoring**: Track render performance and optimization
5. **A/B Testing**: Store-level experiment management

### Architecture Evolution

1. **Micro-frontends**: Store federation across applications
2. **Server State Sync**: Real-time synchronization with backend
3. **State Machines**: XState integration for complex workflows
4. **GraphQL Integration**: Apollo Client state management

---

**Owner**: Frontend Team Lead  
**Last Updated**: 2025-08-10  
**Version**: 1.0.0

This documentation covers the complete Zustand state management implementation for YTEmpire's frontend, providing both technical reference and practical guidance for development teams.