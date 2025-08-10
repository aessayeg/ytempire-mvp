# YTEMPIRE State Management & API Integration Guide
## Zustand Stores & API Communication Patterns

**Document Version**: 1.0  
**Role**: React Engineer  
**Scope**: State Management with Zustand & API Integration

---

## 🏪 Zustand State Architecture

### Core Principles
1. **Simplicity First**: No Redux complexity
2. **Modular Stores**: Separate stores by domain
3. **TypeScript Safety**: Full type coverage
4. **Performance**: Selective subscriptions
5. **Persistence**: Critical data only

### Store Structure Overview
```typescript
// Store organization
stores/
├── useAuthStore.ts       // Authentication & user data
├── useChannelStore.ts    // Channel management
├── useVideoStore.ts      // Video generation & status
├── useDashboardStore.ts  // Dashboard metrics
├── useCostStore.ts       // Cost tracking
├── useNotificationStore.ts // In-app notifications
└── useSettingsStore.ts   // User preferences
```

---

## 🔐 Authentication Store

```typescript
// stores/useAuthStore.ts
import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';
import { authApi } from '@/services/auth';

interface User {
  id: string;
  email: string;
  role: string;
  channelLimit: number;
  subscription: {
    plan: 'free' | 'pro' | 'enterprise';
    videosPerMonth: number;
    expiresAt: string;
  };
}

interface AuthStore {
  // State
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshAuth: () => Promise<void>;
  updateUser: (updates: Partial<User>) => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        user: null,
        accessToken: null,
        refreshToken: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
        
        // Login action
        login: async (email, password) => {
          set({ isLoading: true, error: null });
          
          try {
            const response = await authApi.login({ email, password });
            
            set({
              user: response.user,
              accessToken: response.accessToken,
              refreshToken: response.refreshToken,
              isAuthenticated: true,
              isLoading: false,
            });
            
            // Setup token refresh timer
            const expiresIn = response.expiresIn * 1000; // Convert to ms
            setTimeout(() => get().refreshAuth(), expiresIn - 60000); // Refresh 1 min early
            
          } catch (error) {
            set({
              error: error.message || 'Login failed',
              isLoading: false,
              isAuthenticated: false,
            });
            throw error;
          }
        },
        
        // Logout action
        logout: () => {
          set({
            user: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
            error: null,
          });
          
          // Clear other stores
          useChannelStore.getState().reset();
          useVideoStore.getState().reset();
          useDashboardStore.getState().reset();
        },
        
        // Refresh token action
        refreshAuth: async () => {
          const { refreshToken } = get();
          if (!refreshToken) return;
          
          try {
            const response = await authApi.refresh({ refreshToken });
            
            set({
              accessToken: response.accessToken,
              refreshToken: response.refreshToken,
            });
            
            // Setup next refresh
            const expiresIn = response.expiresIn * 1000;
            setTimeout(() => get().refreshAuth(), expiresIn - 60000);
            
          } catch (error) {
            // Refresh failed, logout user
            get().logout();
          }
        },
        
        // Update user data
        updateUser: (updates) => {
          set((state) => ({
            user: state.user ? { ...state.user, ...updates } : null,
          }));
        },
        
        // Clear error
        clearError: () => set({ error: null }),
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({
          user: state.user,
          refreshToken: state.refreshToken,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// Selector hooks for performance
export const useUser = () => useAuthStore((state) => state.user);
export const useIsAuthenticated = () => useAuthStore((state) => state.isAuthenticated);
```

---

## 📺 Channel Store

```typescript
// stores/useChannelStore.ts
import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { channelApi } from '@/services/channels';

interface Channel {
  id: string;
  name: string;
  youtubeChannelId: string;
  niche: string;
  status: 'active' | 'paused' | 'error';
  automationEnabled: boolean;
  dailyVideoLimit: number;
  statistics: {
    videoCount: number;
    totalViews: number;
    subscribers: number;
    dailyRevenue: number;
    monthlyRevenue: number;
  };
  createdAt: string;
  updatedAt: string;
}

interface ChannelStore {
  // State
  channels: Channel[];
  activeChannelId: string | null;
  loading: boolean;
  error: string | null;
  lastFetch: number | null;
  
  // Computed getters
  activeChannel: Channel | null;
  activeChannels: Channel[];
  
  // Actions
  fetchChannels: () => Promise<void>;
  createChannel: (data: CreateChannelData) => Promise<Channel>;
  updateChannel: (id: string, updates: Partial<Channel>) => Promise<void>;
  deleteChannel: (id: string) => Promise<void>;
  setActiveChannel: (id: string | null) => void;
  toggleAutomation: (id: string) => Promise<void>;
  reset: () => void;
}

export const useChannelStore = create<ChannelStore>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      // Initial state
      channels: [],
      activeChannelId: null,
      loading: false,
      error: null,
      lastFetch: null,
      
      // Computed getters
      get activeChannel() {
        const { channels, activeChannelId } = get();
        return channels.find(ch => ch.id === activeChannelId) || null;
      },
      
      get activeChannels() {
        return get().channels.filter(ch => ch.status === 'active');
      },
      
      // Fetch all channels
      fetchChannels: async () => {
        // Check cache (60 second cache)
        const { lastFetch } = get();
        if (lastFetch && Date.now() - lastFetch < 60000) {
          return; // Use cached data
        }
        
        set({ loading: true, error: null });
        
        try {
          const channels = await channelApi.getChannels();
          
          set({
            channels,
            loading: false,
            lastFetch: Date.now(),
          });
          
        } catch (error) {
          set({
            error: error.message || 'Failed to fetch channels',
            loading: false,
          });
          throw error;
        }
      },
      
      // Create new channel
      createChannel: async (data) => {
        set({ loading: true, error: null });
        
        try {
          const newChannel = await channelApi.createChannel(data);
          
          set((state) => ({
            channels: [...state.channels, newChannel],
            loading: false,
            activeChannelId: newChannel.id,
          }));
          
          return newChannel;
          
        } catch (error) {
          set({
            error: error.message || 'Failed to create channel',
            loading: false,
          });
          throw error;
        }
      },
      
      // Update channel
      updateChannel: async (id, updates) => {
        // Optimistic update
        set((state) => ({
          channels: state.channels.map(ch =>
            ch.id === id ? { ...ch, ...updates } : ch
          ),
        }));
        
        try {
          await channelApi.updateChannel(id, updates);
          // Refetch to ensure consistency
          await get().fetchChannels();
          
        } catch (error) {
          // Revert optimistic update
          await get().fetchChannels();
          throw error;
        }
      },
      
      // Delete channel
      deleteChannel: async (id) => {
        try {
          await channelApi.deleteChannel(id);
          
          set((state) => ({
            channels: state.channels.filter(ch => ch.id !== id),
            activeChannelId: state.activeChannelId === id ? null : state.activeChannelId,
          }));
          
        } catch (error) {
          set({ error: error.message || 'Failed to delete channel' });
          throw error;
        }
      },
      
      // Set active channel
      setActiveChannel: (id) => {
        set({ activeChannelId: id });
      },
      
      // Toggle automation
      toggleAutomation: async (id) => {
        const channel = get().channels.find(ch => ch.id === id);
        if (!channel) return;
        
        const newStatus = !channel.automationEnabled;
        
        // Optimistic update
        set((state) => ({
          channels: state.channels.map(ch =>
            ch.id === id ? { ...ch, automationEnabled: newStatus } : ch
          ),
        }));
        
        try {
          await channelApi.toggleAutomation(id, newStatus);
          
        } catch (error) {
          // Revert on failure
          set((state) => ({
            channels: state.channels.map(ch =>
              ch.id === id ? { ...ch, automationEnabled: !newStatus } : ch
            ),
          }));
          throw error;
        }
      },
      
      // Reset store
      reset: () => {
        set({
          channels: [],
          activeChannelId: null,
          loading: false,
          error: null,
          lastFetch: null,
        });
      },
    })),
    { name: 'ChannelStore' }
  )
);

// Performance-optimized selectors
export const useActiveChannel = () => useChannelStore((state) => state.activeChannel);
export const useChannelById = (id: string) => 
  useChannelStore((state) => state.channels.find(ch => ch.id === id));
```

---

## 🎬 Video Store

```typescript
// stores/useVideoStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { videoApi } from '@/services/videos';

interface Video {
  id: string;
  channelId: string;
  title: string;
  description: string;
  thumbnailUrl: string;
  youtubeUrl: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  stage?: 'script' | 'audio' | 'rendering' | 'uploading';
  progress: number;
  cost: {
    script: number;
    audio: number;
    rendering: number;
    total: number;
  };
  metrics: {
    views: number;
    likes: number;
    comments: number;
    revenue: number;
  };
  error?: {
    stage: string;
    message: string;
    retryable: boolean;
  };
  createdAt: string;
  publishedAt?: string;
}

interface VideoGenerationParams {
  channelId: string;
  topic?: string;
  style: 'educational' | 'entertainment' | 'tutorial';
  length: 'short' | 'medium' | 'long';
  priority: number;
}

interface VideoStore {
  // State
  videos: Video[];
  queue: Video[];
  processing: Video[];
  recentVideos: Video[];
  loading: boolean;
  error: string | null;
  
  // Polling state
  isPolling: boolean;
  pollInterval: NodeJS.Timeout | null;
  
  // Actions
  fetchVideos: (channelId?: string) => Promise<void>;
  generateVideo: (params: VideoGenerationParams) => Promise<Video>;
  fetchVideoStatus: (videoId: string) => Promise<void>;
  fetchQueue: () => Promise<void>;
  retryVideo: (videoId: string) => Promise<void>;
  cancelVideo: (videoId: string) => Promise<void>;
  
  // Polling
  startPolling: () => void;
  stopPolling: () => void;
  
  // Reset
  reset: () => void;
}

export const useVideoStore = create<VideoStore>()(
  devtools((set, get) => ({
    // Initial state
    videos: [],
    queue: [],
    processing: [],
    recentVideos: [],
    loading: false,
    error: null,
    isPolling: false,
    pollInterval: null,
    
    // Fetch videos
    fetchVideos: async (channelId) => {
      set({ loading: true, error: null });
      
      try {
        const videos = await videoApi.getVideos({ channelId });
        
        set({
          videos,
          recentVideos: videos.slice(0, 10),
          loading: false,
        });
        
      } catch (error) {
        set({
          error: error.message || 'Failed to fetch videos',
          loading: false,
        });
      }
    },
    
    // Generate new video
    generateVideo: async (params) => {
      set({ loading: true, error: null });
      
      try {
        const response = await videoApi.generateVideo(params);
        
        // Add to queue
        set((state) => ({
          queue: [...state.queue, response],
          loading: false,
        }));
        
        // Start polling if not already
        if (!get().isPolling) {
          get().startPolling();
        }
        
        return response;
        
      } catch (error) {
        set({
          error: error.message || 'Failed to generate video',
          loading: false,
        });
        throw error;
      }
    },
    
    // Fetch single video status
    fetchVideoStatus: async (videoId) => {
      try {
        const status = await videoApi.getVideoStatus(videoId);
        
        set((state) => ({
          videos: state.videos.map(v =>
            v.id === videoId ? { ...v, ...status } : v
          ),
          queue: state.queue.map(v =>
            v.id === videoId ? { ...v, ...status } : v
          ),
          processing: state.processing.map(v =>
            v.id === videoId ? { ...v, ...status } : v
          ),
        }));
        
        // Move between queues based on status
        if (status.status === 'processing') {
          set((state) => ({
            queue: state.queue.filter(v => v.id !== videoId),
            processing: [...state.processing, status],
          }));
        } else if (status.status === 'completed' || status.status === 'failed') {
          set((state) => ({
            processing: state.processing.filter(v => v.id !== videoId),
            videos: [status, ...state.videos],
          }));
        }
        
      } catch (error) {
        console.error('Failed to fetch video status:', error);
      }
    },
    
    // Fetch queue status
    fetchQueue: async () => {
      try {
        const queueStatus = await videoApi.getQueue();
        
        set({
          queue: queueStatus.queue,
          processing: queueStatus.processing,
        });
        
      } catch (error) {
        console.error('Failed to fetch queue:', error);
      }
    },
    
    // Retry failed video
    retryVideo: async (videoId) => {
      try {
        await videoApi.retryVideo(videoId);
        await get().fetchVideoStatus(videoId);
        
      } catch (error) {
        set({ error: error.message || 'Failed to retry video' });
        throw error;
      }
    },
    
    // Cancel video
    cancelVideo: async (videoId) => {
      try {
        await videoApi.cancelVideo(videoId);
        
        set((state) => ({
          queue: state.queue.filter(v => v.id !== videoId),
          processing: state.processing.filter(v => v.id !== videoId),
        }));
        
      } catch (error) {
        set({ error: error.message || 'Failed to cancel video' });
        throw error;
      }
    },
    
    // Start polling for updates
    startPolling: () => {
      const { isPolling } = get();
      if (isPolling) return;
      
      set({ isPolling: true });
      
      // Poll every 5 seconds for active videos
      const interval = setInterval(async () => {
        const { queue, processing } = get();
        const activeVideos = [...queue, ...processing];
        
        if (activeVideos.length === 0) {
          get().stopPolling();
          return;
        }
        
        // Fetch status for all active videos
        await Promise.all(
          activeVideos.map(video => get().fetchVideoStatus(video.id))
        );
        
      }, 5000); // 5 second polling for active videos
      
      set({ pollInterval: interval });
    },
    
    // Stop polling
    stopPolling: () => {
      const { pollInterval } = get();
      
      if (pollInterval) {
        clearInterval(pollInterval);
      }
      
      set({
        isPolling: false,
        pollInterval: null,
      });
    },
    
    // Reset store
    reset: () => {
      get().stopPolling();
      
      set({
        videos: [],
        queue: [],
        processing: [],
        recentVideos: [],
        loading: false,
        error: null,
      });
    },
  })),
  { name: 'VideoStore' }
);

// Selectors
export const useVideoQueue = () => useVideoStore((state) => state.queue);
export const useProcessingVideos = () => useVideoStore((state) => state.processing);
export const useVideosByChannel = (channelId: string) =>
  useVideoStore((state) => state.videos.filter(v => v.channelId === channelId));
```

---

## 📊 Dashboard Store

```typescript
// stores/useDashboardStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { dashboardApi } from '@/services/dashboard';

interface DashboardMetrics {
  totalChannels: number;
  activeChannels: number;
  videosToday: number;
  videosProcessing: number;
  totalRevenue: number;
  monthlyRevenue: number;
  dailyCost: number;
  monthlyCost: number;
  automationPercentage: number;
  avgVideoTime: number;
  successRate: number;
}

interface ChartData {
  date: string;
  revenue: number;
  cost: number;
  videos: number;
}

interface DashboardStore {
  // State
  metrics: DashboardMetrics | null;
  chartData: ChartData[];
  loading: boolean;
  error: string | null;
  lastUpdate: number | null;
  
  // Polling
  pollInterval: NodeJS.Timeout | null;
  
  // Actions
  fetchDashboard: () => Promise<void>;
  startPolling: () => void;
  stopPolling: () => void;
  reset: () => void;
}

export const useDashboardStore = create<DashboardStore>()(
  devtools((set, get) => ({
    // Initial state
    metrics: null,
    chartData: [],
    loading: false,
    error: null,
    lastUpdate: null,
    pollInterval: null,
    
    // Fetch dashboard data
    fetchDashboard: async () => {
      set({ loading: true, error: null });
      
      try {
        const response = await dashboardApi.getOverview();
        
        set({
          metrics: response.metrics,
          chartData: response.chartData,
          loading: false,
          lastUpdate: Date.now(),
        });
        
      } catch (error) {
        set({
          error: error.message || 'Failed to fetch dashboard',
          loading: false,
        });
      }
    },
    
    // Start 60-second polling
    startPolling: () => {
      const { pollInterval } = get();
      if (pollInterval) return;
      
      // Initial fetch
      get().fetchDashboard();
      
      // Poll every 60 seconds
      const interval = setInterval(() => {
        get().fetchDashboard();
      }, 60000);
      
      set({ pollInterval: interval });
    },
    
    // Stop polling
    stopPolling: () => {
      const { pollInterval } = get();
      
      if (pollInterval) {
        clearInterval(pollInterval);
      }
      
      set({ pollInterval: null });
    },
    
    // Reset store
    reset: () => {
      get().stopPolling();
      
      set({
        metrics: null,
        chartData: [],
        loading: false,
        error: null,
        lastUpdate: null,
      });
    },
  })),
  { name: 'DashboardStore' }
);

// Selectors for specific metrics
export const useTotalRevenue = () => 
  useDashboardStore((state) => state.metrics?.totalRevenue || 0);
export const useDailyCost = () => 
  useDashboardStore((state) => state.metrics?.dailyCost || 0);
export const useAutomationPercentage = () => 
  useDashboardStore((state) => state.metrics?.automationPercentage || 0);
```

---

## 💰 Cost Store

```typescript
// stores/useCostStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { costApi } from '@/services/costs';

interface CostBreakdown {
  ai_generation: number;
  voice_synthesis: number;
  storage: number;
  api_calls: number;
  total: number;
}

interface CostAlert {
  id: string;
  type: 'warning' | 'critical';
  threshold: number;
  currentCost: number;
  message: string;
  timestamp: string;
}

interface CostStore {
  // State
  currentCost: number;
  breakdown: CostBreakdown | null;
  dailyLimit: number;
  monthlyLimit: number;
  alerts: CostAlert[];
  loading: boolean;
  
  // Actions
  fetchCosts: () => Promise<void>;
  fetchBreakdown: () => Promise<void>;
  setLimits: (daily: number, monthly: number) => void;
  dismissAlert: (alertId: string) => void;
  checkAlerts: () => void;
}

export const useCostStore = create<CostStore>()(
  devtools((set, get) => ({
    // Initial state
    currentCost: 0,
    breakdown: null,
    dailyLimit: 50,
    monthlyLimit: 1500,
    alerts: [],
    loading: false,
    
    // Fetch current costs
    fetchCosts: async () => {
      set({ loading: true });
      
      try {
        const response = await costApi.getDailyCosts();
        
        set({
          currentCost: response.totalCost,
          loading: false,
        });
        
        // Check for alerts
        get().checkAlerts();
        
      } catch (error) {
        set({ loading: false });
        console.error('Failed to fetch costs:', error);
      }
    },
    
    // Fetch cost breakdown
    fetchBreakdown: async () => {
      try {
        const breakdown = await costApi.getCostBreakdown();
        set({ breakdown });
        
      } catch (error) {
        console.error('Failed to fetch breakdown:', error);
      }
    },
    
    // Set spending limits
    setLimits: (daily, monthly) => {
      set({ dailyLimit: daily, monthlyLimit: monthly });
      get().checkAlerts();
    },
    
    // Dismiss alert
    dismissAlert: (alertId) => {
      set((state) => ({
        alerts: state.alerts.filter(a => a.id !== alertId),
      }));
    },
    
    // Check for cost alerts
    checkAlerts: () => {
      const { currentCost, dailyLimit, alerts } = get();
      const newAlerts: CostAlert[] = [];
      
      // Warning at 80%
      if (currentCost >= dailyLimit * 0.8 && currentCost < dailyLimit) {
        const existingWarning = alerts.find(a => a.type === 'warning');
        if (!existingWarning) {
          newAlerts.push({
            id: `warning-${Date.now()}`,
            type: 'warning',
            threshold: dailyLimit * 0.8,
            currentCost,
            message: `Daily cost approaching limit: $${currentCost.toFixed(2)} of $${dailyLimit}`,
            timestamp: new Date().toISOString(),
          });
        }
      }
      
      // Critical at 95%
      if (currentCost >= dailyLimit * 0.95) {
        const existingCritical = alerts.find(a => a.type === 'critical');
        if (!existingCritical) {
          newAlerts.push({
            id: `critical-${Date.now()}`,
            type: 'critical',
            threshold: dailyLimit * 0.95,
            currentCost,
            message: `CRITICAL: Daily cost limit nearly reached: $${currentCost.toFixed(2)} of $${dailyLimit}`,
            timestamp: new Date().toISOString(),
          });
        }
      }
      
      if (newAlerts.length > 0) {
        set((state) => ({
          alerts: [...state.alerts, ...newAlerts],
        }));
      }
    },
  })),
  { name: 'CostStore' }
);

// Selectors
export const useCostAlerts = () => useCostStore((state) => state.alerts);
export const useIsCostWarning = () => 
  useCostStore((state) => state.currentCost >= state.dailyLimit * 0.8);
```

---

## 🌐 API Service Layer

### Base API Client

```typescript
// services/api.ts
import { useAuthStore } from '@/stores/useAuthStore';

class ApiClient {
  private baseURL: string;
  private timeout: number;
  
  constructor() {
    this.baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
    this.timeout = 30000; // 30 seconds
  }
  
  private getHeaders(): HeadersInit {
    const token = useAuthStore.getState().accessToken;
    
    return {
      'Content-Type': 'application/json',
      'X-Client-Version': '1.0.0',
      'X-Request-ID': crypto.randomUUID(),
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  }
  
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Unknown error' }));
      
      // Handle specific error codes
      if (response.status === 401) {
        // Token expired, try refresh
        await useAuthStore.getState().refreshAuth();
        // Retry original request
        throw new Error('RETRY_REQUEST');
      }
      
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    
    return response.json();
  }
  
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const url = new URL(`${this.baseURL}${endpoint}`);
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: this.getHeaders(),
        signal: controller.signal,
      });
      
      return await this.handleResponse<T>(response);
      
    } catch (error) {
      if (error.message === 'RETRY_REQUEST') {
        // Retry after token refresh
        const response = await fetch(url.toString(), {
          method: 'GET',
          headers: this.getHeaders(),
        });
        return await this.handleResponse<T>(response);
      }
      throw error;
      
    } finally {
      clearTimeout(timeoutId);
    }
  }
  
  async post<T>(endpoint: string, data?: any): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(data),
        signal: controller.signal,
      });
      
      return await this.handleResponse<T>(response);
      
    } catch (error) {
      if (error.message === 'RETRY_REQUEST') {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
          method: 'POST',
          headers: this.getHeaders(),
          body: JSON.stringify(data),
        });
        return await this.handleResponse<T>(response);
      }
      throw error;
      
    } finally {
      clearTimeout(timeoutId);
    }
  }
  
  async put<T>(endpoint: string, data?: any): Promise<T> {
    // Similar to post
  }
  
  async patch<T>(endpoint: string, data?: any): Promise<T> {
    // Similar to post
  }
  
  async delete<T>(endpoint: string): Promise<T> {
    // Similar to get
  }
}

export const apiClient = new ApiClient();
```

### Channel API Service

```typescript
// services/channels.ts
import { apiClient } from './api';

interface CreateChannelData {
  name: string;
  niche: string;
  dailyVideoLimit: number;
}

interface UpdateChannelData {
  name?: string;
  status?: 'active' | 'paused';
  dailyVideoLimit?: number;
}

class ChannelApi {
  async getChannels(): Promise<Channel[]> {
    return apiClient.get<Channel[]>('/channels');
  }
  
  async getChannel(id: string): Promise<Channel> {
    return apiClient.get<Channel>(`/channels/${id}`);
  }
  
  async createChannel(data: CreateChannelData): Promise<Channel> {
    return apiClient.post<Channel>('/channels', data);
  }
  
  async updateChannel(id: string, data: UpdateChannelData): Promise<Channel> {
    return apiClient.patch<Channel>(`/channels/${id}`, data);
  }
  
  async deleteChannel(id: string): Promise<void> {
    return apiClient.delete<void>(`/channels/${id}`);
  }
  
  async toggleAutomation(id: string, enabled: boolean): Promise<Channel> {
    return apiClient.patch<Channel>(`/channels/${id}/automation`, { enabled });
  }
  
  async getChannelStats(id: string): Promise<ChannelStatistics> {
    return apiClient.get<ChannelStatistics>(`/channels/${id}/stats`);
  }
}

export const channelApi = new ChannelApi();

### Video API Service

```typescript
// services/videos.ts
import { apiClient } from './api';

interface VideoGenerationParams {
  channelId: string;
  topic?: string;
  style: 'educational' | 'entertainment' | 'tutorial';
  length: 'short' | 'medium' | 'long';
  priority: number;
}

interface VideoGenerationResponse {
  videoId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  queuePosition: number;
  estimatedCompletion: string;
  estimatedCost: number;
}

class VideoApi {
  async getVideos(channelId?: string): Promise<Video[]> {
    return apiClient.get<Video[]>('/videos', { channelId });
  }
  
  async getVideoStatus(id: string): Promise<VideoStatus> {
    return apiClient.get<VideoStatus>(`/videos/${id}/status`);
  }
  
  async generateVideo(params: VideoGenerationParams): Promise<VideoGenerationResponse> {
    return apiClient.post<VideoGenerationResponse>('/videos/generate', params);
  }
  
  async getQueue(): Promise<QueueStatus> {
    return apiClient.get<QueueStatus>('/videos/queue');
  }
  
  async retryVideo(id: string): Promise<void> {
    return apiClient.post<void>(`/videos/${id}/retry`);
  }
  
  async cancelVideo(id: string): Promise<void> {
    return apiClient.post<void>(`/videos/${id}/cancel`);
  }
}

export const videoApi = new VideoApi();
```

### Authentication API Service

```typescript
// services/auth.ts
import { apiClient } from './api';

interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

interface RefreshTokenRequest {
  refreshToken: string;
}

interface RefreshTokenResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

class AuthApi {
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    return apiClient.post<LoginResponse>('/auth/login', credentials);
  }
  
  async logout(): Promise<void> {
    return apiClient.post<void>('/auth/logout');
  }
  
  async refresh(data: RefreshTokenRequest): Promise<RefreshTokenResponse> {
    return apiClient.post<RefreshTokenResponse>('/auth/refresh', data);
  }
  
  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>('/auth/me');
  }
  
  async updatePassword(oldPassword: string, newPassword: string): Promise<void> {
    return apiClient.post<void>('/auth/password', { oldPassword, newPassword });
  }
}

export const authApi = new AuthApi();
```

---

## 🔄 Custom Hooks for Data Fetching

### useQuery Hook Implementation

```typescript
// hooks/useQuery.ts
import { useState, useEffect, useCallback, useRef } from 'react';
import { errorHandler } from '@/services/errorHandler';

interface UseQueryOptions<T> {
  enabled?: boolean;
  refetchInterval?: number;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  retry?: number;
  retryDelay?: number;
}

export function useQuery<T>(
  key: string,
  queryFn: () => Promise<T>,
  options: UseQueryOptions<T> = {}
) {
  const {
    enabled = true,
    refetchInterval,
    onSuccess,
    onError,
    retry = 3,
    retryDelay = 1000,
  } = options;
  
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const [isRefetching, setIsRefetching] = useState(false);
  
  const retryCountRef = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout>();
  
  const execute = useCallback(async (isRefetch = false) => {
    if (!enabled) return;
    
    if (isRefetch) {
      setIsRefetching(true);
    } else {
      setLoading(true);
    }
    
    setError(null);
    
    try {
      const result = await queryFn();
      setData(result);
      retryCountRef.current = 0;
      
      if (onSuccess) {
        onSuccess(result);
      }
    } catch (err) {
      const error = err as Error;
      
      if (retryCountRef.current < retry) {
        retryCountRef.current++;
        setTimeout(() => execute(isRefetch), retryDelay * retryCountRef.current);
        return;
      }
      
      setError(error);
      errorHandler.handle(error);
      
      if (onError) {
        onError(error);
      }
    } finally {
      setLoading(false);
      setIsRefetching(false);
    }
  }, [enabled, queryFn, onSuccess, onError, retry, retryDelay]);
  
  const refetch = useCallback(() => {
    execute(true);
  }, [execute]);
  
  // Initial fetch
  useEffect(() => {
    execute();
  }, [key, enabled]);
  
  // Refetch interval
  useEffect(() => {
    if (refetchInterval && enabled) {
      intervalRef.current = setInterval(() => {
        execute(true);
      }, refetchInterval);
      
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [refetchInterval, enabled, execute]);
  
  return {
    data,
    error,
    loading,
    isRefetching,
    refetch,
  };
}
```

### useMutation Hook Implementation

```typescript
// hooks/useMutation.ts
import { useState, useCallback } from 'react';
import { errorHandler } from '@/services/errorHandler';

interface UseMutationOptions<TData, TVariables> {
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (error: Error, variables: TVariables) => void;
  onSettled?: (data: TData | undefined, error: Error | null, variables: TVariables) => void;
}

export function useMutation<TData = unknown, TVariables = void>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options: UseMutationOptions<TData, TVariables> = {}
) {
  const [data, setData] = useState<TData | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  
  const mutate = useCallback(async (variables: TVariables) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await mutationFn(variables);
      setData(result);
      
      if (options.onSuccess) {
        options.onSuccess(result, variables);
      }
      
      if (options.onSettled) {
        options.onSettled(result, null, variables);
      }
      
      return result;
    } catch (err) {
      const error = err as Error;
      setError(error);
      errorHandler.handle(error);
      
      if (options.onError) {
        options.onError(error, variables);
      }
      
      if (options.onSettled) {
        options.onSettled(undefined, error, variables);
      }
      
      throw error;
    } finally {
      setLoading(false);
    }
  }, [mutationFn, options]);
  
  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);
  
  return {
    mutate,
    data,
    error,
    loading,
    reset,
  };
}
```

---

## 🔗 Integration Examples

### Dashboard Integration

```typescript
// pages/Dashboard/useDashboardData.ts
import { useEffect } from 'react';
import { useDashboardStore } from '@/stores/useDashboardStore';
import { useChannelStore } from '@/stores/useChannelStore';
import { useVideoStore } from '@/stores/useVideoStore';
import { useCostStore } from '@/stores/useCostStore';
import { usePolling } from '@/hooks/usePolling';

export const useDashboardData = () => {
  const dashboard = useDashboardStore();
  const channels = useChannelStore();
  const videos = useVideoStore();
  const costs = useCostStore();
  
  // Initial data fetch
  useEffect(() => {
    Promise.all([
      dashboard.fetchDashboard(),
      channels.fetchChannels(),
      videos.fetchVideos(),
      costs.fetchCosts(),
    ]);
  }, []);
  
  // Setup polling for different intervals
  usePolling(dashboard.fetchDashboard, { interval: 60000 }); // 1 minute
  usePolling(costs.fetchCosts, { interval: 30000 }); // 30 seconds
  
  // Poll video queue only when processing
  const hasProcessingVideos = videos.processing.length > 0;
  usePolling(videos.fetchQueue, {
    interval: 5000,
    enabled: hasProcessingVideos,
  });
  
  return {
    metrics: dashboard.metrics,
    channels: channels.activeChannels,
    recentVideos: videos.recentVideos,
    currentCost: costs.currentCost,
    costAlerts: costs.alerts,
    loading: dashboard.loading || channels.loading,
  };
};
```

### Channel Management Integration

```typescript
// pages/Channels/useChannelManagement.ts
import { useCallback } from 'react';
import { useChannelStore } from '@/stores/useChannelStore';
import { useMutation } from '@/hooks/useMutation';
import { channelApi } from '@/services/channels';
import { useNotificationStore } from '@/stores/useNotificationStore';

export const useChannelManagement = () => {
  const { channels, fetchChannels, setActiveChannel } = useChannelStore();
  const { showNotification } = useNotificationStore();
  
  const createChannelMutation = useMutation(
    (data: CreateChannelData) => channelApi.createChannel(data),
    {
      onSuccess: (channel) => {
        showNotification({
          type: 'success',
          message: `Channel "${channel.name}" created successfully!`,
        });
        fetchChannels(); // Refresh list
        setActiveChannel(channel.id);
      },
      onError: (error) => {
        showNotification({
          type: 'error',
          message: `Failed to create channel: ${error.message}`,
        });
      },
    }
  );
  
  const toggleAutomationMutation = useMutation(
    ({ id, enabled }: { id: string; enabled: boolean }) =>
      channelApi.toggleAutomation(id, enabled),
    {
      onSuccess: (_, variables) => {
        const action = variables.enabled ? 'enabled' : 'disabled';
        showNotification({
          type: 'success',
          message: `Automation ${action} successfully!`,
        });
      },
    }
  );
  
  const deleteChannelMutation = useMutation(
    (id: string) => channelApi.deleteChannel(id),
    {
      onSuccess: () => {
        showNotification({
          type: 'success',
          message: 'Channel deleted successfully!',
        });
        fetchChannels();
      },
    }
  );
  
  return {
    channels,
    createChannel: createChannelMutation.mutate,
    toggleAutomation: toggleAutomationMutation.mutate,
    deleteChannel: deleteChannelMutation.mutate,
    loading: createChannelMutation.loading || 
             toggleAutomationMutation.loading || 
             deleteChannelMutation.loading,
  };
};
```

---

## 🎯 Best Practices Summary

### State Management Best Practices

1. **Store Organization**
   - One store per domain (auth, channels, videos, etc.)
   - Keep stores focused and single-purpose
   - Use computed getters for derived state
   - Implement reset methods for cleanup

2. **Performance Optimization**
   - Use selector hooks to prevent unnecessary re-renders
   - Implement caching for expensive operations
   - Debounce rapid state updates
   - Clear intervals and timeouts on cleanup

3. **Error Handling**
   - Centralized error handling in stores
   - Graceful fallbacks for failed operations
   - User-friendly error messages
   - Retry logic for transient failures

4. **Data Synchronization**
   - Polling for non-critical updates (60 seconds)
   - WebSocket for critical real-time events
   - Optimistic updates for better UX
   - Cache invalidation strategies

### API Integration Best Practices

1. **Request Management**
   - Consistent error handling
   - Request deduplication
   - Proper timeout configuration
   - Abort controllers for cancellation

2. **Authentication**
   - Automatic token refresh
   - Secure token storage
   - Request retry after token refresh
   - Logout on auth failures

3. **Performance**
   - Response caching where appropriate
   - Batch requests when possible
   - Lazy loading for large datasets
   - Progress tracking for long operations

4. **Developer Experience**
   - Strong TypeScript typing
   - Consistent API patterns
   - Clear error messages
   - Comprehensive logging

---

## 📚 Additional Resources

### Zustand Advanced Patterns

```typescript
// Advanced Zustand patterns for complex state management
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';

// Using Immer for immutable updates
const useComplexStore = create<ComplexState>()(
  devtools(
    persist(
      immer((set) => ({
        // Nested state
        entities: {
          channels: {},
          videos: {},
        },
        
        // Immer allows direct mutations
        updateChannel: (id: string, updates: Partial<Channel>) =>
          set((state) => {
            state.entities.channels[id] = {
              ...state.entities.channels[id],
              ...updates,
            };
          }),
      })),
      { name: 'complex-store' }
    )
  )
);

// Subscribe to specific state changes
useComplexStore.subscribe(
  (state) => state.entities.channels,
  (channels) => {
    console.log('Channels updated:', channels);
  }
);
```

### Error Recovery Patterns

```typescript
// Implementing circuit breaker pattern
class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  
  constructor(
    private threshold = 5,
    private timeout = 60000 // 1 minute
  ) {}
  
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime > this.timeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit breaker is open');
      }
    }
    
    try {
      const result = await fn();
      if (this.state === 'half-open') {
        this.state = 'closed';
        this.failures = 0;
      }
      return result;
    } catch (error) {
      this.failures++;
      this.lastFailureTime = Date.now();
      
      if (this.failures >= this.threshold) {
        this.state = 'open';
      }
      
      throw error;
    }
  }
}
```

This completes the State Management & API Integration Guide with all the necessary patterns, examples, and best practices for the React Engineer to implement a robust state management system using Zustand and integrate with the backend APIs effectively.