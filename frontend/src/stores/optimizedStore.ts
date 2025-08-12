/**
 * Optimized State Management Store
 * Implements performance optimizations with selective subscriptions and memoization
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { shallow } from 'zustand/shallow';

// Types
export interface AppState {
  // User State
  user: {
    id: string | null;
    email: string | null;
    name: string | null;
    subscription: {
      plan: 'free' | 'starter' | 'pro' | 'enterprise';
      status: 'active' | 'inactive' | 'cancelled';
      expiresAt: Date | null;
    };
    preferences: {
      theme: 'light' | 'dark' | 'system';
      notifications: boolean;
      autoSave: boolean;
    };
  };

  // Channels State
  channels: {
    list: Channel[];
    selected: string | null;
    loading: boolean;
    error: string | null;
    lastFetch: number;
  };

  // Videos State
  videos: {
    queue: VideoQueueItem[];
    processing: string[];
    completed: CompletedVideo[];
    stats: {
      totalGenerated: number;
      totalViews: number;
      totalRevenue: number;
    };
  };

  // Analytics State
  analytics: {
    realtime: RealtimeMetrics | null;
    daily: DailyMetrics[];
    loading: boolean;
    cache: Map<string, any>;
  };

  // UI State
  ui: {
    sidebarOpen: boolean;
    activeModal: string | null;
    notifications: Notification[];
    theme: 'light' | 'dark';
  };

  // WebSocket State
  ws: {
    connected: boolean;
    reconnecting: boolean;
    lastMessage: any;
  };
}

export interface Channel {
  id: string;
  name: string;
  youtubeId: string;
  status: 'active' | 'paused' | 'error';
  metrics: {
    subscribers: number;
    videos: number;
    views: number;
  };
}

export interface VideoQueueItem {
  id: string;
  channelId: string;
  title: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  scheduledAt: Date;
  priority: 'low' | 'normal' | 'high' | 'urgent';
}

export interface CompletedVideo {
  id: string;
  title: string;
  youtubeId: string;
  views: number;
  likes: number;
  revenue: number;
  publishedAt: Date;
}

export interface RealtimeMetrics {
  activeViewers: number;
  videosProcessing: number;
  apiCallsPerMinute: number;
  errorRate: number;
}

export interface DailyMetrics {
  date: string;
  views: number;
  revenue: number;
  newSubscribers: number;
  videosPublished: number;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: Date;
  read: boolean;
}

// Actions
export interface AppActions {
  // User Actions
  setUser: (user: Partial<AppState['user']>) => void;
  updatePreferences: (prefs: Partial<AppState['user']['preferences']>) => void;
  logout: () => void;

  // Channel Actions
  fetchChannels: () => Promise<void>;
  selectChannel: (channelId: string) => void;
  updateChannel: (channelId: string, updates: Partial<Channel>) => void;
  addChannel: (channel: Channel) => void;

  // Video Actions
  addToQueue: (video: Omit<VideoQueueItem, 'id'>) => void;
  updateQueueItem: (id: string, updates: Partial<VideoQueueItem>) => void;
  removeFromQueue: (id: string) => void;
  moveToProcessing: (id: string) => void;
  markCompleted: (id: string, video: CompletedVideo) => void;

  // Analytics Actions
  updateRealtimeMetrics: (metrics: RealtimeMetrics) => void;
  addDailyMetrics: (metrics: DailyMetrics) => void;
  cacheAnalytics: (key: string, data: any) => void;
  getCachedAnalytics: (key: string) => any;

  // UI Actions
  toggleSidebar: () => void;
  openModal: (modalId: string) => void;
  closeModal: () => void;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;
  setTheme: (theme: 'light' | 'dark') => void;

  // WebSocket Actions
  setWsConnected: (connected: boolean) => void;
  setWsReconnecting: (reconnecting: boolean) => void;
  handleWsMessage: (message: any) => void;

  // Utility Actions
  reset: () => void;
  hydrate: () => void;
}

// Initial State
const initialState: AppState = {
  user: {
    id: null,
    email: null,
    name: null,
    subscription: {
      plan: 'free',
      status: 'inactive',
      expiresAt: null,
    },
    preferences: {
      theme: 'system',
      notifications: true,
      autoSave: true,
    },
  },
  channels: {
    list: [],
    selected: null,
    loading: false,
    error: null,
    lastFetch: 0,
  },
  videos: {
    queue: [],
    processing: [],
    completed: [],
    stats: {
      totalGenerated: 0,
      totalViews: 0,
      totalRevenue: 0,
    },
  },
  analytics: {
    realtime: null,
    daily: [],
    loading: false,
    cache: new Map(),
  },
  ui: {
    sidebarOpen: true,
    activeModal: null,
    notifications: [],
    theme: 'light',
  },
  ws: {
    connected: false,
    reconnecting: false,
    lastMessage: null,
  },
};

// Create Store with Middleware
export const useOptimizedStore = create<AppState & AppActions>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          ...initialState,

          // User Actions
          setUser: (user) =>
            set((state) => {
              Object.assign(state.user, user);
            }),

          updatePreferences: (prefs) =>
            set((state) => {
              Object.assign(state.user.preferences, prefs);
            }),

          logout: () =>
            set((state) => {
              state.user = initialState.user;
              state.channels = initialState.channels;
              state.videos = initialState.videos;
            }),

          // Channel Actions
          fetchChannels: async () => {
            const now = Date.now();
            const { channels } = get();
            
            // Cache for 5 minutes
            if (now - channels.lastFetch < 5 * 60 * 1000) {
              return;
            }

            set((state) => {
              state.channels.loading = true;
              state.channels.error = null;
            });

            try {
              const response = await fetch('/api/v1/channels', {
                headers: {
                  Authorization: `Bearer ${localStorage.getItem('token')}`,
                },
              });

              if (!response.ok) throw new Error('Failed to fetch channels');

              const data = await response.json();

              set((state) => {
                state.channels.list = data;
                state.channels.loading = false;
                state.channels.lastFetch = now;
              });
            } catch (error) {
              set((state) => {
                state.channels.loading = false;
                state.channels.error = error instanceof Error ? error.message : 'An error occurred';
              });
            }
          },

          selectChannel: (channelId) =>
            set((state) => {
              state.channels.selected = channelId;
            }),

          updateChannel: (channelId, updates) =>
            set((state) => {
              const channel = state.channels.list.find((c) => c.id === channelId);
              if (channel) {
                Object.assign(channel, updates);
              }
            }),

          addChannel: (channel) =>
            set((state) => {
              state.channels.list.push(channel);
            }),

          // Video Actions
          addToQueue: (video) =>
            set((state) => {
              const id = `video_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
              state.videos.queue.push({ ...video, id });
            }),

          updateQueueItem: (id, updates) =>
            set((state) => {
              const item = state.videos.queue.find((v) => v.id === id);
              if (item) {
                Object.assign(item, updates);
              }
            }),

          removeFromQueue: (id) =>
            set((state) => {
              state.videos.queue = state.videos.queue.filter((v) => v.id !== id);
            }),

          moveToProcessing: (id) =>
            set((state) => {
              const item = state.videos.queue.find((v) => v.id === id);
              if (item) {
                item.status = 'processing';
                state.videos.processing.push(id);
              }
            }),

          markCompleted: (id, video) =>
            set((state) => {
              state.videos.queue = state.videos.queue.filter((v) => v.id !== id);
              state.videos.processing = state.videos.processing.filter((pid) => pid !== id);
              state.videos.completed.push(video);
              state.videos.stats.totalGenerated++;
            }),

          // Analytics Actions
          updateRealtimeMetrics: (metrics) =>
            set((state) => {
              state.analytics.realtime = metrics;
            }),

          addDailyMetrics: (metrics) =>
            set((state) => {
              const existing = state.analytics.daily.findIndex((m) => m.date === metrics.date);
              if (existing >= 0) {
                state.analytics.daily[existing] = metrics;
              } else {
                state.analytics.daily.push(metrics);
                // Keep only last 30 days
                if (state.analytics.daily.length > 30) {
                  state.analytics.daily.shift();
                }
              }
            }),

          cacheAnalytics: (key, data) =>
            set((state) => {
              state.analytics.cache.set(key, {
                data,
                timestamp: Date.now(),
              });
            }),

          getCachedAnalytics: (key) => {
            const cached = get().analytics.cache.get(key);
            if (cached && Date.now() - cached.timestamp < 5 * 60 * 1000) {
              return cached.data;
            }
            return null;
          },

          // UI Actions
          toggleSidebar: () =>
            set((state) => {
              state.ui.sidebarOpen = !state.ui.sidebarOpen;
            }),

          openModal: (modalId) =>
            set((state) => {
              state.ui.activeModal = modalId;
            }),

          closeModal: () =>
            set((state) => {
              state.ui.activeModal = null;
            }),

          addNotification: (notification) =>
            set((state) => {
              const id = `notif_${Date.now()}`;
              state.ui.notifications.unshift({
                ...notification,
                id,
                timestamp: new Date(),
                read: false,
              });
              // Keep max 50 notifications
              if (state.ui.notifications.length > 50) {
                state.ui.notifications.pop();
              }
            }),

          markNotificationRead: (id) =>
            set((state) => {
              const notif = state.ui.notifications.find((n) => n.id === id);
              if (notif) {
                notif.read = true;
              }
            }),

          clearNotifications: () =>
            set((state) => {
              state.ui.notifications = [];
            }),

          setTheme: (theme) =>
            set((state) => {
              state.ui.theme = theme;
              document.documentElement.setAttribute('data-theme', theme);
            }),

          // WebSocket Actions
          setWsConnected: (connected) =>
            set((state) => {
              state.ws.connected = connected;
            }),

          setWsReconnecting: (reconnecting) =>
            set((state) => {
              state.ws.reconnecting = reconnecting;
            }),

          handleWsMessage: (message) =>
            set((state) => {
              state.ws.lastMessage = message;
              
              // Handle different message types
              if (message.type === 'video_update') {
                const { updateQueueItem } = get();
                updateQueueItem(message.videoId, message.data);
              } else if (message.type === 'metrics_update') {
                const { updateRealtimeMetrics } = get();
                updateRealtimeMetrics(message.data);
              } else if (message.type === 'notification') {
                const { addNotification } = get();
                addNotification({
                  type: message.level || 'info',
                  message: message.text,
                });
              }
            }),

          // Utility Actions
          reset: () => set(initialState),

          hydrate: () => {
            // Hydrate from localStorage or API
            const savedState = localStorage.getItem('app-state');
            if (savedState) {
              try {
                const parsed = JSON.parse(savedState);
                set((state) => {
                  Object.assign(state, parsed);
                });
              } catch (error) {
                console.error('Failed to hydrate state:', error);
              }
            }
          },
        }))
      ),
      {
        name: 'ytempire-app-state',
        partialize: (state) => ({
          user: state.user,
          ui: {
            theme: state.ui.theme,
            sidebarOpen: state.ui.sidebarOpen,
          },
        }),
      }
    )
  )
);

// Selectors for optimized re-renders
export const useUser = () => useOptimizedStore((state) => state.user);
export const useChannels = () => useOptimizedStore((state) => state.channels, shallow);
export const useSelectedChannel = () =>
  useOptimizedStore((state) => {
    const { channels } = state;
    return channels.list.find((c) => c.id === channels.selected);
  });
export const useVideoQueue = () => useOptimizedStore((state) => state.videos.queue, shallow);
export const useRealtimeMetrics = () => useOptimizedStore((state) => state.analytics.realtime);
export const useNotifications = () =>
  useOptimizedStore((state) => state.ui.notifications.filter((n) => !n.read), shallow);
export const useTheme = () => useOptimizedStore((state) => state.ui.theme);
export const useWsStatus = () => useOptimizedStore((state) => state.ws);