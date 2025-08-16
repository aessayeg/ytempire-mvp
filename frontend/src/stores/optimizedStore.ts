/**
 * Optimized State Management Store
 * Implements performance optimizations with selective subscriptions and memoization
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Types
export interface User {
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
}

export interface Channel {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'suspended';
  metrics: {
    subscribers: number;
    views: number;
    revenue: number;
  };
}

export interface Video {
  id: string;
  title: string;
  status: 'draft' | 'processing' | 'published' | 'failed';
  progress: number;
  channelId: string;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: Date;
}

export interface AppState {
  // User State
  user: User | null;
  
  // Channels State
  channels: {
    list: Channel[];
    selected: string | null;
    loading: boolean;
  };
  
  // Videos State
  videos: {
    list: Video[];
    selected: string | null;
    processing: string[];
    loading: boolean;
  };
  
  // UI State
  ui: {
    theme: 'light' | 'dark' | 'system';
    sidebarOpen: boolean;
    modalOpen: string | null;
    loading: boolean;
  };
  
  // Notifications
  notifications: Notification[];
  
  // Actions
  setUser: (user: User | null) => void;
  updateUserPreferences: (preferences: Partial<User['preferences']>) => void;
  
  // Channel Actions
  setChannels: (channels: Channel[]) => void;
  selectChannel: (channelId: string | null) => void;
  updateChannel: (channelId: string, data: Partial<Channel>) => void;
  
  // Video Actions
  setVideos: (videos: Video[]) => void;
  selectVideo: (videoId: string | null) => void;
  updateVideo: (videoId: string, data: Partial<Video>) => void;
  addProcessingVideo: (videoId: string) => void;
  removeProcessingVideo: (videoId: string) => void;
  
  // UI Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  toggleSidebar: () => void;
  openModal: (modalId: string) => void;
  closeModal: () => void;
  setLoading: (loading: boolean) => void;
  
  // Notification Actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useOptimizedStore = create<AppState>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set) => ({
          // Initial State
          user: null,
          channels: {
            list: [],
            selected: null,
            loading: false
          },
          videos: {
            list: [],
            selected: null,
            processing: [],
            loading: false
          },
          ui: {
            theme: 'system',
            sidebarOpen: true,
            modalOpen: null,
            loading: false
          },
          notifications: [],
          
          // User Actions
          setUser: (user) => set((state) => {
            state.user = user;
          }),
          
          updateUserPreferences: (preferences) => set((state) => {
            if (state.user) {
              state.user.preferences = { ...state.user.preferences, ...preferences };
            }
          }),
          
          // Channel Actions
          setChannels: (channels) => set((state) => {
            state.channels.list = channels;
          }),
          
          selectChannel: (channelId) => set((state) => {
            state.channels.selected = channelId;
          }),
          
          updateChannel: (channelId, data) => set((state) => {
            const index = state.channels.list.findIndex(c => c.id === channelId);
            if (index !== -1) {
              state.channels.list[index] = { ...state.channels.list[index], ...data };
            }
          }),
          
          // Video Actions
          setVideos: (videos) => set((state) => {
            state.videos.list = videos;
          }),
          
          selectVideo: (videoId) => set((state) => {
            state.videos.selected = videoId;
          }),
          
          updateVideo: (videoId, data) => set((state) => {
            const index = state.videos.list.findIndex(v => v.id === videoId);
            if (index !== -1) {
              state.videos.list[index] = { ...state.videos.list[index], ...data };
            }
          }),
          
          addProcessingVideo: (videoId) => set((state) => {
            if (!state.videos.processing.includes(videoId)) {
              state.videos.processing.push(videoId);
            }
          }),
          
          removeProcessingVideo: (videoId) => set((state) => {
            state.videos.processing = state.videos.processing.filter(id => id !== videoId);
          }),
          
          // UI Actions
          setTheme: (theme) => set((state) => {
            state.ui.theme = theme;
          }),
          
          toggleSidebar: () => set((state) => {
            state.ui.sidebarOpen = !state.ui.sidebarOpen;
          }),
          
          openModal: (modalId) => set((state) => {
            state.ui.modalOpen = modalId;
          }),
          
          closeModal: () => set((state) => {
            state.ui.modalOpen = null;
          }),
          
          setLoading: (loading) => set((state) => {
            state.ui.loading = loading;
          }),
          
          // Notification Actions
          addNotification: (notification) => set((state) => {
            state.notifications.push({
              ...notification,
              id: Date.now().toString(),
              timestamp: new Date()
            });
          }),
          
          removeNotification: (id) => set((state) => {
            state.notifications = state.notifications.filter(n => n.id !== id);
          }),
          
          clearNotifications: () => set((state) => {
            state.notifications = [];
          })
        }))
      ),
      {
        name: 'ytempire-store',
        partialize: (state) => ({
          user: state.user,
          ui: {
            theme: state.ui.theme,
            sidebarOpen: state.ui.sidebarOpen
          }
        })
      }
    )
  )
);

// Selectors
export const selectUser = (state: AppState) => state.user;
export const selectChannels = (state: AppState) => state.channels;
export const selectVideos = (state: AppState) => state.videos;
export const selectUI = (state: AppState) => state.ui;
export const selectNotifications = (state: AppState) => state.notifications;

export default useOptimizedStore;