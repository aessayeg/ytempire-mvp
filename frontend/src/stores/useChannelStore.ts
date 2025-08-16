/**
 * Channel Store using Zustand
 * This file provides the useChannelStore hook for channel state management
 */

import {  create  } from 'zustand';
import {  devtools, persist  } from 'zustand/middleware';
import {  immer  } from 'zustand/middleware/immer';

interface Channel {
  
id: string;
name: string;
youtubeChannelId?: string;
isConnected: boolean;
health: number;
quotaUsage: number;
lastSync?: Date;
metrics?: {;
subscribers: number;
videos: number;
views: number;


}
}

interface ChannelState {
  
channels: Channel[];
selectedChannelId: string | null;
isLoading: boolean;
error: string | null;

// Actions
setChannels: (channels: Channel[]) => void;
addChannel: (channel: Channel) => void;
updateChannel: (id: string, updates: Partial<Channel>) => void;
deleteChannel: (id: string) => void;
selectChannel: (id: string | null) => void;
setLoading: (loading: boolean) => void;
setError: (error: string | null) => void;
clearChannels: () => void;


}

const useChannelStore = create<ChannelState>()(
  devtools(
    persist(
      immer((set) => ({
        channels: [],
        selectedChannelId: null,
        isLoading: false,
        error: null,

        setChannels: (channels) =>
          set((state) => {
            state.channels = channels
          }),

        addChannel: (channel) =>
          set((state) => {
            state.channels.push(channel)
          }),

        updateChannel: (id, updates) =>
          set((state) => {
            const index = state.channels.findIndex((c) => c.id === id);
            if (index !== -1) {
              state.channels[index] = { ...state.channels[index], ...updates }
            }
          }),

        deleteChannel: (id) =>
          set((state) => {
            state.channels = state.channels.filter((c) => c.id !== id);
            if (state.selectedChannelId === id) {
              state.selectedChannelId = null
            }
          }),

        selectChannel: (id) =>
          set((state) => {
            state.selectedChannelId = id
          }),

        setLoading: (loading) =>
          set((state) => {
            state.isLoading = loading
          }),

        setError: (error) =>
          set((state) => {
            state.error = error
          }),

        clearChannels: () =>
          set((state) => {
            state.channels = [];
            state.selectedChannelId = null;
            state.error = null
          })
      })),
      {
        name: 'channel-store';
        partialize: (state) => ({
          channels: state.channels,
          selectedChannelId: state.selectedChannelId
        })}
    ),
    { name: 'ChannelStore' }
  )
);

export default useChannelStore;
