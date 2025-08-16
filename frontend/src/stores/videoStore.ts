import {  create  } from 'zustand';
import {  videosApi  } from '../services/api';

interface Video {
  
id: string;
title: string;
description: string;
status: 'pending' | 'processing' | 'completed' | 'failed';
channelId: string;
createdAt: Date;
publishedAt?: Date;
views?: number;
likes?: number;
comments?: number;
generationCost?: number;


}

interface VideoStore {
  
videos: Video[];
loading: boolean;
error: string | null;
fetchVideos: () => Promise<void>;
createVideo: (videoData: Partial<Video>) => Promise<void>;
updateVideo: (id: string, updates: Partial<Video>) => void;
deleteVideo: (id: string) => void;


}

export const useVideoStore = create<VideoStore>((set) => ({
  videos: [],
  loading: false,
  error: null,

  fetchVideos: async () => {
    set({ loading: true, error: null });
    try {
      const videos = await videosApi.getAll();
      set({ videos, loading: false })
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch videos';
      set({ error: errorMessage, loading: false })}
  },

  createVideo: async (videoData) => {
    set({ loading: true, error: null });
    try {
      const newVideo = await videosApi.generate(videoData);
      set((state) => ({
        videos: [...state.videos, newVideo],
        loading: false
      }))
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create video';
      set({ error: errorMessage, loading: false })}
  },

  updateVideo: (id, updates) => {
    set((state) => ({
      videos: state.videos.map((video) =>
        video.id === id ? { ...video, ...updates } : video
      )
    }))
  },

  deleteVideo: (id) => {
    set((state) => ({
      videos: state.videos.filter((video) => video.id !== id)
    }))
  }
}));
