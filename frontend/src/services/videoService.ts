/**
 * Video Service
 * Owner: Frontend Team Lead
 */

import { apiClient } from '@/utils/api'

export interface Video {
  id: string
  title: string
  description: string
  script: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'published'
  channel_id: string
  topic: string
  quality_score?: number
  total_cost: number
  youtube_video_id?: string
  youtube_url?: string
  thumbnail_url?: string
  created_at: string
  updated_at: string
  completed_at?: string
  metadata?: Record<string, any>
}

export interface VideoGenerationRequest {
  channel_id: string
  topic: string
  target_duration?: number
  style?: string
  voice_id?: string
  visual_style?: string
  custom_instructions?: string
}

export interface VideoQueue {
  id: string
  video_id: string
  status: string
  progress: number
  current_stage: string
  estimated_completion: string
  error_message?: string
}

export interface CostBreakdown {
  script_cost: number
  voice_cost: number
  image_cost: number
  total_cost: number
  breakdown_details: Record<string, any>
}

export class VideoService {
  /**
   * Generate a new video
   */
  static async generateVideo(request: VideoGenerationRequest): Promise<{ video_id: string; queue_id: string }> {
    const response = await apiClient.generateVideo(request)
    return response.data
  }

  /**
   * Get all videos for current user
   */
  static async getVideos(params?: {
    page?: number
    limit?: number
    status?: string
    channel_id?: string
    search?: string
    sort?: string
  }): Promise<{
    videos: Video[]
    total: number
    page: number
    limit: number
    total_pages: number
  }> {
    const response = await apiClient.getVideos(params)
    return response.data
  }

  /**
   * Get video by ID
   */
  static async getVideo(id: string): Promise<Video> {
    const response = await apiClient.getVideo(id)
    return response.data
  }

  /**
   * Update video details
   */
  static async updateVideo(id: string, updates: Partial<Video>): Promise<Video> {
    const response = await apiClient.patch(`/videos/${id}`, updates)
    return response.data
  }

  /**
   * Delete video
   */
  static async deleteVideo(id: string): Promise<void> {
    await apiClient.deleteVideo(id)
  }

  /**
   * Retry failed video generation
   */
  static async retryVideo(id: string): Promise<{ queue_id: string }> {
    const response = await apiClient.retryVideo(id)
    return response.data
  }

  /**
   * Publish video to YouTube
   */
  static async publishVideo(id: string, scheduleTime?: string): Promise<{ youtube_video_id: string }> {
    const response = await apiClient.publishVideo(id, scheduleTime)
    return response.data
  }

  /**
   * Get video generation queue
   */
  static async getQueue(): Promise<VideoQueue[]> {
    const response = await apiClient.get('/videos/queue')
    return response.data
  }

  /**
   * Get queue item by ID
   */
  static async getQueueItem(id: string): Promise<VideoQueue> {
    const response = await apiClient.get(`/videos/queue/${id}`)
    return response.data
  }

  /**
   * Cancel video generation
   */
  static async cancelGeneration(queueId: string): Promise<void> {
    await apiClient.delete(`/videos/queue/${queueId}`)
  }

  /**
   * Get video cost breakdown
   */
  static async getCostBreakdown(id: string): Promise<CostBreakdown> {
    const response = await apiClient.getVideoCost(id)
    return response.data
  }

  /**
   * Get video analytics
   */
  static async getVideoAnalytics(id: string): Promise<{
    views: number
    likes: number
    comments: number
    shares: number
    watch_time: number
    engagement_rate: number
    revenue: number
  }> {
    const response = await apiClient.getVideoAnalytics(id)
    return response.data
  }

  /**
   * Download video file
   */
  static async downloadVideo(id: string): Promise<Blob> {
    const response = await apiClient.get(`/videos/${id}/download`, {
      responseType: 'blob'
    })
    return response.data
  }

  /**
   * Get video thumbnail
   */
  static async getThumbnail(id: string): Promise<Blob> {
    const response = await apiClient.get(`/videos/${id}/thumbnail`, {
      responseType: 'blob'
    })
    return response.data
  }

  /**
   * Update video thumbnail
   */
  static async updateThumbnail(id: string, thumbnailFile: File): Promise<{ thumbnail_url: string }> {
    const formData = new FormData()
    formData.append('thumbnail', thumbnailFile)
    
    const response = await apiClient.post(`/videos/${id}/thumbnail`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  }

  /**
   * Bulk operations
   */
  static async bulkDelete(videoIds: string[]): Promise<{ deleted: number; failed: string[] }> {
    const response = await apiClient.post('/videos/bulk/delete', { video_ids: videoIds })
    return response.data
  }

  static async bulkPublish(videoIds: string[]): Promise<{ published: number; failed: string[] }> {
    const response = await apiClient.post('/videos/bulk/publish', { video_ids: videoIds })
    return response.data
  }

  /**
   * Get video templates
   */
  static async getTemplates(): Promise<{
    id: string
    name: string
    description: string
    style: string
    voice_settings: Record<string, any>
    visual_settings: Record<string, any>
  }[]> {
    const response = await apiClient.get('/videos/templates')
    return response.data
  }

  /**
   * Create video from template
   */
  static async createFromTemplate(templateId: string, params: {
    channel_id: string
    topic: string
    custom_instructions?: string
  }): Promise<{ video_id: string; queue_id: string }> {
    const response = await apiClient.post(`/videos/templates/${templateId}/create`, params)
    return response.data
  }

  /**
   * Get video suggestions based on trends
   */
  static async getVideoSuggestions(channelId: string): Promise<{
    topic: string
    trending_score: number
    estimated_views: number
    difficulty: 'easy' | 'medium' | 'hard'
    keywords: string[]
  }[]> {
    const response = await apiClient.get(`/videos/suggestions?channel_id=${channelId}`)
    return response.data
  }
}

export default VideoService