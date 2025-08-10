/**
 * Channel Service
 * Owner: Frontend Team Lead
 */

import { apiClient } from '@/utils/api'

export interface Channel {
  id: string
  name: string
  description: string
  youtube_channel_id?: string
  youtube_channel_url?: string
  category: string
  target_audience: string
  tone: string
  language: string
  status: 'active' | 'inactive' | 'connecting' | 'failed'
  subscriber_count: number
  video_count: number
  total_views: number
  is_monetized: boolean
  created_at: string
  updated_at: string
  last_upload: string
  avatar_url?: string
  banner_url?: string
  settings: {
    auto_publish: boolean
    publish_schedule?: string
    default_visibility: 'public' | 'unlisted' | 'private'
    voice_id?: string
    visual_style?: string
  }
}

export interface ChannelAnalytics {
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
  top_videos: Array<{
    id: string
    title: string
    views: number
    likes: number
    published_at: string
  }>
}

export interface YouTubeConnectionFlow {
  authorization_url: string
  state: string
}

export class ChannelService {
  /**
   * Get all channels for current user
   */
  static async getChannels(): Promise<Channel[]> {
    const response = await apiClient.getChannels()
    return response.data
  }

  /**
   * Get channel by ID
   */
  static async getChannel(id: string): Promise<Channel> {
    const response = await apiClient.getChannel(id)
    return response.data
  }

  /**
   * Create new channel
   */
  static async createChannel(channel: {
    name: string
    description: string
    category: string
    target_audience: string
    tone: string
    language: string
  }): Promise<Channel> {
    const response = await apiClient.createChannel(channel)
    return response.data
  }

  /**
   * Update channel
   */
  static async updateChannel(id: string, updates: Partial<Channel>): Promise<Channel> {
    const response = await apiClient.updateChannel(id, updates)
    return response.data
  }

  /**
   * Delete channel
   */
  static async deleteChannel(id: string): Promise<void> {
    await apiClient.deleteChannel(id)
  }

  /**
   * Start YouTube connection flow
   */
  static async startYouTubeConnection(channelId: string): Promise<YouTubeConnectionFlow> {
    const response = await apiClient.post(`/channels/${channelId}/youtube/connect`)
    return response.data
  }

  /**
   * Complete YouTube connection
   */
  static async completeYouTubeConnection(channelId: string, authCode: string): Promise<Channel> {
    const response = await apiClient.connectYouTube(channelId, authCode)
    return response.data
  }

  /**
   * Disconnect from YouTube
   */
  static async disconnectYouTube(channelId: string): Promise<void> {
    await apiClient.post(`/channels/${channelId}/youtube/disconnect`)
  }

  /**
   * Sync channel data from YouTube
   */
  static async syncFromYouTube(channelId: string): Promise<Channel> {
    const response = await apiClient.post(`/channels/${channelId}/sync`)
    return response.data
  }

  /**
   * Get channel analytics
   */
  static async getChannelAnalytics(id: string, params?: {
    start_date?: string
    end_date?: string
    metrics?: string[]
  }): Promise<ChannelAnalytics> {
    const response = await apiClient.getChannelAnalytics(id, params)
    return response.data
  }

  /**
   * Get channel statistics
   */
  static async getChannelStats(id: string): Promise<{
    total_videos: number
    processing_videos: number
    published_videos: number
    total_cost: number
    total_views: number
    total_revenue: number
  }> {
    const response = await apiClient.getChannelStats(id)
    return response.data
  }

  /**
   * Get channel videos
   */
  static async getChannelVideos(id: string, params?: {
    page?: number
    limit?: number
    status?: string
    sort?: string
  }): Promise<{
    videos: any[]
    total: number
    page: number
    limit: number
  }> {
    const response = await apiClient.get(`/channels/${id}/videos`, { params })
    return response.data
  }

  /**
   * Update channel settings
   */
  static async updateSettings(id: string, settings: Partial<Channel['settings']>): Promise<Channel> {
    const response = await apiClient.patch(`/channels/${id}/settings`, settings)
    return response.data
  }

  /**
   * Upload channel avatar
   */
  static async uploadAvatar(id: string, avatarFile: File): Promise<{ avatar_url: string }> {
    const formData = new FormData()
    formData.append('avatar', avatarFile)
    
    const response = await apiClient.post(`/channels/${id}/avatar`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  }

  /**
   * Upload channel banner
   */
  static async uploadBanner(id: string, bannerFile: File): Promise<{ banner_url: string }> {
    const formData = new FormData()
    formData.append('banner', bannerFile)
    
    const response = await apiClient.post(`/channels/${id}/banner`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  }

  /**
   * Get channel suggestions
   */
  static async getSuggestions(): Promise<{
    suggested_categories: string[]
    trending_topics: string[]
    recommended_settings: Record<string, any>
  }> {
    const response = await apiClient.get('/channels/suggestions')
    return response.data
  }

  /**
   * Validate YouTube channel
   */
  static async validateYouTubeChannel(channelUrl: string): Promise<{
    valid: boolean
    channel_id?: string
    channel_name?: string
    subscriber_count?: number
    error?: string
  }> {
    const response = await apiClient.post('/channels/validate-youtube', {
      channel_url: channelUrl
    })
    return response.data
  }

  /**
   * Get available voice IDs for channel
   */
  static async getAvailableVoices(): Promise<{
    id: string
    name: string
    language: string
    gender: string
    sample_url: string
    premium: boolean
  }[]> {
    const response = await apiClient.get('/channels/voices')
    return response.data
  }

  /**
   * Test voice with sample text
   */
  static async testVoice(voiceId: string, text: string): Promise<{ audio_url: string }> {
    const response = await apiClient.post('/channels/test-voice', {
      voice_id: voiceId,
      text
    })
    return response.data
  }

  /**
   * Get channel templates
   */
  static async getTemplates(): Promise<{
    id: string
    name: string
    description: string
    category: string
    settings: Record<string, any>
    preview_url?: string
  }[]> {
    const response = await apiClient.get('/channels/templates')
    return response.data
  }

  /**
   * Create channel from template
   */
  static async createFromTemplate(templateId: string, customization: {
    name: string
    description?: string
    target_audience?: string
  }): Promise<Channel> {
    const response = await apiClient.post(`/channels/templates/${templateId}/create`, customization)
    return response.data
  }
}

export default ChannelService