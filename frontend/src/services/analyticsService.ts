/**
 * Analytics Service
 * Owner: Frontend Team Lead
 */

import { apiClient } from '@/utils/api'

export interface DashboardMetrics {
  overview: {
    total_channels: number
    total_videos: number
    total_subscribers: number
    monthly_revenue: number
    total_costs: number
    total_views: number
  }
  performance: {
    top_performing_channels: Array<{
      id: string
      name: string
      revenue: number
      subscribers: number
      views: number
    }>
    recent_videos: Array<{
      id: string
      title: string
      views: number
      engagement_rate: number
      published_at: string
    }>
  }
  trends: {
    subscriber_growth: number
    view_growth: number
    revenue_growth: number
    cost_trend: number
  }
}

export interface ChannelAnalytics {
  channel_id: string
  subscriber_count: {
    current: number
    growth_7d: number
    growth_30d: number
    growth_percentage_7d: number
    growth_percentage_30d: number
  }
  views: {
    total: number
    last_7d: number
    last_30d: number
    average_per_video: number
  }
  engagement: {
    likes: number
    comments: number
    shares: number
    engagement_rate: number
    watch_time_minutes: number
    average_view_duration: number
  }
  revenue: {
    estimated_monthly: number
    last_month: number
    year_to_date: number
    rpm: number
    cpm: number
  }
  video_performance: Array<{
    id: string
    title: string
    views: number
    likes: number
    comments: number
    published_at: string
    engagement_rate: number
    revenue: number
  }>
}

export interface VideoAnalytics {
  video_id: string
  views: number
  likes: number
  dislikes: number
  comments: number
  shares: number
  watch_time: number
  average_view_duration: number
  engagement_rate: number
  click_through_rate: number
  revenue: {
    estimated: number
    rpm: number
    cpm: number
  }
  traffic_sources: Array<{
    source: string
    views: number
    percentage: number
  }>
  demographics: {
    age_groups: Record<string, number>
    genders: Record<string, number>
    top_countries: Record<string, number>
  }
  retention: Array<{
    timestamp: number
    percentage: number
  }>
}

export interface CostAnalytics {
  total_cost: number
  cost_breakdown: {
    script_generation: number
    voice_synthesis: number
    image_generation: number
    video_processing: number
    storage: number
    api_calls: number
  }
  cost_by_channel: Array<{
    channel_id: string
    channel_name: string
    total_cost: number
    video_count: number
    average_cost_per_video: number
  }>
  cost_by_video: Array<{
    video_id: string
    title: string
    total_cost: number
    cost_breakdown: Record<string, number>
    roi: number
  }>
  monthly_trend: Array<{
    month: string
    total_cost: number
    video_count: number
    average_cost: number
  }>
  cost_efficiency: {
    cost_per_view: number
    cost_per_subscriber: number
    cost_per_dollar_revenue: number
  }
}

export interface WeeklyReport {
  week_start: string
  week_end: string
  summary: {
    videos_generated: number
    videos_published: number
    total_views: number
    new_subscribers: number
    revenue: number
    costs: number
    roi: number
  }
  channel_performance: Array<{
    channel_id: string
    channel_name: string
    videos_published: number
    views: number
    subscribers_gained: number
    revenue: number
    costs: number
  }>
  top_videos: Array<{
    id: string
    title: string
    channel_name: string
    views: number
    engagement_rate: number
    revenue: number
  }>
  insights: Array<{
    type: 'positive' | 'negative' | 'neutral'
    title: string
    description: string
    action_required?: boolean
    recommendation?: string
  }>
}

export interface ComparisonData {
  channels: Array<{
    id: string
    name: string
    metrics: {
      views: number
      subscribers: number
      engagement_rate: number
      revenue: number
      costs: number
      roi: number
      video_count: number
    }
  }>
  time_range: {
    start_date: string
    end_date: string
  }
}

export class AnalyticsService {
  /**
   * Get dashboard metrics
   */
  static async getDashboardMetrics(timeRange?: '7d' | '30d' | '90d' | '1y'): Promise<DashboardMetrics> {
    const response = await apiClient.getDashboard()
    return response.data
  }

  /**
   * Get channel analytics
   */
  static async getChannelAnalytics(
    channelId: string,
    params?: {
      start_date?: string
      end_date?: string
      metrics?: string[]
    }
  ): Promise<ChannelAnalytics> {
    const response = await apiClient.getChannelAnalytics(channelId, params)
    return response.data
  }

  /**
   * Get video analytics
   */
  static async getVideoAnalytics(videoId: string): Promise<VideoAnalytics> {
    const response = await apiClient.getVideoAnalytics(videoId)
    return response.data
  }

  /**
   * Get cost analytics
   */
  static async getCostAnalytics(params?: {
    start_date?: string
    end_date?: string
    channel_id?: string
  }): Promise<CostAnalytics> {
    const response = await apiClient.getCostAnalytics(params)
    return response.data
  }

  /**
   * Get weekly report
   */
  static async getWeeklyReport(weekOffset = 0): Promise<WeeklyReport> {
    const response = await apiClient.getWeeklyReport()
    return response.data
  }

  /**
   * Compare multiple channels
   */
  static async compareChannels(
    channelIds: string[],
    timeRange?: {
      start_date: string
      end_date: string
    }
  ): Promise<ComparisonData> {
    const response = await apiClient.get('/analytics/compare/channels', {
      params: {
        channel_ids: channelIds.join(','),
        ...timeRange
      }
    })
    return response.data
  }

  /**
   * Export analytics data
   */
  static async exportAnalyticsData(
    type: 'dashboard' | 'channel' | 'video' | 'costs',
    id?: string,
    format: 'csv' | 'xlsx' | 'pdf' = 'xlsx'
  ): Promise<Blob> {
    const endpoint = id ? `/analytics/${type}/${id}/export` : `/analytics/${type}/export`
    const response = await apiClient.get(endpoint, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

  /**
   * Get real-time metrics
   */
  static async getRealTimeMetrics(): Promise<{
    active_users: number
    videos_processing: number
    api_requests_per_minute: number
    system_status: 'healthy' | 'warning' | 'error'
    cost_rate_per_hour: number
  }> {
    const response = await apiClient.get('/analytics/realtime')
    return response.data
  }

  /**
   * Get trend analysis
   */
  static async getTrendAnalysis(
    metric: 'views' | 'subscribers' | 'revenue' | 'costs',
    timeRange: '7d' | '30d' | '90d' | '1y'
  ): Promise<{
    trend: 'increasing' | 'decreasing' | 'stable'
    percentage_change: number
    forecast: Array<{
      date: string
      predicted_value: number
      confidence: number
    }>
    data_points: Array<{
      date: string
      value: number
    }>
  }> {
    const response = await apiClient.get('/analytics/trends', {
      params: { metric, time_range: timeRange }
    })
    return response.data
  }

  /**
   * Get performance insights
   */
  static async getPerformanceInsights(channelId?: string): Promise<Array<{
    type: 'optimization' | 'alert' | 'opportunity' | 'achievement'
    priority: 'low' | 'medium' | 'high' | 'critical'
    title: string
    description: string
    metric: string
    current_value: number
    target_value?: number
    action_items: string[]
    estimated_impact: {
      views?: number
      subscribers?: number
      revenue?: number
      cost_savings?: number
    }
  }>> {
    const response = await apiClient.get('/analytics/insights', {
      params: channelId ? { channel_id: channelId } : {}
    })
    return response.data
  }

  /**
   * Get competitor analysis
   */
  static async getCompetitorAnalysis(channelId: string): Promise<{
    similar_channels: Array<{
      name: string
      subscriber_count: number
      average_views: number
      upload_frequency: number
      top_keywords: string[]
    }>
    market_position: {
      rank: number
      percentile: number
      growth_rate_compared_to_market: number
    }
    opportunities: Array<{
      type: 'content_gap' | 'trend' | 'optimization'
      description: string
      potential_impact: number
    }>
  }> {
    const response = await apiClient.get(`/analytics/competitor-analysis/${channelId}`)
    return response.data
  }

  /**
   * Get A/B test results
   */
  static async getABTestResults(testId?: string): Promise<Array<{
    id: string
    name: string
    status: 'running' | 'completed' | 'paused'
    variants: Array<{
      name: string
      traffic_percentage: number
      conversion_rate: number
      statistical_significance: number
    }>
    winner?: string
    start_date: string
    end_date?: string
    metrics: Record<string, number>
  }>> {
    const response = await apiClient.get('/analytics/ab-tests', {
      params: testId ? { test_id: testId } : {}
    })
    return response.data
  }

  /**
   * Create custom report
   */
  static async createCustomReport(config: {
    name: string
    metrics: string[]
    filters: Record<string, any>
    time_range: {
      start_date: string
      end_date: string
    }
    grouping?: 'daily' | 'weekly' | 'monthly'
    format?: 'chart' | 'table' | 'both'
  }): Promise<{
    report_id: string
    data: any[]
    chart_config?: any
    table_config?: any
  }> {
    const response = await apiClient.post('/analytics/custom-reports', config)
    return response.data
  }

  /**
   * Schedule automated report
   */
  static async scheduleReport(config: {
    name: string
    report_type: 'weekly' | 'monthly'
    recipients: string[]
    metrics: string[]
    filters?: Record<string, any>
    format: 'email' | 'slack' | 'webhook'
    schedule: {
      day_of_week?: number
      day_of_month?: number
      time: string
    }
  }): Promise<{ schedule_id: string }> {
    const response = await apiClient.post('/analytics/scheduled-reports', config)
    return response.data
  }
}

export default AnalyticsService