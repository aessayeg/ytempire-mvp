/**
 * Services Index - Centralized service exports
 * Owner: Frontend Team Lead
 */

// Core services
export { default as apiService, api, type ApiResponse, type ApiError, type RequestOptions } from './apiService'
export { default as authService } from './authService'
export { default as channelService, type Channel, type ChannelAnalytics, type YouTubeConnectionFlow } from './channelService'
export { default as videoService, type Video, type VideoGenerationRequest, type VideoQueue, type CostBreakdown } from './videoService'
export { default as analyticsService, type DashboardMetrics, type ChannelAnalytics as AnalyticsChannelData, type VideoAnalytics, type CostAnalytics, type WeeklyReport } from './analyticsService'

// Utility services
export { default as uploadService, upload, type UploadOptions, type UploadProgress, type UploadError, type UploadResult } from './uploadService'
export { default as websocketService, websocket, useWebSocket, type WebSocketMessage, type WebSocketOptions, WebSocketStatus } from './websocketService'
export { default as cacheService, cache, type CacheOptions, type CacheItem, type CacheStats } from './cacheService'

// Service utilities
export const services = {
  api: apiService,
  auth: authService,
  channel: channelService,
  video: videoService,
  analytics: analyticsService,
  upload: uploadService,
  websocket: websocketService,
  cache: cacheService
}

// Service health check
export const checkServicesHealth = async () => {
  const results = {
    api: false,
    websocket: false,
    cache: false
  }

  try {
    // Check API health
    const apiHealth = await apiService.healthCheck()
    results.api = apiHealth.status === 'healthy'
  } catch {
    results.api = false
  }

  // Check WebSocket
  results.websocket = websocketService.isConnected()

  // Check cache
  try {
    const stats = cacheService.getStats()
    results.cache = true // Cache is always available locally
  } catch {
    results.cache = false
  }

  return results
}

export default services