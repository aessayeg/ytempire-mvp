# YTEMPIRE API Services & WebSocket Integration
## Complete API Service Layer Implementation

**Document Version**: 1.0  
**Role**: React Engineer  
**Scope**: API Services, WebSocket Handling, and Error Management

---

## 📡 API Service Implementations

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

interface VideoListParams {
  channelId?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

interface QueueStatus {
  queue: Video[];
  processing: Video[];
  stats: {
    queueDepth: number;
    averageWaitTime: number;
    processingCount: number;
  };
}

class VideoApi {
  async getVideos(params?: VideoListParams): Promise<Video[]> {
    return apiClient.get<Video[]>('/videos', params);
  }
  
  async getVideo(id: string): Promise<Video> {
    return apiClient.get<Video>(`/videos/${id}`);
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
  
  async getVideoMetrics(id: string): Promise<VideoMetrics> {
    return apiClient.get<VideoMetrics>(`/videos/${id}/metrics`);
  }
}

export const videoApi = new VideoApi();
```

### Dashboard API Service

```typescript
// services/dashboard.ts
import { apiClient } from './api';

interface DashboardOverview {
  metrics: DashboardMetrics;
  chartData: ChartData[];
  recentActivity: ActivityItem[];
  alerts: DashboardAlert[];
}

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
  views: number;
}

interface ActivityItem {
  id: string;
  type: 'video_completed' | 'channel_created' | 'revenue_milestone';
  message: string;
  timestamp: string;
  metadata?: any;
}

class DashboardApi {
  async getOverview(): Promise<DashboardOverview> {
    return apiClient.get<DashboardOverview>('/dashboard/overview');
  }
  
  async getMetrics(period: 'today' | 'week' | 'month'): Promise<DashboardMetrics> {
    return apiClient.get<DashboardMetrics>('/dashboard/metrics', { period });
  }
  
  async getChartData(
    metric: 'revenue' | 'cost' | 'videos' | 'views',
    period: 'week' | 'month' | 'year'
  ): Promise<ChartData[]> {
    return apiClient.get<ChartData[]>('/dashboard/chart', { metric, period });
  }
  
  async getActivity(limit: number = 20): Promise<ActivityItem[]> {
    return apiClient.get<ActivityItem[]>('/dashboard/activity', { limit });
  }
}

export const dashboardApi = new DashboardApi();
```

### Cost API Service

```typescript
// services/costs.ts
import { apiClient } from './api';

interface CostBreakdown {
  ai_generation: number;
  voice_synthesis: number;
  storage: number;
  api_calls: number;
  total: number;
}

interface CostProjection {
  currentMonth: {
    spent: number;
    projected: number;
    daysRemaining: number;
  };
  nextMonth: {
    projected: number;
    assumptions: {
      videosPerDay: number;
      costPerVideo: number;
      channels: number;
    };
  };
  recommendations: string[];
}

interface DailyCostReport {
  date: string;
  totalCost: number;
  videosGenerated: number;
  costPerVideo: number;
  budget: number;
  budgetRemaining: number;
  projectedMonthly: number;
}

interface CostAlertConfig {
  type: 'daily_limit' | 'per_video' | 'monthly_projection';
  threshold: number;
  email?: string;
}

class CostApi {
  async getCostBreakdown(period: 'today' | 'week' | 'month'): Promise<CostBreakdown> {
    return apiClient.get<CostBreakdown>('/costs/breakdown', { period });
  }
  
  async getCostProjection(): Promise<CostProjection> {
    return apiClient.get<CostProjection>('/costs/projection');
  }
  
  async getDailyCosts(): Promise<DailyCostReport> {
    return apiClient.get<DailyCostReport>('/costs/daily-report');
  }
  
  async setCostAlert(config: CostAlertConfig): Promise<void> {
    return apiClient.post<void>('/costs/alerts', config);
  }
  
  async getCostHistory(days: number = 30): Promise<DailyCostReport[]> {
    return apiClient.get<DailyCostReport[]>('/costs/history', { days });
  }
}

export const costApi = new CostApi();
```

### Analytics API Service

```typescript
// services/analytics.ts
import { apiClient } from './api';

interface ChannelAnalytics {
  channelId: string;
  period: string;
  metrics: {
    views: number;
    watchTime: number;
    subscribers: number;
    revenue: number;
    engagement: number;
  };
  topVideos: VideoPerformance[];
  growthRate: number;
}

interface VideoPerformance {
  videoId: string;
  title: string;
  views: number;
  revenue: number;
  roi: number;
  performanceScore: number;
}

class AnalyticsApi {
  async getChannelAnalytics(
    channelId: string,
    period: 'day' | 'week' | 'month'
  ): Promise<ChannelAnalytics> {
    return apiClient.get<ChannelAnalytics>(`/analytics/channels/${channelId}`, { period });
  }
  
  async getVideoAnalytics(videoId: string): Promise<VideoPerformance> {
    return apiClient.get<VideoPerformance>(`/analytics/videos/${videoId}`);
  }
  
  async getTopPerformers(limit: number = 10): Promise<VideoPerformance[]> {
    return apiClient.get<VideoPerformance[]>('/analytics/top-performers', { limit });
  }
  
  async getRevenueAnalytics(period: 'week' | 'month' | 'year'): Promise<RevenueData> {
    return apiClient.get<RevenueData>('/analytics/revenue', { period });
  }
}

export const analyticsApi = new AnalyticsApi();
```

---

## 🔌 WebSocket Integration

### WebSocket Client Implementation

```typescript
// services/websocket.ts
import { useAuthStore } from '@/stores/useAuthStore';
import { useVideoStore } from '@/stores/useVideoStore';
import { useCostStore } from '@/stores/useCostStore';
import { useNotificationStore } from '@/stores/useNotificationStore';

// MVP: Only 3 critical WebSocket events
enum WSEventType {
  VIDEO_COMPLETED = 'video.completed',
  VIDEO_FAILED = 'video.failed',
  COST_ALERT = 'cost.alert'
}

interface WSMessage {
  type: WSEventType;
  payload: any;
  timestamp: string;
}

class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;
  
  constructor() {
    this.connect = this.connect.bind(this);
    this.disconnect = this.disconnect.bind(this);
    this.handleMessage = this.handleMessage.bind(this);
  }
  
  connect() {
    const token = useAuthStore.getState().accessToken;
    if (!token) {
      console.error('No auth token available for WebSocket connection');
      return;
    }
    
    const wsUrl = `${import.meta.env.VITE_WS_URL}/critical?token=${token}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.scheduleReconnect();
    }
  }
  
  private setupEventHandlers() {
    if (!this.ws) return;
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.startPing();
      
      // Subscribe to critical events only
      this.send({
        type: 'subscribe',
        events: [WSEventType.VIDEO_COMPLETED, WSEventType.VIDEO_FAILED, WSEventType.COST_ALERT]
      });
    };
    
    this.ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.stopPing();
      
      if (!this.isIntentionallyClosed) {
        this.scheduleReconnect();
      }
    };
  }
  
  private handleMessage(message: WSMessage) {
    const { showNotification } = useNotificationStore.getState();
    
    switch (message.type) {
      case WSEventType.VIDEO_COMPLETED:
        this.handleVideoCompleted(message.payload);
        showNotification({
          type: 'success',
          message: `Video "${message.payload.title}" completed successfully!`,
          duration: 5000,
        });
        break;
        
      case WSEventType.VIDEO_FAILED:
        this.handleVideoFailed(message.payload);
        showNotification({
          type: 'error',
          message: `Video generation failed: ${message.payload.error}`,
          duration: 10000,
          action: {
            label: 'Retry',
            onClick: () => useVideoStore.getState().retryVideo(message.payload.videoId)
          }
        });
        break;
        
      case WSEventType.COST_ALERT:
        this.handleCostAlert(message.payload);
        showNotification({
          type: 'warning',
          message: message.payload.message,
          duration: 0, // Don't auto-dismiss cost alerts
          action: {
            label: 'View Details',
            onClick: () => window.location.href = '/settings/billing'
          }
        });
        break;
        
      default:
        console.warn('Unknown WebSocket event type:', message.type);
    }
  }
  
  private handleVideoCompleted(payload: any) {
    const { fetchVideoStatus, fetchVideos } = useVideoStore.getState();
    
    // Update video status
    fetchVideoStatus(payload.videoId);
    
    // Refresh video list
    fetchVideos(payload.channelId);
    
    // Update dashboard metrics
    useDashboardStore.getState().fetchDashboard();
  }
  
  private handleVideoFailed(payload: any) {
    const { fetchVideoStatus } = useVideoStore.getState();
    
    // Update video status with error
    fetchVideoStatus(payload.videoId);
  }
  
  private handleCostAlert(payload: any) {
    const { checkAlerts } = useCostStore.getState();
    
    // Update cost alerts
    checkAlerts();
  }
  
  private send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
  
  private startPing() {
    this.pingInterval = setInterval(() => {
      this.send({ type: 'ping' });
    }, 30000); // Ping every 30 seconds
  }
  
  private stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      useNotificationStore.getState().showNotification({
        type: 'error',
        message: 'Real-time updates disconnected. Please refresh the page.',
        duration: 0,
      });
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  disconnect() {
    this.isIntentionallyClosed = true;
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.stopPing();
  }
}

// Singleton instance
export const wsClient = new WebSocketClient();

// React hook for WebSocket connection management
export const useWebSocket = () => {
  useEffect(() => {
    const isAuthenticated = useAuthStore.getState().isAuthenticated;
    
    if (isAuthenticated) {
      wsClient.connect();
    }
    
    return () => {
      wsClient.disconnect();
    };
  }, []);
};
```

---

## 🚨 Error Handling & Retry Logic

### Global Error Handler

```typescript
// services/errorHandler.ts
import { useNotificationStore } from '@/stores/useNotificationStore';

interface ApiError {
  code: string;
  message: string;
  details?: any;
  statusCode?: number;
}

class ErrorHandler {
  private readonly errorMessages: Record<string, string> = {
    // Authentication errors
    'AUTH_4011': 'Your session has expired. Please log in again.',
    'AUTH_4012': 'Invalid credentials. Please check your email and password.',
    'AUTH_4013': 'You don\'t have permission to perform this action.',
    
    // Validation errors
    'VAL_4001': 'Please check your input and try again.',
    'VAL_4002': 'Invalid request format.',
    'VAL_4003': 'Required fields are missing.',
    
    // Resource errors
    'RES_4041': 'The requested resource was not found.',
    'RES_4042': 'Channel not found.',
    'RES_4043': 'Video not found.',
    
    // Rate limiting
    'RATE_4291': 'Too many requests. Please slow down.',
    'RATE_4292': 'You\'ve reached your quota limit. Please upgrade your plan.',
    
    // Server errors
    'SRV_5001': 'Something went wrong. Please try again later.',
    'SRV_5002': 'Service temporarily unavailable.',
    'SRV_5003': 'Database connection error.',
  };
  
  handle(error: any): void {
    console.error('Error caught:', error);
    
    const { showNotification } = useNotificationStore.getState();
    
    // Network errors
    if (error.name === 'NetworkError' || !navigator.onLine) {
      showNotification({
        type: 'error',
        message: 'No internet connection. Please check your network.',
        duration: 0,
      });
      return;
    }
    
    // Timeout errors
    if (error.name === 'AbortError') {
      showNotification({
        type: 'error',
        message: 'Request timed out. Please try again.',
        duration: 5000,
      });
      return;
    }
    
    // API errors
    if (error.code) {
      const message = this.errorMessages[error.code] || error.message || 'An unexpected error occurred.';
      
      showNotification({
        type: 'error',
        message,
        duration: error.statusCode === 429 ? 10000 : 5000,
      });
      
      // Special handling for auth errors
      if (error.code.startsWith('AUTH_')) {
        // Redirect to login if needed
        if (error.code === 'AUTH_4011') {
          useAuthStore.getState().logout();
          window.location.href = '/login';
        }
      }
      
      // Special handling for quota errors
      if (error.code === 'RATE_4292') {
        // Show upgrade modal
        window.location.href = '/settings/billing';
      }
    } else {
      // Generic error
      showNotification({
        type: 'error',
        message: error.message || 'An unexpected error occurred.',
        duration: 5000,
      });
    }
  }
  
  async withRetry<T>(
    fn: () => Promise<T>,
    options: {
      maxAttempts?: number;
      delay?: number;
      backoff?: boolean;
      onRetry?: (attempt: number) => void;
    } = {}
  ): Promise<T> {
    const {
      maxAttempts = 3,
      delay = 1000,
      backoff = true,
      onRetry
    } = options;
    
    let lastError: any;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        // Don't retry certain errors
        if (
          error.code?.startsWith('AUTH_') ||
          error.code?.startsWith('VAL_') ||
          error.statusCode === 404
        ) {
          throw error;
        }
        
        if (attempt < maxAttempts) {
          const waitTime = backoff ? delay * Math.pow(2, attempt - 1) : delay;
          
          if (onRetry) {
            onRetry(attempt);
          }
          
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }
    }
    
    throw lastError;
  }
}

export const errorHandler = new ErrorHandler();

// React hook for error handling
export const useErrorHandler = () => {
  return useCallback((error: any) => {
    errorHandler.handle(error);
  }, []);
};
```

---

## 📊 API Response Caching

### Cache Implementation

```typescript
// services/cache.ts
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class ApiCache {
  private cache = new Map<string, CacheEntry<any>>();
  private readonly defaultTTL = 60000; // 60 seconds
  
  set<T>(key: string, data: T, ttl: number = this.defaultTTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }
  
  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }
    
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.data;
  }
  
  invalidate(pattern?: string): void {
    if (pattern) {
      // Invalidate keys matching pattern
      for (const key of this.cache.keys()) {
        if (key.includes(pattern)) {
          this.cache.delete(key);
        }
      }
    } else {
      // Clear all cache
      this.cache.clear();
    }
  }
  
  // Cleanup expired entries periodically
  startCleanup(): void {
    setInterval(() => {
      const now = Date.now();
      
      for (const [key, entry] of this.cache.entries()) {
        if (now - entry.timestamp > entry.ttl) {
          this.cache.delete(key);
        }
      }
    }, 60000); // Run every minute
  }
}

export const apiCache = new ApiCache();

// Enhanced API client with caching
export const cachedApiClient = {
  async get<T>(
    endpoint: string,
    params?: any,
    options: { cache?: boolean; ttl?: number } = {}
  ): Promise<T> {
    const { cache = true, ttl } = options;
    
    if (cache) {
      const cacheKey = `${endpoint}:${JSON.stringify(params || {})}`;
      const cached = apiCache.get<T>(cacheKey);
      
      if (cached) {
        return cached;
      }
      
      const data = await apiClient.get<T>(endpoint, params);
      apiCache.set(cacheKey, data, ttl);
      
      return data;
    }
    
    return apiClient.get<T>(endpoint, params);
  }
};
```

---

## 🔄 Polling Utilities

### Custom Polling Hook

```typescript
// hooks/usePolling.ts
import { useEffect, useRef, useCallback } from 'react';

interface UsePollingOptions {
  enabled?: boolean;
  interval?: number;
  onError?: (error: Error) => void;
  immediate?: boolean;
}

export const usePolling = (
  callback: () => void | Promise<void>,
  options: UsePollingOptions = {}
) => {
  const {
    enabled = true,
    interval = 60000, // Default 60 seconds
    onError,
    immediate = true
  } = options;
  
  const savedCallback = useRef(callback);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Update callback if it changes
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);
  
  // Setup polling
  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    
    const poll = async () => {
      try {
        await savedCallback.current();
      } catch (error) {
        console.error('Polling error:', error);
        if (onError) {
          onError(error as Error);
        }
      }
    };
    
    // Run immediately if requested
    if (immediate) {
      poll();
    }
    
    // Setup interval
    intervalRef.current = setInterval(poll, interval);
    
    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, interval, onError, immediate]);
  
  // Manual trigger
  const trigger = useCallback(async () => {
    try {
      await savedCallback.current();
    } catch (error) {
      console.error('Manual polling trigger error:', error);
      if (onError) {
        onError(error as Error);
      }
    }
  }, [onError]);
  
  return { trigger };
};

// Usage example in component
const DashboardPage = () => {
  const { fetchDashboard } = useDashboardStore();
  const { fetchQueue } = useVideoStore();
  const { fetchCosts } = useCostStore();
  
  // Poll dashboard every 60 seconds
  usePolling(fetchDashboard, {
    interval: 60000,
    enabled: true
  });
  
  // Poll video queue every 5 seconds when processing
  const hasProcessingVideos = useVideoStore(state => state.processing.length > 0);
  usePolling(fetchQueue, {
    interval: 5000,
    enabled: hasProcessingVideos
  });
  
  // Poll costs every 30 seconds
  usePolling(fetchCosts, {
    interval: 30000,
    enabled: true
  });
  
  return <Dashboard />;
};
```

---

## 🎯 API Integration Best Practices

### 1. Request Deduplication
```typescript
// Prevent duplicate requests
const pendingRequests = new Map<string, Promise<any>>();

export const dedupeRequest = async <T>(
  key: string,
  request: () => Promise<T>
): Promise<T> => {
  const pending = pendingRequests.get(key);
  
  if (pending) {
    return pending;
  }
  
  const promise = request()
    .finally(() => {
      pendingRequests.delete(key);
    });
  
  pendingRequests.set(key, promise);
  
  return promise;
};
```

### 2. Optimistic Updates
```typescript
// Example: Optimistic channel update
const updateChannelOptimistically = async (id: string, updates: any) => {
  // Update UI immediately
  useChannelStore.setState(state => ({
    channels: state.channels.map(ch =>
      ch.id === id ? { ...ch, ...updates } : ch
    )
  }));
  
  try {
    // Make API call
    await channelApi.updateChannel(id, updates);
  } catch (error) {
    // Revert on failure
    await useChannelStore.getState().fetchChannels();
    throw error;
  }
};
```

### 3. Request Cancellation
```typescript
// Cancel requests when component unmounts
export const useCancelableRequest = () => {
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const makeRequest = useCallback(async (request: (signal: AbortSignal) => Promise<any>) => {
    // Cancel previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new controller
    abortControllerRef.current = new AbortController();
    
    try {
      return await request(abortControllerRef.current.signal);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }
      throw error;
    }
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);
  
  return { makeRequest };
};
```

---

## 📋 API Integration Checklist

### For Each API Integration:
- [ ] TypeScript interfaces defined
- [ ] Error handling implemented
- [ ] Loading states managed
- [ ] Cache strategy defined
- [ ] Retry logic added (where appropriate)
- [ ] Request deduplication
- [ ] Optimistic updates (where applicable)
- [ ] WebSocket fallback for critical updates
- [ ] Performance monitoring
- [ ] Documentation updated

---

**Remember**: Always handle errors gracefully, show meaningful messages to users, and ensure the UI remains responsive even when the API is slow or unavailable. 🚀