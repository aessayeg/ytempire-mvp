# YTEMPIRE React Engineer - API & Integration Reference
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: API Specifications & Integration Guide

---

## 1. API Configuration

### 1.1 Base Configuration

```typescript
// services/api.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { useAuthStore } from '@/stores/useAuthStore';

// API Base URLs by environment
const API_URLS = {
  development: 'http://localhost:8000/api/v1',
  staging: 'https://staging-api.ytempire.com/api/v1',
  production: 'https://api.ytempire.com/api/v1'
};

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_URLS[import.meta.env.MODE] || API_URLS.development,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Client-Version': '1.0.0',
    'X-Platform': 'web'
  }
});

// Request interceptor for auth
apiClient.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add request ID for tracking
    config.headers['X-Request-ID'] = crypto.randomUUID();
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for token refresh
apiClient.interceptors.response.use(
  (response) => response.data,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        await useAuthStore.getState().refreshAuth();
        return apiClient(originalRequest);
      } catch (refreshError) {
        useAuthStore.getState().logout();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
```

### 1.2 Standard Response Types

```typescript
// types/api.types.ts

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata: ResponseMetadata;
  pagination?: PaginationInfo;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  field?: string;
}

export interface ResponseMetadata {
  timestamp: string;
  requestId: string;
  processingTime: number;
  version: string;
}

export interface PaginationInfo {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrevious: boolean;
}
```

---

## 2. Authentication Endpoints

### 2.1 Auth Service Implementation

```typescript
// services/auth.ts
import apiClient from './api';
import type { User, AuthTokens } from '@/types/models.types';

interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

interface RegisterRequest {
  email: string;
  password: string;
  fullName: string;
  timezone: string;
  agreedToTerms: boolean;
}

interface AuthResponse {
  user: User;
  tokens: AuthTokens;
}

class AuthService {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    return apiClient.post('/auth/login', credentials);
  }
  
  async register(data: RegisterRequest): Promise<AuthResponse> {
    return apiClient.post('/auth/register', data);
  }
  
  async logout(): Promise<void> {
    return apiClient.post('/auth/logout');
  }
  
  async refresh(refreshToken: string): Promise<AuthTokens> {
    return apiClient.post('/auth/refresh', { refreshToken });
  }
  
  async verifyEmail(token: string): Promise<void> {
    return apiClient.post('/auth/verify-email', { token });
  }
  
  async forgotPassword(email: string): Promise<void> {
    return apiClient.post('/auth/forgot-password', { email });
  }
  
  async resetPassword(token: string, password: string): Promise<void> {
    return apiClient.post('/auth/reset-password', { token, password });
  }
  
  async getProfile(): Promise<User> {
    return apiClient.get('/auth/profile');
  }
  
  async updateProfile(data: Partial<User>): Promise<User> {
    return apiClient.patch('/auth/profile', data);
  }
}

export const authService = new AuthService();
```

---

## 3. Channel Management API

### 3.1 Channel Endpoints

```typescript
// services/channels.ts
import apiClient from './api';
import type { Channel, ChannelMetrics } from '@/types/models.types';

interface CreateChannelRequest {
  name: string;
  niche: string;
  youtubeChannelId?: string;
  dailyVideoLimit?: number;
  automationEnabled?: boolean;
}

interface UpdateChannelRequest {
  name?: string;
  status?: 'active' | 'paused';
  dailyVideoLimit?: number;
  automationEnabled?: boolean;
}

class ChannelService {
  // GET /channels - List all channels
  async getChannels(): Promise<Channel[]> {
    return apiClient.get('/channels');
  }
  
  // GET /channels/:id - Get single channel
  async getChannel(id: string): Promise<Channel> {
    return apiClient.get(`/channels/${id}`);
  }
  
  // POST /channels - Create channel
  async createChannel(data: CreateChannelRequest): Promise<Channel> {
    return apiClient.post('/channels', data);
  }
  
  // PATCH /channels/:id - Update channel
  async updateChannel(id: string, data: UpdateChannelRequest): Promise<Channel> {
    return apiClient.patch(`/channels/${id}`, data);
  }
  
  // DELETE /channels/:id - Delete channel
  async deleteChannel(id: string): Promise<void> {
    return apiClient.delete(`/channels/${id}`);
  }
  
  // POST /channels/:id/toggle-automation
  async toggleAutomation(id: string, enabled: boolean): Promise<Channel> {
    return apiClient.post(`/channels/${id}/toggle-automation`, { enabled });
  }
  
  // GET /channels/:id/metrics
  async getChannelMetrics(id: string): Promise<ChannelMetrics> {
    return apiClient.get(`/channels/${id}/metrics`);
  }
  
  // POST /channels/:id/connect-youtube
  async connectYouTube(id: string, authCode: string): Promise<Channel> {
    return apiClient.post(`/channels/${id}/connect-youtube`, { authCode });
  }
  
  // POST /channels/:id/disconnect-youtube
  async disconnectYouTube(id: string): Promise<Channel> {
    return apiClient.post(`/channels/${id}/disconnect-youtube`);
  }
  
  // GET /channels/:id/analytics
  async getChannelAnalytics(id: string, period: string): Promise<any> {
    return apiClient.get(`/channels/${id}/analytics`, { 
      params: { period } 
    });
  }
}

export const channelService = new ChannelService();
```

---

## 4. Video Management API

### 4.1 Video Service Implementation

```typescript
// services/videos.ts
import apiClient from './api';
import type { Video, VideoMetrics, VideoQueue } from '@/types/models.types';

interface GenerateVideoRequest {
  channelId: string;
  topic?: string;
  style: 'educational' | 'entertainment' | 'tutorial' | 'news';
  length: 'short' | 'medium' | 'long';
  priority: number; // 1-10
  scheduledFor?: string; // ISO 8601
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
  sortBy?: 'createdAt' | 'title' | 'views';
  sortOrder?: 'asc' | 'desc';
}

class VideoService {
  // GET /videos - List videos with filters
  async getVideos(params?: VideoListParams): Promise<Video[]> {
    return apiClient.get('/videos', { params });
  }
  
  // GET /videos/:id - Get single video
  async getVideo(id: string): Promise<Video> {
    return apiClient.get(`/videos/${id}`);
  }
  
  // POST /videos/generate - Generate new video
  async generateVideo(data: GenerateVideoRequest): Promise<VideoGenerationResponse> {
    return apiClient.post('/videos/generate', data);
  }
  
  // GET /videos/:id/status - Get video generation status
  async getVideoStatus(id: string): Promise<{
    status: string;
    progress: number;
    stage: string;
    estimatedCompletion: string;
  }> {
    return apiClient.get(`/videos/${id}/status`);
  }
  
  // POST /videos/:id/retry - Retry failed video
  async retryVideo(id: string): Promise<VideoGenerationResponse> {
    return apiClient.post(`/videos/${id}/retry`);
  }
  
  // POST /videos/:id/cancel - Cancel video generation
  async cancelVideo(id: string): Promise<void> {
    return apiClient.post(`/videos/${id}/cancel`);
  }
  
  // DELETE /videos/:id - Delete video
  async deleteVideo(id: string): Promise<void> {
    return apiClient.delete(`/videos/${id}`);
  }
  
  // GET /videos/:id/metrics - Get video metrics
  async getVideoMetrics(id: string): Promise<VideoMetrics> {
    return apiClient.get(`/videos/${id}/metrics`);
  }
  
  // GET /videos/queue - Get generation queue
  async getQueue(): Promise<VideoQueue> {
    return apiClient.get('/videos/queue');
  }
  
  // PATCH /videos/:id/metadata - Update video metadata
  async updateVideoMetadata(id: string, metadata: {
    title?: string;
    description?: string;
    tags?: string[];
  }): Promise<Video> {
    return apiClient.patch(`/videos/${id}/metadata`, metadata);
  }
}

export const videoService = new VideoService();
```

---

## 5. Dashboard & Analytics API

### 5.1 Dashboard Service

```typescript
// services/dashboard.ts
import apiClient from './api';

interface DashboardOverview {
  metrics: {
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
  };
  chartData: ChartData[];
  recentActivity: ActivityItem[];
  alerts: Alert[];
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
  metadata?: Record<string, any>;
}

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'error';
  title: string;
  message: string;
  actionUrl?: string;
}

class DashboardService {
  // GET /dashboard/overview
  async getOverview(): Promise<DashboardOverview> {
    return apiClient.get('/dashboard/overview');
  }
  
  // GET /dashboard/metrics
  async getMetrics(period: 'day' | 'week' | 'month' | 'year'): Promise<any> {
    return apiClient.get('/dashboard/metrics', { params: { period } });
  }
  
  // GET /dashboard/revenue
  async getRevenueData(startDate: string, endDate: string): Promise<ChartData[]> {
    return apiClient.get('/dashboard/revenue', {
      params: { startDate, endDate }
    });
  }
  
  // GET /dashboard/costs
  async getCostBreakdown(): Promise<{
    total: number;
    breakdown: {
      ai: number;
      voice: number;
      rendering: number;
      storage: number;
    };
    trend: number; // Percentage change
  }> {
    return apiClient.get('/dashboard/costs');
  }
  
  // GET /dashboard/activity
  async getRecentActivity(limit: number = 10): Promise<ActivityItem[]> {
    return apiClient.get('/dashboard/activity', { params: { limit } });
  }
  
  // GET /dashboard/alerts
  async getAlerts(): Promise<Alert[]> {
    return apiClient.get('/dashboard/alerts');
  }
  
  // POST /dashboard/alerts/:id/dismiss
  async dismissAlert(id: string): Promise<void> {
    return apiClient.post(`/dashboard/alerts/${id}/dismiss`);
  }
}

export const dashboardService = new DashboardService();
```

---

## 6. WebSocket Integration (3 Critical Events Only)

### 6.1 WebSocket Client

```typescript
// services/websocket.ts
import { useAuthStore } from '@/stores/useAuthStore';
import { useVideoStore } from '@/stores/useVideoStore';
import { useCostStore } from '@/stores/useCostStore';
import { toast } from 'react-hot-toast';

// MVP: Only 3 critical WebSocket events
enum WSEventType {
  VIDEO_COMPLETED = 'video.completed',
  VIDEO_FAILED = 'video.failed',
  COST_ALERT = 'cost.alert'
}

interface WSMessage<T = any> {
  id: string;
  type: WSEventType;
  payload: T;
  timestamp: string;
  version: string;
}

interface VideoCompletedPayload {
  videoId: string;
  channelId: string;
  channelName: string;
  title: string;
  thumbnail: string;
  youtubeUrl: string;
  duration: number;
  cost: {
    total: number;
    breakdown: {
      ai: number;
      voice: number;
      rendering: number;
    };
  };
}

interface VideoFailedPayload {
  videoId: string;
  channelId: string;
  channelName: string;
  error: {
    stage: string;
    code: string;
    message: string;
    retryable: boolean;
  };
  cost: {
    incurred: number;
    refundable: number;
  };
}

interface CostAlertPayload {
  alertId: string;
  severity: 'warning' | 'critical';
  type: 'approaching_limit' | 'limit_exceeded';
  message: string;
  currentCost: number;
  limit: number;
  percentage: number;
}

class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;
  
  connect() {
    const token = useAuthStore.getState().accessToken;
    if (!token) {
      console.error('No auth token for WebSocket');
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
    switch (message.type) {
      case WSEventType.VIDEO_COMPLETED:
        this.handleVideoCompleted(message.payload as VideoCompletedPayload);
        break;
        
      case WSEventType.VIDEO_FAILED:
        this.handleVideoFailed(message.payload as VideoFailedPayload);
        break;
        
      case WSEventType.COST_ALERT:
        this.handleCostAlert(message.payload as CostAlertPayload);
        break;
        
      default:
        console.warn('Unknown WebSocket event type:', message.type);
    }
  }
  
  private handleVideoCompleted(payload: VideoCompletedPayload) {
    // Update video store
    useVideoStore.getState().updateVideoStatus(payload.videoId, 'completed');
    
    // Show success notification
    toast.success(
      `Video "${payload.title}" completed! Cost: ${payload.cost.total.toFixed(2)}`,
      { duration: 5000 }
    );
    
    // Refresh dashboard metrics
    // Note: Dashboard will also update via polling
  }
  
  private handleVideoFailed(payload: VideoFailedPayload) {
    // Update video store
    useVideoStore.getState().updateVideoStatus(payload.videoId, 'failed');
    
    // Show error notification
    const message = payload.error.retryable
      ? `Video generation failed: ${payload.error.message}. Click to retry.`
      : `Video generation failed: ${payload.error.message}`;
    
    toast.error(message, {
      duration: 10000,
      onClick: payload.error.retryable
        ? () => videoService.retryVideo(payload.videoId)
        : undefined
    });
  }
  
  private handleCostAlert(payload: CostAlertPayload) {
    // Update cost store
    useCostStore.getState().setCostAlert(payload);
    
    // Show alert based on severity
    const toastFn = payload.severity === 'critical' ? toast.error : toast.warning;
    
    toastFn(payload.message, {
      duration: payload.severity === 'critical' ? 0 : 10000, // Critical alerts don't auto-dismiss
      id: payload.alertId // Prevent duplicate alerts
    });
  }
  
  private startPing() {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
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
      toast.error('Real-time updates disconnected. Please refresh the page.');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  disconnect() {
    this.isIntentionallyClosed = true;
    this.stopPing();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  getReadyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }
}

export const wsClient = new WebSocketClient();
```

### 6.2 WebSocket Hook

```typescript
// hooks/useWebSocket.ts
import { useEffect } from 'react';
import { wsClient } from '@/services/websocket';
import { useAuthStore } from '@/stores/useAuthStore';

export function useWebSocket() {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  
  useEffect(() => {
    if (isAuthenticated) {
      wsClient.connect();
      
      return () => {
        wsClient.disconnect();
      };
    }
  }, [isAuthenticated]);
  
  return {
    isConnected: wsClient.getReadyState() === WebSocket.OPEN
  };
}
```

---

## 7. Error Handling

### 7.1 Error Codes Reference

```typescript
// constants/errorCodes.ts
export const ERROR_CODES = {
  // Authentication (1xxx)
  INVALID_CREDENTIALS: '1001',
  ACCOUNT_NOT_FOUND: '1002',
  ACCOUNT_SUSPENDED: '1003',
  EMAIL_NOT_VERIFIED: '1004',
  TOKEN_EXPIRED: '1005',
  TOKEN_INVALID: '1006',
  INSUFFICIENT_PERMISSIONS: '1007',
  
  // Validation (2xxx)
  INVALID_INPUT: '2001',
  REQUIRED_FIELD_MISSING: '2002',
  VALUE_OUT_OF_RANGE: '2003',
  DUPLICATE_ENTRY: '2004',
  
  // Business Logic (3xxx)
  CHANNEL_LIMIT_EXCEEDED: '3001',
  VIDEO_LIMIT_EXCEEDED: '3002',
  COST_LIMIT_EXCEEDED: '3003',
  INSUFFICIENT_CREDITS: '3004',
  FEATURE_NOT_AVAILABLE: '3005',
  
  // External Services (4xxx)
  YOUTUBE_API_ERROR: '4001',
  OPENAI_API_ERROR: '4002',
  ELEVENLABS_API_ERROR: '4003',
  PAYMENT_ERROR: '4004',
  
  // System (5xxx)
  INTERNAL_SERVER_ERROR: '5001',
  DATABASE_ERROR: '5002',
  SERVICE_UNAVAILABLE: '5003'
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];
```

### 7.2 Error Handler Service

```typescript
// services/errorHandler.ts
import { toast } from 'react-hot-toast';
import { ERROR_CODES } from '@/constants/errorCodes';

interface ErrorHandlerOptions {
  showToast?: boolean;
  logToConsole?: boolean;
  logToSentry?: boolean;
}

class ErrorHandler {
  handle(error: any, options: ErrorHandlerOptions = {}) {
    const {
      showToast = true,
      logToConsole = true,
      logToSentry = true
    } = options;
    
    // Extract error details
    const errorCode = error?.response?.data?.error?.code;
    const errorMessage = error?.response?.data?.error?.message || error.message;
    const statusCode = error?.response?.status;
    
    // Log to console in development
    if (logToConsole && import.meta.env.DEV) {
      console.error('Error:', {
        code: errorCode,
        message: errorMessage,
        status: statusCode,
        full: error
      });
    }
    
    // Send to Sentry in production
    if (logToSentry && import.meta.env.PROD) {
      // Sentry integration would go here
    }
    
    // Show user-friendly toast
    if (showToast) {
      const toastMessage = this.getUserMessage(errorCode, errorMessage);
      toast.error(toastMessage);
    }
    
    return {
      code: errorCode,
      message: errorMessage,
      status: statusCode
    };
  }
  
  private getUserMessage(code?: string, defaultMessage?: string): string {
    switch (code) {
      case ERROR_CODES.INVALID_CREDENTIALS:
        return 'Invalid email or password';
      
      case ERROR_CODES.TOKEN_EXPIRED:
        return 'Your session has expired. Please log in again.';
      
      case ERROR_CODES.CHANNEL_LIMIT_EXCEEDED:
        return 'You have reached the maximum number of channels (5)';
      
      case ERROR_CODES.VIDEO_LIMIT_EXCEEDED:
        return 'Daily video generation limit reached';
      
      case ERROR_CODES.COST_LIMIT_EXCEEDED:
        return 'Cost limit exceeded. Please check your billing settings.';
      
      case ERROR_CODES.YOUTUBE_API_ERROR:
        return 'YouTube service is temporarily unavailable';
      
      case ERROR_CODES.INTERNAL_SERVER_ERROR:
        return 'Something went wrong. Please try again later.';
      
      default:
        return defaultMessage || 'An unexpected error occurred';
    }
  }
}

export const errorHandler = new ErrorHandler();
```

---

**Document Status**: FINAL - Consolidated Version  
**Next Review**: API Contract Review Week 4  
**Owner**: Frontend Team Lead  
**Questions**: Contact via #frontend-team Slack