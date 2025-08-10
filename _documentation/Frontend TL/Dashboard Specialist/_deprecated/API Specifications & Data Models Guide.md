# API Specifications & Data Models Guide
## For: Dashboard Specialist | YTEMPIRE Frontend Team

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**Status**: Implementation Ready

---

## Executive Summary

This document provides complete API specifications, data models, and integration patterns for the YTEMPIRE dashboard. All endpoints, request/response schemas, and TypeScript interfaces are defined to ensure seamless frontend-backend integration.

### API Overview
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: JWT Bearer tokens
- **Content-Type**: `application/json`
- **Response Time Target**: <1 second for all endpoints

---

## 1. Authentication API

### 1.1 Login Endpoint

```typescript
// POST /api/v1/auth/login
interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  success: true;
  data: {
    accessToken: string;      // JWT, expires in 1 hour
    refreshToken: string;     // JWT, expires in 7 days
    user: {
      id: string;
      email: string;
      name: string;
      role: 'user' | 'admin';
      channelLimit: number;   // 5 for MVP
      subscription: {
        plan: 'starter' | 'growth' | 'enterprise';
        status: 'active' | 'trial' | 'cancelled';
        trialEndsAt?: string;
      };
    };
  };
  metadata: {
    timestamp: string;
    requestId: string;
  };
}

// Error Response
interface AuthErrorResponse {
  success: false;
  error: {
    code: 'AUTH_INVALID_CREDENTIALS' | 'AUTH_ACCOUNT_LOCKED';
    message: string;
    details?: any;
  };
  metadata: {
    timestamp: string;
    requestId: string;
  };
}
```

### 1.2 Token Refresh

```typescript
// POST /api/v1/auth/refresh
interface RefreshTokenRequest {
  refreshToken: string;
}

interface RefreshTokenResponse {
  success: true;
  data: {
    accessToken: string;
    refreshToken: string;  // New refresh token
    expiresIn: number;     // Seconds until expiration
  };
}
```

### 1.3 Authentication Headers

```typescript
// Required headers for authenticated requests
interface AuthHeaders {
  'Authorization': `Bearer ${accessToken}`;
  'X-Request-ID': string;  // UUID for request tracking
}

// JWT Token Payload Structure
interface JWTPayload {
  sub: string;        // User ID
  email: string;
  role: string;
  iat: number;        // Issued at
  exp: number;        // Expiration
  jti: string;        // JWT ID for revocation
}
```

---

## 2. Dashboard API

### 2.1 Dashboard Overview

```typescript
// GET /api/v1/dashboard/overview
// Query params: ?period=today|week|month

interface DashboardOverviewRequest {
  period?: 'today' | 'week' | 'month';  // Default: 'today'
}

interface DashboardOverviewResponse {
  success: true;
  data: {
    metrics: {
      // Channel Metrics
      totalChannels: number;
      activeChannels: number;
      pausedChannels: number;
      
      // Video Metrics
      videosToday: number;
      videosThisWeek: number;
      videosProcessing: number;
      videosFailed: number;
      videosQueued: number;
      
      // Financial Metrics
      revenueToday: number;
      revenueThisWeek: number;
      revenueThisMonth: number;
      costToday: number;
      avgCostPerVideo: number;
      projectedMonthlyCost: number;
      
      // Performance Metrics
      automationPercentage: number;  // 0-100
      avgGenerationTime: number;     // seconds
      successRate: number;           // 0-100
      
      // Alerts
      costAlert: {
        active: boolean;
        level: 'none' | 'warning' | 'critical' | 'limit';
        currentCost: number;
        threshold: number;
        message?: string;
      };
    };
    
    channels: ChannelSummary[];
    recentVideos: VideoSummary[];
    
    chartData: {
      revenueChart: ChartDataPoint[];
      videoChart: ChartDataPoint[];
      costChart: ChartDataPoint[];
    };
  };
  metadata: {
    timestamp: string;
    requestId: string;
    cacheAge: number;  // Seconds since last update
  };
}

// Sub-types for Dashboard
interface ChannelSummary {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'error';
  videoCount: number;
  todayVideos: number;
  revenue: number;
  thumbnail?: string;
}

interface VideoSummary {
  id: string;
  channelId: string;
  channelName: string;
  title: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  createdAt: string;
  cost: number;
  stage?: 'script' | 'audio' | 'video' | 'upload';
}

interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}
```

---

## 3. Channel Management API

### 3.1 List Channels

```typescript
// GET /api/v1/channels
interface ChannelListResponse {
  success: true;
  data: {
    channels: Channel[];
    summary: {
      total: number;
      active: number;
      paused: number;
      limit: number;  // 5 for MVP
    };
  };
}

// Complete Channel Model
interface Channel {
  id: string;
  name: string;
  youtubeChannelId: string;
  niche: string;
  status: 'active' | 'paused' | 'error';
  automationEnabled: boolean;
  
  settings: {
    dailyVideoLimit: number;    // 1-3 for MVP
    targetAudience: string;
    primaryLanguage: string;
    videoLength: 'short' | 'medium' | 'long';
    uploadSchedule: {
      enabled: boolean;
      timezone: string;
      slots: string[];          // ['09:00', '14:00', '19:00']
    };
  };
  
  statistics: {
    totalVideos: number;
    videosThisMonth: number;
    totalRevenue: number;
    monthlyRevenue: number;
    avgViews: number;
    avgCTR: number;
    subscribers: number;
  };
  
  costs: {
    totalSpent: number;
    monthlySpent: number;
    avgCostPerVideo: number;
    lastVideoCode: number;
  };
  
  metadata: {
    createdAt: string;
    updatedAt: string;
    lastVideoAt?: string;
    nextVideoAt?: string;
    errorMessage?: string;
    syncStatus: 'synced' | 'syncing' | 'error';
  };
}
```

### 3.2 Create Channel

```typescript
// POST /api/v1/channels
interface CreateChannelRequest {
  name: string;
  niche: string;
  targetAudience: string;
  primaryLanguage: string;
  videoLength: 'short' | 'medium' | 'long';
  dailyVideoLimit: number;  // 1-3
}

interface CreateChannelResponse {
  success: true;
  data: {
    channel: Channel;
    setupUrl: string;  // YouTube OAuth URL
  };
}
```

### 3.3 Update Channel

```typescript
// PATCH /api/v1/channels/:id
interface UpdateChannelRequest {
  name?: string;
  status?: 'active' | 'paused';
  automationEnabled?: boolean;
  settings?: Partial<Channel['settings']>;
}

interface UpdateChannelResponse {
  success: true;
  data: {
    channel: Channel;
    changes: string[];  // List of changed fields
  };
}
```

---

## 4. Video Management API

### 4.1 Video Queue Status

```typescript
// GET /api/v1/videos/queue
interface VideoQueueResponse {
  success: true;
  data: {
    queue: QueuedVideo[];
    processing: ProcessingVideo[];
    completed: CompletedVideo[];
    failed: FailedVideo[];
    
    stats: {
      queueDepth: number;
      processingCount: number;
      avgWaitTime: number;      // seconds
      avgProcessingTime: number; // seconds
      successRate: number;       // percentage
    };
  };
}

// Video Type Definitions
interface BaseVideo {
  id: string;
  channelId: string;
  channelName: string;
  title: string;
  description: string;
  tags: string[];
  createdAt: string;
}

interface QueuedVideo extends BaseVideo {
  status: 'queued';
  position: number;
  priority: number;
  estimatedStartTime: string;
}

interface ProcessingVideo extends BaseVideo {
  status: 'processing';
  stage: 'script' | 'audio' | 'video' | 'upload';
  progress: number;  // 0-100
  startedAt: string;
  estimatedCompletion: string;
  currentCost: number;
}

interface CompletedVideo extends BaseVideo {
  status: 'completed';
  youtubeUrl: string;
  youtubeVideoId: string;
  thumbnail: string;
  duration: number;  // seconds
  completedAt: string;
  metrics: {
    generationTime: number;
    cost: number;
    initialViews?: number;
    initialLikes?: number;
  };
}

interface FailedVideo extends BaseVideo {
  status: 'failed';
  failedAt: string;
  error: {
    stage: 'script' | 'audio' | 'video' | 'upload';
    code: string;
    message: string;
    details?: any;
    retryable: boolean;
  };
  costIncurred: number;
  retryCount: number;
}
```

### 4.2 Generate Video

```typescript
// POST /api/v1/videos/generate
interface GenerateVideoRequest {
  channelId: string;
  topic?: string;              // Optional, AI will choose if not provided
  style: 'educational' | 'entertainment' | 'news' | 'tutorial';
  length: 'short' | 'medium' | 'long';
  priority?: number;           // 1-10, default 5
  scheduledFor?: string;       // ISO datetime for scheduled generation
}

interface GenerateVideoResponse {
  success: true;
  data: {
    video: QueuedVideo;
    estimatedCost: number;
    queuePosition: number;
    estimatedCompletionTime: string;
  };
}
```

---

## 5. Cost Management API

### 5.1 Cost Breakdown

```typescript
// GET /api/v1/costs/breakdown
// Query params: ?period=today|week|month&channelId=xxx

interface CostBreakdownRequest {
  period?: 'today' | 'week' | 'month';
  channelId?: string;  // Optional filter by channel
}

interface CostBreakdownResponse {
  success: true;
  data: {
    period: string;
    totalCost: number;
    breakdown: {
      aiGeneration: number;      // GPT-4 costs
      voiceSynthesis: number;    // ElevenLabs costs
      videoRendering: number;    // GPU costs
      storage: number;           // S3/CDN costs
      apiCalls: number;          // YouTube API costs
      other: number;
    };
    
    byChannel: Array<{
      channelId: string;
      channelName: string;
      cost: number;
      videoCount: number;
      avgCostPerVideo: number;
    }>;
    
    byDay: Array<{
      date: string;
      cost: number;
      videoCount: number;
    }>;
    
    projections: {
      dailyRunRate: number;
      monthlyProjection: number;
      remainingBudget: number;
      daysUntilBudgetLimit: number;
    };
  };
}
```

### 5.2 Cost Alerts Configuration

```typescript
// GET /api/v1/costs/alerts
interface CostAlertsResponse {
  success: true;
  data: {
    alerts: CostAlert[];
    currentCost: {
      today: number;
      month: number;
    };
  };
}

// POST /api/v1/costs/alerts
interface CreateCostAlertRequest {
  type: 'daily' | 'monthly' | 'per_video';
  threshold: number;
  action: 'notify' | 'pause_generation' | 'both';
  channels?: string[];  // Specific channels or all
}

interface CostAlert {
  id: string;
  type: 'daily' | 'monthly' | 'per_video';
  threshold: number;
  action: 'notify' | 'pause_generation' | 'both';
  enabled: boolean;
  channels: string[] | 'all';
  lastTriggered?: string;
  createdAt: string;
}
```

---

## 6. Analytics API

### 6.1 Performance Metrics

```typescript
// GET /api/v1/analytics/performance
// Query params: ?period=day|week|month&channelId=xxx

interface PerformanceMetricsRequest {
  period: 'day' | 'week' | 'month';
  channelId?: string;
  metrics?: string[];  // Filter specific metrics
}

interface PerformanceMetricsResponse {
  success: true;
  data: {
    overview: {
      totalViews: number;
      totalRevenue: number;
      totalVideos: number;
      avgViewsPerVideo: number;
      avgRevenuePerVideo: number;
      growthRate: number;  // Percentage
    };
    
    timeline: Array<{
      date: string;
      views: number;
      revenue: number;
      videos: number;
      subscribers: number;
      ctr: number;
      watchTime: number;  // minutes
    }>;
    
    topVideos: Array<{
      videoId: string;
      title: string;
      channelName: string;
      views: number;
      revenue: number;
      ctr: number;
      thumbnail: string;
    }>;
    
    channelComparison: Array<{
      channelId: string;
      channelName: string;
      metrics: {
        views: number;
        revenue: number;
        videos: number;
        avgPerformance: number;
      };
    }>;
  };
}
```

---

## 7. Error Response Standards

### 7.1 Standard Error Format

```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: string;        // CATEGORY_SPECIFIC_ERROR
    message: string;     // Human-readable message
    field?: string;      // For validation errors
    details?: any;       // Additional error context
  };
  metadata: {
    timestamp: string;
    requestId: string;
    traceId?: string;    // For distributed tracing
  };
}

// Error Code Categories
enum ErrorCategory {
  AUTH = 'AUTH',           // Authentication/Authorization
  VALIDATION = 'VAL',      // Input validation
  RESOURCE = 'RES',        // Resource not found/conflict
  RATE_LIMIT = 'RATE',     // Rate limiting
  PAYMENT = 'PAY',         // Payment/billing issues
  EXTERNAL = 'EXT',        // External service errors
  SERVER = 'SRV'           // Internal server errors
}

// Common Error Codes
const ERROR_CODES = {
  // Authentication
  AUTH_TOKEN_EXPIRED: 'AUTH_001',
  AUTH_TOKEN_INVALID: 'AUTH_002',
  AUTH_INSUFFICIENT_PERMISSIONS: 'AUTH_003',
  
  // Validation
  VAL_REQUIRED_FIELD: 'VAL_001',
  VAL_INVALID_FORMAT: 'VAL_002',
  VAL_OUT_OF_RANGE: 'VAL_003',
  
  // Resources
  RES_NOT_FOUND: 'RES_001',
  RES_ALREADY_EXISTS: 'RES_002',
  RES_LIMIT_EXCEEDED: 'RES_003',  // e.g., channel limit
  
  // Rate Limiting
  RATE_LIMIT_EXCEEDED: 'RATE_001',
  RATE_QUOTA_EXCEEDED: 'RATE_002',
  
  // External Services
  EXT_YOUTUBE_ERROR: 'EXT_001',
  EXT_AI_SERVICE_ERROR: 'EXT_002',
  EXT_PAYMENT_FAILED: 'EXT_003'
};
```

### 7.2 HTTP Status Code Mapping

```typescript
const STATUS_CODE_MAPPING = {
  // Success
  200: 'OK',
  201: 'Created',
  204: 'No Content',
  
  // Client Errors
  400: 'Bad Request',        // Validation errors
  401: 'Unauthorized',       // Missing/invalid auth
  403: 'Forbidden',          // Insufficient permissions
  404: 'Not Found',          // Resource not found
  409: 'Conflict',           // Resource conflict
  422: 'Unprocessable Entity', // Business logic errors
  429: 'Too Many Requests',  // Rate limiting
  
  // Server Errors
  500: 'Internal Server Error',
  502: 'Bad Gateway',        // External service error
  503: 'Service Unavailable',
  504: 'Gateway Timeout'
};
```

---

## 8. WebSocket Event Specifications

### 8.1 WebSocket Connection

```typescript
// WebSocket URL: ws://localhost:8000/ws/{userId}

// Connection Message
interface WSConnectionMessage {
  type: 'connection';
  data: {
    status: 'connected';
    userId: string;
    timestamp: string;
  };
}

// Authentication Message (sent immediately after connection)
interface WSAuthMessage {
  type: 'auth';
  data: {
    token: string;  // JWT access token
  };
}
```

### 8.2 Critical Event Types (MVP)

```typescript
// 1. Video Completed Event
interface VideoCompletedEvent {
  type: 'video.completed';
  data: {
    videoId: string;
    channelId: string;
    channelName: string;
    title: string;
    youtubeUrl: string;
    thumbnail: string;
    cost: number;
    generationTime: number;  // seconds
    metrics: {
      scriptQuality: number;   // 0-100
      audioQuality: number;    // 0-100
      videoQuality: number;    // 0-100
    };
    timestamp: string;
  };
}

// 2. Video Failed Event
interface VideoFailedEvent {
  type: 'video.failed';
  data: {
    videoId: string;
    channelId: string;
    channelName: string;
    error: {
      stage: 'script' | 'audio' | 'video' | 'upload';
      code: string;
      message: string;
      details?: any;
      retryable: boolean;
      suggestedAction?: string;
    };
    costIncurred: number;
    timestamp: string;
  };
}

// 3. Cost Alert Event
interface CostAlertEvent {
  type: 'cost.alert';
  data: {
    alertId: string;
    level: 'warning' | 'critical' | 'limit';
    currentCost: number;
    threshold: number;
    message: string;
    period: 'daily' | 'monthly';
    affectedChannels: string[];
    suggestedAction: string;
    remainingBudget: number;
    projectedOverage?: number;
    timestamp: string;
  };
}

// Union type for all WebSocket events
type WebSocketEvent = 
  | WSConnectionMessage 
  | VideoCompletedEvent 
  | VideoFailedEvent 
  | CostAlertEvent;
```

---

## 9. Type Definitions Index

### 9.1 Core Types

```typescript
// User Type
interface User {
  id: string;
  email: string;
  name: string;
  role: 'user' | 'admin';
  avatar?: string;
  
  subscription: {
    plan: 'starter' | 'growth' | 'enterprise';
    status: 'active' | 'trial' | 'cancelled' | 'past_due';
    trialEndsAt?: string;
    nextBillingDate?: string;
    cancelledAt?: string;
  };
  
  limits: {
    maxChannels: number;      // 5 for MVP
    maxDailyVideos: number;   // 15 for MVP
    maxMonthlyVideos: number; // 450 for MVP
    maxCostPerVideo: number;  // $0.50 for MVP
  };
  
  preferences: {
    theme: 'light' | 'dark';
    timezone: string;
    notifications: {
      email: boolean;
      videoComplete: boolean;
      costAlerts: boolean;
      weeklyReport: boolean;
    };
  };
  
  metadata: {
    createdAt: string;
    updatedAt: string;
    lastLoginAt: string;
    onboardingCompleted: boolean;
  };
}

// Dashboard Metrics Type
interface DashboardMetrics {
  // Real-time metrics (updates via WebSocket)
  videosProcessing: number;
  currentDailyCost: number;
  activeAlerts: Alert[];
  
  // Periodic metrics (60-second polling)
  channelMetrics: {
    total: number;
    active: number;
    paused: number;
    error: number;
  };
  
  videoMetrics: {
    today: number;
    thisWeek: number;
    thisMonth: number;
    allTime: number;
    successRate: number;
    avgGenerationTime: number;
  };
  
  financialMetrics: {
    revenueToday: number;
    revenueThisWeek: number;
    revenueThisMonth: number;
    revenueAllTime: number;
    costToday: number;
    costThisMonth: number;
    profitMargin: number;
    roi: number;
  };
  
  performanceMetrics: {
    automationRate: number;
    avgVideoQuality: number;
    avgViewsPerVideo: number;
    avgCTR: number;
    channelGrowthRate: number;
  };
}

// Alert Type
interface Alert {
  id: string;
  type: 'cost' | 'performance' | 'error' | 'info';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionable: boolean;
  action?: {
    label: string;
    url?: string;
    handler?: string;
  };
}
```

### 9.2 Utility Types

```typescript
// Pagination
interface PaginationParams {
  page?: number;      // Default: 1
  limit?: number;     // Default: 20, Max: 100
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    totalPages: number;
    totalItems: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// Time Range
interface TimeRange {
  start: string;  // ISO datetime
  end: string;    // ISO datetime
}

// API Response Wrapper
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata: ResponseMetadata;
}

interface ApiError {
  code: string;
  message: string;
  field?: string;
  details?: any;
}

interface ResponseMetadata {
  timestamp: string;
  requestId: string;
  processingTime?: number;
  version?: string;
}
```

---

## 10. API Client Implementation

### 10.1 Base API Client

```typescript
// services/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/stores/authStore';

class ApiClient {
  private client: AxiosInstance;
  private refreshPromise: Promise<void> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = useAuthStore.getState().accessToken;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        // Add request ID
        config.headers['X-Request-ID'] = crypto.randomUUID();
        
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response.data,
      async (error) => {
        const originalRequest = error.config;

        // Handle token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          if (!this.refreshPromise) {
            this.refreshPromise = this.refreshToken();
          }

          await this.refreshPromise;
          this.refreshPromise = null;

          return this.client(originalRequest);
        }

        // Transform error to standard format
        const apiError: ApiError = {
          code: error.response?.data?.error?.code || 'UNKNOWN_ERROR',
          message: error.response?.data?.error?.message || 'An unexpected error occurred',
          details: error.response?.data?.error?.details,
        };

        return Promise.reject(apiError);
      }
    );
  }

  private async refreshToken(): Promise<void> {
    try {
      const refreshToken = useAuthStore.getState().refreshToken;
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await this.client.post<RefreshTokenResponse>('/auth/refresh', {
        refreshToken,
      });

      useAuthStore.getState().setTokens({
        accessToken: response.data.accessToken,
        refreshToken: response.data.refreshToken,
      });
    } catch (error) {
      useAuthStore.getState().logout();
      throw error;
    }
  }

  // Generic request methods
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.get(url, config);
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.client.post(url, data, config);
  }

  async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.client.patch(url, data, config);
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.delete(url, config);
  }
}

export const apiClient = new ApiClient();
```

### 10.2 API Service Layer

```typescript
// services/api/dashboard.ts
import { apiClient } from './client';

export const dashboardApi = {
  async getOverview(period: 'today' | 'week' | 'month' = 'today') {
    return apiClient.get<DashboardOverviewResponse>(
      `/dashboard/overview?period=${period}`
    );
  },

  async getMetrics() {
    return apiClient.get<DashboardMetrics>('/dashboard/metrics');
  },

  async exportData(format: 'csv' | 'json', period: TimeRange) {
    return apiClient.post('/dashboard/export', { format, period }, {
      responseType: 'blob',
    });
  },
};

// services/api/channels.ts
export const channelsApi = {
  async list() {
    return apiClient.get<ChannelListResponse>('/channels');
  },

  async create(data: CreateChannelRequest) {
    return apiClient.post<CreateChannelResponse>('/channels', data);
  },

  async update(id: string, data: UpdateChannelRequest) {
    return apiClient.patch<UpdateChannelResponse>(`/channels/${id}`, data);
  },

  async delete(id: string) {
    return apiClient.delete(`/channels/${id}`);
  },

  async toggleAutomation(id: string, enabled: boolean) {
    return apiClient.patch(`/channels/${id}/automation`, { enabled });
  },
};

// services/api/videos.ts
export const videosApi = {
  async getQueue() {
    return apiClient.get<VideoQueueResponse>('/videos/queue');
  },

  async generate(data: GenerateVideoRequest) {
    return apiClient.post<GenerateVideoResponse>('/videos/generate', data);
  },

  async getStatus(id: string) {
    return apiClient.get<ProcessingVideo>(`/videos/${id}/status`);
  },

  async retry(id: string) {
    return apiClient.post(`/videos/${id}/retry`);
  },

  async cancel(id: string) {
    return apiClient.post(`/videos/${id}/cancel`);
  },
};

// services/api/costs.ts
export const costsApi = {
  async getBreakdown(params: CostBreakdownRequest) {
    const query = new URLSearchParams(params as any).toString();
    return apiClient.get<CostBreakdownResponse>(`/costs/breakdown?${query}`);
  },

  async getAlerts() {
    return apiClient.get<CostAlertsResponse>('/costs/alerts');
  },

  async createAlert(data: CreateCostAlertRequest) {
    return apiClient.post('/costs/alerts', data);
  },

  async updateAlert(id: string, enabled: boolean) {
    return apiClient.patch(`/costs/alerts/${id}`, { enabled });
  },

  async deleteAlert(id: string) {
    return apiClient.delete(`/costs/alerts/${id}`);
  },
};
```

---

## Next Steps

1. Review all type definitions and ensure they match your component props
2. Implement the API client with proper error handling
3. Create TypeScript declaration files for global types
4. Set up API mocking for development (using MSW or similar)
5. Test all endpoints with the backend team
6. Document any deviations from these specifications

**Remember**: These specifications are the contract between frontend and backend. Any changes should be communicated and documented immediately.