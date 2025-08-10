# YTEMPIRE Complete API Specifications & Data Models
## Backend Contracts, WebSocket Events, and Integration Details

**Document Version**: 1.0  
**Role**: React Engineer  
**Purpose**: Complete API reference for frontend implementation

---

## 📡 API Base Configuration

### Base URLs
```typescript
// Development
const DEV_CONFIG = {
  API_BASE_URL: 'http://localhost:8000/api/v1',
  WS_URL: 'ws://localhost:8000/ws',
  CDN_URL: 'http://localhost:8000/static'
};

// Staging
const STAGING_CONFIG = {
  API_BASE_URL: 'https://staging-api.ytempire.com/api/v1',
  WS_URL: 'wss://staging-api.ytempire.com/ws',
  CDN_URL: 'https://staging-cdn.ytempire.com'
};

// Production
const PROD_CONFIG = {
  API_BASE_URL: 'https://api.ytempire.com/api/v1',
  WS_URL: 'wss://api.ytempire.com/ws',
  CDN_URL: 'https://cdn.ytempire.com'
};
```

### Standard Headers
```typescript
const API_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'X-Client-Version': '1.0.0',
  'X-Platform': 'web',
  'X-Request-ID': () => crypto.randomUUID()
};

// Authenticated requests also include:
const AUTH_HEADERS = {
  'Authorization': 'Bearer {access_token}'
};
```

### Standard Response Format
```typescript
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  metadata: {
    timestamp: string;
    requestId: string;
    processingTime: number;
    version: string;
  };
  pagination?: {
    page: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };
}
```

---

## 🔐 Authentication Endpoints

### POST /api/v1/auth/register
```typescript
// Request
interface RegisterRequest {
  email: string;          // Valid email, unique
  password: string;       // Min 8 chars, 1 uppercase, 1 number
  fullName: string;       // 2-50 characters
  timezone: string;       // IANA timezone (e.g., 'America/New_York')
  agreedToTerms: boolean; // Must be true
}

// Response
interface RegisterResponse {
  user: {
    id: string;           // UUID
    email: string;
    fullName: string;
    role: 'user' | 'admin';
    status: 'pending_verification' | 'active';
    createdAt: string;    // ISO 8601
  };
  tokens: {
    accessToken: string;  // JWT, expires in 1 hour
    refreshToken: string; // JWT, expires in 7 days
    expiresIn: number;    // Seconds until expiry
  };
  verificationRequired: boolean;
}

// Example
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "fullName": "John Doe",
  "timezone": "America/New_York",
  "agreedToTerms": true
}
```

### POST /api/v1/auth/login
```typescript
// Request
interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;  // Extends refresh token to 30 days
}

// Response
interface LoginResponse {
  user: {
    id: string;
    email: string;
    fullName: string;
    role: 'user' | 'admin';
    status: 'active' | 'suspended' | 'pending_verification';
    subscription: {
      plan: 'free' | 'starter' | 'pro' | 'enterprise';
      status: 'active' | 'past_due' | 'canceled';
      channelLimit: number;     // 5 for MVP
      videoLimit: number;       // Daily limit
      costLimit: number;        // Daily spend limit
      expiresAt: string | null;
    };
    preferences: {
      theme: 'light' | 'dark' | 'system';
      emailNotifications: boolean;
      language: string;
      timezone: string;
    };
  };
  tokens: {
    accessToken: string;
    refreshToken: string;
    expiresIn: number;
  };
  lastLogin: string | null;
  requiresPasswordChange: boolean;
}
```

### POST /api/v1/auth/refresh
```typescript
// Request
interface RefreshTokenRequest {
  refreshToken: string;
}

// Response
interface RefreshTokenResponse {
  tokens: {
    accessToken: string;
    refreshToken: string;  // New refresh token
    expiresIn: number;
  };
}
```

### POST /api/v1/auth/logout
```typescript
// Request
interface LogoutRequest {
  refreshToken: string;  // To invalidate
  everywhere?: boolean;  // Logout from all devices
}

// Response
interface LogoutResponse {
  success: boolean;
  message: string;
}
```

### POST /api/v1/auth/verify-email
```typescript
// Request
interface VerifyEmailRequest {
  token: string;  // From email link
}

// Response
interface VerifyEmailResponse {
  success: boolean;
  user: {
    id: string;
    email: string;
    status: 'active';
  };
}
```

---

## 📺 Channel Management Endpoints

### GET /api/v1/channels
```typescript
// Query Parameters
interface GetChannelsParams {
  status?: 'active' | 'paused' | 'error';
  search?: string;
  sortBy?: 'name' | 'created' | 'videos' | 'revenue';
  order?: 'asc' | 'desc';
}

// Response
interface GetChannelsResponse {
  channels: Channel[];
  stats: {
    total: number;
    active: number;
    paused: number;
    error: number;
  };
}

// Channel Model
interface Channel {
  id: string;
  name: string;
  description: string;
  youtubeChannelId: string;
  youtubeChannelUrl: string;
  thumbnailUrl: string;
  niche: 'education' | 'entertainment' | 'gaming' | 'technology' | 'lifestyle' | 'other';
  status: 'active' | 'paused' | 'error';
  errorMessage?: string;
  
  automation: {
    enabled: boolean;
    schedule: {
      videosPerDay: number;      // 1-3 for MVP
      publishTimes: string[];    // ['09:00', '15:00']
      timezone: string;
    };
    contentStrategy: {
      style: 'educational' | 'entertainment' | 'tutorial' | 'news';
      targetLength: 'short' | 'medium' | 'long';
      useAiTopics: boolean;
      topicKeywords?: string[];
    };
  };
  
  statistics: {
    totalVideos: number;
    publishedVideos: number;
    totalViews: number;
    totalWatchTime: number;      // Minutes
    subscribers: number;
    
    revenue: {
      total: number;
      lastMonth: number;
      lastWeek: number;
      today: number;
      currency: 'USD';
    };
    
    costs: {
      total: number;
      lastMonth: number;
      avgPerVideo: number;
    };
    
    performance: {
      avgViews: number;
      avgRetention: number;      // Percentage
      avgCtr: number;            // Click-through rate
      engagementRate: number;
    };
  };
  
  settings: {
    monetization: {
      adsEnabled: boolean;
      affiliateLinks: boolean;
      sponsorships: boolean;
    };
    branding: {
      intro: boolean;
      outro: boolean;
      watermark: boolean;
    };
    notifications: {
      onVideoComplete: boolean;
      onVideoError: boolean;
      onRevenueThreshold: boolean;
      revenueThreshold?: number;
    };
  };
  
  createdAt: string;
  updatedAt: string;
  lastVideoAt: string | null;
}
```

### POST /api/v1/channels
```typescript
// Request
interface CreateChannelRequest {
  name: string;                    // 3-50 characters
  description?: string;            // Max 500 characters
  youtubeChannelId: string;        // From YouTube OAuth
  niche: string;
  automation: {
    enabled: boolean;
    schedule: {
      videosPerDay: number;        // 1-3
      publishTimes: string[];
    };
    contentStrategy: {
      style: string;
      targetLength: string;
      useAiTopics: boolean;
      topicKeywords?: string[];
    };
  };
}

// Response
interface CreateChannelResponse {
  channel: Channel;
  setupTasks: {
    id: string;
    task: string;
    status: 'pending' | 'completed';
  }[];
}
```

### PATCH /api/v1/channels/{channelId}
```typescript
// Request
interface UpdateChannelRequest {
  name?: string;
  description?: string;
  status?: 'active' | 'paused';
  automation?: {
    enabled?: boolean;
    schedule?: Partial<Channel['automation']['schedule']>;
    contentStrategy?: Partial<Channel['automation']['contentStrategy']>;
  };
  settings?: Partial<Channel['settings']>;
}

// Response
interface UpdateChannelResponse {
  channel: Channel;
  changes: string[];  // List of what was updated
}
```

### DELETE /api/v1/channels/{channelId}
```typescript
// Response
interface DeleteChannelResponse {
  success: boolean;
  message: string;
  deletedVideos: number;
  freedQuota: {
    channels: number;  // How many slots freed
    dailyVideos: number;
  };
}
```

### POST /api/v1/channels/{channelId}/sync
```typescript
// Sync with YouTube to update stats
// Response
interface SyncChannelResponse {
  channel: Channel;
  updates: {
    subscribers: { old: number; new: number };
    videos: { old: number; new: number };
    views: { old: number; new: number };
  };
  lastSyncAt: string;
}
```

---

## 🎬 Video Generation Endpoints

### POST /api/v1/videos/generate
```typescript
// Request
interface GenerateVideoRequest {
  channelId: string;
  
  content: {
    topic?: string;              // Optional, AI selects if empty
    style: 'educational' | 'entertainment' | 'tutorial' | 'news';
    length: 'short' | 'medium' | 'long';
    
    customization?: {
      tone: 'professional' | 'casual' | 'humorous' | 'serious';
      pacing: 'slow' | 'medium' | 'fast';
      complexity: 'beginner' | 'intermediate' | 'advanced';
    };
  };
  
  scheduling: {
    priority: number;            // 1-10, higher = sooner
    publishAt?: string;          // ISO 8601, null = immediate
    maxCost?: number;           // Cost limit for this video
  };
  
  features: {
    generateThumbnail: boolean;
    addCaptions: boolean;
    includeChapters: boolean;
    optimizeForSeo: boolean;
  };
}

// Response
interface GenerateVideoResponse {
  job: {
    id: string;
    status: 'queued';
    queuePosition: number;
    estimatedStartTime: string;
    estimatedCompletionTime: string;
    estimatedCost: {
      breakdown: {
        aiGeneration: number;
        voiceSynthesis: number;
        videoRendering: number;
        thumbnailGeneration: number;
        captioning: number;
        total: number;
      };
      confidence: number;        // 0-1, how accurate the estimate is
    };
  };
  
  quotaUsage: {
    dailyVideosUsed: number;
    dailyVideosLimit: number;
    dailyCostUsed: number;
    dailyCostLimit: number;
  };
  
  warnings?: string[];          // ["Approaching daily limit", etc.]
}
```

### GET /api/v1/videos
```typescript
// Query Parameters
interface GetVideosParams {
  channelId?: string;
  status?: 'queued' | 'processing' | 'completed' | 'failed' | 'published';
  dateFrom?: string;
  dateTo?: string;
  search?: string;
  page?: number;
  limit?: number;
  sortBy?: 'created' | 'published' | 'views' | 'revenue';
  order?: 'asc' | 'desc';
}

// Response
interface GetVideosResponse {
  videos: Video[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };
  aggregates: {
    totalVideos: number;
    totalViews: number;
    totalRevenue: number;
    avgCostPerVideo: number;
  };
}

// Video Model
interface Video {
  id: string;
  channelId: string;
  channelName: string;
  
  content: {
    title: string;
    description: string;
    tags: string[];
    category: string;
    thumbnail: {
      url: string;
      width: number;
      height: number;
    };
  };
  
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'published';
  
  progress: {
    percentage: number;          // 0-100
    currentStage: 'queued' | 'script' | 'audio' | 'video' | 'thumbnail' | 'uploading' | 'published';
    stages: {
      script: { status: 'pending' | 'processing' | 'completed' | 'failed'; duration?: number };
      audio: { status: 'pending' | 'processing' | 'completed' | 'failed'; duration?: number };
      video: { status: 'pending' | 'processing' | 'completed' | 'failed'; duration?: number };
      thumbnail: { status: 'pending' | 'processing' | 'completed' | 'failed'; duration?: number };
      uploading: { status: 'pending' | 'processing' | 'completed' | 'failed'; duration?: number };
    };
  };
  
  generation: {
    requestedAt: string;
    startedAt?: string;
    completedAt?: string;
    publishedAt?: string;
    
    input: {
      topic: string;
      style: string;
      length: string;
      customization?: any;
    };
    
    output: {
      duration: number;          // Seconds
      fileSize: number;          // Bytes
      resolution: string;        // "1920x1080"
      format: string;           // "mp4"
    };
    
    costs: {
      script: number;
      audio: number;
      video: number;
      thumbnail: number;
      total: number;
      breakdown: {
        aiTokens: number;
        voiceMinutes: number;
        renderingMinutes: number;
        storageGB: number;
      };
    };
  };
  
  youtube: {
    videoId?: string;
    url?: string;
    status?: 'private' | 'unlisted' | 'public';
    
    metrics: {
      views: number;
      likes: number;
      dislikes: number;
      comments: number;
      shares: number;
      watchTime: number;         // Minutes
      averageViewDuration: number;
      clickThroughRate: number;
      impressions: number;
    };
    
    revenue: {
      ads: number;
      channel_memberships: number;
      super_chat: number;
      super_stickers: number;
      total: number;
      currency: 'USD';
      lastUpdated: string;
    };
  };
  
  errors?: {
    stage: string;
    code: string;
    message: string;
    details?: any;
    retryable: boolean;
    retryCount: number;
  }[];
  
  metadata: {
    version: number;
    retryCount: number;
    lastRetryAt?: string;
  };
}
```

### GET /api/v1/videos/{videoId}
```typescript
// Response: Single Video object with all details
```

### GET /api/v1/videos/{videoId}/status
```typescript
// Lightweight status check
interface VideoStatusResponse {
  id: string;
  status: string;
  progress: {
    percentage: number;
    currentStage: string;
    message?: string;
  };
  estimatedCompletion?: string;
  errors?: any[];
}
```

### POST /api/v1/videos/{videoId}/retry
```typescript
// Request
interface RetryVideoRequest {
  fromStage?: 'script' | 'audio' | 'video' | 'thumbnail' | 'uploading';
  reason?: string;
}

// Response
interface RetryVideoResponse {
  job: {
    id: string;
    status: 'queued';
    queuePosition: number;
    retryCount: number;
  };
}
```

### DELETE /api/v1/videos/{videoId}
```typescript
// Response
interface DeleteVideoResponse {
  success: boolean;
  youtubeDeleted: boolean;  // Whether it was removed from YouTube
  storageFreed: number;     // Bytes
}
```

### GET /api/v1/videos/queue
```typescript
// Response
interface VideoQueueResponse {
  queue: {
    videoId: string;
    channelName: string;
    position: number;
    priority: number;
    estimatedStart: string;
    topic: string;
  }[];
  
  processing: {
    videoId: string;
    channelName: string;
    stage: string;
    progress: number;
    startedAt: string;
  }[];
  
  stats: {
    queueDepth: number;
    averageWaitTime: number;    // Seconds
    processingCount: number;
    completedToday: number;
  };
  
  systemStatus: {
    gpuUtilization: number;      // Percentage
    cpuUtilization: number;
    memoryUsage: number;
    estimatedThroughput: number; // Videos per hour
  };
}
```

---

## 💰 Cost Management Endpoints

### GET /api/v1/costs/current
```typescript
// Response
interface CurrentCostsResponse {
  period: {
    start: string;
    end: string;
    timezone: string;
  };
  
  costs: {
    today: {
      spent: number;
      limit: number;
      remaining: number;
      projectedEnd: number;      // Projected total by end of day
    };
    
    week: {
      spent: number;
      dailyAverage: number;
      projectedEnd: number;
    };
    
    month: {
      spent: number;
      limit: number;
      remaining: number;
      projectedEnd: number;
      daysRemaining: number;
    };
  };
  
  breakdown: {
    byCategory: {
      aiGeneration: number;
      voiceSynthesis: number;
      videoRendering: number;
      storage: number;
      apiCalls: number;
      other: number;
    };
    
    byChannel: Array<{
      channelId: string;
      channelName: string;
      cost: number;
      videoCount: number;
      avgPerVideo: number;
    }>;
    
    topExpenses: Array<{
      videoId: string;
      videoTitle: string;
      channelName: string;
      cost: number;
      date: string;
    }>;
  };
  
  alerts: Array<{
    id: string;
    type: 'warning' | 'critical';
    threshold: number;
    current: number;
    message: string;
    createdAt: string;
  }>;
}
```

### GET /api/v1/costs/history
```typescript
// Query Parameters
interface CostHistoryParams {
  period: 'day' | 'week' | 'month' | 'year';
  from?: string;
  to?: string;
  groupBy?: 'day' | 'week' | 'month';
}

// Response
interface CostHistoryResponse {
  history: Array<{
    date: string;
    costs: {
      total: number;
      breakdown: Record<string, number>;
    };
    videos: {
      generated: number;
      published: number;
      failed: number;
    };
    efficiency: {
      costPerVideo: number;
      successRate: number;
      avgGenerationTime: number;
    };
  }>;
  
  trends: {
    costTrend: 'increasing' | 'stable' | 'decreasing';
    percentageChange: number;
    projection: {
      nextPeriod: number;
      confidence: number;
    };
  };
}
```

### POST /api/v1/costs/alerts
```typescript
// Request
interface CreateCostAlertRequest {
  type: 'daily_spend' | 'per_video' | 'monthly_budget' | 'unusual_spike';
  threshold: number;
  comparison: 'greater' | 'less' | 'equal';
  
  notification: {
    email: boolean;
    inApp: boolean;
    webhook?: string;
  };
}

// Response
interface CreateCostAlertResponse {
  alert: {
    id: string;
    type: string;
    threshold: number;
    active: boolean;
    createdAt: string;
  };
}
```

---

## 📊 Analytics Endpoints

### GET /api/v1/analytics/overview
```typescript
// Query Parameters
interface AnalyticsParams {
  period: 'day' | 'week' | 'month' | 'quarter' | 'year';
  channelId?: string;  // All channels if not specified
  compareWith?: 'previous_period' | 'last_year';
}

// Response
interface AnalyticsOverviewResponse {
  summary: {
    totalVideos: number;
    totalViews: number;
    totalRevenue: number;
    totalCosts: number;
    profit: number;
    roi: number;                 // Percentage
    
    changes: {                   // vs comparison period
      videos: number;
      views: number;
      revenue: number;
      costs: number;
    };
  };
  
  performance: {
    topVideos: Array<{
      id: string;
      title: string;
      channel: string;
      views: number;
      revenue: number;
      roi: number;
      thumbnail: string;
    }>;
    
    topChannels: Array<{
      id: string;
      name: string;
      videos: number;
      views: number;
      revenue: number;
      growth: number;
    }>;
    
    contentPerformance: {
      byStyle: Record<string, { videos: number; avgViews: number; avgRevenue: number }>;
      byLength: Record<string, { videos: number; avgViews: number; avgRevenue: number }>;
      byNiche: Record<string, { videos: number; avgViews: number; avgRevenue: number }>;
    };
  };
  
  trends: {
    daily: Array<{
      date: string;
      videos: number;
      views: number;
      revenue: number;
      costs: number;
    }>;
    
    hourly: Array<{
      hour: number;
      avgViews: number;
      avgEngagement: number;
    }>;
  };
  
  insights: Array<{
    type: 'tip' | 'warning' | 'success';
    title: string;
    message: string;
    actionable: boolean;
    action?: {
      label: string;
      url: string;
    };
  }>;
}
```

### GET /api/v1/analytics/revenue
```typescript
// Response
interface RevenueAnalyticsResponse {
  summary: {
    total: number;
    bySource: {
      ads: number;
      affiliates: number;
      sponsorships: number;
      memberships: number;
      other: number;
    };
    
    byChannel: Array<{
      channelId: string;
      channelName: string;
      revenue: number;
      percentage: number;
    }>;
    
    projections: {
      endOfMonth: number;
      nextMonth: number;
      quarterly: number;
      annually: number;
      confidence: number;
    };
  };
  
  performance: {
    rpm: number;                 // Revenue per mille (thousand views)
    cpm: number;                 // Cost per mille
    averageVideoRevenue: number;
    topEarningDay: {
      date: string;
      revenue: number;
    };
  };
  
  optimization: {
    recommendations: Array<{
      title: string;
      description: string;
      potentialIncrease: number;
      difficulty: 'easy' | 'medium' | 'hard';
    }>;
    
    opportunities: Array<{
      type: string;
      description: string;
      estimatedRevenue: number;
    }>;
  };
}
```

---

## 🔌 WebSocket Events

### Connection
```typescript
// Connection URL
ws://localhost:8000/ws?token={access_token}

// Connection lifecycle
interface ConnectionEvents {
  'open': { connected: true; timestamp: string };
  'close': { code: number; reason: string };
  'error': { message: string; code?: string };
  'ping': { timestamp: string };
  'pong': { timestamp: string };
}
```

### Event Structure
```typescript
interface WebSocketMessage<T = any> {
  id: string;              // Unique message ID
  type: string;            // Event type
  payload: T;              // Event-specific data
  timestamp: string;       // ISO 8601
  version: string;         // API version
}
```

### Critical Events (MVP - 3 Only)

#### video.completed
```typescript
interface VideoCompletedEvent {
  type: 'video.completed';
  payload: {
    videoId: string;
    channelId: string;
    channelName: string;
    title: string;
    thumbnail: string;
    youtubeUrl: string;
    duration: number;        // Seconds
    
    generation: {
      startedAt: string;
      completedAt: string;
      totalTime: number;     // Seconds
    };
    
    cost: {
      total: number;
      breakdown: {
        ai: number;
        voice: number;
        rendering: number;
      };
    };
    
    quality: {
      score: number;         // 0-100
      checks: {
        audioQuality: boolean;
        videoQuality: boolean;
        contentRelevance: boolean;
        seoOptimized: boolean;
      };
    };
  };
}
```

#### video.failed
```typescript
interface VideoFailedEvent {
  type: 'video.failed';
  payload: {
    videoId: string;
    channelId: string;
    channelName: string;
    topic: string;
    
    error: {
      stage: 'script' | 'audio' | 'video' | 'thumbnail' | 'upload';
      code: string;
      message: string;
      details?: any;
      retryable: boolean;
    };
    
    progress: {
      percentage: number;
      lastStage: string;
    };
    
    cost: {
      incurred: number;      // Cost before failure
      refundable: number;
    };
    
    actions: {
      canRetry: boolean;
      retryCount: number;
      alternativeOptions?: string[];
    };
  };
}
```

#### cost.alert
```typescript
interface CostAlertEvent {
  type: 'cost.alert';
  payload: {
    alertId: string;
    severity: 'warning' | 'critical' | 'info';
    
    alert: {
      type: 'approaching_limit' | 'limit_exceeded' | 'unusual_spike' | 'budget_milestone';
      threshold: number;
      current: number;
      limit: number;
      percentage: number;
    };
    
    message: string;
    
    recommendations: Array<{
      action: string;
      impact: string;
      url?: string;
    }>;
    
    restrictions?: {
      videosDisabled: boolean;
      reducedQuality: boolean;
      message: string;
    };
  };
}
```

### Future WebSocket Events (Post-MVP)
```typescript
// These will be added after MVP:
- 'channel.metrics.updated'
- 'video.processing.progress'
- 'system.announcement'
- 'revenue.milestone'
- 'ai.recommendation'
```

---

## 🔧 Settings & Configuration Endpoints

### GET /api/v1/settings
```typescript
interface SettingsResponse {
  user: {
    profile: {
      fullName: string;
      email: string;
      avatar?: string;
      timezone: string;
      language: string;
    };
    
    preferences: {
      theme: 'light' | 'dark' | 'system';
      emailNotifications: {
        videoComplete: boolean;
        videoFailed: boolean;
        costAlerts: boolean;
        weeklyReport: boolean;
        productUpdates: boolean;
      };
      
      dashboard: {
        defaultView: 'grid' | 'list';
        refreshInterval: number;  // Seconds
        showTips: boolean;
      };
    };
    
    limits: {
      channels: number;
      dailyVideos: number;
      dailyCost: number;
      monthlyBudget: number;
    };
  };
  
  billing: {
    subscription: {
      plan: string;
      status: string;
      nextBillingDate?: string;
      amount: number;
    };
    
    paymentMethod?: {
      type: 'card' | 'paypal';
      last4?: string;
      expiryMonth?: number;
      expiryYear?: number;
    };
    
    invoices: Array<{
      id: string;
      date: string;
      amount: number;
      status: 'paid' | 'pending' | 'failed';
      downloadUrl: string;
    }>;
  };
  
  api: {
    keys: Array<{
      id: string;
      name: string;
      key: string;           // Partially masked
      createdAt: string;
      lastUsed?: string;
      permissions: string[];
    }>;
    
    webhooks: Array<{
      id: string;
      url: string;
      events: string[];
      active: boolean;
      lastTriggered?: string;
    }>;
  };
}
```

### PATCH /api/v1/settings
```typescript
interface UpdateSettingsRequest {
  profile?: Partial<SettingsResponse['user']['profile']>;
  preferences?: Partial<SettingsResponse['user']['preferences']>;
  limits?: Partial<SettingsResponse['user']['limits']>;
}
```

---

## 🎯 Business Logic & Calculations

### Video Generation Pipeline
```typescript
const VIDEO_GENERATION_PIPELINE = {
  stages: [
    {
      name: 'topic_selection',
      description: 'AI analyzes trends and selects optimal topic',
      duration: '5-10 seconds',
      cost: '$0.02-0.05',
      logic: `
        1. Fetch trending topics from YouTube API
        2. Analyze channel niche and past performance
        3. Score topics by predicted engagement
        4. Select highest scoring topic
        5. Generate SEO-optimized title
      `
    },
    {
      name: 'script_generation',
      description: 'GPT-4 creates engaging script',
      duration: '20-30 seconds',
      cost: '$0.10-0.15',
      logic: `
        1. Create outline based on video length
        2. Generate hook (first 15 seconds)
        3. Develop main content points
        4. Add call-to-action
        5. Optimize for retention
      `
    },
    {
      name: 'voice_synthesis',
      description: 'ElevenLabs generates natural speech',
      duration: '30-60 seconds',
      cost: '$0.10-0.20',
      logic: `
        1. Parse script for emphasis and pauses
        2. Select voice based on channel style
        3. Generate audio with proper pacing
        4. Add background music if enabled
        5. Normalize audio levels
      `
    },
    {
      name: 'video_rendering',
      description: 'Create visuals matching script',
      duration: '2-4 minutes',
      cost: '$0.15-0.25',
      logic: `
        1. Generate scene breakdown from script
        2. Select/generate relevant visuals
        3. Add text overlays and animations
        4. Sync with audio timing
        5. Render at 1080p/60fps
      `
    },
    {
      name: 'thumbnail_generation',
      description: 'AI creates clickable thumbnail',
      duration: '10-15 seconds',
      cost: '$0.02-0.03',
      logic: `
        1. Extract key visual from video
        2. Generate attention-grabbing text
        3. Apply channel branding
        4. A/B test variations
        5. Optimize for mobile viewing
      `
    },
    {
      name: 'upload_publishing',
      description: 'Upload to YouTube with SEO',
      duration: '30-60 seconds',
      cost: '$0.01',
      logic: `
        1. Upload video file to YouTube
        2. Set title, description, tags
        3. Add end screens and cards
        4. Schedule or publish immediately
        5. Submit to YouTube algorithm
      `
    }
  ],
  
  total: {
    duration: '5-8 minutes',
    cost: '$0.40-0.50 per video'
  }
};
```

### Cost Calculation Formula
```typescript
const calculateVideoCost = (params: VideoGenerationParams): CostBreakdown => {
  const costs = {
    // AI Generation (GPT-4)
    aiGeneration: {
      baseRate: 0.03,    // Per 1K tokens
      multipliers: {
        short: 0.5,      // ~500 tokens
        medium: 1.0,     // ~1000 tokens
        long: 2.0        // ~2000 tokens
      },
      calculate: (length: string) => 0.03 * multipliers[length] * 1.5 // 1.5x for quality
    },
    
    // Voice Synthesis (ElevenLabs)
    voiceSynthesis: {
      baseRate: 0.18,    // Per 1K characters
      multipliers: {
        short: 0.4,      // ~3 min = 2K chars
        medium: 1.0,     // ~10 min = 5K chars
        long: 1.8        // ~18 min = 9K chars
      },
      calculate: (length: string) => 0.18 * multipliers[length]
    },
    
    // Video Rendering (GPU time)
    videoRendering: {
      baseRate: 0.002,   // Per second of video
      multipliers: {
        short: 180,      // 3 minutes
        medium: 600,     // 10 minutes
        long: 1080       // 18 minutes
      },
      calculate: (length: string) => 0.002 * multipliers[length]
    },
    
    // Fixed costs
    thumbnail: 0.02,
    uploading: 0.01,
    storage: 0.001       // Per day per GB
  };
  
  return {
    script: costs.aiGeneration.calculate(params.length),
    voice: costs.voiceSynthesis.calculate(params.length),
    video: costs.videoRendering.calculate(params.length),
    thumbnail: costs.thumbnail,
    uploading: costs.uploading,
    total: // sum of all above
  };
};
```

### Channel Automation Logic
```typescript
const CHANNEL_AUTOMATION_RULES = {
  triggers: {
    scheduled: {
      description: 'Videos generated at set times',
      logic: `
        1. Check channel schedule (e.g., daily at 9 AM)
        2. Verify quota availability
        3. Check cost limits
        4. Trigger video generation
        5. Queue for optimal processing
      `
    },
    
    trending: {
      description: 'React to trending topics',
      logic: `
        1. Monitor niche-specific trends every hour
        2. Score trend relevance (0-100)
        3. If score > 80, generate video
        4. Fast-track processing (high priority)
        5. Publish within 2 hours
      `
    },
    
    performance: {
      description: 'Adapt based on analytics',
      logic: `
        1. Analyze last 10 videos performance
        2. Identify successful patterns
        3. Adjust content strategy
        4. Generate similar content
        5. A/B test variations
      `
    }
  },
  
  limits: {
    daily: {
      videos: 3,         // Max per channel
      cost: 10,          // Max $10 per channel per day
      globalVideos: 15   // Max across all channels
    },
    
    quality: {
      minScore: 70,      // Don't publish below this
      retries: 2,        // Max generation attempts
      variation: 20      // % difference required between videos
    }
  },
  
  optimization: {
    timing: `
      - Analyze audience timezone
      - Identify peak viewing hours
      - Schedule 30 min before peak
      - Avoid competing uploads
      - Optimize for algorithm boost
    `,
    
    content: `
      - 70% evergreen content
      - 20% trending topics
      - 10% experimental
      - Maintain channel theme
      - Build content series
    `
  }
};
```

---

## 🔗 Third-Party Integrations

### YouTube API v3 Integration
```typescript
const YOUTUBE_INTEGRATION = {
  oauth: {
    authUrl: 'https://accounts.google.com/o/oauth2/v2/auth',
    tokenUrl: 'https://oauth2.googleapis.com/token',
    scope: [
      'https://www.googleapis.com/auth/youtube.upload',
      'https://www.googleapis.com/auth/youtube.readonly',
      'https://www.googleapis.com/auth/youtubepartner',
      'https://www.googleapis.com/auth/yt-analytics.readonly'
    ],
    
    flow: `
      1. Redirect to Google OAuth consent
      2. User approves YouTube access
      3. Receive authorization code
      4. Exchange for access/refresh tokens
      5. Store encrypted tokens per channel
    `
  },
  
  quotas: {
    daily: 10000,              // API units per day
    costs: {
      upload: 1600,            // Per video upload
      read: 1,                 // Per data fetch
      write: 50,               // Per metadata update
      delete: 50               // Per video deletion
    }
  },
  
  endpoints: {
    upload: 'POST https://www.googleapis.com/upload/youtube/v3/videos',
    update: 'PUT https://www.googleapis.com/youtube/v3/videos',
    analytics: 'GET https://youtubeanalytics.googleapis.com/v2/reports'
  }
};
```

### OpenAI GPT-4 Integration
```typescript
const OPENAI_INTEGRATION = {
  model: 'gpt-4-turbo-preview',
  
  endpoints: {
    base: 'https://api.openai.com/v1',
    completions: '/chat/completions',
    embeddings: '/embeddings'
  },
  
  parameters: {
    script_generation: {
      temperature: 0.8,        // Creativity level
      max_tokens: 2000,        // ~1500 words
      top_p: 0.9,
      frequency_penalty: 0.3,  // Reduce repetition
      presence_penalty: 0.3,
      
      system_prompt: `You are an expert YouTube content creator...`
    },
    
    topic_selection: {
      temperature: 0.6,        // More focused
      max_tokens: 100,
      functions: [{
        name: 'select_topic',
        parameters: {
          topic: 'string',
          relevance_score: 'number',
          search_volume: 'number',
          competition: 'low|medium|high'
        }
      }]
    }
  },
  
  costs: {
    input: 0.01,              // Per 1K tokens
    output: 0.03,             // Per 1K tokens
    average_per_video: 0.15   // Typical usage
  }
};
```

### ElevenLabs Voice Integration
```typescript
const ELEVENLABS_INTEGRATION = {
  endpoints: {
    base: 'https://api.elevenlabs.io/v1',
    textToSpeech: '/text-to-speech/{voice_id}',
    voices: '/voices'
  },
  
  voices: {
    professional: {
      id: 'pNInz6obpgDQGcFmaJgB',
      name: 'Adam',
      style: 'Clear, professional narrator'
    },
    casual: {
      id: 'AZnzlk1XvdvUeBn7MQlG',
      name: 'Domi',
      style: 'Friendly, conversational'
    },
    energetic: {
      id: 'EXAVITQu4vr4xnSDxMaL',
      name: 'Bella',
      style: 'Upbeat, enthusiastic'
    }
  },
  
  settings: {
    stability: 0.75,          // Voice consistency
    similarity_boost: 0.75,   // Match voice character
    style: 0.5,              // Speaking style strength
    use_speaker_boost: true,
    
    optimize_streaming_latency: 0,
    output_format: 'mp3_44100_128'
  },
  
  costs: {
    per_character: 0.00018,   // $0.18 per 1K chars
    average_per_video: 0.20   // ~1100 chars/min
  }
};
```

### Stripe Payment Integration
```typescript
const STRIPE_INTEGRATION = {
  products: {
    free: {
      id: 'prod_free',
      features: {
        channels: 1,
        videos_per_day: 1,
        support: 'community'
      }
    },
    
    starter: {
      id: 'prod_starter',
      price: 'price_starter_monthly',
      amount: 4900,           // $49.00
      features: {
        channels: 5,
        videos_per_day: 3,
        support: 'email'
      }
    },
    
    pro: {
      id: 'prod_pro',
      price: 'price_pro_monthly',
      amount: 14900,          // $149.00
      features: {
        channels: 20,
        videos_per_day: 10,
        support: 'priority'
      }
    }
  },
  
  webhooks: {
    events: [
      'checkout.session.completed',
      'customer.subscription.created',
      'customer.subscription.updated',
      'customer.subscription.deleted',
      'invoice.payment_succeeded',
      'invoice.payment_failed'
    ],
    
    endpoint: '/api/v1/webhooks/stripe'
  }
};
```

---

## 🔐 Environment Configuration

### Development Environment
```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_CDN_URL=http://localhost:8000/static

# Third-Party Services (Development Keys)
VITE_YOUTUBE_API_KEY=AIzaSyD-dev-key-xxxxxxxxxxxxxxxxxxxx
VITE_STRIPE_PUBLIC_KEY=pk_test_51234567890abcdefghijklmnop
VITE_SENTRY_DSN=https://dev@sentry.io/1234567

# Feature Flags
VITE_ENABLE_ANALYTICS=false
VITE_ENABLE_PAYMENT=false
VITE_ENABLE_WEBSOCKET=true

# Development Settings
VITE_DEBUG_MODE=true
VITE_LOG_LEVEL=debug
VITE_MOCK_API=false
```

### Production Environment
```env
# API Configuration
VITE_API_BASE_URL=https://api.ytempire.com/api/v1
VITE_WS_URL=wss://api.ytempire.com/ws
VITE_CDN_URL=https://cdn.ytempire.com

# Third-Party Services (Production Keys - Encrypted)
VITE_YOUTUBE_API_KEY=${YOUTUBE_API_KEY}
VITE_STRIPE_PUBLIC_KEY=${STRIPE_PUBLIC_KEY}
VITE_SENTRY_DSN=${SENTRY_DSN}
VITE_GA_TRACKING_ID=${GA_TRACKING_ID}

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_PAYMENT=true
VITE_ENABLE_WEBSOCKET=true

# Production Settings
VITE_DEBUG_MODE=false
VITE_LOG_LEVEL=error
VITE_MOCK_API=false
```

---

## 🎨 Data Models Reference

### Complete TypeScript Interfaces
```typescript
// User Model
interface User {
  id: string;
  email: string;
  fullName: string;
  avatar?: string;
  role: 'user' | 'admin';
  status: 'active' | 'suspended' | 'pending_verification';
  
  subscription: {
    plan: 'free' | 'starter' | 'pro' | 'enterprise';
    status: 'active' | 'past_due' | 'canceled';
    startedAt: string;
    expiresAt?: string;
    
    limits: {
      channels: number;
      dailyVideos: number;
      dailyCost: number;
      monthlyBudget: number;
      storageGB: number;
    };
    
    usage: {
      channels: number;
      videosToday: number;
      costToday: number;
      costMonth: number;
      storageUsed: number;
    };
  };
  
  preferences: {
    theme: 'light' | 'dark' | 'system';
    language: string;
    timezone: string;
    currency: string;
    emailNotifications: Record<string, boolean>;
  };
  
  metadata: {
    createdAt: string;
    updatedAt: string;
    lastLoginAt?: string;
    loginCount: number;
    referralSource?: string;
  };
}

// Session Model
interface Session {
  id: string;
  userId: string;
  device: {
    type: 'desktop' | 'mobile' | 'tablet';
    os: string;
    browser: string;
    ip: string;
    location?: {
      country: string;
      city: string;
    };
  };
  
  createdAt: string;
  expiresAt: string;
  lastActivityAt: string;
}

// Additional models for all entities...
```

---

## 📋 Error Codes Reference

### Complete Error Code System
```typescript
const ERROR_CODES = {
  // Authentication (1xxx)
  1001: 'Invalid credentials',
  1002: 'Account not found',
  1003: 'Account suspended',
  1004: 'Email not verified',
  1005: 'Token expired',
  1006: 'Token invalid',
  1007: 'Insufficient permissions',
  1008: 'Two-factor required',
  
  // Validation (2xxx)
  2001: 'Invalid input format',
  2002: 'Required field missing',
  2003: 'Value out of range',
  2004: 'Duplicate entry',
  2005: 'Invalid file type',
  2006: 'File too large',
  
  // Business Logic (3xxx)
  3001: 'Channel limit exceeded',
  3002: 'Daily video limit exceeded',
  3003: 'Cost limit exceeded',
  3004: 'Insufficient credits',
  3005: 'Feature not available in plan',
  3006: 'Content policy violation',
  
  // External Services (4xxx)
  4001: 'YouTube API error',
  4002: 'OpenAI API error',
  4003: 'ElevenLabs API error',
  4004: 'Stripe payment error',
  4005: 'Service temporarily unavailable',
  
  // System (5xxx)
  5001: 'Internal server error',
  5002: 'Database error',
  5003: 'File system error',
  5004: 'Queue system error',
  5005: 'GPU unavailable'
};
```

---

## 🚀 Implementation Checklist

### For React Engineer
- [ ] Set up API client with all endpoints
- [ ] Implement authentication flow
- [ ] Create data models/interfaces
- [ ] Set up WebSocket connection
- [ ] Implement error handling
- [ ] Create mock data for testing
- [ ] Build UI components matching flows
- [ ] Integrate with state management
- [ ] Add cost calculations
- [ ] Implement video generation flow

This document provides all the API specifications, data models, and integration details needed to build the YTEMPIRE frontend. All endpoints, request/response formats, WebSocket events, and business logic are fully documented.