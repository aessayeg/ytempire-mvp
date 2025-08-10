# YTEMPIRE Documentation - API & Integrations

## 5.1 API Specifications

### Base Configuration

```typescript
// API Base Configuration
const API_CONFIG = {
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Version': '1.0'
  }
};
```

### Core API Endpoints

#### Authentication Endpoints

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

// POST /api/v1/auth/refresh
interface RefreshTokenRequest {
  refreshToken: string;
}

interface RefreshTokenResponse {
  success: true;
  data: {
    accessToken: string;
    refreshToken: string;
    expiresIn: number;
  };
}

// POST /api/v1/auth/logout
interface LogoutRequest {
  refreshToken: string;
}
```

#### Dashboard Endpoints

```typescript
// GET /api/v1/dashboard/overview
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
}
```

#### Channel Management Endpoints

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

// POST /api/v1/channels
interface CreateChannelRequest {
  name: string;
  niche: string;
  targetAudience: string;
  primaryLanguage: string;
  videoLength: 'short' | 'medium' | 'long';
  dailyVideoLimit: number;  // 1-3
}

// PATCH /api/v1/channels/:id
interface UpdateChannelRequest {
  name?: string;
  status?: 'active' | 'paused';
  automationEnabled?: boolean;
  settings?: Partial<ChannelSettings>;
}

// DELETE /api/v1/channels/:id
// Returns: 204 No Content
```

#### Video Management Endpoints

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

// POST /api/v1/videos/generate
interface GenerateVideoRequest {
  channelId: string;
  topic?: string;              // Optional, AI will choose if not provided
  style: 'educational' | 'entertainment' | 'news' | 'tutorial';
  length: 'short' | 'medium' | 'long';
  priority?: number;           // 1-10, default 5
  scheduledFor?: string;       // ISO datetime for scheduled generation
}

// GET /api/v1/videos/:id
// Returns detailed video information

// POST /api/v1/videos/:id/retry
// Retry failed video generation

// DELETE /api/v1/videos/:id
// Cancel queued video
```

#### Cost Management Endpoints

```typescript
// GET /api/v1/costs/breakdown
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

// POST /api/v1/costs/alerts
interface CreateCostAlertRequest {
  type: 'daily' | 'monthly' | 'per_video';
  threshold: number;
  action: 'notify' | 'pause_generation' | 'both';
  channels?: string[];  // Specific channels or all
}
```

## 5.2 Authentication & Authorization

### JWT Implementation

```typescript
// JWT Token Structure
interface JWTPayload {
  sub: string;        // User ID
  email: string;
  role: string;
  iat: number;        // Issued at
  exp: number;        // Expiration
  jti: string;        // JWT ID for revocation
}

// Auth Service Implementation
class AuthService {
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private refreshPromise: Promise<void> | null = null;
  
  constructor() {
    this.loadTokens();
  }
  
  private loadTokens() {
    this.accessToken = localStorage.getItem('access_token');
    this.refreshToken = localStorage.getItem('refresh_token');
  }
  
  private saveTokens(access: string, refresh: string) {
    this.accessToken = access;
    this.refreshToken = refresh;
    localStorage.setItem('access_token', access);
    localStorage.setItem('refresh_token', refresh);
  }
  
  async login(email: string, password: string): Promise<User> {
    const response = await api.post<LoginResponse>('/auth/login', {
      email,
      password
    });
    
    this.saveTokens(
      response.data.accessToken,
      response.data.refreshToken
    );
    
    return response.data.user;
  }
  
  async refreshAccessToken(): Promise<void> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }
    
    this.refreshPromise = (async () => {
      try {
        const response = await api.post<RefreshTokenResponse>(
          '/auth/refresh',
          { refreshToken: this.refreshToken }
        );
        
        this.saveTokens(
          response.data.accessToken,
          response.data.refreshToken
        );
      } catch (error) {
        this.logout();
        throw error;
      } finally {
        this.refreshPromise = null;
      }
    })();
    
    return this.refreshPromise;
  }
  
  getAccessToken(): string | null {
    return this.accessToken;
  }
  
  isTokenExpired(): boolean {
    if (!this.accessToken) return true;
    
    try {
      const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
      return Date.now() >= payload.exp * 1000;
    } catch {
      return true;
    }
  }
  
  logout() {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = '/login';
  }
}

export const authService = new AuthService();
```

### API Client with Auth

```typescript
// Axios Instance with Interceptors
import axios, { AxiosInstance } from 'axios';

class APIClient {
  private client: AxiosInstance;
  
  constructor() {
    this.client = axios.create({
      baseURL: API_CONFIG.baseURL,
      timeout: API_CONFIG.timeout,
      headers: API_CONFIG.headers
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors() {
    // Request interceptor - add auth token
    this.client.interceptors.request.use(
      async (config) => {
        const token = authService.getAccessToken();
        
        if (token) {
          // Check if token is expired
          if (authService.isTokenExpired()) {
            await authService.refreshAccessToken();
          }
          
          config.headers.Authorization = `Bearer ${authService.getAccessToken()}`;
        }
        
        // Add request ID for tracking
        config.headers['X-Request-ID'] = crypto.randomUUID();
        
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor - handle errors
    this.client.interceptors.response.use(
      (response) => response.data,
      async (error) => {
        const originalRequest = error.config;
        
        // Handle token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            await authService.refreshAccessToken();
            originalRequest.headers.Authorization = 
              `Bearer ${authService.getAccessToken()}`;
            return this.client(originalRequest);
          } catch {
            authService.logout();
            return Promise.reject(error);
          }
        }
        
        // Transform error to standard format
        const apiError: APIError = {
          code: error.response?.data?.error?.code || 'UNKNOWN',
          message: error.response?.data?.error?.message || 'An error occurred',
          details: error.response?.data?.error?.details
        };
        
        return Promise.reject(apiError);
      }
    );
  }
  
  // HTTP Methods
  async get<T>(url: string, params?: any): Promise<T> {
    return this.client.get(url, { params });
  }
  
  async post<T>(url: string, data?: any): Promise<T> {
    return this.client.post(url, data);
  }
  
  async patch<T>(url: string, data?: any): Promise<T> {
    return this.client.patch(url, data);
  }
  
  async delete<T>(url: string): Promise<T> {
    return this.client.delete(url);
  }
}

export const apiClient = new APIClient();
```

## 5.3 External Service Integrations

### YouTube API Integration

```typescript
// YouTube Service
class YouTubeService {
  private clientId = import.meta.env.VITE_YOUTUBE_CLIENT_ID;
  private apiKey = import.meta.env.VITE_YOUTUBE_API_KEY;
  
  async authenticateChannel(channelId: string): Promise<void> {
    // OAuth 2.0 flow
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${this.clientId}&` +
      `redirect_uri=${window.location.origin}/youtube/callback&` +
      `response_type=token&` +
      `scope=https://www.googleapis.com/auth/youtube.upload`;
    
    window.location.href = authUrl;
  }
  
  async uploadVideo(
    channelId: string,
    videoData: VideoUploadData
  ): Promise<string> {
    const response = await fetch(
      'https://www.googleapis.com/upload/youtube/v3/videos',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.getChannelToken(channelId)}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          snippet: {
            title: videoData.title,
            description: videoData.description,
            tags: videoData.tags,
            categoryId: videoData.categoryId
          },
          status: {
            privacyStatus: 'public',
            selfDeclaredMadeForKids: false
          }
        })
      }
    );
    
    const result = await response.json();
    return result.id;
  }
  
  async getChannelAnalytics(
    channelId: string,
    startDate: string,
    endDate: string
  ): Promise<ChannelAnalytics> {
    const response = await fetch(
      `https://youtubeanalytics.googleapis.com/v2/reports?` +
      `ids=channel==${channelId}&` +
      `startDate=${startDate}&` +
      `endDate=${endDate}&` +
      `metrics=views,estimatedRevenue,averageViewDuration&` +
      `dimensions=day`,
      {
        headers: {
          'Authorization': `Bearer ${this.getChannelToken(channelId)}`
        }
      }
    );
    
    return response.json();
  }
  
  private getChannelToken(channelId: string): string {
    // Get stored OAuth token for channel
    return localStorage.getItem(`youtube_token_${channelId}`) || '';
  }
}

export const youtubeService = new YouTubeService();
```

### OpenAI Integration

```typescript
// OpenAI Service
class OpenAIService {
  private apiKey = import.meta.env.VITE_OPENAI_API_KEY;
  private baseURL = 'https://api.openai.com/v1';
  
  async generateScript(params: ScriptParams): Promise<string> {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'You are a YouTube content script writer.'
          },
          {
            role: 'user',
            content: this.buildScriptPrompt(params)
          }
        ],
        temperature: 0.7,
        max_tokens: 2000
      })
    });
    
    const result = await response.json();
    return result.choices[0].message.content;
  }
  
  private buildScriptPrompt(params: ScriptParams): string {
    return `Create a ${params.length} YouTube video script about ${params.topic} 
            for ${params.targetAudience}. Style: ${params.style}. 
            Include hooks, main content, and call-to-action.`;
  }
}
```

### ElevenLabs Integration

```typescript
// Voice Synthesis Service
class ElevenLabsService {
  private apiKey = import.meta.env.VITE_ELEVENLABS_API_KEY;
  private baseURL = 'https://api.elevenlabs.io/v1';
  
  async synthesizeVoice(
    text: string,
    voiceId: string
  ): Promise<ArrayBuffer> {
    const response = await fetch(
      `${this.baseURL}/text-to-speech/${voiceId}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': this.apiKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_monolingual_v1',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.5
          }
        })
      }
    );
    
    return response.arrayBuffer();
  }
  
  async getVoices(): Promise<Voice[]> {
    const response = await fetch(`${this.baseURL}/voices`, {
      headers: {
        'xi-api-key': this.apiKey
      }
    });
    
    const result = await response.json();
    return result.voices;
  }
}
```

## 5.4 Error Handling Standards

### Error Response Format

```typescript
// Standard Error Response
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

### Error Handling Service

```typescript
// Error Handler
class ErrorHandler {
  handle(error: any): void {
    // Log error
    this.logError(error);
    
    // Display user-friendly message
    const message = this.getUserMessage(error);
    this.showNotification(message);
    
    // Report to monitoring service
    if (this.isCritical(error)) {
      this.reportToMonitoring(error);
    }
  }
  
  private getUserMessage(error: any): string {
    const errorMessages: Record<string, string> = {
      'AUTH_001': 'Your session has expired. Please login again.',
      'VAL_001': 'Please fill in all required fields.',
      'RES_001': 'The requested resource was not found.',
      'RES_003': 'You have reached the maximum limit of 5 channels.',
      'RATE_001': 'Too many requests. Please try again later.',
      'EXT_001': 'YouTube service is temporarily unavailable.',
      'DEFAULT': 'An unexpected error occurred. Please try again.'
    };
    
    return errorMessages[error.code] || errorMessages.DEFAULT;
  }
  
  private isCritical(error: any): boolean {
    const criticalCodes = ['SRV', 'PAY', 'AUTH_003'];
    return criticalCodes.some(code => error.code?.startsWith(code));
  }
  
  private logError(error: any): void {
    console.error('[API Error]', {
      code: error.code,
      message: error.message,
      details: error.details,
      timestamp: new Date().toISOString()
    });
  }
  
  private showNotification(message: string): void {
    // Use toast notification system
    toast.error(message);
  }
  
  private reportToMonitoring(error: any): void {
    // Send to Sentry or similar service
    if (window.Sentry) {
      window.Sentry.captureException(error);
    }
  }
}

export const errorHandler = new ErrorHandler();
```

### Retry Logic

```typescript
// Retry Configuration
interface RetryConfig {
  maxAttempts: number;
  backoffMultiplier: number;
  maxDelay: number;
  retryableErrors: string[];
}

// Retry Wrapper
async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig = {
    maxAttempts: 3,
    backoffMultiplier: 2,
    maxDelay: 10000,
    retryableErrors: ['RATE_001', 'EXT_001', 'EXT_002']
  }
): Promise<T> {
  let lastError;
  
  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      
      // Check if error is retryable
      if (!config.retryableErrors.includes(error.code)) {
        throw error;
      }
      
      // Calculate delay
      const delay = Math.min(
        Math.pow(config.backoffMultiplier, attempt - 1) * 1000,
        config.maxDelay
      );
      
      console.log(`Retry attempt ${attempt} after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError;
}

// Usage Example
const fetchWithRetry = () => withRetry(
  () => apiClient.get('/dashboard/metrics'),
  { maxAttempts: 5 }
);
```