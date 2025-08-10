# YTEMPIRE Frontend & API Specifications - Complete Guide

**Document Version**: 1.0  
**Date**: January 2025  
**Status**: FINAL - READY FOR IMPLEMENTATION  
**Author**: Full-Stack Architecture Team  
**For**: Analytics Engineer - Frontend & API Implementation

---

## 1. Frontend Specifications

### 1.1 Technology Stack

```yaml
frontend_stack:
  framework:
    name: "React"
    version: "18.2.0"
    setup_method: "Vite"
    
  build_tool:
    name: "Vite"
    version: "5.0.0"
    config: "TypeScript + SWC"
    
  language:
    name: "TypeScript"
    version: "5.3.0"
    strict_mode: true
    
  styling:
    framework: "Tailwind CSS"
    version: "3.4.0"
    component_library: "shadcn/ui"
    
  state_management:
    global: "Zustand"
    server_state: "TanStack Query (React Query)"
    forms: "React Hook Form + Zod"
    
  routing:
    library: "React Router"
    version: "6.20.0"
    
  authentication:
    method: "JWT with refresh tokens"
    storage: "HttpOnly cookies + localStorage"
```

### 1.2 Project Structure

```typescript
// Frontend project structure
frontend/
├── src/
│   ├── components/
│   │   ├── ui/               // shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   └── ...
│   │   ├── dashboard/
│   │   │   ├── ChannelGrid.tsx
│   │   │   ├── MetricsOverview.tsx
│   │   │   ├── VideoQueue.tsx
│   │   │   └── RevenueChart.tsx
│   │   ├── video/
│   │   │   ├── VideoGenerator.tsx
│   │   │   ├── VideoPreview.tsx
│   │   │   ├── VideoEditor.tsx
│   │   │   └── VideoAnalytics.tsx
│   │   └── layout/
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       └── Footer.tsx
│   │
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Channels.tsx
│   │   ├── Videos.tsx
│   │   ├── Analytics.tsx
│   │   ├── Settings.tsx
│   │   └── Onboarding.tsx
│   │
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useChannels.ts
│   │   ├── useVideos.ts
│   │   └── useWebSocket.ts
│   │
│   ├── services/
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   ├── websocket.ts
│   │   └── storage.ts
│   │
│   ├── stores/
│   │   ├── authStore.ts
│   │   ├── channelStore.ts
│   │   ├── videoStore.ts
│   │   └── settingsStore.ts
│   │
│   ├── types/
│   │   ├── api.types.ts
│   │   ├── channel.types.ts
│   │   ├── video.types.ts
│   │   └── user.types.ts
│   │
│   └── utils/
│       ├── constants.ts
│       ├── helpers.ts
│       └── validators.ts
```

### 1.3 UI Framework and Components

```typescript
// shadcn/ui configuration with Tailwind CSS
// tailwind.config.js
module.exports = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        youtube: {
          red: "#FF0000",
          dark: "#282828",
          light: "#F9F9F9",
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}

// Component examples
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { toast } from "sonner"
```

### 1.4 Authentication Flow Implementation

```typescript
// Authentication service implementation
class AuthenticationService {
  private baseURL = import.meta.env.VITE_API_URL;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  
  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await fetch(`${this.baseURL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include', // For httpOnly cookies
      body: JSON.stringify({ email, password })
    });
    
    if (!response.ok) throw new Error('Login failed');
    
    const data = await response.json();
    
    // Store tokens
    this.accessToken = data.accessToken;
    this.refreshToken = data.refreshToken;
    
    // Store in localStorage for persistence
    localStorage.setItem('accessToken', data.accessToken);
    
    // Refresh token is in httpOnly cookie (secure)
    
    return data;
  }
  
  async refreshAccessToken(): Promise<string> {
    const response = await fetch(`${this.baseURL}/api/auth/refresh`, {
      method: 'POST',
      credentials: 'include', // Sends httpOnly refresh token cookie
    });
    
    if (!response.ok) {
      // Refresh failed, redirect to login
      this.logout();
      throw new Error('Session expired');
    }
    
    const data = await response.json();
    this.accessToken = data.accessToken;
    localStorage.setItem('accessToken', data.accessToken);
    
    return data.accessToken;
  }
  
  async authenticatedFetch(url: string, options: RequestInit = {}): Promise<Response> {
    // Add auth header
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${this.accessToken}`
    };
    
    let response = await fetch(url, { ...options, headers });
    
    // If 401, try refresh
    if (response.status === 401) {
      await this.refreshAccessToken();
      
      // Retry with new token
      headers.Authorization = `Bearer ${this.accessToken}`;
      response = await fetch(url, { ...options, headers });
    }
    
    return response;
  }
  
  logout(): void {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('accessToken');
    
    // Clear httpOnly cookie
    fetch(`${this.baseURL}/api/auth/logout`, {
      method: 'POST',
      credentials: 'include'
    });
    
    window.location.href = '/login';
  }
}

export const authService = new AuthenticationService();
```

### 1.5 Dashboard Layout Specifications

```typescript
// Main Dashboard Component
import React from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const Dashboard: React.FC = () => {
  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-card">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-youtube-red">YTEMPIRE</h1>
        </div>
        <nav className="space-y-2 px-3">
          <NavItem icon={Home} label="Dashboard" active />
          <NavItem icon={PlayCircle} label="Channels" />
          <NavItem icon={Video} label="Videos" />
          <NavItem icon={TrendingUp} label="Analytics" />
          <NavItem icon={DollarSign} label="Revenue" />
          <NavItem icon={Settings} label="Settings" />
        </nav>
      </aside>
      
      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Header */}
        <header className="border-b bg-card px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Dashboard Overview</h2>
            <div className="flex items-center gap-4">
              <Button variant="outline" size="sm">
                <RefreshCw className="mr-2 h-4 w-4" />
                Sync Data
              </Button>
              <Button size="sm">
                <Plus className="mr-2 h-4 w-4" />
                Generate Video
              </Button>
            </div>
          </div>
        </header>
        
        {/* Dashboard Content */}
        <div className="p-6 space-y-6">
          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Total Views"
              value="1,234,567"
              change="+12.3%"
              icon={Eye}
            />
            <MetricCard
              title="Revenue"
              value="$12,345"
              change="+8.7%"
              icon={DollarSign}
            />
            <MetricCard
              title="Subscribers"
              value="45,678"
              change="+5.2%"
              icon={Users}
            />
            <MetricCard
              title="Videos"
              value="234"
              change="+15"
              icon={Video}
            />
          </div>
          
          {/* Charts and Tables */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Revenue Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <RevenueChart />
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Channel Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <ChannelPerformanceTable />
              </CardContent>
            </Card>
          </div>
          
          {/* Video Queue */}
          <Card>
            <CardHeader>
              <CardTitle>Video Generation Queue</CardTitle>
            </CardHeader>
            <CardContent>
              <VideoQueueList />
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};
```

---

## 2. API Structure Specifications

### 2.1 API Architecture

```yaml
api_architecture:
  type: "RESTful API"
  protocol: "HTTPS"
  format: "JSON"
  authentication: "JWT Bearer Token"
  versioning: "URL path versioning (/api/v1)"
  documentation: "OpenAPI 3.0 / Swagger"
  rate_limiting: "Per-user token bucket"
  
real_time_updates:
  protocol: "WebSocket"
  library: "Socket.io"
  events:
    - video.generation.progress
    - channel.metrics.update
    - system.notification
```

### 2.2 Complete API Endpoints

```typescript
// API Endpoint Definitions
interface APIEndpoints {
  // Authentication Endpoints
  '/api/v1/auth/register': {
    method: 'POST';
    body: {
      email: string;
      password: string;
      name: string;
    };
    response: {
      user: User;
      accessToken: string;
      refreshToken: string;
    };
  };
  
  '/api/v1/auth/login': {
    method: 'POST';
    body: {
      email: string;
      password: string;
    };
    response: {
      user: User;
      accessToken: string;
      refreshToken: string;
    };
  };
  
  '/api/v1/auth/refresh': {
    method: 'POST';
    body: {
      refreshToken: string;
    };
    response: {
      accessToken: string;
    };
  };
  
  '/api/v1/auth/logout': {
    method: 'POST';
    response: {
      success: boolean;
    };
  };
  
  // Channel Management Endpoints
  '/api/v1/channels': {
    method: 'GET';
    query?: {
      page?: number;
      limit?: number;
      status?: 'active' | 'paused' | 'deleted';
    };
    response: {
      channels: Channel[];
      total: number;
      page: number;
      limit: number;
    };
  };
  
  '/api/v1/channels': {
    method: 'POST';
    body: {
      name: string;
      niche: string;
      targetAudience: string;
      automationLevel: 'full' | 'semi' | 'manual';
    };
    response: Channel;
  };
  
  '/api/v1/channels/:id': {
    method: 'GET';
    response: ChannelDetails;
  };
  
  '/api/v1/channels/:id': {
    method: 'PATCH';
    body: Partial<Channel>;
    response: Channel;
  };
  
  '/api/v1/channels/:id': {
    method: 'DELETE';
    response: {
      success: boolean;
    };
  };
  
  '/api/v1/channels/:id/sync': {
    method: 'POST';
    response: {
      jobId: string;
      status: 'queued' | 'processing' | 'completed';
    };
  };
  
  // Video Management Endpoints
  '/api/v1/videos': {
    method: 'GET';
    query?: {
      channelId?: string;
      status?: 'draft' | 'processing' | 'published' | 'failed';
      page?: number;
      limit?: number;
    };
    response: {
      videos: Video[];
      total: number;
    };
  };
  
  '/api/v1/videos/generate': {
    method: 'POST';
    body: {
      channelId: string;
      topic: string;
      style: string;
      duration: number;
      publish: boolean;
    };
    response: {
      jobId: string;
      estimatedTime: number;
    };
  };
  
  '/api/v1/videos/:id': {
    method: 'GET';
    response: VideoDetails;
  };
  
  '/api/v1/videos/:id/publish': {
    method: 'POST';
    body: {
      scheduledTime?: string;
      privacy: 'public' | 'unlisted' | 'private';
    };
    response: {
      success: boolean;
      youtubeId: string;
    };
  };
  
  // Analytics Endpoints
  '/api/v1/analytics/overview': {
    method: 'GET';
    query: {
      startDate: string;
      endDate: string;
    };
    response: {
      totalViews: number;
      totalRevenue: number;
      totalSubscribers: number;
      totalVideos: number;
      trends: TrendData[];
    };
  };
  
  '/api/v1/analytics/channels/:id': {
    method: 'GET';
    query: {
      startDate: string;
      endDate: string;
      metrics: string[]; // ['views', 'revenue', 'engagement']
    };
    response: ChannelAnalytics;
  };
  
  '/api/v1/analytics/videos/:id': {
    method: 'GET';
    response: VideoAnalytics;
  };
  
  // AI & Content Generation Endpoints
  '/api/v1/ai/generate-script': {
    method: 'POST';
    body: {
      topic: string;
      style: string;
      duration: number;
      keywords: string[];
    };
    response: {
      script: string;
      title: string;
      description: string;
      tags: string[];
    };
  };
  
  '/api/v1/ai/generate-thumbnail': {
    method: 'POST';
    body: {
      title: string;
      style: string;
      elements: string[];
    };
    response: {
      thumbnailUrl: string;
      variations: string[];
    };
  };
  
  '/api/v1/ai/suggest-topics': {
    method: 'POST';
    body: {
      niche: string;
      channelId: string;
    };
    response: {
      topics: Topic[];
    };
  };
  
  // System & Settings Endpoints
  '/api/v1/settings': {
    method: 'GET';
    response: UserSettings;
  };
  
  '/api/v1/settings': {
    method: 'PATCH';
    body: Partial<UserSettings>;
    response: UserSettings;
  };
  
  '/api/v1/system/health': {
    method: 'GET';
    response: {
      status: 'healthy' | 'degraded' | 'down';
      services: ServiceStatus[];
    };
  };
}
```

### 2.3 Authentication Method (JWT Implementation)

```typescript
// Backend JWT Implementation (FastAPI example)
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid token type"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )

# Protected endpoint example
@app.get("/api/v1/channels")
async def get_channels(current_user = Depends(verify_token)):
    # User is authenticated
    user_id = current_user.get("sub")
    # Return user's channels
    return {"channels": [], "user_id": user_id}
```

### 2.4 Request/Response Formats

```typescript
// Standard API Response Format
interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  meta?: {
    timestamp: string;
    version: string;
    requestId: string;
  };
}

// Pagination Format
interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

// Error Response Format
interface ErrorResponse {
  success: false;
  error: {
    code: string; // e.g., "VALIDATION_ERROR", "NOT_FOUND", "UNAUTHORIZED"
    message: string;
    field?: string; // For validation errors
    details?: Record<string, any>;
  };
  meta: {
    timestamp: string;
    requestId: string;
  };
}

// Example Requests and Responses

// POST /api/v1/videos/generate
// Request:
{
  "channelId": "ch_123456",
  "topic": "10 Python Tips for Beginners",
  "style": "educational",
  "duration": 10,
  "publish": true,
  "scheduledTime": "2024-01-15T10:00:00Z"
}

// Response:
{
  "success": true,
  "data": {
    "jobId": "job_789012",
    "videoId": "vid_345678",
    "status": "processing",
    "estimatedCompletionTime": "2024-01-15T09:45:00Z",
    "progress": 0
  },
  "meta": {
    "timestamp": "2024-01-15T09:30:00Z",
    "version": "1.0.0",
    "requestId": "req_abc123"
  }
}

// GET /api/v1/analytics/overview?startDate=2024-01-01&endDate=2024-01-31
// Response:
{
  "success": true,
  "data": {
    "totalViews": 1234567,
    "totalRevenue": 12345.67,
    "totalSubscribers": 45678,
    "totalVideos": 234,
    "viewsChange": 12.3,
    "revenueChange": 8.7,
    "subscribersChange": 5.2,
    "trends": [
      {
        "date": "2024-01-01",
        "views": 42000,
        "revenue": 420.50,
        "subscribers": 150
      }
      // ... more daily data
    ]
  },
  "meta": {
    "timestamp": "2024-01-15T10:00:00Z",
    "version": "1.0.0",
    "requestId": "req_def456"
  }
}
```

### 2.5 WebSocket Real-time Updates

```typescript
// WebSocket Implementation for Real-time Updates
import { io, Socket } from 'socket.io-client';

class WebSocketService {
  private socket: Socket | null = null;
  private listeners: Map<string, Set<Function>> = new Map();
  
  connect(token: string): void {
    this.socket = io(import.meta.env.VITE_WS_URL, {
      auth: { token },
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    
    this.setupEventHandlers();
  }
  
  private setupEventHandlers(): void {
    if (!this.socket) return;
    
    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.emit('connected', true);
    });
    
    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.emit('connected', false);
    });
    
    // Video generation events
    this.socket.on('video.generation.started', (data) => {
      this.emit('videoGenerationStarted', data);
    });
    
    this.socket.on('video.generation.progress', (data) => {
      this.emit('videoGenerationProgress', data);
    });
    
    this.socket.on('video.generation.completed', (data) => {
      this.emit('videoGenerationCompleted', data);
    });
    
    this.socket.on('video.generation.failed', (data) => {
      this.emit('videoGenerationFailed', data);
    });
    
    // Analytics updates
    this.socket.on('analytics.realtime', (data) => {
      this.emit('analyticsUpdate', data);
    });
    
    // System notifications
    this.socket.on('notification', (data) => {
      this.emit('notification', data);
    });
  }
  
  subscribe(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }
  
  unsubscribe(event: string, callback: Function): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.delete(callback);
    }
  }
  
  private emit(event: string, data: any): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.forEach(callback => callback(data));
    }
  }
  
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const wsService = new WebSocketService();

// Usage in React component
const VideoGenerator: React.FC = () => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  
  useEffect(() => {
    // Subscribe to video generation events
    const handleProgress = (data: any) => {
      setProgress(data.progress);
      setStatus(data.status);
    };
    
    wsService.subscribe('videoGenerationProgress', handleProgress);
    
    return () => {
      wsService.unsubscribe('videoGenerationProgress', handleProgress);
    };
  }, []);
  
  // Component rendering...
};
```

---

## 3. Integration Examples

### 3.1 Complete Frontend API Integration

```typescript
// API Client Service
class APIClient {
  private baseURL = import.meta.env.VITE_API_URL;
  
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await authService.authenticatedFetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new APIError(error);
    }
    
    return response.json();
  }
  
  // Channel methods
  async getChannels(params?: ChannelParams) {
    const query = new URLSearchParams(params as any).toString();
    return this.request<ChannelResponse>(`/api/v1/channels?${query}`);
  }
  
  async createChannel(data: CreateChannelDTO) {
    return this.request<Channel>('/api/v1/channels', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
  
  // Video methods
  async generateVideo(data: GenerateVideoDTO) {
    return this.request<VideoJob>('/api/v1/videos/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
  
  async getVideoStatus(jobId: string) {
    return this.request<JobStatus>(`/api/v1/jobs/${jobId}`);
  }
  
  // Analytics methods
  async getAnalytics(startDate: string, endDate: string) {
    const params = new URLSearchParams({ startDate, endDate });
    return this.request<AnalyticsData>(`/api/v1/analytics/overview?${params}`);
  }
}

export const apiClient = new APIClient();
```

### 3.2 State Management with Zustand

```typescript
// Channel Store Example
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface ChannelState {
  channels: Channel[];
  selectedChannel: Channel | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  fetchChannels: () => Promise<void>;
  selectChannel: (channelId: string) => void;
  createChannel: (data: CreateChannelDTO) => Promise<void>;
  updateChannel: (channelId: string, data: Partial<Channel>) => Promise<void>;
  deleteChannel: (channelId: string) => Promise<void>;
}

export const useChannelStore = create<ChannelState>()(
  devtools(
    persist(
      (set, get) => ({
        channels: [],
        selectedChannel: null,
        isLoading: false,
        error: null,
        
        fetchChannels: async () => {
          set({ isLoading: true, error: null });
          try {
            const response = await apiClient.getChannels();
            set({ channels: response.channels, isLoading: false });
          } catch (error) {
            set({ error: error.message, isLoading: false });
          }
        },
        
        selectChannel: (channelId) => {
          const channel = get().channels.find(c => c.id === channelId);
          set({ selectedChannel: channel });
        },
        
        createChannel: async (data) => {
          set({ isLoading: true, error: null });
          try {
            const newChannel = await apiClient.createChannel(data);
            set(state => ({
              channels: [...state.channels, newChannel],
              isLoading: false
            }));
          } catch (error) {
            set({ error: error.message, isLoading: false });
          }
        },
        
        updateChannel: async (channelId, data) => {
          try {
            const updated = await apiClient.updateChannel(channelId, data);
            set(state => ({
              channels: state.channels.map(c =>
                c.id === channelId ? updated : c
              )
            }));
          } catch (error) {
            set({ error: error.message });
          }
        },
        
        deleteChannel: async (channelId) => {
          try {
            await apiClient.deleteChannel(channelId);
            set(state => ({
              channels: state.channels.filter(c => c.id !== channelId)
            }));
          } catch (error) {
            set({ error: error.message });
          }
        },
      }),
      {
        name: 'channel-storage',
      }
    )
  )
);
```

---

## Summary

This document provides complete specifications for:

1. **Frontend Stack**: React 18 with Vite, TypeScript, Tailwind CSS, shadcn/ui
2. **State Management**: Zustand for global state, TanStack Query for server state
3. **API Structure**: RESTful with JWT authentication, WebSocket for real-time
4. **Authentication**: JWT with refresh tokens, secure httpOnly cookies
5. **Dashboard Layout**: Complete component structure and organization

All specifications are production-ready and optimized for the YTEMPIRE platform requirements.