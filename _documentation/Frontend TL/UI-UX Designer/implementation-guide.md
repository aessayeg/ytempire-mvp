# YTEMPIRE Implementation Guide

## 5.1 Development Setup

### Prerequisites

```bash
# Required Software
node >= 18.0.0
npm >= 9.0.0
git >= 2.30.0

# Recommended IDE
VSCode with extensions:
- ESLint
- Prettier
- TypeScript and JavaScript
- Material-UI Snippets
- React Snippets
```

### Project Initialization

```bash
# Clone repository
git clone https://github.com/ytempire/frontend.git
cd frontend

# Install dependencies
npm install

# Setup environment variables
cp .env.example .env.local

# Required environment variables
VITE_API_URL=http://localhost:3001
VITE_WEBSOCKET_URL=ws://localhost:3001
VITE_YOUTUBE_API_KEY=your_key_here
VITE_STRIPE_PUBLIC_KEY=your_key_here
VITE_SENTRY_DSN=your_dsn_here
```

### Development Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives",
    "format": "prettier --write \"src/**/*.{ts,tsx,json,css,scss}\"",
    "type-check": "tsc --noEmit",
    "analyze": "vite build --mode analyze"
  }
}
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@services': path.resolve(__dirname, './src/services'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@types': path.resolve(__dirname, './src/types')
    }
  },
  build: {
    target: 'es2020',
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui': ['@mui/material', '@emotion/react', '@emotion/styled'],
          'charts': ['recharts'],
          'state': ['zustand', 'immer']
        }
      }
    },
    // Bundle size limit: 1MB
    chunkSizeWarningLimit: 1000
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true
      }
    }
  }
});
```

## 5.2 Coding Standards

### TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@pages/*": ["src/pages/*"],
      "@stores/*": ["src/stores/*"],
      "@services/*": ["src/services/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### ESLint Configuration

```javascript
// .eslintrc.js
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
    'plugin:react/recommended',
    'prettier'
  ],
  ignorePatterns: ['dist', '.eslintrc.js'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true }
    ],
    'react/react-in-jsx-scope': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'error',
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_' }
    ],
    'no-console': ['warn', { allow: ['warn', 'error'] }]
  }
};
```

### Code Style Guidelines

```typescript
// Component Structure
import { FC, useState, useEffect } from 'react';
import { Box, Card, Typography } from '@mui/material';
import { useChannelStore } from '@stores/channelStore';
import { formatCurrency } from '@utils/formatters';
import type { Channel } from '@types/channel';

interface ChannelCardProps {
  channel: Channel;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
}

export const ChannelCard: FC<ChannelCardProps> = ({ 
  channel, 
  onEdit, 
  onDelete 
}) => {
  // State hooks first
  const [isLoading, setIsLoading] = useState(false);
  
  // Store hooks
  const { updateChannel } = useChannelStore();
  
  // Effects
  useEffect(() => {
    // Effect logic
  }, [channel.id]);
  
  // Handlers
  const handlePause = async () => {
    setIsLoading(true);
    try {
      await updateChannel(channel.id, { status: 'paused' });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Render
  return (
    <Card>
      <Typography variant="h6">{channel.name}</Typography>
      <Typography>{formatCurrency(channel.revenue)}</Typography>
    </Card>
  );
};

// Type definitions
export interface Channel {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'deleted';
  revenue: number;
  videoCount: number;
  createdAt: Date;
  updatedAt: Date;
}

// Constants
export const CHANNEL_LIMITS = {
  MAX_CHANNELS: 5,
  MAX_NAME_LENGTH: 50,
  MIN_NAME_LENGTH: 3
} as const;

// Utility functions
export const isChannelActive = (channel: Channel): boolean => {
  return channel.status === 'active';
};
```

### Naming Conventions

```typescript
// Files and folders
channelCard.tsx       // React components (camelCase)
useChannel.ts         // Custom hooks
channelStore.ts       // Zustand stores
api.service.ts        // Services
formatters.utils.ts   // Utilities
channel.types.ts      // Type definitions

// Variables and functions
const channelName = 'Tech Reviews';    // camelCase
const MAX_CHANNELS = 5;                // UPPER_SNAKE_CASE for constants
const isActive = true;                  // Boolean prefix with is/has/should

// React Components
const ChannelCard = () => {};          // PascalCase
const useChannelData = () => {};       // Hooks start with 'use'

// Types and Interfaces
interface Channel {}                    // PascalCase, singular
type ChannelStatus = 'active' | 'paused'; // PascalCase
enum ChannelType {}                     // PascalCase
```

## 5.3 State Management

### Zustand Store Pattern

```typescript
// stores/channelStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { channelApi } from '@services/api/channel';
import type { Channel } from '@types/channel';

interface ChannelState {
  // State
  channels: Channel[];
  selectedChannelId: string | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  fetchChannels: () => Promise<void>;
  createChannel: (data: CreateChannelDto) => Promise<Channel>;
  updateChannel: (id: string, data: UpdateChannelDto) => Promise<void>;
  deleteChannel: (id: string) => Promise<void>;
  selectChannel: (id: string | null) => void;
  
  // Computed getters
  getActiveChannels: () => Channel[];
  getChannelById: (id: string) => Channel | undefined;
}

export const useChannelStore = create<ChannelState>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      channels: [],
      selectedChannelId: null,
      isLoading: false,
      error: null,
      
      // Actions
      fetchChannels: async () => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        
        try {
          const channels = await channelApi.getAll();
          set((state) => {
            state.channels = channels;
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error.message;
            state.isLoading = false;
          });
        }
      },
      
      createChannel: async (data) => {
        const channels = get().channels;
        
        if (channels.length >= 5) {
          throw new Error('Maximum 5 channels allowed');
        }
        
        set((state) => {
          state.isLoading = true;
        });
        
        try {
          const newChannel = await channelApi.create(data);
          set((state) => {
            state.channels.push(newChannel);
            state.isLoading = false;
          });
          return newChannel;
        } catch (error) {
          set((state) => {
            state.error = error.message;
            state.isLoading = false;
          });
          throw error;
        }
      },
      
      updateChannel: async (id, data) => {
        await channelApi.update(id, data);
        set((state) => {
          const index = state.channels.findIndex(c => c.id === id);
          if (index !== -1) {
            state.channels[index] = { ...state.channels[index], ...data };
          }
        });
      },
      
      deleteChannel: async (id) => {
        await channelApi.delete(id);
        set((state) => {
          state.channels = state.channels.filter(c => c.id !== id);
          if (state.selectedChannelId === id) {
            state.selectedChannelId = null;
          }
        });
      },
      
      selectChannel: (id) => {
        set((state) => {
          state.selectedChannelId = id;
        });
      },
      
      // Computed getters
      getActiveChannels: () => {
        return get().channels.filter(c => c.status === 'active');
      },
      
      getChannelById: (id) => {
        return get().channels.find(c => c.id === id);
      }
    })),
    {
      name: 'channel-store'
    }
  )
);
```

### Store Organization

```typescript
// stores/index.ts
export { useAuthStore } from './authStore';
export { useChannelStore } from './channelStore';
export { useVideoStore } from './videoStore';
export { useDashboardStore } from './dashboardStore';
export { useCostStore } from './costStore';

// Store types
export type { AuthState } from './authStore';
export type { ChannelState } from './channelStore';
export type { VideoState } from './videoStore';
export type { DashboardState } from './dashboardStore';
export type { CostState } from './costStore';
```

## 5.4 API Integration

### API Client Setup

```typescript
// services/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { useAuthStore } from '@stores/authStore';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

class ApiClient {
  private client: AxiosInstance;
  
  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = useAuthStore.getState().token;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response.data,
      async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            await useAuthStore.getState().refreshToken();
            return this.client(originalRequest);
          } catch (refreshError) {
            useAuthStore.getState().logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }
        
        return Promise.reject(error);
      }
    );
  }
  
  public async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.get<T>(url, config);
  }
  
  public async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return this.client.post<T>(url, data, config);
  }
  
  public async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return this.client.put<T>(url, data, config);
  }
  
  public async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.delete<T>(url, config);
  }
}

export const apiClient = new ApiClient();
```

### Service Layer

```typescript
// services/api/channel.ts
import { apiClient } from './client';
import type { Channel, CreateChannelDto, UpdateChannelDto } from '@types/channel';

export const channelApi = {
  async getAll(): Promise<Channel[]> {
    return apiClient.get<Channel[]>('/api/channels');
  },
  
  async getById(id: string): Promise<Channel> {
    return apiClient.get<Channel>(`/api/channels/${id}`);
  },
  
  async create(data: CreateChannelDto): Promise<Channel> {
    return apiClient.post<Channel>('/api/channels', data);
  },
  
  async update(id: string, data: UpdateChannelDto): Promise<Channel> {
    return apiClient.put<Channel>(`/api/channels/${id}`, data);
  },
  
  async delete(id: string): Promise<void> {
    return apiClient.delete(`/api/channels/${id}`);
  },
  
  async pause(id: string): Promise<Channel> {
    return apiClient.post<Channel>(`/api/channels/${id}/pause`);
  },
  
  async resume(id: string): Promise<Channel> {
    return apiClient.post<Channel>(`/api/channels/${id}/resume`);
  },
  
  async generateVideo(id: string, topic?: string): Promise<Video> {
    return apiClient.post<Video>(`/api/channels/${id}/generate-video`, { topic });
  }
};
```

### WebSocket Integration

```typescript
// services/websocket.ts
import { useVideoStore } from '@stores/videoStore';
import { useCostStore } from '@stores/costStore';

const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:3001';

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect(token: string): void {
    this.ws = new WebSocket(`${WEBSOCKET_URL}?token=${token}`);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect(token);
    };
  }
  
  private handleMessage(data: WebSocketMessage): void {
    switch (data.type) {
      case 'video.completed':
        useVideoStore.getState().handleVideoCompleted(data.payload);
        break;
        
      case 'video.failed':
        useVideoStore.getState().handleVideoFailed(data.payload);
        break;
        
      case 'cost.alert':
        useCostStore.getState().handleCostAlert(data.payload);
        break;
        
      default:
        console.warn('Unknown WebSocket message type:', data.type);
    }
  }
  
  private attemptReconnect(token: string): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    
    this.reconnectTimeout = setTimeout(() => {
      console.log(`Attempting reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect(token);
    }, delay);
  }
  
  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  send(message: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }
}

export const websocketService = new WebSocketService();
```

## 5.5 Performance Optimization

### Code Splitting

```typescript
// App.tsx - Route-based code splitting
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { LoadingScreen } from '@components/common/LoadingScreen';

// Lazy load pages
const Dashboard = lazy(() => import('@pages/Dashboard'));
const Channels = lazy(() => import('@pages/Channels'));
const Videos = lazy(() => import('@pages/Videos'));
const Analytics = lazy(() => import('@pages/Analytics'));
const Settings = lazy(() => import('@pages/Settings'));

export const App = () => {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/channels/*" element={<Channels />} />
        <Route path="/videos/*" element={<Videos />} />
        <Route path="/analytics/*" element={<Analytics />} />
        <Route path="/settings/*" element={<Settings />} />
      </Routes>
    </Suspense>
  );
};
```

### React Performance Patterns

```typescript
// Memoization for expensive components
import { memo, useMemo, useCallback } from 'react';

export const ExpensiveChart = memo(({ data, onDataPoint }) => {
  // Memoize expensive calculations
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      value: item.value * 100,
      formatted: formatCurrency(item.value)
    }));
  }, [data]);
  
  // Memoize callbacks
  const handleClick = useCallback((point) => {
    onDataPoint(point);
  }, [onDataPoint]);
  
  return (
    <LineChart data={processedData} onClick={handleClick} />
  );
});

// Virtual scrolling for large lists
import { FixedSizeList } from 'react-window';

export const VideoList = ({ videos }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <VideoRow video={videos[index]} />
    </div>
  );
  
  return (
    <FixedSizeList
      height={600}
      itemCount={videos.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
};
```

### Bundle Optimization

```typescript
// Dynamic imports for heavy libraries
const loadChartLibrary = async () => {
  const { LineChart, BarChart, PieChart } = await import('recharts');
  return { LineChart, BarChart, PieChart };
};

// Image optimization
const OptimizedImage = ({ src, alt, ...props }) => {
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      decoding="async"
      {...props}
    />
  );
};

// Debounced search
import { useMemo } from 'react';
import { debounce } from 'lodash-es';

export const useDebounceSearch = (searchFn: (query: string) => void, delay = 300) => {
  return useMemo(
    () => debounce(searchFn, delay),
    [searchFn, delay]
  );
};
```

### Performance Monitoring

```typescript
// Performance observer
export const measurePerformance = () => {
  // First Contentful Paint
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.name === 'first-contentful-paint') {
        console.log('FCP:', entry.startTime);
        // Send to analytics
      }
    }
  });
  
  observer.observe({ entryTypes: ['paint'] });
  
  // Component render time
  if (process.env.NODE_ENV === 'development') {
    const { Profiler } = require('react');
    
    return (Component: React.ComponentType) => {
      return (props: any) => (
        <Profiler
          id={Component.name}
          onRender={(id, phase, actualDuration) => {
            console.log(`${id} (${phase}) took ${actualDuration}ms`);
          }}
        >
          <Component {...props} />
        </Profiler>
      );
    };
  }
};
```