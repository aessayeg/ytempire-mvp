# YTEMPIRE React Engineer - Technical Architecture
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: Architecture & Technology Stack

---

## 1. Technology Stack

### 1.1 Core Framework & Languages

```typescript
const techStack = {
  // Core
  framework: "React 18.2.0",
  language: "TypeScript 5.3",
  runtime: "Node.js 18+",
  
  // Build & Development
  bundler: "Vite 5.0",
  packageManager: "npm 9+",
  
  // State Management - CRITICAL: NO Redux
  stateManagement: "Zustand 4.4",
  asyncState: "React Query 3.39",
  
  // Routing
  routing: "React Router 6.20",
  
  // UI Framework
  uiLibrary: "Material-UI 5.14",
  icons: "@mui/icons-material 5.14",
  
  // Data Visualization - CRITICAL: NO D3.js
  charts: "Recharts 2.10",
  
  // Forms & Validation
  forms: "React Hook Form 7.x",
  validation: "Zod 3.x",
  
  // HTTP & WebSocket
  httpClient: "Axios 1.6",
  websocket: "Native WebSocket API",
  
  // Testing
  testRunner: "Vitest 1.0",
  testingLibrary: "React Testing Library 14",
  e2e: "Playwright (Post-MVP)",
  
  // Quality Tools
  linter: "ESLint 8.50",
  formatter: "Prettier 3.0",
  typeChecker: "TypeScript Strict Mode"
};
```

### 1.2 Critical Technical Decisions

| Decision | Choice | Rationale | Impact |
|----------|--------|-----------|---------|
| **State Management** | Zustand over Redux | Simpler API, less boilerplate, 8KB vs 50KB | -42KB bundle, faster development |
| **Charts** | Recharts over D3.js | React-native, easier implementation | Faster delivery, sufficient for MVP |
| **UI Library** | Material-UI | Pre-built components, consistent design | +300KB bundle (accepted tradeoff) |
| **Build Tool** | Vite over CRA/Webpack | 10x faster cold starts, better DX | Improved developer productivity |
| **Polling** | 60-second intervals | Simpler than WebSockets | Reduced complexity, sufficient for MVP |

### 1.3 Bundle Size Budget

```yaml
Total Budget: 1000KB (1MB)

Breakdown:
  React + React-DOM: 140KB
  Material-UI: 300KB
  Recharts: 150KB
  React Router: 35KB
  Zustand: 8KB
  Axios: 25KB
  Other Libraries: 100KB
  Application Code: 200KB
  Buffer: 42KB
  
Current Estimate: 958KB (under budget)
```

---

## 2. Project Structure

### 2.1 Directory Organization

```
ytempire-frontend/
├── public/
│   ├── favicon.ico
│   └── robots.txt
│
├── src/
│   ├── components/         # Reusable UI components (30-40 total)
│   │   ├── common/         # Generic components
│   │   │   ├── Button/
│   │   │   ├── Input/
│   │   │   ├── Modal/
│   │   │   ├── Card/
│   │   │   └── LoadingSpinner/
│   │   │
│   │   ├── layout/         # Layout components
│   │   │   ├── AppLayout/
│   │   │   ├── Header/
│   │   │   ├── Sidebar/
│   │   │   └── Footer/
│   │   │
│   │   ├── features/       # Business-specific components
│   │   │   ├── channels/
│   │   │   ├── videos/
│   │   │   ├── dashboard/
│   │   │   └── settings/
│   │   │
│   │   └── charts/         # Recharts wrappers
│   │       ├── LineChart/
│   │       ├── BarChart/
│   │       └── PieChart/
│   │
│   ├── pages/              # Route components (20-25 screens)
│   │   ├── Dashboard/
│   │   ├── Channels/
│   │   ├── Videos/
│   │   ├── Analytics/
│   │   ├── Settings/
│   │   └── Auth/
│   │
│   ├── stores/             # Zustand state management
│   │   ├── useAuthStore.ts
│   │   ├── useChannelStore.ts
│   │   ├── useVideoStore.ts
│   │   ├── useDashboardStore.ts
│   │   └── useCostStore.ts
│   │
│   ├── services/           # API layer
│   │   ├── api.ts          # Axios instance
│   │   ├── auth.ts
│   │   ├── channels.ts
│   │   ├── videos.ts
│   │   └── websocket.ts
│   │
│   ├── hooks/              # Custom React hooks
│   │   ├── usePolling.ts
│   │   ├── useWebSocket.ts
│   │   ├── useAuth.ts
│   │   └── useDebounce.ts
│   │
│   ├── utils/              # Helper functions
│   │   ├── formatters.ts
│   │   ├── validators.ts
│   │   ├── constants.ts
│   │   └── helpers.ts
│   │
│   ├── types/              # TypeScript definitions
│   │   ├── api.types.ts
│   │   ├── models.types.ts
│   │   └── components.types.ts
│   │
│   ├── styles/             # Global styles
│   │   ├── theme.ts        # MUI theme
│   │   └── globals.css
│   │
│   ├── App.tsx
│   ├── main.tsx
│   └── vite-env.d.ts
│
├── tests/                  # Test files
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .env.example
├── .eslintrc.json
├── .prettierrc
├── tsconfig.json
├── vite.config.ts
└── package.json
```

### 2.2 File Naming Conventions

```typescript
// Component files
ComponentName.tsx           // Component implementation
ComponentName.test.tsx      // Component tests
ComponentName.types.ts      // TypeScript interfaces
ComponentName.styles.ts     // Styled components (if used)
index.ts                   // Public exports

// Store files
useStoreName.ts            // Zustand store
useStoreName.test.ts       // Store tests

// Service files
serviceName.ts             // API service
serviceName.test.ts        // Service tests

// Hook files
useHookName.ts             // Custom hook
useHookName.test.ts        // Hook tests
```

---

## 3. State Management Architecture

### 3.1 Zustand Store Pattern

```typescript
// stores/useChannelStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { channelApi } from '@/services/channels';

interface Channel {
  id: string;
  name: string;
  status: 'active' | 'paused';
  automationEnabled: boolean;
  videosToday: number;
  totalVideos: number;
}

interface ChannelStore {
  // State
  channels: Channel[];
  selectedChannelId: string | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchChannels: () => Promise<void>;
  selectChannel: (id: string) => void;
  updateChannel: (id: string, data: Partial<Channel>) => void;
  toggleAutomation: (id: string) => Promise<void>;
  
  // Computed (selectors)
  getSelectedChannel: () => Channel | undefined;
  getActiveChannels: () => Channel[];
}

export const useChannelStore = create<ChannelStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        channels: [],
        selectedChannelId: null,
        loading: false,
        error: null,
        
        // Actions
        fetchChannels: async () => {
          set({ loading: true, error: null });
          try {
            const channels = await channelApi.getChannels();
            set({ channels, loading: false });
          } catch (error) {
            set({ 
              error: error instanceof Error ? error.message : 'Failed to fetch channels',
              loading: false 
            });
          }
        },
        
        selectChannel: (id) => set({ selectedChannelId: id }),
        
        updateChannel: (id, data) => set((state) => ({
          channels: state.channels.map(ch => 
            ch.id === id ? { ...ch, ...data } : ch
          )
        })),
        
        toggleAutomation: async (id) => {
          const channel = get().channels.find(ch => ch.id === id);
          if (!channel) return;
          
          try {
            const updated = await channelApi.toggleAutomation(
              id, 
              !channel.automationEnabled
            );
            get().updateChannel(id, updated);
          } catch (error) {
            set({ 
              error: error instanceof Error ? error.message : 'Failed to toggle automation'
            });
          }
        },
        
        // Computed
        getSelectedChannel: () => {
          const { channels, selectedChannelId } = get();
          return channels.find(ch => ch.id === selectedChannelId);
        },
        
        getActiveChannels: () => {
          return get().channels.filter(ch => ch.status === 'active');
        }
      }),
      {
        name: 'channel-store',
        partialize: (state) => ({ selectedChannelId: state.selectedChannelId })
      }
    )
  )
);
```

### 3.2 Store Organization

```yaml
Stores (5 total for MVP):
  
  useAuthStore:
    - User authentication state
    - Token management
    - Login/logout actions
    - Permission checks
    
  useChannelStore:
    - Channel list and selection
    - Channel CRUD operations
    - Automation toggles
    - Channel metrics
    
  useVideoStore:
    - Video queue management
    - Generation status tracking
    - Video history
    - Processing updates
    
  useDashboardStore:
    - Overview metrics
    - Chart data
    - Real-time updates
    - Activity feed
    
  useCostStore:
    - Cost tracking
    - Budget management
    - Cost alerts
    - Billing information
```

### 3.3 Polling Implementation

```typescript
// hooks/usePolling.ts
import { useEffect, useRef } from 'react';

interface UsePollingOptions {
  enabled?: boolean;
  interval?: number;
  onError?: (error: Error) => void;
}

export function usePolling(
  callback: () => void | Promise<void>,
  options: UsePollingOptions = {}
) {
  const {
    enabled = true,
    interval = 60000, // 60 seconds default
    onError
  } = options;
  
  const savedCallback = useRef(callback);
  const intervalId = useRef<NodeJS.Timeout>();
  
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);
  
  useEffect(() => {
    if (!enabled) {
      if (intervalId.current) {
        clearInterval(intervalId.current);
      }
      return;
    }
    
    const tick = async () => {
      try {
        await savedCallback.current();
      } catch (error) {
        onError?.(error as Error);
      }
    };
    
    // Initial call
    tick();
    
    // Setup interval
    intervalId.current = setInterval(tick, interval);
    
    return () => {
      if (intervalId.current) {
        clearInterval(intervalId.current);
      }
    };
  }, [enabled, interval, onError]);
}

// Usage in component
function Dashboard() {
  const { fetchDashboard } = useDashboardStore();
  
  usePolling(fetchDashboard, {
    interval: 60000, // 60 seconds
    enabled: true,
    onError: (error) => console.error('Polling error:', error)
  });
  
  // Component rendering...
}
```

---

## 4. Performance Optimization Strategy

### 4.1 Code Splitting

```typescript
// App.tsx - Route-based code splitting
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import LoadingSpinner from '@/components/common/LoadingSpinner';

// Lazy load all route components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Channels = lazy(() => import('./pages/Channels'));
const Videos = lazy(() => import('./pages/Videos'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Settings = lazy(() => import('./pages/Settings'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner fullScreen />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/channels/*" element={<Channels />} />
        <Route path="/videos/*" element={<Videos />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/settings/*" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

### 4.2 Bundle Optimization

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({ open: true, gzipSize: true })
  ],
  build: {
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui': ['@mui/material', '@emotion/react', '@emotion/styled'],
          'charts': ['recharts'],
          'state': ['zustand', 'react-query'],
          'utils': ['axios', 'date-fns', 'zod']
        }
      }
    },
    chunkSizeWarningLimit: 500
  }
});
```

### 4.3 Component Optimization

```typescript
// Memoization pattern for expensive components
import { memo, useMemo, useCallback } from 'react';

interface ChannelCardProps {
  channel: Channel;
  onSelect?: (id: string) => void;
}

export const ChannelCard = memo<ChannelCardProps>(({ 
  channel, 
  onSelect 
}) => {
  // Memoize expensive computations
  const metrics = useMemo(() => {
    return calculateChannelMetrics(channel);
  }, [channel.id, channel.videosToday, channel.totalVideos]);
  
  // Memoize callbacks
  const handleClick = useCallback(() => {
    onSelect?.(channel.id);
  }, [channel.id, onSelect]);
  
  return (
    <Card onClick={handleClick}>
      {/* Component content */}
    </Card>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for memo
  return (
    prevProps.channel.id === nextProps.channel.id &&
    prevProps.channel.updatedAt === nextProps.channel.updatedAt
  );
});
```

---

## 5. Security Architecture

### 5.1 Authentication Flow

```typescript
// JWT token management
interface AuthTokens {
  accessToken: string;  // 1 hour expiry
  refreshToken: string; // 7 days expiry
}

// Axios interceptor for token management
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token expired, try refresh
      const refreshToken = localStorage.getItem('refreshToken');
      if (refreshToken) {
        try {
          const { data } = await authApi.refresh(refreshToken);
          localStorage.setItem('accessToken', data.accessToken);
          // Retry original request
          return axios.request(error.config);
        } catch {
          // Refresh failed, redirect to login
          window.location.href = '/login';
        }
      }
    }
    return Promise.reject(error);
  }
);
```

### 5.2 Security Best Practices

```yaml
Authentication:
  - JWT tokens with secure httpOnly cookies
  - Automatic token refresh
  - Session timeout after 30 minutes idle
  
Data Protection:
  - HTTPS only in production
  - XSS protection via React's default escaping
  - CSRF tokens for state-changing operations
  - Input validation with Zod schemas
  
Storage:
  - Sensitive data in memory only
  - Non-sensitive preferences in localStorage
  - No credentials in code or env files
  
API Security:
  - Request rate limiting
  - Input sanitization
  - Error messages don't leak sensitive info
```

---

## 6. Environment Configuration

### 6.1 Environment Variables

```bash
# .env.example
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_ANALYTICS=false
VITE_ENABLE_DEBUG=true

# Third-party Services
VITE_SENTRY_DSN=
VITE_ANALYTICS_ID=

# Build Configuration
VITE_BUILD_VERSION=${npm_package_version}
VITE_BUILD_TIME=${BUILD_TIME}
```

### 6.2 Configuration by Environment

```typescript
// config/index.ts
interface Config {
  apiBaseUrl: string;
  wsUrl: string;
  pollingInterval: number;
  features: {
    websocket: boolean;
    analytics: boolean;
    debug: boolean;
  };
}

const config: Config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL,
  wsUrl: import.meta.env.VITE_WS_URL,
  pollingInterval: 60000, // 60 seconds
  features: {
    websocket: import.meta.env.VITE_ENABLE_WEBSOCKET === 'true',
    analytics: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
    debug: import.meta.env.VITE_ENABLE_DEBUG === 'true'
  }
};

export default config;
```

---

**Document Status**: FINAL - Consolidated Version  
**Next Review**: Technical Architecture Review Week 4  
**Owner**: Frontend Team Lead  
**Questions**: Contact via #frontend-team Slack