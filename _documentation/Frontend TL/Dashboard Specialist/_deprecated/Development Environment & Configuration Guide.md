# Development Environment & Configuration Guide
## For: Dashboard Specialist | YTEMPIRE Frontend Team

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**Status**: Implementation Ready

---

## Executive Summary

This guide provides complete setup instructions for the YTEMPIRE dashboard development environment, including environment variables, local backend configuration, design assets, and development tools.

### Development Stack
- **Node.js**: 18.x LTS
- **Package Manager**: npm (not yarn/pnpm for MVP)
- **Development Server**: Vite 5.0
- **Testing**: Jest + React Testing Library
- **Linting**: ESLint + Prettier
- **Git Hooks**: Husky + lint-staged

---

## 1. Environment Configuration

### 1.1 Environment Variables

```bash
# .env.example - Commit this to repository
# Copy to .env.local for local development

# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# Environment
VITE_ENV=development
VITE_DEBUG=true

# Feature Flags
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_COST_ALERTS=true
VITE_ENABLE_EXPORT=true

# External Services (development keys)
VITE_YOUTUBE_CLIENT_ID=your-dev-client-id
VITE_GOOGLE_ANALYTICS_ID=
VITE_SENTRY_DSN=

# Development Tools
VITE_ENABLE_MOCK_API=false
VITE_ENABLE_DEV_TOOLS=true

# Performance Monitoring
VITE_ENABLE_PERFORMANCE_MONITOR=true
VITE_PERFORMANCE_THRESHOLD_MS=2000
```

```bash
# .env.local - DO NOT COMMIT
# Actual values for local development

VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# Get these from the backend team
VITE_YOUTUBE_CLIENT_ID=actual-client-id-here
VITE_GOOGLE_ANALYTICS_ID=G-XXXXXXXXXX
VITE_SENTRY_DSN=https://xxxxxx@sentry.io/xxxxxx
```

```bash
# .env.production - Production values (CI/CD will inject)
VITE_API_BASE_URL=https://api.ytempire.com/v1
VITE_WS_URL=wss://api.ytempire.com/ws
VITE_ENV=production
VITE_DEBUG=false
```

### 1.2 Environment Type Definitions

```typescript
// src/types/env.d.ts
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_ENV: 'development' | 'staging' | 'production';
  readonly VITE_DEBUG: string;
  readonly VITE_ENABLE_WEBSOCKET: string;
  readonly VITE_ENABLE_COST_ALERTS: string;
  readonly VITE_ENABLE_EXPORT: string;
  readonly VITE_YOUTUBE_CLIENT_ID: string;
  readonly VITE_GOOGLE_ANALYTICS_ID?: string;
  readonly VITE_SENTRY_DSN?: string;
  readonly VITE_ENABLE_MOCK_API: string;
  readonly VITE_ENABLE_DEV_TOOLS: string;
  readonly VITE_ENABLE_PERFORMANCE_MONITOR: string;
  readonly VITE_PERFORMANCE_THRESHOLD_MS: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### 1.3 Configuration Service

```typescript
// src/config/index.ts
export const config = {
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL,
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
  },
  
  websocket: {
    url: import.meta.env.VITE_WS_URL,
    enabled: import.meta.env.VITE_ENABLE_WEBSOCKET === 'true',
    reconnectAttempts: 5,
    reconnectDelay: 1000,
    pingInterval: 30000,
  },
  
  features: {
    websocket: import.meta.env.VITE_ENABLE_WEBSOCKET === 'true',
    costAlerts: import.meta.env.VITE_ENABLE_COST_ALERTS === 'true',
    export: import.meta.env.VITE_ENABLE_EXPORT === 'true',
    mockApi: import.meta.env.VITE_ENABLE_MOCK_API === 'true',
    devTools: import.meta.env.VITE_ENABLE_DEV_TOOLS === 'true',
    performanceMonitor: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITOR === 'true',
  },
  
  performance: {
    threshold: parseInt(import.meta.env.VITE_PERFORMANCE_THRESHOLD_MS || '2000'),
    enableMonitoring: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITOR === 'true',
  },
  
  environment: {
    isDevelopment: import.meta.env.VITE_ENV === 'development',
    isStaging: import.meta.env.VITE_ENV === 'staging',
    isProduction: import.meta.env.VITE_ENV === 'production',
    debug: import.meta.env.VITE_DEBUG === 'true',
  },
  
  external: {
    youtubeClientId: import.meta.env.VITE_YOUTUBE_CLIENT_ID,
    googleAnalyticsId: import.meta.env.VITE_GOOGLE_ANALYTICS_ID,
    sentryDsn: import.meta.env.VITE_SENTRY_DSN,
  },
};

// Type-safe config access
export type Config = typeof config;
```

---

## 2. Local Development Setup

### 2.1 Initial Setup Script

```bash
#!/bin/bash
# setup.sh - Run this first time only

echo "🚀 Setting up YTEMPIRE Dashboard Development Environment..."

# Check Node version
NODE_VERSION=$(node -v)
echo "Node version: $NODE_VERSION"

# Install dependencies
echo "📦 Installing dependencies..."
npm ci

# Copy environment file
echo "🔧 Setting up environment..."
if [ ! -f .env.local ]; then
  cp .env.example .env.local
  echo "✅ Created .env.local - Please update with your values"
else
  echo "✅ .env.local already exists"
fi

# Setup git hooks
echo "🪝 Setting up git hooks..."
npm run prepare

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p src/assets/images
mkdir -p src/assets/icons
mkdir -p public/fonts

# Download local development data
echo "📊 Setting up mock data..."
npm run setup:mock-data

# Verify setup
echo "🔍 Verifying setup..."
npm run verify:setup

echo "✅ Setup complete! Run 'npm run dev' to start development"
```

### 2.2 Development Scripts

```json
// package.json scripts section
{
  "scripts": {
    // Development
    "dev": "vite",
    "dev:mock": "VITE_ENABLE_MOCK_API=true vite",
    "dev:prod": "VITE_ENV=production vite",
    
    // Building
    "build": "tsc && vite build",
    "build:analyze": "ANALYZE=true vite build",
    
    // Testing
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:ui": "jest --ui",
    
    // Linting & Formatting
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,css}\"",
    
    // Type Checking
    "type-check": "tsc --noEmit",
    "type-check:watch": "tsc --noEmit --watch",
    
    // Git Hooks
    "prepare": "husky install",
    
    // Utilities
    "clean": "rm -rf dist node_modules/.vite",
    "setup:mock-data": "node scripts/setup-mock-data.js",
    "verify:setup": "node scripts/verify-setup.js",
    "update:deps": "npm update && npm audit fix",
    
    // Performance
    "analyze:bundle": "vite-bundle-visualizer",
    "lighthouse": "lighthouse http://localhost:3000 --view"
  }
}
```

### 2.3 VS Code Configuration

```json
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "files.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.vite": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/coverage": true
  },
  "emmet.includeLanguages": {
    "javascript": "javascriptreact",
    "typescript": "typescriptreact"
  },
  "tailwindCSS.experimental.classRegex": [
    ["sx={{([^}]*)}", "[\"'`]([^\"'`]*)[\"'`]"]
  ]
}
```

```json
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "dsznajder.es7-react-js-snippets",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense",
    "mikestead.dotenv",
    "yoavbls.pretty-ts-errors"
  ]
}
```

---

## 3. Local Backend Setup

### 3.1 Docker Compose Configuration

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Mock API Server
  mock-api:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./mock-server:/app
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=development
    command: npm start
    
  # Redis for caching/sessions
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
      
  # PostgreSQL for data persistence
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ytempire_dev
      POSTGRES_USER: ytempire
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      
  # MinIO for S3-compatible storage
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: ytempire
      MINIO_ROOT_PASSWORD: dev_password
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data

volumes:
  redis-data:
  postgres-data:
  minio-data:
```

### 3.2 Mock API Server

```typescript
// mock-server/index.js
const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const { faker } = require('@faker-js/faker');

const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

// Mock authentication
app.post('/api/v1/auth/login', (req, res) => {
  const { email, password } = req.body;
  
  if (email === 'demo@ytempire.com' && password === 'demo123') {
    res.json({
      success: true,
      data: {
        accessToken: 'mock-jwt-token',
        refreshToken: 'mock-refresh-token',
        user: {
          id: '123',
          email: 'demo@ytempire.com',
          name: 'Demo User',
          role: 'user',
          channelLimit: 5,
          subscription: {
            plan: 'growth',
            status: 'active'
          }
        }
      },
      metadata: {
        timestamp: new Date(),
        requestId: faker.string.uuid()
      }
    });
  } else {
    res.status(401).json({
      success: false,
      error: {
        code: 'AUTH_INVALID_CREDENTIALS',
        message: 'Invalid email or password'
      }
    });
  }
});

// Mock dashboard data
app.get('/api/v1/dashboard/overview', (req, res) => {
  res.json({
    success: true,
    data: {
      metrics: {
        totalChannels: 5,
        activeChannels: 3,
        pausedChannels: 2,
        videosToday: faker.number.int({ min: 5, max: 15 }),
        videosThisWeek: faker.number.int({ min: 30, max: 100 }),
        videosProcessing: faker.number.int({ min: 0, max: 3 }),
        videosFailed: faker.number.int({ min: 0, max: 2 }),
        videosQueued: faker.number.int({ min: 0, max: 5 }),
        revenueToday: faker.number.float({ min: 50, max: 200, precision: 0.01 }),
        revenueThisWeek: faker.number.float({ min: 300, max: 1000, precision: 0.01 }),
        revenueThisMonth: faker.number.float({ min: 1000, max: 5000, precision: 0.01 }),
        costToday: faker.number.float({ min: 10, max: 50, precision: 0.01 }),
        avgCostPerVideo: faker.number.float({ min: 0.30, max: 0.50, precision: 0.01 }),
        projectedMonthlyCost: faker.number.float({ min: 300, max: 1500, precision: 0.01 }),
        automationPercentage: faker.number.int({ min: 85, max: 99 }),
        avgGenerationTime: faker.number.int({ min: 180, max: 480 }),
        successRate: faker.number.int({ min: 90, max: 98 }),
        costAlert: {
          active: false,
          level: 'none',
          currentCost: 0,
          threshold: 0
        }
      },
      channels: generateMockChannels(5),
      recentVideos: generateMockVideos(10),
      chartData: generateMockChartData()
    },
    metadata: {
      timestamp: new Date(),
      requestId: faker.string.uuid(),
      cacheAge: 0
    }
  });
});

// Mock data generators
function generateMockChannels(count) {
  return Array.from({ length: count }, () => ({
    id: faker.string.uuid(),
    name: faker.company.name() + ' Channel',
    status: faker.helpers.arrayElement(['active', 'paused', 'error']),
    videoCount: faker.number.int({ min: 10, max: 100 }),
    todayVideos: faker.number.int({ min: 0, max: 5 }),
    revenue: faker.number.float({ min: 100, max: 1000, precision: 0.01 }),
    thumbnail: `https://picsum.photos/200/200?random=${faker.number.int({ min: 1, max: 100 })}`
  }));
}

function generateMockVideos(count) {
  return Array.from({ length: count }, () => ({
    id: faker.string.uuid(),
    channelId: faker.string.uuid(),
    channelName: faker.company.name() + ' Channel',
    title: faker.lorem.sentence(),
    status: faker.helpers.arrayElement(['queued', 'processing', 'completed', 'failed']),
    createdAt: faker.date.recent().toISOString(),
    cost: faker.number.float({ min: 0.30, max: 0.50, precision: 0.01 })
  }));
}

function generateMockChartData() {
  const days = 7;
  const now = new Date();
  
  return {
    revenueChart: Array.from({ length: days }, (_, i) => {
      const date = new Date(now);
      date.setDate(date.getDate() - (days - i - 1));
      return {
        timestamp: date.toISOString(),
        value: faker.number.float({ min: 50, max: 200, precision: 0.01 })
      };
    }),
    videoChart: Array.from({ length: days }, (_, i) => {
      const date = new Date(now);
      date.setDate(date.getDate() - (days - i - 1));
      return {
        timestamp: date.toISOString(),
        value: faker.number.int({ min: 5, max: 20 })
      };
    }),
    costChart: Array.from({ length: days }, (_, i) => {
      const date = new Date(now);
      date.setDate(date.getDate() - (days - i - 1));
      return {
        timestamp: date.toISOString(),
        value: faker.number.float({ min: 10, max: 50, precision: 0.01 })
      };
    })
  };
}

// WebSocket server
const wss = new WebSocket.Server({ port: 8001 });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Send initial connection message
  ws.send(JSON.stringify({
    type: 'connection',
    data: {
      status: 'connected',
      timestamp: new Date()
    }
  }));
  
  // Simulate events
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      // Randomly send different event types
      const eventType = faker.helpers.arrayElement([
        'video.completed',
        'video.failed',
        'cost.alert'
      ]);
      
      ws.send(JSON.stringify(generateMockEvent(eventType)));
    }
  }, 30000); // Every 30 seconds
  
  ws.on('close', () => {
    clearInterval(interval);
    console.log('WebSocket client disconnected');
  });
});

function generateMockEvent(type) {
  switch (type) {
    case 'video.completed':
      return {
        type: 'video.completed',
        data: {
          videoId: faker.string.uuid(),
          channelId: faker.string.uuid(),
          channelName: faker.company.name() + ' Channel',
          title: faker.lorem.sentence(),
          youtubeUrl: `https://youtube.com/watch?v=${faker.string.alphanumeric(11)}`,
          thumbnail: `https://picsum.photos/320/180?random=${faker.number.int({ min: 1, max: 100 })}`,
          cost: faker.number.float({ min: 0.30, max: 0.50, precision: 0.01 }),
          generationTime: faker.number.int({ min: 180, max: 480 }),
          timestamp: new Date()
        }
      };
      
    case 'video.failed':
      return {
        type: 'video.failed',
        data: {
          videoId: faker.string.uuid(),
          channelId: faker.string.uuid(),
          channelName: faker.company.name() + ' Channel',
          error: {
            stage: faker.helpers.arrayElement(['script', 'audio', 'video', 'upload']),
            code: 'VIDEO_GENERATION_FAILED',
            message: faker.lorem.sentence(),
            retryable: faker.datatype.boolean()
          },
          costIncurred: faker.number.float({ min: 0.10, max: 0.30, precision: 0.01 }),
          timestamp: new Date()
        }
      };
      
    case 'cost.alert':
      return {
        type: 'cost.alert',
        data: {
          alertId: faker.string.uuid(),
          level: faker.helpers.arrayElement(['warning', 'critical']),
          currentCost: faker.number.float({ min: 35, max: 50, precision: 0.01 }),
          threshold: 40,
          message: 'Daily cost limit approaching',
          period: 'daily',
          affectedChannels: [],
          suggestedAction: 'Consider pausing low-performing channels',
          remainingBudget: faker.number.float({ min: 5, max: 15, precision: 0.01 }),
          timestamp: new Date()
        }
      };
  }
}

// Start server
app.listen(PORT, () => {
  console.log(`Mock API server running at http://localhost:${PORT}`);
  console.log(`WebSocket server running at ws://localhost:8001`);
});
```

---

## 4. Design System & Assets

### 4.1 Design Tokens

```typescript
// src/theme/tokens.ts
export const designTokens = {
  // Colors
  colors: {
    // Primary - YouTube Red
    primary: {
      50: '#ffebee',
      100: '#ffcdd2',
      200: '#ef9a9a',
      300: '#e57373',
      400: '#ef5350',
      500: '#ff0000', // YouTube Red
      600: '#e53935',
      700: '#d32f2f',
      800: '#c62828',
      900: '#b71c1c',
    },
    
    // Secondary - Dark Blue
    secondary: {
      50: '#e3f2fd',
      100: '#bbdefb',
      200: '#90caf9',
      300: '#64b5f6',
      400: '#42a5f5',
      500: '#2196f3',
      600: '#1e88e5',
      700: '#1976d2',
      800: '#1565c0',
      900: '#0d47a1',
    },
    
    // Success - Green
    success: {
      50: '#e8f5e9',
      100: '#c8e6c9',
      200: '#a5d6a7',
      300: '#81c784',
      400: '#66bb6a',
      500: '#4caf50',
      600: '#43a047',
      700: '#388e3c',
      800: '#2e7d32',
      900: '#1b5e20',
    },
    
    // Warning - Orange
    warning: {
      50: '#fff3e0',
      100: '#ffe0b2',
      200: '#ffcc80',
      300: '#ffb74d',
      400: '#ffa726',
      500: '#ff9800',
      600: '#fb8c00',
      700: '#f57c00',
      800: '#ef6c00',
      900: '#e65100',
    },
    
    // Error - Red
    error: {
      50: '#ffebee',
      100: '#ffcdd2',
      200: '#ef9a9a',
      300: '#e57373',
      400: '#ef5350',
      500: '#f44336',
      600: '#e53935',
      700: '#d32f2f',
      800: '#c62828',
      900: '#b71c1c',
    },
    
    // Neutral - Grays
    neutral: {
      0: '#ffffff',
      50: '#fafafa',
      100: '#f5f5f5',
      200: '#eeeeee',
      300: '#e0e0e0',
      400: '#bdbdbd',
      500: '#9e9e9e',
      600: '#757575',
      700: '#616161',
      800: '#424242',
      900: '#212121',
      1000: '#000000',
    },
  },
  
  // Typography
  typography: {
    fontFamily: {
      primary: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      mono: '"Fira Code", "Consolas", "Monaco", monospace',
    },
    
    fontSize: {
      xs: '0.75rem',    // 12px
      sm: '0.875rem',   // 14px
      base: '1rem',     // 16px
      lg: '1.125rem',   // 18px
      xl: '1.25rem',    // 20px
      '2xl': '1.5rem',  // 24px
      '3xl': '1.875rem', // 30px
      '4xl': '2.25rem', // 36px
      '5xl': '3rem',    // 48px
    },
    
    fontWeight: {
      light: 300,
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    
    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.75,
    },
  },
  
  // Spacing
  spacing: {
    0: '0',
    1: '0.25rem',   // 4px
    2: '0.5rem',    // 8px
    3: '0.75rem',   // 12px
    4: '1rem',      // 16px
    5: '1.25rem',   // 20px
    6: '1.5rem',    // 24px
    8: '2rem',      // 32px
    10: '2.5rem',   // 40px
    12: '3rem',     // 48px
    16: '4rem',     // 64px
    20: '5rem',     // 80px
    24: '6rem',     // 96px
  },
  
  // Border Radius
  borderRadius: {
    none: '0',
    sm: '0.125rem',  // 2px
    base: '0.25rem', // 4px
    md: '0.375rem',  // 6px
    lg: '0.5rem',    // 8px
    xl: '0.75rem',   // 12px
    '2xl': '1rem',   // 16px
    full: '9999px',
  },
  
  // Shadows
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
    none: 'none',
  },
  
  // Transitions
  transitions: {
    fast: '150ms ease-in-out',
    base: '200ms ease-in-out',
    slow: '300ms ease-in-out',
    slower: '500ms ease-in-out',
  },
  
  // Z-index
  zIndex: {
    auto: 'auto',
    0: 0,
    10: 10,
    20: 20,
    30: 30,
    40: 40,
    50: 50,
    60: 60,    // Modals
    70: 70,    // Popovers
    80: 80,    // Tooltips
    90: 90,    // Notifications
    100: 100,  // Critical overlays
  },
};
```

### 4.2 MUI Theme Configuration

```typescript
// src/theme/mui-theme.ts
import { createTheme } from '@mui/material/styles';
import { designTokens } from './tokens';

export const muiTheme = createTheme({
  palette: {
    primary: {
      main: designTokens.colors.primary[500],
      light: designTokens.colors.primary[300],
      dark: designTokens.colors.primary[700],
    },
    secondary: {
      main: designTokens.colors.secondary[500],
      light: designTokens.colors.secondary[300],
      dark: designTokens.colors.secondary[700],
    },
    success: {
      main: designTokens.colors.success[500],
      light: designTokens.colors.success[300],
      dark: designTokens.colors.success[700],
    },
    warning: {
      main: designTokens.colors.warning[500],
      light: designTokens.colors.warning[300],
      dark: designTokens.colors.warning[700],
    },
    error: {
      main: designTokens.colors.error[500],
      light: designTokens.colors.error[300],
      dark: designTokens.colors.error[700],
    },
    grey: designTokens.colors.neutral,
    background: {
      default: designTokens.colors.neutral[50],
      paper: designTokens.colors.neutral[0],
    },
    text: {
      primary: designTokens.colors.neutral[900],
      secondary: designTokens.colors.neutral[600],
      disabled: designTokens.colors.neutral[400],
    },
  },
  
  typography: {
    fontFamily: designTokens.typography.fontFamily.primary,
    h1: {
      fontSize: designTokens.typography.fontSize['5xl'],
      fontWeight: designTokens.typography.fontWeight.bold,
      lineHeight: designTokens.typography.lineHeight.tight,
    },
    h2: {
      fontSize: designTokens.typography.fontSize['4xl'],
      fontWeight: designTokens.typography.fontWeight.bold,
      lineHeight: designTokens.typography.lineHeight.tight,
    },
    h3: {
      fontSize: designTokens.typography.fontSize['3xl'],
      fontWeight: designTokens.typography.fontWeight.semibold,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    h4: {
      fontSize: designTokens.typography.fontSize['2xl'],
      fontWeight: designTokens.typography.fontWeight.semibold,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    h5: {
      fontSize: designTokens.typography.fontSize.xl,
      fontWeight: designTokens.typography.fontWeight.medium,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    h6: {
      fontSize: designTokens.typography.fontSize.lg,
      fontWeight: designTokens.typography.fontWeight.medium,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    body1: {
      fontSize: designTokens.typography.fontSize.base,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    body2: {
      fontSize: designTokens.typography.fontSize.sm,
      lineHeight: designTokens.typography.lineHeight.normal,
    },
    button: {
      fontSize: designTokens.typography.fontSize.base,
      fontWeight: designTokens.typography.fontWeight.medium,
      textTransform: 'none', // Don't uppercase buttons
    },
  },
  
  shape: {
    borderRadius: parseInt(designTokens.borderRadius.md),
  },
  
  shadows: [
    'none',
    designTokens.shadows.sm,
    designTokens.shadows.base,
    designTokens.shadows.md,
    designTokens.shadows.lg,
    designTokens.shadows.xl,
    ...Array(19).fill(designTokens.shadows.xl), // Fill remaining shadows
  ],
  
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: designTokens.borderRadius.md,
          textTransform: 'none',
          fontWeight: designTokens.typography.fontWeight.medium,
          transition: designTokens.transitions.fast,
        },
        sizeLarge: {
          padding: `${designTokens.spacing[3]} ${designTokens.spacing[6]}`,
        },
        sizeMedium: {
          padding: `${designTokens.spacing[2]} ${designTokens.spacing[4]}`,
        },
        sizeSmall: {
          padding: `${designTokens.spacing[1]} ${designTokens.spacing[3]}`,
        },
      },
    },
    
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: designTokens.borderRadius.lg,
          boxShadow: designTokens.shadows.base,
          transition: designTokens.transitions.base,
          '&:hover': {
            boxShadow: designTokens.shadows.md,
          },
        },
      },
    },
    
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
        size: 'medium',
      },
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: designTokens.borderRadius.md,
          },
        },
      },
    },
    
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: designTokens.borderRadius.lg,
        },
        elevation1: {
          boxShadow: designTokens.shadows.sm,
        },
        elevation2: {
          boxShadow: designTokens.shadows.base,
        },
        elevation3: {
          boxShadow: designTokens.shadows.md,
        },
      },
    },
  },
});
```

### 4.3 Asset Structure

```bash
# Asset directory structure
src/assets/
├── images/
│   ├── logo/
│   │   ├── logo-light.svg
│   │   ├── logo-dark.svg
│   │   └── logo-icon.svg
│   ├── illustrations/
│   │   ├── empty-state.svg
│   │   ├── error-404.svg
│   │   ├── error-500.svg
│   │   └── success.svg
│   └── placeholders/
│       ├── channel-thumbnail.png
│       └── video-thumbnail.png
├── icons/
│   ├── custom/
│   │   ├── youtube.svg
│   │   ├── automation.svg
│   │   └── cost.svg
│   └── index.ts
└── fonts/
    ├── Inter-Regular.woff2
    ├── Inter-Medium.woff2
    ├── Inter-SemiBold.woff2
    └── Inter-Bold.woff2
```

### 4.4 Icon Management

```typescript
// src/assets/icons/index.ts
import YouTubeIcon from './custom/youtube.svg?react';
import AutomationIcon from './custom/automation.svg?react';
import CostIcon from './custom/cost.svg?react';

// Re-export Material Icons for consistency
export {
  Dashboard as DashboardIcon,
  VideoLibrary as VideoIcon,
  AttachMoney as RevenueIcon,
  TrendingUp as GrowthIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
  PlayCircle as PlayIcon,
  PauseCircle as PauseIcon,
  Settings as SettingsIcon,
  Notifications as NotificationIcon,
  Person as UserIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';

// Custom icons
export {
  YouTubeIcon,
  AutomationIcon,
  CostIcon,
};

// Icon wrapper for consistent sizing
interface IconProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
}

export const iconSizes = {
  small: 16,
  medium: 24,
  large: 32,
};

export const Icon: React.FC<IconProps & { icon: React.FC<any> }> = ({
  icon: IconComponent,
  size = 'medium',
  color,
}) => {
  return (
    <IconComponent
      style={{
        width: iconSizes[size],
        height: iconSizes[size],
        color,
      }}
    />
  );
};
```

---

## 5. Development Tools

### 5.1 Mock Data Generators

```typescript
// src/utils/mock-data.ts
import { faker } from '@faker-js/faker';

export const mockData = {
  generateChannel: (overrides?: Partial<Channel>): Channel => ({
    id: faker.string.uuid(),
    name: faker.company.name() + ' Channel',
    youtubeChannelId: 'UC' + faker.string.alphanumeric(22),
    niche: faker.helpers.arrayElement(['tech', 'gaming', 'education', 'lifestyle']),
    status: faker.helpers.arrayElement(['active', 'paused', 'error']),
    automationEnabled: faker.datatype.boolean(),
    settings: {
      dailyVideoLimit: faker.number.int({ min: 1, max: 3 }),
      targetAudience: faker.lorem.words(3),
      primaryLanguage: 'en',
      videoLength: faker.helpers.arrayElement(['short', 'medium', 'long']),
      uploadSchedule: {
        enabled: faker.datatype.boolean(),
        timezone: 'America/New_York',
        slots: ['09:00', '14:00', '19:00'],
      },
    },
    statistics: {
      totalVideos: faker.number.int({ min: 10, max: 1000 }),
      videosThisMonth: faker.number.int({ min: 1, max: 100 }),
      totalRevenue: faker.number.float({ min: 100, max: 10000, precision: 0.01 }),
      monthlyRevenue: faker.number.float({ min: 10, max: 1000, precision: 0.01 }),
      avgViews: faker.number.int({ min: 100, max: 10000 }),
      avgCTR: faker.number.float({ min: 1, max: 10, precision: 0.1 }),
      subscribers: faker.number.int({ min: 100, max: 100000 }),
    },
    costs: {
      totalSpent: faker.number.float({ min: 50, max: 500, precision: 0.01 }),
      monthlySpent: faker.number.float({ min: 10, max: 150, precision: 0.01 }),
      avgCostPerVideo: faker.number.float({ min: 0.30, max: 0.50, precision: 0.01 }),
      lastVideoCode: faker.number.float({ min: 0.30, max: 0.50, precision: 0.01 }),
    },
    metadata: {
      createdAt: faker.date.past().toISOString(),
      updatedAt: faker.date.recent().toISOString(),
      lastVideoAt: faker.date.recent().toISOString(),
      syncStatus: 'synced',
    },
    ...overrides,
  }),

  generateVideo: (overrides?: Partial<Video>): Video => ({
    id: faker.string.uuid(),
    channelId: faker.string.uuid(),
    channelName: faker.company.name() + ' Channel',
    title: faker.lorem.sentence(),
    description: faker.lorem.paragraph(),
    tags: faker.lorem.words(5).split(' '),
    status: faker.helpers.arrayElement(['queued', 'processing', 'completed', 'failed']),
    createdAt: faker.date.recent().toISOString(),
    ...overrides,
  }),

  generateDashboardMetrics: (): DashboardMetrics => ({
    videosProcessing: faker.number.int({ min: 0, max: 5 }),
    currentDailyCost: faker.number.float({ min: 10, max: 50, precision: 0.01 }),
    activeAlerts: [],
    channelMetrics: {
      total: 5,
      active: faker.number.int({ min: 3, max: 5 }),
      paused: faker.number.int({ min: 0, max: 2 }),
      error: 0,
    },
    videoMetrics: {
      today: faker.number.int({ min: 5, max: 15 }),
      thisWeek: faker.number.int({ min: 30, max: 100 }),
      thisMonth: faker.number.int({ min: 100, max: 450 }),
      allTime: faker.number.int({ min: 1000, max: 5000 }),
      successRate: faker.number.int({ min: 90, max: 98 }),
      avgGenerationTime: faker.number.int({ min: 180, max: 480 }),
    },
    financialMetrics: {
      revenueToday: faker.number.float({ min: 50, max: 200, precision: 0.01 }),
      revenueThisWeek: faker.number.float({ min: 300, max: 1000, precision: 0.01 }),
      revenueThisMonth: faker.number.float({ min: 1000, max: 5000, precision: 0.01 }),
      revenueAllTime: faker.number.float({ min: 10000, max: 50000, precision: 0.01 }),
      costToday: faker.number.float({ min: 10, max: 50, precision: 0.01 }),
      costThisMonth: faker.number.float({ min: 300, max: 1500, precision: 0.01 }),
      profitMargin: faker.number.float({ min: 60, max: 90, precision: 0.1 }),
      roi: faker.number.float({ min: 200, max: 500, precision: 0.1 }),
    },
    performanceMetrics: {
      automationRate: faker.number.int({ min: 85, max: 99 }),
      avgVideoQuality: faker.number.int({ min: 80, max: 95 }),
      avgViewsPerVideo: faker.number.int({ min: 1000, max: 10000 }),
      avgCTR: faker.number.float({ min: 3, max: 8, precision: 0.1 }),
      channelGrowthRate: faker.number.float({ min: 5, max: 25, precision: 0.1 }),
    },
  }),
};

// Delay utility for simulating API latency
export const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock API wrapper
export const mockApi = {
  async get<T>(generator: () => T, latency = 300): Promise<T> {
    await delay(latency);
    return generator();
  },

  async post<T>(data: any, generator: () => T, latency = 500): Promise<T> {
    await delay(latency);
    console.log('Mock POST:', data);
    return generator();
  },
};
```

### 5.2 Development Utilities

```typescript
// src/utils/dev-tools.ts
export const devTools = {
  // Performance monitoring
  measurePerformance: (name: string, fn: () => void) => {
    if (!config.features.performanceMonitor) return fn();
    
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    
    if (duration > config.performance.threshold) {
      console.warn(`Performance warning: ${name} took ${duration}ms`);
    }
    
    return result;
  },

  // Component render tracking
  trackRender: (componentName: string) => {
    if (!config.environment.isDevelopment) return;
    
    console.log(`[Render] ${componentName} at ${new Date().toISOString()}`);
  },

  // State debugging
  logStateChange: (storeName: string, prevState: any, newState: any) => {
    if (!config.environment.debug) return;
    
    console.groupCollapsed(`[State] ${storeName}`);
    console.log('Previous:', prevState);
    console.log('New:', newState);
    console.log('Diff:', diff(prevState, newState));
    console.groupEnd();
  },

  // API request logging
  logApiRequest: (method: string, url: string, data?: any) => {
    if (!config.environment.debug) return;
    
    console.groupCollapsed(`[API] ${method} ${url}`);
    if (data) console.log('Data:', data);
    console.trace();
    console.groupEnd();
  },

  // WebSocket event logging
  logWebSocketEvent: (event: any) => {
    if (!config.environment.debug) return;
    
    console.log(`[WS] ${event.type}:`, event.data);
  },
};

// React DevTools integration
if (config.environment.isDevelopment && config.features.devTools) {
  // Enable React DevTools
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    ...window.__REACT_DEVTOOLS_GLOBAL_HOOK__,
    onCommitFiberRoot: (id: any, root: any) => {
      // Custom commit logging
    },
  };
}
```

### 5.3 Testing Utilities

```typescript
// src/test/test-utils.tsx
import React from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { muiTheme } from '@/theme/mui-theme';

// Create a test query client
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      cacheTime: 0,
    },
  },
});

// Custom render function
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialRoute?: string;
  queryClient?: QueryClient;
}

export const renderWithProviders = (
  ui: React.ReactElement,
  {
    initialRoute = '/',
    queryClient = createTestQueryClient(),
    ...options
  }: CustomRenderOptions = {}
) => {
  window.history.pushState({}, 'Test page', initialRoute);

  const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={muiTheme}>
        <BrowserRouter>
          {children}
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );

  return {
    ...render(ui, { wrapper: Wrapper, ...options }),
    queryClient,
  };
};

// Re-export everything
export * from '@testing-library/react';
export { renderWithProviders as render };

// Mock data factories
export const factories = {
  user: (overrides = {}) => ({
    id: '123',
    email: 'test@example.com',
    name: 'Test User',
    role: 'user',
    ...overrides,
  }),

  channel: (overrides = {}) => ({
    id: '456',
    name: 'Test Channel',
    status: 'active',
    videoCount: 10,
    revenue: 100,
    ...overrides,
  }),
};
```

---

## 6. Git Hooks & Quality Checks

### 6.1 Husky Configuration

```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

npx lint-staged
```

```bash
# .husky/pre-push
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

npm run type-check
npm run test
```

### 6.2 Lint-staged Configuration

```json
// .lintstagedrc.json
{
  "*.{ts,tsx}": [
    "eslint --fix",
    "prettier --write"
  ],
  "*.{css,scss}": [
    "prettier --write"
  ],
  "*.{json,md}": [
    "prettier --write"
  ]
}
```

### 6.3 ESLint Configuration

```javascript
// .eslintrc.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:jsx-a11y/recommended',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2021,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
    project: './tsconfig.json',
  },
  plugins: ['react', '@typescript-eslint', 'jsx-a11y'],
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    'no-console': ['warn', { allow: ['warn', 'error'] }],
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};
```

### 6.4 Prettier Configuration

```json
// .prettierrc
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "arrowParens": "always",
  "endOfLine": "lf"
}
```

---

## 7. Performance Monitoring Setup

### 7.1 Web Vitals Monitoring

```typescript
// src/utils/performance.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export const initPerformanceMonitoring = () => {
  if (!config.features.performanceMonitor) return;

  // Core Web Vitals
  getCLS(console.log);  // Cumulative Layout Shift
  getFID(console.log);  // First Input Delay
  getFCP(console.log);  // First Contentful Paint
  getLCP(console.log);  // Largest Contentful Paint
  getTTFB(console.log); // Time to First Byte

  // Custom metrics
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      console.log(`[Performance] ${entry.name}: ${entry.duration}ms`);
    }
  });

  observer.observe({ entryTypes: ['measure'] });

  // API timing
  if ('PerformanceObserver' in window) {
    const apiObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.name.includes('/api/')) {
          const duration = entry.responseEnd - entry.startTime;
          if (duration > config.performance.threshold) {
            console.warn(`Slow API call: ${entry.name} took ${duration}ms`);
          }
        }
      });
    });

    apiObserver.observe({ entryTypes: ['resource'] });
  }
};
```

---

## Quick Start Checklist

### Initial Setup (One-time)
- [ ] Install Node.js 18.x LTS
- [ ] Clone repository
- [ ] Run `npm ci` to install dependencies
- [ ] Copy `.env.example` to `.env.local`
- [ ] Update `.env.local` with your values
- [ ] Run `npm run prepare` to setup git hooks
- [ ] Install recommended VS Code extensions

### Daily Development
- [ ] Pull latest changes: `git pull origin develop`
- [ ] Update dependencies: `npm ci`
- [ ] Start mock backend: `docker-compose -f docker-compose.dev.yml up`
- [ ] Start dev server: `npm run dev`
- [ ] Run tests in watch mode: `npm run test:watch`

### Before Committing
- [ ] Run type check: `npm run type-check`
- [ ] Run tests: `npm run test`
- [ ] Check bundle size: `npm run build:analyze`
- [ ] Verify no console.logs in code

### Troubleshooting
- **Port 3000 in use**: Kill process: `lsof -ti:3000 | xargs kill`
- **Mock API not working**: Check Docker is running
- **WebSocket connection failed**: Verify VITE_WS_URL in .env.local
- **Type errors**: Run `npm run type-check` for details

**Need Help?** Contact the Frontend Team Lead or check the architecture documentation.