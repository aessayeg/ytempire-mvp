# YTEMPIRE React Engineer Implementation Guide

**Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**For**: React Engineers (2 positions)  
**Document Type**: Technical Implementation Guide  
**Classification**: Internal - Engineering

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Component Development Standards](#2-component-development-standards)
3. [TypeScript Implementation](#3-typescript-implementation)
4. [State Management Architecture](#4-state-management-architecture)
5. [React Hooks Patterns](#5-react-hooks-patterns)
6. [Performance Optimization](#6-performance-optimization)
7. [Code Quality Standards](#7-code-quality-standards)
8. [Development Workflow](#8-development-workflow)

---

## 1. Executive Overview

### 1.1 Project Context

YTEMPIRE is an AI-powered YouTube automation platform enabling users to manage multiple YouTube channels with minimal manual intervention. As a React Engineer, you are responsible for building the user interface layer that makes complex automation accessible through intuitive interactions.

### 1.2 Technical Scope

```yaml
Platform Specifications:
  Framework: React 18.2.0 with TypeScript 5.3
  State Management: Zustand 4.4 (Critical: NO Redux)
  UI Library: Material-UI 5.14
  Charts: Recharts 2.10 (Critical: NO D3.js)
  Build Tool: Vite 5.0
  Testing: Vitest + React Testing Library
  Target Platform: Desktop (1280px minimum)
  Browser Support: Chrome 90+, Firefox 88+, Safari 14+
  
MVP Constraints:
  Timeline: 12 weeks to launch
  Components: 30-40 total (maximum)
  Screens: 20-25 total
  Users: 50 beta users
  Channels per User: 5 maximum
  Performance: <2 second page load
  Bundle Size: <1MB total
```

### 1.3 Team Structure & Your Role

```
Frontend Team (6 members)
├── Frontend Team Lead (Reports to CTO)
├── React Engineers (2) ← Your Position
│   ├── Core component development
│   ├── State management implementation
│   └── API integration layer
├── Dashboard Specialist (1)
│   └── Data visualization focus
└── UI/UX Designers (2)
    └── Design system & user research
```

### 1.4 Success Metrics

Your work directly impacts:
- **User Experience**: Page load <2 seconds, Time to Interactive <3 seconds
- **Code Quality**: 70% test coverage minimum, 0 critical bugs in production
- **Development Velocity**: 2-week sprint cycles, 85% sprint completion rate
- **Platform Reliability**: 99.9% uptime, <5% error rate

---

## 2. Component Development Standards

### 2.1 Component Architecture Philosophy

#### 2.1.1 Core Principles

```typescript
/**
 * YTEMPIRE Component Development Principles
 * 
 * 1. Composition over Inheritance
 * 2. Single Responsibility per Component
 * 3. Props Interface First Development
 * 4. Accessibility by Default
 * 5. Performance through Memoization
 */

// Example: Well-Structured Component Following All Principles
import React, { useState, useEffect, useMemo, useCallback, memo } from 'react';
import { Box, Card, Typography, Skeleton } from '@mui/material';
import { useChannelStore } from '@/stores/channelStore';
import { formatCurrency, formatDate } from '@/utils/formatters';
import type { Channel, ChannelMetrics } from '@/types';

// 1. Props Interface First - Define contract before implementation
interface ChannelCardProps {
  channel: Channel;
  variant?: 'default' | 'compact' | 'detailed';
  onSelect?: (channelId: string) => void;
  onToggleAutomation?: (channelId: string) => Promise<void>;
  showMetrics?: boolean;
  className?: string;
  testId?: string;
}

// 2. Component Implementation with Performance Optimization
export const ChannelCard = memo<ChannelCardProps>(({
  channel,
  variant = 'default',
  onSelect,
  onToggleAutomation,
  showMetrics = true,
  className,
  testId = 'channel-card'
}) => {
  // State Management - Minimal local state
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Store Integration - Selective subscriptions
  const updateChannel = useChannelStore(state => state.updateChannel);
  const channelMetrics = useChannelStore(
    state => state.metricsById[channel.id],
    // Shallow comparison for performance
    (prev, next) => prev?.updatedAt === next?.updatedAt
  );
  
  // Memoized Computations - Expensive calculations cached
  const formattedMetrics = useMemo(() => {
    if (!showMetrics || !channelMetrics) return null;
    
    return {
      revenue: formatCurrency(channelMetrics.revenue),
      views: channelMetrics.views.toLocaleString(),
      videos: channelMetrics.videoCount,
      lastUpdated: formatDate(channelMetrics.updatedAt)
    };
  }, [showMetrics, channelMetrics]);
  
  // Callbacks - Prevent unnecessary re-renders
  const handleToggleAutomation = useCallback(async () => {
    if (!onToggleAutomation || isProcessing) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      await onToggleAutomation(channel.id);
      updateChannel(channel.id, { 
        automationEnabled: !channel.automationEnabled 
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Operation failed';
      setError(errorMessage);
      console.error('Failed to toggle automation:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [channel.id, channel.automationEnabled, onToggleAutomation, isProcessing, updateChannel]);
  
  // Effect Management - Cleanup and dependencies
  useEffect(() => {
    // Poll for metrics updates every 60 seconds (MVP requirement)
    const pollInterval = setInterval(() => {
      if (channel.status === 'active') {
        // Trigger metrics refresh via store action
        useChannelStore.getState().refreshMetrics(channel.id);
      }
    }, 60000);
    
    return () => clearInterval(pollInterval);
  }, [channel.id, channel.status]);
  
  // Render Logic - Clear component structure
  return (
    <Card 
      className={className}
      data-testid={testId}
      sx={{ 
        p: 2,
        cursor: onSelect ? 'pointer' : 'default',
        opacity: isProcessing ? 0.7 : 1,
        transition: 'opacity 0.2s'
      }}
      onClick={() => onSelect?.(channel.id)}
    >
      <ChannelHeader channel={channel} variant={variant} />
      
      {showMetrics && formattedMetrics && (
        <ChannelMetrics metrics={formattedMetrics} />
      )}
      
      {variant === 'detailed' && (
        <ChannelActions 
          channel={channel}
          onToggle={handleToggleAutomation}
          disabled={isProcessing}
        />
      )}
      
      {error && (
        <ErrorMessage message={error} onDismiss={() => setError(null)} />
      )}
    </Card>
  );
}, 
// Memo comparison function for performance
(prevProps, nextProps) => {
  return (
    prevProps.channel.id === nextProps.channel.id &&
    prevProps.channel.updatedAt === nextProps.channel.updatedAt &&
    prevProps.variant === nextProps.variant &&
    prevProps.showMetrics === nextProps.showMetrics
  );
});

ChannelCard.displayName = 'ChannelCard';
```

#### 2.1.2 Component Hierarchy & Organization

```typescript
/**
 * Component Organization Structure
 * Maximum 40 components for MVP (Current: 35)
 */

// src/components/index.ts - Central export registry
export { AppLayout } from './layout/AppLayout';
export { ChannelCard } from './channels/ChannelCard';
export { VideoQueue } from './videos/VideoQueue';
// ... all component exports

// File Structure Pattern
src/
├── components/
│   ├── layout/                 # Layout Components (5)
│   │   ├── AppLayout/
│   │   │   ├── index.ts
│   │   │   ├── AppLayout.tsx
│   │   │   ├── AppLayout.test.tsx
│   │   │   └── AppLayout.types.ts
│   │   ├── Header/
│   │   ├── Sidebar/
│   │   ├── Footer/
│   │   └── Container/
│   │
│   ├── common/                 # Shared Components (10)
│   │   ├── Button/
│   │   ├── Input/
│   │   ├── Modal/
│   │   ├── Card/
│   │   ├── Table/
│   │   ├── LoadingSpinner/
│   │   ├── ErrorBoundary/
│   │   ├── Toast/
│   │   ├── Tooltip/
│   │   └── Select/
│   │
│   ├── features/               # Business Components (15)
│   │   ├── channels/
│   │   │   ├── ChannelCard/
│   │   │   ├── ChannelList/
│   │   │   └── ChannelSelector/
│   │   ├── videos/
│   │   │   ├── VideoRow/
│   │   │   ├── VideoQueue/
│   │   │   └── GenerateButton/
│   │   ├── metrics/
│   │   │   ├── MetricCard/
│   │   │   ├── CostBreakdown/
│   │   │   └── PerformanceMetrics/
│   │   └── settings/
│   │       ├── UserSettings/
│   │       └── ApiKeyManager/
│   │
│   └── charts/                 # Recharts Wrappers (5)
│       ├── LineChart/
│       ├── BarChart/
│       ├── PieChart/
│       ├── AreaChart/
│       └── MetricChart/
```

### 2.2 Component Implementation Patterns

#### 2.2.1 Loading States Pattern

```typescript
/**
 * Standardized Loading State Implementation
 * Consistent UX across all data-fetching components
 */

interface LoadingStateProps {
  isLoading: boolean;
  error: Error | null;
  isEmpty: boolean;
  retry?: () => void;
  children: React.ReactNode;
  emptyMessage?: string;
  loadingComponent?: React.ComponentType;
  errorComponent?: React.ComponentType<{ error: Error; retry?: () => void }>;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  isLoading,
  error,
  isEmpty,
  retry,
  children,
  emptyMessage = 'No data available',
  loadingComponent: LoadingComponent = DefaultLoadingComponent,
  errorComponent: ErrorComponent = DefaultErrorComponent
}) => {
  // Error state takes precedence
  if (error) {
    return <ErrorComponent error={error} retry={retry} />;
  }
  
  // Loading state
  if (isLoading) {
    return <LoadingComponent />;
  }
  
  // Empty state
  if (isEmpty) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography color="text.secondary">{emptyMessage}</Typography>
      </Box>
    );
  }
  
  // Success state with data
  return <>{children}</>;
};

// Usage Example
const ChannelList: React.FC = () => {
  const { channels, loading, error, refetch } = useChannelStore();
  
  return (
    <LoadingState
      isLoading={loading}
      error={error}
      isEmpty={channels.length === 0}
      retry={refetch}
      emptyMessage="No channels created yet. Create your first channel to get started!"
    >
      <Grid container spacing={2}>
        {channels.map(channel => (
          <Grid item xs={12} md={6} lg={4} key={channel.id}>
            <ChannelCard channel={channel} />
          </Grid>
        ))}
      </Grid>
    </LoadingState>
  );
};
```

#### 2.2.2 Error Boundary Pattern

```typescript
/**
 * Error Boundary Implementation
 * The ONLY class component allowed in the codebase
 */

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  isolate?: boolean; // If true, only affects this component tree
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state to trigger fallback UI
    return {
      hasError: true,
      error
    };
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to monitoring service
    console.error('ErrorBoundary caught:', error, errorInfo);
    
    // Send to Sentry or monitoring service
    if (typeof window !== 'undefined' && window.Sentry) {
      window.Sentry.captureException(error, {
        contexts: {
          react: {
            componentStack: errorInfo.componentStack
          }
        }
      });
    }
    
    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);
    
    // Store error info for display
    this.setState({ errorInfo });
  }
  
  resetError = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };
  
  render(): React.ReactNode {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      
      return (
        <FallbackComponent 
          error={this.state.error} 
          resetError={this.resetError}
        />
      );
    }
    
    return this.props.children;
  }
}

// Default Error Fallback Component
const DefaultErrorFallback: React.FC<{ error: Error; resetError: () => void }> = ({ 
  error, 
  resetError 
}) => (
  <Card sx={{ p: 3, textAlign: 'center' }}>
    <Typography variant="h6" color="error" gutterBottom>
      Something went wrong
    </Typography>
    <Typography variant="body2" color="text.secondary" paragraph>
      {error.message}
    </Typography>
    <Button variant="contained" onClick={resetError}>
      Try Again
    </Button>
  </Card>
);

// Usage: Wrap critical component trees
const App: React.FC = () => (
  <ErrorBoundary>
    <AppLayout>
      <ErrorBoundary isolate>
        <Dashboard />
      </ErrorBoundary>
    </AppLayout>
  </ErrorBoundary>
);
```

---

## 3. TypeScript Implementation

### 3.1 TypeScript Configuration

#### 3.1.1 Project Configuration

```json
// tsconfig.json - Strict TypeScript configuration for type safety
{
  "compilerOptions": {
    // Language and Environment
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    
    // Modules
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowImportingTsExtensions": true,
    
    // JavaScript Support
    "allowJs": false,
    "checkJs": false,
    
    // Emit
    "noEmit": true,
    "sourceMap": true,
    "removeComments": true,
    
    // Interop Constraints
    "isolatedModules": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    
    // Type Checking - STRICT MODE ENABLED
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    "allowUnusedLabels": false,
    "allowUnreachableCode": false,
    
    // Advanced
    "stripInternal": true,
    "preserveConstEnums": true,
    "declaration": true,
    "declarationMap": true,
    
    // Path Mapping
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@/components": ["src/components/index"],
      "@/stores": ["src/stores/index"],
      "@/hooks": ["src/hooks/index"],
      "@/utils": ["src/utils/index"],
      "@/types": ["src/types/index"],
      "@/services": ["src/services/index"],
      "@/constants": ["src/constants/index"],
      "@/assets": ["src/assets/*"]
    }
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.tsx",
    "src/**/*.d.ts"
  ],
  "exclude": [
    "node_modules",
    "dist",
    "build",
    "coverage",
    "**/*.spec.ts",
    "**/*.test.ts",
    "**/*.spec.tsx",
    "**/*.test.tsx"
  ],
  "references": [
    { "path": "./tsconfig.node.json" }
  ]
}
```

### 3.2 Type System Architecture

#### 3.2.1 Core Domain Types

```typescript
// src/types/domain.types.ts - Business domain type definitions

/**
 * User Domain Types
 */
export interface User {
  readonly id: string;
  email: string;
  role: UserRole;
  subscription: SubscriptionTier;
  channelLimit: number; // Max 5 for MVP
  preferences: UserPreferences;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  BETA_TESTER = 'beta_tester'
}

export enum SubscriptionTier {
  FREE = 'free',
  STARTER = 'starter',
  PROFESSIONAL = 'professional',
  ENTERPRISE = 'enterprise'
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  notifications: NotificationPreferences;
  dashboard: DashboardPreferences;
}

/**
 * Channel Domain Types
 */
export interface Channel {
  readonly id: string;
  name: string;
  youtubeChannelId: string;
  niche: ChannelNiche;
  status: ChannelStatus;
  automationEnabled: boolean;
  settings: ChannelSettings;
  metrics: ChannelMetrics;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export type ChannelStatus = 'active' | 'paused' | 'error' | 'suspended';

export interface ChannelSettings {
  dailyVideoLimit: number; // Max 3 for MVP
  publishingSchedule: PublishingSchedule;
  contentPreferences: ContentPreferences;
  monetization: MonetizationSettings;
}

export interface ChannelMetrics {
  videoCount: number;
  totalViews: number;
  subscribers: number;
  revenue: MonetaryAmount;
  averageVideoPerformance: VideoPerformanceMetrics;
  readonly lastUpdated: Date;
}

/**
 * Video Domain Types
 */
export interface Video {
  readonly id: string;
  channelId: string;
  title: string;
  description: string;
  tags: string[];
  status: VideoStatus;
  stage: VideoGenerationStage;
  progress: number; // 0-100
  cost: VideoCost;
  performance?: VideoPerformanceMetrics;
  readonly generatedAt: Date;
  publishedAt?: Date;
  youtubeVideoId?: string;
}

export type VideoStatus = 
  | 'queued'
  | 'processing'
  | 'completed'
  | 'published'
  | 'failed'
  | 'cancelled';

export type VideoGenerationStage = 
  | 'initialization'
  | 'script_generation'
  | 'voice_synthesis'
  | 'video_assembly'
  | 'thumbnail_creation'
  | 'quality_check'
  | 'uploading'
  | 'published';

export interface VideoCost {
  scriptGeneration: number;
  voiceSynthesis: number;
  videoRendering: number;
  thumbnailGeneration: number;
  total: number;
  currency: CurrencyCode;
}

/**
 * Utility Types
 */
export interface MonetaryAmount {
  value: number;
  currency: CurrencyCode;
  formatted?: string;
}

export type CurrencyCode = 'USD' | 'EUR' | 'GBP';

export interface DateRange {
  start: Date;
  end: Date;
}

export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}
```

#### 3.2.2 API Types

```typescript
// src/types/api.types.ts - API communication types

/**
 * Base API Types
 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata: ResponseMetadata;
}

export interface ApiError {
  code: ErrorCode;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
  traceId: string;
}

export interface ResponseMetadata {
  timestamp: string;
  requestId: string;
  processingTime: number;
  version: string;
}

/**
 * Request Types
 */
export interface CreateChannelRequest {
  name: string;
  niche: string;
  youtubeChannelId?: string;
  settings?: Partial<ChannelSettings>;
}

export interface UpdateChannelRequest {
  name?: string;
  status?: ChannelStatus;
  automationEnabled?: boolean;
  settings?: Partial<ChannelSettings>;
}

export interface GenerateVideoRequest {
  channelId: string;
  topic?: string;
  style: VideoStyle;
  length: VideoLength;
  priority: number; // 1-10
  scheduledFor?: Date;
}

export type VideoStyle = 'educational' | 'entertainment' | 'tutorial' | 'news';
export type VideoLength = 'short' | 'medium' | 'long';

/**
 * Response Types
 */
export interface DashboardOverviewResponse {
  metrics: GlobalMetrics;
  channels: ChannelSummary[];
  recentVideos: VideoSummary[];
  costBreakdown: CostBreakdown;
  alerts: SystemAlert[];
}

export interface GlobalMetrics {
  totalChannels: number;
  activeChannels: number;
  videosToday: number;
  videosProcessing: number;
  totalRevenue: MonetaryAmount;
  totalCost: MonetaryAmount;
  profitMargin: number;
  automationPercentage: number;
}

/**
 * WebSocket Event Types
 */
export interface WebSocketEvent<T = unknown> {
  type: WebSocketEventType;
  payload: T;
  timestamp: string;
  correlationId?: string;
}

export enum WebSocketEventType {
  // Critical events only for MVP
  VIDEO_COMPLETED = 'video.completed',
  VIDEO_FAILED = 'video.failed',
  COST_ALERT = 'cost.alert'
}

export interface VideoCompletedPayload {
  videoId: string;
  channelId: string;
  title: string;
  youtubeUrl: string;
  cost: VideoCost;
}

export interface VideoFailedPayload {
  videoId: string;
  channelId: string;
  error: string;
  stage: VideoGenerationStage;
  canRetry: boolean;
}

export interface CostAlertPayload {
  type: 'daily_limit' | 'per_video' | 'monthly_projection';
  current: number;
  limit: number;
  message: string;
}
```

#### 3.2.3 Advanced TypeScript Patterns

```typescript
// src/types/utility.types.ts - Advanced type utilities

/**
 * Branded Types for Type Safety
 * Prevents mixing up similar primitive types
 */
export type Brand<K, T> = K & { __brand: T };

export type UserId = Brand<string, 'UserId'>;
export type ChannelId = Brand<string, 'ChannelId'>;
export type VideoId = Brand<string, 'VideoId'>;

// Type-safe ID creators
export const createUserId = (id: string): UserId => id as UserId;
export const createChannelId = (id: string): ChannelId => id as ChannelId;
export const createVideoId = (id: string): VideoId => id as VideoId;

/**
 * Discriminated Unions for State Management
 */
export type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: Error };

// Type guards for discriminated unions
export const isIdle = <T>(state: AsyncState<T>): state is { status: 'idle' } =>
  state.status === 'idle';

export const isLoading = <T>(state: AsyncState<T>): state is { status: 'loading' } =>
  state.status === 'loading';

export const isSuccess = <T>(state: AsyncState<T>): state is { status: 'success'; data: T } =>
  state.status === 'success';

export const isError = <T>(state: AsyncState<T>): state is { status: 'error'; error: Error } =>
  state.status === 'error';

/**
 * Template Literal Types for API Endpoints
 */
export type ApiEndpoint = 
  | `/api/v1/channels`
  | `/api/v1/channels/${string}`
  | `/api/v1/videos`
  | `/api/v1/videos/${string}`
  | `/api/v1/dashboard/overview`
  | `/api/v1/metrics/${string}`;

/**
 * Conditional Types for Feature Flags
 */
export type FeatureFlag<T extends boolean> = T extends true
  ? { enabled: true; config: FeatureConfig }
  : { enabled: false };

export interface FeatureConfig {
  rolloutPercentage: number;
  enabledForUsers: UserId[];
  metadata: Record<string, unknown>;
}

/**
 * Mapped Types for Form Handling
 */
export type FormErrors<T> = {
  [K in keyof T]?: string;
};

export type FormTouched<T> = {
  [K in keyof T]?: boolean;
};

export type FormState<T> = {
  values: T;
  errors: FormErrors<T>;
  touched: FormTouched<T>;
  isSubmitting: boolean;
  isValid: boolean;
};

/**
 * Utility Types
 */
export type DeepPartial<T> = T extends object
  ? { [P in keyof T]?: DeepPartial<T[P]> }
  : T;

export type DeepReadonly<T> = T extends object
  ? { readonly [P in keyof T]: DeepReadonly<T[P]> }
  : T;

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type Maybe<T> = T | null | undefined;

export type AsyncFunction<T = void> = () => Promise<T>;
export type VoidFunction = () => void;

/**
 * Extract Types from Arrays
 */
export type ArrayElement<ArrayType extends readonly unknown[]> = 
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;

// Usage: type MyType = ArrayElement<typeof myArray>;
```

---

## 4. State Management Architecture

### 4.1 Zustand Store Implementation

#### 4.1.1 Store Architecture

```typescript
// src/stores/architecture.ts - Zustand store architecture

/**
 * CRITICAL: We use Zustand, NOT Redux for state management
 * This decision is final for MVP
 */

import { create, StateCreator } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

/**
 * Store Slice Pattern for Modular State Management
 */

// Channel Slice
export interface ChannelSlice {
  // State
  channels: Channel[];
  activeChannelId: ChannelId | null;
  channelsLoading: boolean;
  channelsError: Error | null;
  
  // Actions
  fetchChannels: () => Promise<void>;
  createChannel: (data: CreateChannelRequest) => Promise<Channel>;
  updateChannel: (id: ChannelId, updates: Partial<Channel>) => Promise<void>;
  deleteChannel: (id: ChannelId) => Promise<void>;
  setActiveChannel: (id: ChannelId | null) => void;
  
  // Selectors (computed values)
  getActiveChannel: () => Channel | undefined;
  getChannelById: (id: ChannelId) => Channel | undefined;
  getActiveChannelsCount: () => number;
}

// Video Slice
export interface VideoSlice {
  // State
  videos: Video[];
  videoQueue: Video[];
  videosLoading: boolean;
  videosError: Error | null;
  
  // Actions
  fetchVideos: (channelId?: ChannelId) => Promise<void>;
  generateVideo: (request: GenerateVideoRequest) => Promise<Video>;
  cancelVideo: (videoId: VideoId) => Promise<void>;
  retryVideo: (videoId: VideoId) => Promise<void>;
  
  // Selectors
  getVideosByChannel: (channelId: ChannelId) => Video[];
  getProcessingVideos: () => Video[];
  getVideoById: (id: VideoId) => Video | undefined;
}

// Metrics Slice
export interface MetricsSlice {
  // State
  globalMetrics: GlobalMetrics | null;
  channelMetrics: Map<ChannelId, ChannelMetrics>;
  costBreakdown: CostBreakdown | null;
  metricsLoading: boolean;
  
  // Actions
  fetchGlobalMetrics: () => Promise<void>;
  fetchChannelMetrics: (channelId: ChannelId) => Promise<void>;
  updateCostBreakdown: (costs: CostBreakdown) => void;
  
  // Selectors
  getChannelMetrics: (channelId: ChannelId) => ChannelMetrics | undefined;
  getTotalCost: () => number;
  getProfitMargin: () => number;
}

// Combined Store Type
export type AppStore = ChannelSlice & VideoSlice & MetricsSlice;
```

#### 4.1.2 Store Implementation

```typescript
// src/stores/appStore.ts - Main application store

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { api } from '@/services/api';

/**
 * Channel Slice Implementation
 */
const createChannelSlice: StateCreator<
  AppStore,
  [['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/immer', never]],
  [],
  ChannelSlice
> = (set, get) => ({
  // Initial State
  channels: [],
  activeChannelId: null,
  channelsLoading: false,
  channelsError: null,
  
  // Actions
  fetchChannels: async () => {
    set(state => {
      state.channelsLoading = true;
      state.channelsError = null;
    });
    
    try {
      const response = await api.get<ApiResponse<Channel[]>>('/api/v1/channels');
      
      if (response.data.success && response.data.data) {
        set(state => {
          state.channels = response.data.data!;
          state.channelsLoading = false;
        });
      }
    } catch (error) {
      set(state => {
        state.channelsError = error as Error;
        state.channelsLoading = false;
      });
      throw error;
    }
  },
  
  createChannel: async (data) => {
    // Check MVP limit
    if (get().channels.length >= 5) {
      throw new Error('Channel limit reached. Maximum 5 channels allowed in MVP.');
    }
    
    set(state => {
      state.channelsLoading = true;
    });
    
    try {
      const response = await api.post<ApiResponse<Channel>>(
        '/api/v1/channels',
        data
      );
      
      if (response.data.success && response.data.data) {
        const newChannel = response.data.data;
        
        set(state => {
          state.channels.push(newChannel);
          state.channelsLoading = false;
        });
        
        return newChannel;
      }
      
      throw new Error('Failed to create channel');
    } catch (error) {
      set(state => {
        state.channelsError = error as Error;
        state.channelsLoading = false;
      });
      throw error;
    }
  },
  
  updateChannel: async (id, updates) => {
    try {
      const response = await api.patch<ApiResponse<Channel>>(
        `/api/v1/channels/${id}`,
        updates
      );
      
      if (response.data.success && response.data.data) {
        set(state => {
          const index = state.channels.findIndex(ch => ch.id === id);
          if (index !== -1) {
            state.channels[index] = { ...state.channels[index], ...updates };
          }
        });
      }
    } catch (error) {
      set(state => {
        state.channelsError = error as Error;
      });
      throw error;
    }
  },
  
  deleteChannel: async (id) => {
    try {
      await api.delete(`/api/v1/channels/${id}`);
      
      set(state => {
        state.channels = state.channels.filter(ch => ch.id !== id);
        if (state.activeChannelId === id) {
          state.activeChannelId = null;
        }
      });
    } catch (error) {
      set(state => {
        state.channelsError = error as Error;
      });
      throw error;
    }
  },
  
  setActiveChannel: (id) => {
    set(state => {
      state.activeChannelId = id;
    });
  },
  
  // Selectors
  getActiveChannel: () => {
    const state = get();
    return state.channels.find(ch => ch.id === state.activeChannelId);
  },
  
  getChannelById: (id) => {
    return get().channels.find(ch => ch.id === id);
  },
  
  getActiveChannelsCount: () => {
    return get().channels.filter(ch => ch.status === 'active').length;
  }
});

/**
 * Video Slice Implementation
 */
const createVideoSlice: StateCreator<
  AppStore,
  [['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/immer', never]],
  [],
  VideoSlice
> = (set, get) => ({
  // Initial State
  videos: [],
  videoQueue: [],
  videosLoading: false,
  videosError: null,
  
  // Actions
  fetchVideos: async (channelId?) => {
    set(state => {
      state.videosLoading = true;
      state.videosError = null;
    });
    
    try {
      const endpoint = channelId 
        ? `/api/v1/channels/${channelId}/videos`
        : '/api/v1/videos';
        
      const response = await api.get<ApiResponse<Video[]>>(endpoint);
      
      if (response.data.success && response.data.data) {
        set(state => {
          state.videos = response.data.data!;
          state.videosLoading = false;
        });
      }
    } catch (error) {
      set(state => {
        state.videosError = error as Error;
        state.videosLoading = false;
      });
    }
  },
  
  generateVideo: async (request) => {
    try {
      const response = await api.post<ApiResponse<Video>>(
        '/api/v1/videos/generate',
        request
      );
      
      if (response.data.success && response.data.data) {
        const newVideo = response.data.data;
        
        set(state => {
          state.videos.push(newVideo);
          state.videoQueue.push(newVideo);
        });
        
        return newVideo;
      }
      
      throw new Error('Failed to generate video');
    } catch (error) {
      set(state => {
        state.videosError = error as Error;
      });
      throw error;
    }
  },
  
  cancelVideo: async (videoId) => {
    try {
      await api.post(`/api/v1/videos/${videoId}/cancel`);
      
      set(state => {
        const video = state.videos.find(v => v.id === videoId);
        if (video) {
          video.status = 'cancelled';
        }
        state.videoQueue = state.videoQueue.filter(v => v.id !== videoId);
      });
    } catch (error) {
      set(state => {
        state.videosError = error as Error;
      });
      throw error;
    }
  },
  
  retryVideo: async (videoId) => {
    try {
      const response = await api.post<ApiResponse<Video>>(
        `/api/v1/videos/${videoId}/retry`
      );
      
      if (response.data.success && response.data.data) {
        set(state => {
          const index = state.videos.findIndex(v => v.id === videoId);
          if (index !== -1) {
            state.videos[index] = response.data.data!;
          }
          state.videoQueue.push(response.data.data!);
        });
      }
    } catch (error) {
      set(state => {
        state.videosError = error as Error;
      });
      throw error;
    }
  },
  
  // Selectors
  getVideosByChannel: (channelId) => {
    return get().videos.filter(v => v.channelId === channelId);
  },
  
  getProcessingVideos: () => {
    return get().videos.filter(v => v.status === 'processing');
  },
  
  getVideoById: (id) => {
    return get().videos.find(v => v.id === id);
  }
});

/**
 * Combined Store with All Slices
 */
export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      immer((...a) => ({
        ...createChannelSlice(...a),
        ...createVideoSlice(...a),
        // Add metrics slice here
      })),
      {
        name: 'ytempire-storage',
        partialize: (state) => ({
          // Only persist non-sensitive data
          activeChannelId: state.activeChannelId,
          // Don't persist: tokens, sensitive data, temporary UI state
        })
      }
    ),
    {
      name: 'YTEmpire Store'
    }
  )
);
```

---

## 5. React Hooks Patterns

### 5.1 Custom Hooks Library

#### 5.1.1 Data Management Hooks

```typescript
// src/hooks/useDataManagement.ts - Data fetching and caching hooks

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/services/api';

/**
 * Polling Hook with Automatic Cleanup
 * Used for dashboard metrics that update every 60 seconds
 */
export function usePolling<T>(
  fetchFn: () => Promise<T>,
  interval: number = 60000,
  options?: {
    enabled?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: Error) => void;
    retryOnError?: boolean;
  }
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const intervalRef = useRef<NodeJS.Timeout>();
  
  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await fetchFn();
      setData(result);
      options?.onSuccess?.(result);
    } catch (err) {
      const error = err as Error;
      setError(error);
      options?.onError?.(error);
      
      if (!options?.retryOnError) {
        // Stop polling on error unless retry is enabled
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      }
    } finally {
      setLoading(false);
    }
  }, [fetchFn, options]);
  
  useEffect(() => {
    if (options?.enabled === false) return;
    
    // Initial fetch
    fetch();
    
    // Set up polling
    intervalRef.current = setInterval(fetch, interval);
    
    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetch, interval, options?.enabled]);
  
  return {
    data,
    loading,
    error,
    refetch: fetch
  };
}

/**
 * Optimistic Update Hook
 * Updates UI immediately, rolls back on error
 */
export function useOptimisticUpdate<T>(
  initialData: T,
  updateFn: (newData: T) => Promise<T>
) {
  const [data, setData] = useState(initialData);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const previousDataRef = useRef<T>(initialData);
  
  const update = useCallback(async (newData: T) => {
    // Store current data for rollback
    previousDataRef.current = data;
    
    // Optimistic update
    setData(newData);
    setIsUpdating(true);
    setError(null);
    
    try {
      const result = await updateFn(newData);
      setData(result); // Update with server response
      return result;
    } catch (err) {
      // Rollback on error
      setData(previousDataRef.current);
      setError(err as Error);
      throw err;
    } finally {
      setIsUpdating(false);
    }
  }, [data, updateFn]);
  
  return {
    data,
    update,
    isUpdating,
    error,
    reset: () => setData(initialData)
  };
}

/**
 * Infinite Scroll Hook
 * For paginated lists
 */
export function useInfiniteScroll<T>(
  fetchFn: (page: number) => Promise<{ data: T[]; hasMore: boolean }>,
  options?: {
    initialPage?: number;
    threshold?: number;
  }
) {
  const [items, setItems] = useState<T[]>([]);
  const [page, setPage] = useState(options?.initialPage || 1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  const observerRef = useRef<IntersectionObserver>();
  const lastElementRef = useCallback((node: HTMLElement | null) => {
    if (loading) return;
    
    if (observerRef.current) {
      observerRef.current.disconnect();
    }
    
    observerRef.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMore) {
        loadMore();
      }
    }, {
      threshold: options?.threshold || 0.1
    });
    
    if (node) {
      observerRef.current.observe(node);
    }
  }, [loading, hasMore]);
  
  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await fetchFn(page);
      setItems(prev => [...prev, ...result.data]);
      setHasMore(result.hasMore);
      setPage(prev => prev + 1);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [page, loading, hasMore, fetchFn]);
  
  useEffect(() => {
    loadMore(); // Load initial data
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  
  return {
    items,
    loading,
    error,
    hasMore,
    lastElementRef,
    refresh: () => {
      setItems([]);
      setPage(options?.initialPage || 1);
      setHasMore(true);
      loadMore();
    }
  };
}
```

#### 5.1.2 UI Enhancement Hooks

```typescript
// src/hooks/useUIEnhancements.ts - UI and UX enhancement hooks

/**
 * Debounce Hook
 * Delays value updates for search inputs
 */
export function useDebounce<T>(value: T, delay: number = 500): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [value, delay]);
  
  return debouncedValue;
}

/**
 * Click Outside Hook
 * Detects clicks outside of element
 */
export function useClickOutside(
  ref: RefObject<HTMLElement>,
  handler: (event: MouseEvent | TouchEvent) => void,
  enabled: boolean = true
) {
  useEffect(() => {
    if (!enabled) return;
    
    const listener = (event: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return;
      }
      handler(event);
    };
    
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);
    
    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler, enabled]);
}

/**
 * Keyboard Shortcut Hook
 * Registers keyboard shortcuts
 */
export function useKeyboardShortcut(
  key: string,
  callback: (event: KeyboardEvent) => void,
  options?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    enabled?: boolean;
  }
) {
  useEffect(() => {
    if (options?.enabled === false) return;
    
    const handler = (event: KeyboardEvent) => {
      if (
        event.key === key &&
        (!options?.ctrl || event.ctrlKey) &&
        (!options?.alt || event.altKey) &&
        (!options?.shift || event.shiftKey)
      ) {
        event.preventDefault();
        callback(event);
      }
    };
    
    window.addEventListener('keydown', handler);
    
    return () => {
      window.removeEventListener('keydown', handler);
    };
  }, [key, callback, options]);
}

/**
 * Copy to Clipboard Hook
 * Handles clipboard operations with feedback
 */
export function useClipboard(timeout: number = 2000) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setError(null);
      
      setTimeout(() => {
        setCopied(false);
      }, timeout);
      
      return true;
    } catch (err) {
      setError(err as Error);
      setCopied(false);
      return false;
    }
  }, [timeout]);
  
  return { copy, copied, error };
}
```

---

## 6. Performance Optimization

### 6.1 Code Splitting Strategy

```typescript
// src/routes/AppRoutes.tsx - Route-based code splitting

import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { LoadingScreen } from '@/components/common/LoadingScreen';

// Lazy load all route components
const Dashboard = lazy(() => 
  import(/* webpackChunkName: "dashboard" */ '@/pages/Dashboard')
);

const Channels = lazy(() => 
  import(/* webpackChunkName: "channels" */ '@/pages/Channels')
);

const Videos = lazy(() => 
  import(/* webpackChunkName: "videos" */ '@/pages/Videos')
);

const Analytics = lazy(() => 
  import(/* webpackChunkName: "analytics" */ '@/pages/Analytics')
);

const Settings = lazy(() => 
  import(/* webpackChunkName: "settings" */ '@/pages/Settings')
);

export const AppRoutes = () => {
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

### 6.2 Performance Monitoring

```typescript
// src/utils/performance.ts - Performance monitoring utilities

class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();
  
  /**
   * Measure component render time
   */
  measureRender(componentName: string, callback: () => void) {
    const startTime = performance.now();
    callback();
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Store metric
    if (!this.metrics.has(componentName)) {
      this.metrics.set(componentName, []);
    }
    this.metrics.get(componentName)!.push(duration);
    
    // Log slow renders (>16ms = 60fps threshold)
    if (duration > 16) {
      console.warn(`Slow render detected: ${componentName} took ${duration.toFixed(2)}ms`);
    }
    
    // Send to monitoring service if configured
    if (window.__MONITORING_ENABLED__) {
      this.reportMetric(componentName, duration);
    }
  }
  
  /**
   * Track Web Vitals
   */
  trackWebVitals() {
    // First Contentful Paint
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          this.reportMetric('FCP', entry.startTime);
        }
      }
    }).observe({ entryTypes: ['paint'] });
    
    // Largest Contentful Paint
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.reportMetric('LCP', lastEntry.startTime);
    }).observe({ entryTypes: ['largest-contentful-paint'] });
    
    // First Input Delay
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const eventEntry = entry as PerformanceEventTiming;
        const delay = eventEntry.processingStart - eventEntry.startTime;
        this.reportMetric('FID', delay);
      }
    }).observe({ entryTypes: ['first-input'] });
    
    // Cumulative Layout Shift
    let clsScore = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!(entry as any).hadRecentInput) {
          clsScore += (entry as any).value;
        }
      }
      this.reportMetric('CLS', clsScore);
    }).observe({ entryTypes: ['layout-shift'] });
  }
  
  private reportMetric(name: string, value: number) {
    // Send to monitoring service
    if (typeof window !== 'undefined' && window.__MONITORING_ENABLED__) {
      fetch('/api/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metric: name,
          value,
          timestamp: Date.now(),
          url: window.location.href,
          userAgent: navigator.userAgent
        })
      }).catch(console.error);
    }
  }
  
  getReport() {
    const report: Record<string, any> = {};
    
    this.metrics.forEach((values, key) => {
      report[key] = {
        count: values.length,
        average: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        p95: this.percentile(values, 95)
      };
    });
    
    return report;
  }
  
  private percentile(values: number[], p: number): number {
    const sorted = values.sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[index];
  }
}

export const performanceMonitor = new PerformanceMonitor();
```

---

## 7. Code Quality Standards

### 7.1 Linting and Formatting

```javascript
// .eslintrc.js - ESLint configuration

module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended',
    'prettier'
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true
    },
    project: './tsconfig.json'
  },
  plugins: [
    '@typescript-eslint',
    'react',
    'react-hooks',
    'jsx-a11y',
    'import'
  ],
  settings: {
    react: {
      version: 'detect'
    }
  },
  rules: {
    // TypeScript Rules
    '@typescript-eslint/explicit-function-return-type': 'error',
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-non-null-assertion': 'error',
    
    // React Rules
    'react/prop-types': 'off', // Using TypeScript
    'react/react-in-jsx-scope': 'off', // React 17+
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    
    // Import Rules
    'import/order': ['error', {
      groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
      'newlines-between': 'always',
      alphabetize: {
        order: 'asc',
        caseInsensitive: true
      }
    }],
    
    // General Rules
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-debugger': 'error',
    'prefer-const': 'error',
    'no-var': 'error'
  }
};
```

### 7.2 Pre-commit Hooks

```json
// package.json - Husky and lint-staged configuration

{
  "scripts": {
    "prepare": "husky install",
    "lint": "eslint src --ext .ts,.tsx",
    "format": "prettier --write \"src/**/*.{ts,tsx,css,scss}\"",
    "type-check": "tsc --noEmit",
    "test": "vitest",
    "test:coverage": "vitest --coverage"
  },
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "pre-push": "npm run test"
    }
  },
  "lint-staged": {
    "src/**/*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write",
      "git add"
    ],
    "src/**/*.{css,scss}": [
      "prettier --write",
      "git add"
    ]
  }
}
```

---

## 8. Development Workflow

### 8.1 Component Development Checklist

```markdown
## New Component Checklist

### Before Starting
- [ ] Component count under 40 limit
- [ ] Design approved by UI/UX team
- [ ] API contracts reviewed
- [ ] TypeScript interfaces defined

### During Development
- [ ] Functional component (no class components except ErrorBoundary)
- [ ] TypeScript strict mode compliance
- [ ] Zustand for state (NOT Redux)
- [ ] MUI imports use tree-shaking pattern
- [ ] Props validated with TypeScript
- [ ] Loading and error states handled
- [ ] Memoization applied where needed
- [ ] Accessibility requirements met (WCAG 2.1 AA)

### Before PR
- [ ] Unit tests written (70% coverage minimum)
- [ ] Component documented in Storybook (post-MVP)
- [ ] Performance tested (<16ms render time)
- [ ] Bundle size impact checked (<10KB per component)
- [ ] Cross-browser tested (Chrome, Firefox, Safari)
- [ ] Responsive behavior verified (desktop-only for MVP)
- [ ] Code reviewed by peer
- [ ] No console.log statements
```

### 8.2 Git Workflow

```bash
# Branch naming convention
feature/YTE-123-channel-card
bugfix/YTE-456-video-queue-error
hotfix/YTE-789-critical-dashboard-fix

# Commit message format
feat(channels): add channel creation modal
fix(videos): resolve queue polling issue
refactor(dashboard): optimize metric calculations
test(auth): add integration tests for login flow
docs(readme): update setup instructions

# PR workflow
1. Create feature branch from develop
2. Implement feature with tests
3. Run local checks: npm run lint && npm run test
4. Push branch and create PR
5. Await code review and CI/CD checks
6. Merge to develop after approval
7. Deploy to staging for QA
8. Merge to main for production release
```

---

## Quick Reference

### Critical Reminders

```yaml
DO:
  ✅ Use Zustand for state management
  ✅ Use Recharts for data visualization
  ✅ Use functional components
  ✅ Use TypeScript strict mode
  ✅ Test to 70% coverage minimum
  ✅ Keep bundle under 1MB
  ✅ Target 2-second page load

DON'T:
  ❌ Use Redux (Zustand only)
  ❌ Use D3.js (Recharts only)
  ❌ Create class components (except ErrorBoundary)
  ❌ Exceed 40 components
  ❌ Skip TypeScript types
  ❌ Ignore accessibility
  ❌ Leave console.logs
```

### Performance Targets

```yaml
Core Web Vitals:
  FCP: <1.8s
  LCP: <2.5s
  FID: <100ms
  CLS: <0.1
  
Custom Metrics:
  Component Render: <16ms
  API Response: <1s
  Bundle Size: <1MB
  Code Coverage: ≥70%
```

---

**Document Status**: COMPLETE  
**Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: Week 6 Sprint Review  
**Author**: Frontend Team Lead  
**Approved By**: CTO/Technical Director