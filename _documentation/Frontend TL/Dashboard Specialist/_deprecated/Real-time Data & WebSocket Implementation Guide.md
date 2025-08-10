# Real-time Data & WebSocket Implementation Guide
## For: Dashboard Specialist | YTEMPIRE Frontend Team

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**Status**: Implementation Ready

---

## Executive Summary

This guide covers the real-time data architecture for YTEMPIRE's dashboard, focusing on the hybrid approach of **60-second polling for general updates** and **WebSocket connections for 3 critical events only**. This approach balances real-time responsiveness with system efficiency for our MVP.

### Key Real-time Constraints (MVP)
- **Polling interval**: 60 seconds for dashboard metrics
- **WebSocket events**: Only 3 critical events
- **Maximum latency**: 2 seconds for updates
- **Concurrent connections**: Support 100 users
- **Data freshness**: Near real-time for critical events

---

## 1. Real-time Data Requirements

### 1.1 Data Update Categories

```typescript
// Data update strategy by category
const dataUpdateStrategy = {
  // Critical real-time (WebSocket)
  critical: {
    events: ['video.completed', 'video.failed', 'cost.alert'],
    latency: '<1 second',
    priority: 'highest',
    implementation: 'WebSocket'
  },
  
  // Near real-time (60s polling)
  nearRealtime: {
    data: ['dashboard_metrics', 'channel_stats', 'revenue_data'],
    latency: '60 seconds',
    priority: 'high',
    implementation: 'HTTP polling'
  },
  
  // Periodic updates (5-minute polling)
  periodic: {
    data: ['analytics_trends', 'cost_breakdown', 'performance_history'],
    latency: '5 minutes',
    priority: 'medium',
    implementation: 'HTTP polling with cache'
  },
  
  // On-demand only
  onDemand: {
    data: ['detailed_reports', 'export_data', 'historical_analysis'],
    latency: 'User triggered',
    priority: 'low',
    implementation: 'Manual fetch'
  }
};
```

### 1.2 Data Freshness Requirements

```typescript
interface DataFreshnessRequirements {
  // MVP Requirements
  videoStatus: {
    completed: 'immediate', // WebSocket
    failed: 'immediate',    // WebSocket
    processing: '60s'       // Polling
  };
  
  costTracking: {
    alert: 'immediate',     // WebSocket when >$0.40
    current: '60s',         // Polling
    breakdown: '5min'       // Less frequent
  };
  
  dashboardMetrics: {
    channelCount: '60s',
    videoCount: '60s',
    revenue: '60s',
    automation: '5min'
  };
  
  analytics: {
    realtimeViews: 'Not in MVP',
    dailyStats: '60s',
    trends: '5min'
  };
}
```

---

## 2. WebSocket Events Specification

### 2.1 WebSocket Connection Management

```typescript
// WebSocketManager.ts - Singleton WebSocket connection manager
class WebSocketManager {
  private static instance: WebSocketManager;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<string, Set<Function>> = new Map();
  private connectionState: 'connecting' | 'connected' | 'disconnected' = 'disconnected';

  private constructor() {}

  static getInstance(): WebSocketManager {
    if (!WebSocketManager.instance) {
      WebSocketManager.instance = new WebSocketManager();
    }
    return WebSocketManager.instance;
  }

  connect(userId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    this.connectionState = 'connecting';
    const wsUrl = `${import.meta.env.VITE_WS_URL}/ws/${userId}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.scheduleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.connectionState = 'connected';
      this.reconnectAttempts = 0;
      this.authenticate();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      this.connectionState = 'disconnected';
      this.scheduleReconnect();
    };
  }

  private handleMessage(data: WebSocketEvent): void {
    const handlers = this.eventHandlers.get(data.type);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }

    // Global event for monitoring
    const globalHandlers = this.eventHandlers.get('*');
    if (globalHandlers) {
      globalHandlers.forEach(handler => handler(data));
    }
  }

  subscribe(eventType: string, handler: Function): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }
    
    this.eventHandlers.get(eventType)!.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.eventHandlers.get(eventType)?.delete(handler);
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.notifyConnectionFailure();
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    setTimeout(() => {
      this.reconnectAttempts++;
      const userId = useAuthStore.getState().user?.id;
      if (userId) {
        this.connect(userId);
      }
    }, delay);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.eventHandlers.clear();
    this.connectionState = 'disconnected';
  }

  getConnectionState(): string {
    return this.connectionState;
  }
}

export const wsManager = WebSocketManager.getInstance();
```

### 2.2 Critical WebSocket Events (MVP)

```typescript
// WebSocket event types - MVP only includes 3 critical events
enum WebSocketEventType {
  VIDEO_COMPLETED = 'video.completed',
  VIDEO_FAILED = 'video.failed',
  COST_ALERT = 'cost.alert'
}

// Event payload interfaces
interface VideoCompletedEvent {
  type: 'video.completed';
  payload: {
    videoId: string;
    channelId: string;
    channelName: string;
    title: string;
    url: string;
    cost: number;
    generationTime: number;
    timestamp: string;
  };
}

interface VideoFailedEvent {
  type: 'video.failed';
  payload: {
    videoId: string;
    channelId: string;
    channelName: string;
    error: {
      stage: 'script' | 'audio' | 'video' | 'upload';
      message: string;
      retryable: boolean;
    };
    cost: number; // Cost incurred before failure
    timestamp: string;
  };
}

interface CostAlertEvent {
  type: 'cost.alert';
  payload: {
    level: 'warning' | 'critical' | 'limit';
    currentCost: number;
    threshold: number;
    message: string;
    remainingBudget: number;
    projectedDailyCost: number;
    timestamp: string;
  };
}

type WebSocketEvent = VideoCompletedEvent | VideoFailedEvent | CostAlertEvent;
```

### 2.3 WebSocket Hook Implementation

```typescript
// useWebSocketEvents.ts - React hook for WebSocket events
import { useEffect, useCallback } from 'react';
import { wsManager } from '@/services/websocket';
import { useNotificationStore } from '@/stores/notificationStore';
import { useDashboardStore } from '@/stores/dashboardStore';

export const useWebSocketEvents = (eventTypes: WebSocketEventType[]) => {
  const { showNotification } = useNotificationStore();
  const { updateMetrics, addRecentVideo } = useDashboardStore();

  const handleVideoCompleted = useCallback((event: VideoCompletedEvent) => {
    // Update dashboard metrics
    updateMetrics({
      videosToday: (prev) => prev + 1,
      totalCost: (prev) => prev + event.payload.cost
    });

    // Add to recent videos
    addRecentVideo({
      id: event.payload.videoId,
      channelName: event.payload.channelName,
      title: event.payload.title,
      status: 'completed',
      cost: event.payload.cost,
      timestamp: event.payload.timestamp
    });

    // Show success notification
    showNotification({
      type: 'success',
      title: 'Video Completed',
      message: `"${event.payload.title}" published to ${event.payload.channelName}`,
      duration: 5000
    });
  }, [updateMetrics, addRecentVideo, showNotification]);

  const handleVideoFailed = useCallback((event: VideoFailedEvent) => {
    // Show error notification
    showNotification({
      type: 'error',
      title: 'Video Generation Failed',
      message: `Failed at ${event.payload.error.stage}: ${event.payload.error.message}`,
      duration: 10000,
      action: event.payload.error.retryable ? {
        label: 'Retry',
        onClick: () => retryVideoGeneration(event.payload.videoId)
      } : undefined
    });

    // Update failure metrics
    updateMetrics({
      failedToday: (prev) => prev + 1,
      totalCost: (prev) => prev + event.payload.cost
    });
  }, [showNotification, updateMetrics]);

  const handleCostAlert = useCallback((event: CostAlertEvent) => {
    const severity = {
      warning: 'warning',
      critical: 'error',
      limit: 'error'
    }[event.payload.level];

    // Show cost alert
    showNotification({
      type: severity as any,
      title: 'Cost Alert',
      message: event.payload.message,
      duration: event.payload.level === 'limit' ? null : 10000, // Persistent for limit
      action: {
        label: 'View Details',
        onClick: () => navigateToCostDashboard()
      }
    });

    // Update cost alert status
    useDashboardStore.setState({
      costAlert: {
        active: true,
        level: event.payload.level,
        currentCost: event.payload.currentCost,
        remainingBudget: event.payload.remainingBudget
      }
    });
  }, [showNotification]);

  useEffect(() => {
    const unsubscribers: (() => void)[] = [];

    // Subscribe to requested event types
    if (eventTypes.includes(WebSocketEventType.VIDEO_COMPLETED)) {
      unsubscribers.push(
        wsManager.subscribe('video.completed', handleVideoCompleted)
      );
    }

    if (eventTypes.includes(WebSocketEventType.VIDEO_FAILED)) {
      unsubscribers.push(
        wsManager.subscribe('video.failed', handleVideoFailed)
      );
    }

    if (eventTypes.includes(WebSocketEventType.COST_ALERT)) {
      unsubscribers.push(
        wsManager.subscribe('cost.alert', handleCostAlert)
      );
    }

    // Cleanup
    return () => {
      unsubscribers.forEach(unsub => unsub());
    };
  }, [eventTypes, handleVideoCompleted, handleVideoFailed, handleCostAlert]);

  return {
    connectionState: wsManager.getConnectionState()
  };
};
```

---

## 3. Polling Implementation

### 3.1 Dashboard Polling Hook

```typescript
// useDashboardPolling.ts - 60-second polling for dashboard data
import { useEffect, useRef, useCallback } from 'react';
import { useDashboardStore } from '@/stores/dashboardStore';
import { api } from '@/services/api';

export const useDashboardPolling = (interval: number = 60000) => {
  const intervalRef = useRef<NodeJS.Timeout>();
  const { updateMetrics, setChannels, setLoading, setError } = useDashboardStore();

  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Fetch dashboard overview
      const response = await api.get<DashboardOverviewResponse>(
        '/api/v1/dashboard/overview'
      );
      
      // Update store with fresh data
      updateMetrics(response.data.metrics);
      setChannels(response.data.channels);
      
      // Update last fetch timestamp
      useDashboardStore.setState({
        lastFetch: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('Dashboard fetch error:', error);
      setError('Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  }, [updateMetrics, setChannels, setLoading, setError]);

  useEffect(() => {
    // Initial fetch
    fetchDashboardData();

    // Setup polling interval
    intervalRef.current = setInterval(fetchDashboardData, interval);

    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchDashboardData, interval]);

  // Return manual refresh function
  return { refresh: fetchDashboardData };
};
```

### 3.2 Smart Polling with Visibility API

```typescript
// useSmartPolling.ts - Pause polling when tab is not visible
import { useEffect, useRef } from 'react';

export const useSmartPolling = (
  callback: () => void,
  interval: number,
  options: { immediate?: boolean; pauseOnHidden?: boolean } = {}
) => {
  const { immediate = true, pauseOnHidden = true } = options;
  const savedCallback = useRef(callback);
  const intervalRef = useRef<NodeJS.Timeout>();

  // Update callback ref
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    const tick = () => {
      savedCallback.current();
    };

    const startPolling = () => {
      if (immediate) tick();
      intervalRef.current = setInterval(tick, interval);
    };

    const stopPolling = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };

    if (pauseOnHidden) {
      const handleVisibilityChange = () => {
        if (document.hidden) {
          stopPolling();
        } else {
          startPolling();
        }
      };

      // Start polling if page is visible
      if (!document.hidden) {
        startPolling();
      }

      // Listen for visibility changes
      document.addEventListener('visibilitychange', handleVisibilityChange);

      return () => {
        stopPolling();
        document.removeEventListener('visibilitychange', handleVisibilityChange);
      };
    } else {
      startPolling();
      return stopPolling;
    }
  }, [interval, immediate, pauseOnHidden]);
};
```

---

## 4. Data Synchronization Strategy

### 4.1 Optimistic Updates

```typescript
// Optimistic update pattern for better UX
const useOptimisticUpdate = () => {
  const { channels, updateChannel } = useDashboardStore();

  const toggleChannelStatus = async (channelId: string) => {
    const channel = channels.find(ch => ch.id === channelId);
    if (!channel) return;

    const newStatus = channel.status === 'active' ? 'paused' : 'active';
    
    // Optimistic update
    updateChannel(channelId, { status: newStatus });

    try {
      // API call
      await api.patch(`/api/v1/channels/${channelId}/status`, {
        status: newStatus
      });
    } catch (error) {
      // Revert on failure
      updateChannel(channelId, { status: channel.status });
      
      showNotification({
        type: 'error',
        message: 'Failed to update channel status'
      });
    }
  };

  return { toggleChannelStatus };
};
```

### 4.2 Data Conflict Resolution

```typescript
// Handle conflicts between polling and WebSocket updates
class DataSyncManager {
  private lastUpdateTimestamps: Map<string, number> = new Map();

  shouldUpdateData(dataType: string, newTimestamp: number): boolean {
    const lastTimestamp = this.lastUpdateTimestamps.get(dataType) || 0;
    
    if (newTimestamp > lastTimestamp) {
      this.lastUpdateTimestamps.set(dataType, newTimestamp);
      return true;
    }
    
    return false;
  }

  mergeUpdates(
    currentData: any,
    pollingData: any,
    websocketData: any,
    dataType: string
  ): any {
    // WebSocket data takes precedence for real-time fields
    const realtimeFields = ['status', 'currentCost', 'activeAlerts'];
    
    // Polling data for aggregate metrics
    const aggregateFields = ['totalVideos', 'revenue', 'avgMetrics'];
    
    const merged = { ...currentData };
    
    // Apply real-time updates from WebSocket
    realtimeFields.forEach(field => {
      if (websocketData?.[field] !== undefined) {
        merged[field] = websocketData[field];
      }
    });
    
    // Apply aggregate updates from polling
    aggregateFields.forEach(field => {
      if (pollingData?.[field] !== undefined) {
        merged[field] = pollingData[field];
      }
    });
    
    return merged;
  }
}

export const dataSyncManager = new DataSyncManager();
```

---

## 5. Performance Monitoring

### 5.1 Real-time Performance Metrics

```typescript
// Monitor real-time data performance
class RealtimePerformanceMonitor {
  private metrics = {
    websocketLatency: [] as number[],
    pollingDuration: [] as number[],
    updateFrequency: [] as number[],
    dataSize: [] as number[]
  };

  measureWebSocketLatency(event: WebSocketEvent): void {
    const now = Date.now();
    const eventTime = new Date(event.payload.timestamp).getTime();
    const latency = now - eventTime;
    
    this.metrics.websocketLatency.push(latency);
    
    // Alert if latency exceeds threshold
    if (latency > 1000) {
      console.warn(`High WebSocket latency: ${latency}ms`);
    }
  }

  measurePollingDuration(duration: number): void {
    this.metrics.pollingDuration.push(duration);
    
    // Alert if polling takes too long
    if (duration > 5000) {
      console.warn(`Slow polling response: ${duration}ms`);
    }
  }

  getAverageMetrics(): PerformanceMetrics {
    return {
      avgWebSocketLatency: this.average(this.metrics.websocketLatency),
      avgPollingDuration: this.average(this.metrics.pollingDuration),
      p95WebSocketLatency: this.percentile(this.metrics.websocketLatency, 95),
      p95PollingDuration: this.percentile(this.metrics.pollingDuration, 95)
    };
  }

  private average(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[index];
  }
}

export const perfMonitor = new RealtimePerformanceMonitor();
```

### 5.2 Connection Status Component

```typescript
// ConnectionStatus.tsx - Visual indicator for real-time connection
import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import { useWebSocketEvents } from '@/hooks/useWebSocketEvents';

export const ConnectionStatus: React.FC = () => {
  const { connectionState } = useWebSocketEvents([]);
  const [pollingActive, setPollingActive] = useState(true);

  const getStatusColor = () => {
    if (connectionState === 'connected' && pollingActive) {
      return 'success';
    } else if (connectionState === 'connecting' || pollingActive) {
      return 'warning';
    } else {
      return 'error';
    }
  };

  const getStatusText = () => {
    if (connectionState === 'connected' && pollingActive) {
      return 'Live';
    } else if (connectionState === 'connecting') {
      return 'Connecting...';
    } else if (pollingActive) {
      return 'Polling Only';
    } else {
      return 'Offline';
    }
  };

  return (
    <Tooltip title={`WebSocket: ${connectionState}, Polling: ${pollingActive ? 'Active' : 'Inactive'}`}>
      <Chip
        icon={<FiberManualRecordIcon />}
        label={getStatusText()}
        color={getStatusColor()}
        size="small"
        variant="outlined"
      />
    </Tooltip>
  );
};
```

---

## 6. Testing Real-time Features

### 6.1 WebSocket Testing

```typescript
// __tests__/websocket.test.ts
import WS from 'jest-websocket-mock';
import { wsManager } from '@/services/websocket';

describe('WebSocket Manager', () => {
  let server: WS;

  beforeEach(async () => {
    server = new WS('ws://localhost:8000/ws/test-user');
    await server.connected;
  });

  afterEach(() => {
    WS.clean();
  });

  test('handles video.completed event', async () => {
    const handler = jest.fn();
    wsManager.subscribe('video.completed', handler);

    const event = {
      type: 'video.completed',
      payload: {
        videoId: '123',
        channelId: '456',
        channelName: 'Test Channel',
        title: 'Test Video',
        url: 'https://youtube.com/watch?v=123',
        cost: 0.45,
        generationTime: 300,
        timestamp: new Date().toISOString()
      }
    };

    server.send(JSON.stringify(event));

    await expect(handler).toHaveBeenCalledWith(event);
  });

  test('reconnects on connection loss', async () => {
    wsManager.connect('test-user');
    
    // Simulate connection loss
    server.close();
    
    // Wait for reconnection attempt
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Verify reconnection attempted
    expect(server).toHaveReceivedMessages([
      expect.stringContaining('authenticate')
    ]);
  });
});
```

---

## Best Practices Summary

### Do's ✅
1. **Use 60-second polling** for general dashboard updates
2. **Limit WebSocket** to 3 critical events only
3. **Implement smart polling** that pauses when tab is hidden
4. **Show connection status** to users
5. **Handle reconnection** gracefully with exponential backoff
6. **Optimize updates** with memoization and shallow comparisons
7. **Monitor performance** of real-time features

### Don'ts ❌
1. **Don't use WebSocket** for all data updates
2. **Don't poll more frequently** than 60 seconds
3. **Don't ignore connection failures**
4. **Don't update state** without checking timestamps
5. **Don't forget error boundaries** for real-time components
6. **Don't block UI** during polling updates

---

## Next Steps

1. Implement the WebSocket manager singleton
2. Create the dashboard polling hooks
3. Add connection status indicator to header
4. Set up performance monitoring
5. Test with simulated connection issues
6. Review the [Data Visualization & Recharts Standards](dashboard-visualization-guide)

**Remember**: The hybrid approach (polling + critical WebSockets) provides the best balance of responsiveness and efficiency for our MVP.