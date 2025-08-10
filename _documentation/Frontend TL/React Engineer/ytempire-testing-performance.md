# YTEMPIRE React Engineer - Testing & Performance Guide
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: Testing Strategy & Performance Standards

---

## 1. Testing Strategy Overview

### 1.1 Testing Pyramid

```yaml
Testing Distribution:
  Unit Tests: 70%
    - Components
    - Hooks
    - Utilities
    - Store logic
    
  Integration Tests: 25%
    - API integration
    - User flows
    - Store interactions
    
  E2E Tests: 5% (Post-MVP)
    - Critical paths only
    - Login flow
    - Channel creation
    - Video generation
```

### 1.2 Coverage Requirements

| Category | Minimum | Target | Notes |
|----------|---------|--------|-------|
| **Global** | 70% | 75% | Overall project coverage |
| **Critical Paths** | 90% | 95% | Auth, payments, video generation |
| **Components** | 70% | 80% | All UI components |
| **Stores** | 80% | 85% | Zustand stores |
| **Services** | 75% | 80% | API services |
| **Utilities** | 95% | 100% | Pure functions |

---

## 2. Unit Testing

### 2.1 Component Testing Pattern

```typescript
// ChannelCard.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ChannelCard } from './ChannelCard';
import { TestWrapper } from '@/test/utils';
import type { Channel } from '@/types/models.types';

describe('ChannelCard', () => {
  const mockChannel: Channel = {
    id: 'channel-1',
    name: 'Tech Reviews',
    niche: 'Technology',
    status: 'active',
    automationEnabled: true,
    dailyVideoLimit: 3,
    totalVideos: 150,
    estimatedRevenue: 4500.00,
    totalCost: 450.00,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
  };
  
  describe('Rendering', () => {
    it('should render channel information correctly', () => {
      render(
        <TestWrapper>
          <ChannelCard channel={mockChannel} />
        </TestWrapper>
      );
      
      expect(screen.getByText('Tech Reviews')).toBeInTheDocument();
      expect(screen.getByText(/Technology/)).toBeInTheDocument();
      expect(screen.getByText(/active/)).toBeInTheDocument();
    });
    
    it('should display metrics when showMetrics prop is true', () => {
      render(
        <TestWrapper>
          <ChannelCard channel={mockChannel} showMetrics />
        </TestWrapper>
      );
      
      expect(screen.getByText(/150 videos/i)).toBeInTheDocument();
      expect(screen.getByText(/\$4,500/)).toBeInTheDocument();
      expect(screen.getByText(/\$450/)).toBeInTheDocument();
    });
    
    it('should render different variants correctly', () => {
      const { rerender } = render(
        <TestWrapper>
          <ChannelCard channel={mockChannel} variant="compact" />
        </TestWrapper>
      );
      
      expect(screen.queryByRole('button', { name: /toggle/i })).not.toBeInTheDocument();
      
      rerender(
        <TestWrapper>
          <ChannelCard channel={mockChannel} variant="detailed" />
        </TestWrapper>
      );
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });
  });
  
  describe('Interactions', () => {
    it('should call onSelect when clicked', async () => {
      const user = userEvent.setup();
      const handleSelect = vi.fn();
      
      render(
        <TestWrapper>
          <ChannelCard 
            channel={mockChannel} 
            onSelect={handleSelect}
          />
        </TestWrapper>
      );
      
      await user.click(screen.getByRole('article'));
      expect(handleSelect).toHaveBeenCalledWith('channel-1');
    });
    
    it('should toggle automation status', async () => {
      const user = userEvent.setup();
      const { mockToggleAutomation } = vi.hoisted(() => ({
        mockToggleAutomation: vi.fn()
      }));
      
      vi.mock('@/stores/useChannelStore', () => ({
        useChannelStore: () => ({
          toggleAutomation: mockToggleAutomation
        })
      }));
      
      render(
        <TestWrapper>
          <ChannelCard 
            channel={mockChannel} 
            variant="detailed"
          />
        </TestWrapper>
      );
      
      const toggleButton = screen.getByRole('button');
      await user.click(toggleButton);
      
      await waitFor(() => {
        expect(mockToggleAutomation).toHaveBeenCalledWith('channel-1');
      });
    });
  });
  
  describe('Loading States', () => {
    it('should show loading state while processing', async () => {
      const user = userEvent.setup();
      const slowToggle = vi.fn(() => 
        new Promise(resolve => setTimeout(resolve, 100))
      );
      
      vi.mock('@/stores/useChannelStore', () => ({
        useChannelStore: () => ({
          toggleAutomation: slowToggle
        })
      }));
      
      render(
        <TestWrapper>
          <ChannelCard 
            channel={mockChannel} 
            variant="detailed"
          />
        </TestWrapper>
      );
      
      const button = screen.getByRole('button');
      await user.click(button);
      
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute('aria-busy', 'true');
      
      await waitFor(() => {
        expect(button).not.toBeDisabled();
      });
    });
  });
  
  describe('Accessibility', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <TestWrapper>
          <ChannelCard channel={mockChannel} />
        </TestWrapper>
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
    
    it('should be keyboard navigable', async () => {
      const user = userEvent.setup();
      const handleSelect = vi.fn();
      
      render(
        <TestWrapper>
          <ChannelCard 
            channel={mockChannel} 
            onSelect={handleSelect}
          />
        </TestWrapper>
      );
      
      const card = screen.getByRole('article');
      card.focus();
      
      await user.keyboard('{Enter}');
      expect(handleSelect).toHaveBeenCalledWith('channel-1');
      
      await user.keyboard(' '); // Space key
      expect(handleSelect).toHaveBeenCalledTimes(2);
    });
  });
});
```

### 2.2 Store Testing

```typescript
// useChannelStore.test.ts
import { renderHook, act, waitFor } from '@testing-library/react';
import { useChannelStore } from './useChannelStore';
import { channelService } from '@/services/channels';
import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('@/services/channels');

describe('useChannelStore', () => {
  beforeEach(() => {
    // Reset store between tests
    useChannelStore.setState({
      channels: [],
      selectedChannelId: null,
      loading: false,
      error: null
    });
    vi.clearAllMocks();
  });
  
  describe('fetchChannels', () => {
    it('should fetch and set channels', async () => {
      const mockChannels = [
        { id: '1', name: 'Channel 1', status: 'active' },
        { id: '2', name: 'Channel 2', status: 'paused' }
      ];
      
      (channelService.getChannels as any).mockResolvedValue(mockChannels);
      
      const { result } = renderHook(() => useChannelStore());
      
      expect(result.current.loading).toBe(false);
      expect(result.current.channels).toHaveLength(0);
      
      await act(async () => {
        await result.current.fetchChannels();
      });
      
      expect(result.current.channels).toEqual(mockChannels);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
    });
    
    it('should handle fetch errors', async () => {
      const error = new Error('Network error');
      (channelService.getChannels as any).mockRejectedValue(error);
      
      const { result } = renderHook(() => useChannelStore());
      
      await act(async () => {
        await result.current.fetchChannels();
      });
      
      expect(result.current.channels).toHaveLength(0);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe('Network error');
    });
    
    it('should set loading state during fetch', async () => {
      (channelService.getChannels as any).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve([]), 100))
      );
      
      const { result } = renderHook(() => useChannelStore());
      
      act(() => {
        result.current.fetchChannels();
      });
      
      expect(result.current.loading).toBe(true);
      
      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });
    });
  });
  
  describe('selectChannel', () => {
    it('should set selected channel ID', () => {
      const { result } = renderHook(() => useChannelStore());
      
      act(() => {
        result.current.selectChannel('channel-1');
      });
      
      expect(result.current.selectedChannelId).toBe('channel-1');
    });
  });
  
  describe('toggleAutomation', () => {
    it('should toggle channel automation', async () => {
      const mockChannel = { 
        id: '1', 
        name: 'Channel 1', 
        automationEnabled: false 
      };
      
      const { result } = renderHook(() => useChannelStore());
      
      // Set initial channels
      act(() => {
        useChannelStore.setState({ channels: [mockChannel] });
      });
      
      (channelService.toggleAutomation as any).mockResolvedValue({
        ...mockChannel,
        automationEnabled: true
      });
      
      await act(async () => {
        await result.current.toggleAutomation('1');
      });
      
      expect(result.current.channels[0].automationEnabled).toBe(true);
    });
  });
});
```

### 2.3 Hook Testing

```typescript
// usePolling.test.ts
import { renderHook, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { usePolling } from './usePolling';

describe('usePolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  
  afterEach(() => {
    vi.useRealTimers();
  });
  
  it('should call callback at specified interval', () => {
    const callback = vi.fn();
    
    renderHook(() => usePolling(callback, { interval: 5000 }));
    
    // Initial call
    expect(callback).toHaveBeenCalledTimes(1);
    
    // After 5 seconds
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(callback).toHaveBeenCalledTimes(2);
    
    // After another 5 seconds
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(callback).toHaveBeenCalledTimes(3);
  });
  
  it('should stop polling when enabled is false', () => {
    const callback = vi.fn();
    
    const { rerender } = renderHook(
      ({ enabled }) => usePolling(callback, { enabled, interval: 5000 }),
      { initialProps: { enabled: true } }
    );
    
    expect(callback).toHaveBeenCalledTimes(1);
    
    // Disable polling
    rerender({ enabled: false });
    
    act(() => {
      vi.advanceTimersByTime(10000);
    });
    
    // Should not have been called again
    expect(callback).toHaveBeenCalledTimes(1);
  });
  
  it('should handle async callbacks', async () => {
    const asyncCallback = vi.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 100));
    });
    
    renderHook(() => usePolling(asyncCallback, { interval: 5000 }));
    
    // Wait for initial async call
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    
    expect(asyncCallback).toHaveBeenCalled();
  });
  
  it('should handle errors with onError callback', () => {
    const callback = vi.fn(() => {
      throw new Error('Polling error');
    });
    const onError = vi.fn();
    
    renderHook(() => usePolling(callback, { interval: 5000, onError }));
    
    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    expect(onError.mock.calls[0][0].message).toBe('Polling error');
  });
});
```

---

## 3. Integration Testing

### 3.1 API Integration Tests

```typescript
// api.integration.test.ts
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { rest } from 'msw';
import { channelService } from '@/services/channels';
import { videoService } from '@/services/videos';

const server = setupServer(
  rest.get('/api/v1/channels', (req, res, ctx) => {
    return res(ctx.json({
      success: true,
      data: [
        { id: '1', name: 'Channel 1', status: 'active' },
        { id: '2', name: 'Channel 2', status: 'paused' }
      ]
    }));
  }),
  
  rest.post('/api/v1/videos/generate', (req, res, ctx) => {
    return res(ctx.json({
      success: true,
      data: {
        videoId: 'video-123',
        status: 'queued',
        queuePosition: 5,
        estimatedCompletion: '2024-01-15T12:00:00Z',
        estimatedCost: 2.50
      }
    }));
  })
);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('API Integration', () => {
  describe('Channel API', () => {
    it('should fetch channels successfully', async () => {
      const channels = await channelService.getChannels();
      
      expect(channels).toHaveLength(2);
      expect(channels[0]).toHaveProperty('id', '1');
      expect(channels[0]).toHaveProperty('name', 'Channel 1');
    });
    
    it('should handle API errors gracefully', async () => {
      server.use(
        rest.get('/api/v1/channels', (req, res, ctx) => {
          return res(
            ctx.status(500),
            ctx.json({
              success: false,
              error: {
                code: 'INTERNAL_ERROR',
                message: 'Database connection failed'
              }
            })
          );
        })
      );
      
      await expect(channelService.getChannels()).rejects.toThrow();
    });
    
    it('should handle network timeouts', async () => {
      server.use(
        rest.get('/api/v1/channels', (req, res, ctx) => {
          return res(ctx.delay(35000)); // Exceed 30s timeout
        })
      );
      
      await expect(channelService.getChannels()).rejects.toThrow(/timeout/i);
    }, 40000);
  });
  
  describe('Video Generation API', () => {
    it('should submit video generation request', async () => {
      const request = {
        channelId: 'channel-1',
        topic: 'Tech Review',
        style: 'educational' as const,
        length: 'medium' as const,
        priority: 5
      };
      
      const response = await videoService.generateVideo(request);
      
      expect(response.videoId).toBe('video-123');
      expect(response.status).toBe('queued');
      expect(response.estimatedCost).toBe(2.50);
    });
  });
});
```

### 3.2 User Flow Tests

```typescript
// userFlows.integration.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, beforeEach } from 'vitest';
import { App } from '@/App';
import { TestWrapper } from '@/test/utils';
import { setupMockServer } from '@/test/mocks';

describe('User Flows', () => {
  beforeEach(() => {
    setupMockServer();
  });
  
  describe('Channel Creation Flow', () => {
    it('should complete channel creation successfully', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <App />
        </TestWrapper>
      );
      
      // Navigate to channels
      await user.click(screen.getByRole('link', { name: /channels/i }));
      
      // Open creation modal
      await user.click(screen.getByRole('button', { name: /create channel/i }));
      
      // Fill form
      await user.type(
        screen.getByLabelText(/channel name/i),
        'My Tech Channel'
      );
      
      await user.selectOptions(
        screen.getByLabelText(/niche/i),
        'Technology'
      );
      
      await user.type(
        screen.getByLabelText(/daily limit/i),
        '3'
      );
      
      // Submit form
      await user.click(screen.getByRole('button', { name: /create/i }));
      
      // Verify success
      await waitFor(() => {
        expect(screen.getByText('Channel created successfully')).toBeInTheDocument();
        expect(screen.getByText('My Tech Channel')).toBeInTheDocument();
      });
    });
  });
  
  describe('Video Generation Flow', () => {
    it('should complete video generation request', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <App />
        </TestWrapper>
      );
      
      // Select channel
      await user.click(screen.getByText('Channel 1'));
      
      // Click generate button
      await user.click(screen.getByRole('button', { name: /generate video/i }));
      
      // Fill generation form
      await user.type(
        screen.getByLabelText(/topic/i),
        'Latest iPhone Review'
      );
      
      await user.click(screen.getByLabelText(/educational/i));
      
      await user.selectOptions(
        screen.getByLabelText(/length/i),
        'medium'
      );
      
      // Submit
      await user.click(screen.getByRole('button', { name: /generate/i }));
      
      // Verify queued
      await waitFor(() => {
        expect(screen.getByText(/video queued/i)).toBeInTheDocument();
        expect(screen.getByText(/position #5/i)).toBeInTheDocument();
      });
    });
  });
});
```

---

## 4. Performance Testing

### 4.1 Component Performance Tests

```typescript
// performance.test.tsx
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { measureRenderTime } from '@/test/utils/performance';
import { Dashboard } from '@/pages/Dashboard';
import { ChannelList } from '@/components/channels/ChannelList';

describe('Performance Tests', () => {
  describe('Render Performance', () => {
    it('Dashboard should render in less than 100ms', () => {
      const renderTime = measureRenderTime(() => {
        render(<Dashboard />);
      });
      
      expect(renderTime).toBeLessThan(100);
      console.log(`Dashboard render time: ${renderTime}ms`);
    });
    
    it('ChannelList with 100 items should render in less than 200ms', () => {
      const channels = Array.from({ length: 100 }, (_, i) => ({
        id: `channel-${i}`,
        name: `Channel ${i}`,
        status: 'active'
      }));
      
      const renderTime = measureRenderTime(() => {
        render(<ChannelList channels={channels} />);
      });
      
      expect(renderTime).toBeLessThan(200);
      console.log(`ChannelList (100 items) render time: ${renderTime}ms`);
    });
  });
  
  describe('Re-render Performance', () => {
    it('should minimize unnecessary re-renders', () => {
      const { rerender } = render(<Dashboard />);
      
      const rerenderTime = measureRenderTime(() => {
        rerender(<Dashboard />);
      });
      
      expect(rerenderTime).toBeLessThan(16); // 60fps threshold
    });
  });
  
  describe('Memory Usage', () => {
    it('should not have memory leaks', async () => {
      if (!performance.memory) {
        console.warn('Memory API not available');
        return;
      }
      
      const initialMemory = performance.memory.usedJSHeapSize;
      
      // Render and unmount multiple times
      for (let i = 0; i < 10; i++) {
        const { unmount } = render(<Dashboard />);
        unmount();
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const finalMemory = performance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Allow for some memory increase but flag potential leaks
      expect(memoryIncrease).toBeLessThan(5 * 1024 * 1024); // 5MB threshold
    });
  });
});
```

### 4.2 Bundle Size Analysis

```typescript
// vite.config.ts - Bundle analysis configuration
import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    visualizer({
      filename: './dist/stats.html',
      open: true,
      gzipSize: true,
      brotliSize: true,
      template: 'treemap', // or 'sunburst', 'network'
    })
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui': ['@mui/material', '@emotion/react'],
          'charts': ['recharts'],
          'state': ['zustand'],
        }
      }
    },
    // Generate source maps for analysis
    sourcemap: true,
    // Report compressed sizes
    reportCompressedSize: true,
    // Chunk size warnings
    chunkSizeWarningLimit: 500, // 500KB warning
  }
});
```

---

## 5. Performance Monitoring

### 5.1 Runtime Performance Monitoring

```typescript
// utils/performanceMonitor.ts
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();
  
  measureComponent(componentName: string, fn: () => void): void {
    const startTime = performance.now();
    fn();
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    if (!this.metrics.has(componentName)) {
      this.metrics.set(componentName, []);
    }
    
    this.metrics.get(componentName)!.push(duration);
    
    // Log slow renders
    if (duration > 16) { // 60fps threshold
      console.warn(`Slow render detected: ${componentName} took ${duration}ms`);
    }
  }
  
  getMetrics(componentName: string): {
    count: number;
    average: number;
    min: number;
    max: number;
    p95: number;
  } | null {
    const values = this.metrics.get(componentName);
    if (!values || values.length === 0) return null;
    
    const sorted = [...values].sort((a, b) => a - b);
    
    return {
      count: values.length,
      average: values.reduce((a, b) => a + b, 0) / values.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p95: sorted[Math.floor(sorted.length * 0.95)]
    };
  }
  
  logAllMetrics(): void {
    console.table(
      Array.from(this.metrics.entries()).map(([name, values]) => ({
        Component: name,
        ...this.getMetrics(name)
      }))
    );
  }
  
  reset(): void {
    this.metrics.clear();
  }
}

export const performanceMonitor = new PerformanceMonitor();

// Hook for monitoring component performance
export function usePerformanceMonitor(componentName: string) {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const duration = performance.now() - startTime;
      performanceMonitor.measureComponent(componentName, () => {});
      
      if (import.meta.env.DEV) {
        console.debug(`${componentName} lifecycle: ${duration}ms`);
      }
    };
  }, [componentName]);
}
```

### 5.2 Web Vitals Monitoring

```typescript
// utils/webVitals.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

interface VitalMetric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
}

class WebVitalsMonitor {
  private vitals: Map<string, VitalMetric> = new Map();
  
  init(): void {
    getCLS(this.handleMetric);
    getFID(this.handleMetric);
    getFCP(this.handleMetric);
    getLCP(this.handleMetric);
    getTTFB(this.handleMetric);
  }
  
  private handleMetric = (metric: any): void => {
    const vital: VitalMetric = {
      name: metric.name,
      value: metric.value,
      rating: metric.rating || this.getRating(metric.name, metric.value)
    };
    
    this.vitals.set(metric.name, vital);
    
    // Log to console in development
    if (import.meta.env.DEV) {
      console.log(`Web Vital: ${metric.name}`, vital);
    }
    
    // Send to analytics in production
    if (import.meta.env.PROD) {
      this.sendToAnalytics(vital);
    }
  };
  
  private getRating(name: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    const thresholds: Record<string, [number, number]> = {
      CLS: [0.1, 0.25],
      FID: [100, 300],
      FCP: [1800, 3000],
      LCP: [2500, 4000],
      TTFB: [800, 1800]
    };
    
    const [good, poor] = thresholds[name] || [0, 0];
    
    if (value <= good) return 'good';
    if (value <= poor) return 'needs-improvement';
    return 'poor';
  }
  
  private sendToAnalytics(vital: VitalMetric): void {
    // Send to analytics service
    if (window.gtag) {
      window.gtag('event', 'web_vital', {
        event_category: 'Performance',
        event_label: vital.name,
        value: Math.round(vital.value),
        metric_rating: vital.rating
      });
    }
  }
  
  getVitals(): VitalMetric[] {
    return Array.from(this.vitals.values());
  }
}

export const webVitalsMonitor = new WebVitalsMonitor();
```

---

## 6. Test Utilities

### 6.1 Test Wrapper Component

```typescript
// test/utils/TestWrapper.tsx
import { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ThemeProvider } from '@mui/material/styles';
import { MemoryRouter } from 'react-router-dom';
import { theme } from '@/styles/theme';

interface TestWrapperProps {
  children: ReactNode;
  initialEntries?: string[];
}

export function TestWrapper({ 
  children, 
  initialEntries = ['/'] 
}: TestWrapperProps) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });
  
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <MemoryRouter initialEntries={initialEntries}>
          {children}
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
```

### 6.2 Mock Data Factories

```typescript
// test/factories/channel.factory.ts
import { faker } from '@faker-js/faker';
import type { Channel } from '@/types/models.types';

export function createMockChannel(overrides?: Partial<Channel>): Channel {
  return {
    id: faker.string.uuid(),
    name: faker.company.name(),
    niche: faker.helpers.arrayElement(['Technology', 'Gaming', 'Education', 'Entertainment']),
    status: faker.helpers.arrayElement(['active', 'paused', 'suspended']),
    automationEnabled: faker.datatype.boolean(),
    dailyVideoLimit: faker.number.int({ min: 1, max: 5 }),
    totalVideos: faker.number.int({ min: 0, max: 1000 }),
    estimatedRevenue: faker.number.float({ min: 0, max: 10000, precision: 2 }),
    totalCost: faker.number.float({ min: 0, max: 1000, precision: 2 }),
    createdAt: faker.date.past().toISOString(),
    updatedAt: faker.date.recent().toISOString(),
    ...overrides
  };
}

export function createMockChannels(count: number): Channel[] {
  return Array.from({ length: count }, () => createMockChannel());
}
```

---

## 7. Continuous Testing

### 7.1 Pre-commit Hooks

```json
// package.json
{
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "pre-push": "npm run test:coverage"
    }
  },
  "lint-staged": {
    "src/**/*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write",
      "vitest related --run"
    ]
  }
}
```

### 7.2 CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Type check
        run: npm run type-check
      
      - name: Lint
        run: npm run lint
      
      - name: Unit tests
        run: npm run test:coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
      
      - name: Build
        run: npm run build
      
      - name: Bundle size check
        run: npx bundlesize
        env:
          BUNDLESIZE_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

**Document Status**: FINAL - Consolidated Version  
**Next Review**: Performance Review Week 8  
**Owner**: Frontend Team Lead  
**Questions**: Contact via #frontend-team Slack