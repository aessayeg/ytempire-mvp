# YTEMPIRE Component Testing & Performance Guide

**Version**: 1.0  
**Date**: January 2025  
**Author**: Frontend Team Lead  
**For**: React Engineers  
**Document Type**: Testing & Performance Standards  
**Classification**: Internal - Engineering

---

## Table of Contents

1. [Component Testing Requirements](#1-component-testing-requirements)
2. [Testing Strategy & Architecture](#2-testing-strategy--architecture)
3. [Unit Testing Implementation](#3-unit-testing-implementation)
4. [Integration Testing](#4-integration-testing)
5. [Performance Testing](#5-performance-testing)
6. [Performance Metrics Implementation](#6-performance-metrics-implementation)
7. [Accessibility Standards (WCAG 2.1)](#7-accessibility-standards-wcag-21)
8. [Code Splitting Strategy](#8-code-splitting-strategy)
9. [Bundle Optimization](#9-bundle-optimization)
10. [Monitoring & Alerting](#10-monitoring--alerting)

---

## 1. Component Testing Requirements

### 1.1 Testing Coverage Requirements

```yaml
MVP Testing Targets:
  Global Coverage:
    minimum: 70%
    target: 75%
    ideal: 80%
    
  Critical Paths (90% Required):
    - Authentication flow
    - Payment processing
    - Video generation pipeline
    - Channel management
    - Cost tracking
    
  Component Categories:
    Business Logic: 85%
    UI Components: 60%
    Utilities: 95%
    API Services: 90%
    Stores (Zustand): 80%
    
  Metrics to Track:
    - Line coverage
    - Branch coverage
    - Function coverage
    - Statement coverage
```

### 1.2 Testing Framework Setup

```javascript
// vitest.config.ts - Testing framework configuration
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'src/test/',
        '*.config.ts',
        '*.d.ts',
        'src/types/',
        'src/vite-env.d.ts'
      ],
      thresholds: {
        global: {
          branches: 70,
          functions: 70,
          lines: 70,
          statements: 70
        },
        './src/services/': {
          branches: 90,
          functions: 90,
          lines: 90,
          statements: 90
        },
        './src/utils/': {
          branches: 95,
          functions: 95,
          lines: 95,
          statements: 95
        }
      }
    },
    testTimeout: 10000,
    hookTimeout: 10000,
    teardownTimeout: 10000,
    isolate: true,
    threads: true,
    mockReset: true
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/stores': path.resolve(__dirname, './src/stores'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/utils': path.resolve(__dirname, './src/utils'),
      '@/services': path.resolve(__dirname, './src/services'),
      '@/types': path.resolve(__dirname, './src/types')
    }
  }
});
```

```typescript
// src/test/setup.ts - Global test setup
import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeAll, afterAll, vi } from 'vitest';

// Auto cleanup after each test
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// Mock window APIs
beforeAll(() => {
  // Mock IntersectionObserver
  global.IntersectionObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn()
  }));
  
  // Mock ResizeObserver
  global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn()
  }));
  
  // Mock matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  });
  
  // Mock clipboard API
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
      readText: vi.fn().mockResolvedValue('')
    },
    writable: true
  });
});

// Clean up after all tests
afterAll(() => {
  vi.restoreAllMocks();
});
```

---

## 2. Testing Strategy & Architecture

### 2.1 Testing Pyramid

```typescript
/**
 * YTEMPIRE Testing Pyramid
 * 
 * E2E Tests (5%) - Deferred to Post-MVP
 * ↑
 * Integration Tests (25%) - Critical user flows
 * ↑
 * Unit Tests (70%) - Component and utility testing
 */

// Testing categories and their purposes
export const TestingStrategy = {
  unit: {
    purpose: "Test individual components and functions in isolation",
    coverage: "70% of all tests",
    tools: ["Vitest", "React Testing Library"],
    focus: [
      "Component rendering",
      "Event handlers",
      "State changes",
      "Props validation",
      "Utility functions"
    ]
  },
  
  integration: {
    purpose: "Test component interactions and data flow",
    coverage: "25% of all tests",
    tools: ["Vitest", "React Testing Library", "MSW"],
    focus: [
      "API integration",
      "Store interactions",
      "Multi-component workflows",
      "Route transitions",
      "Form submissions"
    ]
  },
  
  e2e: {
    purpose: "Test complete user journeys",
    coverage: "5% of all tests",
    status: "DEFERRED TO POST-MVP",
    tools: ["Playwright or Cypress"],
    focus: [
      "Critical user paths",
      "Cross-browser compatibility",
      "Performance under load"
    ]
  }
};
```

### 2.2 Test File Organization

```typescript
// File structure for tests
src/
├── components/
│   └── channels/
│       └── ChannelCard/
│           ├── ChannelCard.tsx
│           ├── ChannelCard.test.tsx      // Unit tests
│           └── ChannelCard.types.ts
├── services/
│   ├── api.ts
│   └── api.test.ts                       // Service tests
├── stores/
│   ├── channelStore.ts
│   └── channelStore.test.ts              // Store tests
├── test/
│   ├── setup.ts                          // Global setup
│   ├── utils.tsx                         // Test utilities
│   ├── fixtures/                         // Test data
│   │   ├── channels.ts
│   │   ├── videos.ts
│   │   └── users.ts
│   └── mocks/                            // API mocks
│       ├── handlers.ts
│       └── server.ts
```

---

## 3. Unit Testing Implementation

### 3.1 Component Testing Patterns

```typescript
// src/components/channels/ChannelCard/ChannelCard.test.tsx
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ChannelCard } from './ChannelCard';
import { mockChannel } from '@/test/fixtures/channels';
import { renderWithProviders } from '@/test/utils';

describe('ChannelCard Component', () => {
  // Test props
  const defaultProps = {
    channel: mockChannel,
    onSelect: vi.fn(),
    onToggleAutomation: vi.fn(),
    variant: 'default' as const,
    showMetrics: true
  };
  
  beforeEach(() => {
    vi.clearAllMocks();
  });
  
  describe('Rendering', () => {
    it('should render channel information correctly', () => {
      render(<ChannelCard {...defaultProps} />);
      
      // Assert channel name is displayed
      expect(screen.getByText(mockChannel.name)).toBeInTheDocument();
      
      // Assert channel status is displayed
      expect(screen.getByText(mockChannel.status)).toBeInTheDocument();
      
      // Assert metrics are displayed when showMetrics is true
      expect(screen.getByText(/Revenue:/)).toBeInTheDocument();
      expect(screen.getByText(/Videos:/)).toBeInTheDocument();
    });
    
    it('should not render metrics when showMetrics is false', () => {
      render(<ChannelCard {...defaultProps} showMetrics={false} />);
      
      expect(screen.queryByText(/Revenue:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Videos:/)).not.toBeInTheDocument();
    });
    
    it('should apply correct variant styles', () => {
      const { rerender } = render(<ChannelCard {...defaultProps} variant="compact" />);
      
      const compactCard = screen.getByTestId('channel-card');
      expect(compactCard).toHaveClass('channel-card--compact');
      
      rerender(<ChannelCard {...defaultProps} variant="detailed" />);
      
      const detailedCard = screen.getByTestId('channel-card');
      expect(detailedCard).toHaveClass('channel-card--detailed');
    });
  });
  
  describe('User Interactions', () => {
    it('should call onSelect when card is clicked', async () => {
      const user = userEvent.setup();
      render(<ChannelCard {...defaultProps} />);
      
      const card = screen.getByTestId('channel-card');
      await user.click(card);
      
      expect(defaultProps.onSelect).toHaveBeenCalledWith(mockChannel.id);
      expect(defaultProps.onSelect).toHaveBeenCalledTimes(1);
    });
    
    it('should toggle automation when button is clicked', async () => {
      const user = userEvent.setup();
      render(<ChannelCard {...defaultProps} variant="detailed" />);
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      await user.click(toggleButton);
      
      await waitFor(() => {
        expect(defaultProps.onToggleAutomation).toHaveBeenCalledWith(mockChannel.id);
      });
    });
    
    it('should show loading state during automation toggle', async () => {
      const slowToggle = vi.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
      render(
        <ChannelCard 
          {...defaultProps} 
          onToggleAutomation={slowToggle}
          variant="detailed"
        />
      );
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      fireEvent.click(toggleButton);
      
      // Check loading state appears
      expect(toggleButton).toBeDisabled();
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
      
      // Wait for operation to complete
      await waitFor(() => {
        expect(toggleButton).not.toBeDisabled();
        expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
      });
    });
  });
  
  describe('Error Handling', () => {
    it('should display error message when toggle fails', async () => {
      const failingToggle = vi.fn().mockRejectedValue(new Error('API Error'));
      render(
        <ChannelCard 
          {...defaultProps} 
          onToggleAutomation={failingToggle}
          variant="detailed"
        />
      );
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      fireEvent.click(toggleButton);
      
      await waitFor(() => {
        expect(screen.getByText(/API Error/)).toBeInTheDocument();
      });
    });
    
    it('should allow dismissing error messages', async () => {
      const failingToggle = vi.fn().mockRejectedValue(new Error('API Error'));
      const user = userEvent.setup();
      
      render(
        <ChannelCard 
          {...defaultProps} 
          onToggleAutomation={failingToggle}
          variant="detailed"
        />
      );
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      await user.click(toggleButton);
      
      await waitFor(() => {
        expect(screen.getByText(/API Error/)).toBeInTheDocument();
      });
      
      const dismissButton = screen.getByRole('button', { name: /dismiss/i });
      await user.click(dismissButton);
      
      expect(screen.queryByText(/API Error/)).not.toBeInTheDocument();
    });
  });
  
  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      render(<ChannelCard {...defaultProps} />);
      
      const card = screen.getByTestId('channel-card');
      expect(card).toHaveAttribute('role', 'article');
      expect(card).toHaveAttribute('aria-label', `Channel: ${mockChannel.name}`);
    });
    
    it('should be keyboard navigable', async () => {
      render(<ChannelCard {...defaultProps} variant="detailed" />);
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      
      // Tab to button
      toggleButton.focus();
      expect(toggleButton).toHaveFocus();
      
      // Trigger with Enter key
      fireEvent.keyDown(toggleButton, { key: 'Enter', code: 'Enter' });
      
      await waitFor(() => {
        expect(defaultProps.onToggleAutomation).toHaveBeenCalled();
      });
    });
    
    it('should announce status changes to screen readers', async () => {
      render(<ChannelCard {...defaultProps} variant="detailed" />);
      
      const toggleButton = screen.getByRole('button', { name: /toggle automation/i });
      fireEvent.click(toggleButton);
      
      await waitFor(() => {
        const liveRegion = screen.getByRole('status');
        expect(liveRegion).toHaveTextContent(/Automation toggled/);
      });
    });
  });
  
  describe('Performance', () => {
    it('should memoize expensive computations', () => {
      const computeSpy = vi.spyOn(console, 'log');
      const { rerender } = render(<ChannelCard {...defaultProps} />);
      
      // Re-render with same props
      rerender(<ChannelCard {...defaultProps} />);
      
      // Expensive computation should not run again
      expect(computeSpy).not.toHaveBeenCalledWith('Computing metrics');
    });
    
    it('should not re-render unnecessarily', () => {
      const renderSpy = vi.fn();
      const TrackedChannelCard = () => {
        renderSpy();
        return <ChannelCard {...defaultProps} />;
      };
      
      const { rerender } = render(<TrackedChannelCard />);
      expect(renderSpy).toHaveBeenCalledTimes(1);
      
      // Re-render parent with same props
      rerender(<TrackedChannelCard />);
      
      // Component should not re-render if props haven't changed
      expect(renderSpy).toHaveBeenCalledTimes(1);
    });
  });
});
```

### 3.2 Hook Testing Patterns

```typescript
// src/hooks/usePolling.test.ts
import { renderHook, act, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { usePolling } from './usePolling';

describe('usePolling Hook', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });
  
  it('should fetch data immediately on mount', async () => {
    const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });
    
    const { result } = renderHook(() => 
      usePolling(fetchFn, 60000)
    );
    
    // Initial state
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBe(null);
    
    // Wait for initial fetch
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.data).toEqual({ data: 'test' });
    });
    
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
  
  it('should poll at specified interval', async () => {
    const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });
    
    renderHook(() => usePolling(fetchFn, 5000)); // 5 second interval
    
    // Initial fetch
    await waitFor(() => {
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });
    
    // Advance time by 5 seconds
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    
    await waitFor(() => {
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });
    
    // Advance time by another 5 seconds
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    
    await waitFor(() => {
      expect(fetchFn).toHaveBeenCalledTimes(3);
    });
  });
  
  it('should handle errors gracefully', async () => {
    const error = new Error('Fetch failed');
    const fetchFn = vi.fn().mockRejectedValue(error);
    const onError = vi.fn();
    
    const { result } = renderHook(() => 
      usePolling(fetchFn, 60000, { onError })
    );
    
    await waitFor(() => {
      expect(result.current.error).toEqual(error);
      expect(result.current.loading).toBe(false);
      expect(onError).toHaveBeenCalledWith(error);
    });
  });
  
  it('should stop polling when disabled', async () => {
    const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });
    
    const { rerender } = renderHook(
      ({ enabled }) => usePolling(fetchFn, 5000, { enabled }),
      { initialProps: { enabled: true } }
    );
    
    // Initial fetch when enabled
    await waitFor(() => {
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });
    
    // Disable polling
    rerender({ enabled: false });
    
    // Advance time - should not trigger fetch
    act(() => {
      vi.advanceTimersByTime(10000);
    });
    
    // Still only 1 call
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
  
  it('should cleanup interval on unmount', async () => {
    const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });
    
    const { unmount } = renderHook(() => 
      usePolling(fetchFn, 5000)
    );
    
    await waitFor(() => {
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });
    
    // Unmount hook
    unmount();
    
    // Advance time after unmount
    act(() => {
      vi.advanceTimersByTime(10000);
    });
    
    // Should not fetch after unmount
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
});
```

### 3.3 Store Testing (Zustand)

```typescript
// src/stores/channelStore.test.ts
import { act, renderHook } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { useChannelStore } from './channelStore';
import { api } from '@/services/api';
import { mockChannels } from '@/test/fixtures/channels';

// Mock API module
vi.mock('@/services/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn()
  }
}));

describe('Channel Store (Zustand)', () => {
  beforeEach(() => {
    // Reset store state before each test
    useChannelStore.setState({
      channels: [],
      activeChannelId: null,
      channelsLoading: false,
      channelsError: null
    });
    
    vi.clearAllMocks();
  });
  
  describe('fetchChannels', () => {
    it('should fetch and store channels', async () => {
      vi.mocked(api.get).mockResolvedValue({
        data: {
          success: true,
          data: mockChannels
        }
      });
      
      const { result } = renderHook(() => useChannelStore());
      
      // Initial state
      expect(result.current.channels).toHaveLength(0);
      expect(result.current.channelsLoading).toBe(false);
      
      // Fetch channels
      await act(async () => {
        await result.current.fetchChannels();
      });
      
      // Verify state updates
      expect(result.current.channels).toEqual(mockChannels);
      expect(result.current.channelsLoading).toBe(false);
      expect(result.current.channelsError).toBe(null);
      
      // Verify API was called correctly
      expect(api.get).toHaveBeenCalledWith('/api/v1/channels');
    });
    
    it('should handle fetch errors', async () => {
      const error = new Error('Network error');
      vi.mocked(api.get).mockRejectedValue(error);
      
      const { result } = renderHook(() => useChannelStore());
      
      // Fetch channels
      await act(async () => {
        try {
          await result.current.fetchChannels();
        } catch (e) {
          // Expected to throw
        }
      });
      
      // Verify error state
      expect(result.current.channelsError).toEqual(error);
      expect(result.current.channelsLoading).toBe(false);
      expect(result.current.channels).toHaveLength(0);
    });
  });
  
  describe('createChannel', () => {
    it('should create a new channel', async () => {
      const newChannel = {
        id: 'new-1',
        name: 'New Channel',
        status: 'active'
      };
      
      vi.mocked(api.post).mockResolvedValue({
        data: {
          success: true,
          data: newChannel
        }
      });
      
      const { result } = renderHook(() => useChannelStore());
      
      // Create channel
      let createdChannel;
      await act(async () => {
        createdChannel = await result.current.createChannel({
          name: 'New Channel',
          niche: 'gaming'
        });
      });
      
      // Verify channel was added
      expect(result.current.channels).toHaveLength(1);
      expect(result.current.channels[0]).toEqual(newChannel);
      expect(createdChannel).toEqual(newChannel);
    });
    
    it('should enforce channel limit (5 max for MVP)', async () => {
      const { result } = renderHook(() => useChannelStore());
      
      // Set 5 channels (max limit)
      act(() => {
        useChannelStore.setState({
          channels: Array(5).fill(null).map((_, i) => ({
            id: `ch-${i}`,
            name: `Channel ${i}`,
            status: 'active'
          }))
        });
      });
      
      // Try to create 6th channel
      await expect(async () => {
        await act(async () => {
          await result.current.createChannel({
            name: 'Channel 6',
            niche: 'tech'
          });
        });
      }).rejects.toThrow('Channel limit reached');
      
      // Verify no API call was made
      expect(api.post).not.toHaveBeenCalled();
    });
  });
  
  describe('updateChannel', () => {
    it('should update channel data', async () => {
      const channel = mockChannels[0];
      
      const { result } = renderHook(() => useChannelStore());
      
      // Set initial channel
      act(() => {
        useChannelStore.setState({ channels: [channel] });
      });
      
      vi.mocked(api.patch).mockResolvedValue({
        data: { success: true, data: { ...channel, name: 'Updated Name' } }
      });
      
      // Update channel
      await act(async () => {
        await result.current.updateChannel(channel.id, { name: 'Updated Name' });
      });
      
      // Verify update
      expect(result.current.channels[0].name).toBe('Updated Name');
      expect(api.patch).toHaveBeenCalledWith(
        `/api/v1/channels/${channel.id}`,
        { name: 'Updated Name' }
      );
    });
  });
  
  describe('Selectors', () => {
    it('should compute active channel correctly', () => {
      const { result } = renderHook(() => useChannelStore());
      
      const channels = [
        { id: 'ch-1', name: 'Channel 1', status: 'active' },
        { id: 'ch-2', name: 'Channel 2', status: 'active' }
      ];
      
      act(() => {
        useChannelStore.setState({
          channels,
          activeChannelId: 'ch-2'
        });
      });
      
      const activeChannel = result.current.getActiveChannel();
      expect(activeChannel).toEqual(channels[1]);
    });
    
    it('should count active channels', () => {
      const { result } = renderHook(() => useChannelStore());
      
      act(() => {
        useChannelStore.setState({
          channels: [
            { id: 'ch-1', status: 'active' },
            { id: 'ch-2', status: 'paused' },
            { id: 'ch-3', status: 'active' },
            { id: 'ch-4', status: 'error' },
            { id: 'ch-5', status: 'active' }
          ]
        });
      });
      
      expect(result.current.getActiveChannelsCount()).toBe(3);
    });
  });
});
```

---

## 4. Integration Testing

### 4.1 API Integration Testing

```typescript
// src/services/api.integration.test.ts
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { rest } from 'msw';
import { api } from './api';

// Setup MSW server
const server = setupServer(
  rest.get('/api/v1/channels', (req, res, ctx) => {
    return res(
      ctx.json({
        success: true,
        data: [
          { id: '1', name: 'Channel 1', status: 'active' },
          { id: '2', name: 'Channel 2', status: 'paused' }
        ],
        metadata: {
          timestamp: new Date().toISOString(),
          requestId: 'test-123',
          processingTime: 50
        }
      })
    );
  }),
  
  rest.post('/api/v1/videos/generate', (req, res, ctx) => {
    return res(
      ctx.json({
        success: true,
        data: {
          id: 'video-1',
          status: 'queued',
          estimatedCompletion: new Date(Date.now() + 300000).toISOString()
        }
      })
    );
  }),
  
  rest.get('/api/v1/dashboard/overview', (req, res, ctx) => {
    return res(
      ctx.json({
        success: true,
        data: {
          metrics: {
            totalChannels: 5,
            activeChannels: 3,
            videosToday: 15,
            totalRevenue: { value: 1234.56, currency: 'USD' }
          }
        }
      })
    );
  })
);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('API Integration Tests', () => {
  describe('Channel API', () => {
    it('should fetch channels list', async () => {
      const response = await api.get('/api/v1/channels');
      
      expect(response.data.success).toBe(true);
      expect(response.data.data).toHaveLength(2);
      expect(response.data.data[0]).toHaveProperty('id');
      expect(response.data.data[0]).toHaveProperty('name');
      expect(response.data.data[0]).toHaveProperty('status');
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
                message: 'Internal server error'
              }
            })
          );
        })
      );
      
      await expect(api.get('/api/v1/channels')).rejects.toThrow();
    });
    
    it('should handle network timeouts', async () => {
      server.use(
        rest.get('/api/v1/channels', (req, res, ctx) => {
          return res(ctx.delay(35000)); // Exceed 30s timeout
        })
      );
      
      await expect(api.get('/api/v1/channels')).rejects.toThrow(/timeout/i);
    }, 40000);
  });
  
  describe('Video Generation API', () => {
    it('should submit video generation request', async () => {
      const request = {
        channelId: 'channel-1',
        topic: 'Tech Review',
        style: 'educational',
        length: 'medium',
        priority: 5
      };
      
      const response = await api.post('/api/v1/videos/generate', request);
      
      expect(response.data.success).toBe(true);
      expect(response.data.data.status).toBe('queued');
      expect(response.data.data.estimatedCompletion).toBeDefined();
    });
  });
  
  describe('Dashboard API', () => {
    it('should fetch dashboard metrics', async () => {
      const response = await api.get('/api/v1/dashboard/overview');
      
      expect(response.data.success).toBe(true);
      expect(response.data.data.metrics).toBeDefined();
      expect(response.data.data.metrics.totalChannels).toBe(5);
      expect(response.data.data.metrics.totalRevenue.value).toBe(1234.56);
    });
  });
});
```

### 4.2 Component Integration Testing

```typescript
// src/features/Dashboard.integration.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { rest } from 'msw';
import { Dashboard } from './Dashboard';
import { TestWrapper } from '@/test/utils';

const server = setupServer(
  // Mock all required endpoints for Dashboard
  rest.get('/api/v1/dashboard/overview', (req, res, ctx) => {
    return res(ctx.json({
      success: true,
      data: {
        metrics: {
          totalChannels: 5,
          activeChannels: 3,
          videosToday: 15,
          videosProcessing: 2,
          totalRevenue: { value: 5432.10, currency: 'USD' },
          totalCost: { value: 543.21, currency: 'USD' },
          profitMargin: 90,
          automationPercentage: 95
        },
        channels: [
          { id: '1', name: 'Tech Channel', status: 'active', videoCount: 50 },
          { id: '2', name: 'Gaming Channel', status: 'active', videoCount: 30 },
          { id: '3', name: 'Education Channel', status: 'paused', videoCount: 20 }
        ],
        recentVideos: [
          { id: 'v1', title: 'Latest Tech News', status: 'published' },
          { id: 'v2', title: 'Game Review', status: 'processing' }
        ]
      }
    }));
  }),
  
  rest.get('/api/v1/costs/breakdown', (req, res, ctx) => {
    return res(ctx.json({
      success: true,
      data: {
        ai_generation: 200,
        voice_synthesis: 150,
        storage: 50,
        api_calls: 143.21,
        total: 543.21
      }
    }));
  })
);

beforeAll(() => server.listen());
afterAll(() => server.close());

describe('Dashboard Integration', () => {
  it('should load and display all dashboard sections', async () => {
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );
    
    // Check loading state appears first
    expect(screen.getByTestId('dashboard-loading')).toBeInTheDocument();
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('dashboard-loading')).not.toBeInTheDocument();
    });
    
    // Verify metrics are displayed
    expect(screen.getByText('5')).toBeInTheDocument(); // Total channels
    expect(screen.getByText('3')).toBeInTheDocument(); // Active channels
    expect(screen.getByText('15')).toBeInTheDocument(); // Videos today
    expect(screen.getByText('$5,432.10')).toBeInTheDocument(); // Revenue
    expect(screen.getByText('90%')).toBeInTheDocument(); // Profit margin
    expect(screen.getByText('95%')).toBeInTheDocument(); // Automation
    
    // Verify channels are listed
    expect(screen.getByText('Tech Channel')).toBeInTheDocument();
    expect(screen.getByText('Gaming Channel')).toBeInTheDocument();
    expect(screen.getByText('Education Channel')).toBeInTheDocument();
    
    // Verify recent videos
    expect(screen.getByText('Latest Tech News')).toBeInTheDocument();
    expect(screen.getByText('Game Review')).toBeInTheDocument();
    
    // Verify cost breakdown loaded
    expect(screen.getByText(/AI Generation.*200/)).toBeInTheDocument();
    expect(screen.getByText(/Voice Synthesis.*150/)).toBeInTheDocument();
  });
  
  it('should handle polling updates (60 second intervals)', async () => {
    vi.useFakeTimers();
    
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );
    
    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('15')).toBeInTheDocument(); // Initial videos today
    });
    
    // Update server response for next poll
    server.use(
      rest.get('/api/v1/dashboard/overview', (req, res, ctx) => {
        return res(ctx.json({
          success: true,
          data: {
            metrics: {
              videosToday: 20 // Updated value
              // ... other metrics
            }
          }
        }));
      })
    );
    
    // Advance time by 60 seconds
    act(() => {
      vi.advanceTimersByTime(60000);
    });
    
    // Check updated value appears
    await waitFor(() => {
      expect(screen.getByText('20')).toBeInTheDocument();
    });
    
    vi.useRealTimers();
  });
  
  it('should handle error states gracefully', async () => {
    server.use(
      rest.get('/api/v1/dashboard/overview', (req, res, ctx) => {
        return res(
          ctx.status(500),
          ctx.json({
            success: false,
            error: {
              code: 'SERVER_ERROR',
              message: 'Failed to fetch dashboard data'
            }
          })
        );
      })
    );
    
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );
    
    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch dashboard data/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });
  });
});
```

---

## 5. Performance Testing

### 5.1 Component Performance Testing

```typescript
// src/test/performance/componentPerformance.test.ts
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { measureRenderTime, measureReRenderTime } from '@/test/utils/performance';
import { ChannelCard } from '@/components/channels/ChannelCard';
import { Dashboard } from '@/features/Dashboard';
import { mockChannel } from '@/test/fixtures';

describe('Component Performance Tests', () => {
  describe('Render Time Benchmarks', () => {
    it('ChannelCard should render in less than 16ms', () => {
      const renderTime = measureRenderTime(() => {
        render(<ChannelCard channel={mockChannel} />);
      });
      
      expect(renderTime).toBeLessThan(16); // 60fps threshold
    });
    
    it('Dashboard should render in less than 100ms', () => {
      const renderTime = measureRenderTime(() => {
        render(<Dashboard />);
      });
      
      expect(renderTime).toBeLessThan(100);
    });
    
    it('Large list (100 items) should render in less than 200ms', () => {
      const items = Array.from({ length: 100 }, (_, i) => ({
        ...mockChannel,
        id: `channel-${i}`
      }));
      
      const renderTime = measureRenderTime(() => {
        render(
          <div>
            {items.map(item => (
              <ChannelCard key={item.id} channel={item} />
            ))}
          </div>
        );
      });
      
      expect(renderTime).toBeLessThan(200);
    });
  });
  
  describe('Re-render Performance', () => {
    it('should not re-render when props are unchanged', () => {
      const { rerender } = render(<ChannelCard channel={mockChannel} />);
      
      const reRenderTime = measureReRenderTime(() => {
        rerender(<ChannelCard channel={mockChannel} />);
      });
      
      // Should be near 0 if properly memoized
      expect(reRenderTime).toBeLessThan(1);
    });
    
    it('should efficiently update when single prop changes', () => {
      const { rerender } = render(
        <ChannelCard channel={mockChannel} variant="default" />
      );
      
      const reRenderTime = measureReRenderTime(() => {
        rerender(<ChannelCard channel={mockChannel} variant="compact" />);
      });
      
      expect(reRenderTime).toBeLessThan(10);
    });
  });
  
  describe('Memory Usage', () => {
    it('should not leak memory on repeated mounting/unmounting', () => {
      const initialMemory = performance.memory?.usedJSHeapSize || 0;
      
      // Mount and unmount component 100 times
      for (let i = 0; i < 100; i++) {
        const { unmount } = render(<ChannelCard channel={mockChannel} />);
        unmount();
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = performance.memory?.usedJSHeapSize || 0;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Should not increase by more than 1MB
      expect(memoryIncrease).toBeLessThan(1024 * 1024);
    });
  });
});
```

### 5.2 Bundle Size Testing

```typescript
// src/test/bundleSize.test.ts
import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('Bundle Size Constraints', () => {
  const distPath = path.resolve(__dirname, '../../dist');
  
  it('should keep main bundle under 500KB', () => {
    const mainBundle = fs.readdirSync(distPath)
      .find(file => file.startsWith('index') && file.endsWith('.js'));
    
    if (mainBundle) {
      const stats = fs.statSync(path.join(distPath, mainBundle));
      const sizeInKB = stats.size / 1024;
      
      expect(sizeInKB).toBeLessThan(500);
    }
  });
  
  it('should keep total bundle size under 1MB', () => {
    const files = fs.readdirSync(distPath)
      .filter(file => file.endsWith('.js') || file.endsWith('.css'));
    
    const totalSize = files.reduce((acc, file) => {
      const stats = fs.statSync(path.join(distPath, file));
      return acc + stats.size;
    }, 0);
    
    const sizeInMB = totalSize / (1024 * 1024);
    
    expect(sizeInMB).toBeLessThan(1);
  });
  
  it('should properly split vendor chunks', () => {
    const vendorChunks = fs.readdirSync(distPath)
      .filter(file => file.includes('vendor'));
    
    // Should have separate vendor chunks
    expect(vendorChunks.length).toBeGreaterThan(0);
    
    // Each vendor chunk should be reasonably sized
    vendorChunks.forEach(chunk => {
      const stats = fs.statSync(path.join(distPath, chunk));
      const sizeInKB = stats.size / 1024;
      
      // No single vendor chunk should exceed 200KB
      expect(sizeInKB).toBeLessThan(200);
    });
  });
});
```

---

## 6. Performance Metrics Implementation

### 6.1 Real-time Performance Monitoring

```typescript
// src/utils/performanceMonitor.ts
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, PerformanceMetric[]> = new Map();
  private observers: PerformanceObserver[] = [];
  
  private constructor() {
    this.initializeObservers();
  }
  
  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }
  
  private initializeObservers(): void {
    // Observe paint timing
    this.observePaintTiming();
    
    // Observe layout shifts
    this.observeLayoutShifts();
    
    // Observe long tasks
    this.observeLongTasks();
    
    // Observe resource timing
    this.observeResourceTiming();
  }
  
  private observePaintTiming(): void {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          this.recordMetric('FCP', entry.startTime);
        } else if (entry.name === 'largest-contentful-paint') {
          this.recordMetric('LCP', entry.startTime);
        }
      }
    });
    
    observer.observe({ entryTypes: ['paint', 'largest-contentful-paint'] });
    this.observers.push(observer);
  }
  
  private observeLayoutShifts(): void {
    let clsScore = 0;
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const layoutShift = entry as any;
        if (!layoutShift.hadRecentInput) {
          clsScore += layoutShift.value;
          this.recordMetric('CLS', clsScore);
        }
      }
    });
    
    observer.observe({ entryTypes: ['layout-shift'] });
    this.observers.push(observer);
  }
  
  private observeLongTasks(): void {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > 50) { // Tasks longer than 50ms
          this.recordMetric('LongTask', entry.duration, {
            name: entry.name,
            startTime: entry.startTime
          });
          
          // Alert if task is extremely long
          if (entry.duration > 200) {
            console.warn(`Long task detected: ${entry.duration}ms`, entry);
          }
        }
      }
    });
    
    if ('PerformanceLongTaskTiming' in window) {
      observer.observe({ entryTypes: ['longtask'] });
      this.observers.push(observer);
    }
  }
  
  private observeResourceTiming(): void {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const resource = entry as PerformanceResourceTiming;
        
        // Track slow resources
        if (resource.duration > 1000) {
          this.recordMetric('SlowResource', resource.duration, {
            name: resource.name,
            type: resource.initiatorType,
            size: resource.transferSize
          });
        }
      }
    });
    
    observer.observe({ entryTypes: ['resource'] });
    this.observers.push(observer);
  }
  
  recordMetric(
    name: string, 
    value: number, 
    metadata?: Record<string, any>
  ): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    
    const metric: PerformanceMetric = {
      name,
      value,
      timestamp: Date.now(),
      metadata
    };
    
    this.metrics.get(name)!.push(metric);
    
    // Send to monitoring service if critical threshold exceeded
    this.checkThresholds(name, value);
    
    // Keep only last 100 metrics per type to prevent memory leak
    const metrics = this.metrics.get(name)!;
    if (metrics.length > 100) {
      metrics.shift();
    }
  }
  
  private checkThresholds(name: string, value: number): void {
    const thresholds: Record<string, number> = {
      FCP: 1800,      // 1.8 seconds
      LCP: 2500,      // 2.5 seconds
      FID: 100,       // 100ms
      CLS: 0.1,       // 0.1 cumulative
      LongTask: 200,  // 200ms
      SlowResource: 3000 // 3 seconds
    };
    
    if (thresholds[name] && value > thresholds[name]) {
      this.reportToMonitoring({
        type: 'threshold_exceeded',
        metric: name,
        value,
        threshold: thresholds[name],
        url: window.location.href
      });
    }
  }
  
  private reportToMonitoring(data: any): void {
    // Send to monitoring endpoint
    if (process.env.NODE_ENV === 'production') {
      fetch('/api/v1/metrics/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      }).catch(console.error);
    }
  }
  
  getMetrics(): Map<string, PerformanceMetric[]> {
    return this.metrics;
  }
  
  getSummary(): PerformanceSummary {
    const summary: PerformanceSummary = {};
    
    this.metrics.forEach((metrics, name) => {
      if (metrics.length > 0) {
        const values = metrics.map(m => m.value);
        summary[name] = {
          count: metrics.length,
          average: values.reduce((a, b) => a + b, 0) / values.length,
          min: Math.min(...values),
          max: Math.max(...values),
          p50: this.percentile(values, 50),
          p75: this.percentile(values, 75),
          p95: this.percentile(values, 95),
          p99: this.percentile(values, 99)
        };
      }
    });
    
    return summary;
  }
  
  private percentile(values: number[], p: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }
  
  cleanup(): void {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
    this.metrics.clear();
  }
}

// Types
interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

interface PerformanceSummary {
  [metric: string]: {
    count: number;
    average: number;
    min: number;
    max: number;
    p50: number;
    p75: number;
    p95: number;
    p99: number;
  };
}

// React Hook for performance monitoring
export function usePerformanceMonitor(componentName: string) {
  const monitor = useRef(PerformanceMonitor.getInstance());
  
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      monitor.current.recordMetric(`Component.${componentName}`, renderTime);
    };
  }, [componentName]);
  
  return monitor.current;
}
```

---

## 7. Accessibility Standards (WCAG 2.1)

### 7.1 Accessibility Testing Implementation

```typescript
// src/test/accessibility/a11y.test.tsx
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { describe, it, expect } from 'vitest';
import { ChannelCard } from '@/components/channels/ChannelCard';
import { Dashboard } from '@/features/Dashboard';
import { mockChannel } from '@/test/fixtures';

// Extend expect with jest-axe matchers
expect.extend(toHaveNoViolations);

describe('Accessibility Tests (WCAG 2.1 AA)', () => {
  describe('Component Accessibility', () => {
    it('ChannelCard should have no accessibility violations', async () => {
      const { container } = render(
        <ChannelCard channel={mockChannel} />
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
    
    it('Dashboard should have no accessibility violations', async () => {
      const { container } = render(<Dashboard />);
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
    
    it('Forms should have proper labels and error associations', async () => {
      const { container } = render(
        <form>
          <label htmlFor="channel-name">Channel Name</label>
          <input 
            id="channel-name" 
            type="text" 
            aria-required="true"
            aria-invalid="false"
            aria-describedby="channel-name-error"
          />
          <span id="channel-name-error" role="alert">
            Channel name is required
          </span>
        </form>
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
  
  describe('Keyboard Navigation', () => {
    it('should support keyboard navigation through interactive elements', () => {
      const { getAllByRole } = render(
        <div>
          <button>First Button</button>
          <a href="#">Link</a>
          <input type="text" />
          <button>Last Button</button>
        </div>
      );
      
      const interactiveElements = [
        ...getAllByRole('button'),
        ...getAllByRole('link'),
        ...getAllByRole('textbox')
      ];
      
      interactiveElements.forEach(element => {
        expect(element).toHaveAttribute('tabindex');
        // Verify tab index is not negative (unless intentionally removed from tab order)
        const tabIndex = element.getAttribute('tabindex');
        if (tabIndex) {
          expect(parseInt(tabIndex)).toBeGreaterThanOrEqual(-1);
        }
      });
    });
  });
  
  describe('ARIA Attributes', () => {
    it('should have proper ARIA labels for icon buttons', () => {
      const { getByRole } = render(
        <button aria-label="Delete channel">
          <svg>{/* Trash icon */}</svg>
        </button>
      );
      
      const button = getByRole('button', { name: /delete channel/i });
      expect(button).toHaveAttribute('aria-label');
    });
    
    it('should have live regions for dynamic content', () => {
      const { getByRole } = render(
        <div role="status" aria-live="polite" aria-atomic="true">
          Processing video...
        </div>
      );
      
      const status = getByRole('status');
      expect(status).toHaveAttribute('aria-live', 'polite');
      expect(status).toHaveAttribute('aria-atomic', 'true');
    });
  });
  
  describe('Color Contrast', () => {
    it('should maintain WCAG AA contrast ratios', () => {
      // This would typically be tested with tools like Pa11y or Lighthouse
      // For unit tests, we verify that proper CSS classes are applied
      
      const { container } = render(
        <div className="text-primary bg-white">
          Text with proper contrast
        </div>
      );
      
      const element = container.firstChild;
      expect(element).toHaveClass('text-primary');
      expect(element).toHaveClass('bg-white');
      
      // In real implementation, you'd check computed styles
      // and calculate actual contrast ratios
    });
  });
});
```

### 7.2 Accessibility Implementation Guidelines

```typescript
// src/components/common/AccessibleButton.tsx
import React from 'react';
import { Button, ButtonProps } from '@mui/material';

interface AccessibleButtonProps extends ButtonProps {
  label: string;
  description?: string;
  loading?: boolean;
  loadingText?: string;
}

export const AccessibleButton: React.FC<AccessibleButtonProps> = ({
  label,
  description,
  loading = false,
  loadingText = 'Loading...',
  disabled,
  onClick,
  ...props
}) => {
  return (
    <Button
      {...props}
      disabled={disabled || loading}
      onClick={onClick}
      aria-label={label}
      aria-describedby={description ? `${label}-description` : undefined}
      aria-busy={loading}
      aria-disabled={disabled || loading}
    >
      {loading ? (
        <>
          <span className="sr-only">{loadingText}</span>
          <CircularProgress size={20} aria-hidden="true" />
        </>
      ) : (
        label
      )}
      {description && (
        <span id={`${label}-description`} className="sr-only">
          {description}
        </span>
      )}
    </Button>
  );
};

// Accessible Form Field
export const AccessibleTextField: React.FC<TextFieldProps> = ({
  label,
  error,
  helperText,
  required,
  ...props
}) => {
  const fieldId = props.id || `field-${label?.toLowerCase().replace(/\s/g, '-')}`;
  const errorId = `${fieldId}-error`;
  const helperId = `${fieldId}-helper`;
  
  return (
    <div>
      <label htmlFor={fieldId}>
        {label}
        {required && <span aria-label="required">*</span>}
      </label>
      <TextField
        {...props}
        id={fieldId}
        error={error}
        aria-required={required}
        aria-invalid={error}
        aria-describedby={
          error ? errorId : helperText ? helperId : undefined
        }
      />
      {error && (
        <span id={errorId} role="alert" className="error-text">
          {helperText}
        </span>
      )}
      {!error && helperText && (
        <span id={helperId} className="helper-text">
          {helperText}
        </span>
      )}
    </div>
  );
};

// Skip Navigation Link
export const SkipToContent: React.FC = () => {
  return (
    <a 
      href="#main-content" 
      className="skip-link"
      onFocus={(e) => e.currentTarget.classList.add('focused')}
      onBlur={(e) => e.currentTarget.classList.remove('focused')}
    >
      Skip to main content
    </a>
  );
};

// Accessible Data Table
export const AccessibleTable: React.FC<TableProps> = ({
  caption,
  headers,
  rows,
  ...props
}) => {
  return (
    <table {...props} role="table">
      <caption>{caption}</caption>
      <thead>
        <tr role="row">
          {headers.map((header, index) => (
            <th key={index} role="columnheader" scope="col">
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex} role="row">
            {row.map((cell, cellIndex) => (
              <td key={cellIndex} role="cell">
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};
```

---

## 8. Code Splitting Strategy

### 8.1 Route-Based Code Splitting

```typescript
// src/routes/index.tsx
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { LoadingScreen } from '@/components/common/LoadingScreen';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

// Lazy load all route components with named chunks
const Dashboard = lazy(() => 
  import(
    /* webpackChunkName: "dashboard" */
    /* webpackPrefetch: true */
    '@/features/Dashboard'
  )
);

const Channels = lazy(() => 
  import(
    /* webpackChunkName: "channels" */
    '@/features/Channels'
  )
);

const Videos = lazy(() => 
  import(
    /* webpackChunkName: "videos" */
    '@/features/Videos'
  )
);

const Analytics = lazy(() => 
  import(
    /* webpackChunkName: "analytics" */
    '@/features/Analytics'
  )
);

const Settings = lazy(() => 
  import(
    /* webpackChunkName: "settings" */
    '@/features/Settings'
  )
);

export const AppRoutes: React.FC = () => {
  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingScreen />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/channels/*" element={<Channels />} />
          <Route path="/videos/*" element={<Videos />} />
          <Route path="/analytics/*" element={<Analytics />} />
          <Route path="/settings/*" element={<Settings />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
};
```

### 8.2 Component-Level Code Splitting

```typescript
// src/components/heavy/HeavyComponent.tsx
import { lazy, Suspense } from 'react';
import { Skeleton } from '@mui/material';

// Split heavy components that aren't immediately visible
const HeavyChart = lazy(() => 
  import(
    /* webpackChunkName: "heavy-chart" */
    './HeavyChart'
  )
);

const VideoEditor = lazy(() => 
  import(
    /* webpackChunkName: "video-editor" */
    './VideoEditor'
  )
);

export const LazyHeavyComponent: React.FC = () => {
  const [showChart, setShowChart] = useState(false);
  
  return (
    <div>
      <button onClick={() => setShowChart(true)}>
        Load Chart
      </button>
      
      {showChart && (
        <Suspense fallback={<Skeleton height={400} />}>
          <HeavyChart />
        </Suspense>
      )}
    </div>
  );
};

// Intersection Observer for lazy loading
export const LazyLoadOnScroll: React.FC<{ children: React.ReactNode }> = ({ 
  children 
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    
    if (ref.current) {
      observer.observe(ref.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div ref={ref}>
      {isVisible ? children : <Skeleton height={200} />}
    </div>
  );
};