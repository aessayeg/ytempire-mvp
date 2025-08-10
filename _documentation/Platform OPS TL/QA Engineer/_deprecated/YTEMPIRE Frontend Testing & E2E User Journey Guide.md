# YTEMPIRE Frontend Testing & E2E User Journey Guide
**Version 1.0 | January 2025**  
**Owner: QA Engineering Team**  
**Primary Author: QA Engineer**  
**Approved By: Platform Operations Lead**

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Frontend Architecture Overview](#2-frontend-architecture-overview)
3. [Testing Strategy](#3-testing-strategy)
4. [Unit Testing Framework](#4-unit-testing-framework)
5. [Component Testing](#5-component-testing)
6. [Integration Testing](#6-integration-testing)
7. [End-to-End Testing](#7-end-to-end-testing)
8. [User Journey Testing](#8-user-journey-testing)
9. [Performance Testing](#9-performance-testing)
10. [Accessibility Testing](#10-accessibility-testing)
11. [Cross-Browser Testing](#11-cross-browser-testing)
12. [Visual Regression Testing](#12-visual-regression-testing)
13. [Test Automation & CI/CD](#13-test-automation--cicd)

---

## 1. Executive Summary

This comprehensive guide defines the testing strategy for YTEMPIRE's frontend application, covering everything from unit tests to complete end-to-end user journeys.

### Core Testing Objectives
- **Quality**: Zero critical bugs in production
- **Coverage**: >90% code coverage for frontend
- **Performance**: <2s page load time
- **Accessibility**: WCAG 2.1 AA compliance
- **Compatibility**: Support for all major browsers

### Testing Scope
- React 18 component architecture
- Redux state management
- API integration layer
- Real-time WebSocket features
- Multi-channel dashboard
- Video generation workflows

### Success Metrics
- **Bug Detection Rate**: >95% before production
- **Test Execution Time**: <10 minutes for full suite
- **Flakiness Rate**: <1% of tests
- **Automation Coverage**: >80% of test cases

---

## 2. Frontend Architecture Overview

### 2.1 Technology Stack

```yaml
frontend_stack:
  framework: React 18.2
  state_management: Redux Toolkit
  routing: React Router v6
  ui_components: Material-UI v5
  styling: Tailwind CSS
  build_tool: Vite
  testing_libraries:
    - Jest
    - React Testing Library
    - Cypress
    - Playwright
  
  key_features:
    - Multi-channel dashboard
    - Real-time analytics
    - Video generation wizard
    - Content calendar
    - Revenue tracking
    - User management
```

### 2.2 Component Architecture

```typescript
// Core component structure
interface ComponentArchitecture {
  pages: {
    Dashboard: 'Main overview page',
    Channels: 'Channel management',
    Videos: 'Video library and generation',
    Analytics: 'Performance metrics',
    Settings: 'User and system settings'
  },
  
  components: {
    common: ['Header', 'Sidebar', 'Footer', 'LoadingSpinner'],
    dashboard: ['StatsCard', 'RevenueChart', 'ChannelList'],
    videos: ['VideoCard', 'VideoPlayer', 'GenerationWizard'],
    forms: ['ChannelForm', 'VideoForm', 'SettingsForm']
  },
  
  services: {
    api: 'API communication layer',
    websocket: 'Real-time updates',
    auth: 'Authentication service',
    storage: 'Local storage management'
  }
}
```

---

## 3. Testing Strategy

### 3.1 Testing Pyramid

```yaml
testing_pyramid:
  unit_tests: 60%
    - Individual functions
    - React hooks
    - Utility functions
    - Redux reducers
    
  integration_tests: 25%
    - Component interactions
    - API integrations
    - State management
    
  e2e_tests: 15%
    - Critical user journeys
    - Cross-browser scenarios
    - Production workflows
```

### 3.2 Test Coverage Requirements

```javascript
// coverage.config.js
module.exports = {
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/index.tsx',
    '!src/serviceWorker.ts',
    '!src/**/*.d.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 90,
      statements: 90,
    },
    './src/components/': {
      branches: 90,
      functions: 90,
      lines: 95,
      statements: 95,
    },
    './src/utils/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100,
    },
  },
};
```

### 3.3 Testing Principles

```yaml
testing_principles:
  isolation: "Tests should not depend on external services"
  deterministic: "Same input always produces same output"
  fast: "Unit tests <100ms, integration <1s, E2E <30s"
  maintainable: "Clear naming, single responsibility"
  comprehensive: "Test happy path, edge cases, and errors"

---

## 4. Unit Testing Framework

### 4.1 Unit Test Setup

```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleNameMapper: {
    '^@/(.*): '<rootDir>/src/$1',
    '\\.(css|less|scss|sass): 'identity-obj-proxy',
  },
  transform: {
    '^.+\\.(ts|tsx): 'ts-jest',
    '^.+\\.(js|jsx): 'babel-jest',
  },
  testMatch: [
    '**/__tests__/**/*.[jt]s?(x)',
    '**/?(*.)+(spec|test).[jt]s?(x)',
  ],
};

// setupTests.ts
import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';
import 'jest-canvas-mock';

// Configure Testing Library
configure({ testIdAttribute: 'data-testid' });

// Mock window objects
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
```

### 4.2 Testing React Hooks

```typescript
// hooks/useChannelData.test.tsx
import { renderHook, act, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { useChannelData } from './useChannelData';
import { mockChannelData } from '../__mocks__/channelData';

describe('useChannelData Hook', () => {
  let store: any;
  
  beforeEach(() => {
    store = configureStore({
      reducer: {
        channels: (state = { data: [] }) => state,
      },
    });
  });

  it('should fetch channel data on mount', async () => {
    const wrapper = ({ children }: any) => (
      <Provider store={store}>{children}</Provider>
    );

    const { result } = renderHook(() => useChannelData(), { wrapper });

    expect(result.current.loading).toBe(true);
    expect(result.current.channels).toEqual([]);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.channels).toEqual(mockChannelData);
  });

  it('should handle errors gracefully', async () => {
    const wrapper = ({ children }: any) => (
      <Provider store={store}>{children}</Provider>
    );

    // Mock API failure
    jest.spyOn(global, 'fetch').mockRejectedValueOnce(
      new Error('Network error')
    );

    const { result } = renderHook(() => useChannelData(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Failed to fetch channel data');
    expect(result.current.channels).toEqual([]);
  });

  it('should refresh data when requested', async () => {
    const wrapper = ({ children }: any) => (
      <Provider store={store}>{children}</Provider>
    );

    const { result } = renderHook(() => useChannelData(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    act(() => {
      result.current.refresh();
    });

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
```

### 4.3 Testing Redux Store

```typescript
// store/channelSlice.test.ts
import channelReducer, {
  addChannel,
  updateChannel,
  deleteChannel,
  setChannels,
  setLoading,
  setError,
} from './channelSlice';

describe('Channel Slice', () => {
  const initialState = {
    channels: [],
    loading: false,
    error: null,
  };

  it('should handle initial state', () => {
    expect(channelReducer(undefined, { type: 'unknown' })).toEqual(initialState);
  });

  it('should handle addChannel', () => {
    const newChannel = {
      id: '1',
      name: 'Tech Reviews',
      niche: 'Technology',
      status: 'active',
    };

    const actual = channelReducer(initialState, addChannel(newChannel));
    expect(actual.channels).toEqual([newChannel]);
  });

  it('should handle updateChannel', () => {
    const existingState = {
      ...initialState,
      channels: [
        { id: '1', name: 'Old Name', niche: 'Tech', status: 'active' },
      ],
    };

    const update = { id: '1', name: 'New Name' };
    const actual = channelReducer(existingState, updateChannel(update));
    
    expect(actual.channels[0].name).toBe('New Name');
    expect(actual.channels[0].niche).toBe('Tech'); // Unchanged
  });

  it('should handle deleteChannel', () => {
    const existingState = {
      ...initialState,
      channels: [
        { id: '1', name: 'Channel 1', niche: 'Tech', status: 'active' },
        { id: '2', name: 'Channel 2', niche: 'Gaming', status: 'active' },
      ],
    };

    const actual = channelReducer(existingState, deleteChannel('1'));
    expect(actual.channels).toHaveLength(1);
    expect(actual.channels[0].id).toBe('2');
  });

  it('should handle loading states', () => {
    const actual = channelReducer(initialState, setLoading(true));
    expect(actual.loading).toBe(true);
  });

  it('should handle errors', () => {
    const error = 'Failed to fetch channels';
    const actual = channelReducer(initialState, setError(error));
    expect(actual.error).toBe(error);
    expect(actual.loading).toBe(false);
  });
});
```

---

## 5. Component Testing

### 5.1 Component Testing Setup

```typescript
// test-utils.tsx
import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { configureStore } from '@reduxjs/toolkit';
import { theme } from './theme';
import rootReducer from './store/rootReducer';

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialState?: any;
  store?: any;
}

function customRender(
  ui: ReactElement,
  {
    initialState,
    store = configureStore({ reducer: rootReducer, preloadedState: initialState }),
    ...renderOptions
  }: CustomRenderOptions = {}
) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <Provider store={store}>
        <BrowserRouter>
          <ThemeProvider theme={theme}>
            {children}
          </ThemeProvider>
        </BrowserRouter>
      </Provider>
    );
  }

  return render(ui, { wrapper: Wrapper, ...renderOptions });
}

export * from '@testing-library/react';
export { customRender as render };
```

### 5.2 Testing React Components

```typescript
// components/ChannelCard.test.tsx
import { render, screen, fireEvent, waitFor } from '../test-utils';
import userEvent from '@testing-library/user-event';
import { ChannelCard } from './ChannelCard';
import { mockChannel } from '../__mocks__/channelData';

describe('ChannelCard Component', () => {
  const defaultProps = {
    channel: mockChannel,
    onEdit: jest.fn(),
    onDelete: jest.fn(),
    onViewAnalytics: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render channel information correctly', () => {
    render(<ChannelCard {...defaultProps} />);

    expect(screen.getByText(mockChannel.name)).toBeInTheDocument();
    expect(screen.getByText(mockChannel.niche)).toBeInTheDocument();
    expect(screen.getByText(`${mockChannel.videoCount} videos`)).toBeInTheDocument();
    expect(screen.getByText(`${mockChannel.revenue}`)).toBeInTheDocument();
  });

  it('should show active status indicator', () => {
    render(<ChannelCard {...defaultProps} />);
    
    const statusIndicator = screen.getByTestId('status-indicator');
    expect(statusIndicator).toHaveClass('bg-green-500');
  });

  it('should handle edit action', async () => {
    render(<ChannelCard {...defaultProps} />);
    
    const editButton = screen.getByLabelText('Edit channel');
    await userEvent.click(editButton);

    expect(defaultProps.onEdit).toHaveBeenCalledWith(mockChannel.id);
  });

  it('should handle delete with confirmation', async () => {
    render(<ChannelCard {...defaultProps} />);
    
    const deleteButton = screen.getByLabelText('Delete channel');
    await userEvent.click(deleteButton);

    // Confirmation dialog should appear
    expect(screen.getByText('Delete Channel?')).toBeInTheDocument();
    expect(screen.getByText(/This action cannot be undone/)).toBeInTheDocument();

    const confirmButton = screen.getByText('Confirm');
    await userEvent.click(confirmButton);

    expect(defaultProps.onDelete).toHaveBeenCalledWith(mockChannel.id);
  });

  it('should cancel delete action', async () => {
    render(<ChannelCard {...defaultProps} />);
    
    const deleteButton = screen.getByLabelText('Delete channel');
    await userEvent.click(deleteButton);

    const cancelButton = screen.getByText('Cancel');
    await userEvent.click(cancelButton);

    expect(defaultProps.onDelete).not.toHaveBeenCalled();
  });

  it('should display monetization badge when applicable', () => {
    const monetizedChannel = {
      ...mockChannel,
      monetizationStatus: 'approved',
    };

    render(<ChannelCard {...defaultProps} channel={monetizedChannel} />);
    
    expect(screen.getByTestId('monetization-badge')).toBeInTheDocument();
    expect(screen.getByText('Monetized')).toBeInTheDocument();
  });

  it('should handle loading state during actions', async () => {
    defaultProps.onEdit.mockImplementation(() => {
      return new Promise(resolve => setTimeout(resolve, 1000));
    });

    render(<ChannelCard {...defaultProps} />);
    
    const editButton = screen.getByLabelText('Edit channel');
    await userEvent.click(editButton);

    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
});
```

### 5.3 Testing Form Components

```typescript
// components/VideoGenerationForm.test.tsx
import { render, screen, waitFor } from '../test-utils';
import userEvent from '@testing-library/user-event';
import { VideoGenerationForm } from './VideoGenerationForm';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.post('/api/videos/generate', (req, res, ctx) => {
    return res(ctx.json({ id: '123', status: 'processing' }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('VideoGenerationForm', () => {
  const onSuccess = jest.fn();
  const onError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should validate required fields', async () => {
    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    const submitButton = screen.getByRole('button', { name: /generate video/i });
    await userEvent.click(submitButton);

    expect(screen.getByText('Title is required')).toBeInTheDocument();
    expect(screen.getByText('Topic is required')).toBeInTheDocument();
    expect(screen.getByText('Channel is required')).toBeInTheDocument();
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('should validate title length', async () => {
    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    const titleInput = screen.getByLabelText('Video Title');
    const longTitle = 'a'.repeat(101);
    
    await userEvent.type(titleInput, longTitle);
    await userEvent.tab(); // Trigger blur validation

    expect(screen.getByText('Title must be 100 characters or less')).toBeInTheDocument();
  });

  it('should submit form with valid data', async () => {
    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    // Fill form
    await userEvent.type(screen.getByLabelText('Video Title'), 'Test Video');
    await userEvent.type(screen.getByLabelText('Topic'), 'Technology Review');
    await userEvent.selectOptions(screen.getByLabelText('Channel'), 'channel-1');
    await userEvent.selectOptions(screen.getByLabelText('Voice'), 'voice-1');
    await userEvent.type(screen.getByLabelText('Keywords'), 'tech, review, gadget');

    const submitButton = screen.getByRole('button', { name: /generate video/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith({
        id: '123',
        status: 'processing',
      });
    });
  });

  it('should handle API errors', async () => {
    server.use(
      rest.post('/api/videos/generate', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Server error' }));
      })
    );

    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    // Fill minimal required fields
    await userEvent.type(screen.getByLabelText('Video Title'), 'Test Video');
    await userEvent.type(screen.getByLabelText('Topic'), 'Technology');
    await userEvent.selectOptions(screen.getByLabelText('Channel'), 'channel-1');

    const submitButton = screen.getByRole('button', { name: /generate video/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Server error');
    });

    expect(screen.getByText('Failed to generate video')).toBeInTheDocument();
  });

  it('should show cost estimation', async () => {
    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    await userEvent.selectOptions(screen.getByLabelText('Video Length'), '10');
    await userEvent.selectOptions(screen.getByLabelText('Voice'), 'premium-voice');

    await waitFor(() => {
      expect(screen.getByText(/Estimated Cost: \$0.85/)).toBeInTheDocument();
    });
  });

  it('should disable submit during processing', async () => {
    render(
      <VideoGenerationForm onSuccess={onSuccess} onError={onError} />
    );

    // Fill form
    await userEvent.type(screen.getByLabelText('Video Title'), 'Test Video');
    await userEvent.type(screen.getByLabelText('Topic'), 'Technology');
    await userEvent.selectOptions(screen.getByLabelText('Channel'), 'channel-1');

    const submitButton = screen.getByRole('button', { name: /generate video/i });
    await userEvent.click(submitButton);

    expect(submitButton).toBeDisabled();
    expect(screen.getByText(/generating/i)).toBeInTheDocument();
  });
});
```

---

## 6. Integration Testing

### 6.1 API Integration Tests

```typescript
// integration/api.test.ts
import { render, screen, waitFor } from '../test-utils';
import { Dashboard } from '../pages/Dashboard';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { mockDashboardData } from '../__mocks__/dashboardData';

const server = setupServer(
  rest.get('/api/dashboard', (req, res, ctx) => {
    return res(ctx.json(mockDashboardData));
  }),
  rest.get('/api/channels', (req, res, ctx) => {
    return res(ctx.json({ channels: mockDashboardData.channels }));
  }),
  rest.get('/api/analytics/summary', (req, res, ctx) => {
    return res(ctx.json(mockDashboardData.analytics));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Dashboard API Integration', () => {
  it('should load and display dashboard data', async () => {
    render(<Dashboard />);

    // Initially shows loading
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });

    // Verify data is displayed
    expect(screen.getByText('5 Active Channels')).toBeInTheDocument();
    expect(screen.getByText('$1,234.56')).toBeInTheDocument(); // Revenue
    expect(screen.getByText('45 Videos')).toBeInTheDocument();
  });

  it('should handle API error states', async () => {
    server.use(
      rest.get('/api/dashboard', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Internal server error' }));
      })
    );

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load dashboard/)).toBeInTheDocument();
    });

    // Should show retry button
    const retryButton = screen.getByRole('button', { name: /retry/i });
    expect(retryButton).toBeInTheDocument();
  });

  it('should refresh data on interval', async () => {
    jest.useFakeTimers();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });

    const initialRevenue = screen.getByText('$1,234.56');
    expect(initialRevenue).toBeInTheDocument();

    // Update mock data for next request
    server.use(
      rest.get('/api/dashboard', (req, res, ctx) => {
        return res(ctx.json({
          ...mockDashboardData,
          analytics: {
            ...mockDashboardData.analytics,
            revenue: 2345.67,
          },
        }));
      })
    );

    // Fast-forward 30 seconds (refresh interval)
    jest.advanceTimersByTime(30000);

    await waitFor(() => {
      expect(screen.getByText('$2,345.67')).toBeInTheDocument();
    });

    jest.useRealTimers();
  });
});
```

### 6.2 WebSocket Integration Tests

```typescript
// integration/websocket.test.tsx
import { render, screen, waitFor } from '../test-utils';
import { VideoGenerationStatus } from '../components/VideoGenerationStatus';
import WS from 'jest-websocket-mock';

describe('WebSocket Integration', () => {
  let server: WS;

  beforeEach(() => {
    server = new WS('ws://localhost:8080/ws');
  });

  afterEach(() => {
    WS.clean();
  });

  it('should receive real-time video generation updates', async () => {
    render(<VideoGenerationStatus videoId="123" />);

    await server.connected;

    // Initial status
    expect(screen.getByText('Waiting for updates...')).toBeInTheDocument();

    // Send status update
    server.send(JSON.stringify({
      type: 'VIDEO_STATUS',
      videoId: '123',
      status: 'processing',
      progress: 25,
    }));

    await waitFor(() => {
      expect(screen.getByText('Processing: 25%')).toBeInTheDocument();
    });

    // Send completion update
    server.send(JSON.stringify({
      type: 'VIDEO_STATUS',
      videoId: '123',
      status: 'completed',
      progress: 100,
      youtubeUrl: 'https://youtube.com/watch?v=abc123',
    }));

    await waitFor(() => {
      expect(screen.getByText('Video generated successfully!')).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /view on youtube/i })).toHaveAttribute(
        'href',
        'https://youtube.com/watch?v=abc123'
      );
    });
  });

  it('should handle connection errors', async () => {
    render(<VideoGenerationStatus videoId="123" />);

    await server.connected;
    server.error();

    await waitFor(() => {
      expect(screen.getByText('Connection lost. Reconnecting...')).toBeInTheDocument();
    });
  });

  it('should reconnect automatically', async () => {
    jest.useFakeTimers();
    render(<VideoGenerationStatus videoId="123" />);

    await server.connected;
    server.close();

    await waitFor(() => {
      expect(screen.getByText('Connection lost. Reconnecting...')).toBeInTheDocument();
    });

    // Create new server to simulate reconnection
    const newServer = new WS('ws://localhost:8080/ws');
    
    jest.advanceTimersByTime(5000); // Reconnection delay

    await newServer.connected;

    newServer.send(JSON.stringify({
      type: 'VIDEO_STATUS',
      videoId: '123',
      status: 'processing',
      progress: 50,
    }));

    await waitFor(() => {
      expect(screen.getByText('Processing: 50%')).toBeInTheDocument();
    });

    jest.useRealTimers();
  });
});
```

---

## 7. End-to-End Testing

### 7.1 Cypress E2E Setup

```javascript
// cypress.config.js
const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    viewportWidth: 1920,
    viewportHeight: 1080,
    video: true,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    requestTimeout: 10000,
    responseTimeout: 10000,
    
    setupNodeEvents(on, config) {
      // Custom tasks
      on('task', {
        seedDatabase() {
          // Seed test data
          return null;
        },
        clearDatabase() {
          // Clear test data
          return null;
        },
      });
    },
  },
  
  env: {
    apiUrl: 'http://localhost:8000/api',
    testUser: {
      email: 'test@ytempire.com',
      password: 'TestPassword123!',
    },
  },
});

// cypress/support/commands.js
Cypress.Commands.add('login', (email, password) => {
  cy.session([email, password], () => {
    cy.visit('/login');
    cy.get('[data-testid="email-input"]').type(email);
    cy.get('[data-testid="password-input"]').type(password);
    cy.get('[data-testid="login-button"]').click();
    cy.url().should('include', '/dashboard');
  });
});

Cypress.Commands.add('createChannel', (channelData) => {
  cy.visit('/channels/new');
  cy.get('[data-testid="channel-name"]').type(channelData.name);
  cy.get('[data-testid="channel-niche"]').select(channelData.niche);
  cy.get('[data-testid="youtube-channel-id"]').type(channelData.youtubeId);
  cy.get('[data-testid="create-channel-button"]').click();
  cy.url().should('include', '/channels');
  cy.contains(channelData.name).should('be.visible');
});
```

### 7.2 Critical User Journey Tests

```javascript
// cypress/e2e/user-journey.cy.js
describe('Complete User Journey', () => {
  beforeEach(() => {
    cy.task('clearDatabase');
    cy.task('seedDatabase');
  });

  it('should complete full video generation workflow', () => {
    // Step 1: Login
    cy.login(Cypress.env('testUser').email, Cypress.env('testUser').password);
    
    // Step 2: Navigate to dashboard
    cy.visit('/dashboard');
    cy.get('[data-testid="dashboard-header"]').should('contain', 'Dashboard');
    
    // Step 3: Create new channel
    cy.get('[data-testid="create-channel-button"]').click();
    cy.url().should('include', '/channels/new');
    
    cy.get('[data-testid="channel-name"]').type('Tech Reviews Channel');
    cy.get('[data-testid="channel-niche"]').select('Technology');
    cy.get('[data-testid="youtube-channel-id"]').type('UC_TEST_123456');
    cy.get('[data-testid="submit-channel"]').click();
    
    // Verify channel created
    cy.url().should('include', '/channels');
    cy.contains('Tech Reviews Channel').should('be.visible');
    
    // Step 4: Generate video
    cy.get('[data-testid="channel-card-Tech Reviews Channel"]')
      .find('[data-testid="generate-video-button"]')
      .click();
    
    cy.url().should('include', '/videos/generate');
    
    // Fill video generation form
    cy.get('[data-testid="video-title"]').type('iPhone 15 Pro Review');
    cy.get('[data-testid="video-topic"]').type('Comprehensive review of the new iPhone');
    cy.get('[data-testid="video-length"]').select('10');
    cy.get('[data-testid="voice-selection"]').select('professional-male');
    cy.get('[data-testid="keywords"]').type('iphone, review, tech, apple');
    
    // Check cost estimation
    cy.get('[data-testid="cost-estimate"]').should('contain', '$0.85');
    
    // Submit generation
    cy.get('[data-testid="generate-button"]').click();
    
    // Step 5: Monitor generation progress
    cy.get('[data-testid="generation-status"]').should('be.visible');
    cy.get('[data-testid="progress-bar"]').should('be.visible');
    
    // Wait for generation (mocked in test environment)
    cy.get('[data-testid="generation-complete"]', { timeout: 30000 })
      .should('be.visible');
    
    // Step 6: Verify video in library
    cy.visit('/videos');
    cy.contains('iPhone 15 Pro Review').should('be.visible');
    
    // Step 7: Check analytics
    cy.get('[data-testid="video-card-iPhone 15 Pro Review"]')
      .find('[data-testid="view-analytics"]')
      .click();
    
    cy.url().should('include', '/analytics');
    cy.get('[data-testid="video-stats"]').should('be.visible');
  });

  it('should handle bulk video generation', () => {
    cy.login(Cypress.env('testUser').email, Cypress.env('testUser').password);
    
    // Navigate to bulk generation
    cy.visit('/videos/bulk-generate');
    
    // Upload CSV file
    cy.get('[data-testid="csv-upload"]').selectFile('cypress/fixtures/bulk-videos.csv');
    
    // Review parsed data
    cy.get('[data-testid="parsed-videos-table"]').should('be.visible');
    cy.get('[data-testid="video-row"]').should('have.length', 5);
    
    // Select channel for all
    cy.get('[data-testid="select-all-channel"]').select('Tech Reviews Channel');
    
    // Start bulk generation
    cy.get('[data-testid="start-bulk-generation"]').click();
    
    // Monitor progress
    cy.get('[data-testid="bulk-progress"]').should('be.visible');
    cy.get('[data-testid="completed-count"]').should('contain', '5 / 5');
    
    // Verify all videos created
    cy.visit('/videos');
    cy.get('[data-testid="video-card"]').should('have.length.at.least', 5);
  });
});
```

---

## 8. User Journey Testing

### 8.1 New User Onboarding Journey

```javascript
// cypress/e2e/onboarding-journey.cy.js
describe('New User Onboarding Journey', () => {
  it('should complete entire onboarding flow', () => {
    // Step 1: Registration
    cy.visit('/register');
    
    cy.get('[data-testid="email"]').type('newuser@test.com');
    cy.get('[data-testid="username"]').type('newuser123');
    cy.get('[data-testid="password"]').type('SecurePass123!');
    cy.get('[data-testid="confirm-password"]').type('SecurePass123!');
    cy.get('[data-testid="terms-checkbox"]').check();
    cy.get('[data-testid="register-button"]').click();
    
    // Step 2: Email verification (mocked)
    cy.url().should('include', '/verify-email');
    cy.get('[data-testid="verification-code"]').type('123456');
    cy.get('[data-testid="verify-button"]').click();
    
    // Step 3: Onboarding wizard
    cy.url().should('include', '/onboarding');
    
    // Welcome screen
    cy.get('[data-testid="welcome-message"]').should('be.visible');
    cy.get('[data-testid="get-started-button"]').click();
    
    // Business goals
    cy.get('[data-testid="goal-passive-income"]').click();
    cy.get('[data-testid="goal-brand-building"]').click();
    cy.get('[data-testid="next-button"]').click();
    
    // Experience level
    cy.get('[data-testid="experience-beginner"]').click();
    cy.get('[data-testid="next-button"]').click();
    
    // Niche selection
    cy.get('[data-testid="niche-technology"]').click();
    cy.get('[data-testid="niche-gaming"]').click();
    cy.get('[data-testid="next-button"]').click();
    
    // Subscription plan
    cy.get('[data-testid="plan-starter"]').click();
    cy.get('[data-testid="next-button"]').click();
    
    // Step 4: First channel setup
    cy.url().should('include', '/channels/setup');
    
    cy.get('[data-testid="channel-name"]').type('My First Channel');
    cy.get('[data-testid="channel-niche"]').select('Technology');
    cy.get('[data-testid="target-audience"]').type('Tech enthusiasts aged 18-35');
    cy.get('[data-testid="content-frequency"]').select('3-per-week');
    cy.get('[data-testid="create-first-channel"]').click();
    
    // Step 5: Tutorial
    cy.get('[data-testid="tutorial-overlay"]').should('be.visible');
    cy.get('[data-testid="tutorial-step-1"]').should('be.visible');
    cy.get('[data-testid="next-step"]').click();
    cy.get('[data-testid="tutorial-step-2"]').should('be.visible');
    cy.get('[data-testid="next-step"]').click();
    cy.get('[data-testid="tutorial-step-3"]').should('be.visible');
    cy.get('[data-testid="finish-tutorial"]').click();
    
    // Step 6: Dashboard
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="welcome-banner"]').should('contain', 'Welcome to YTEMPIRE');
    cy.get('[data-testid="quick-actions"]').should('be.visible');
  });
});
```

### 8.2 Revenue Generation Journey

```javascript
// cypress/e2e/revenue-journey.cy.js
describe('Revenue Generation Journey', () => {
  beforeEach(() => {
    cy.login('user@test.com', 'password');
  });

  it('should track complete revenue generation flow', () => {
    // Step 1: Check monetization eligibility
    cy.visit('/monetization');
    
    cy.get('[data-testid="eligibility-checker"]').should('be.visible');
    cy.get('[data-testid="check-channel"]').select('Tech Reviews');
    cy.get('[data-testid="check-eligibility"]').click();
    
    // View requirements
    cy.get('[data-testid="requirements-list"]').should('be.visible');
    cy.get('[data-testid="subscribers-requirement"]').should('contain', '✓ 1,000+');
    cy.get('[data-testid="watch-hours-requirement"]').should('contain', '✓ 4,000+');
    
    // Step 2: Enable monetization
    cy.get('[data-testid="enable-monetization"]').click();
    
    // Connect AdSense
    cy.get('[data-testid="connect-adsense"]').click();
    cy.get('[data-testid="adsense-account"]').type('pub-123456789');
    cy.get('[data-testid="confirm-adsense"]').click();
    
    // Step 3: Configure revenue streams
    cy.visit('/revenue/configure');
    
    // Enable ads
    cy.get('[data-testid="enable-ads"]').check();
    cy.get('[data-testid="ad-frequency"]').select('moderate');
    
    // Add affiliate links
    cy.get('[data-testid="add-affiliate"]').click();
    cy.get('[data-testid="affiliate-program"]').select('Amazon Associates');
    cy.get('[data-testid="affiliate-id"]').type('ytempire-20');
    cy.get('[data-testid="save-affiliate"]').click();
    
    // Step 4: Generate monetized video
    cy.visit('/videos/generate');
    
    cy.get('[data-testid="video-title"]').type('Best Tech Gadgets 2025');
    cy.get('[data-testid="monetization-toggle"]').check();
    cy.get('[data-testid="include-affiliates"]').check();
    cy.get('[data-testid="generate-button"]').click();
    
    // Step 5: Track revenue
    cy.visit('/analytics/revenue');
    
    cy.get('[data-testid="revenue-dashboard"]').should('be.visible');
    cy.get('[data-testid="total-revenue"]').should('exist');
    cy.get('[data-testid="revenue-chart"]').should('be.visible');
    
    // View detailed breakdown
    cy.get('[data-testid="revenue-breakdown"]').click();
    cy.get('[data-testid="ad-revenue"]').should('be.visible');
    cy.get('[data-testid="affiliate-revenue"]').should('be.visible');
  });
});
```

---

## 9. Performance Testing

### 9.1 Frontend Performance Tests

```javascript
// performance/lighthouse.test.js
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

describe('Performance Metrics', () => {
  let chrome;

  beforeAll(async () => {
    chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
  });

  afterAll(async () => {
    await chrome.kill();
  });

  test('Dashboard performance should meet targets', async () => {
    const options = {
      logLevel: 'info',
      output: 'json',
      port: chrome.port,
    };

    const runnerResult = await lighthouse('http://localhost:3000/dashboard', options);
    const report = runnerResult.lhr;

    // Performance score
    expect(report.categories.performance.score).toBeGreaterThanOrEqual(0.9);

    // Core Web Vitals
    const metrics = report.audits.metrics.details.items[0];
    
    // First Contentful Paint < 1.8s
    expect(metrics.firstContentfulPaint).toBeLessThan(1800);
    
    // Largest Contentful Paint < 2.5s
    expect(metrics.largestContentfulPaint).toBeLessThan(2500);
    
    // Cumulative Layout Shift < 0.1
    expect(metrics.cumulativeLayoutShift).toBeLessThan(0.1);
    
    // Total Blocking Time < 200ms
    expect(metrics.totalBlockingTime).toBeLessThan(200);
  });

  test('Bundle size should be within limits', async () => {
    const stats = require('../build/bundle-stats.json');
    
    const mainBundle = stats.assets.find(a => a.name.startsWith('main'));
    const vendorBundle = stats.assets.find(a => a.name.startsWith('vendor'));
    
    // Main bundle < 200KB
    expect(mainBundle.size).toBeLessThan(200 * 1024);
    
    // Vendor bundle < 500KB
    expect(vendorBundle.size).toBeLessThan(500 * 1024);
    
    // Total JS < 700KB
    const totalJS = stats.assets
      .filter(a => a.name.endsWith('.js'))
      .reduce((sum, a) => sum + a.size, 0);
    
    expect(totalJS).toBeLessThan(700 * 1024);
  });
});
```

### 9.2 Load Testing

```javascript
// performance/load-test.js
const { test, expect } = require('@playwright/test');

test.describe('Load Testing', () => {
  test('should handle 50 concurrent users', async ({ browser }) => {
    const users = [];
    const userCount = 50;
    
    // Create concurrent user sessions
    for (let i = 0; i < userCount; i++) {
      users.push(browser.newContext());
    }
    
    const contexts = await Promise.all(users);
    const pages = await Promise.all(contexts.map(ctx => ctx.newPage()));
    
    const startTime = Date.now();
    
    // All users navigate to dashboard simultaneously
    const navigationPromises = pages.map(page => 
      page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' })
    );
    
    await Promise.all(navigationPromises);
    
    const loadTime = Date.now() - startTime;
    
    // Average load time should be under 3 seconds
    expect(loadTime / userCount).toBeLessThan(3000);
    
    // Verify all pages loaded successfully
    for (const page of pages) {
      const title = await page.title();
      expect(title).toContain('YTEMPIRE');
    }
    
    // Cleanup
    await Promise.all(contexts.map(ctx => ctx.close()));
  });

  test('should handle rapid user interactions', async ({ page }) => {
    await page.goto('http://localhost:3000/channels');
    
    // Rapidly click multiple buttons
    const actions = [];
    
    for (let i = 0; i < 20; i++) {
      actions.push(
        page.click('[data-testid="refresh-button"]'),
        page.click('[data-testid="filter-toggle"]'),
        page.click('[data-testid="sort-button"]')
      );
    }
    
    const startTime = Date.now();
    await Promise.all(actions);
    const duration = Date.now() - startTime;
    
    // Should handle all interactions within 5 seconds
    expect(duration).toBeLessThan(5000);
    
    // Page should still be responsive
    await expect(page.locator('[data-testid="channel-list"]')).toBeVisible();
  });
});
```

---

## 10. Accessibility Testing

### 10.1 WCAG Compliance Tests

```javascript
// accessibility/wcag.test.js
const { test, expect } = require('@playwright/test');
const { injectAxe, checkA11y } = require('axe-playwright');

test.describe('Accessibility Compliance', () => {
  test('Dashboard should be accessible', async ({ page }) => {
    await page.goto('http://localhost:3000/dashboard');
    await injectAxe(page);
    
    const violations = await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: {
        html: true,
      },
    });
    
    expect(violations).toBeNull();
  });

  test('Forms should be keyboard navigable', async ({ page }) => {
    await page.goto('http://localhost:3000/videos/generate');
    
    // Tab through form elements
    await page.keyboard.press('Tab'); // Focus on first input
    await expect(page.locator('[data-testid="video-title"]')).toBeFocused();
    
    await page.keyboard.press('Tab'); // Next input
    await expect(page.locator('[data-testid="video-topic"]')).toBeFocused();
    
    // Enter key should submit when on button
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await expect(page.locator('[data-testid="generate-button"]')).toBeFocused();
    
    await page.keyboard.press('Enter');
    // Should trigger validation
    await expect(page.locator('.error-message')).toBeVisible();
  });

  test('Screen reader compatibility', async ({ page }) => {
    await page.goto('http://localhost:3000');
    
    // Check for ARIA labels
    const buttons = await page.$('button');
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      expect(ariaLabel || text).toBeTruthy();
    }
    
    // Check for alt text on images
    const images = await page.$('img');
    for (const img of images) {
      const alt = await img.getAttribute('alt');
      expect(alt).toBeTruthy();
    }
    
    // Check for proper heading hierarchy
    const h1Count = await page.$eval('h1', elements => elements.length);
    expect(h1Count).toBe(1);
    
    const headings = await page.$eval('h1, h2, h3, h4, h5, h6', elements =>
      elements.map(el => ({ tag: el.tagName, text: el.textContent }))
    );
    
    // Verify logical heading order
    let lastLevel = 0;
    for (const heading of headings) {
      const level = parseInt(heading.tag.substring(1));
      expect(level - lastLevel).toBeLessThanOrEqual(1);
      lastLevel = level;
    }
  });

  test('Color contrast should meet WCAG AA standards', async ({ page }) => {
    await page.goto('http://localhost:3000');
    await injectAxe(page);
    
    const violations = await checkA11y(page, null, {
      runOnly: ['color-contrast'],
    });
    
    expect(violations).toBeNull();
  });
});
```

---

## 11. Cross-Browser Testing

### 11.1 Browser Compatibility Tests

```javascript
// cross-browser/compatibility.test.js
const { chromium, firefox, webkit } = require('playwright');

describe('Cross-Browser Compatibility', () => {
  const browsers = [
    { name: 'Chrome', launch: chromium },
    { name: 'Firefox', launch: firefox },
    { name: 'Safari', launch: webkit },
  ];

  browsers.forEach(({ name, launch }) => {
    describe(`${name} Browser`, () => {
      let browser;
      let page;

      beforeAll(async () => {
        browser = await launch();
        const context = await browser.newContext();
        page = await context.newPage();
      });

      afterAll(async () => {
        await browser.close();
      });

      test('should render dashboard correctly', async () => {
        await page.goto('http://localhost:3000/dashboard');
        
        // Check main elements are visible
        await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
        await expect(page.locator('[data-testid="stats-grid"]')).toBeVisible();
        await expect(page.locator('[data-testid="channel-list"]')).toBeVisible();
        
        // Take screenshot for visual comparison
        await page.screenshot({ 
          path: `screenshots/${name.toLowerCase()}-dashboard.png`,
          fullPage: true,
        });
      });

      test('should handle video playback', async () => {
        await page.goto('http://localhost:3000/videos/preview/123');
        
        const video = page.locator('video');
        await expect(video).toBeVisible();
        
        // Play video
        await video.click();
        await page.waitForTimeout(1000);
        
        // Check if playing
        const isPlaying = await video.evaluate(vid => !vid.paused);
        expect(isPlaying).toBe(true);
      });

      test('should support drag and drop', async () => {
        await page.goto('http://localhost:3000/content-calendar');
        
        const draggable = page.locator('[data-testid="draggable-video-1"]');
        const dropzone = page.locator('[data-testid="calendar-slot-monday"]');
        
        await draggable.dragTo(dropzone);
        
        // Verify item was moved
        await expect(dropzone.locator('[data-testid="draggable-video-1"]')).toBeVisible();
      });
    });
  });
});
```

---

## 12. Visual Regression Testing

### 12.1 Visual Regression Setup

```javascript
// visual-regression/visual.test.js
const { test, expect } = require('@playwright/test');

test.describe('Visual Regression Tests', () => {
  test('Dashboard visual consistency', async ({ page }) => {
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Hide dynamic content for consistent screenshots
    await page.evaluate(() => {
      // Hide timestamps
      document.querySelectorAll('[data-testid*="timestamp"]').forEach(el => {
        el.textContent = '2025-01-01 00:00:00';
      });
      
      // Hide random IDs
      document.querySelectorAll('[data-testid*="id"]').forEach(el => {
        el.textContent = 'ID-PLACEHOLDER';
      });
    });
    
    await expect(page).toHaveScreenshot('dashboard.png', {
      fullPage: true,
      threshold: 0.2, // 20% difference threshold
      maxDiffPixels: 100,
    });
  });

  test('Component visual tests', async ({ page }) => {
    await page.goto('http://localhost:3000/components');
    
    // Test each component variant
    const components = [
      'button-primary',
      'button-secondary',
      'card-default',
      'card-highlighted',
      'input-normal',
      'input-error',
      'modal-open',
    ];
    
    for (const component of components) {
      const element = page.locator(`[data-testid="${component}"]`);
      await expect(element).toHaveScreenshot(`${component}.png`);
    }
  });

  test('Responsive design breakpoints', async ({ page }) => {
    const viewports = [
      { name: 'mobile', width: 375, height: 667 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'desktop', width: 1920, height: 1080 },
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto('http://localhost:3000/dashboard');
      await page.waitForLoadState('networkidle');
      
      await expect(page).toHaveScreenshot(`dashboard-${viewport.name}.png`, {
        fullPage: true,
      });
    }
  });
});
```

---

## 13. Test Automation & CI/CD

### 13.1 CI/CD Pipeline Configuration

```yaml
# .github/workflows/frontend-tests.yml
name: Frontend Testing Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run unit tests
        run: npm run test:unit -- --coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
          flags: frontend-unit

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Start test server
        run: |
          npm run build
          npm run preview &
          npx wait-on http://localhost:3000
      
      - name: Run integration tests
        run: npm run test:integration

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Install Playwright browsers
        run: npx playwright install --with-deps
      
      - name: Start application
        run: |
          npm run build
          npm run preview &
          npx wait-on http://localhost:3000
      
      - name: Run E2E tests
        run: npm run test:e2e
      
      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: e2e-artifacts
          path: |
            test-results/
            screenshots/
            videos/

  visual-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Install Playwright
        run: npx playwright install --with-deps chromium
      
      - name: Run visual tests
        run: npm run test:visual
      
      - name: Upload visual diff
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: visual-diff
          path: test-results/

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build application
        run: npm run build
      
      - name: Run Lighthouse CI
        run: |
          npm install -g @lhci/cli
          lhci autorun
      
      - name: Upload Lighthouse report
        uses: actions/upload-artifact@v3
        with:
          name: lighthouse-report
          path: .lighthouseci/
```

### 13.2 Test Execution Scripts

```json
// package.json
{
  "scripts": {
    "test": "npm run test:unit && npm run test:integration",
    "test:unit": "jest --testPathPattern=.*\\.test\\.(ts|tsx)$",
    "test:integration": "jest --testPathPattern=.*\\.integration\\.(ts|tsx)$",
    "test:e2e": "cypress run",
    "test:e2e:open": "cypress open",
    "test:visual": "playwright test visual-regression/",
    "test:a11y": "playwright test accessibility/",
    "test:performance": "node performance/run-tests.js",
    "test:all": "npm run test && npm run test:e2e && npm run test:visual",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:ci": "npm run test:coverage && npm run test:e2e"
  }
}
```

---

## Appendix A: Test Data Management

### Mock Data Factories

```typescript
// __mocks__/factories.ts
export const createMockUser = (overrides = {}) => ({
  id: 'user-123',
  email: 'test@ytempire.com',
  username: 'testuser',
  subscriptionTier: 'professional',
  channels: [],
  createdAt: '2025-01-01T00:00:00Z',
  ...overrides,
});

export const createMockChannel = (overrides = {}) => ({
  id: 'channel-123',
  name: 'Test Channel',
  niche: 'Technology',
  youtubeChannelId: 'UC_TEST_123',
  status: 'active',
  videoCount: 10,
  subscriberCount: 1000,
  revenue: 123.45,
  ...overrides,
});

export const createMockVideo = (overrides = {}) => ({
  id: 'video-123',
  channelId: 'channel-123',
  title: 'Test Video',
  description: 'Test description',
  youtubeVideoId: 'abc123',
  status: 'published',
  generationCost: 0.85,
  viewCount: 1000,
  revenue: 12.34,
  ...overrides,
});
```

---

## Document Control

- **Version**: 1.0
- **Last Updated**: January 2025
- **Review Schedule**: Bi-weekly during MVP, Monthly post-launch
- **Owner**: QA Engineering Team
- **Next Review**: End of Week 2

**Approval Chain:**
1. QA Engineer (Author) ✅
2. Platform Operations Lead (Review) ✅
3. Frontend Team Lead (Technical Review)