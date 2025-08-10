# YTEMPIRE Documentation - Development Guide

## 7.1 Environment Setup

### Prerequisites

```bash
# Required Software Versions
Node.js: 18.x LTS
npm: 9.x
Git: 2.x
Docker: 24.x
Docker Compose: 2.x
```

### Initial Setup Script

```bash
#!/bin/bash
# setup.sh - Run this first time only

echo "🚀 Setting up YTEMPIRE Development Environment..."

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

### Environment Variables

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

### VS Code Configuration

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
  }
}
```

## 7.2 Development Workflow

### Git Workflow

```bash
# Branch naming convention
feature/JIRA-123-description  # New features
bugfix/JIRA-456-description   # Bug fixes
hotfix/JIRA-789-description   # Production hotfixes
chore/JIRA-012-description    # Maintenance tasks

# Commit message format
type(scope): subject

# Examples:
feat(dashboard): add revenue chart
fix(auth): resolve token refresh issue
docs(api): update endpoint documentation
style(ui): improve button consistency
refactor(state): simplify zustand stores
test(charts): add unit tests for recharts
chore(deps): update dependencies
```

### Development Scripts

```json
// package.json scripts
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
    "test": "vitest",
    "test:watch": "vitest --watch",
    "test:coverage": "vitest --coverage",
    "test:ui": "vitest --ui",
    
    // Linting & Formatting
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,css}\"",
    
    // Type Checking
    "type-check": "tsc --noEmit",
    "type-check:watch": "tsc --noEmit --watch",
    
    // Utilities
    "clean": "rm -rf dist node_modules/.vite",
    "setup:mock-data": "node scripts/setup-mock-data.js",
    "verify:setup": "node scripts/verify-setup.js"
  }
}
```

### Code Review Process

```markdown
## Pull Request Template

### Description
Brief description of changes

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

### Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No console.logs left
- [ ] Bundle size checked (<1MB)

### Screenshots (if applicable)
[Add screenshots here]

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
```

## 7.3 Testing Strategy

### Unit Testing

```typescript
// Example unit test for a component
import { render, screen, fireEvent } from '@testing-library/react';
import { MetricCard } from '@/components/MetricCard';

describe('MetricCard', () => {
  it('renders with correct value', () => {
    render(
      <MetricCard
        title="Revenue"
        value={1000}
        format="currency"
      />
    );
    
    expect(screen.getByText('Revenue')).toBeInTheDocument();
    expect(screen.getByText('$1,000.00')).toBeInTheDocument();
  });
  
  it('shows trend indicator when change provided', () => {
    render(
      <MetricCard
        title="Revenue"
        value={1000}
        change={10}
      />
    );
    
    expect(screen.getByText('10%')).toBeInTheDocument();
    expect(screen.getByTestId('trend-up-icon')).toBeInTheDocument();
  });
});
```

### Integration Testing

```typescript
// Integration test for dashboard
import { renderWithProviders } from '@/test/test-utils';
import { Dashboard } from '@/pages/Dashboard';
import { server } from '@/mocks/server';
import { rest } from 'msw';

describe('Dashboard Integration', () => {
  it('loads and displays dashboard data', async () => {
    const { findByText } = renderWithProviders(<Dashboard />);
    
    // Wait for data to load
    await findByText('5 Active Channels');
    await findByText('$10,000');
    
    // Verify charts rendered
    expect(screen.getByTestId('revenue-chart')).toBeInTheDocument();
    expect(screen.getByTestId('cost-breakdown')).toBeInTheDocument();
  });
  
  it('handles API errors gracefully', async () => {
    // Mock API error
    server.use(
      rest.get('/api/v1/dashboard/overview', (req, res, ctx) => {
        return res(ctx.status(500));
      })
    );
    
    const { findByText } = renderWithProviders(<Dashboard />);
    
    await findByText('Failed to load dashboard data');
  });
});
```

### E2E Testing

```typescript
// Playwright E2E test
import { test, expect } from '@playwright/test';

test.describe('User Journey', () => {
  test('complete channel setup flow', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'password');
    await page.click('button[type="submit"]');
    
    // Navigate to channels
    await page.waitForURL('/dashboard');
    await page.click('text=Channels');
    
    // Create new channel
    await page.click('text=New Channel');
    await page.fill('[name="name"]', 'Test Channel');
    await page.selectOption('[name="niche"]', 'tech');
    await page.click('text=Create Channel');
    
    // Verify channel created
    await expect(page.locator('text=Test Channel')).toBeVisible();
  });
});
```

## 7.4 Mock Data & Local Development

### Mock Server Setup

```javascript
// mock-server/index.js
const express = require('express');
const cors = require('cors');
const { faker } = require('@faker-js/faker');

const app = express();
app.use(cors());
app.use(express.json());

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
          channelLimit: 5
        }
      }
    });
  } else {
    res.status(401).json({
      success: false,
      error: {
        code: 'AUTH_INVALID_CREDENTIALS',
        message: 'Invalid credentials'
      }
    });
  }
});

// Mock dashboard data
app.get('/api/v1/dashboard/overview', (req, res) => {
  res.json({
    success: true,
    data: {
      metrics: generateMockMetrics(),
      channels: generateMockChannels(5),
      recentVideos: generateMockVideos(10)
    }
  });
});

function generateMockMetrics() {
  return {
    totalChannels: 5,
    activeChannels: 3,
    videosToday: faker.number.int({ min: 5, max: 15 }),
    revenueToday: faker.number.float({ min: 50, max: 200 }),
    costToday: faker.number.float({ min: 10, max: 50 }),
    automationPercentage: faker.number.int({ min: 85, max: 99 })
  };
}

app.listen(8000, () => {
  console.log('Mock server running on http://localhost:8000');
});
```

### Mock Data Utilities

```typescript
// src/utils/mock-data.ts
import { faker } from '@faker-js/faker';

export const mockData = {
  generateChannel: (overrides?: Partial<Channel>): Channel => ({
    id: faker.string.uuid(),
    name: faker.company.name() + ' Channel',
    status: faker.helpers.arrayElement(['active', 'paused', 'error']),
    videoCount: faker.number.int({ min: 10, max: 100 }),
    revenue: faker.number.float({ min: 100, max: 1000 }),
    ...overrides
  }),
  
  generateVideo: (overrides?: Partial<Video>): Video => ({
    id: faker.string.uuid(),
    title: faker.lorem.sentence(),
    status: faker.helpers.arrayElement(['queued', 'processing', 'completed', 'failed']),
    cost: faker.number.float({ min: 0.30, max: 0.50 }),
    createdAt: faker.date.recent().toISOString(),
    ...overrides
  }),
  
  generateMetrics: (): DashboardMetrics => ({
    videosProcessing: faker.number.int({ min: 0, max: 5 }),
    currentDailyCost: faker.number.float({ min: 10, max: 50 }),
    channelMetrics: {
      total: 5,
      active: faker.number.int({ min: 3, max: 5 }),
      paused: faker.number.int({ min: 0, max: 2 }),
      error: 0
    },
    videoMetrics: {
      today: faker.number.int({ min: 5, max: 15 }),
      successRate: faker.number.int({ min: 90, max: 98 })
    },
    financialMetrics: {
      revenueToday: faker.number.float({ min: 50, max: 200 }),
      costToday: faker.number.float({ min: 10, max: 50 }),
      roi: faker.number.float({ min: 200, max: 500 })
    }
  })
};
```

## 7.5 Code Standards & Best Practices

### TypeScript Standards

```typescript
// Type definitions should be explicit
interface UserProps {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';  // Use union types for known values
}

// Use const assertions for constants
const ROUTES = {
  dashboard: '/dashboard',
  channels: '/channels',
  settings: '/settings'
} as const;

// Avoid any - use unknown or generics
function processData<T>(data: T): T {
  // Process data
  return data;
}

// Use optional chaining and nullish coalescing
const userName = user?.profile?.name ?? 'Guest';
```

### React Best Practices

```typescript
// Use functional components with TypeScript
interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  variant = 'primary',
  disabled = false
}) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
};

// Use custom hooks for logic reuse
export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [value, delay]);
  
  return debouncedValue;
};
```

### Performance Guidelines

```typescript
// Memoize expensive computations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// Memoize callbacks
const handleClick = useCallback(() => {
  doSomething(id);
}, [id]);

// Use React.lazy for code splitting
const Analytics = lazy(() => 
  import(/* webpackChunkName: "analytics" */ './pages/Analytics')
);

// Virtualize long lists
import { FixedSizeList } from 'react-window';

const VirtualList = ({ items }) => (
  <FixedSizeList
    height={600}
    itemCount={items.length}
    itemSize={50}
    width="100%"
  >
    {({ index, style }) => (
      <div style={style}>
        {items[index].name}
      </div>
    )}
  </FixedSizeList>
);
```

### Accessibility Standards

```typescript
// Always include ARIA labels
<button
  aria-label="Close dialog"
  onClick={onClose}
>
  <CloseIcon />
</button>

// Use semantic HTML
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/dashboard">Dashboard</a></li>
    <li><a href="/channels">Channels</a></li>
  </ul>
</nav>

// Keyboard navigation support
const handleKeyDown = (e: React.KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    onClick();
  }
};

// Focus management
useEffect(() => {
  if (isOpen) {
    firstFocusableElement.current?.focus();
  }
}, [isOpen]);
```

### Error Handling

```typescript
// Error boundaries for components
class ErrorBoundary extends React.Component<Props, State> {
  state = { hasError: false };
  
  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Component error:', error, errorInfo);
    // Log to error reporting service
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback />;
    }
    
    return this.props.children;
  }
}

// Async error handling
const fetchData = async () => {
  try {
    setLoading(true);
    const data = await api.getData();
    setData(data);
  } catch (error) {
    setError(error.message);
    // Show user-friendly error message
    toast.error('Failed to load data');
  } finally {
    setLoading(false);
  }
};
```