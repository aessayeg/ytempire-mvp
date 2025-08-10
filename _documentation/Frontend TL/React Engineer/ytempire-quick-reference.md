# YTEMPIRE React Engineer - Quick Reference
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: Quick Reference & Cheat Sheet

---

## 1. Critical Information

### 1.1 MVP Constraints

```yaml
MUST HAVE:
  Components: 30-40 maximum (track count!)
  Screens: 20-25 total
  Stores: 5 Zustand stores
  Charts: 5-7 Recharts only
  Bundle: < 1MB
  Load Time: < 2 seconds
  Coverage: 70% minimum
  Users: 50 beta users
  Channels: 5 per user max

MUST NOT:
  ❌ Redux (use Zustand)
  ❌ D3.js (use Recharts)
  ❌ Mobile responsive (desktop only)
  ❌ Real-time WebSocket (except 3 events)
  ❌ Class components (functional only)
  ❌ Complex animations
  ❌ Exceed 40 components
```

### 1.2 Technology Stack

```typescript
// CONFIRMED STACK - DO NOT DEVIATE
const techStack = {
  framework: "React 18.2.0",
  language: "TypeScript 5.3",
  state: "Zustand 4.4",        // NOT Redux
  ui: "Material-UI 5.14",      // NOT Tailwind
  charts: "Recharts 2.10",     // NOT D3.js
  build: "Vite 5.0",
  routing: "React Router 6.20",
  http: "Axios 1.6",
  testing: "Vitest + RTL"
};
```

---

## 2. Common Commands

### 2.1 Development

```bash
# Start development server
npm run dev

# Type checking (watch mode)
npm run type-check:watch

# Run tests (watch mode)
npm run test:watch

# Check bundle size
npm run build:analyze

# Format code
npm run format

# Lint and fix
npm run lint:fix
```

### 2.2 Git Workflow

```bash
# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/YTE-123-description

# Commit with conventional message
git add .
git commit -m "feat(scope): description"

# Push and create PR
git push origin feature/YTE-123-description

# Common commit types
feat:     # New feature
fix:      # Bug fix
refactor: # Code restructuring
test:     # Adding tests
docs:     # Documentation
perf:     # Performance improvement
```

---

## 3. Code Patterns

### 3.1 Component Template

```typescript
// components/ComponentName/ComponentName.tsx
import { FC, memo } from 'react';
import { Box, Typography } from '@mui/material';

interface ComponentNameProps {
  // Props
}

export const ComponentName: FC<ComponentNameProps> = memo(({
  // Destructured props
}) => {
  // Hooks first
  
  // Local state
  
  // Effects
  
  // Handlers
  
  // Render
  return (
    <Box>
      <Typography>Component</Typography>
    </Box>
  );
});

ComponentName.displayName = 'ComponentName';
```

### 3.2 Zustand Store Template

```typescript
// stores/useStoreName.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface StoreState {
  // State
  data: any[];
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchData: () => Promise<void>;
  updateData: (id: string, data: any) => void;
  reset: () => void;
}

export const useStoreName = create<StoreState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        data: [],
        loading: false,
        error: null,
        
        // Actions
        fetchData: async () => {
          set({ loading: true, error: null });
          try {
            // Fetch logic
            set({ data: [], loading: false });
          } catch (error) {
            set({ 
              error: error.message, 
              loading: false 
            });
          }
        },
        
        updateData: (id, updates) => {
          set((state) => ({
            data: state.data.map(item =>
              item.id === id ? { ...item, ...updates } : item
            )
          }));
        },
        
        reset: () => set({ 
          data: [], 
          loading: false, 
          error: null 
        })
      }),
      {
        name: 'store-name',
        partialize: (state) => ({ 
          // Only persist what's needed
          data: state.data 
        })
      }
    )
  )
);
```

### 3.3 API Service Template

```typescript
// services/serviceName.ts
import apiClient from './api';

interface RequestDTO {
  // Request shape
}

interface ResponseDTO {
  // Response shape
}

class ServiceName {
  async getAll(): Promise<ResponseDTO[]> {
    return apiClient.get('/endpoint');
  }
  
  async getById(id: string): Promise<ResponseDTO> {
    return apiClient.get(`/endpoint/${id}`);
  }
  
  async create(data: RequestDTO): Promise<ResponseDTO> {
    return apiClient.post('/endpoint', data);
  }
  
  async update(id: string, data: Partial<RequestDTO>): Promise<ResponseDTO> {
    return apiClient.patch(`/endpoint/${id}`, data);
  }
  
  async delete(id: string): Promise<void> {
    return apiClient.delete(`/endpoint/${id}`);
  }
}

export const serviceName = new ServiceName();
```

---

## 4. Material-UI Quick Reference

### 4.1 Import Pattern

```typescript
// ✅ CORRECT - Tree-shaking friendly
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';

// ❌ WRONG - Imports entire library
import { Button, TextField, Box } from '@mui/material';
```

### 4.2 Common Components

```typescript
// Layout
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';

// Typography
import Typography from '@mui/material/Typography';

// Inputs
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import Checkbox from '@mui/material/Checkbox';
import Switch from '@mui/material/Switch';

// Buttons
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';

// Display
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Modal from '@mui/material/Modal';
import Dialog from '@mui/material/Dialog';

// Feedback
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import CircularProgress from '@mui/material/CircularProgress';
```

### 4.3 Theme Usage

```typescript
// Using theme in sx prop
<Box
  sx={{
    p: 2,                    // padding: theme.spacing(2)
    mt: 3,                   // marginTop: theme.spacing(3)
    bgcolor: 'primary.main', // theme.palette.primary.main
    color: 'text.secondary', // theme.palette.text.secondary
    borderRadius: 1,         // theme.shape.borderRadius
  }}
/>

// Using theme in styled component
import { styled } from '@mui/material/styles';

const StyledCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  '&:hover': {
    boxShadow: theme.shadows[4]
  }
}));
```

---

## 5. Testing Quick Reference

### 5.1 Test File Structure

```typescript
// ComponentName.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { ComponentName } from './ComponentName';

describe('ComponentName', () => {
  describe('Rendering', () => {
    it('should render correctly', () => {
      // Test rendering
    });
  });
  
  describe('Interactions', () => {
    it('should handle user interaction', async () => {
      const user = userEvent.setup();
      // Test interactions
    });
  });
  
  describe('Accessibility', () => {
    it('should be accessible', async () => {
      // Test accessibility
    });
  });
});
```

### 5.2 Common Testing Patterns

```typescript
// Render with wrapper
render(
  <TestWrapper>
    <Component />
  </TestWrapper>
);

// Query elements
screen.getByRole('button', { name: /submit/i });
screen.getByLabelText(/email/i);
screen.getByText(/loading/i);
screen.queryByText(/error/i); // Returns null if not found

// User interactions
const user = userEvent.setup();
await user.click(button);
await user.type(input, 'text');
await user.selectOptions(select, 'option');

// Mock functions
const mockFn = vi.fn();
mockFn.mockResolvedValue(data);
mockFn.mockRejectedValue(error);

// Wait for async
await waitFor(() => {
  expect(screen.getByText('Done')).toBeInTheDocument();
});
```

---

## 6. Performance Checklist

### 6.1 Component Optimization

```typescript
// Memoization
const MemoizedComponent = memo(Component);

// Callback memoization
const handleClick = useCallback(() => {
  // Handler logic
}, [dependency]);

// Value memoization
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// Lazy loading
const LazyComponent = lazy(() => import('./Component'));

// Virtualization for lists
import { FixedSizeList } from 'react-window';
```

### 6.2 Bundle Optimization

```typescript
// Dynamic imports
const Dashboard = lazy(() => import('./pages/Dashboard'));

// Tree shaking
import debounce from 'lodash/debounce'; // ✅
import { debounce } from 'lodash';      // ❌

// Code splitting in vite.config.ts
manualChunks: {
  'vendor': ['react', 'react-dom'],
  'ui': ['@mui/material'],
  'charts': ['recharts'],
}
```

---

## 7. Error Codes

### 7.1 Common Error Codes

```typescript
// Authentication (1xxx)
1001: 'Invalid credentials'
1005: 'Token expired'
1007: 'Insufficient permissions'

// Validation (2xxx)
2001: 'Invalid input'
2002: 'Required field missing'

// Business Logic (3xxx)
3001: 'Channel limit exceeded'
3002: 'Video limit exceeded'
3003: 'Cost limit exceeded'

// External Services (4xxx)
4001: 'YouTube API error'
4004: 'Payment error'

// System (5xxx)
5001: 'Internal server error'
5003: 'Service unavailable'
```

---

## 8. API Endpoints

### 8.1 REST Endpoints

```typescript
// Authentication
POST   /api/v1/auth/login
POST   /api/v1/auth/register
POST   /api/v1/auth/logout
POST   /api/v1/auth/refresh
GET    /api/v1/auth/profile

// Channels
GET    /api/v1/channels
GET    /api/v1/channels/:id
POST   /api/v1/channels
PATCH  /api/v1/channels/:id
DELETE /api/v1/channels/:id

// Videos
GET    /api/v1/videos
GET    /api/v1/videos/:id
POST   /api/v1/videos/generate
GET    /api/v1/videos/:id/status
POST   /api/v1/videos/:id/retry
GET    /api/v1/videos/queue

// Dashboard
GET    /api/v1/dashboard/overview
GET    /api/v1/dashboard/metrics
GET    /api/v1/dashboard/activity
```

### 8.2 WebSocket Events (3 Only)

```typescript
// Critical events only
'video.completed'  // Video generation complete
'video.failed'     // Video generation failed
'cost.alert'       // Cost limit warning
```

---

## 9. Team Contacts

```yaml
Frontend Team:
  Team Lead: @frontend-lead
  Dashboard Specialist: @dashboard-dev  
  UI/UX Designer: @design-team
  Channel: #frontend-team

Cross-Team:
  Backend API: @api-developer
  DevOps: @platform-ops
  QA: @qa-engineer
  
Escalation:
  1. Check docs
  2. Ask in Slack
  3. Team Lead
  4. CTO if blocking
```

---

## 10. Week Milestones

| Week | Key Deliverable | Success Metric |
|------|----------------|----------------|
| 1 | Environment Setup | Dev server running |
| 2 | Authentication | Login working |
| 3 | Channel UI | 5 channels displayed |
| 4 | Video Generation | Form complete |
| 5 | API Integration | Stores connected |
| 6 | Dashboard | Metrics displayed |
| 7 | WebSocket | 3 events working |
| 8 | Settings | Preferences saved |
| 9 | Polish | Flows complete |
| 10 | Performance | <1MB bundle |
| 11 | Testing | 70% coverage |
| 12 | Launch | 50 users live |

---

**Quick Help**:
- Bundle too large? Check Material-UI imports
- Slow renders? Add React.memo
- State not updating? Check Zustand devtools
- API failing? Check network tab
- Tests failing? Run with --no-coverage first

**Remember**: 
- Components: 40 max
- Screens: 25 max  
- Bundle: <1MB
- Load: <2 seconds
- Coverage: 70% min

---

**Document Status**: FINAL - Quick Reference  
**Keep This Handy**: Print or bookmark  
**Questions**: #frontend-team Slack