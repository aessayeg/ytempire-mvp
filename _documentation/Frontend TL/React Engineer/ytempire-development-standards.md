# YTEMPIRE React Engineer - Development Standards
**Document Version**: 2.0  
**Last Updated**: January 2025  
**Document Type**: Coding Standards & Best Practices

---

## 1. Component Development Standards

### 1.1 Component Structure Pattern

```typescript
// components/features/channels/ChannelCard/ChannelCard.tsx

import { FC, memo, useCallback, useMemo } from 'react';
import { Card, CardContent, Typography, IconButton } from '@mui/material';
import { PlayArrow, Pause } from '@mui/icons-material';
import { useChannelStore } from '@/stores/useChannelStore';
import { formatNumber, formatCurrency } from '@/utils/formatters';
import type { Channel } from '@/types/models.types';

// Types should be in separate file for complex components
interface ChannelCardProps {
  channel: Channel;
  variant?: 'compact' | 'detailed';
  showMetrics?: boolean;
  onSelect?: (channelId: string) => void;
  className?: string;
}

/**
 * ChannelCard displays channel information and controls
 * 
 * @component
 * @example
 * <ChannelCard 
 *   channel={channelData}
 *   variant="detailed"
 *   showMetrics
 *   onSelect={handleChannelSelect}
 * />
 */
export const ChannelCard: FC<ChannelCardProps> = memo(({
  channel,
  variant = 'compact',
  showMetrics = false,
  onSelect,
  className
}) => {
  // Store hooks
  const { toggleAutomation, updateChannel } = useChannelStore();
  
  // Local state (if needed)
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Memoized values
  const formattedMetrics = useMemo(() => {
    if (!showMetrics) return null;
    
    return {
      videos: formatNumber(channel.totalVideos),
      revenue: formatCurrency(channel.estimatedRevenue),
      cost: formatCurrency(channel.totalCost)
    };
  }, [showMetrics, channel.totalVideos, channel.estimatedRevenue, channel.totalCost]);
  
  // Callbacks
  const handleToggleAutomation = useCallback(async () => {
    setIsProcessing(true);
    try {
      await toggleAutomation(channel.id);
    } catch (error) {
      console.error('Failed to toggle automation:', error);
      // Handle error (show toast, etc.)
    } finally {
      setIsProcessing(false);
    }
  }, [channel.id, toggleAutomation]);
  
  const handleCardClick = useCallback(() => {
    if (!isProcessing && onSelect) {
      onSelect(channel.id);
    }
  }, [channel.id, isProcessing, onSelect]);
  
  // Render helpers (for complex rendering logic)
  const renderMetrics = () => {
    if (!formattedMetrics) return null;
    
    return (
      <div className="metrics-container">
        <Typography variant="body2">
          Videos: {formattedMetrics.videos}
        </Typography>
        <Typography variant="body2">
          Revenue: {formattedMetrics.revenue}
        </Typography>
        <Typography variant="body2">
          Cost: {formattedMetrics.cost}
        </Typography>
      </div>
    );
  };
  
  // Main render
  return (
    <Card 
      className={className}
      onClick={handleCardClick}
      sx={{ 
        cursor: onSelect ? 'pointer' : 'default',
        opacity: isProcessing ? 0.7 : 1,
        transition: 'opacity 0.2s'
      }}
    >
      <CardContent>
        <Typography variant="h6" component="h3">
          {channel.name}
        </Typography>
        
        <Typography variant="body2" color="text.secondary">
          {channel.niche} • {channel.status}
        </Typography>
        
        {renderMetrics()}
        
        {variant === 'detailed' && (
          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              handleToggleAutomation();
            }}
            disabled={isProcessing}
            size="small"
          >
            {channel.automationEnabled ? <Pause /> : <PlayArrow />}
          </IconButton>
        )}
      </CardContent>
    </Card>
  );
});

// Display name for debugging
ChannelCard.displayName = 'ChannelCard';

// Default export for lazy loading
export default ChannelCard;
```

### 1.2 Component Checklist

Before creating a new component, verify:

```markdown
## New Component Checklist

### Planning
- [ ] Component count under 40 limit (current: X/40)
- [ ] Design approved by UI/UX Designer
- [ ] Similar component doesn't already exist
- [ ] Clear single responsibility defined

### Implementation
- [ ] TypeScript interfaces defined
- [ ] Props validated with TypeScript
- [ ] Memoization applied where beneficial
- [ ] Loading and error states handled
- [ ] Accessibility attributes included
- [ ] Material-UI theme utilized

### Testing
- [ ] Unit tests written (minimum 70% coverage)
- [ ] Accessibility tests pass
- [ ] Cross-browser tested
- [ ] Performance benchmarked (<16ms render)

### Documentation
- [ ] JSDoc comments added
- [ ] Props documented
- [ ] Usage example provided
- [ ] Storybook story created (post-MVP)
```

---

## 2. TypeScript Standards

### 2.1 Type Definition Patterns

```typescript
// types/models.types.ts

// Use interfaces for objects that might be extended
export interface Channel {
  readonly id: string;
  name: string;
  niche: string;
  status: ChannelStatus;
  automationEnabled: boolean;
  dailyVideoLimit: number;
  createdAt: string; // ISO 8601
  updatedAt: string; // ISO 8601
  metrics?: ChannelMetrics;
}

// Use type aliases for unions, primitives, and utilities
export type ChannelStatus = 'active' | 'paused' | 'suspended';

export type ChannelUpdate = Partial<Omit<Channel, 'id' | 'createdAt'>>;

// Use enums sparingly, prefer const objects
export const VideoLength = {
  SHORT: 'short',
  MEDIUM: 'medium',
  LONG: 'long'
} as const;

export type VideoLengthType = typeof VideoLength[keyof typeof VideoLength];

// Generic types for reusable patterns
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata: ResponseMetadata;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}

// Utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Nullable<T> = T | null;

export type AsyncState<T> = {
  data: T | null;
  loading: boolean;
  error: Error | null;
};
```

### 2.2 TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    
    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    
    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitReturns": true,
    "noImplicitThis": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitOverride": true,
    
    /* Paths */
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@pages/*": ["src/pages/*"],
      "@stores/*": ["src/stores/*"],
      "@services/*": ["src/services/*"],
      "@hooks/*": ["src/hooks/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## 3. Code Style Guidelines

### 3.1 ESLint Configuration

```javascript
// .eslintrc.js
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended-type-checked',
    'plugin:react-hooks/recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'prettier'
  ],
  ignorePatterns: ['dist', '.eslintrc.js'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json'],
    tsconfigRootDir: __dirname
  },
  plugins: ['react-refresh'],
  rules: {
    // TypeScript
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-unused-vars': ['error', { 
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_'
    }],
    '@typescript-eslint/explicit-function-return-type': ['warn', {
      allowExpressions: true,
      allowTypedFunctionExpressions: true
    }],
    
    // React
    'react-refresh/only-export-components': ['warn', { 
      allowConstantExport: true 
    }],
    'react/prop-types': 'off',
    'react/display-name': 'off',
    
    // General
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'prefer-const': 'error',
    'no-debugger': 'error',
    
    // Hooks
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn'
  },
  settings: {
    react: {
      version: 'detect'
    }
  }
};
```

### 3.2 Prettier Configuration

```json
// .prettierrc
{
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "semi": true,
  "singleQuote": true,
  "quoteProps": "as-needed",
  "jsxSingleQuote": false,
  "trailingComma": "es5",
  "bracketSpacing": true,
  "bracketSameLine": false,
  "arrowParens": "always",
  "endOfLine": "lf",
  "singleAttributePerLine": false
}
```

### 3.3 Naming Conventions

```typescript
// File naming
ComponentName.tsx          // React components (PascalCase)
useHookName.ts            // Custom hooks (camelCase with 'use' prefix)
serviceName.ts            // Services (camelCase)
fileName.utils.ts         // Utility files (camelCase with descriptor)
constants.ts              // Constants file (lowercase)

// Variable naming
const MAX_RETRIES = 3;              // Constants (UPPER_SNAKE_CASE)
const channelName = 'Tech Reviews'; // Variables (camelCase)
let isLoading = false;              // Booleans (is/has/should prefix)

// Function naming
function handleClick() {}           // Event handlers (handle prefix)
function validateEmail() {}         // Validators (validate prefix)
function formatCurrency() {}        // Formatters (format prefix)
async function fetchData() {}       // Async functions (action verb)

// Component props interfaces
interface ButtonProps {}            // Component props (ComponentName + Props)
interface ModalState {}            // Component state (ComponentName + State)

// API types
interface CreateChannelRequest {}   // API requests (Action + Entity + Request)
interface ChannelResponse {}        // API responses (Entity + Response)
```

---

## 4. Material-UI Integration

### 4.1 Theme Configuration

```typescript
// styles/theme.ts
import { createTheme, ThemeOptions } from '@mui/material/styles';

const themeOptions: ThemeOptions = {
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5070',
      dark: '#b8003f',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    info: {
      main: '#2196f3',
    },
    success: {
      main: '#4caf50',
    },
    background: {
      default: '#fafafa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
    button: {
      textTransform: 'none', // Disable uppercase transformation
    },
  },
  shape: {
    borderRadius: 8,
  },
  spacing: 8, // Base spacing unit (8px)
  
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          padding: '8px 16px',
        },
      },
      defaultProps: {
        disableElevation: true,
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
        size: 'small',
      },
    },
  },
};

export const theme = createTheme(themeOptions);
```

### 4.2 Material-UI Best Practices

```typescript
// ✅ CORRECT - Tree-shaking friendly imports
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import { styled } from '@mui/material/styles';

// ❌ WRONG - Imports entire library
import { Button, TextField } from '@mui/material';

// ✅ CORRECT - Use sx prop for one-off styles
<Button
  sx={{
    mt: 2,
    mb: 1,
    backgroundColor: 'primary.main',
    '&:hover': {
      backgroundColor: 'primary.dark',
    },
  }}
>
  Click Me
</Button>

// ✅ CORRECT - Use styled for reusable styled components
const StyledCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(2),
  transition: 'transform 0.2s',
  '&:hover': {
    transform: 'translateY(-2px)',
  },
}));
```

---

## 5. Testing Standards

### 5.1 Unit Testing Pattern

```typescript
// ChannelCard.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ChannelCard } from './ChannelCard';
import { useChannelStore } from '@/stores/useChannelStore';
import type { Channel } from '@/types/models.types';

// Mock the store
vi.mock('@/stores/useChannelStore');

describe('ChannelCard', () => {
  const mockChannel: Channel = {
    id: 'channel-1',
    name: 'Tech Reviews',
    niche: 'Technology',
    status: 'active',
    automationEnabled: true,
    dailyVideoLimit: 3,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  };
  
  const mockToggleAutomation = vi.fn();
  const mockOnSelect = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    (useChannelStore as any).mockReturnValue({
      toggleAutomation: mockToggleAutomation,
    });
  });
  
  describe('Rendering', () => {
    it('should render channel name and niche', () => {
      render(<ChannelCard channel={mockChannel} />);
      
      expect(screen.getByText('Tech Reviews')).toBeInTheDocument();
      expect(screen.getByText(/Technology/)).toBeInTheDocument();
    });
    
    it('should render metrics when showMetrics is true', () => {
      render(
        <ChannelCard 
          channel={{ ...mockChannel, totalVideos: 100 }} 
          showMetrics 
        />
      );
      
      expect(screen.getByText(/Videos: 100/)).toBeInTheDocument();
    });
    
    it('should not render metrics when showMetrics is false', () => {
      render(<ChannelCard channel={mockChannel} showMetrics={false} />);
      
      expect(screen.queryByText(/Videos:/)).not.toBeInTheDocument();
    });
  });
  
  describe('Interactions', () => {
    it('should call onSelect when card is clicked', async () => {
      const user = userEvent.setup();
      
      render(
        <ChannelCard 
          channel={mockChannel} 
          onSelect={mockOnSelect}
        />
      );
      
      const card = screen.getByRole('article'); // Card component role
      await user.click(card);
      
      expect(mockOnSelect).toHaveBeenCalledWith('channel-1');
    });
    
    it('should toggle automation when button is clicked', async () => {
      const user = userEvent.setup();
      mockToggleAutomation.mockResolvedValue(undefined);
      
      render(
        <ChannelCard 
          channel={mockChannel}
          variant="detailed"
        />
      );
      
      const toggleButton = screen.getByRole('button');
      await user.click(toggleButton);
      
      await waitFor(() => {
        expect(mockToggleAutomation).toHaveBeenCalledWith('channel-1');
      });
    });
    
    it('should disable interactions while processing', async () => {
      const user = userEvent.setup();
      mockToggleAutomation.mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 100))
      );
      
      render(
        <ChannelCard 
          channel={mockChannel}
          variant="detailed"
          onSelect={mockOnSelect}
        />
      );
      
      const toggleButton = screen.getByRole('button');
      await user.click(toggleButton);
      
      // Button should be disabled while processing
      expect(toggleButton).toBeDisabled();
      
      // Card should not trigger onSelect while processing
      const card = screen.getByRole('article');
      await user.click(card);
      expect(mockOnSelect).not.toHaveBeenCalled();
      
      // Wait for processing to complete
      await waitFor(() => {
        expect(toggleButton).not.toBeDisabled();
      });
    });
  });
  
  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      render(
        <ChannelCard 
          channel={mockChannel}
          onSelect={mockOnSelect}
        />
      );
      
      const card = screen.getByRole('article');
      expect(card).toHaveAttribute('tabIndex', '0');
      expect(card).toHaveStyle({ cursor: 'pointer' });
    });
    
    it('should be keyboard navigable', async () => {
      const user = userEvent.setup();
      
      render(
        <ChannelCard 
          channel={mockChannel}
          onSelect={mockOnSelect}
        />
      );
      
      const card = screen.getByRole('article');
      card.focus();
      
      await user.keyboard('{Enter}');
      expect(mockOnSelect).toHaveBeenCalledWith('channel-1');
    });
  });
});
```

### 5.2 Test Coverage Requirements

```yaml
Coverage Targets:
  Global Minimum: 70%
  
  By Category:
    Critical Paths: 90%
      - Authentication flow
      - Channel management
      - Video generation
      - Payment processing
      
    Components: 70%
      - Props rendering
      - User interactions
      - Error states
      - Loading states
      
    Store Logic: 80%
      - State updates
      - Async actions
      - Error handling
      
    Utilities: 95%
      - Pure functions
      - Formatters
      - Validators
      
    Services: 75%
      - API calls
      - Error handling
      - Response transformation
```

---

## 6. Git Workflow

### 6.1 Branch Strategy

```bash
# Branch naming convention
feature/YTE-{ticket}-{description}   # New features
fix/YTE-{ticket}-{description}       # Bug fixes
refactor/YTE-{ticket}-{description}  # Code refactoring
test/YTE-{ticket}-{description}      # Test additions
docs/YTE-{ticket}-{description}      # Documentation

# Examples
feature/YTE-123-channel-creation-modal
fix/YTE-456-dashboard-loading-error
refactor/YTE-789-optimize-store-updates
```

### 6.2 Commit Standards

```bash
# Commit message format
<type>(<scope>): <subject>

[optional body]

[optional footer]

# Types
feat     # New feature
fix      # Bug fix
refactor # Code refactoring
test     # Test additions/changes
docs     # Documentation changes
style    # Code style changes (formatting, etc.)
perf     # Performance improvements
chore    # Build process or auxiliary tool changes

# Examples
feat(channels): add channel creation modal
fix(auth): resolve token refresh race condition
refactor(dashboard): optimize metric calculations
test(api): add channel service integration tests
docs(readme): update setup instructions
```

### 6.3 Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No console.log statements
- [ ] Bundle size impact checked

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] Cross-browser tested

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Related Issues
Closes YTE-XXX
```

---

**Document Status**: FINAL - Consolidated Version  
**Next Review**: Code Review Standards Update Week 4  
**Owner**: Frontend Team Lead  
**Questions**: Contact via #frontend-team Slack