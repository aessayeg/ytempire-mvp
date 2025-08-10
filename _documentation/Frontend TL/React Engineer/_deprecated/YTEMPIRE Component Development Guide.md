# YTEMPIRE Component Development Guide
## Complete Component Architecture & Implementation

**Document Version**: 1.0  
**Role**: React Engineer  
**Scope**: Component Development Standards & Patterns

---

## 📋 Component Inventory (30-40 Total)

### Layout Components (5)
```typescript
// 1. AppLayout.tsx - Main application wrapper
interface AppLayoutProps {
  children: React.ReactNode;
}

// 2. Header.tsx - Top navigation bar
interface HeaderProps {
  user: User;
  onLogout: () => void;
}

// 3. Sidebar.tsx - Side navigation
interface SidebarProps {
  activeRoute: string;
  channels: Channel[];
  onChannelSelect: (id: string) => void;
}

// 4. ContentContainer.tsx - Main content wrapper
interface ContentContainerProps {
  children: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl';
}

// 5. PageHeader.tsx - Page title and actions
interface PageHeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}
```

### Common Components (10)
```typescript
// 6. Button.tsx - Extends MUI Button
interface ButtonProps extends MuiButtonProps {
  loading?: boolean;
  loadingText?: string;
}

// 7. TextField.tsx - Extends MUI TextField
interface TextFieldProps extends MuiTextFieldProps {
  errorText?: string;
}

// 8. Select.tsx - Dropdown select
interface SelectProps<T> {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
  placeholder?: string;
}

// 9. Modal.tsx - Reusable modal wrapper
interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}

// 10. Card.tsx - Content card wrapper
interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}

// 11. Table.tsx - Data table wrapper
interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  loading?: boolean;
}

// 12. LoadingSpinner.tsx - Loading indicator
interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  fullScreen?: boolean;
}

// 13. ErrorBoundary.tsx - Error handling (Class component)
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error }>;
}

// 14. Toast.tsx - Notification component
interface ToastProps {
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
  onClose: () => void;
}

// 15. EmptyState.tsx - No data display
interface EmptyStateProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
  icon?: React.ReactNode;
}
```

### Business Components (15)
```typescript
// 16. ChannelCard.tsx - Channel display card
interface ChannelCardProps {
  channel: Channel;
  onEdit: (id: string) => void;
  onToggle: (id: string) => void;
}

// 17. VideoRow.tsx - Video list item
interface VideoRowProps {
  video: Video;
  showChannel?: boolean;
  onView: (id: string) => void;
}

// 18. MetricCard.tsx - Metric display
interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: React.ReactNode;
  loading?: boolean;
}

// 19. CostBreakdown.tsx - Cost visualization
interface CostBreakdownProps {
  costs: CostData;
  period: 'daily' | 'weekly' | 'monthly';
}

// 20. VideoQueue.tsx - Video generation queue
interface VideoQueueProps {
  queue: QueueItem[];
  onCancel: (id: string) => void;
}

// 21. ChannelSelector.tsx - Channel dropdown
interface ChannelSelectorProps {
  channels: Channel[];
  selected?: string;
  onChange: (channelId: string) => void;
  allowAll?: boolean;
}

// 22. VideoGenerator.tsx - Video creation form
interface VideoGeneratorProps {
  channelId: string;
  onGenerate: (params: VideoParams) => Promise<void>;
}

// 23. RevenueDisplay.tsx - Revenue metrics
interface RevenueDisplayProps {
  revenue: RevenueData;
  period: TimePeriod;
}

// 24. AutomationToggle.tsx - Enable/disable automation
interface AutomationToggleProps {
  channelId: string;
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

// 25. ChannelSetupWizard.tsx - New channel creation
interface ChannelSetupWizardProps {
  onComplete: (channel: Channel) => void;
  onCancel: () => void;
}

// 26. VideoStatusBadge.tsx - Video status indicator
interface VideoStatusBadgeProps {
  status: VideoStatus;
  showLabel?: boolean;
}

// 27. CostAlert.tsx - Cost warning display
interface CostAlertProps {
  currentCost: number;
  threshold: number;
  onDismiss: () => void;
}

// 28. PerformanceChart.tsx - Channel performance
interface PerformanceChartProps {
  data: PerformanceData[];
  metric: 'views' | 'revenue' | 'subscribers';
}

// 29. BulkActions.tsx - Multi-select actions
interface BulkActionsProps {
  selectedCount: number;
  actions: BulkAction[];
}

// 30. QuickStats.tsx - Dashboard summary
interface QuickStatsProps {
  stats: DashboardStats;
  period: TimePeriod;
}
```

### Chart Components (5) - Recharts Only
```typescript
// 31. LineChart.tsx - Trend visualization
interface LineChartProps {
  data: TimeSeriesData[];
  dataKey: string;
  height?: number;
}

// 32. BarChart.tsx - Comparison chart
interface BarChartProps {
  data: CategoryData[];
  dataKey: string;
  categoryKey: string;
}

// 33. PieChart.tsx - Distribution chart
interface PieChartProps {
  data: DistributionData[];
  innerRadius?: number;
}

// 34. AreaChart.tsx - Stacked metrics
interface AreaChartProps {
  data: TimeSeriesData[];
  areas: AreaConfig[];
}

// 35. MetricSparkline.tsx - Mini trend chart
interface MetricSparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}
```

---

## 🏗️ Component Implementation Patterns

### Base Component Template
```typescript
import React, { memo } from 'react';
import { Box, Typography } from '@mui/material';
import { useChannelStore } from '@/stores/useChannelStore';

interface ComponentNameProps {
  // Props interface
  required: string;
  optional?: number;
  children?: React.ReactNode;
}

export const ComponentName = memo<ComponentNameProps>(({ 
  required,
  optional = 0,
  children 
}) => {
  // Hooks at the top
  const { data, loading } = useChannelStore();
  
  // Local state if needed
  const [state, setState] = useState(false);
  
  // Memoized values
  const computed = useMemo(() => {
    return expensive(data);
  }, [data]);
  
  // Callbacks
  const handleClick = useCallback(() => {
    // Handle click
  }, [dependency]);
  
  // Effects
  useEffect(() => {
    // Side effects
    return () => {
      // Cleanup
    };
  }, [dependency]);
  
  // Early returns for edge cases
  if (loading) return <LoadingSpinner />;
  if (!data) return <EmptyState />;
  
  // Main render
  return (
    <Box>
      <Typography>{required}</Typography>
      {children}
    </Box>
  );
});

ComponentName.displayName = 'ComponentName';
```

---

## 📐 Component Architecture Rules

### 1. State Management
```typescript
// ✅ CORRECT - Use Zustand for shared state
const ChannelList = () => {
  const { channels, fetchChannels } = useChannelStore();
  const [localFilter, setLocalFilter] = useState('');
  
  // Local state for UI-only concerns
  // Zustand for shared application state
};

// ❌ WRONG - Don't prop drill shared state
const BadComponent = ({ channels, updateChannels, loading }) => {
  // Should use Zustand store instead
};
```

### 2. Performance Optimization
```typescript
// ✅ CORRECT - Optimize expensive components
export const ExpensiveChart = memo(({ data }) => {
  const processedData = useMemo(() => 
    processData(data), [data]
  );
  
  return <LineChart data={processedData} />;
});

// ✅ CORRECT - Debounce user input
const SearchInput = () => {
  const [search, setSearch] = useState('');
  const debouncedSearch = useDebouncedValue(search, 500);
  
  useEffect(() => {
    if (debouncedSearch) {
      searchChannels(debouncedSearch);
    }
  }, [debouncedSearch]);
};
```

### 3. Error Handling
```typescript
// ✅ CORRECT - Handle errors gracefully
const DataComponent = () => {
  const { data, error, loading } = useQuery('key', fetchData);
  
  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay error={error} retry={refetch} />;
  if (!data) return <EmptyState />;
  
  return <DataDisplay data={data} />;
};

// ✅ CORRECT - Use Error Boundary for critical sections
<ErrorBoundary fallback={<ErrorFallback />}>
  <CriticalDashboard />
</ErrorBoundary>
```

### 4. TypeScript Best Practices
```typescript
// ✅ CORRECT - Strong typing
interface ChannelActionsProps {
  channel: Channel;
  onEdit: (channel: Channel) => void;
  onDelete: (id: string) => Promise<void>;
}

// ❌ WRONG - Avoid any types
const BadComponent = (props: any) => {
  const handleClick = (e: any) => {
    // No type safety
  };
};
```

---

## 🎨 Material-UI Integration

### Theme Customization
```typescript
// src/theme/theme.ts
import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    primary: {
      main: '#2196F3',
      light: '#64B5F6',
      dark: '#1976D2',
    },
    secondary: {
      main: '#FF9800',
    },
    success: {
      main: '#4CAF50',
    },
    error: {
      main: '#F44336',
    },
  },
  typography: {
    fontFamily: '"Inter", -apple-system, sans-serif',
    h1: {
      fontSize: '2rem',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
});
```

### Component Styling Patterns
```typescript
// ✅ CORRECT - Use sx prop for one-off styles
<Box sx={{ 
  p: 2, 
  bgcolor: 'background.paper',
  borderRadius: 1,
  boxShadow: 1 
}}>
  Content
</Box>

// ✅ CORRECT - Use styled for reusable styled components
const StyledCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(2),
  transition: 'transform 0.2s',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[4],
  },
}));

// ❌ WRONG - Don't use inline styles
<div style={{ padding: '16px' }}>Content</div>
```

---

## 📊 Chart Implementation Guide

### Recharts Setup
```typescript
// ✅ CORRECT - Recharts implementation
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip 
} from 'recharts';

export const RevenueChart = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="date" 
          tickFormatter={(date) => format(date, 'MMM d')}
        />
        <YAxis 
          tickFormatter={(value) => `$${value}`}
        />
        <Tooltip 
          formatter={(value) => [`$${value}`, 'Revenue']}
        />
        <Line 
          type="monotone" 
          dataKey="revenue" 
          stroke="#2196F3" 
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

// ❌ WRONG - Don't use D3.js for MVP
// No direct D3 manipulation
// No complex custom visualizations
```

---

## 🧪 Component Testing

### Testing Patterns
```typescript
// ChannelCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChannelCard } from './ChannelCard';
import { mockChannel } from '@/test/fixtures';

describe('ChannelCard', () => {
  const defaultProps = {
    channel: mockChannel,
    onEdit: jest.fn(),
    onToggle: jest.fn(),
  };
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  it('renders channel information', () => {
    render(<ChannelCard {...defaultProps} />);
    
    expect(screen.getByText(mockChannel.name)).toBeInTheDocument();
    expect(screen.getByText(`${mockChannel.videoCount} videos`)).toBeInTheDocument();
  });
  
  it('handles edit action', () => {
    render(<ChannelCard {...defaultProps} />);
    
    fireEvent.click(screen.getByLabelText('Edit channel'));
    expect(defaultProps.onEdit).toHaveBeenCalledWith(mockChannel.id);
  });
  
  it('handles toggle automation', async () => {
    render(<ChannelCard {...defaultProps} />);
    
    const toggle = screen.getByRole('switch');
    fireEvent.click(toggle);
    
    expect(defaultProps.onToggle).toHaveBeenCalledWith(mockChannel.id);
  });
  
  it('shows loading state during toggle', async () => {
    const slowToggle = jest.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
    render(<ChannelCard {...defaultProps} onToggle={slowToggle} />);
    
    fireEvent.click(screen.getByRole('switch'));
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
```

### Testing Checklist
- [ ] Component renders without errors
- [ ] Props are properly typed
- [ ] User interactions work correctly
- [ ] Loading states display properly
- [ ] Error states handle gracefully
- [ ] Accessibility requirements met
- [ ] Edge cases covered

---

## 🚀 Component Development Workflow

### Step-by-Step Process
1. **Review Design**
   - Check Figma for specifications
   - Understand user interactions
   - Note responsive behaviors

2. **Create Component File**
   ```bash
   src/components/business/NewComponent.tsx
   src/components/business/NewComponent.test.tsx
   ```

3. **Define TypeScript Interface**
   ```typescript
   interface NewComponentProps {
     // Define all props with types
   }
   ```

4. **Implement Component**
   - Start with static markup
   - Add interactivity
   - Connect to Zustand store
   - Add loading/error states

5. **Style Component**
   - Use Material-UI components
   - Apply theme consistently
   - Ensure desktop layout (1280px+)

6. **Write Tests**
   - Cover all user interactions
   - Test edge cases
   - Achieve 70% coverage minimum

7. **Optimize Performance**
   - Add React.memo if needed
   - Implement useMemo for expensive operations
   - Profile with React DevTools

8. **Document Usage**
   - Add JSDoc comments
   - Create Storybook story (if applicable)
   - Update component inventory

---

## 📋 Component Checklist

Before marking a component as complete:

### Functionality
- [ ] All requirements implemented
- [ ] Connected to Zustand store (if needed)
- [ ] API integration working
- [ ] Error handling in place
- [ ] Loading states implemented

### Code Quality
- [ ] TypeScript interfaces defined
- [ ] No `any` types used
- [ ] Props validated
- [ ] Component under 200 lines
- [ ] Follows naming conventions

### Testing
- [ ] Unit tests written
- [ ] 70% coverage achieved
- [ ] Edge cases tested
- [ ] Accessibility tested
- [ ] Browser testing done

### Performance
- [ ] No unnecessary re-renders
- [ ] Images optimized
- [ ] Code split if needed
- [ ] Bundle size checked

### Documentation
- [ ] Props documented
- [ ] Usage examples provided
- [ ] Complex logic commented
- [ ] README updated

---

## 🎯 Quality Standards

### Component Quality Metrics
- **Size**: < 200 lines per component
- **Complexity**: Cyclomatic complexity < 10
- **Dependencies**: < 5 imports from other components
- **Test Coverage**: ≥ 70%
- **Performance**: Renders < 16ms

### Code Review Focus Areas
1. **Correctness**: Does it meet requirements?
2. **Performance**: Any unnecessary renders?
3. **Maintainability**: Is it easy to understand?
4. **Testability**: Can it be easily tested?
5. **Reusability**: Can other components use it?

---

## 📚 Resources

### Component Examples
- Authentication Flow: `src/pages/Login`
- Data Table: `src/components/common/Table`
- Form Handling: `src/components/business/VideoGenerator`
- Chart Integration: `src/components/charts/RevenueChart`

### Useful Utilities
```typescript
// src/utils/formatters.ts
export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
};

export const formatDate = (date: Date): string => {
  return format(date, 'MMM d, yyyy');
};

export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('en-US').format(num);
};
```

---

**Remember**: Focus on building simple, reusable components that make complex operations feel effortless. Quality over quantity! 🚀