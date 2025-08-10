# YTEMPIRE Design System

## 3.1 Design Principles

### Core Design Philosophy

#### Clarity First
Complex operations should feel simple. Every interface element must have a clear purpose and obvious function. Ambiguity is the enemy of automation.

#### Data-Dense Yet Digestible
Display maximum information without overwhelming. Use progressive disclosure, clear visual hierarchy, and smart defaults to manage complexity.

#### Performance-Oriented
Every design decision considers load time impact. Prefer system fonts, optimize images, and minimize decorative elements that slow performance.

#### Accessibility by Default
WCAG 2.1 AA compliance minimum. Every feature must be keyboard navigable, screen reader compatible, and usable by people with disabilities.

#### Desktop-First for MVP
Optimized for 1280px minimum width. No mobile responsiveness in MVP. Focus on power-user workflows on desktop displays.

### Design Values

```yaml
values:
  simplicity:
    description: "Reduce cognitive load"
    implementation: "Hide complexity behind smart defaults"
    
  consistency:
    description: "Predictable interactions"
    implementation: "Reuse patterns throughout platform"
    
  efficiency:
    description: "Minimize clicks and time"
    implementation: "Bulk actions, keyboard shortcuts"
    
  feedback:
    description: "Always inform the user"
    implementation: "Loading states, progress indicators"
    
  forgiveness:
    description: "Prevent and recover from errors"
    implementation: "Confirmations, undo actions"
```

## 3.2 Component Library

### Component Architecture (35 Components Total)

#### Base Components (10)

```typescript
// 1. Button Component
interface ButtonProps {
  variant: 'contained' | 'outlined' | 'text';
  color: 'primary' | 'secondary' | 'error' | 'warning' | 'success';
  size: 'small' | 'medium' | 'large';
  fullWidth?: boolean;
  disabled?: boolean;
  loading?: boolean;
  startIcon?: React.ReactNode;
  endIcon?: React.ReactNode;
}

// Button Specifications
const buttonSpecs = {
  small: { height: '32px', padding: '6px 16px', fontSize: '13px' },
  medium: { height: '40px', padding: '8px 22px', fontSize: '14px' },
  large: { height: '48px', padding: '11px 24px', fontSize: '15px' }
};

// 2. TextField Component
interface TextFieldProps {
  variant: 'outlined' | 'filled' | 'standard';
  size: 'small' | 'medium';
  type: 'text' | 'password' | 'email' | 'number';
  error?: boolean;
  helperText?: string;
  required?: boolean;
  multiline?: boolean;
  rows?: number;
}

// 3. Select Component
interface SelectProps {
  multiple?: boolean;
  native?: boolean;
  size: 'small' | 'medium';
  variant: 'outlined' | 'filled' | 'standard';
  error?: boolean;
}

// 4. Checkbox
// 5. Radio Button
// 6. Switch
// 7. Chip
// 8. Badge
// 9. Avatar
// 10. Tooltip
```

#### Layout Components (5)

```typescript
// 11. Card Component
interface CardProps {
  elevation?: 0 | 1 | 2 | 3 | 4 | 5;
  variant?: 'elevation' | 'outlined';
  square?: boolean;
}

// 12. Container
// 13. Grid
// 14. Stack
// 15. Divider
```

#### Chart Components (5) - Recharts Only

```typescript
// 16. LineChart - Revenue trends
// 17. BarChart - Channel comparison
// 18. PieChart - Cost breakdown
// 19. AreaChart - Video performance
// 20. GaugeChart - Cost per video indicator
```

#### Business Components (15)

```typescript
// 21. ChannelCard
interface ChannelCardProps {
  channel: Channel;
  onEdit: () => void;
  onPause: () => void;
  onDelete: () => void;
  onGenerateVideo: () => void;
}

// 22. VideoRow
// 23. MetricCard
// 24. CostIndicator (<$3 threshold)
// 25. StatusBadge
// 26. ProgressRing
// 27. EmptyState
// 28. ErrorBoundary
// 29. LoadingSkeleton
// 30. NotificationToast
// 31. ConfirmDialog
// 32. Sidebar
// 33. Header
// 34. PageHeader
// 35. DataTable
```

### Component Specifications

```scss
// Component Spacing System (8px grid)
$spacing-unit: 8px;
$spacing: (
  0: 0,
  1: 8px,
  2: 16px,
  3: 24px,
  4: 32px,
  5: 40px,
  6: 48px,
  7: 56px,
  8: 64px
);

// Component Border Radius
$radius: (
  small: 4px,
  medium: 8px,
  large: 12px,
  round: 999px
);

// Component Shadows (Material Design)
$shadows: (
  0: none,
  1: 0px 1px 3px rgba(0,0,0,0.12),
  2: 0px 3px 6px rgba(0,0,0,0.16),
  3: 0px 10px 20px rgba(0,0,0,0.19),
  4: 0px 14px 28px rgba(0,0,0,0.25),
  5: 0px 19px 38px rgba(0,0,0,0.30)
);
```

## 3.3 Visual Language

### Color System

```scss
// Primary Colors (Material Blue)
$primary: (
  main: #2196F3,
  light: #64B5F6,
  dark: #1976D2,
  contrast: #FFFFFF
);

// Secondary Colors (Material Orange)
$secondary: (
  main: #FF9800,
  light: #FFB74D,
  dark: #F57C00,
  contrast: #000000
);

// Semantic Colors
$semantic: (
  success: #4CAF50,  // Green
  warning: #FFC107,  // Amber
  error: #F44336,    // Red
  info: #00BCD4      // Cyan
);

// Cost-Specific Colors (Critical for MVP)
$cost-colors: (
  safe: #4CAF50,     // Under $2.50
  warning: #FF9800,  // $2.50-$2.90
  danger: #F44336    // Over $2.90 (approaching $3 limit)
);

// Neutral Palette
$grey: (
  50: #FAFAFA,
  100: #F5F5F5,
  200: #EEEEEE,
  300: #E0E0E0,
  400: #BDBDBD,
  500: #9E9E9E,
  600: #757575,
  700: #616161,
  800: #424242,
  900: #212121
);

// Background Colors
$backgrounds: (
  default: #FAFAFA,
  paper: #FFFFFF,
  dark: #121212
);
```

### Typography System

```scss
// Font Stack
$font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
$font-mono: 'Roboto Mono', 'Courier New', monospace;

// Type Scale (Material Design)
$typography: (
  h1: (size: 96px, weight: 300, line-height: 1.167, letter-spacing: -1.5px),
  h2: (size: 60px, weight: 300, line-height: 1.2, letter-spacing: -0.5px),
  h3: (size: 48px, weight: 400, line-height: 1.167, letter-spacing: 0px),
  h4: (size: 34px, weight: 400, line-height: 1.235, letter-spacing: 0.25px),
  h5: (size: 24px, weight: 400, line-height: 1.334, letter-spacing: 0px),
  h6: (size: 20px, weight: 500, line-height: 1.6, letter-spacing: 0.15px),
  body1: (size: 16px, weight: 400, line-height: 1.5, letter-spacing: 0.15px),
  body2: (size: 14px, weight: 400, line-height: 1.43, letter-spacing: 0.15px),
  caption: (size: 12px, weight: 400, line-height: 1.66, letter-spacing: 0.4px),
  button: (size: 14px, weight: 500, line-height: 1.75, letter-spacing: 0.4px),
  overline: (size: 10px, weight: 400, line-height: 2.66, letter-spacing: 1px)
);

// Specialized Typography for Metrics
$metric-typography: (
  large: (size: 48px, weight: 700, line-height: 1),
  medium: (size: 32px, weight: 600, line-height: 1),
  small: (size: 24px, weight: 600, line-height: 1)
);
```

### Iconography

```yaml
icon_system:
  library: "Material Icons Outlined"
  sizes: [16px, 20px, 24px, 32px]
  style: "Outlined with 2px stroke"
  
categories:
  navigation:
    - Menu
    - Close
    - ArrowBack
    - ArrowForward
    - ExpandMore
    
  actions:
    - Add
    - Delete
    - Edit
    - Save
    - PlayArrow
    - Pause
    - Stop
    
  status:
    - CheckCircle
    - Error
    - Warning
    - Info
    - Schedule
    
  ytempire_specific:
    - VideoLibrary
    - AttachMoney
    - TrendingUp
    - Analytics
    - YouTube
```

## 3.4 Interaction Patterns

### Motion Principles

```scss
// Timing Functions
$easing: (
  standard: cubic-bezier(0.4, 0.0, 0.2, 1),
  decelerate: cubic-bezier(0.0, 0.0, 0.2, 1),
  accelerate: cubic-bezier(0.4, 0.0, 1, 1),
  sharp: cubic-bezier(0.4, 0.0, 0.6, 1)
);

// Duration Scale
$duration: (
  shortest: 150ms,
  short: 200ms,
  standard: 300ms,
  long: 500ms,
  longest: 1000ms
);

// Standard Transitions
.transition-all {
  transition: all 300ms cubic-bezier(0.4, 0.0, 0.2, 1);
}

.transition-opacity {
  transition: opacity 200ms cubic-bezier(0.4, 0.0, 0.2, 1);
}

.transition-transform {
  transition: transform 300ms cubic-bezier(0.4, 0.0, 0.2, 1);
}
```

### Interaction States

```typescript
// Component State Definitions
interface InteractionStates {
  default: 'Base state';
  hover: 'Cursor over element';
  active: 'Being clicked/pressed';
  focus: 'Keyboard navigation';
  disabled: 'Not interactive';
  loading: 'Processing action';
  error: 'Action failed';
  success: 'Action completed';
}

// State Visual Feedback
const stateStyles = {
  hover: {
    elevation: 'Increase by 1',
    opacity: '0.9',
    cursor: 'pointer'
  },
  active: {
    scale: '0.98',
    elevation: 'Decrease by 1'
  },
  focus: {
    outline: '2px solid primary',
    outlineOffset: '2px'
  },
  disabled: {
    opacity: '0.5',
    cursor: 'not-allowed'
  }
};
```

### Loading Patterns

```scss
// Skeleton Loading
.skeleton-loading {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

// Spinner Loading
.spinner {
  border: 2px solid rgba(0, 0, 0, 0.1);
  border-left-color: #2196F3;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Feedback Patterns

```typescript
// Toast Notifications
const toastConfig = {
  position: 'bottom-right',
  duration: 5000,
  maxStack: 3,
  types: {
    success: { icon: 'CheckCircle', color: 'success' },
    error: { icon: 'Error', color: 'error' },
    warning: { icon: 'Warning', color: 'warning' },
    info: { icon: 'Info', color: 'info' }
  }
};

// Progress Indicators
const progressPatterns = {
  quick: {
    duration: '<1s',
    indicator: 'None needed'
  },
  medium: {
    duration: '1-3s',
    indicator: 'Circular progress'
  },
  long: {
    duration: '3-10s',
    indicator: 'Linear progress with percentage'
  },
  extended: {
    duration: '>10s',
    indicator: 'Stepped progress with status messages'
  }
};
```

## 3.5 Accessibility Standards

### WCAG 2.1 AA Compliance

#### Color Contrast Requirements

```scss
// Minimum Contrast Ratios
$contrast-requirements: (
  normal-text: 4.5,      // Regular text
  large-text: 3,         // 24px+ or 19px+ bold
  ui-components: 3,      // Interactive elements
  graphics: 3            // Important graphics
);

// Contrast Validation
@mixin ensure-contrast($foreground, $background, $ratio: 4.5) {
  // Automated testing will verify contrast ratios
}
```

#### Keyboard Navigation

```typescript
// Keyboard Shortcuts
const keyboardShortcuts = {
  global: {
    'Cmd/Ctrl + K': 'Open command palette',
    'Cmd/Ctrl + /': 'Toggle help',
    'Escape': 'Close modal/dropdown',
    '?': 'Show keyboard shortcuts'
  },
  
  navigation: {
    'Tab': 'Next focusable element',
    'Shift + Tab': 'Previous focusable element',
    'Enter': 'Activate element',
    'Space': 'Toggle selection',
    'Arrow keys': 'Navigate within component'
  },
  
  dashboard: {
    'G then D': 'Go to Dashboard',
    'G then C': 'Go to Channels',
    'G then V': 'Go to Videos',
    'G then A': 'Go to Analytics',
    'N': 'Create new video',
    'P': 'Pause/Resume selected channel'
  }
};
```

#### Screen Reader Support

```scss
// Screen Reader Only Content
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

// Skip Links
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: $primary-main;
  color: white;
  padding: 8px 16px;
  text-decoration: none;
  border-radius: 0 0 4px 0;
  z-index: 100;
  
  &:focus {
    top: 0;
  }
}
```

#### ARIA Implementation

```typescript
// ARIA Patterns
const ariaPatterns = {
  liveRegions: {
    polite: 'Status messages, non-critical updates',
    assertive: 'Errors, time-sensitive alerts',
    off: 'Decorative or redundant content'
  },
  
  landmarks: {
    main: 'Primary content area',
    navigation: 'Site navigation',
    complementary: 'Supporting content',
    contentinfo: 'Footer information'
  },
  
  states: {
    'aria-busy': 'Loading content',
    'aria-disabled': 'Non-interactive',
    'aria-expanded': 'Accordion/dropdown state',
    'aria-selected': 'Selection state',
    'aria-checked': 'Checkbox state'
  }
};
```

### Focus Management

```typescript
// Focus Trap for Modals
const trapFocus = (element: HTMLElement) => {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const firstFocusable = focusableElements[0];
  const lastFocusable = focusableElements[focusableElements.length - 1];
  
  // Trap focus within element
  element.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      if (e.shiftKey && document.activeElement === firstFocusable) {
        lastFocusable.focus();
        e.preventDefault();
      } else if (!e.shiftKey && document.activeElement === lastFocusable) {
        firstFocusable.focus();
        e.preventDefault();
      }
    }
  });
};
```

### Responsive Design (Post-MVP)

```scss
// Desktop-Only for MVP
$breakpoints: (
  minimum: 1280px,  // MVP requirement
  standard: 1920px, // Target resolution
  wide: 2560px      // 4K support
);

// Future Mobile Breakpoints (Phase 2)
$future-breakpoints: (
  mobile: 320px,
  tablet: 768px,
  desktop: 1280px,
  wide: 1920px
);
```