# YTEMPIRE Design System Documentation

**Version**: 1.0  
**Date**: January 2025  
**Owner**: UI/UX Designer  
**Approved By**: Frontend Team Lead  

---

## 1. Design System Overview

### 1.1 Purpose & Principles

The YTEMPIRE Design System serves as the single source of truth for all UI components, patterns, and guidelines. It ensures consistency across our platform while enabling rapid development and maintaining our brand identity.

**Core Design Principles:**
- **Clarity First**: Complex operations should feel simple
- **Data-Dense Yet Digestible**: Display maximum information without overwhelming
- **Performance-Oriented**: Every design decision considers load time impact
- **Accessibility by Default**: WCAG 2.1 AA compliance minimum
- **Desktop-First for MVP**: 1280px minimum width requirement

### 1.2 Technology Alignment

```yaml
design_system_tech_stack:
  ui_framework: Material-UI v5
  css_approach: MUI sx prop (no Tailwind for MVP)
  charts: Recharts only (no D3.js for MVP)
  icons: Material Icons
  typography: Inter font family
  grid: 8px baseline grid
```

---

## 2. Foundation Elements

### 2.1 Color Palette

```scss
// Primary Colors
$primary-main: #2196F3;      // Material Blue
$primary-light: #64B5F6;
$primary-dark: #1976D2;
$primary-contrast: #FFFFFF;

// Secondary Colors  
$secondary-main: #FF9800;    // Material Orange
$secondary-light: #FFB74D;
$secondary-dark: #F57C00;
$secondary-contrast: #000000;

// Semantic Colors
$success-main: #4CAF50;      // Green
$warning-main: #FFC107;      // Amber
$error-main: #F44336;        // Red
$info-main: #00BCD4;         // Cyan

// Neutral Palette
$grey-50: #FAFAFA;
$grey-100: #F5F5F5;
$grey-200: #EEEEEE;
$grey-300: #E0E0E0;
$grey-400: #BDBDBD;
$grey-500: #9E9E9E;
$grey-600: #757575;
$grey-700: #616161;
$grey-800: #424242;
$grey-900: #212121;

// Background Colors
$background-default: #FAFAFA;
$background-paper: #FFFFFF;
$background-dark: #121212;

// Cost-Specific Colors (Critical for MVP)
$cost-safe: #4CAF50;         // Under $0.35
$cost-warning: #FF9800;      // $0.35-$0.45
$cost-danger: #F44336;       // Over $0.45
```

### 2.2 Typography System

```scss
// Font Family
$font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

// Type Scale (Material Design)
.typography-h1 {
  font-size: 96px;
  font-weight: 300;
  line-height: 1.167;
  letter-spacing: -1.5px;
}

.typography-h2 {
  font-size: 60px;
  font-weight: 300;
  line-height: 1.2;
  letter-spacing: -0.5px;
}

.typography-h3 {
  font-size: 48px;
  font-weight: 400;
  line-height: 1.167;
  letter-spacing: 0px;
}

.typography-h4 {
  font-size: 34px;
  font-weight: 400;
  line-height: 1.235;
  letter-spacing: 0.25px;
}

.typography-h5 {
  font-size: 24px;
  font-weight: 400;
  line-height: 1.334;
  letter-spacing: 0px;
}

.typography-h6 {
  font-size: 20px;
  font-weight: 500;
  line-height: 1.6;
  letter-spacing: 0.15px;
}

.typography-body1 {
  font-size: 16px;
  font-weight: 400;
  line-height: 1.5;
  letter-spacing: 0.15px;
}

.typography-body2 {
  font-size: 14px;
  font-weight: 400;
  line-height: 1.43;
  letter-spacing: 0.15px;
}

.typography-caption {
  font-size: 12px;
  font-weight: 400;
  line-height: 1.66;
  letter-spacing: 0.4px;
}

// Specialized Typography for Data
.metric-large {
  font-size: 48px;
  font-weight: 700;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}

.metric-medium {
  font-size: 32px;
  font-weight: 600;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}

.metric-small {
  font-size: 24px;
  font-weight: 600;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
```

### 2.3 Spacing System

```scss
// 8px Grid System
$spacing-unit: 8px;

// Spacing Scale
$spacing-0: 0;
$spacing-1: 8px;
$spacing-2: 16px;
$spacing-3: 24px;
$spacing-4: 32px;
$spacing-5: 40px;
$spacing-6: 48px;
$spacing-7: 56px;
$spacing-8: 64px;
$spacing-9: 72px;
$spacing-10: 80px;

// Component Spacing
$component-padding: $spacing-2;
$section-margin: $spacing-4;
$page-margin: $spacing-5;
```

### 2.4 Elevation & Shadows

```scss
// Material Design Elevation
$elevation-0: none;
$elevation-1: 0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24);
$elevation-2: 0px 3px 6px rgba(0,0,0,0.16), 0px 3px 6px rgba(0,0,0,0.23);
$elevation-3: 0px 10px 20px rgba(0,0,0,0.19), 0px 6px 6px rgba(0,0,0,0.23);
$elevation-4: 0px 14px 28px rgba(0,0,0,0.25), 0px 10px 10px rgba(0,0,0,0.22);
$elevation-5: 0px 19px 38px rgba(0,0,0,0.30), 0px 15px 12px rgba(0,0,0,0.22);

// Component Elevations
$card-elevation: $elevation-1;
$modal-elevation: $elevation-5;
$dropdown-elevation: $elevation-2;
$tooltip-elevation: $elevation-2;
```

---

## 3. Core Components

### 3.1 Button Component

```typescript
// Button Variants
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
  small: {
    height: '32px',
    padding: '6px 16px',
    fontSize: '13px'
  },
  medium: {
    height: '40px',
    padding: '8px 22px',
    fontSize: '14px'
  },
  large: {
    height: '48px',
    padding: '11px 24px',
    fontSize: '15px'
  }
};

// Special MVP Buttons
const mvpCriticalButtons = {
  generateVideo: {
    variant: 'contained',
    color: 'primary',
    size: 'large',
    text: 'Generate Video',
    icon: 'VideoLibrary'
  },
  emergencyStop: {
    variant: 'contained',
    color: 'error',
    size: 'medium',
    text: 'Emergency Stop',
    icon: 'Stop'
  },
  viewCosts: {
    variant: 'outlined',
    color: 'warning',
    size: 'medium',
    text: 'View Costs',
    icon: 'AttachMoney'
  }
};
```

### 3.2 Card Component

```scss
// Card Specifications
.ytempire-card {
  background: $background-paper;
  border-radius: 8px;
  padding: $spacing-3;
  box-shadow: $card-elevation;
  transition: box-shadow 0.3s ease;
  
  &:hover {
    box-shadow: $elevation-2;
  }
  
  // Card Header
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: $spacing-2;
    
    .card-title {
      @extend .typography-h6;
      color: $grey-900;
    }
    
    .card-actions {
      display: flex;
      gap: $spacing-1;
    }
  }
  
  // Card Content
  .card-content {
    color: $grey-700;
  }
  
  // Card Footer
  .card-footer {
    margin-top: $spacing-2;
    padding-top: $spacing-2;
    border-top: 1px solid $grey-200;
  }
}

// Specialized Cards
.channel-card {
  @extend .ytempire-card;
  
  .channel-status {
    display: inline-flex;
    align-items: center;
    gap: $spacing-1;
    
    &.active {
      color: $success-main;
    }
    
    &.paused {
      color: $warning-main;
    }
  }
  
  .channel-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: $spacing-2;
    margin-top: $spacing-2;
  }
}

.video-card {
  @extend .ytempire-card;
  position: relative;
  
  .video-thumbnail {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 4px;
    margin-bottom: $spacing-2;
  }
  
  .video-status-badge {
    position: absolute;
    top: $spacing-1;
    right: $spacing-1;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    
    &.processing {
      background: $info-main;
      color: white;
    }
    
    &.completed {
      background: $success-main;
      color: white;
    }
    
    &.failed {
      background: $error-main;
      color: white;
    }
  }
}
```

### 3.3 Form Components

```typescript
// Form Field Specifications
interface FormFieldProps {
  label: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  required?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
}

// Text Field Component
const textFieldStyles = {
  standard: {
    height: '56px',
    padding: '16px',
    fontSize: '16px',
    borderRadius: '4px',
    border: '1px solid rgba(0, 0, 0, 0.23)',
    '&:hover': {
      borderColor: 'rgba(0, 0, 0, 0.87)'
    },
    '&:focus': {
      borderColor: '$primary-main',
      borderWidth: '2px'
    }
  },
  error: {
    borderColor: '$error-main',
    '&:hover': {
      borderColor: '$error-main'
    }
  }
};

// Select Component
const selectStyles = {
  minWidth: '200px',
  height: '56px',
  '& .MuiSelect-select': {
    padding: '16px',
    display: 'flex',
    alignItems: 'center'
  }
};

// MVP Critical Forms
const mvpForms = {
  channelSetup: {
    fields: [
      { name: 'channelName', type: 'text', required: true },
      { name: 'niche', type: 'select', required: true },
      { name: 'dailyVideoLimit', type: 'number', min: 1, max: 3 }
    ]
  },
  videoGeneration: {
    fields: [
      { name: 'topic', type: 'text', required: false },
      { name: 'style', type: 'select', required: true },
      { name: 'length', type: 'select', required: true },
      { name: 'priority', type: 'slider', min: 1, max: 10 }
    ]
  }
};
```

### 3.4 Data Display Components

```scss
// Table Component
.data-table {
  width: 100%;
  background: $background-paper;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: $card-elevation;
  
  .table-header {
    background: $grey-50;
    border-bottom: 2px solid $grey-200;
    
    th {
      padding: $spacing-2;
      text-align: left;
      font-weight: 600;
      color: $grey-700;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }
  
  .table-body {
    tr {
      border-bottom: 1px solid $grey-100;
      
      &:hover {
        background: $grey-50;
      }
      
      &:last-child {
        border-bottom: none;
      }
    }
    
    td {
      padding: $spacing-2;
      color: $grey-800;
      font-size: 14px;
    }
  }
}

// Metric Display
.metric-display {
  display: flex;
  flex-direction: column;
  
  .metric-value {
    @extend .metric-large;
    color: $grey-900;
    margin-bottom: $spacing-1;
  }
  
  .metric-label {
    @extend .typography-caption;
    color: $grey-600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .metric-change {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: $spacing-1;
    font-size: 14px;
    font-weight: 500;
    
    &.positive {
      color: $success-main;
    }
    
    &.negative {
      color: $error-main;
    }
  }
}
```

---

## 4. Chart Components (Recharts Only)

### 4.1 Chart Color Schemes

```typescript
// Chart Color Palettes
const chartColors = {
  primary: ['#2196F3', '#1976D2', '#0D47A1', '#64B5F6', '#90CAF9'],
  categorical: ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4'],
  sequential: ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0'],
  diverging: ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50'],
  cost: ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
};

// Chart Specifications
const chartSpecs = {
  lineChart: {
    height: 300,
    margin: { top: 5, right: 20, bottom: 5, left: 0 },
    strokeWidth: 2,
    dot: false,
    animationDuration: 1500
  },
  barChart: {
    height: 300,
    margin: { top: 20, right: 20, bottom: 5, left: 0 },
    barGap: 4,
    barCategoryGap: '20%'
  },
  pieChart: {
    height: 300,
    innerRadius: 0,
    outerRadius: 100,
    paddingAngle: 2
  }
};
```

### 4.2 MVP Dashboard Charts

```typescript
// Revenue Trend Chart
const RevenueTrendChart = {
  type: 'LineChart',
  dataKey: 'revenue',
  color: chartColors.primary[0],
  yAxisFormat: '$0,0.00',
  tooltipFormat: 'Revenue: $0,0.00'
};

// Cost Breakdown Chart
const CostBreakdownChart = {
  type: 'PieChart',
  dataKeys: ['ai_generation', 'voice_synthesis', 'storage', 'api_calls'],
  colors: chartColors.categorical,
  labelFormat: '$0,0.00'
};

// Channel Performance Chart
const ChannelPerformanceChart = {
  type: 'BarChart',
  dataKey: 'videoCount',
  color: chartColors.primary[1],
  yAxisLabel: 'Videos Generated'
};

// Video Generation Timeline
const VideoGenerationChart = {
  type: 'AreaChart',
  dataKey: 'count',
  color: chartColors.primary[0],
  fillOpacity: 0.3,
  strokeWidth: 2
};
```

---

## 5. Layout System

### 5.1 Desktop-Only Grid (MVP)

```scss
// MVP: Fixed 1280px minimum width
.app-container {
  min-width: 1280px;
  max-width: 1920px;
  margin: 0 auto;
  padding: 0 $spacing-3;
}

// 12-Column Grid System
.grid-container {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: $spacing-3;
  
  // Common Grid Patterns
  .col-12 { grid-column: span 12; }
  .col-8 { grid-column: span 8; }
  .col-6 { grid-column: span 6; }
  .col-4 { grid-column: span 4; }
  .col-3 { grid-column: span 3; }
  .col-2 { grid-column: span 2; }
}

// Dashboard Layout
.dashboard-layout {
  display: grid;
  grid-template-columns: 240px 1fr;
  grid-template-rows: 64px 1fr;
  min-height: 100vh;
  
  .sidebar {
    grid-row: 2;
    background: $grey-50;
    border-right: 1px solid $grey-200;
    padding: $spacing-3;
  }
  
  .header {
    grid-column: 1 / -1;
    background: $background-paper;
    border-bottom: 1px solid $grey-200;
    padding: 0 $spacing-3;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  
  .main-content {
    grid-row: 2;
    padding: $spacing-4;
    background: $background-default;
    overflow-y: auto;
  }
}
```

### 5.2 Component Layout Patterns

```scss
// Card Grid Layout
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: $spacing-3;
}

// Metric Cards Layout
.metric-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: $spacing-3;
  margin-bottom: $spacing-4;
}

// Two-Column Layout
.two-column-layout {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: $spacing-4;
  
  @media (max-width: 1280px) {
    // Show minimum resolution warning
    display: none;
  }
}
```

---

## 6. Animation & Transitions

### 6.1 Motion Principles

```scss
// Timing Functions
$ease-standard: cubic-bezier(0.4, 0.0, 0.2, 1);
$ease-decelerate: cubic-bezier(0.0, 0.0, 0.2, 1);
$ease-accelerate: cubic-bezier(0.4, 0.0, 1, 1);

// Duration Scale
$duration-shortest: 150ms;
$duration-short: 200ms;
$duration-standard: 300ms;
$duration-long: 500ms;

// Standard Transitions
.transition-all {
  transition: all $duration-standard $ease-standard;
}

.transition-opacity {
  transition: opacity $duration-short $ease-standard;
}

.transition-transform {
  transition: transform $duration-standard $ease-standard;
}

// Component Transitions
.button-transition {
  transition: background-color $duration-shortest $ease-standard,
              box-shadow $duration-shortest $ease-standard,
              transform $duration-shortest $ease-standard;
  
  &:hover {
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0);
  }
}

// Loading States
.skeleton-loading {
  background: linear-gradient(
    90deg,
    $grey-200 25%,
    $grey-100 50%,
    $grey-200 75%
  );
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

---

## 7. Iconography

### 7.1 Icon System

```typescript
// Material Icons Usage
const iconCategories = {
  navigation: [
    'Menu', 'Close', 'ArrowBack', 'ArrowForward', 
    'ExpandMore', 'ExpandLess', 'ChevronRight'
  ],
  
  actions: [
    'Add', 'Delete', 'Edit', 'Save', 'Cancel',
    'PlayArrow', 'Pause', 'Stop', 'Refresh'
  ],
  
  status: [
    'CheckCircle', 'Error', 'Warning', 'Info',
    'Schedule', 'HourglassEmpty', 'Done'
  ],
  
  ytempire_specific: [
    'VideoLibrary', 'AttachMoney', 'TrendingUp',
    'Analytics', 'YouTube', 'AutoAwesome'
  ]
};

// Icon Sizes
const iconSizes = {
  small: '20px',
  medium: '24px',
  large: '32px',
  xlarge: '48px'
};

// Icon Colors
const iconColors = {
  default: '$grey-700',
  primary: '$primary-main',
  secondary: '$secondary-main',
  success: '$success-main',
  error: '$error-main',
  warning: '$warning-main'
};
```

---

## 8. Accessibility Guidelines

### 8.1 WCAG 2.1 AA Compliance

```scss
// Color Contrast Requirements
.accessibility-requirements {
  // Normal text: 4.5:1 minimum
  // Large text (24px+): 3:1 minimum
  // UI components: 3:1 minimum
}

// Focus Indicators
*:focus {
  outline: 2px solid $primary-main;
  outline-offset: 2px;
}

// Skip Links
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: $primary-main;
  color: white;
  padding: $spacing-1 $spacing-2;
  text-decoration: none;
  border-radius: 0 0 4px 0;
  
  &:focus {
    top: 0;
  }
}

// Screen Reader Only
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
```

### 8.2 Keyboard Navigation

```typescript
// Keyboard Shortcuts
const keyboardShortcuts = {
  global: {
    'Cmd/Ctrl + K': 'Open command palette',
    'Cmd/Ctrl + /': 'Toggle help',
    'Escape': 'Close modal/dropdown'
  },
  
  navigation: {
    'Tab': 'Next focusable element',
    'Shift + Tab': 'Previous focusable element',
    'Enter': 'Activate element',
    'Space': 'Toggle selection'
  },
  
  dashboard: {
    'G then D': 'Go to Dashboard',
    'G then C': 'Go to Channels',
    'G then V': 'Go to Videos',
    'N': 'Create new video'
  }
};
```

---

## 9. Component States

### 9.1 Interactive States

```scss
// Button States
.button {
  // Default
  background: $primary-main;
  color: white;
  
  // Hover
  &:hover:not(:disabled) {
    background: $primary-dark;
    box-shadow: $elevation-1;
  }
  
  // Active
  &:active:not(:disabled) {
    background: darken($primary-dark, 5%);
  }
  
  // Focus
  &:focus-visible {
    outline: 2px solid $primary-main;
    outline-offset: 2px;
  }
  
  // Disabled
  &:disabled {
    background: $grey-300;
    color: $grey-500;
    cursor: not-allowed;
  }
  
  // Loading
  &.loading {
    position: relative;
    color: transparent;
    
    &::after {
      content: '';
      position: absolute;
      width: 16px;
      height: 16px;
      top: 50%;
      left: 50%;
      margin-left: -8px;
      margin-top: -8px;
      border: 2px solid white;
      border-radius: 50%;
      border-top-color: transparent;
      animation: spinner 0.8s linear infinite;
    }
  }
}

@keyframes spinner {
  to { transform: rotate(360deg); }
}
```

---

## 10. Design Tokens

### 10.1 Token Structure

```json
{
  "ytempire": {
    "color": {
      "primary": {
        "value": "#2196F3",
        "type": "color"
      },
      "cost": {
        "safe": {
          "value": "#4CAF50",
          "type": "color"
        },
        "warning": {
          "value": "#FF9800",
          "type": "color"
        },
        "danger": {
          "value": "#F44336",
          "type": "color"
        }
      }
    },
    "spacing": {
      "xs": {
        "value": "8px",
        "type": "spacing"
      },
      "sm": {
        "value": "16px",
        "type": "spacing"
      },
      "md": {
        "value": "24px",
        "type": "spacing"
      },
      "lg": {
        "value": "32px",
        "type": "spacing"
      },
      "xl": {
        "value": "48px",
        "type": "spacing"
      }
    },
    "borderRadius": {
      "sm": {
        "value": "4px",
        "type": "borderRadius"
      },
      "md": {
        "value": "8px",
        "type": "borderRadius"
      },
      "lg": {
        "value": "16px",
        "type": "borderRadius"
      }
    }
  }
}
```

---

## Implementation Notes

### MVP Component Priority (35-40 Components Max)

1. **Critical Components (Week 1-2)**
   - Button, TextField, Select
   - Card, MetricDisplay
   - Table, Loading states
   - Header, Sidebar, Navigation

2. **Dashboard Components (Week 3-4)**
   - ChannelCard, VideoCard
   - CostDisplay, AlertBanner
   - Charts (4-5 Recharts only)

3. **Secondary Components (Week 5-6)**
   - Modal, Tooltip, Snackbar
   - Tabs, Pagination
   - EmptyState, ErrorBoundary

### Design System Maintenance

- Review and update monthly
- Document all component changes
- Maintain Figma-Code parity
- Version control design tokens
- Regular accessibility audits

---

**Document Status**: APPROVED FOR MVP IMPLEMENTATION  
**Next Review**: End of Week 2  
**Owner**: UI/UX Designer  
**Distribution**: Frontend Team, Product Owner