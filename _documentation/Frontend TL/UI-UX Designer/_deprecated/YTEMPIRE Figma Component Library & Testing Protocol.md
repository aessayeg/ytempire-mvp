# YTEMPIRE Figma Component Library & Testing Protocol

**Version**: 1.0  
**Date**: January 2025  
**Owner**: UI/UX Designer  
**Approved By**: Frontend Team Lead  
**Status**: MVP Implementation Ready

---

## Table of Contents

1. [Figma Library Structure](#1-figma-library-structure)
2. [Component Organization](#2-component-organization)
3. [Design Tokens Setup](#3-design-tokens-setup)
4. [Component Specifications](#4-component-specifications)
5. [Prototyping Guidelines](#5-prototyping-guidelines)
6. [Developer Handoff](#6-developer-handoff)
7. [Usability Testing Protocol](#7-usability-testing-protocol)
8. [Brand Guidelines](#8-brand-guidelines)

---

## 1. Figma Library Structure

### 1.1 File Organization

```
YTEMPIRE Design System/
├── 📁 01_Foundations
│   ├── 📄 Colors & Themes
│   ├── 📄 Typography
│   ├── 📄 Grid & Spacing
│   ├── 📄 Icons & Illustrations
│   └── 📄 Effects & Elevation
│
├── 📁 02_Components
│   ├── 📄 Base Components
│   ├── 📄 Form Elements
│   ├── 📄 Cards & Containers
│   ├── 📄 Navigation
│   ├── 📄 Data Display
│   └── 📄 Feedback & Modals
│
├── 📁 03_Patterns
│   ├── 📄 Page Layouts
│   ├── 📄 Common Patterns
│   └── 📄 Empty States
│
├── 📁 04_Screens
│   ├── 📄 Dashboard
│   ├── 📄 Channels
│   ├── 📄 Videos
│   ├── 📄 Analytics
│   └── 📄 Settings
│
└── 📁 05_Prototypes
    ├── 📄 Onboarding Flow
    ├── 📄 Video Generation Flow
    └── 📄 Daily Workflow
```

### 1.2 Naming Conventions

```yaml
component_naming:
  format: "Category/Component/Variant/State"
  examples:
    - "Button/Primary/Large/Default"
    - "Card/Channel/Active/Hover"
    - "Input/Text/Error/Focused"
    
layer_naming:
  icons: "icon-{name}"
  text: "text-{description}"
  shapes: "shape-{type}"
  groups: "group-{purpose}"
  
color_naming:
  format: "{color}-{shade}"
  semantic: "{purpose}-{variant}"
  examples:
    - "blue-500"
    - "primary-default"
    - "status-error"
```

### 1.3 Version Control

```yaml
versioning_strategy:
  major_changes: "New version file"
  minor_updates: "Branch within file"
  
  naming_format: "YTEMPIRE_v{major}.{minor}_{date}"
  example: "YTEMPIRE_v1.0_2025-01"
  
  change_log:
    location: "File description"
    format: |
      v1.1 (2025-01-15)
      - Added new video card variants
      - Updated color system
      - Fixed button padding issues
```

---

## 2. Component Organization

### 2.1 Component Structure

```
Component Structure in Figma:
┌─────────────────────────────────────┐
│ Component Set Name                  │
├─────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│ │Focused  │ │Disabled │ │Loading  ││
│ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────┘
```

### 2.2 Component Properties

```typescript
// Figma Component Properties Setup
interface ComponentProperties {
  // Button Component Example
  Button: {
    variant: "primary" | "secondary" | "danger" | "ghost";
    size: "small" | "medium" | "large";
    state: "default" | "hover" | "active" | "disabled" | "loading";
    iconLeft?: boolean;
    iconRight?: boolean;
    fullWidth?: boolean;
  };
  
  // Card Component Example
  Card: {
    type: "default" | "metric" | "channel" | "video";
    elevation: "none" | "low" | "medium" | "high";
    state: "default" | "hover" | "selected" | "disabled";
    hasAction?: boolean;
    hasFooter?: boolean;
  };
  
  // Input Component Example
  Input: {
    type: "text" | "number" | "password" | "search";
    state: "default" | "hover" | "focused" | "error" | "success" | "disabled";
    hasLabel?: boolean;
    hasHelper?: boolean;
    hasIcon?: boolean;
  };
}
```

### 2.3 Component Checklist

```yaml
component_requirements:
  before_publishing:
    ✓ All states designed (default, hover, active, focus, disabled)
    ✓ Proper constraints applied
    ✓ Auto-layout configured
    ✓ Consistent naming
    ✓ Description added
    ✓ Preview image set
    
  properties_setup:
    ✓ Variants properly configured
    ✓ Boolean properties for optional elements
    ✓ Text properties for customizable content
    ✓ Instance swap properties for icons
    
  accessibility_check:
    ✓ Color contrast passes WCAG AA
    ✓ Touch targets ≥44px
    ✓ Focus states visible
    ✓ Interactive elements labeled
```

---

## 3. Design Tokens Setup

### 3.1 Token Structure

```json
{
  "global": {
    "color": {
      "primary": {
        "50": { "value": "#E3F2FD" },
        "100": { "value": "#BBDEFB" },
        "200": { "value": "#90CAF9" },
        "300": { "value": "#64B5F6" },
        "400": { "value": "#42A5F5" },
        "500": { "value": "#2196F3" },
        "600": { "value": "#1E88E5" },
        "700": { "value": "#1976D2" },
        "800": { "value": "#1565C0" },
        "900": { "value": "#0D47A1" }
      },
      "semantic": {
        "success": { "value": "{color.green.500}" },
        "warning": { "value": "{color.orange.500}" },
        "error": { "value": "{color.red.500}" },
        "info": { "value": "{color.blue.500}" }
      }
    },
    "spacing": {
      "xs": { "value": "4px" },
      "sm": { "value": "8px" },
      "md": { "value": "16px" },
      "lg": { "value": "24px" },
      "xl": { "value": "32px" },
      "xxl": { "value": "48px" }
    },
    "typography": {
      "fontFamily": {
        "primary": { "value": "Inter" },
        "mono": { "value": "Roboto Mono" }
      },
      "fontSize": {
        "xs": { "value": "12px" },
        "sm": { "value": "14px" },
        "md": { "value": "16px" },
        "lg": { "value": "20px" },
        "xl": { "value": "24px" },
        "xxl": { "value": "32px" }
      },
      "fontWeight": {
        "regular": { "value": "400" },
        "medium": { "value": "500" },
        "semibold": { "value": "600" },
        "bold": { "value": "700" }
      }
    },
    "borderRadius": {
      "sm": { "value": "4px" },
      "md": { "value": "8px" },
      "lg": { "value": "12px" },
      "full": { "value": "999px" }
    },
    "shadow": {
      "sm": { "value": "0 1px 3px rgba(0,0,0,0.12)" },
      "md": { "value": "0 3px 6px rgba(0,0,0,0.16)" },
      "lg": { "value": "0 10px 20px rgba(0,0,0,0.19)" },
      "xl": { "value": "0 14px 28px rgba(0,0,0,0.25)" }
    }
  }
}
```

### 3.2 Token Application

```yaml
token_usage:
  colors:
    - Use semantic tokens for UI elements
    - Use primitive tokens for illustrations
    - Never hardcode color values
    
  spacing:
    - Use spacing tokens for all margins/padding
    - Combine tokens for larger spaces (md + lg)
    - Keep consistent rhythm
    
  typography:
    - Create text styles from tokens
    - Use semantic naming (heading-1, body-text)
    - Include line-height in text styles
```

### 3.3 Token Export

```typescript
// Figma Tokens Plugin Configuration
const tokenExportConfig = {
  platforms: {
    css: {
      transformGroup: 'css',
      buildPath: 'tokens/css/',
      files: [{
        destination: 'variables.css',
        format: 'css/variables'
      }]
    },
    scss: {
      transformGroup: 'scss',
      buildPath: 'tokens/scss/',
      files: [{
        destination: '_variables.scss',
        format: 'scss/variables'
      }]
    },
    js: {
      transformGroup: 'js',
      buildPath: 'tokens/js/',
      files: [{
        destination: 'tokens.js',
        format: 'javascript/es6'
      }]
    }
  }
};
```

---

## 4. Component Specifications

### 4.1 Core Component Library (35 Components)

#### Base Components (10)

```yaml
1_button:
  variants: [primary, secondary, danger, ghost]
  sizes: [small, medium, large]
  states: [default, hover, active, focus, disabled, loading]
  specs:
    height: [32px, 40px, 48px]
    padding: ["8px 16px", "10px 20px", "12px 24px"]
    font: ["14px/20px", "14px/20px", "16px/24px"]

2_input:
  types: [text, number, password, search, textarea]
  states: [default, hover, focused, error, success, disabled]
  specs:
    height: 40px (48px with label)
    padding: 12px 16px
    border: 1px solid
    
3_select:
  states: [closed, open, selected]
  specs:
    min-width: 200px
    max-height: 240px (dropdown)
    
4_checkbox:
  states: [unchecked, checked, indeterminate, disabled]
  specs:
    size: 20px × 20px
    icon: 14px checkmark
    
5_radio:
  states: [unselected, selected, disabled]
  specs:
    size: 20px × 20px
    inner-circle: 8px
    
6_switch:
  states: [off, on, disabled]
  specs:
    width: 44px
    height: 24px
    thumb: 20px
    
7_badge:
  types: [status, count, label]
  colors: [primary, success, warning, error, neutral]
  
8_chip:
  variants: [default, deletable, clickable]
  sizes: [small, medium]
  
9_avatar:
  sizes: [small(32px), medium(40px), large(48px)]
  types: [image, initial, icon]
  
10_tooltip:
  positions: [top, right, bottom, left]
  max-width: 200px
```

#### Card Components (5)

```yaml
11_base_card:
  structure:
    - Optional header
    - Body content
    - Optional footer
  elevation: [none, low, medium, high]
  
12_metric_card:
  structure:
    - Large value
    - Label
    - Change indicator
    - Sparkline (optional)
  size: 300px × 120px (minimum)
  
13_channel_card:
  structure:
    - Channel avatar
    - Channel name & niche
    - Status indicator
    - 3 key metrics
    - Action buttons
  size: 320px × 240px
  
14_video_card:
  structure:
    - Thumbnail preview
    - Title (2 lines max)
    - Channel name
    - Metrics row
    - Status badge
  size: 320px × 280px
  
15_stat_card:
  structure:
    - Icon
    - Label
    - Value
    - Trend
  orientation: [horizontal, vertical]
```

#### Data Display Components (5)

```yaml
16_table:
  structure:
    - Sticky header
    - Sortable columns
    - Row hover state
    - Row selection
    - Pagination
  row-height: 48px
  
17_list:
  types: [simple, detailed, grouped]
  item-spacing: 8px
  dividers: optional
  
18_progress_bar:
  types: [determinate, indeterminate]
  height: 4px
  labels: optional
  
19_status_indicator:
  types: [dot, badge, icon]
  colors: [green, orange, red, blue, gray]
  size: 8px (dot), 20px (icon)
  
20_empty_state:
  structure:
    - Illustration (200×150px)
    - Title
    - Description
    - CTA button
  max-width: 400px
```

#### Navigation Components (5)

```yaml
21_sidebar:
  width: 240px
  structure:
    - Logo area (64px height)
    - Primary nav
    - Secondary nav
    - User section
  item-height: 40px
  
22_header:
  height: 64px
  structure:
    - Logo/Brand
    - Search (optional)
    - Actions/Navigation
    - User menu
  responsive: fixed
  
23_tabs:
  types: [line, contained]
  min-width: 80px per tab
  indicator: 2px line or background
  
24_breadcrumb:
  separator: "/"
  max-items: 4 (collapse middle)
  
25_pagination:
  structure:
    - Previous button
    - Page numbers
    - Next button
    - Items per page
  max-visible-pages: 7
```

#### Form Components (5)

```yaml
26_form_group:
  structure:
    - Label
    - Input/Control
    - Helper text
    - Error message
  spacing: 4px between elements
  
27_date_picker:
  types: [single, range]
  calendar-size: 280px × 280px
  
28_file_upload:
  types: [button, dropzone]
  dropzone-size: 100% × 120px
  
29_slider:
  types: [single, range]
  track-height: 4px
  thumb-size: 20px
  
30_stepper:
  types: [horizontal, vertical]
  step-size: 32px
  connector: 2px line
```

#### Feedback Components (5)

```yaml
31_modal:
  sizes: [small(400px), medium(600px), large(800px)]
  structure:
    - Header (optional)
    - Body
    - Footer
  overlay: rgba(0,0,0,0.5)
  
32_toast:
  position: bottom-right
  width: 360px (max)
  auto-dismiss: 5s
  
33_alert:
  types: [info, success, warning, error]
  variants: [standard, filled, outlined]
  
34_loading:
  types: [spinner, skeleton, progress]
  spinner-size: [16px, 24px, 32px]
  
35_popover:
  max-width: 320px
  arrow: 8px
  offset: 8px from trigger
```

---

## 5. Prototyping Guidelines

### 5.1 Interaction Standards

```yaml
prototype_interactions:
  click_tap:
    - Use for buttons, links, cards
    - Instant transition (0ms)
    
  hover:
    - Desktop only
    - Show tooltips, button states
    - 200ms ease-out
    
  drag:
    - Reserved for sliders, reordering
    - Follow cursor precisely
    
  scroll:
    - Vertical scroll for long content
    - Smooth scrolling enabled
```

### 5.2 Flow Connections

```yaml
user_flows:
  onboarding:
    screens: 6-8
    connections: Linear with back option
    completion: Dashboard
    
  video_generation:
    screens: 4
    connections: Modal overlay
    completion: Return to origin
    
  channel_management:
    screens: 5-7
    connections: Navigate and return
    completion: Save or cancel
```

### 5.3 Prototype Animations

```yaml
animations:
  page_transitions:
    type: "Smart animate"
    duration: 300ms
    easing: "Ease out"
    
  micro_interactions:
    type: "Smart animate"
    duration: 200ms
    easing: "Ease in out"
    
  loading_states:
    type: "After delay"
    delay: 1000ms
    show: Loading skeleton
```

---

## 6. Developer Handoff

### 6.1 Figma Dev Mode Setup

```yaml
dev_mode_preparation:
  components:
    ✓ All components properly named
    ✓ Properties clearly defined
    ✓ Constraints set correctly
    ✓ Text styles applied
    ✓ Color styles used
    
  spacing:
    ✓ Auto-layout everywhere possible
    ✓ Consistent padding values
    ✓ Proper alignment
    
  assets:
    ✓ Icons exported as SVG
    ✓ Images optimized
    ✓ Export settings configured
```

### 6.2 Annotation Standards

```yaml
annotations:
  measurements:
    - Show spacing between elements
    - Indicate component sizes
    - Mark responsive breakpoints
    
  interactions:
    - Describe hover states
    - Note animation timings
    - Specify transitions
    
  edge_cases:
    - Empty states
    - Error states
    - Loading states
    - Overflow behavior
```

### 6.3 Export Specifications

```yaml
export_settings:
  icons:
    format: SVG
    size: Original
    naming: "icon-{name}.svg"
    
  images:
    format: WebP with PNG fallback
    sizes: [1x, 2x]
    naming: "{name}@{scale}x.{ext}"
    
  colors:
    format: CSS variables
    naming: "--color-{name}-{shade}"
    
  typography:
    format: CSS classes
    naming: ".text-{style}"
```

---

## 7. Usability Testing Protocol

### 7.1 Testing Methodology

```yaml
testing_approach:
  type: "Moderated remote testing"
  participants: 5-8 per round
  duration: 45-60 minutes
  frequency: Every 2 weeks during MVP
  
recruitment_criteria:
  primary:
    - YouTube content creators
    - Managing 2+ channels
    - Basic technical literacy
    
  secondary:
    - Aspiring creators
    - Digital marketers
    - Online entrepreneurs
```

### 7.2 Test Scenarios

```yaml
core_scenarios:
  1_first_time_setup:
    goal: "Create first channel"
    success_metrics:
      - Time to complete < 10 minutes
      - Error rate < 10%
      - Satisfaction > 4/5
      
  2_daily_monitoring:
    goal: "Check performance and make decisions"
    tasks:
      - Review overnight performance
      - Identify best performing video
      - Pause underperforming channel
      
  3_video_generation:
    goal: "Generate a new video"
    tasks:
      - Select channel
      - Choose/approve topic
      - Monitor progress
      - Handle errors
      
  4_bulk_operations:
    goal: "Manage multiple channels"
    tasks:
      - Select 3 channels
      - Apply same settings
      - Bulk pause/resume
```

### 7.3 Testing Script

```markdown
## Introduction (5 minutes)
- Welcome and comfort building
- Explain think-aloud protocol
- Set expectations (testing design, not user)
- Get consent to record

## Background Questions (5 minutes)
1. Tell me about your YouTube experience
2. How many channels do you manage?
3. What tools do you currently use?
4. What's your biggest challenge?

## Task Scenarios (30 minutes)
[Present scenarios based on participant profile]

### Scenario 1: First Channel Setup
"Imagine you just signed up for YTEMPIRE. Your goal is to create your first automated YouTube channel in the tech niche."

Observe:
- Navigation clarity
- Form completion
- Error recovery
- Time to complete

### Scenario 2: Daily Dashboard Review
"It's morning, and you want to check how your channels performed overnight."

Observe:
- Information finding
- Metric understanding
- Action taking
- Navigation patterns

## Post-Test Questions (10 minutes)
1. What was your overall impression?
2. What was confusing or frustrating?
3. What did you like most?
4. Would you recommend this to others?
5. Any features missing?

## Debrief (5 minutes)
- Thank participant
- Explain next steps
- Provide compensation
```

### 7.4 Analysis Framework

```yaml
data_collection:
  quantitative:
    - Task completion rate
    - Time on task
    - Error frequency
    - Click/tap counts
    
  qualitative:
    - Verbal feedback
    - Facial expressions
    - Hesitation points
    - Confusion moments
    
analysis_method:
  1. Review recordings
  2. Tag issues by severity
  3. Identify patterns
  4. Prioritize fixes
  5. Create action items
  
severity_scale:
  critical: "Prevents task completion"
  major: "Causes significant delay/frustration"
  minor: "Noticeable but doesn't impede"
  enhancement: "Would improve experience"
```

### 7.5 Reporting Template

```markdown
# Usability Test Report - [Date]

## Executive Summary
- Participants: X
- Overall Success Rate: X%
- Critical Issues: X
- Key Recommendations: [List]

## Methodology
- Remote moderated testing
- [X] participants
- [Date range]

## Key Findings

### 1. [Finding Title]
- Severity: [Critical/Major/Minor]
- Frequency: X/X participants
- Description: [What happened]
- User Quote: "[Relevant quote]"
- Recommendation: [Specific fix]
- Screenshot: [Annotated image]

## Success Metrics
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Task Success | >90% | X% | ✓/✗ |
| Time on Task | <5min | Xmin | ✓/✗ |
| Satisfaction | >4/5 | X/5 | ✓/✗ |

## Recommendations Priority
1. [Critical fix]
2. [Major improvement]
3. [Enhancement]

## Next Steps
- [ ] Address critical issues
- [ ] Update designs
- [ ] Retest problem areas
```

---

## 8. Brand Guidelines

### 8.1 Brand Foundation

```yaml
brand_personality:
  core_traits:
    - Innovative
    - Reliable
    - Empowering
    - Approachable
    
  voice_attributes:
    - Professional but friendly
    - Clear and concise
    - Encouraging
    - Action-oriented
    
  not_this:
    - Overly technical
    - Condescending
    - Hyperbolic
    - Pushy
```

### 8.2 Visual Identity

```yaml
logo_usage:
  primary:
    - Full color on light
    - Minimum size: 120px width
    - Clear space: 1x height
    
  variations:
    - Horizontal layout
    - Icon only (favicon)
    - Monochrome
    
  dont:
    - Stretch or distort
    - Change colors
    - Add effects
    - Crowd with elements
```

### 8.3 Color Application

```scss
// Brand Color Usage
$brand-colors: (
  // Primary - Actions, links, focus
  primary: #2196F3,
  primary-use: "CTAs, primary buttons, active states",
  
  // Accent - Highlights, alerts
  accent: #FF9800,
  accent-use: "Important alerts, promotional elements",
  
  // Success - Positive states
  success: #4CAF50,
  success-use: "Success messages, positive metrics",
  
  // Error - Problems, warnings
  error: #F44336,
  error-use: "Error states, critical alerts",
  
  // Neutral - UI elements
  neutral: #9E9E9E,
  neutral-use: "Borders, backgrounds, secondary text"
);

// Color Accessibility
$color-contrast: (
  // Text on backgrounds
  primary-on-white: 4.5:1,  // WCAG AA
  white-on-primary: 7.1:1,  // WCAG AAA
  
  // Important elements
  error-on-white: 4.5:1,
  success-on-white: 4.5:1
);
```

### 8.4 Typography Usage

```yaml
typography_hierarchy:
  display:
    font: "Inter"
    weight: 700
    size: "48-64px"
    use: "Marketing headlines only"
    
  headings:
    h1: "32px/40px, 600 weight - Page titles"
    h2: "24px/32px, 600 weight - Section headers"
    h3: "20px/28px, 500 weight - Card titles"
    h4: "16px/24px, 500 weight - Subsections"
    
  body:
    large: "16px/24px - Primary content"
    medium: "14px/20px - Default UI text"
    small: "12px/16px - Secondary info"
    
  data:
    numbers: "Roboto Mono - Metrics, currency"
    tables: "14px/20px - Consistent alignment"
```

### 8.5 Iconography

```yaml
icon_style:
  style: "Outlined"
  weight: 2px stroke
  size: [16px, 20px, 24px]
  grid: 24px base
  
icon_sources:
  primary: "Material Icons Outlined"
  custom: "Created on 24px grid"
  
usage_guidelines:
  - Always pair with text for clarity
  - Maintain consistent sizing
  - Use semantic meaning
  - Ensure sufficient contrast
```

### 8.6 Motion Principles

```yaml
animation_brand:
  personality: "Smooth and purposeful"
  
  timing:
    instant: 100ms - "Immediate feedback"
    fast: 200ms - "Micro-interactions"
    standard: 300ms - "State changes"
    slow: 500ms - "Complex transitions"
    
  easing:
    standard: "cubic-bezier(0.4, 0.0, 0.2, 1)"
    emphasize: "cubic-bezier(0.0, 0.0, 0.2, 1)"
    
  principles:
    - Animations serve a purpose
    - Natural, not mechanical
    - Consistent timing
    - Respect reduced motion preference
```

---

## Implementation Timeline

### Week 1: Foundation Setup
- [ ] Create Figma file structure
- [ ] Set up design tokens
- [ ] Build color and type styles
- [ ] Create grid templates

### Week 2: Component Building
- [ ] Design base components (1-10)
- [ ] Create card components (11-15)
- [ ] Build data display (16-20)

### Week 3: Complex Components
- [ ] Design navigation (21-25)
- [ ] Create form components (26-30)
- [ ] Build feedback components (31-35)

### Week 4: Screen Design
- [ ] Dashboard layouts
- [ ] Channel management screens
- [ ] Video generation flow
- [ ] Empty states

### Week 5: Prototyping
- [ ] Connect user flows
- [ ] Add micro-interactions
- [ ] Create demo prototype
- [ ] Prepare for testing

### Week 6: Testing & Handoff
- [ ] Conduct usability tests
- [ ] Refine based on feedback
- [ ] Prepare dev documentation
- [ ] Final handoff

---

## Quality Checklist

### Before Handoff
- [ ] All 35 components complete
- [ ] States for every component
- [ ] Consistent naming throughout
- [ ] Design tokens applied
- [ ] Prototypes functional
- [ ] Documentation complete
- [ ] Assets exported
- [ ] Dev mode configured
- [ ] Accessibility checked
- [ ] Usability tested

---

## Resources & Links

### Figma Files
- Design System: [Link]
- Components: [Link]
- Prototypes: [Link]

### Documentation
- Token Documentation: [Link]
- Component Specs: [Link]
- Brand Guidelines: [Link]

### Tools & Plugins
- Figma Tokens
- Able (Accessibility)
- Figma to Code
- Design Lint

---

**Remember**: This component library is the foundation of YTEMPIRE's user experience. Every component should embody our principle of making complex operations feel simple. Quality over quantity - better to have 35 polished components than 50 rough ones. │Default  │ │Hover    │ │Active   ││
│ └─────────┘ └─────────┘ └─────────┘│
│ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│