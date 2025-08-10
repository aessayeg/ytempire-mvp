# YTEmpire Design System Documentation

## Brand Identity

### Logo
- **Primary Logo**: YTEmpire text with crown icon
- **Icon**: Stylized "Y" with play button integration
- **Minimum Size**: 32px height
- **Clear Space**: 1x icon height on all sides

### Brand Values
- **Innovation**: Cutting-edge AI technology
- **Efficiency**: Automation and optimization
- **Growth**: Scaling YouTube success
- **Reliability**: Consistent performance

## Color Palette

### Primary Colors
```css
--primary-600: #667eea;  /* Main brand color */
--primary-500: #7c8ff0;
--primary-400: #92a1f6;
--primary-300: #a8b3fc;
--primary-200: #c4ccfd;
--primary-100: #e0e5fe;
```

### Secondary Colors
```css
--secondary-600: #764ba2;  /* Accent color */
--secondary-500: #8b5db5;
--secondary-400: #a070c8;
--secondary-300: #b582db;
--secondary-200: #ca95ee;
--secondary-100: #e5ccf7;
```

### Semantic Colors
```css
/* Success */
--success-600: #10b981;
--success-500: #34d399;
--success-100: #d1fae5;

/* Warning */
--warning-600: #f59e0b;
--warning-500: #fbbf24;
--warning-100: #fef3c7;

/* Error */
--error-600: #ef4444;
--error-500: #f87171;
--error-100: #fee2e2;

/* Info */
--info-600: #3b82f6;
--info-500: #60a5fa;
--info-100: #dbeafe;
```

### Neutral Colors
```css
--gray-900: #111827;  /* Primary text */
--gray-800: #1f2937;
--gray-700: #374151;
--gray-600: #4b5563;
--gray-500: #6b7280;  /* Secondary text */
--gray-400: #9ca3af;
--gray-300: #d1d5db;
--gray-200: #e5e7eb;
--gray-100: #f3f4f6;
--gray-50:  #f9fafb;  /* Background */
--white:    #ffffff;
```

## Typography

### Font Families
```css
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
--font-mono: 'Fira Code', 'Courier New', monospace;
```

### Font Sizes
```css
--text-xs:   0.75rem;   /* 12px */
--text-sm:   0.875rem;  /* 14px */
--text-base: 1rem;      /* 16px */
--text-lg:   1.125rem;  /* 18px */
--text-xl:   1.25rem;   /* 20px */
--text-2xl:  1.5rem;    /* 24px */
--text-3xl:  1.875rem;  /* 30px */
--text-4xl:  2.25rem;   /* 36px */
--text-5xl:  3rem;      /* 48px */
```

### Font Weights
```css
--font-light:    300;
--font-regular:  400;
--font-medium:   500;
--font-semibold: 600;
--font-bold:     700;
```

### Line Heights
```css
--leading-tight:   1.25;
--leading-snug:    1.375;
--leading-normal:  1.5;
--leading-relaxed: 1.625;
--leading-loose:   2;
```

## Spacing System

### Base Unit: 4px
```css
--space-1:  0.25rem;  /* 4px */
--space-2:  0.5rem;   /* 8px */
--space-3:  0.75rem;  /* 12px */
--space-4:  1rem;     /* 16px */
--space-5:  1.25rem;  /* 20px */
--space-6:  1.5rem;   /* 24px */
--space-8:  2rem;     /* 32px */
--space-10: 2.5rem;   /* 40px */
--space-12: 3rem;     /* 48px */
--space-16: 4rem;     /* 64px */
--space-20: 5rem;     /* 80px */
--space-24: 6rem;     /* 96px */
```

## Border Radius
```css
--radius-sm: 0.125rem;  /* 2px */
--radius-md: 0.375rem;  /* 6px */
--radius-lg: 0.5rem;    /* 8px */
--radius-xl: 0.75rem;   /* 12px */
--radius-2xl: 1rem;     /* 16px */
--radius-full: 9999px;  /* Full round */
```

## Shadows
```css
--shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
--shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
```

## Components

### Buttons

#### Primary Button
```tsx
<Button variant="primary" size="md">
  Generate Video
</Button>
```
- Background: `--primary-600`
- Text: `white`
- Hover: `--primary-500`
- Active: `--primary-700`
- Disabled: `--gray-300`

#### Secondary Button
```tsx
<Button variant="secondary" size="md">
  Cancel
</Button>
```
- Background: `white`
- Border: `--gray-300`
- Text: `--gray-700`
- Hover: `--gray-50`

#### Sizes
- `sm`: Height 32px, padding 8px 12px, font-size 14px
- `md`: Height 40px, padding 10px 16px, font-size 16px
- `lg`: Height 48px, padding 12px 20px, font-size 18px

### Input Fields

```tsx
<Input 
  type="text"
  placeholder="Enter video title"
  error={false}
/>
```
- Height: 40px
- Border: `--gray-300`
- Focus border: `--primary-500`
- Error border: `--error-500`
- Border radius: `--radius-md`

### Cards

```tsx
<Card variant="elevated">
  <CardHeader>
    <CardTitle>Video Analytics</CardTitle>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```
- Background: `white`
- Border: `--gray-200` (outlined variant)
- Shadow: `--shadow-md` (elevated variant)
- Border radius: `--radius-lg`
- Padding: `--space-6`

### Navigation

#### Sidebar
- Width: 280px (expanded), 80px (collapsed)
- Background: `--gray-900`
- Text: `white`
- Active item: `--primary-600` background

#### Header
- Height: 64px
- Background: `white`
- Shadow: `--shadow-sm`
- Z-index: 1000

### Data Visualization

#### Charts Color Scheme
```javascript
const chartColors = [
  '#667eea', // Primary
  '#764ba2', // Secondary
  '#10b981', // Success
  '#f59e0b', // Warning
  '#3b82f6', // Info
  '#ef4444', // Error
  '#8b5d15', // Brown
  '#ec4899', // Pink
];
```

### Modals

```tsx
<Modal
  open={isOpen}
  onClose={handleClose}
  size="md"
>
  <ModalHeader>Confirm Action</ModalHeader>
  <ModalContent>
    {/* Content */}
  </ModalContent>
  <ModalFooter>
    <Button variant="secondary">Cancel</Button>
    <Button variant="primary">Confirm</Button>
  </ModalFooter>
</Modal>
```
- Overlay: `rgba(0, 0, 0, 0.5)`
- Background: `white`
- Border radius: `--radius-xl`
- Sizes: `sm` (400px), `md` (600px), `lg` (800px), `xl` (1000px)

### Alerts

```tsx
<Alert severity="success">
  Video published successfully!
</Alert>
```
- Success: Green background `--success-100`, border `--success-600`
- Warning: Yellow background `--warning-100`, border `--warning-600`
- Error: Red background `--error-100`, border `--error-600`
- Info: Blue background `--info-100`, border `--info-600`

## Layout Grid

### Container
```css
.container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 var(--space-4);
}
```

### Grid System
- 12-column grid
- Gutter: `--space-6` (24px)
- Breakpoints:
  - `sm`: 640px
  - `md`: 768px
  - `lg`: 1024px
  - `xl`: 1280px
  - `2xl`: 1536px

## Icons

### Icon Library: Heroicons & Material Icons
- Size variants: 16px, 20px, 24px, 32px
- Stroke width: 1.5px (outline), filled
- Color: Inherit from parent

### Common Icons
- Dashboard: `<HomeIcon />`
- Videos: `<VideoCameraIcon />`
- Analytics: `<ChartBarIcon />`
- Settings: `<CogIcon />`
- Notifications: `<BellIcon />`
- User: `<UserCircleIcon />`

## Animation

### Transitions
```css
--transition-fast: 150ms ease-in-out;
--transition-base: 250ms ease-in-out;
--transition-slow: 350ms ease-in-out;
```

### Common Animations
```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { transform: translateY(10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

## Responsive Design

### Mobile First Approach
```css
/* Mobile (default) */
.component { font-size: 14px; }

/* Tablet */
@media (min-width: 768px) {
  .component { font-size: 16px; }
}

/* Desktop */
@media (min-width: 1024px) {
  .component { font-size: 18px; }
}
```

### Breakpoint Usage
- `sm`: Mobile landscape
- `md`: Tablet portrait
- `lg`: Tablet landscape / Small desktop
- `xl`: Desktop
- `2xl`: Large desktop

## Accessibility

### Color Contrast
- Normal text: Minimum 4.5:1 ratio
- Large text: Minimum 3:1 ratio
- Interactive elements: Minimum 3:1 ratio

### Focus States
```css
:focus-visible {
  outline: 2px solid var(--primary-500);
  outline-offset: 2px;
}
```

### ARIA Labels
- All interactive elements must have accessible names
- Use semantic HTML where possible
- Provide skip links for navigation

## Dark Mode

### Dark Theme Colors
```css
[data-theme="dark"] {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --border: #334155;
}
```

### Implementation
```tsx
const toggleTheme = () => {
  document.documentElement.setAttribute(
    'data-theme',
    theme === 'light' ? 'dark' : 'light'
  );
};
```

## Component States

### Interactive States
1. **Default**: Base appearance
2. **Hover**: Cursor over element
3. **Active**: Being clicked/pressed
4. **Focus**: Keyboard navigation
5. **Disabled**: Non-interactive
6. **Loading**: Processing action
7. **Error**: Invalid state

## Usage Guidelines

### Do's
- ✅ Use consistent spacing
- ✅ Follow color semantic meaning
- ✅ Maintain visual hierarchy
- ✅ Test on multiple devices
- ✅ Ensure accessibility compliance

### Don'ts
- ❌ Mix different design patterns
- ❌ Use colors outside the palette
- ❌ Create custom breakpoints
- ❌ Ignore focus states
- ❌ Use inline styles for common patterns

## Figma Integration

### File Structure
```
YTEmpire Design System/
├── 📁 Foundations
│   ├── Colors
│   ├── Typography
│   ├── Spacing
│   └── Effects
├── 📁 Components
│   ├── Buttons
│   ├── Forms
│   ├── Cards
│   ├── Navigation
│   └── Data Display
├── 📁 Patterns
│   ├── Forms
│   ├── Tables
│   └── Dashboards
└── 📁 Pages
    ├── Dashboard
    ├── Video Management
    └── Analytics
```

### Component Naming
- Use slash naming: `Button/Primary/Medium`
- Include all variants
- Document properties
- Add usage notes

---

*Design System Version: 1.0*
*Last Updated: 2024*
*Maintained by: YTEmpire UI/UX Team*