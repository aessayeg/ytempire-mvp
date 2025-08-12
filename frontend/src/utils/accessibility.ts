// WCAG 2.1 AA Compliance Utilities

// Color contrast checking
export const getContrastRatio = (color1: string, color2: string): number => {
  const getLuminance = (color: string): number => {
    const rgb = color.match(/\d+/g);
    if (!rgb || rgb.length < 3) return 0;
    
    const [r, g, b] = rgb.map((x) => {
      const val = parseInt(x) / 255;
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  
  return (brightest + 0.05) / (darkest + 0.05);
};

// Check if color combination meets WCAG AA standards
export const meetsWCAGAA = (
  foreground: string,
  background: string,
  largeText = false
): boolean => {
  const ratio = getContrastRatio(foreground, background);
  return largeText ? ratio >= 3 : ratio >= 4.5;
};

// Check if color combination meets WCAG AAA standards
export const meetsWCAGAAA = (
  foreground: string,
  background: string,
  largeText = false
): boolean => {
  const ratio = getContrastRatio(foreground, background);
  return largeText ? ratio >= 4.5 : ratio >= 7;
};

// Keyboard navigation utilities
export const FOCUSABLE_ELEMENTS = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
];

export const getFocusableElements = (container: HTMLElement): HTMLElement[] => {
  const elements = container.querySelectorAll<HTMLElement>(
    FOCUSABLE_ELEMENTS.join(',')
  );
  return Array.from(elements);
};

export const trapFocus = (container: HTMLElement) => {
  const focusableElements = getFocusableElements(container);
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);
  
  // Focus first element
  firstElement?.focus();

  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', handleKeyDown);
  };
};

// Screen reader announcements
export const announce = (
  message: string,
  priority: 'polite' | 'assertive' = 'polite'
) => {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.style.position = 'absolute';
  announcement.style.left = '-10000px';
  announcement.style.width = '1px';
  announcement.style.height = '1px';
  announcement.style.overflow = 'hidden';
  
  announcement.textContent = message;
  document.body.appendChild(announcement);
  
  // Remove after announcement
  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
};

// Skip navigation link management
export const createSkipLink = (targetId: string, text = 'Skip to main content') => {
  const link = document.createElement('a');
  link.href = `#${targetId}`;
  link.className = 'skip-link';
  link.textContent = text;
  
  link.style.position = 'absolute';
  link.style.left = '-9999px';
  link.style.zIndex = '999';
  
  link.addEventListener('focus', () => {
    link.style.left = '0';
    link.style.top = '0';
  });
  
  link.addEventListener('blur', () => {
    link.style.left = '-9999px';
  });
  
  return link;
};

// ARIA attributes helper
export const getAriaProps = (props: {
  label?: string;
  labelledBy?: string;
  describedBy?: string;
  required?: boolean;
  invalid?: boolean;
  expanded?: boolean;
  selected?: boolean;
  checked?: boolean;
  disabled?: boolean;
  hidden?: boolean;
  busy?: boolean;
  live?: 'polite' | 'assertive' | 'off';
  role?: string;
}) => {
  const ariaProps: Record<string, any> = {};
  
  if (props.label) ariaProps['aria-label'] = props.label;
  if (props.labelledBy) ariaProps['aria-labelledby'] = props.labelledBy;
  if (props.describedBy) ariaProps['aria-describedby'] = props.describedBy;
  if (props.required !== undefined) ariaProps['aria-required'] = props.required;
  if (props.invalid !== undefined) ariaProps['aria-invalid'] = props.invalid;
  if (props.expanded !== undefined) ariaProps['aria-expanded'] = props.expanded;
  if (props.selected !== undefined) ariaProps['aria-selected'] = props.selected;
  if (props.checked !== undefined) ariaProps['aria-checked'] = props.checked;
  if (props.disabled !== undefined) ariaProps['aria-disabled'] = props.disabled;
  if (props.hidden !== undefined) ariaProps['aria-hidden'] = props.hidden;
  if (props.busy !== undefined) ariaProps['aria-busy'] = props.busy;
  if (props.live) ariaProps['aria-live'] = props.live;
  if (props.role) ariaProps['role'] = props.role;
  
  return ariaProps;
};

// Heading hierarchy validation
export const validateHeadingHierarchy = (container: HTMLElement): boolean => {
  const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
  let lastLevel = 0;
  let isValid = true;
  
  headings.forEach((heading) => {
    const level = parseInt(heading.tagName.substring(1));
    if (lastLevel > 0 && level > lastLevel + 1) {
      console.warn(`Heading hierarchy issue: h${lastLevel} followed by h${level}`);
      isValid = false;
    }
    lastLevel = level;
  });
  
  return isValid;
};

// Focus visible management
export const manageFocusVisible = () => {
  let hadKeyboardEvent = false;
  
  const onPointerDown = () => {
    hadKeyboardEvent = false;
  };
  
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Tab') {
      hadKeyboardEvent = true;
    }
  };
  
  const onFocus = (e: FocusEvent) => {
    if (hadKeyboardEvent || (e.target as HTMLElement).matches(':focus-visible')) {
      document.body.classList.add('keyboard-focused');
    }
  };
  
  const onBlur = () => {
    document.body.classList.remove('keyboard-focused');
  };
  
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('pointerdown', onPointerDown);
  document.addEventListener('focus', onFocus, true);
  document.addEventListener('blur', onBlur, true);
  
  return () => {
    document.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('pointerdown', onPointerDown);
    document.removeEventListener('focus', onFocus, true);
    document.removeEventListener('blur', onBlur, true);
  };
};

// Reduced motion preference
export const prefersReducedMotion = (): boolean => {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

// High contrast mode detection
export const prefersHighContrast = (): boolean => {
  return window.matchMedia('(prefers-contrast: high)').matches;
};

// Dark mode preference
export const prefersDarkMode = (): boolean => {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

// Text spacing utilities for readability
export const getReadableTextStyles = () => ({
  lineHeight: 1.5,
  letterSpacing: '0.12em',
  wordSpacing: '0.16em',
  paragraphSpacing: '2em',
});

// Alternative text validation
export const validateAltText = (img: HTMLImageElement): boolean => {
  const alt = img.getAttribute('alt');
  if (alt === null) {
    console.warn('Image missing alt attribute:', img.src);
    return false;
  }
  if (alt === '' && !img.getAttribute('role')) {
    console.warn('Decorative image should have role="presentation":', img.src);
    return false;
  }
  return true;
};

// Form label association validation
export const validateFormLabels = (form: HTMLFormElement): boolean => {
  const inputs = form.querySelectorAll('input, select, textarea');
  let isValid = true;
  
  inputs.forEach((input) => {
    const id = input.getAttribute('id');
    const ariaLabel = input.getAttribute('aria-label');
    const ariaLabelledBy = input.getAttribute('aria-labelledby');
    
    if (!id || (!document.querySelector(`label[for="${id}"]`) && !ariaLabel && !ariaLabelledBy)) {
      console.warn('Form input missing label:', input);
      isValid = false;
    }
  });
  
  return isValid;
};