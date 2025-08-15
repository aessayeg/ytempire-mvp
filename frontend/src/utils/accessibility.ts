// WCAG 2.1 AA Compliance Utilities

// Color contrast checking
export const getContrastRatio = (color1: string, color2: string): number => {
  const getLuminance = (color: string): number => {
    const rgb = color.match(/\d+/g);
    if (!rgb || rgb.length < 3) return 0;
    
    const [r, g, b] = rgb.map((x) => {
      const val = parseInt(x) / 255;
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4)});
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  
  return (brightest + 0.05) / (darkest + 0.05)};

// Check if color combination meets WCAG AA standards


// Check if color combination meets WCAG AAA standards


// Keyboard navigation utilities
export const FOCUSABLE_ELEMENTS = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])'];

export const getFocusableElements = (container: HTMLElement): HTMLElement[] => {
  const elements = container.querySelectorAll<HTMLElement>(
    FOCUSABLE_ELEMENTS.join(',')
  );
  return Array.from(elements)};



  container.addEventListener('keydown', handleKeyDown);
  
  // Focus first element
  firstElement?.focus();

  // Return cleanup function
  return () => {
    
    container.removeEventListener('keydown', handleKeyDown)};

// Screen reader announcements


// Skip navigation link management


// ARIA attributes helper

  
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


// Focus visible management

  
  const onKeyDown = (_: KeyboardEvent) => {
    if (e.key === 'Tab') {
      hadKeyboardEvent = true;
    }
  };
  
  const onFocus = (_: FocusEvent) => {
    if (hadKeyboardEvent || (e.target as HTMLElement).matches(':focus-visible')) {
      document.body.classList.add('keyboard-focused')}
  };
  
  const onBlur = () => {
    document.body.classList.remove('keyboard-focused')};
  
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('pointerdown', onPointerDown);
  document.addEventListener('focus', onFocus, true);
  document.addEventListener('blur', onBlur, true);
  
  return () => {
    
    document.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('pointerdown', onPointerDown);
    document.removeEventListener('focus', onFocus, true);
    document.removeEventListener('blur', onBlur, true)};

// Reduced motion preference


// High contrast mode detection


// Dark mode preference


// Text spacing utilities for readability


// Form label association validation
`