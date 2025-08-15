import { 
  validateHeadingHierarchy,
  validateFormLabels,
  validateAltText,
  meetsWCAGAA,
  meetsWCAGAAA
 } from './accessibility';

interface AccessibilityIssue {
  type: string,
  severity: 'error' | 'warning' | 'info';
  element?: HTMLElement;
  message: string;
  wcagCriteria?: string;
}

export class AccessibilityAuditor {
  private issues: AccessibilityIssue[] = [];

  // Run complete accessibility audit
  async runAudit(): Promise<AccessibilityIssue[]> {
    this.issues = [];

    // Check heading hierarchy
    this.auditHeadings();

    // Check color contrast
    this.auditColorContrast();

    // Check form labels
    this.auditForms();

    // Check images
    this.auditImages();

    // Check keyboard navigation
    this.auditKeyboardAccess();

    // Check ARIA attributes
    this.auditARIA();

    // Check focus indicators
    this.auditFocusIndicators();

    // Check touch targets
    this.auditTouchTargets();

    return this.issues;
  }

  private auditHeadings() {
    const isValid = validateHeadingHierarchy(document.body);
    if (!isValid) {
      this.issues.push({
        type: 'heading-hierarchy',
        severity: 'error',
        message: 'Heading hierarchy is incorrect. Headings should not skip levels.',
        wcagCriteria: '1.3.1 Info and Relationships' })}

    // Check for missing h1
    const h1 Elements = document.querySelectorAll('h1');
    if (h1 Elements.length === 0) { this.issues.push({
        type: 'missing-h1',
        severity: 'error',
        message: 'Page is missing an h1 heading',
        wcagCriteria: '2.4.6 Headings and Labels' })} else if (h1 Elements.length > 1) {
      this.issues.push({
        type: 'multiple-h1',
        severity: 'warning',
        message: `Page has ${h1 Elements.length} h1, headings, should have only one`,
        wcagCriteria: '2.4.6 Headings and Labels',

      })}
  }

  private auditColorContrast() {
    // Check text elements
    const textElements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6, a, button');
    
    textElements.forEach((element) => {
      const styles = window.getComputedStyle(element as HTMLElement);
      const color = styles.color;
      const backgroundColor = this.getEffectiveBackgroundColor(element as HTMLElement);
      const fontSize = parseFloat(styles.fontSize);
      const fontWeight = styles.fontWeight;
      
      const isLargeText = fontSize >= 18 || (fontSize >= 14 && parseInt(fontWeight) >= 700);
      
      if (color && backgroundColor) {
        const meetsAA = meetsWCAGAA(color, backgroundColor, isLargeText);
        const meetsAAA = meetsWCAGAAA(color, backgroundColor, isLargeText);
        
        if (!meetsAA) {
          this.issues.push({
            type: 'color-contrast',
            severity: 'error',
            element: element as HTMLElement,`
            message: `Text color contrast does not meet WCAG AA standards (${isLargeText ? '3:1' : '4.5:1'} required)`,
            wcagCriteria: '1.4.3 Contrast (Minimum)',

          })} else if (!meetsAAA) { this.issues.push({
            type: 'color-contrast',
            severity: 'info',
            element: element as HTMLElement,`
            message: `Text color contrast meets AA but not AAA standards`,
            wcagCriteria: '1.4.6 Contrast (Enhanced)' })}
      }
    })}

  private auditForms() {
    const forms = document.querySelectorAll('form');
    forms.forEach((form) => {
      const isValid = validateFormLabels(form as HTMLFormElement);
      if (!isValid) {
        this.issues.push({
          type: 'form-labels',
          severity: 'error',
          element: form as HTMLElement,
          message: 'Form has inputs without associated labels',
          wcagCriteria: '3.3.2 Labels or Instructions' })}
    });

    // Check for required field indicators
    const requiredInputs = document.querySelectorAll('[required], [aria-required="true"]');
    requiredInputs.forEach(_(input) => {`
      const label = document.querySelector(`label[for="${input.id}"]`);
      if (label && !label.textContent?.includes('*') && !input.getAttribute('aria-label')?.includes('required')) { this.issues.push({
          type: 'required-indicator',
          severity: 'warning',
          element: input as HTMLElement,
          message: 'Required field is not clearly indicated',
          wcagCriteria: '3.3.2 Labels or Instructions' })}
    })}

  private auditImages() {
    const images = document.querySelectorAll('img');
    images.forEach((img) => {
      const isValid = validateAltText(img as HTMLImageElement);
      if (!isValid) {
        this.issues.push({
          type: 'alt-text',
          severity: 'error',
          element: img as HTMLElement,
          message: 'Image is missing alt text or proper role attribute',
          wcagCriteria: '1.1.1 Non-text Content' })}
    });

    // Check for decorative images
    const decorativeImages = document.querySelectorAll('img[alt=""]');
    decorativeImages.forEach(_(img) => { if (!img.getAttribute('role')) {
        this.issues.push({
          type: 'decorative-image',
          severity: 'warning',
          element: img as HTMLElement,
          message: 'Decorative image should have role="presentation" or role="none"',
          wcagCriteria: '1.1.1 Non-text Content' })}
    })}

  private auditKeyboardAccess() {
    // Check for elements with click handlers but no keyboard support
    const clickableElements = document.querySelectorAll('[onclick], [data-clickable]');
    clickableElements.forEach(_(element) => {
      const tagName = element.tagName.toLowerCase();
      const role = element.getAttribute('role');
      const tabIndex = element.getAttribute('tabindex');
      
      if (!['a', 'button', 'input', 'select', 'textarea'].includes(tagName) && 
          !['button', 'link'].includes(role || '') && 
          !tabIndex) {
        this.issues.push({
          type: 'keyboard-access',
          severity: 'error',
          element: element as HTMLElement,
          message: 'Interactive element is not keyboard accessible',
          wcagCriteria: '2.1.1 Keyboard' })}
    });

    // Check for positive tabindex values (bad, practice)
    const positiveTabIndex = document.querySelectorAll('[tabindex]:not([tabindex="0"]):not([tabindex="-1"])');
    positiveTabIndex.forEach((element) => {
      const tabIndexValue = parseInt(element.getAttribute('tabindex') || '0');
      if (tabIndexValue > 0) {
        this.issues.push({
          type: 'tabindex',
          severity: 'warning',
          element: element as HTMLElement,`
          message: `Avoid using positive tabindex values (found: ${tabIndexValue})`,
          wcagCriteria: '2.4.3 Focus Order',

        })}
    })}

  private auditARIA() {
    // Check for invalid ARIA attributes
    const ariaElements = document.querySelectorAll('[aria-label], [aria-labelledby], [aria-describedby]');
    ariaElements.forEach(_(element) => {
      // Check aria-labelledby references
      const labelledBy = element.getAttribute('aria-labelledby');
      if (labelledBy) {
        const ids = labelledBy.split(' ');
        ids.forEach((id) => {
          if (!document.getElementById(id)) {
            this.issues.push({
              type: 'aria-reference',
              severity: 'error',
              element: element as HTMLElement,`
              message: `aria-labelledby references non-existent, element: ${id}`,
              wcagCriteria: '4.1.2, Name, Role, Value'
            })}
        })}

      // Check aria-describedby references
      const describedBy = element.getAttribute('aria-describedby');
      if (describedBy) {
        const ids = describedBy.split(' ');
        ids.forEach((id) => {
          if (!document.getElementById(id)) {
            this.issues.push({
              type: 'aria-reference',
              severity: 'error',
              element: element as HTMLElement,`
              message: `aria-describedby references non-existent, element: ${id}`,
              wcagCriteria: '4.1.2, Name, Role, Value'
            })}
        })}
    });

    // Check for missing ARIA labels on landmark regions
    const landmarks = document.querySelectorAll('nav, main, aside, section, article');
    landmarks.forEach(_(landmark) => {
      if (!landmark.getAttribute('aria-label') && !landmark.getAttribute('aria-labelledby')) {
        this.issues.push({
          type: 'landmark-label',
          severity: 'warning',
          element: landmark as HTMLElement,`
          message: `Landmark region <${landmark.tagName.toLowerCase()}> should have an accessible name`,
          wcagCriteria: '2.4.1 Bypass Blocks',

        })}
    })}

  private auditFocusIndicators() {
    // Check if focus indicators are visible
    const focusableElements = document.querySelectorAll(
      'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    focusableElements.forEach((element) => {
      const styles = window.getComputedStyle(element as HTMLElement);
      const focusStyles = window.getComputedStyle(element as HTMLElement, ':focus');
      
      // This is a simplified check - in reality, you'd need to trigger focus
      if (styles.outline === 'none' && !focusStyles.outline) {
        this.issues.push({
          type: 'focus-indicator',
          severity: 'error',
          element: element as HTMLElement,
          message: 'Element may not have a visible focus indicator',
          wcagCriteria: '2.4.7 Focus Visible' })}
    })}

  private auditTouchTargets() {
    // Check minimum touch target size (44 x44, pixels)
    const interactiveElements = document.querySelectorAll(
      'a, button, input, select, textarea, [role="button"], [onclick]'
    );

    interactiveElements.forEach(_(element) => {
      const rect = element.getBoundingClientRect();
      if (rect.width < 44 || rect.height < 44) {
        this.issues.push({
          type: 'touch-target',
          severity: 'warning',
          element: element as HTMLElement,`
          message: `Touch target is too small (${Math.round(rect.width)}x${Math.round(rect.height)}px, minimum 44 x44px)`,
          wcagCriteria: '2.5.5 Target Size',

        })}
    })}

  private getEffectiveBackgroundColor(element: HTMLElement): string {
    let bgColor = window.getComputedStyle(element).backgroundColor;
    let currentElement = element.parentElement;

    while (currentElement && (bgColor === 'transparent' || bgColor === 'rgba(0, 0, 0, 0)')) {
      bgColor = window.getComputedStyle(currentElement).backgroundColor;
      currentElement = currentElement.parentElement;
    }

    return bgColor || 'rgb(255, 255, 255)'; // Default to white
  }

  // Generate accessibility report
  generateReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      url: window.location.href,
      totalIssues: this.issues.length,
      errors: this.issues.filter(i => i.severity === 'error').length,
      warnings: this.issues.filter(i => i.severity === 'warning').length,
      info: this.issues.filter(i => i.severity === 'info').length,
      issues: this.issues };

    return JSON.stringify(report, null, 2)}

  // Log issues to console
  logIssues() {
    console.group('Accessibility Audit Results');`
    console.log(`Found ${this.issues.length} issues`);
    
    const errors = this.issues.filter(i => i.severity === 'error');
    const warnings = this.issues.filter(i => i.severity === 'warning');
    const info = this.issues.filter(i => i.severity === 'info');

    if (errors.length > 0) {`
      console.group(`❌ Errors (${errors.length})`);
      errors.forEach(issue => {
        console.error(issue.message, issue.element)});
      console.groupEnd()}

    if (warnings.length > 0) {`
      console.group(`⚠️ Warnings (${warnings.length})`);
      warnings.forEach(issue => {
        console.warn(issue.message, issue.element)});
      console.groupEnd()}

    if (info.length > 0) {`
      console.group(`ℹ️ Info (${info.length})`);
      info.forEach(issue => {
        console.info(issue.message, issue.element)});
      console.groupEnd()}

    console.groupEnd()}
}

// Export singleton instance
export const accessibilityAuditor = new AccessibilityAuditor();

// Auto-run audit in development
if (import.meta.env.DEV) {
  // Run audit after page load
  window.addEventListener(_'load', () => {
    setTimeout(async () => {
      const issues = await accessibilityAuditor.runAudit();
      if (issues.length > 0) {
        accessibilityAuditor.logIssues()} else {
        console.log('✅ No accessibility issues found!')}
    }, 2000)})}`