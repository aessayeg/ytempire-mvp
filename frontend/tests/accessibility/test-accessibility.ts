/**
 * Accessibility Testing Script
 * Run this script to validate WCAG 2.1 AA compliance
 */

import { AccessibilityAuditor } from '../frontend/src/utils/accessibilityTesting';

// Test configuration
const TEST_PAGES = [
  '/dashboard',
  '/channels',
  '/videos',
  '/analytics',
  '/settings',
];

interface TestResult {
  page: string;
  errors: number;
  warnings: number;
  info: number;
  details: unknown[];
}

class AccessibilityTester {
  private auditor: AccessibilityAuditor;
  private results: TestResult[] = [];

  constructor() {
    this.auditor = new AccessibilityAuditor();
  }

  async runFullTest(): Promise<void> {
    console.log('🔍 Starting WCAG 2.1 AA Compliance Test Suite');
    console.log('='.repeat(50));

    for (const page of TEST_PAGES) {
      await this.testPage(page);
    }

    this.printSummary();
    this.generateReport();
  }

  private async testPage(path: string): Promise<void> {
    console.log(`\nTesting: ${path}`);
    
    // Navigate to page (in a real test, this would use Puppeteer or Playwright)
    // For now, we'll simulate by checking if we're on the right page
    if (window.location.pathname !== path) {
      console.log(`  ⏭️  Skipping (not on page)`);
      return;
    }

    // Run audit
    const issues = await this.auditor.runAudit();
    
    // Categorize issues
    const errors = issues.filter(i => i.severity === 'error');
    const warnings = issues.filter(i => i.severity === 'warning');
    const info = issues.filter(i => i.severity === 'info');

    // Store results
    const result: TestResult = {
      page: path,
      errors: errors.length,
      warnings: warnings.length,
      info: info.length,
      details: issues,
    };
    this.results.push(result);

    // Print page results
    if (errors.length === 0 && warnings.length === 0) {
      console.log('  ✅ PASSED - No accessibility issues found');
    } else {
      if (errors.length > 0) {
        console.log(`  ❌ FAILED - ${errors.length} error(s) found`);
        errors.forEach(e => {
          console.log(`     - ${e.message}`);
          if (e.wcagCriteria) {
            console.log(`       WCAG: ${e.wcagCriteria}`);
          }
        });
      }
      if (warnings.length > 0) {
        console.log(`  ⚠️  ${warnings.length} warning(s) found`);
      }
    }
  }

  private printSummary(): void {
    console.log('\n' + '='.repeat(50));
    console.log('📊 Test Summary');
    console.log('='.repeat(50));

    const totalErrors = this.results.reduce((sum, r) => sum + r.errors, 0);
    const totalWarnings = this.results.reduce((sum, r) => sum + r.warnings, 0);
    const totalInfo = this.results.reduce((sum, r) => sum + r.info, 0);
    const passedPages = this.results.filter(r => r.errors === 0).length;

    console.log(`\nPages Tested: ${this.results.length}`);
    console.log(`Pages Passed: ${passedPages}/${this.results.length}`);
    console.log(`\nTotal Issues:`);
    console.log(`  ❌ Errors: ${totalErrors}`);
    console.log(`  ⚠️  Warnings: ${totalWarnings}`);
    console.log(`  ℹ️  Info: ${totalInfo}`);

    // WCAG Compliance Status
    console.log('\n📋 WCAG 2.1 AA Compliance Status:');
    if (totalErrors === 0) {
      console.log('  ✅ COMPLIANT - All critical requirements met');
    } else {
      console.log('  ❌ NON-COMPLIANT - Critical issues must be resolved');
    }

    // Recommendations
    if (totalErrors > 0 || totalWarnings > 0) {
      console.log('\n💡 Recommendations:');
      const issueTypes = new Set<string>();
      this.results.forEach(r => {
        r.details.forEach(d => issueTypes.add(d.type));
      });

      const recommendations: Record<string, string> = {
        'color-contrast': 'Review and adjust color combinations to meet contrast ratios',
        'alt-text': 'Add descriptive alt text to all informative images',
        'heading-hierarchy': 'Ensure headings follow proper sequential order',
        'form-labels': 'Associate all form inputs with descriptive labels',
        'keyboard-access': 'Make all interactive elements keyboard accessible',
        'focus-indicator': 'Add visible focus indicators to all focusable elements',
        'touch-target': 'Increase size of interactive elements to 44x44 pixels minimum',
        'aria-reference': 'Fix broken ARIA attribute references',
      };

      issueTypes.forEach(type => {
        if (recommendations[type]) {
          console.log(`  • ${recommendations[type]}`);
        }
      });
    }
  }

  private generateReport(): void {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        pagesTestedred: this.results.length,
        pagesPassed: this.results.filter(r => r.errors === 0).length,
        totalErrors: this.results.reduce((sum, r) => sum + r.errors, 0),
        totalWarnings: this.results.reduce((sum, r) => sum + r.warnings, 0),
        totalInfo: this.results.reduce((sum, r) => sum + r.info, 0),
      },
      wcagCompliance: {
        level: 'AA',
        status: this.results.every(r => r.errors === 0) ? 'COMPLIANT' : 'NON-COMPLIANT',
        criteria: {
          '1.1.1': 'Non-text Content',
          '1.3.1': 'Info and Relationships',
          '1.4.3': 'Contrast (Minimum)',
          '2.1.1': 'Keyboard',
          '2.4.3': 'Focus Order',
          '2.4.6': 'Headings and Labels',
          '2.4.7': 'Focus Visible',
          '3.3.2': 'Labels or Instructions',
          '4.1.2': 'Name, Role, Value',
        },
      },
      details: this.results,
    };

    // Save report
    const reportJson = JSON.stringify(report, null, 2);
    console.log('\n📄 Full report available in console (copy below):');
    console.log(reportJson);

    // In a real implementation, save to file
    // fs.writeFileSync('accessibility-report.json', reportJson);
  }
}

// Auto-run test if this is the main module
if (typeof window !== 'undefined') {
  // Browser environment
  const tester = new AccessibilityTester();
  
  // Add to window for manual testing
  (window as unknown as { accessibilityTest: () => void }).accessibilityTest = () => tester.runFullTest();
  
  console.log('Accessibility testing ready!');
  console.log('Run: window.accessibilityTest() to start testing');
  
  // Auto-run in development
  if (process.env.NODE_ENV === 'development') {
    console.log('Auto-running accessibility test in 3 seconds...');
    setTimeout(() => {
      tester.runFullTest();
    }, 3000);
  }
}

export { AccessibilityTester };